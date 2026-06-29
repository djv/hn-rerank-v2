import socket
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import httpx
import pytest
from http.server import ThreadingHTTPServer
from hypothesis import HealthCheck, given, settings, strategies as st

from typing import Any

from server import Handler, SKELETON_HTML
from pipeline import Config, Embedder
from database import Database, Story

import numpy as np


class MockEmbedder(Embedder):
    """Drop-in stand-in for pipeline.Embedder with no model load.

    Subclasses Embedder for type compatibility with Handler.embedder (typed
    as Embedder), but overrides __init__ to skip the real
    AutoTokenizer.from_pretrained + ort.InferenceSession path — that's
    ~0.2s and a fresh ~30MB ONNX arena per instance, paid 30× in the
    cache-version property test alone.
    """

    def __init__(self) -> None:
        pass

    def encode(self, texts: list[str], batch_size: int = 64) -> Any:
        return np.zeros((len(texts), 384), dtype=np.float32)


def _read_template_and_static() -> tuple[str, str]:
    """Read both the Jinja2 template and the inline <script> block.

    Tests that look for JS code should check the inline script returned here,
    while tests that look for HTML attributes / Jinja2 directives check the
    template. (The script is inline again after the static extraction was
    rolled back — see WORKLOG 2026-06-27.)
    """
    repo_root = Path(__file__).resolve().parents[1]
    template = (repo_root / "templates" / "index.html").read_text(encoding="utf-8")
    start = template.find("  <script>\n")
    end = template.find("  </script>\n", start)
    inline_script = template[start:end] if start >= 0 and end >= 0 else ""
    return template, inline_script


@pytest.fixture(scope="module")
def mock_embedder() -> MockEmbedder:
    """One MockEmbedder for the whole test_server module.

    Avoids re-allocating the (cheap) instance, but more importantly keeps
    embedding-dependent handler state (warmup-in-flight threads, dashboard
    caches keyed by user) consistent across tests that share the module.
    """
    return MockEmbedder()


def _start_handler_server(
    db: Database, embedder: MockEmbedder, port: int = 0
) -> tuple[ThreadingHTTPServer, int, type[Handler]]:
    """Spin up a ThreadingHTTPServer on 127.0.0.1:<port>.

    Returns (server, port, TestHandler). The TestHandler is dynamically
    created with a fresh cache state and bound to (db, embedder, regen_event).
    Caller is responsible for cleanup (drain warmup, shutdown, db.close).
    """
    regen_event = threading.Event()

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=port)
    TestHandler.db = db
    TestHandler.embedder = embedder
    TestHandler.regen_event = regen_event
    TestHandler._dashboard_cache = {}
    TestHandler._dashboard_versions = {}
    TestHandler._render_locks = {}
    TestHandler._warmup_in_flight = set()
    TestHandler._warmup_in_flight_guard = threading.Lock()
    TestHandler._WARM_DEBOUNCE_S = 0.01

    server = ThreadingHTTPServer(("127.0.0.1", port), TestHandler)
    bound_port = server.server_address[1]
    return server, bound_port, TestHandler


def _drain_and_shutdown(server: ThreadingHTTPServer, handler: type[Handler]) -> None:
    drain_deadline = time.time() + 3.0
    while handler._warmup_in_flight and time.time() < drain_deadline:
        time.sleep(0.1)
    server.socket.shutdown(socket.SHUT_RDWR)
    server.shutdown()


@pytest.fixture(scope="module")
def app_env(tmp_path_factory, mock_embedder):
    """Module-scoped HTTP server for the small set of read-only server tests
    (redirects, static serving, CORS, tldr 404).

    Yields the same 5-tuple shape as test_env: (port, db, regen_event,
    TestHandler, user). regen_event is None here because no read-only test
    uses it; the rest of the positional shape is preserved so test bodies
    can swap `test_env` -> `app_env` with no other change.

    The TestHandler and server live for the whole module; cache state is
    reset between tests by a function-scoped autouse fixture
    (see _reset_app_env). Stateful tests (feedback POST/clear, dashboard
    renders that depend on cache state) must use test_env instead.
    """
    tmp_dir = tmp_path_factory.mktemp("app_env")
    db_file = tmp_dir / "app_env.db"
    db = Database(str(db_file))
    user = db.create_user("test_token")
    server, port, handler = _start_handler_server(db, mock_embedder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield port, db, None, handler, user

    _drain_and_shutdown(server, handler)
    db.close()


@pytest.fixture
def test_env(tmp_path, mock_embedder):
    """Per-test fresh DB + server. Used by stateful tests that mutate the
    cache, feedback table, or story table (the autouse app_env reset would
    not give them a clean DB).

    Reuses the module-scoped mock_embedder so we don't re-allocate the
    (cheap) instance per test.
    """
    db_file = tmp_path / "test_server.db"
    db = Database(str(db_file))

    # Create test user
    user = db.create_user("test_token")

    server, port, TestHandler = _start_handler_server(db, mock_embedder)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    regen_event = TestHandler.regen_event

    yield port, db, regen_event, TestHandler, user

    # Drain any in-flight warm threads on the fixture's TestHandler before
    # teardown so they don't outlive the fixture and pollute the next test's
    # monkeypatched pipeline functions.
    _drain_and_shutdown(server, TestHandler)
    db.close()


def test_token_redirect(app_env):
    port, _, _, _, user = app_env
    resp = httpx.get(f"http://127.0.0.1:{port}/u/{user.token}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"] == "../"
    assert "hn_token" in resp.headers.get("Set-Cookie", "")


def test_first_visit_redirect(app_env):
    port, _, _, _, _ = app_env
    resp = httpx.get(f"http://127.0.0.1:{port}/", follow_redirects=False)
    assert resp.status_code == 302
    location = resp.headers["Location"]
    assert location.startswith("u/")
    assert "hn_token" in resp.headers.get("Set-Cookie", "")


def test_dashboard_route_no_user_creates_token_and_redirects(app_env):
    port, db, _, _, _ = app_env
    with db._conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    resp = httpx.get(f"http://127.0.0.1:{port}/", follow_redirects=False)

    assert resp.status_code == 302
    assert resp.headers["Location"].startswith("u/")
    assert "hn_token" in resp.headers.get("Set-Cookie", "")
    with db._conn() as conn:
        after = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert after == before

    follow = httpx.get(
        f"http://127.0.0.1:{port}/{resp.headers['Location']}",
        follow_redirects=False,
    )
    assert follow.status_code == 302
    with db._conn() as conn:
        persisted = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert persisted == before + 1


def test_static_serving(test_env):
    port, _, _, handler, user = test_env
    # Pre-warm cache so HTTP request hits cached dashboard.
    result = handler._render_dashboard_for_user(user)
    assert result == SKELETON_HTML
    # Wait for warm to complete (debounce is 10ms, so ~20ms total)
    _wait_for_cache(handler, user, 0, timeout=3.0)
    # Now HTTP request should hit the cache
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert 'data-sort="recommended"' in resp.text
    assert 'data-sort="popular"' in resp.text
    assert 'data-sort="explore"' in resp.text
    assert 'data-sort="date"' in resp.text
    assert 'data-age="recent"' in resp.text
    assert 'data-age="archive"' in resp.text
    assert 'id="toast"' in resp.text
    assert 'class="refresh-progress"' not in resp.text
    assert 'id="sort-toggle"' not in resp.text
    assert 'id="queue-status"' not in resp.text
    assert 'id="refresh-banner"' not in resp.text
    assert 'id="refresh-now-btn"' not in resp.text


def test_feedback_post(test_env):
    port, db, regen_event, _, user = test_env
    db.upsert_story(
        Story(
            id=999,
            title="Feedback story",
            url="https://example.com",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    feedback_payload = {
        "story_id": 999,
        "action": "up",
    }
    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json=feedback_payload,
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": True}

    records = db.get_all_feedback(user.id)
    assert len(records) == 1
    assert records[0].story_id == 999
    assert records[0].action == "up"
    assert regen_event.is_set()


def test_feedback_post_invalidates_cache_on_every_vote(test_env):
    """Every vote bumps the dashboard version and triggers a warm, regardless
    of client-supplied ``queue_remaining`` or ``refresh_ranking`` hints.

    Regression: the previous "defer until queue low / every 5 votes" gating
    left the cached HTML stale for up to ~9s per burst; the SWR stale-hit
    path then re-injected already-voted stories via ``refillQueue`` (the bug
    observed on 2026-06-28).
    """
    port, db, regen_event, handler, user = test_env
    db.upsert_story(
        Story(
            id=1000,
            title="Always invalidate story",
            url="https://example.com",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    regen_event.clear()

    starting_version = handler._dashboard_version(user.id)
    assert starting_version == 0

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1000, "action": "up", "queue_remaining": 8},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": True}
    assert len(db.get_all_feedback(user.id)) == 1
    assert handler._dashboard_version(user.id) == starting_version + 1
    assert regen_event.is_set()


def test_feedback_post_invalidates_cache_with_low_queue(test_env):
    """Even a vote with ``queue_remaining: 4`` (low watermark) invalidates
    the cache. Belt for the client-side ``votedStoryIds`` filter (which
    also catches stale SWR refills).
    """
    port, db, regen_event, handler, user = test_env
    db.upsert_story(
        Story(
            id=1001,
            title="Low queue invalidate story",
            url="https://example.com",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    regen_event.clear()

    starting_version = handler._dashboard_version(user.id)

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1001, "action": "up", "queue_remaining": 4},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": True}
    assert handler._dashboard_version(user.id) == starting_version + 1
    assert regen_event.is_set()


def test_feedback_post_refreshes_when_client_requests_ranking(test_env):
    port, db, regen_event, _, user = test_env
    db.upsert_story(
        Story(
            id=1002,
            title="Batch refresh story",
            url="https://example.com",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    regen_event.clear()

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={
            "story_id": 1002,
            "action": "up",
            "queue_remaining": 20,
            "refresh_ranking": True,
        },
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": True}
    assert len(db.get_all_feedback(user.id)) == 1
    assert regen_event.is_set()


def test_feedback_post_bumps_cache_version_for_warm_rerender(test_env, monkeypatch):
    """End-to-end: vote on a story → cache version bumps → warm renders →
    cached HTML no longer contains the voted story.

    Regression for the 2026-06-28 bug where the dashboard served a
    pre-vote HTML deck via the SWR stale-hit path, re-injecting the
    just-voted story into the refill queue.
    """
    port, db, _, handler, user = test_env

    voted_story = Story(
        id=4242,
        title="Voted story",
        url="https://example.com/voted",
        score=100,
        time=1600000000,
        text_content="Voted body",
        source="hn",
    )
    db.upsert_story(voted_story)

    def fake_fast_rerank_for_user(database, config, embedder, user_id):
        return []

    def fake_generate_dashboard_bytes(ranked, config, database, user_id, user_token):
        version = handler._dashboard_version(user_id)
        body = f"version={version}"
        if voted_story.id not in (s.id for s in ranked):
            body += f" excluded={voted_story.id}"
        return body.encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    pre_version = handler._dashboard_version(user.id)
    assert pre_version == 0

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": voted_story.id, "action": "up", "queue_remaining": 6},
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": True}

    post_version = handler._dashboard_version(user.id)
    assert post_version == pre_version + 1, (
        "vote must bump the dashboard version so the SWR stale-hit "
        "cannot return the pre-vote HTML"
    )

    fresh_html = _wait_for_cache(handler, user, post_version)
    assert f"version={post_version}" in fresh_html.decode()


def test_feedback_clear(test_env):
    port, db, regen_event, _, user = test_env
    db.upsert_story(
        Story(
            id=999,
            title="Title",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text",
            source="hn",
        )
    )
    db.upsert_feedback(user.id, 999, "up")
    assert len(db.get_all_feedback(user.id)) == 1

    regen_event.clear()

    clear_payload = {
        "story_id": 999,
        "action": "clear",
    }
    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json=clear_payload,
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": True}

    assert len(db.get_all_feedback(user.id)) == 0
    assert regen_event.is_set()


def test_feedback_clear_then_revote_creates_new_record(test_env):
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=1003,
            title="Revote story",
            url=None,
            score=100,
            time=1600000000,
            text_content="Revote body",
            source="hn",
        )
    )

    for action in ("up", "clear", "down"):
        resp = httpx.post(
            f"http://127.0.0.1:{port}/api/feedback",
            json={"story_id": 1003, "action": action},
            cookies={"hn_token": user.token},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    records = db.get_all_feedback(user.id)
    assert len(records) == 1
    assert records[0].story_id == 1003
    assert records[0].action == "down"


def _wait_for_cache(handler, user, expected_version, timeout=3.0):
    key = f"dashboard_{user.id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        cached = handler._dashboard_cache.get(key)
        if cached and cached[2] == expected_version:
            return cached[0]
        time.sleep(0.1)
    raise AssertionError(
        f"Cache for user {user.id} version {expected_version} not populated within {timeout}s"
    )


def test_dashboard_cache_uses_feedback_versions(test_env, mock_embedder, monkeypatch):
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = mock_embedder
    TestHandler._dashboard_cache = {}
    TestHandler._dashboard_versions = {}
    TestHandler._render_locks = {}
    TestHandler._warmup_in_flight = set()
    TestHandler._warmup_in_flight_guard = threading.Lock()
    TestHandler._WARM_DEBOUNCE_S = 0.01

    calls = []

    def fake_fast_rerank_for_user(database, config, embedder, user_id):
        calls.append(("rank", user_id))
        return []

    def fake_generate_dashboard_bytes(ranked, config, database, user_id, user_token):
        version = TestHandler._dashboard_version(user_id)
        return f"version={version}".encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    # SWR: first call returns skeleton, warm thread renders async
    assert TestHandler._render_dashboard_for_user(user) == SKELETON_HTML
    assert len(calls) == 0
    assert len(TestHandler._warmup_in_flight) == 1

    # Wait for warm to complete version 0
    html_v0 = _wait_for_cache(TestHandler, user, 0)
    assert html_v0 == b"version=0"
    assert len(calls) == 1

    # Second call hits cache
    assert TestHandler._render_dashboard_for_user(user) == b"version=0"
    assert len(calls) == 1

    # Invalidate bumps version
    version = TestHandler._invalidate_dashboard_cache(user.id)
    assert version == 1

    # SWR: returns stale (version 0), triggers warm for version 1
    assert TestHandler._render_dashboard_for_user(user) == b"version=0"

    # Wait for warm to complete version 1
    html_v1 = _wait_for_cache(TestHandler, user, 1)
    assert html_v1 == b"version=1"
    assert len(calls) == 2


def test_stale_warm_render_does_not_overwrite_current_cache(
    test_env, mock_embedder, monkeypatch
):
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = mock_embedder
    TestHandler._dashboard_cache = {}
    TestHandler._dashboard_versions = {}
    TestHandler._render_locks = {}
    TestHandler._warmup_in_flight = set()
    TestHandler._warmup_in_flight_guard = threading.Lock()
    TestHandler._WARM_DEBOUNCE_S = 0.01

    def fake_fast_rerank_for_user(database, config, embedder, user_id):
        return []

    def fake_generate_dashboard_bytes(ranked, config, database, user_id, user_token):
        return f"requested={user_token}".encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    old_version = TestHandler._invalidate_dashboard_cache(user.id)
    new_version = TestHandler._invalidate_dashboard_cache(user.id)
    assert (old_version, new_version) == (1, 2)

    cache_key = f"dashboard_{user.id}"

    # SWR: returns skeleton, triggers warm for version 2
    assert (
        TestHandler._render_dashboard_for_user(user, expected_version=new_version)
        == SKELETON_HTML
    )

    # Wait for warm to complete
    current_html = _wait_for_cache(TestHandler, user, new_version)
    assert TestHandler._dashboard_cache[cache_key][2] == new_version

    # Stale hit: request with old_version returns current (version 2) cached content
    stale_html = TestHandler._render_dashboard_for_user(
        user, expected_version=old_version
    )
    assert stale_html == current_html
    assert stale_html == current_html
    # Cache should NOT have been overwritten — still version 2
    assert TestHandler._dashboard_cache[cache_key][2] == new_version


@pytest.fixture(scope="module")
def prop_db():
    with TemporaryDirectory() as temp_dir:
        db = Database(str(Path(temp_dir) / "prop_server.db"))
        yield db
        db.close()


@given(
    operations=st.lists(
        st.sampled_from(["invalidate", "render_current", "render_stale"]),
        min_size=1,
        max_size=40,
    )
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_dashboard_cache_version_invariant_property(
    operations, prop_db, mock_embedder, monkeypatch
):
    with prop_db._conn() as conn:
        with conn:
            conn.execute("DELETE FROM users")
            conn.execute("DELETE FROM stories")
            conn.execute("DELETE FROM feedback")
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM tldr_cache")
            conn.execute("DELETE FROM article_fetch_failures")
    user = prop_db.create_user("prop_user")

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=prop_db.db_path, server_port=0)
    TestHandler.db = prop_db
    TestHandler.embedder = mock_embedder
    TestHandler._dashboard_cache = {}
    TestHandler._dashboard_versions = {}
    TestHandler._render_locks = {}
    TestHandler._warmup_in_flight = set()
    TestHandler._warmup_in_flight_guard = threading.Lock()
    TestHandler._WARM_DEBOUNCE_S = 0.01

    def fake_fast_rerank_for_user(database, config, embedder, user_id):
        return []

    def fake_generate_dashboard_bytes(ranked, config, database, user_id, user_token):
        return f"v={TestHandler._dashboard_version(user_id)}".encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    cache_key = f"dashboard_{user.id}"
    for operation in operations:
        current_version = TestHandler._dashboard_version(user.id)
        if operation == "invalidate":
            TestHandler._invalidate_dashboard_cache(user.id)
        elif operation == "render_current":
            TestHandler._render_dashboard_for_user(user)
        else:
            stale_version = max(0, current_version - 1)
            TestHandler._render_dashboard_for_user(user, expected_version=stale_version)
            # stale hit MUST return content that was in cache before
            # (cache version must be ≥ stale_version)

        # Wait for any in-flight warm to settle before checking invariant.
        # SWR allows stale cache entries (cache version < current version)
        # between invalidation and warm completion.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            cached = TestHandler._dashboard_cache.get(cache_key)
            cur_ver = TestHandler._dashboard_version(user.id)
            if cached is None or cached[2] <= cur_ver:
                break
            time.sleep(0.1)
        cached = TestHandler._dashboard_cache.get(cache_key)
        if cached is not None:
            cur_ver = TestHandler._dashboard_version(user.id)
            assert cached[2] <= cur_ver, (
                f"cache version {cached[2]} > dashboard version "
                f"{cur_ver} after op={operation}"
            )

    # Drain in-flight warm threads before monkeypatch cleanup so they don't
    # capture our fakes and leak into subsequent tests.
    drain_deadline = time.time() + 3.0
    while TestHandler._warmup_in_flight and time.time() < drain_deadline:
        time.sleep(0.1)


def test_cors_headers(app_env):
    port, _, _, _, _ = app_env
    resp = httpx.options(f"http://127.0.0.1:{port}/api/feedback")
    assert resp.status_code == 204
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


def test_normalize_tldr_markdown_repairs_inline_bullets():
    import server

    raw = (
        "Article\n"
        "Consensus: Open-source models are improving.\n"
        "Notable Caveats:\n"
        "- Quantization may degrade quality. - Providers reduce hardware barriers."
    )

    normalized = server._normalize_tldr_markdown(raw)

    assert "### Article" in normalized
    assert "- **Consensus:** Open-source models are improving." in normalized
    assert "- **Notable Caveats:**" in normalized
    assert (
        "- Quantization may degrade quality.\n- Providers reduce hardware barriers."
        in normalized
    )


def test_reddit_rss_helpers_extract_post_and_comment_text():
    import server

    assert (
        server._reddit_post_rss_url(
            "https://www.reddit.com/r/LocalLLaMA/comments/1u7qti8/title/"
        )
        == "https://www.reddit.com/r/LocalLLaMA/comments/1u7qti8/title/.rss"
    )

    raw = (
        '<table><tr><td><div class="md">'
        '<p><a href="https://x.com/a/status/1">https://x.com/a/status/1</a></p>'
        "</div> submitted by /u/test</td></tr></table>"
    )

    assert server._clean_reddit_rss_html(raw) == "https://x.com/a/status/1"


def test_tldr_cache_key_truncates_prompt_inputs(monkeypatch):
    import server

    monkeypatch.setattr(server, "SELF_TEXT_PROMPT_CHAR_LIMIT", 5)
    monkeypatch.setattr(server, "COMMENT_PROMPT_CHAR_LIMIT", 6)
    monkeypatch.setattr(server, "ARTICLE_BODY_CHAR_LIMIT", 7)

    key1 = server._tldr_cache_key(
        title="Same title",
        self_text="abcde-left",
        top_comments="abcdef-left",
        article_body="abcdefg-left",
    )
    key2 = server._tldr_cache_key(
        title="Same title",
        self_text="abcde-right",
        top_comments="abcdef-right",
        article_body="abcdefg-right",
    )
    key3 = server._tldr_cache_key(
        title="Same title",
        self_text="xbcde-right",
        top_comments="abcdef-right",
        article_body="abcdefg-right",
    )

    assert key1 == key2
    assert key1 != key3


def test_reddit_low_signal_comment_filter():
    import server

    assert server._is_low_signal_reddit_comment(
        "withoutreason1729",
        "This is long enough but comes from a known noisy bot account.",
    )
    assert server._is_low_signal_reddit_comment("/u/AutoModerator", "Useful length.")
    assert server._is_low_signal_reddit_comment("/u/alice", "[deleted]")
    assert server._is_low_signal_reddit_comment("/u/alice", "[removed]")
    assert server._is_low_signal_reddit_comment(
        "/u/alice",
        "I am a bot and this action was performed automatically.",
    )
    assert server._is_low_signal_reddit_comment(
        "/u/alice",
        "Your post is getting popular and something something.",
    )
    assert server._is_low_signal_reddit_comment("/u/alice", "too short")
    assert not server._is_low_signal_reddit_comment(
        "/u/alice",
        "This is a substantive Reddit comment with enough content to summarize.",
    )


@pytest.mark.asyncio
async def test_reddit_rss_context_caps_comments_and_cached_chars(monkeypatch):
    import server

    def item(title, author, body):
        return f"""
        <item>
          <title>{title}</title>
          <dc:creator>{author}</dc:creator>
          <description><![CDATA[<div class="md"><p>{body}</p></div>]]></description>
        </item>
        """

    comments = "\n".join(
        item(
            f"comment {i}",
            f"user{i}",
            f"This is substantive comment number {i} with enough useful detail to keep.",
        )
        for i in range(1, 8)
    )
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel>
        {item("post", "poster", "Post self text with an embedded link.")}
        {comments}
      </channel>
    </rss>
    """

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, *args, **kwargs):
            return httpx.Response(200, text=rss)

    monkeypatch.setattr(server.httpx, "AsyncClient", MockAsyncClient)
    monkeypatch.setattr(server, "REDDIT_COMMENT_LIMIT", 3)
    monkeypatch.setattr(server, "REDDIT_COMMENTS_CACHE_CHAR_LIMIT", 160)

    context = await server._fetch_reddit_rss_context(
        "https://www.reddit.com/r/LocalLLaMA/comments/abc123/test/"
    )

    assert context is not None
    assert context.self_text == "Post self text with an embedded link."
    assert context.comment_count <= 3
    assert len(context.top_comments) <= 160
    assert "comment number 1" in context.top_comments
    assert "comment number 4" not in context.top_comments


def test_tldr_handler_returns_404_for_missing_story(app_env):
    port, _, _, _, user = app_env

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 987654321},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 404
    assert resp.json()["error"] == "Story not found in database"


def test_tldr_detail_fetches_reddit_rss_comments(test_env, monkeypatch):
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=-1234,
            title="Reddit test",
            url="https://www.reddit.com/r/LocalLLaMA/comments/abc123/reddit_test/",
            score=0,
            time=1600000000,
            text_content="Reddit test. https://x.com/example/status/1",
            source="rss_reddit_localllama",
            comment_count=None,
            discussion_url=None,
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    async def mock_fetch_reddit_rss_context(url):
        return server.RedditRssContext(
            self_text="https://x.com/example/status/1",
            top_comments="/u/alice: Useful Reddit comment about the model.",
            comment_count=1,
        )

    async def mock_fetch_article_body(url):
        raise AssertionError("Reddit comments pages should not be scraped as articles")

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return f"TLDR: {self_text} | {top_comments}"

    import server

    monkeypatch.setattr(
        server, "_fetch_reddit_rss_context", mock_fetch_reddit_rss_context
    )
    monkeypatch.setattr(server, "_fetch_article_body", mock_fetch_article_body)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": -1234},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "Useful Reddit comment" in data["tldr"]

    updated_story = db.get_story(-1234)
    assert updated_story.self_text == "https://x.com/example/status/1"
    assert "Useful Reddit comment" in updated_story.top_comments
    assert updated_story.discussion_url == (
        "https://www.reddit.com/r/LocalLLaMA/comments/abc123/reddit_test/"
    )


def test_tldr_detail_dynamic_fetch(test_env, monkeypatch):
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=777,
            title="Dynamic test",
            url="https://example.com/dynamic-test",
            score=100,
            time=1600000000,
            text_content="Dynamic test.",
            source="hn",
            comment_count=5,
            discussion_url="https://news.ycombinator.com/item?id=777",
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    # Mock fetch_story and _fetch_article_body
    async def mock_fetch_story(client, sid, database):
        story = database.get_story(sid)
        from dataclasses import replace

        updated = replace(
            story,
            top_comments="Fetched comments",
            text_content="Dynamic test. Fetched comments",
        )
        database.upsert_story(updated)
        return updated

    async def mock_fetch_article_body(url):
        return "Fetched article body text"

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return f"TLDR: {title} | {top_comments} | {article_body}"

    import server
    import pipeline

    monkeypatch.setattr(pipeline, "fetch_story", mock_fetch_story)
    monkeypatch.setattr(server, "_fetch_article_body", mock_fetch_article_body)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    # Request TLDR
    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 777},
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "Fetched comments" in data["tldr"]
    assert "Fetched article body text" in data["tldr"]

    # Verify database was updated
    updated_story = db.get_story(777)
    assert updated_story.top_comments == "Fetched comments"
    assert updated_story.article_body == "Fetched article body text"


def test_tldr_detail_dynamic_fetch_for_bq_seed(test_env, monkeypatch):
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=779,
            title="BQ dynamic test",
            url="https://example.com/bq-dynamic-test",
            score=100,
            time=1600000000,
            text_content="BQ dynamic test.",
            source="bq_seed",
            comment_count=5,
            discussion_url="https://news.ycombinator.com/item?id=779",
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    async def mock_fetch_story(client, sid, database):
        story = database.get_story(sid)
        from dataclasses import replace

        updated = replace(
            story,
            top_comments="Fetched BQ comments",
            text_content="BQ dynamic test. Fetched BQ comments",
        )
        database.upsert_story(updated)
        return updated

    async def mock_fetch_article_body(url):
        return None

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return f"TLDR: {title} | {top_comments}"

    import server
    import pipeline

    monkeypatch.setattr(pipeline, "fetch_story", mock_fetch_story)
    monkeypatch.setattr(server, "_fetch_article_body", mock_fetch_article_body)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 779},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert "Fetched BQ comments" in resp.json()["tldr"]
    assert db.get_story(779).top_comments == "Fetched BQ comments"


def test_tldr_detail_dynamic_fetch_for_ch_seed(test_env, monkeypatch):
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=771,
            title="CH dynamic test",
            url="https://example.com/ch-dynamic-test",
            score=100,
            time=1600000000,
            text_content="CH dynamic test.",
            source="ch_seed",
            comment_count=5,
            discussion_url="https://news.ycombinator.com/item?id=771",
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    async def mock_fetch_story(client, sid, database):
        story = database.get_story(sid)
        from dataclasses import replace

        updated = replace(
            story,
            top_comments="Fetched CH comments",
            text_content="CH dynamic test. Fetched CH comments",
        )
        database.upsert_story(updated)
        return updated

    async def mock_fetch_article_body(url):
        return None

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return f"TLDR: {title} | {top_comments}"

    import server
    import pipeline

    monkeypatch.setattr(pipeline, "fetch_story", mock_fetch_story)
    monkeypatch.setattr(server, "_fetch_article_body", mock_fetch_article_body)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 771},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert "Fetched CH comments" in resp.json()["tldr"]
    assert db.get_story(771).top_comments == "Fetched CH comments"


def test_tldr_detail_uses_cached_summary(test_env, monkeypatch):
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=778,
            title="Cached TLDR test",
            url="https://example.com/cached-tldr",
            score=12,
            time=1600000000,
            text_content="Cached TLDR test. Existing article body.",
            source="hn",
            comment_count=0,
            discussion_url="https://news.ycombinator.com/item?id=778",
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="Existing article body.",
        )
    )

    calls = 0

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        nonlocal calls
        calls += 1
        return f"cached-result-{calls}: {title} | {article_body}"

    import server

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp1 = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 778},
        cookies={"hn_token": user.token},
    )
    resp2 = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 778},
        cookies={"hn_token": user.token},
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert calls == 1
    assert resp1.json()["cached"] is False
    assert resp2.json()["cached"] is True
    assert resp2.json()["tldr"] == resp1.json()["tldr"]


def test_tldr_detail_does_not_cache_placeholder(test_env, monkeypatch):
    """A story with no content returns the placeholder but does not cache it."""
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=779,
            title="Empty story",
            url=None,
            score=5,
            time=1600000000,
            text_content="x",
            source="hn",
            comment_count=0,
            discussion_url=None,
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    call_count = 0

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        nonlocal call_count
        call_count += 1
        return "No article body or discussion available to summarize for this story."

    import server

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp1 = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 779},
        cookies={"hn_token": user.token},
    )
    resp2 = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 779},
        cookies={"hn_token": user.token},
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert call_count == 2  # both requests regenerated (no cache write)
    assert resp1.json()["cached"] is False
    assert resp2.json()["cached"] is False
    assert db.get_tldr_cache(779, "") is None  # no cache entry written


@pytest.mark.asyncio
async def test_generate_detailed_tldr_splits_article_and_comments(monkeypatch):
    import server

    calls = []

    async def mock_call_llm_chat(*, api_key, base_url, model, prompt, max_tokens):
        calls.append(prompt)
        if "Summarize the article" in prompt:
            return "- **Article** summary"
        if "Summarize the discussion" in prompt:
            return "- **Discussion** summary"
        return "- **Fallback** summary"

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Split summary test",
        self_text="Author text",
        top_comments="Comment text",
        article_body="Article body",
    )

    assert len(calls) == 2
    assert "Article body" in calls[0]
    assert "Comments:" in calls[1]
    assert "Points:" not in calls[1]
    assert "Age hours:" not in calls[1]
    assert "### Article" in result
    assert "- **Article** summary" in result
    assert "### Discussion" in result
    assert "- **Discussion** summary" in result


async def test_unified_fallback_omits_article_when_no_article_body(
    monkeypatch,
) -> None:
    """Discussion-only stories must not produce an ### Article or ### Story section."""
    import server

    calls = []

    async def mock_call_llm_chat(*, api_key, base_url, model, prompt, max_tokens):
        calls.append(prompt)
        return "- **Discussion** summary"

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Story links to OpenAI",
        self_text="",
        top_comments="comment 1\ncomment 2",
        article_body="",
    )

    assert "### Article" not in result
    assert "### Story" not in result
    assert len(calls) == 1


async def test_generate_detailed_tldr_returns_stub_when_no_content(
    monkeypatch,
) -> None:
    """No article + no comments → short stub, zero LLM calls."""
    import server

    calls: list = []

    async def mock_call_llm_chat(*, api_key, base_url, model, prompt, max_tokens):
        calls.append(prompt)
        return "should not be called"

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Empty story",
        self_text="",
        top_comments="",
        article_body="",
    )

    assert calls == []
    assert "No article body or discussion" in result


def test_keydown_guard_excludes_buttons_and_anchors():
    """Regression: the global keydown handler in static/dashboard.js must not
    bail out when a <button> or <a> has focus, otherwise clicking a mode tab
    or vote button blocks the next ArrowUp/ArrowDown from registering.
    """
    _, static = _read_template_and_static()
    assert "addEventListener('keydown'" in static, (
        "keydown handler not found in static/dashboard.js"
    )

    # Locate the closest(...) call in the guard
    idx = static.index("closest?.(")
    end = static.index(")", idx)
    guard = static[idx : end + 1]

    assert "button" not in guard, "button should not block global shortcuts"
    assert "'a'" not in guard, "a should not block global shortcuts"
    assert "input" in guard, "input should still block"
    assert "textarea" in guard, "textarea should still block"
    assert "select" in guard, "select should still block"
    assert '[contenteditable="true"]' in guard, "contenteditable should still block"


def test_dashboard_has_no_refresh_button_or_progress_bar():
    """The refresh button and the 5-vote progress bar were removed (2026-06-29):
    the server still invalidates the cache on every vote, but the client
    silently refills the queue on every successful vote save and on every
    sort/age/source tab click — no user-facing button or bar to wait for.
    """
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "index.html"
    ).read_text(encoding="utf-8")
    # Old refresh UI is gone
    assert 'id="refresh-banner"' not in template
    assert 'id="refresh-now-btn"' not in template
    assert 'class="refresh-progress"' not in template
    assert 'class="refresh-segment"' not in template
    assert 'class="refresh-wrapper"' not in template
    assert 'class="refresh-label"' not in template
    assert "pulse-segment" not in template
    assert 'role="progressbar"' not in template
    assert "VOTES_PER_RANKING_REFRESH" not in template
    assert "refresh-progress" not in template.split("</style>", 1)[0]
    # New toast element is present and accessible
    assert 'id="toast"' in template
    assert 'role="status"' in template
    assert 'aria-live="polite"' in template
    assert 'class="toast"' in template
    assert "showToast" in template
    assert "silentRefill" in template
    # 'u undo' hint removed from the legend
    assert '<span class="key-hint">u</span> undo' not in template, (
        "undo hint should be hidden from the visible legend"
    )
    # Vote counts element still present
    assert 'class="vote-counts"' in template
    assert template.count('data-vote-count="up"') == 1
    assert template.count('data-vote-count="neutral"') == 1
    assert template.count('data-vote-count="down"') == 1
    # Old text "Loading queue..." should be gone from the template
    assert "Loading queue..." not in template


def test_extract_lesswrong_post_id():
    import server

    assert (
        server._extract_lesswrong_post_id(
            "https://www.lesswrong.com/posts/3TpvKNKAvFGDc5b5k/and-what-happens-next"
        )
        == "3TpvKNKAvFGDc5b5k"
    )
    assert (
        server._extract_lesswrong_post_id("https://www.lesswrong.com/posts/abc123/slug")
        == "abc123"
    )
    assert server._extract_lesswrong_post_id("https://example.com/foo") is None
    assert server._extract_lesswrong_post_id("") is None
    assert server._extract_lesswrong_post_id(None) is None


def test_clean_lesswrong_html():
    import server

    raw = '<p>See also: <a href="https://example.com">a post</a>.</p>'
    cleaned = server._clean_lesswrong_html(raw)
    assert "See also:" in cleaned
    assert "a post" in cleaned

    assert server._clean_lesswrong_html("") == ""
    assert server._clean_lesswrong_html(None) == ""
    assert (
        server._clean_lesswrong_html("<p>  <b>Hello</b>   world  </p>") == "Hello world"
    )


async def test_lesswrong_context_fetches_post_and_comments(monkeypatch):
    import server

    graphql_response = {
        "data": {
            "post": {
                "result": {
                    "_id": "3TpvKNKAvFGDc5b5k",
                    "commentCount": 4,
                    "contents": {"html": "<p>Post body with <b>key</b> insight.</p>"},
                }
            },
            "comments": {
                "results": [
                    {
                        "_id": "c1",
                        "postId": "3TpvKNKAvFGDc5b5k",
                        "author": "gwern",
                        "baseScore": 5,
                        "htmlBody": "<p>Great point about X.</p>",
                        "postedAt": "2026-06-23T20:20:47.723Z",
                    },
                    {
                        "_id": "c2",
                        "postId": "3TpvKNKAvFGDc5b5k",
                        "author": "",
                        "baseScore": 3,
                        "htmlBody": "<p>Short reply.</p>",
                        "postedAt": "2026-06-23T21:00:00.000Z",
                    },
                ]
            },
        }
    }

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return httpx.Response(200, json=graphql_response)

    monkeypatch.setattr(server.httpx, "AsyncClient", MockAsyncClient)

    ctx = await server._fetch_lesswrong_context("3TpvKNKAvFGDc5b5k")

    assert ctx is not None
    assert "key insight" in ctx.self_text
    assert "gwern" in ctx.top_comments
    assert "Great point about X" in ctx.top_comments
    assert "/u/gwern" in ctx.top_comments
    assert ctx.comment_count == 2


def test_tldr_detail_fetches_lesswrong_comments(test_env, monkeypatch):
    import server

    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=-3000,
            title="And What Happens Next",
            url="https://www.lesswrong.com/posts/3TpvKNKAvFGDc5b5k/and-what-happens-next",
            score=0,
            time=1600000000,
            text_content="And What Happens Next.",
            source="rss_lesswrong_com",
            comment_count=None,
            discussion_url=None,
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    async def mock_fetch_lesswrong_context(post_id):
        return server.LessWrongContext(
            self_text="Post body with key insight.",
            top_comments="/u/gwern: Great point about X.",
            comment_count=1,
        )

    async def mock_fetch_article_body(url):
        raise AssertionError("LessWrong should not be scraped as articles")

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return f"TLDR: {self_text} | {top_comments}"

    monkeypatch.setattr(
        server, "_fetch_lesswrong_context", mock_fetch_lesswrong_context
    )
    monkeypatch.setattr(server, "_fetch_article_body", mock_fetch_article_body)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": -3000},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "Great point about X" in data["tldr"]

    updated_story = db.get_story(-3000)
    assert updated_story.self_text == "Post body with key insight."
    assert "Great point about X" in updated_story.top_comments
    assert updated_story.discussion_url == (
        "https://www.lesswrong.com/posts/3TpvKNKAvFGDc5b5k/and-what-happens-next"
    )
    assert updated_story.comment_count == 1


def test_dashboard_has_source_filter_toggle():
    """The side rail must expose a 3-way source filter (Mixed/HN/Non-HN)
    that narrows the deck by story source. Mixed is the default.
    """
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "index.html"
    ).read_text(encoding="utf-8")
    assert 'class="source-tabs"' in template
    assert 'data-source="mixed"' in template
    assert 'data-source="hn"' in template
    assert 'data-source="non-hn"' in template
    # Mixed is the default active tab
    assert 'class="source-tab active" data-source="mixed"' in template
    # Source filter appears before swipe keys in the DOM
    assert template.index("source-tabs") < template.index("swipe-keys")


def test_story_cards_emit_is_hn_attribute():
    """Each .story-card must carry a data-is-hn flag so the client source
    filter can use a single source of truth (matches is_hn_source on the
    server). HN/CH-archive/BQ-archive → 1; RSS → 0.
    """
    template, static = _read_template_and_static()
    assert (
        "data-is-hn=\"{{ '0' if item.is_non_hn else '1' }}\"".replace("{{", "{{")
        in template
    )
    # The client filter must use data-is-hn, not the old prefix-based check
    assert "card.dataset.isHn" in static
    assert "s.startsWith('rss_')" not in static
    assert "s === 'hn' || s === 'bq_seed'" not in static


def test_dashboard_js_loaded_via_static_endpoint():
    """The inline <script> is served from the template (the static/dashboard.js
    extraction was rolled back — see WORKLOG 2026-06-27)."""
    repo_root = Path(__file__).resolve().parents[1]
    template = (repo_root / "templates" / "index.html").read_text(encoding="utf-8")
    # The template must contain an inline <script> block
    assert "  <script>\n" in template
    # The template must NOT reference a /static/ JS file
    assert 'src="/static/dashboard.js"' not in template


def test_static_dashboard_js_has_no_jinja():
    """The inline <script> in the template is served as-is by Jinja2, so it
    must not contain Jinja2 directives."""
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "index.html"
    ).read_text(encoding="utf-8")
    start = template.find("  <script>\n")
    end = template.find("  </script>\n", start)
    inline_script = template[start:end] if start >= 0 and end >= 0 else ""
    assert "{{" not in inline_script, "inline script must not contain Jinja2 {{ }}"
    assert "{%" not in inline_script, "inline script must not contain Jinja2 {% %}"


def test_keydown_uses_letter_keys():
    """The global keydown handler maps j/k/l to down/up/neutral
    and ArrowUp/ArrowDown scroll the active card.
    """
    template, static = _read_template_and_static()
    assert "key === 'j'" in static
    assert "key === 'k'" in static
    assert "key === 'l'" in static
    # arrow bindings present for card scrolling
    handler = static.split("document.addEventListener('keydown'")[1].split("});", 1)[0]
    assert "arrowup" in handler
    assert "arrowdown" in handler
    assert "ArrowRight" not in handler
    # legend shows the new label
    assert "skip (neutral)" in template.lower()
    # first-time tip present and uses floating overlay
    assert "first-time-tip" in template
    assert "position: fixed" in template
    assert 'aria-label="Keyboard shortcuts"' in template
    assert "first-time-tip-inner" in template
    # open article / open comments keys present
    assert "key === 'o'" in static
    assert "key === 'c'" in static
    assert "open article" in template.lower()
    assert "open comments" in template.lower()
    assert "data-article-url" in template
    assert "data-comments-url" in template
    # card sizing: shrink-to-fit for short, full width for enriched,
    # max-height caps at viewport so page never scrolls
    assert "width: fit-content" in template
    assert "max-width: 902px" in template
    assert "max-width: none" in template
    # page never scrolls — overflow hidden on html and body
    assert "html {\n      overflow: hidden;\n    }" in template
    assert "overflow: hidden;" in template.split("body {")[1].split("}")[0]
    active_block = template.split(".story-card.active {", 1)[1].split("}", 1)[0]
    assert "max-height: calc(100vh - 5rem)" in active_block
    assert "min-height: 18rem;" in active_block
    # active card has bottom padding to clear the fixed vote bar
    assert "padding-bottom: 4rem;" in active_block
    enriched_block = template.split(".story-card.enriched {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in enriched_block
    # long unbroken text wraps instead of overflowing the card
    assert (
        "overflow-wrap: anywhere;"
        in template.split(".story-title a {", 1)[1].split("}", 1)[0]
    )
    assert (
        "overflow-wrap: anywhere;"
        in template.split(".match-reason {", 1)[1].split("}", 1)[0]
    )
    assert (
        "overflow-wrap: anywhere;"
        in template.split(".tldr-detail-content {", 1)[1].split("}", 1)[0]
    )
    assert (
        "overflow-wrap: anywhere;"
        in template.split(".story-header {", 1)[1].split("}", 1)[0]
    )
    # no min-height on #stories, rail caps via max-height
    stories_block = template.split("#stories {", 1)[1].split("}", 1)[0]
    assert "min-height" not in stories_block
    side_block = template.split(".swipe-side {", 1)[1].split("}", 1)[0]
    assert "max-height: calc(100vh - 1.5rem)" in side_block
    assert "overflow-y: auto" in side_block
    assert "sticky" not in side_block
    # layout uses flex-start, not stretch
    assert "align-items: flex-start" in template
    # mobile side-rail stack (column, keys hidden)
    assert ".swipe-keys { display: none; }" in template
    assert "width: 100%;" in template
    assert "flex-direction: column" in template
    # bigger touch buttons on mobile
    assert "padding: 0.6rem 0.9rem" in template
    assert "min-width: 2.75rem" in template
    assert "min-height: 2.75rem" in template
    # flex scroll container on mobile
    assert "height: calc(100vh - 1.5rem)" in template
    assert "100dvh" in template
    assert ".story-card.active {\n        max-height: 100%;" in template
    # global vote bar at the bottom of the viewport
    assert (
        "position: fixed;\n      bottom: 0;\n      left: 0;\n      right: 0;"
        in template
    )
    assert ".vote-bar[hidden]" in template
    assert '<div class="vote-bar" hidden>' in template
    # vote counts live in the vote bar; no refresh bar/label/progress
    assert 'class="vote-counts"' in template.split('<div class="vote-bar"')[1]
    assert 'class="refresh-progress"' not in template
    assert 'class="refresh-label"' not in template
    assert "Votes until refresh" not in template
    assert "width: 120px" not in template
    # vote bar is flex-end (no wrapper on the left after removing the progress bar)
    assert "justify-content: flex-end" in template
    assert (
        "margin-left: auto" not in template.split(".vote-counts", 1)[1].split("}", 1)[0]
    )
    # toast is positioned fixed at the top
    assert ".toast {" in template
    assert 'role="status"' in template
    assert 'aria-live="polite"' in template
    # mode and source tabs have filled active style
    assert (
        ".sort-tab.active, .age-tab.active {\n      background: var(--pico-primary);"
        in template
    )
    assert ".source-tab.active {\n      background: #6c757d;" in template
    # feedback button has filled, shadowed style
    assert "box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);" in template
    # click handler no longer passes null card
    assert "submitVote(btn.dataset.fb, btn.closest('.story-card'))" not in static
    assert "submitVote(btn.dataset.fb);" in static


def test_dashboard_renders_user_vote_counts_zero_for_no_feedback(test_env):
    """Fresh user with no feedback → all three counts are 0."""
    port, db, regen_event, handler, user = test_env
    assert handler._render_dashboard_for_user(user) == SKELETON_HTML
    _wait_for_cache(handler, user, 0, timeout=3.0)
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert 'data-vote-count="up">0<' in resp.text
    assert 'data-vote-count="neutral">0<' in resp.text
    assert 'data-vote-count="down">0<' in resp.text


def test_dashboard_renders_user_vote_counts_with_feedback(test_env):
    """Seeded feedback → counts rendered in the dashboard."""
    port, db, regen_event, handler, user = test_env
    for i in range(3):
        db.upsert_story(
            Story(
                id=2000 + i,
                title=f"Up {i}",
                url=None,
                score=100 - i,
                time=0,
                text_content="text",
            )
        )
        db.upsert_feedback(user.id, 2000 + i, "up")
    db.upsert_story(
        Story(id=3000, title="Neutral", url=None, score=90, time=0, text_content="text")
    )
    db.upsert_feedback(user.id, 3000, "neutral")
    for i in range(2):
        db.upsert_story(
            Story(
                id=4000 + i,
                title=f"Down {i}",
                url=None,
                score=80 - i,
                time=0,
                text_content="text",
            )
        )
        db.upsert_feedback(user.id, 4000 + i, "down")

    assert handler._render_dashboard_for_user(user) == SKELETON_HTML
    _wait_for_cache(handler, user, 0, timeout=3.0)
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert 'data-vote-count="up">3<' in resp.text
    assert 'data-vote-count="neutral">1<' in resp.text
    assert 'data-vote-count="down">2<' in resp.text


def test_dashboard_vote_counts_aggregate_across_refreshes(test_env):
    """Vote counts persist across ranking refreshes (all-time, not session)."""
    port, db, regen_event, handler, user = test_env
    for i in range(5):
        db.upsert_story(
            Story(
                id=5000 + i,
                title=f"S{i}",
                url=None,
                score=100,
                time=0,
                text_content="text",
            )
        )
        db.upsert_feedback(user.id, 5000 + i, "up")
    for i in range(2):
        db.upsert_story(
            Story(
                id=6000 + i,
                title=f"T{i}",
                url=None,
                score=80,
                time=0,
                text_content="text",
            )
        )
        db.upsert_feedback(user.id, 6000 + i, "down")

    assert handler._render_dashboard_for_user(user) == SKELETON_HTML
    _wait_for_cache(handler, user, 0, timeout=3.0)
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert 'data-vote-count="up">5<' in resp.text
    assert 'data-vote-count="neutral">0<' in resp.text
    assert 'data-vote-count="down">2<' in resp.text


# SWR / model cache integration tests
# ------------------------------------


def test_dashboard_skeleton_returns_when_no_cache(test_env):
    port, db, regen_event, _, user = test_env
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": "test_token"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Loading your personalized dashboard" in resp.content
    assert b'meta http-equiv="refresh" content="1"' in resp.content


@pytest.fixture
def swr_handler(test_env, mock_embedder):
    _, db, _, _, user = test_env

    class SwrHandler(Handler):
        pass

    SwrHandler.config = Config(db_path=db.db_path, server_port=0)
    SwrHandler.db = db
    SwrHandler.embedder = mock_embedder
    SwrHandler._dashboard_cache = {}
    SwrHandler._dashboard_versions = {}
    SwrHandler._render_locks = {}
    SwrHandler._warmup_in_flight = set()
    SwrHandler._warmup_in_flight_guard = threading.Lock()
    SwrHandler._WARM_DEBOUNCE_S = 0.01

    import pipeline

    old_fast_rerank = pipeline.fast_rerank_for_user
    old_gen_bytes = pipeline.generate_dashboard_bytes
    pipeline.fast_rerank_for_user = lambda db, c, e, uid: []  # type: ignore
    pipeline.generate_dashboard_bytes = lambda *a, **kw: b""  # type: ignore

    yield user, SwrHandler

    pipeline.fast_rerank_for_user = old_fast_rerank
    pipeline.generate_dashboard_bytes = old_gen_bytes


def test_dashboard_stale_hit_returns_when_version_mismatch(swr_handler):
    user, h = swr_handler
    stale_html = b"stale content"
    h._dashboard_cache[f"dashboard_{user.id}"] = (stale_html, time.time(), 0)
    h._dashboard_versions[user.id] = 1

    result = h._render_dashboard_for_user(user)
    assert result == stale_html


def test_dashboard_cache_hit_returns_when_version_matches(swr_handler):
    user, h = swr_handler
    fresh_html = b"fresh content"
    h._dashboard_cache[f"dashboard_{user.id}"] = (fresh_html, time.time(), 0)
    h._dashboard_versions[user.id] = 0

    result = h._render_dashboard_for_user(user)
    assert result == fresh_html


def test_trigger_warm_dedup(swr_handler):
    user, h = swr_handler
    h._trigger_warm(user, version=42)
    h._trigger_warm(user, version=42)

    with h._warmup_in_flight_guard:
        assert (user.id, 42) in h._warmup_in_flight
        assert len(h._warmup_in_flight) == 1


def test_trigger_warm_different_versions_not_deduped(swr_handler):
    user, h = swr_handler
    h._trigger_warm(user, version=1)
    h._trigger_warm(user, version=2)

    with h._warmup_in_flight_guard:
        assert len(h._warmup_in_flight) == 2


def test_enforce_cache_cap(swr_handler):
    user, h = swr_handler
    for i in range(102):
        h._dashboard_cache[f"dashboard_{i}"] = (b"", float(i), 0)

    h._enforce_cache_cap(max_entries=100)

    assert len(h._dashboard_cache) == 100
    assert "dashboard_0" not in h._dashboard_cache
    assert "dashboard_1" not in h._dashboard_cache
    assert "dashboard_99" in h._dashboard_cache
    assert "dashboard_101" in h._dashboard_cache


def test_bump_all_cached_versions(swr_handler):
    user, h = swr_handler
    h._dashboard_versions = {1: 5, 2: 10, 3: 0}
    h._bump_all_cached_versions()

    assert h._dashboard_versions[1] == 6
    assert h._dashboard_versions[2] == 11
    assert h._dashboard_versions[3] == 1


def test_setSort_triggers_silent_refill_on_tab_click() -> None:
    """Clicking a sort tab silently refills the queue (no manual refresh
    button — the user picks a sort, the new ranking is fetched in the
    background)."""
    _, static = _read_template_and_static()
    idx = static.index("function setSort(")
    end = static.index("function setAge(", idx)
    body = static[idx:end]
    assert "silentRefill" in body
    assert "refillQueued" not in body
    assert "refillWhenReady" not in body


def test_setAge_triggers_silent_refill_on_tab_click() -> None:
    """Clicking an age tab silently refills the queue."""
    _, static = _read_template_and_static()
    idx = static.index("function setAge(")
    end = static.index("function setSource(", idx)
    body = static[idx:end]
    assert "silentRefill" in body
    assert "refillQueued" not in body
    assert "refillWhenReady" not in body


def test_setSource_triggers_silent_refill_on_tab_click() -> None:
    """Clicking a source tab silently refills the queue."""
    _, static = _read_template_and_static()
    idx = static.index("function setSource(")
    # setSource is the last function in the file's first script block; the
    # body ends with the next blank line.
    end = static.index("\n\n    applyGradient();", idx)
    body = static[idx:end]
    assert "silentRefill" in body
    assert "refillQueued" not in body
    assert "refillWhenReady" not in body


def test_archive_age_tab_button_exists():
    """Archive age-tab button exists and idle prefetch pre-warms the other age."""
    template, static = _read_template_and_static()
    assert 'data-age="archive"' in template
    # Idle prefetch uses other-age logic.
    assert "function scheduleIdleAgePrefetch" in static
    assert "otherAge" in static
    assert "cardsForAge" in static


def test_matchesCurrentAxes_filters_by_age_and_sort():
    """matchesCurrentAxes filters by age (currentAge + data-is-recent)
    and sort (currentSort + data-sort-popular / data-sort-explore)."""
    _, static = _read_template_and_static()
    idx = static.index("function matchesCurrentAxes(")
    end = static.index("function queuedCards(", idx)
    body = static[idx:end]
    # Age checks: archive/recent branches reference isRecent.
    assert "currentAge === 'archive'" in body
    assert "currentAge === 'recent'" in body
    assert "card.dataset.isRecent" in body
    # Sort checks: popular/explore reference sortPopular / sortExplore.
    assert "card.dataset.sortPopular" in body
    assert "card.dataset.sortExplore" in body


def test_orderForCurrentSort_uses_orderByRank_for_recommended():
    """Recommended sort uses orderByRank (score desc); popular/explore shuffle."""
    _, static = _read_template_and_static()
    idx = static.index("function orderForCurrentSort(")
    # orderForCurrentSort is now the last function in its block (no
    # updateRefreshProgress follows). End at the blank line.
    end = static.index("\n\n    function setActiveCard(", idx)
    body = static[idx:end]
    assert "currentSort === 'recommended'" in body
    assert "orderByRank" in body
    assert "shuffleStories" in body


def test_data_is_recent_attribute_emitted(test_env):
    """Per-card data-is-recent attribute is set correctly based on story age."""
    import re

    port, db, regen_event, handler, user = test_env
    now = int(time.time())
    # 2 recent HN stories, 2 old archive stories. The reranker should
    # surface at least one of each group in the final deck.
    recent_id = 5001
    old_id = 5002
    db.upsert_story(
        Story(
            id=recent_id,
            title="Recent HN story",
            url=None,
            score=300,
            time=now - 3600,  # 1h old
            text_content="A recent HN story with content.",
            source="hn",
            comment_count=10,
        )
    )
    db.upsert_story(
        Story(
            id=old_id,
            title="Old archive story",
            url=None,
            score=200,
            time=now - 365 * 86400,  # 1 year old
            text_content="An old archive story with content.",
            source="ch_seed",
            comment_count=5,
        )
    )
    handler._render_dashboard_for_user(user)
    _wait_for_cache(handler, user, 0, timeout=3.0)
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    text = resp.text
    # Both cards should be in the HTML.
    recent_card_pat = re.search(
        rf'<article class="story-card[^"]*"[^>]*data-story-id="{recent_id}"[^>]*>',
        text,
        re.DOTALL,
    )
    old_card_pat = re.search(
        rf'<article class="story-card[^"]*"[^>]*data-story-id="{old_id}"[^>]*>',
        text,
        re.DOTALL,
    )
    assert recent_card_pat is not None, "Recent story card not in HTML"
    assert old_card_pat is not None, "Old story card not in HTML"
    recent_card = recent_card_pat.group(0)
    old_card = old_card_pat.group(0)
    assert 'data-is-recent="1"' in recent_card
    assert 'data-is-recent="0"' in old_card


def test_inline_script_has_voted_story_ids_filter():
    """The client-side ``votedStoryIds`` Set and the refillQueue filter are
    the defense-in-depth for the 2026-06-28 stale-fetch bug: even if a SWR
    stale-hit returns the pre-vote HTML, refillQueue suppresses any incoming
    card whose storyId is in the session-scoped voted set.
    """
    _, inline_script = _read_template_and_static()
    assert "votedStoryIds = new Set()" in inline_script
    # submitVote adds to the set; undoLastVote removes from it
    submit_vote_block = inline_script.split("function submitVote(", 1)[1].split(
        "function ", 1
    )[0]
    assert "votedStoryIds.add(storyId)" in submit_vote_block
    undo_block = inline_script.split("function undoLastVote()", 1)[1].split(
        "function ", 1
    )[0]
    assert "votedStoryIds.delete(storyId)" in undo_block
    # refillQueue must skip incoming cards whose id is in the set
    refill_block = inline_script.split("function refillQueue(", 1)[1].split(
        "function ", 1
    )[0]
    assert "votedStoryIds.has(Number(storyId))" in refill_block
    # The sendFeedback catch handler must roll back the in-memory "voted"
    # state when the request fails. Otherwise a transient network error
    # would leave the storyId in votedStoryIds for the rest of the session
    # and refillQueue would suppress that story from the next refill even
    # though no vote was actually saved to the DB.
    submit_catch = inline_script.split("Network error submitting feedback", 1)[1].split(
        ".finally", 1
    )[0]
    assert "votedStoryIds.delete(storyId)" in submit_catch
    assert "delete card.dataset.voted" in submit_catch


def test_submitVote_silently_refills_on_success() -> None:
    """A successful vote save triggers a silent queue refill (no manual
    refresh button — the user voted, the next queue is fetched in the
    background)."""
    _, inline_script = _read_template_and_static()
    submit_vote_block = inline_script.split("function submitVote(", 1)[1].split(
        "function ", 1
    )[0]
    assert "silentRefill()" in submit_vote_block
    # On a failed save, the catch handler must surface a toast (not the old
    # refresh banner, which is gone).
    submit_catch = submit_vote_block.split("Network error submitting feedback", 1)[1]
    assert "showToast(" in submit_catch
    assert "refreshBannerText" not in submit_catch
    assert "refreshBanner.hidden" not in submit_catch


def test_undoLastVote_silently_refills_on_success() -> None:
    """A successful undo triggers a silent queue refill; a failed undo
    surfaces a toast."""
    _, inline_script = _read_template_and_static()
    undo_block = inline_script.split("function undoLastVote()", 1)[1].split(
        "function ", 1
    )[0]
    assert "silentRefill()" in undo_block
    undo_catch = undo_block.split("Network error undoing feedback", 1)[1]
    assert "showToast(" in undo_catch
    assert "refreshBannerText" not in undo_catch


def test_silentRefill_serializes_concurrent_calls() -> None:
    """silentRefill returns early if a refill is already in flight, so
    bursty votes coalesce into one fetch per cycle instead of N parallel
    fetches."""
    _, inline_script = _read_template_and_static()
    block = inline_script.split("async function silentRefill()", 1)[1].split(
        "function ", 1
    )[0]
    assert "if (isRefilling) return" in block
    assert "refillQueue({ forceFetch: true })" in block


def test_refillQueue_reorders_deterministic_modes_only() -> None:
    """After appending new cards from the server (which always returns them
    in recommended order), refillQueue re-applies the active sort for the
    deterministic modes (recommended/date). Popular/explore (shuffle) are
    skipped to avoid reshuffling on every vote."""
    _, inline_script = _read_template_and_static()
    block = inline_script.split("async function refillQueue(", 1)[1].split(
        "function ", 1
    )[0]
    # The reorder call must be guarded by the deterministic modes.
    assert "orderForCurrentSort()" in block
    assert "currentSort === 'recommended'" in block
    assert "currentSort === 'date'" in block
    assert "if (currentSort === 'recommended' || currentSort === 'date')" in block


def test_showToast_dismisses_after_3s() -> None:
    """showToast shows the toast and auto-dismisses after 3000ms."""
    _, inline_script = _read_template_and_static()
    block = inline_script.split("function showToast(message, variant)", 1)[1].split(
        "function ", 1
    )[0]
    assert "toastEl.hidden = false" in block
    assert "toastEl.hidden = true" in block
    assert "3000" in block
    assert "clearTimeout(toastTimer)" in block
