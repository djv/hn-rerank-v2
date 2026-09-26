import threading
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import httpx
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from werkzeug.serving import make_server

from collections.abc import Callable
from typing import Any, cast

from server import DeckState, Handler, SKELETON_HTML, create_app
from pipeline import Config, Embedder, RankedStory
from database import Database, Story

import numpy as np


class _LocalHttp:
    """httpx's module-level helpers build a fresh client per call, and with
    it a TLS context loaded from the CA bundle (20-50 ms each). These tests
    only speak plain HTTP to 127.0.0.1, so skip certificate loading."""

    @staticmethod
    def get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.get(url, verify=False, **kwargs)

    @staticmethod
    def post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.post(url, verify=False, **kwargs)

    @staticmethod
    def options(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.options(url, verify=False, **kwargs)


local_http = _LocalHttp()


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

    def encode(self, texts: list[str], batch_size: int | None = None) -> Any:
        return np.zeros((len(texts), 384), dtype=np.float32)


def _read_template_and_static() -> tuple[str, str]:
    """Read both the Jinja2 template and the inline <script> block.

    Tests that look for JS code should check the inline script returned here,
    while tests that look for HTML attributes / Jinja2 directives check the full
    template source, including component partials. (The script is inline again
    after the static extraction was rolled back — see WORKLOG 2026-06-27.)
    """
    repo_root = Path(__file__).resolve().parents[1]
    index_template = (repo_root / "templates" / "index.html").read_text(
        encoding="utf-8"
    )
    component_dir = repo_root / "templates" / "components"
    component_sources = [
        path.read_text(encoding="utf-8")
        for path in sorted(component_dir.glob("*.html"))
    ]
    template = "\n".join([index_template, *component_sources])
    start = index_template.find("  <script>\n")
    end = index_template.find("  </script>\n", start)
    inline_script = index_template[start:end] if start >= 0 and end >= 0 else ""
    return template, inline_script


@pytest.fixture(scope="module")
def mock_embedder() -> MockEmbedder:
    """One MockEmbedder for the whole test_server module.

    Avoids re-allocating the (cheap) instance, but more importantly keeps
    embedding-dependent handler state (warmup-in-flight threads, dashboard
    caches keyed by user) consistent across tests that share the module.
    """
    return MockEmbedder()


def _reset_warm_state(handler: type[Handler]) -> None:
    feedback_regen_timer = getattr(handler, "_feedback_regen_timer", None)
    if feedback_regen_timer is not None:
        feedback_regen_timer.cancel()
    handler._scheduler = None
    handler._pool_generation = 1
    handler._feedback_warm_counts = {}
    handler._feedback_warm_guard = threading.Lock()
    handler._feedback_regen_timer = None
    handler._feedback_regen_guard = threading.Lock()


def _has_pending_warm(handler: type[Handler]) -> bool:
    scheduler = handler.__dict__.get("_scheduler")
    return scheduler is not None and scheduler.busy()


def _has_pending_feedback_regen(handler: type[Handler]) -> bool:
    with handler._feedback_regen_guard:
        return handler._feedback_regen_timer is not None


class _ControllableTimer(threading.Timer):
    """`threading.Timer` stand-in for deterministic timer tests.

    `start()` does not arm a real background wait; the callback only runs
    when `.fire()` is called explicitly. `.fire()` still executes on the
    timer's own thread (via the real `Thread.start`/`join`), so
    `threading.current_thread()` inside the callback is the timer object
    itself -- production code (`Handler._feedback_regen_idle_fired`) guards
    on that identity, and a stub that ran the callback inline would make
    that guard untestable. `cancel()` is inherited unmodified, so debounced
    timers behave exactly as in production: a `fire()` after `cancel()` is
    correctly a no-op.
    """

    def start(self) -> None:  # do not arm a real wait
        pass

    def fire(self) -> None:
        self.interval = 0
        threading.Thread.start(self)
        self.join(timeout=1.0)


def _controllable_timer_factory(
    created: list[_ControllableTimer],
) -> Callable[..., _ControllableTimer]:
    """Build a `server._TIMER_FACTORY` replacement that records every timer
    it creates into `created`, in creation order."""

    def factory(*args: Any, **kwargs: Any) -> _ControllableTimer:
        timer = _ControllableTimer(*args, **kwargs)
        created.append(timer)
        return timer

    return factory


def _drain_warms(handler: type[Handler], timeout_s: float = 3.0) -> None:
    scheduler = handler.__dict__.get("_scheduler")
    if scheduler is not None:
        scheduler.wait_idle(timeout_s)


def _cancel_warms(handler: type[Handler]) -> None:
    """Teardown: drop queued (e.g. vote-debounced) warms, finish running ones."""
    scheduler = handler.__dict__.get("_scheduler")
    if scheduler is not None:
        scheduler.clear_pending()
        scheduler.wait_idle(3.0)


def test_warm_job_collects_after_failed_warm(
    prop_db: Database, mock_embedder: MockEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=prop_db.db_path, server_port=0)
    TestHandler.db = prop_db
    TestHandler.embedder = mock_embedder
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)

    user = prop_db.create_user("warm_gc")
    calls: list[str] = []

    def fake_run(cls, user_arg, version):
        calls.append(f"run:{version}")
        raise RuntimeError("boom")

    def fake_collect(cls):
        calls.append("collect")

    monkeypatch.setattr(TestHandler, "_run_warm_attempt", classmethod(fake_run))
    monkeypatch.setattr(
        TestHandler, "_collect_after_warm_attempt", classmethod(fake_collect)
    )

    TestHandler._trigger_warm(user, 2)
    _drain_warms(TestHandler)

    assert calls == ["run:2", "collect"]
    assert not _has_pending_warm(TestHandler)


def _start_handler_server(
    db: Database, embedder: MockEmbedder, port: int = 0
) -> tuple[Any, int, type[Handler]]:
    """Spin up a Flask server on 127.0.0.1:<port>.

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
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)
    TestHandler.reset_public_demo_limiter()

    app = create_app(TestHandler)
    server = make_server("127.0.0.1", port, app, threaded=True)
    bound_port = server.server_port
    return server, bound_port, TestHandler


def _drain_and_shutdown(server: Any, handler: type[Handler]) -> None:
    with handler._feedback_warm_guard:
        handler._feedback_warm_counts.clear()
    handler._cancel_feedback_regen()
    _cancel_warms(handler)
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
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
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
    t = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    t.start()

    regen_event = TestHandler.regen_event

    yield port, db, regen_event, TestHandler, user

    # Drain any in-flight warm threads on the fixture's TestHandler before
    # teardown so they don't outlive the fixture and pollute the next test's
    # monkeypatched pipeline functions.
    _drain_and_shutdown(server, TestHandler)
    db.close()


@pytest.mark.parametrize("size", [0, 1499, 1500, 5000, 10440, 12000, 20000, 30000])
def test_fixed_pane_budget(size: int) -> None:
    import server

    assert server._section_budget(size, single_section=True).startswith(
        "6-8 bullets, aim for 240 words"
    )
    assert server._section_budget(size).startswith("3-4 bullets, aim for 120 words")
    assert "never pad or invent" in server._section_budget(size)


@pytest.mark.parametrize(
    ("template", "fields"),
    [
        ("discussion_only_v4.txt", {"comments_section": "a comment"}),
        (
            "article_only_v4.txt",
            {"article_section": "Author's text:\nsome article content"},
        ),
        ("article_v4.txt", {"article_section": "Author's text:\nsome article content"}),
        ("discussion_v4.txt", {"comments_section": "a comment"}),
    ],
)
def test_prompts_render_budget_placeholder(template: str, fields: dict) -> None:
    """Pins the {budget} placeholder contract in every TLDR prompt -- if a
    future prompt edit drops it, .format() must fail loudly here rather
    than silently ignoring the scaled budget."""
    import server

    prompt = server._load_prompt(template).format(
        title="Some story",
        budget=server._section_budget(5_000),
        **fields,
    )
    assert "3-4 bullets, aim for 120 words" in prompt
    assert "at most one `####` heading" in prompt


@pytest.mark.parametrize("template", ["article_v4.txt", "article_only_v4.txt"])
def test_article_prompts_require_full_piece_coverage(template: str) -> None:
    """Article prompts must instruct coverage of every major section — a
    lead-only summary drops trailing sections of long newsletters (Import AI
    473 lost its third topic starting 88% into the text)."""
    import server

    prompt = server._load_prompt(template)
    assert "each major section" in prompt


@pytest.mark.parametrize("template", ["discussion_only_v4.txt", "discussion_v4.txt"])
def test_discussion_prompts_use_freeform_headings(template: str) -> None:
    """Discussion prompts must ask for freeform thread-specific headings,
    never a fixed Consensus/Disagreement/Caveat label set — the fixed set
    makes the model repeat the same buckets on every story."""
    import server

    prompt = server._load_prompt(template)
    assert "####" in prompt
    for label in ("Consensus:", "Disagreement:", "Caveat:"):
        assert label not in prompt


def test_token_redirect(app_env):
    port, _, _, _, user = app_env
    resp = local_http.get(
        f"http://127.0.0.1:{port}/u/{user.token}", follow_redirects=False
    )
    assert resp.status_code == 302
    assert resp.headers["Location"] == "../"
    assert "hn_token" in resp.headers.get("Set-Cookie", "")


def test_token_redirect_unknown_token_does_not_create_user(app_env):
    port, db, _, _, _ = app_env
    with db.conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    resp = local_http.get(
        f"http://127.0.0.1:{port}/u/not-a-real-token",
        follow_redirects=False,
    )

    assert resp.status_code == 404
    with db.conn() as conn:
        after = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert after == before


def test_first_visit_serves_dashboard_and_sets_cookie(app_env):
    port, _, _, _, _ = app_env
    resp = local_http.get(f"http://127.0.0.1:{port}/", follow_redirects=False)
    assert resp.status_code == 200
    assert "Location" not in resp.headers
    assert "hn_token" in resp.headers.get("Set-Cookie", "")
    assert resp.content


def test_dashboard_route_no_user_creates_token_inline(app_env):
    port, db, _, _, _ = app_env
    with db.conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    resp = local_http.get(f"http://127.0.0.1:{port}/", follow_redirects=False)

    assert resp.status_code == 200
    assert "Location" not in resp.headers
    assert "hn_token" in resp.headers.get("Set-Cookie", "")
    with db.conn() as conn:
        after = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert after == before + 1

    cookie = resp.headers["Set-Cookie"].split(";", 1)[0].split("=", 1)[1]
    follow = local_http.get(f"http://127.0.0.1:{port}/", cookies={"hn_token": cookie})
    assert follow.status_code == 200
    with db.conn() as conn:
        persisted = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert persisted == after


def test_unknown_cookie_does_not_create_user(app_env) -> None:
    port, db, _, _, _ = app_env
    with db.conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    resp = local_http.get(
        f"http://127.0.0.1:{port}/api/user",
        cookies={"hn_token": "forged-token"},
    )

    assert resp.status_code == 401
    with db.conn() as conn:
        after = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert after == before


def test_dashboard_route_session_creation_limit_uses_forwarded_for(
    test_env,
) -> None:
    port, _, _, TestHandler, _ = test_env
    TestHandler.config = replace(
        TestHandler.config,
        session_create_per_ip_limit=1,
        session_create_per_ip_window_seconds=3600,
    )
    headers = {"X-Forwarded-For": "203.0.113.10, 127.0.0.1"}

    first = local_http.get(
        f"http://127.0.0.1:{port}/",
        headers=headers,
        follow_redirects=False,
    )
    second = local_http.get(
        f"http://127.0.0.1:{port}/",
        headers=headers,
        follow_redirects=False,
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "3600"


def test_session_creation_limit_ignores_spoofed_leftmost_forwarded_for(
    test_env,
) -> None:
    """Rotating a client-supplied leftmost XFF value must not mint a fresh
    bucket: the key is the rightmost non-loopback hop our edge appended."""
    port, _, _, TestHandler, _ = test_env
    TestHandler.config = replace(
        TestHandler.config,
        session_create_per_ip_limit=1,
        session_create_per_ip_window_seconds=3600,
    )

    statuses = [
        local_http.get(
            f"http://127.0.0.1:{port}/",
            headers={"X-Forwarded-For": f"198.51.100.{i}, 203.0.113.20, ::1"},
            follow_redirects=False,
        ).status_code
        for i in range(3)
    ]

    assert statuses == [200, 429, 429]


def test_fixed_window_limiter_sweeps_idle_buckets() -> None:
    from server import FixedWindowLimiter

    limiter = FixedWindowLimiter()
    sweep = FixedWindowLimiter.SWEEP_INTERVAL_SECONDS
    for i in range(50):
        assert limiter.try_acquire([(f"ip:{i}", 5, 60)], now=0.0).allowed
    assert limiter.try_acquire([("long", 1, 3600)], now=0.0).allowed

    # After the sweep interval, expired 60s buckets are dropped while the
    # still-active 1h bucket keeps enforcing its limit.
    assert not limiter.try_acquire([("long", 1, 3600)], now=sweep).allowed
    assert set(limiter._buckets) == {"long"}
    assert limiter.try_acquire([("ip:0", 5, 60)], now=sweep).allowed


def test_dashboard_authenticated_visit_does_not_consume_session_creation_quota(
    test_env,
) -> None:
    port, _, _, TestHandler, user = test_env
    TestHandler.config = replace(
        TestHandler.config,
        session_create_per_ip_limit=1,
        session_create_per_ip_window_seconds=3600,
    )
    headers = {"X-Forwarded-For": "203.0.113.11"}

    authenticated = local_http.get(
        f"http://127.0.0.1:{port}/",
        headers=headers,
        cookies={"hn_token": user.token},
        follow_redirects=False,
    )
    anonymous = local_http.get(
        f"http://127.0.0.1:{port}/",
        headers=headers,
        follow_redirects=False,
    )

    assert authenticated.status_code == 200
    assert anonymous.status_code == 200


def test_token_redirect_profile_link_limit_uses_forwarded_for(test_env) -> None:
    port, _, _, TestHandler, user = test_env
    TestHandler.config = replace(
        TestHandler.config,
        profile_link_per_ip_limit=1,
        profile_link_per_ip_window_seconds=3600,
    )
    headers = {"X-Forwarded-For": "203.0.113.12, 127.0.0.1"}

    first = local_http.get(
        f"http://127.0.0.1:{port}/u/{user.token}",
        headers=headers,
        follow_redirects=False,
    )
    second = local_http.get(
        f"http://127.0.0.1:{port}/u/{user.token}",
        headers=headers,
        follow_redirects=False,
    )

    assert first.status_code == 302
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "3600"


def test_static_serving(test_env):
    port, _, _, handler, user = test_env
    # Pre-warm cache so HTTP request hits cached dashboard.
    result = handler._render_dashboard_for_user(user)
    assert result == SKELETON_HTML
    # The skeleton queued a warm for the live version.
    _wait_for_cache(handler, user, handler._dashboard_version(user.id), timeout=3.0)
    # Now HTTP request should hit the cache
    resp = local_http.get(
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
    assert 'id="queue-loading"' in resp.text
    assert "Loading more stories…" in resp.text
    assert 'class="refresh-progress"' not in resp.text
    assert 'id="sort-toggle"' not in resp.text
    assert 'id="queue-status"' not in resp.text
    assert 'id="refresh-banner"' not in resp.text
    assert 'id="refresh-now-btn"' not in resp.text


def test_feedback_post(test_env):
    port, db, regen_event, handler, user = test_env
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
    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json=feedback_payload,
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "ranking_refresh_queued": False,
        "target_version": 2,
        "ranking_idle_seconds": 3.0,
    }

    records = db.get_all_feedback(user.id)
    assert len(records) == 1
    assert records[0].story_id == 999
    assert records[0].action == "up"
    assert not regen_event.is_set()
    assert _has_pending_feedback_regen(handler)


def test_feedback_post_rejects_invalid_action(test_env: Any) -> None:
    port, db, regen_event, handler, user = test_env
    db.upsert_story(
        Story(
            id=998,
            title="Invalid action story",
            url="https://example.com/invalid-action",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    regen_event.clear()

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 998, "action": "sideways"},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 400
    assert resp.json() == {"error": "Invalid feedback"}
    assert db.get_all_feedback(user.id) == []
    assert handler._dashboard_version(user.id) == handler._pool_generation
    assert not regen_event.is_set()
    assert not _has_pending_feedback_regen(handler)


def test_feedback_post_rejects_malformed_story_id(test_env: Any) -> None:
    port, db, regen_event, handler, user = test_env
    regen_event.clear()

    for story_id in ("999", None, True):
        resp = local_http.post(
            f"http://127.0.0.1:{port}/api/feedback",
            json={"story_id": story_id, "action": "up"},
            cookies={"hn_token": user.token},
        )
        assert resp.status_code == 400
        assert resp.json() == {"error": "Invalid feedback"}

    assert db.get_all_feedback(user.id) == []
    assert handler._dashboard_version(user.id) == handler._pool_generation
    assert not regen_event.is_set()


def test_feedback_post_invalidates_cache_and_defers_warm_until_idle(test_env):
    """A vote persists immediately while ranking waits for the idle cadence."""
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
    assert starting_version == handler._pool_generation

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1000, "action": "up", "queue_remaining": 8},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "ranking_refresh_queued": False,
        "target_version": starting_version + 1,
        "ranking_idle_seconds": handler.config.dashboard_warm_idle_seconds,
    }
    assert len(db.get_all_feedback(user.id)) == 1
    assert handler._dashboard_version(user.id) == starting_version + 1
    assert not regen_event.is_set()
    assert _has_pending_feedback_regen(handler)


def test_feedback_vote_threshold_queues_one_latest_warm(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    port, db, _, handler, user = test_env
    handler.config = replace(
        handler.config,
        dashboard_warm_vote_threshold=10,
        dashboard_warm_idle_seconds=60.0,
    )
    calls: list[tuple[int, int, float]] = []

    def fake_trigger_warm(
        cls: type[Handler], warm_user: Any, version: int, delay_s: float = 0.0
    ) -> None:
        calls.append((warm_user.id, version, delay_s))

    monkeypatch.setattr(handler, "_trigger_warm", classmethod(fake_trigger_warm))
    for story_id in range(1200, 1210):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Threshold story {story_id}",
                url=f"https://example.com/{story_id}",
                score=100,
                time=1600000000,
                text_content="Feedback body text",
                source="hn",
            )
        )

    responses = [
        local_http.post(
            f"http://127.0.0.1:{port}/api/feedback",
            json={"story_id": story_id, "action": "neutral"},
            cookies={"hn_token": user.token},
        ).json()
        for story_id in range(1200, 1210)
    ]

    assert [response["ranking_refresh_queued"] for response in responses] == [
        False
    ] * 9 + [True]
    # Each vote below the threshold (re)starts the idle wait; the 10th vote
    # asks for the latest version immediately.
    assert calls == [(user.id, 1 + n, 60.0) for n in range(1, 10)] + [
        (user.id, 11, 0.0)
    ]


def test_feedback_idle_threshold_queues_latest_warm(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Votes below the threshold debounce: one warm, for the latest version,
    after the idle pause."""
    _, _, _, handler, user = test_env
    handler.config = replace(
        handler.config,
        dashboard_warm_vote_threshold=10,
        dashboard_warm_idle_seconds=0.05,
    )
    ran: list[int] = []
    monkeypatch.setattr(
        handler,
        "_run_warm_attempt",
        classmethod(lambda cls, warm_user, version: ran.append(version)),
    )
    handler._schedule_feedback_warm(user, 2)
    time.sleep(0.02)
    handler._schedule_feedback_warm(user, 3)
    assert ran == []
    _drain_warms(handler)
    assert ran == [3]


def test_feedback_regen_timer_resets_across_users_and_signals_once(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All users share one trailing regeneration request.

    Uses a controllable timer (see `_controllable_timer_factory`) instead of
    a short real `feedback_regen_idle_seconds` + `time.sleep` budget: the
    prior version raced the wall clock and failed intermittently under
    parallel test workers (each of the three HTTP round-trips plus two 30ms
    sleeps had to fit inside a 150ms window). The invariant under test --
    each vote replaces the pending timer, and a burst produces exactly one
    regen signal -- does not need real time to verify.
    """
    import server

    port, db, regen_event, handler, user = test_env
    other_user = db.create_user("other_feedback_regen_user")
    timers: list[_ControllableTimer] = []
    monkeypatch.setattr(server, "_TIMER_FACTORY", _controllable_timer_factory(timers))
    set_calls: list[float] = []
    original_set = regen_event.set

    def counted_set() -> None:
        set_calls.append(time.monotonic())
        original_set()

    monkeypatch.setattr(regen_event, "set", counted_set)

    for story_id in (1300, 1301, 1302):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Shared regen story {story_id}",
                url=f"https://example.com/shared-regen-{story_id}",
                score=100,
                time=1600000000,
                text_content="Feedback body text",
                source="hn",
            )
        )

    first = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1300, "action": "up"},
        cookies={"hn_token": user.token},
    )
    assert first.status_code == 200
    second = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1301, "action": "down"},
        cookies={"hn_token": other_user.token},
    )
    assert second.status_code == 200
    third = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1302, "action": "neutral"},
        cookies={"hn_token": user.token},
    )
    assert third.status_code == 200
    assert user.id != other_user.id

    # Each vote cancelled the previous timer and armed a new one: three
    # distinct timer objects were created, and only the last is still live.
    assert len(timers) == 3
    assert timers[0].finished.is_set()
    assert timers[1].finished.is_set()
    assert not timers[2].finished.is_set()
    assert not regen_event.is_set()

    timers[2].fire()

    assert regen_event.is_set()
    assert len(set_calls) == 1
    assert not _has_pending_feedback_regen(handler)


def test_regeneration_start_cancels_pending_feedback_timer(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A periodic regeneration satisfies the pending delayed request."""
    import server

    _, _, regen_event, handler, _ = test_env
    timers: list[_ControllableTimer] = []
    monkeypatch.setattr(server, "_TIMER_FACTORY", _controllable_timer_factory(timers))
    set_calls: list[None] = []
    monkeypatch.setattr(regen_event, "set", lambda: set_calls.append(None))

    handler._schedule_feedback_regen()
    assert _has_pending_feedback_regen(handler)
    handler._cancel_feedback_regen()

    assert timers[0].finished.is_set()
    timers[0].fire()

    assert set_calls == []
    assert not _has_pending_feedback_regen(handler)


def test_feedback_post_limit_returns_429_without_write(test_env) -> None:
    port, db, regen_event, handler, user = test_env
    handler.config = Config(
        db_path=db.db_path,
        server_port=port,
        feedback_per_user_limit=1,
        feedback_per_user_window_seconds=600,
        feedback_global_limit=100,
    )
    for story_id in (1100, 1101):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Limited feedback story {story_id}",
                url=f"https://example.com/limited-feedback-{story_id}",
                score=100,
                time=1600000000,
                text_content="Feedback body text",
                source="hn",
            )
        )
    regen_event.clear()

    first = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1100, "action": "up"},
        cookies={"hn_token": user.token},
    )
    feedback_regen_timer = handler._feedback_regen_timer
    second = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1101, "action": "down"},
        cookies={"hn_token": user.token},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) > 0
    assert second.json()["retry_after"] == int(second.headers["Retry-After"])
    records = db.get_all_feedback(user.id)
    assert [(record.story_id, record.action) for record in records] == [(1100, "up")]
    assert handler._feedback_regen_timer is feedback_regen_timer


@pytest.mark.parametrize(
    "headers",
    [
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "https://attacker.example"},
    ],
)
def test_feedback_post_rejects_cross_site_posts(
    test_env, headers: dict[str, str]
) -> None:
    port, db, regen_event, handler, user = test_env
    db.upsert_story(
        Story(
            id=1102,
            title="Cross-site feedback story",
            url="https://example.com/cross-site-feedback",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    regen_event.clear()

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1102, "action": "up"},
        headers=headers,
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 403
    assert resp.json() == {"error": "Cross-site POSTs are not allowed"}
    assert db.get_all_feedback(user.id) == []
    assert handler._dashboard_version(user.id) == handler._pool_generation
    assert not regen_event.is_set()


def test_feedback_post_accepts_same_origin_post(test_env) -> None:
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=1103,
            title="Same-origin feedback story",
            url="https://example.com/same-origin-feedback",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1103, "action": "up"},
        headers={"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin"},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert len(db.get_all_feedback(user.id)) == 1


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

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 1001, "action": "up", "queue_remaining": 4},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "ranking_refresh_queued": False,
        "target_version": starting_version + 1,
        "ranking_idle_seconds": handler.config.dashboard_warm_idle_seconds,
    }
    assert handler._dashboard_version(user.id) == starting_version + 1
    assert not regen_event.is_set()
    assert _has_pending_feedback_regen(handler)


def test_feedback_post_refreshes_when_client_requests_ranking(test_env):
    port, db, regen_event, handler, user = test_env
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

    resp = local_http.post(
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
    assert resp.json() == {
        "ok": True,
        "ranking_refresh_queued": False,
        "target_version": 2,
        "ranking_idle_seconds": handler.config.dashboard_warm_idle_seconds,
    }
    assert len(db.get_all_feedback(user.id)) == 1
    assert not regen_event.is_set()
    assert _has_pending_feedback_regen(handler)


def test_feedback_post_bumps_cache_version_for_warm_rerender(test_env, monkeypatch):
    """End-to-end: vote on a story → cache version bumps → warm renders →
    cached HTML no longer contains the voted story.

    Regression for the 2026-06-28 bug where the dashboard served a
    pre-vote HTML deck via the SWR stale-hit path, re-injecting the
    just-voted story into the refill queue.
    """
    port, db, _, handler, user = test_env
    handler.config = replace(handler.config, dashboard_warm_vote_threshold=1)

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

    def fake_fast_rerank_for_user(database, config, embedder, user_id, **kwargs):
        return []

    def fake_generate_dashboard_bytes(
        ranked, config, database, user_id, user_token, **kwargs
    ):
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
    assert pre_version == handler._pool_generation

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": voted_story.id, "action": "up", "queue_remaining": 6},
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "ranking_refresh_queued": True,
        "target_version": pre_version + 1,
        "ranking_idle_seconds": handler.config.dashboard_warm_idle_seconds,
    }

    post_version = handler._dashboard_version(user.id)
    assert post_version == pre_version + 1, (
        "vote must bump the dashboard version so the SWR stale-hit "
        "cannot return the pre-vote HTML"
    )

    _wait_for_cache(handler, user, post_version)
    fresh_html = handler._render_dashboard_for_user(user)
    assert f"version={post_version} excluded={voted_story.id}" in fresh_html.decode()


def test_feedback_clear(test_env):
    port, db, regen_event, handler, user = test_env
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
    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json=clear_payload,
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "ranking_refresh_queued": False,
        "target_version": 2,
        "ranking_idle_seconds": handler.config.dashboard_warm_idle_seconds,
    }

    assert len(db.get_all_feedback(user.id)) == 0
    assert not regen_event.is_set()
    assert _has_pending_feedback_regen(handler)


def test_feedback_clear_without_existing_vote_is_noop(test_env) -> None:
    """Spec (specs/ranking-feedback.allium ClearVote): clearing a story with
    no existing vote must not queue a dashboard refresh, since nothing
    changed.
    """
    port, db, regen_event, handler, user = test_env
    db.upsert_story(
        Story(
            id=4242,
            title="No vote yet",
            url=None,
            score=1,
            time=1_600_000_000,
            text_content="Text",
            source="hn",
        )
    )
    assert db.get_all_feedback(user.id) == []
    regen_event.clear()

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/feedback",
        json={"story_id": 4242, "action": "clear"},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["ranking_refresh_queued"] is False
    assert db.get_all_feedback(user.id) == []
    assert not regen_event.is_set()
    assert not _has_pending_feedback_regen(handler)


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
        resp = local_http.post(
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


@pytest.mark.parametrize("version", ["", "abc", "-1", "1.2"])
def test_ranking_ready_rejects_invalid_version(test_env, version: str) -> None:
    port, _, _, _, user = test_env

    resp = local_http.get(
        f"http://127.0.0.1:{port}/api/ranking-ready?version={version}",
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 400


def test_ranking_ready_false_when_cache_missing_or_older(test_env, monkeypatch) -> None:
    port, _, _, handler, user = test_env
    calls: list[tuple[int, int, bool]] = []

    def fake_trigger_warm(
        cls, warm_user, version: int, delay_s: float = 0.0, expedite: bool = True
    ) -> None:
        calls.append((warm_user.id, version, expedite))

    monkeypatch.setattr(handler, "_trigger_warm", classmethod(fake_trigger_warm))
    target_version = handler._bump_user_version(user.id)

    missing_resp = local_http.get(
        f"http://127.0.0.1:{port}/api/ranking-ready?version={target_version}",
        cookies={"hn_token": user.token},
    )

    assert missing_resp.status_code == 200
    assert missing_resp.json() == {
        "ok": True,
        "ready": False,
        "ready_version": None,
        "min_version": target_version,
        "target_version": target_version,
        "current_version": target_version,
        "cached_version": None,
    }

    handler._decks[user.id] = DeckState([], time.time(), target_version - 1)
    older_resp = local_http.get(
        f"http://127.0.0.1:{port}/api/ranking-ready?version={target_version}",
        cookies={"hn_token": user.token},
    )

    assert older_resp.status_code == 200
    assert older_resp.json()["ready"] is False
    assert older_resp.json()["cached_version"] == target_version - 1
    # Polls are passive: they must not cut short a queued vote-debounce warm.
    assert calls == [(user.id, target_version, False), (user.id, target_version, False)]


def test_ranking_ready_true_only_from_cached_version(test_env, monkeypatch) -> None:
    port, _, _, handler, user = test_env
    calls: list[tuple[int, int]] = []

    def fake_trigger_warm(cls, warm_user, version: int, **_: Any) -> None:
        calls.append((warm_user.id, version))

    monkeypatch.setattr(handler, "_trigger_warm", classmethod(fake_trigger_warm))
    target_version = handler._bump_user_version(user.id)
    handler._decks[user.id] = DeckState([], time.time(), target_version)

    resp = local_http.get(
        f"http://127.0.0.1:{port}/api/ranking-ready?version={target_version}",
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "ready": True,
        "ready_version": target_version,
        "min_version": target_version,
        "target_version": target_version,
        "current_version": target_version,
        "cached_version": target_version,
    }
    assert calls == []


def test_ranking_ready_true_for_older_requested_version(test_env) -> None:
    port, _, _, handler, user = test_env
    newer_version = handler._bump_user_version(user.id)
    handler._decks[user.id] = DeckState([], time.time(), newer_version)

    resp = local_http.get(
        f"http://127.0.0.1:{port}/api/ranking-ready?version={newer_version - 1}",
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json()["ready"] is True
    assert resp.json()["ready_version"] == newer_version
    assert resp.json()["min_version"] == newer_version - 1
    assert resp.json()["cached_version"] == newer_version


def test_ranking_ready_returns_intermediate_cached_version(
    test_env, monkeypatch
) -> None:
    port, _, _, handler, user = test_env
    calls: list[tuple[int, int]] = []

    def fake_trigger_warm(cls, warm_user, version: int, **_: Any) -> None:
        calls.append((warm_user.id, version))

    monkeypatch.setattr(handler, "_trigger_warm", classmethod(fake_trigger_warm))
    for expected_version in (2, 3, 4):
        assert handler._bump_user_version(user.id) == expected_version
    handler._decks[user.id] = DeckState([], time.time(), 3)

    resp = local_http.get(
        f"http://127.0.0.1:{port}/api/ranking-ready?min_version=2&target_version=4",
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "ready": True,
        "ready_version": 3,
        "min_version": 2,
        "target_version": 4,
        "current_version": 4,
        "cached_version": 3,
    }
    assert calls == [(user.id, 4)]


def test_ranking_ready_version_param_remains_compat_alias(test_env) -> None:
    port, _, _, handler, user = test_env
    target_version = handler._bump_user_version(user.id)
    handler._decks[user.id] = DeckState([], time.time(), target_version)

    resp = local_http.get(
        f"http://127.0.0.1:{port}/api/ranking-ready?version={target_version}",
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json()["ready"] is True
    assert resp.json()["ready_version"] == target_version
    assert resp.json()["min_version"] == target_version


def _wait_for_cache(handler, user, expected_version, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        deck = handler._decks.get(user.id)
        if deck is not None and deck.version == expected_version:
            return deck
        time.sleep(0.001)
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
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)

    calls = []

    def fake_fast_rerank_for_user(database, config, embedder, user_id, **kwargs):
        calls.append(("rank", user_id))
        return []

    def fake_generate_dashboard_bytes(
        ranked, config, database, user_id, user_token, **kwargs
    ):
        version = TestHandler._dashboard_version(user_id)
        return f"version={version}".encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    # No deck and an empty pool: skeleton now, warm queued for version 1.
    assert TestHandler._render_dashboard_for_user(user) == SKELETON_HTML
    _wait_for_cache(TestHandler, user, 1)
    assert len(calls) == 1

    # Second call hits the cached deck.
    assert TestHandler._render_dashboard_for_user(user) == b"version=1"
    assert len(calls) == 1

    # A vote bumps the version.
    version = TestHandler._bump_user_version(user.id)
    assert version == 2

    # SWR: the stale deck still renders immediately and a warm is queued.
    TestHandler._render_dashboard_for_user(user)
    _wait_for_cache(TestHandler, user, 2)
    assert TestHandler._render_dashboard_for_user(user) == b"version=2"
    assert len(calls) == 2


@pytest.mark.parametrize("vote_counter", [3, 0])
def test_no_cache_user_gets_cold_deck_and_warm_is_scheduled(
    test_env, monkeypatch: pytest.MonkeyPatch, vote_counter: int
) -> None:
    """A voted user without a deck gets the cold deck (as version 0) and a
    warm. The pool generation starts at 1, so the target is always ahead of
    the cold deck -- including right after a restart, when the in-memory vote
    counter is 0 -- and clients poll for the personalized deck."""
    _, db, _, handler, user = test_env
    story = Story(
        id=991,
        title="Cold fallback",
        url="https://example.com/cold",
        score=99,
        time=int(time.time()) - 3600,
        text_content="Cold fallback body",
        source="hn",
        comment_count=1,
    )
    db.upsert_story(story)
    db.upsert_feedback(user.id, 991, "up")
    # Unvoted story that survives the per-user cold-deck filter.
    unvoted = Story(
        id=992,
        title="Remaining story",
        url="https://example.com/remain",
        score=50,
        time=int(time.time()) - 3600,
        text_content="Remaining body",
        source="hn",
        comment_count=1,
    )
    db.upsert_story(unvoted)
    calls: list[tuple[int, int]] = []
    rendered: list[dict[str, object]] = []
    handler._decks = {}
    handler._dashboard_versions = {user.id: vote_counter}
    target_version = handler._pool_generation + vote_counter

    def fake_generate_dashboard_bytes(
        ranked: list[RankedStory],
        config: Config,
        database: Database,
        user_id: int | None,
        user_token: str | None,
        **kwargs: object,
    ) -> bytes:
        rendered.append(
            {
                "ranked": ranked,
                "user_id": user_id,
                "user_token": user_token,
                **kwargs,
            }
        )
        return b"cold html"

    def fake_trigger_warm(
        cls, warm_user, version: int, delay_s: float = 0.0, **_: Any
    ) -> None:
        calls.append((warm_user.id, version))

    import pipeline

    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )
    monkeypatch.setattr(handler, "_trigger_warm", classmethod(fake_trigger_warm))

    html = handler._render_dashboard_for_user(user)

    assert html == b"cold html"
    assert calls == [(user.id, target_version)]
    rank = rendered[0]
    ranked_list = cast(list[RankedStory], rank["ranked"])
    story_ids = [rs.story.id for rs in ranked_list]
    assert story_ids == [992]
    assert 991 not in story_ids
    assert rank["user_id"] == user.id
    assert rank["user_token"] == user.token
    assert rank["dashboard_version"] == 0
    assert rank["dashboard_latest_version"] == target_version


def test_no_cache_zero_feedback_user_gets_cold_deck_no_warm(
    test_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold deck for a 0-feedback user — instant, no redundant warm."""
    _, db, _, handler, user = test_env
    cold = [
        RankedStory(
            story=Story(
                id=992,
                title="Cold zero",
                url="https://example.com/cold0",
                score=50,
                time=int(time.time()) - 3600,
                text_content="body",
                source="hn",
                comment_count=1,
            ),
            score=50.0,
            best_match_title="",
            is_recent=True,
            combo_keys="recent_hn recent_mixed",
        )
    ]
    calls: list[tuple[int, int]] = []
    handler._decks = {}
    handler._dashboard_versions = {user.id: 0}
    handler._cold_stories = cold

    def fake_generate_dashboard_bytes(
        ranked: list[RankedStory],
        config: Config,
        database: Database,
        user_id: int | None,
        user_token: str | None,
        **kwargs: object,
    ) -> bytes:
        return b"cold html"

    def fake_trigger_warm(cls, warm_user, version: int, **_: Any) -> None:
        calls.append((warm_user.id, version))

    import pipeline

    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )
    monkeypatch.setattr(handler, "_trigger_warm", classmethod(fake_trigger_warm))

    html = handler._render_dashboard_for_user(user)

    assert html == b"cold html"
    assert calls == []


def test_stale_warm_render_does_not_overwrite_current_cache(
    test_env, mock_embedder, monkeypatch
):
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = mock_embedder
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)

    def fake_fast_rerank_for_user(database, config, embedder, user_id, **kwargs):
        return []

    def fake_generate_dashboard_bytes(
        ranked, config, database, user_id, user_token, **kwargs
    ):
        return f"requested={user_token}".encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    TestHandler._bump_user_version(user.id)
    new_version = TestHandler._bump_user_version(user.id)
    assert new_version == 3

    TestHandler._trigger_warm(user, new_version)
    current = _wait_for_cache(TestHandler, user, new_version)

    # A late job for an older version must not replace the newer deck.
    TestHandler._run_warm_attempt(user, new_version - 1)
    assert TestHandler._decks[user.id] is current


def test_active_warm_commits_when_dashboard_version_advances(
    test_env, mock_embedder, monkeypatch
) -> None:
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = mock_embedder
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)

    rank_started = threading.Event()
    allow_rank_to_finish = threading.Event()

    def fake_fast_rerank_for_user(database, config, embedder, user_id, **kwargs):
        rank_started.set()
        assert allow_rank_to_finish.wait(timeout=2.0)
        return []

    def fake_generate_dashboard_bytes(
        ranked, config, database, user_id, user_token, **kwargs
    ):
        return b"fresh content"

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    TestHandler._decks[user.id] = DeckState([], time.time(), 0)
    TestHandler._dashboard_versions[user.id] = 1

    TestHandler._trigger_warm(user, version=2)
    assert rank_started.wait(timeout=2.0)

    bumped_version = TestHandler._bump_user_version(user.id)
    assert bumped_version == 3

    allow_rank_to_finish.set()

    _drain_warms(TestHandler)

    # The in-flight warm still commits the version it was asked for; the
    # newer version is behind it, so the next read queues another warm.
    assert TestHandler._decks[user.id].version == 2

    rank_perf_rows = db.execute("SELECT COUNT(*) FROM rank_perf")
    assert rank_perf_rows[0][0] == 1


def test_rapid_vote_warms_coalesce_to_latest_version(
    test_env, mock_embedder, monkeypatch
) -> None:
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = mock_embedder
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)

    ranked_versions: list[int] = []

    def fake_fast_rerank_for_user(database, config, embedder, user_id, **kwargs):
        ranked_versions.append(TestHandler._dashboard_version(user_id))
        return []

    def fake_generate_dashboard_bytes(
        ranked, config, database, user_id, user_token, **kwargs
    ):
        version = TestHandler._dashboard_version(user_id)
        return f"version={version}".encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    TestHandler.config = replace(
        TestHandler.config,
        dashboard_warm_vote_threshold=10,
        dashboard_warm_idle_seconds=0.05,
    )
    for expected_version in (2, 3, 4):
        version = TestHandler._bump_user_version(user.id)
        assert version == expected_version
        TestHandler._schedule_feedback_warm(user, version)

    _wait_for_cache(TestHandler, user, expected_version=4)

    assert TestHandler._render_dashboard_for_user(user) == b"version=4"
    assert ranked_versions == [4]


def test_warm_loops_to_newer_version_requested_while_ranking(
    test_env, mock_embedder, monkeypatch
) -> None:
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = mock_embedder
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)

    rank_started = threading.Event()
    allow_first_rank_to_finish = threading.Event()
    ranked_versions: list[int] = []

    def fake_fast_rerank_for_user(database, config, embedder, user_id, **kwargs):
        ranked_versions.append(TestHandler._dashboard_version(user_id))
        if len(ranked_versions) == 1:
            rank_started.set()
            assert allow_first_rank_to_finish.wait(timeout=2.0)
        return []

    def fake_generate_dashboard_bytes(
        ranked, config, database, user_id, user_token, **kwargs
    ):
        version = TestHandler._dashboard_version(user_id)
        return f"version={version}".encode()

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes
    )

    version_1 = TestHandler._bump_user_version(user.id)
    assert version_1 == 2
    TestHandler._trigger_warm(user, version=version_1)
    assert rank_started.wait(timeout=2.0)

    version_2 = TestHandler._bump_user_version(user.id)
    assert version_2 == 3
    TestHandler._trigger_warm(user, version=version_2)
    allow_first_rank_to_finish.set()

    _wait_for_cache(TestHandler, user, expected_version=3)

    assert TestHandler._render_dashboard_for_user(user) == b"version=3"
    assert ranked_versions == [2, 3]


@pytest.fixture(scope="module")
def prop_db():
    with TemporaryDirectory() as temp_dir:
        db = Database(str(Path(temp_dir) / "prop_server.db"))
        yield db
        db.close()


@given(
    operations=st.lists(
        st.sampled_from(["invalidate", "pool_changed", "render"]),
        min_size=1,
        max_size=40,
    )
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_dashboard_cache_version_invariant_property(
    operations: list[str],
    prop_db: Database,
    mock_embedder: MockEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with prop_db.conn() as conn:
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
    TestHandler._decks = {}
    TestHandler._dashboard_versions = {}
    TestHandler._cold_stories = []
    _reset_warm_state(TestHandler)

    def fake_fast_rerank_for_user(database, config, embedder, user_id, **kwargs):
        return []

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(pipeline, "generate_dashboard_bytes", _fake_render)
    monkeypatch.setattr(
        TestHandler, "_rebuild_cold_deck", classmethod(lambda cls: None)
    )

    for operation in operations:
        if operation == "invalidate":
            TestHandler._bump_user_version(user.id)
        elif operation == "pool_changed":
            TestHandler._pool_changed()
        else:
            current = TestHandler._dashboard_version(user.id)
            rendered = TestHandler._render_dashboard_for_user(user)
            if rendered != SKELETON_HTML:
                # Every render reports the live version as its target and
                # never claims to be newer than it.
                page_version, target = map(
                    int, rendered.decode().split()[0][2:].split("/")
                )
                assert target == current
                assert page_version <= current
            # Rendering queues whatever warm is needed: the deck catches up.
            _wait_for_cache(TestHandler, user, current)

        deck = TestHandler._decks.get(user.id)
        if deck is not None:
            assert deck.version <= TestHandler._dashboard_version(user.id)

    # Drain in-flight warm threads before monkeypatch cleanup so they don't
    # capture our fakes and leak into subsequent tests.
    _drain_warms(TestHandler)


def test_cors_headers(app_env):
    port, _, _, _, _ = app_env
    resp = local_http.options(f"http://127.0.0.1:{port}/api/feedback")
    assert resp.status_code == 204
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/user", None),
        ("GET", "/api/feed", None),
        ("GET", "/api/deck-cards", None),
        ("GET", "/api/ranking-ready?version=0", None),
        ("GET", "/api/tldr-cache/1", None),
        ("POST", "/api/feedback", {"story_id": 1, "action": "up"}),
    ],
)
def test_session_scoped_endpoints_reject_missing_cookie(
    app_env: Any, method: str, path: str, body: dict[str, int | str] | None
) -> None:
    """Without a session cookie every per-user endpoint answers 401 and none
    of them silently creates a user (only GET / does that)."""
    _, db, _, handler, _ = app_env
    client = create_app(handler).test_client()
    with db.conn() as conn:
        users_before = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    resp = client.open(
        path, method=method, json=body, headers={"Origin": "http://localhost"}
    )

    assert resp.status_code == 401
    with db.conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == users_before


def test_flask_test_client_user_returns_session(app_env: Any) -> None:
    _, _, _, handler, user = app_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    resp = client.get("/api/user")

    assert resp.status_code == 200
    assert resp.get_json() == {"user_id": user.id, "token": user.token}


def test_flask_test_client_options_preserves_cors(app_env) -> None:
    _, _, _, handler, _ = app_env
    client = create_app(handler).test_client()

    resp = client.options("/api/feedback")

    assert resp.status_code == 204
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


def test_flask_test_client_first_visit_sets_cookie(app_env) -> None:
    _, db, _, handler, _ = app_env
    client = create_app(handler).test_client()
    with db.conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    resp = client.get("/")

    assert resp.status_code == 200
    assert resp.headers.get("Set-Cookie", "").startswith("hn_token=")
    assert resp.data
    with db.conn() as conn:
        after = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert after == before + 1


def test_flask_test_client_first_visit_limit_sets_retry_after(
    test_env: Any,
) -> None:
    _, _, _, handler, _ = test_env
    handler.config = replace(
        handler.config,
        session_create_per_ip_limit=1,
        session_create_per_ip_window_seconds=3600,
    )
    client = create_app(handler).test_client(use_cookies=False)
    headers = {"X-Forwarded-For": "203.0.113.30, 127.0.0.1"}

    first = client.get("/", headers=headers)
    second = client.get("/", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "3600"
    assert second.get_json() == {
        "error": "Too many new demo sessions from this address. Please try again later.",
        "retry_after": 3600,
    }


def test_flask_test_client_profile_link_sets_existing_session(
    app_env: Any,
) -> None:
    _, _, _, handler, user = app_env
    client = create_app(handler).test_client()

    resp = client.get(f"/u/{user.token}")

    assert resp.status_code == 302
    assert resp.headers["Location"] == "../"
    assert resp.headers.get("Set-Cookie", "").startswith(f"hn_token={user.token}")


def test_flask_test_client_profile_link_limit_uses_forwarded_for(
    test_env: Any,
) -> None:
    _, _, _, handler, user = test_env
    handler.config = replace(
        handler.config,
        profile_link_per_ip_limit=1,
        profile_link_per_ip_window_seconds=3600,
    )
    client = create_app(handler).test_client()
    headers = {"X-Forwarded-For": "203.0.113.31, 127.0.0.1"}

    first = client.get(f"/u/{user.token}", headers=headers)
    second = client.get(f"/u/{user.token}", headers=headers)

    assert first.status_code == 302
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "3600"
    assert second.get_json() == {
        "error": "Too many profile-link attempts. Please try again later.",
        "retry_after": 3600,
    }


def test_flask_test_client_ranking_ready_validates_version(app_env: Any) -> None:
    _, _, _, handler, user = app_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    resp = client.get("/api/ranking-ready?version=-1")

    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Invalid version"}


def test_flask_test_client_ranking_ready_reports_missing_cache(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, handler, user = test_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    calls: list[tuple[int, int]] = []

    def fake_trigger_warm(
        cls: type[Handler], warm_user: Any, version: int, **_: Any
    ) -> None:
        calls.append((warm_user.id, version))

    monkeypatch.setattr(handler, "_trigger_warm", classmethod(fake_trigger_warm))
    version = handler._bump_user_version(user.id)

    resp = client.get(f"/api/ranking-ready?min_version={version}&target_version=3")

    assert resp.status_code == 200
    assert resp.get_json() == {
        "ok": True,
        "ready": False,
        "ready_version": None,
        "min_version": version,
        "target_version": 3,
        "current_version": version,
        "cached_version": None,
    }
    assert calls == [(user.id, version)]


def test_flask_test_client_feedback_rejects_cross_site(app_env: Any) -> None:
    _, _, _, handler, _ = app_env
    client = create_app(handler).test_client()

    resp = client.post(
        "/api/feedback",
        json={"story_id": 1, "action": "up"},
        headers={"Origin": "https://evil.example", "Host": "localhost"},
    )

    assert resp.status_code == 403
    assert resp.get_json() == {"error": "Cross-site POSTs are not allowed"}


def test_flask_test_client_feedback_writes_and_queues_refresh(test_env: Any) -> None:
    _, db, regen_event, handler, user = test_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    db.upsert_story(
        Story(
            id=1710,
            title="Flask feedback story",
            url="https://example.com/flask-feedback",
            score=100,
            time=1600000000,
            text_content="Feedback body text",
            source="hn",
        )
    )
    regen_event.clear()

    resp = client.post("/api/feedback", json={"story_id": 1710, "action": "up"})

    assert resp.status_code == 200
    assert resp.get_json() == {
        "ok": True,
        "ranking_refresh_queued": False,
        "target_version": 2,
        "ranking_idle_seconds": handler.config.dashboard_warm_idle_seconds,
    }
    records = db.get_all_feedback(user.id)
    assert [(record.story_id, record.action) for record in records] == [(1710, "up")]
    assert handler._dashboard_version(user.id) == 2
    assert not regen_event.is_set()
    assert _has_pending_feedback_regen(handler)


def test_flask_test_client_feedback_rejects_invalid_payload(test_env: Any) -> None:
    _, db, regen_event, handler, user = test_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    regen_event.clear()

    resp = client.post("/api/feedback", json={"story_id": "1711", "action": "up"})

    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Invalid feedback"}
    assert db.get_all_feedback(user.id) == []
    assert handler._dashboard_version(user.id) == handler._pool_generation
    assert not regen_event.is_set()


def test_flask_test_client_events_require_same_origin_and_session(app_env: Any) -> None:
    _, _, _, handler, _ = app_env
    client = create_app(handler).test_client()

    cross_site = client.post(
        "/api/interaction",
        json={"events": []},
        headers={"Origin": "https://evil.example", "Host": "localhost"},
    )
    assert cross_site.status_code == 403

    no_session = client.post("/api/interaction", json={"events": []})
    assert no_session.status_code == 401
    assert no_session.get_json() == {"error": "No session"}


def test_flask_test_client_events_batch_and_replay_are_idempotent(
    test_env: Any,
) -> None:
    _, db, _, handler, user = test_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    db.upsert_story(
        Story(
            id=2710,
            title="Ledger API story",
            url="https://example.com/ledger-api",
            score=10,
            time=1600000000,
            text_content="Ledger body",
            source="hn",
        )
    )
    event = {
        "event_id": "55555555-5555-4555-8555-555555555555",
        "client_session_id": "66666666-6666-4666-8666-666666666666",
        "story_id": 2710,
        "event_type": "impression",
        "dashboard_version": 0,
        "position": 0,
        "sort_mode": "recommended",
        "age_filter": "recent",
        "source_filter": "mixed",
        "ranker_arm": "baseline",
        "occurred_at": 1_700_000_000.0,
    }

    first = client.post("/api/interaction", json={"events": [event, event]})
    replay = client.post("/api/interaction", json={"events": [event]})

    assert first.status_code == 200
    assert first.get_json() == {
        "ok": True,
        "inserted": 1,
        "duplicates": 1,
        "rejected": 0,
    }
    assert replay.status_code == 200
    assert replay.get_json() == {
        "ok": True,
        "inserted": 0,
        "duplicates": 1,
        "rejected": 0,
    }
    with db.conn() as conn:
        assert conn.execute(
            "SELECT user_id, story_id, event_type FROM interaction_events"
        ).fetchall() == [(user.id, 2710, "impression")]


def _ledger_event(event_id: str, story_id: int) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "client_session_id": "88888888-8888-4888-8888-888888888888",
        "story_id": story_id,
        "event_type": "impression",
        "dashboard_version": 0,
        "position": 0,
        "sort_mode": "recommended",
        "age_filter": "recent",
        "source_filter": "mixed",
        "ranker_arm": "baseline",
        "occurred_at": 1_700_000_000.0,
    }


def test_flask_test_client_events_accept_negative_story_ids(test_env: Any) -> None:
    _, db, _, handler, user = test_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    db.upsert_story(
        Story(
            id=-42,
            title="Synthetic RSS story",
            url="https://example.com/rss-item",
            score=0,
            time=1600000000,
            text_content="Feed body",
            source="rss_example_com",
        )
    )
    event = _ledger_event("77777777-7777-4777-8777-777777777777", -42)
    response = client.post("/api/interaction", json={"events": [event]})
    assert response.status_code == 200
    assert response.get_json() == {
        "ok": True,
        "inserted": 1,
        "duplicates": 0,
        "rejected": 0,
    }
    with db.conn() as conn:
        assert conn.execute("SELECT story_id FROM interaction_events").fetchall() == [
            (-42,)
        ]


def test_flask_test_client_events_skip_invalid_and_unknown_per_event(
    test_env: Any,
) -> None:
    _, db, _, handler, user = test_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    db.upsert_story(
        Story(
            id=2711,
            title="Ledger valid story",
            url="https://example.com/ledger-valid",
            score=10,
            time=1600000000,
            text_content="Ledger body",
            source="hn",
        )
    )
    unknown = _ledger_event("99999999-9999-4999-8999-999999999999", 999999)
    malformed = _ledger_event("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", 2711)
    malformed["event_type"] = "hover"
    valid = _ledger_event("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", 2711)
    response = client.post(
        "/api/interaction", json={"events": [unknown, malformed, valid]}
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "ok": True,
        "inserted": 1,
        "duplicates": 0,
        "rejected": 2,
    }
    with db.conn() as conn:
        assert conn.execute("SELECT story_id FROM interaction_events").fetchall() == [
            (2711,)
        ]


def test_flask_test_client_feedback_limit_sets_retry_after(test_env: Any) -> None:
    _, db, regen_event, handler, user = test_env
    handler.config = replace(
        handler.config,
        feedback_per_user_limit=1,
        feedback_per_user_window_seconds=600,
        feedback_global_limit=100,
    )
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    for story_id in (1712, 1713):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Flask limited feedback story {story_id}",
                url=f"https://example.com/flask-limited-feedback-{story_id}",
                score=100,
                time=1600000000,
                text_content="Feedback body text",
                source="hn",
            )
        )
    regen_event.clear()

    first = client.post("/api/feedback", json={"story_id": 1712, "action": "up"})
    second = client.post("/api/feedback", json={"story_id": 1713, "action": "down"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) > 0
    assert second.get_json()["retry_after"] == int(second.headers["Retry-After"])
    records = db.get_all_feedback(user.id)
    assert [(record.story_id, record.action) for record in records] == [(1712, "up")]


def test_flask_test_client_tldr_missing_story(app_env: Any) -> None:
    _, _, _, handler, _ = app_env
    client = create_app(handler).test_client()

    resp = client.post("/api/tldr-detail", json={"story_id": 999999})

    assert resp.status_code == 404
    assert resp.get_json() == {"error": "Story not found in database"}


def test_flask_test_client_tldr_rejects_cross_site_before_handler(
    app_env: Any,
) -> None:
    _, _, _, handler, _ = app_env
    client = create_app(handler).test_client()

    resp = client.post(
        "/api/tldr-detail",
        json={"story_id": 999999},
        headers={"Sec-Fetch-Site": "cross-site"},
    )

    assert resp.status_code == 403
    assert resp.get_json() == {"error": "Cross-site POSTs are not allowed"}


def test_flask_test_client_tldr_cached_bypasses_uncached_quota(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import server

    _, db, _, handler, user = test_env
    handler.config = replace(
        handler.config,
        tldr_uncached_per_user_limit=1,
        tldr_uncached_per_user_window_seconds=3600,
        tldr_uncached_global_limit=100,
    )
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    uncached_story = Story(
        id=1714,
        title="Flask uncached quota consumer",
        url="https://example.com/flask-uncached-quota",
        score=10,
        time=1600000000,
        text_content="Flask uncached quota consumer. Body.",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Body.",
    )
    cached_story = Story(
        id=1715,
        title="Flask cached quota bypass",
        url="https://example.com/flask-cached-quota",
        score=10,
        time=1600000000,
        text_content="Flask cached quota bypass. Cached body.",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Cached body.",
    )
    db.upsert_story(uncached_story)
    db.upsert_story(cached_story)
    cached_key = server._tldr_cache_key(
        title=cached_story.title,
        self_text="",
        top_comments="",
        article_body="Cached body.",
    )
    db.upsert_tldr_cache(cached_story.id, cached_key, "Already cached")

    calls = 0

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        nonlocal calls
        calls += 1
        return server.TldrResult(kind="ok", tldr=f"generated-{calls}: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    first = client.post("/api/tldr-detail", json={"story_id": uncached_story.id})
    second = client.post("/api/tldr-detail", json={"story_id": cached_story.id})

    assert first.status_code == 200
    assert first.get_json()["cached"] is False
    assert second.status_code == 200
    assert second.get_json() == {
        "ok": True,
        "tldr": "Already cached",
        "cached": True,
    }
    assert calls == 1


def test_flask_test_client_tldr_uncached_limit_sets_retry_after(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import server

    _, db, _, handler, user = test_env
    handler.config = replace(
        handler.config,
        tldr_uncached_per_user_limit=1,
        tldr_uncached_per_user_window_seconds=3600,
        tldr_uncached_global_limit=100,
    )
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    for story_id in (1716, 1717):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Flask per-user TLDR story {story_id}",
                url=f"https://example.com/flask-per-user-tldr-{story_id}",
                score=10,
                time=1600000000,
                text_content="Story body.",
                source="hn",
                comment_count=0,
                self_text="",
                top_comments="",
                article_body="Story body.",
            )
        )

    calls: list[str] = []

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        calls.append(title)
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    first = client.post("/api/tldr-detail", json={"story_id": 1716})
    second = client.post("/api/tldr-detail", json={"story_id": 1717})

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) > 0
    assert second.get_json()["retry_after"] == int(second.headers["Retry-After"])
    assert calls == ["Flask per-user TLDR story 1716"]


def test_flask_test_client_tldr_forces_refresh_for_active_thread(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recent, high-velocity HN thread should trigger a force=True
    fetch_story call even though top_comments is already populated from
    prewarm -- CH prewarm data can be 1-24h stale for brand-new comments."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    active_story = Story(
        id=1720,
        title="Active thread story",
        url="https://example.com/active-thread",
        score=200,
        time=int(now - 2 * 3600),  # 2h old
        text_content="Active thread story. Body.",
        source="hn",
        comment_count=200,  # 100 comments/hour
        comment_count_at_fetch=200,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(active_story)
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": active_story.id})

    assert resp.status_code == 200
    assert calls == [{"sid": active_story.id, "force": True}]


def test_flask_test_client_tldr_forces_refresh_for_active_thread_even_when_cached(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An active thread must bypass a pre-existing cached TLDR too -- the
    force-refresh must not be short-circuited by the early cache-hit check,
    since serving a cached summary forever was the original bug."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    active_story = Story(
        id=1730,
        title="Active thread story with stale cache",
        url="https://example.com/active-thread-cached",
        score=200,
        time=int(now - 2 * 3600),  # 2h old
        text_content="Active thread story with stale cache. Body.",
        source="hn",
        comment_count=200,  # 100 comments/hour
        comment_count_at_fetch=200,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(active_story)
    cached_key = server._tldr_cache_key(
        title=active_story.title,
        self_text=active_story.self_text or "",
        top_comments=active_story.top_comments or "",
        article_body=active_story.article_body or "",
    )
    db.upsert_tldr_cache(active_story.id, cached_key, "Stale cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": active_story.id})

    assert resp.status_code == 200
    assert calls == [{"sid": active_story.id, "force": True}]
    # The stale cached summary must not be the one returned -- fetch_story
    # is mocked to a no-op, so the post-refresh cache_key is unchanged and
    # the (still-stale) cached summary is what's served back, but the
    # important assertion is that the refresh path fired at all.
    assert resp.get_json()["tldr"] == "Stale cached TLDR"


def test_flask_test_client_tldr_skips_refresh_for_cached_quiet_recent_thread(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached, quiet/recent HN thread whose live count hasn't moved (tap
    probe miss) must still take the early cache-hit path and must NOT
    trigger fetch_story -- regression guard for the cache-bypass added
    alongside the active-thread force-refresh."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    quiet_story = Story(
        id=1731,
        title="Quiet recent story with cache",
        url="https://example.com/quiet-recent-cached",
        score=10,
        time=int(now - 2 * 3600),  # 2h old
        text_content="Quiet recent story with cache. Body.",
        source="hn",
        comment_count=5,  # 2.5 comments/hour, below default floor + velocity
        comment_count_at_fetch=5,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(quiet_story)
    cached_key = server._tldr_cache_key(
        title=quiet_story.title,
        self_text=quiet_story.self_text or "",
        top_comments=quiet_story.top_comments or "",
        article_body=quiet_story.article_body or "",
    )
    db.upsert_tldr_cache(quiet_story.id, cached_key, "Cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    probe_calls: list[list[int]] = []

    async def mock_probe_live_counts(stories, timeout_s):
        probe_calls.append([s.id for s in stories])
        return {}

    monkeypatch.setattr("pipeline._probe_live_counts", mock_probe_live_counts)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": quiet_story.id})

    assert resp.status_code == 200
    assert probe_calls == [[quiet_story.id]]
    assert calls == []
    assert resp.get_json()["tldr"] == "Cached TLDR"
    assert resp.get_json()["cached"] is True


@pytest.mark.parametrize("stored_count", [20, 60])
def test_flask_test_client_tldr_tap_probe_hydrates_confirmed_growth(
    test_env: Any, monkeypatch: pytest.MonkeyPatch, stored_count: int
) -> None:
    """Unfetched live comments trigger hydration even after count healing."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    grown_story = Story(
        id=1732,
        title="Grown thread story",
        url="https://example.com/grown-thread",
        score=50,
        time=int(now - 10 * 3600),  # below the active velocity threshold
        text_content="Grown thread story. Body.",
        source="hn",
        comment_count=stored_count,
        comment_count_at_fetch=20,  # tap must compare live against this count
        self_text="",
        top_comments="Old prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(grown_story)
    cached_key = server._tldr_cache_key(
        title=grown_story.title,
        self_text=grown_story.self_text or "",
        top_comments=grown_story.top_comments or "",
        article_body=grown_story.article_body or "",
    )
    db.upsert_tldr_cache(grown_story.id, cached_key, "Stale cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    probe_timeouts: list[float] = []

    async def mock_probe_live_counts(stories, timeout_s):
        probe_timeouts.append(timeout_s)
        return {grown_story.id: 60}

    monkeypatch.setattr("pipeline._probe_live_counts", mock_probe_live_counts)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        current = db_.get_story(sid)
        assert current is not None
        updated = replace(current, top_comments="Freshly hydrated comments.")
        db_.upsert_story(updated)
        return updated

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"Fresh TLDR: {top_comments}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": grown_story.id})

    assert resp.status_code == 200
    assert probe_timeouts == [handler.config.tldr_tap_probe_timeout_seconds]
    assert calls == [{"sid": grown_story.id, "force": True}]
    body = resp.get_json()
    assert body["tldr"] == "Fresh TLDR: Freshly hydrated comments."
    assert body["cached"] is False
    healed = db.get_story(grown_story.id)
    assert healed is not None
    assert healed.comment_count == 60


def test_flask_test_client_tldr_heals_count_past_lagging_hydrate(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Algolia lags Firebase: when hydration returns fewer comments than
    the probe-confirmed live count, the DB count keeps the live number
    and the response reports both live and summarized counts."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    grown_story = Story(
        id=1734,
        title="Lagging hydrate story",
        url="https://example.com/lagging-hydrate",
        score=50,
        time=int(now - 2 * 3600),  # 2h old
        text_content="Lagging hydrate story. Body.",
        source="hn",
        comment_count=20,  # below the 30-comment active floor
        comment_count_at_fetch=20,  # ... so only the tap probe notices growth
        self_text="",
        top_comments="Old prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(grown_story)
    cached_key = server._tldr_cache_key(
        title=grown_story.title,
        self_text=grown_story.self_text or "",
        top_comments=grown_story.top_comments or "",
        article_body=grown_story.article_body or "",
    )
    db.upsert_tldr_cache(grown_story.id, cached_key, "Stale cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    async def mock_probe_live_counts(stories, timeout_s):
        return {grown_story.id: 60}

    monkeypatch.setattr("pipeline._probe_live_counts", mock_probe_live_counts)

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        current = db_.get_story(sid)
        assert current is not None
        # Algolia view of the world: only 40 of the 60 live comments.
        updated = replace(
            current,
            top_comments="Freshly hydrated comments.",
            comment_count=40,
            comment_count_at_fetch=40,
        )
        db_.upsert_story(updated)
        return updated

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"Fresh TLDR: {top_comments}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": grown_story.id})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["cached"] is False
    assert body["comment_count_live"] == 60
    assert body["comment_count_summarized"] == 40
    healed = db.get_story(grown_story.id)
    assert healed is not None
    assert healed.comment_count == 60
    assert healed.comment_count_at_fetch == 40


def test_flask_test_client_tldr_tap_probe_failure_serves_cached(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A throwing tap probe must never block the tap: the cached TLDR is
    served with no hydration attempt."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    quiet_story = Story(
        id=1733,
        title="Probe failure story",
        url="https://example.com/probe-failure",
        score=10,
        time=int(now - 2 * 3600),
        text_content="Probe failure story. Body.",
        source="hn",
        comment_count=5,
        comment_count_at_fetch=5,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(quiet_story)
    cached_key = server._tldr_cache_key(
        title=quiet_story.title,
        self_text=quiet_story.self_text or "",
        top_comments=quiet_story.top_comments or "",
        article_body=quiet_story.article_body or "",
    )
    db.upsert_tldr_cache(quiet_story.id, cached_key, "Cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    async def mock_probe_live_counts(stories, timeout_s):
        raise ConnectionError("firebase unreachable")

    monkeypatch.setattr("pipeline._probe_live_counts", mock_probe_live_counts)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        raise AssertionError("LLM must not run on probe failure")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": quiet_story.id})

    assert resp.status_code == 200
    assert calls == []
    assert resp.get_json()["tldr"] == "Cached TLDR"
    assert resp.get_json()["cached"] is True


def test_flask_test_client_tldr_tap_probe_skipped_for_old_thread(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threads older than the refresh window are never probed on tap,
    even with a cached TLDR and live-looking counts."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    old_story = Story(
        id=1734,
        title="Old thread story",
        url="https://example.com/old-thread",
        score=500,
        time=int(now - 100 * 3600),  # 100h old, beyond the 72h window
        text_content="Old thread story. Body.",
        source="hn",
        comment_count=500,
        comment_count_at_fetch=500,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(old_story)
    cached_key = server._tldr_cache_key(
        title=old_story.title,
        self_text=old_story.self_text or "",
        top_comments=old_story.top_comments or "",
        article_body=old_story.article_body or "",
    )
    db.upsert_tldr_cache(old_story.id, cached_key, "Cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    async def mock_probe_live_counts(stories, timeout_s):
        raise AssertionError("old threads must not be probed")

    monkeypatch.setattr("pipeline._probe_live_counts", mock_probe_live_counts)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    resp = client.post("/api/tldr-detail", json={"story_id": old_story.id})

    assert resp.status_code == 200
    assert calls == []
    assert resp.get_json()["tldr"] == "Cached TLDR"
    assert resp.get_json()["cached"] is True


def test_flask_test_client_tldr_force_refresh_regenerates(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force_refresh bypasses both cache hits (early + post-enrich) and
    force-hydrates, so even a quiet cached thread gets a fresh summary.
    The probe is skipped — the user already confirmed intent."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    quiet_story = Story(
        id=1735,
        title="Forced refresh story",
        url="https://example.com/forced-refresh",
        score=10,
        time=int(now - 2 * 3600),
        text_content="Forced refresh story. Body.",
        source="hn",
        comment_count=5,
        comment_count_at_fetch=5,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(quiet_story)
    cached_key = server._tldr_cache_key(
        title=quiet_story.title,
        self_text=quiet_story.self_text or "",
        top_comments=quiet_story.top_comments or "",
        article_body=quiet_story.article_body or "",
    )
    db.upsert_tldr_cache(quiet_story.id, cached_key, "Cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    async def mock_probe_live_counts(stories, timeout_s):
        raise AssertionError("force_refresh must skip the tap probe")

    monkeypatch.setattr("pipeline._probe_live_counts", mock_probe_live_counts)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None  # no new comments; post-enrich key is unchanged

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    llm_calls = 0

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        nonlocal llm_calls
        llm_calls += 1
        return server.TldrResult(kind="ok", tldr="Fresh forced TLDR")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post(
        "/api/tldr-detail",
        json={"story_id": quiet_story.id, "force_refresh": True},
    )

    assert resp.status_code == 200
    assert calls == [{"sid": quiet_story.id, "force": True}]
    assert llm_calls == 1  # post-enrich hit skipped despite unchanged key
    body = resp.get_json()
    assert body["tldr"] == "Fresh forced TLDR"
    assert body["cached"] is False


def test_flask_test_client_tldr_force_refresh_serves_cached_on_cooldown(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force_refresh still fail-fasts on provider cooldown: the cached TLDR
    is served instead of deepening the ban."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    quiet_story = Story(
        id=1736,
        title="Forced cooldown story",
        url="https://example.com/forced-cooldown",
        score=10,
        time=int(now - 2 * 3600),
        text_content="Forced cooldown story. Body.",
        source="hn",
        comment_count=5,
        comment_count_at_fetch=5,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(quiet_story)
    cached_key = server._tldr_cache_key(
        title=quiet_story.title,
        self_text=quiet_story.self_text or "",
        top_comments=quiet_story.top_comments or "",
        article_body=quiet_story.article_body or "",
    )
    db.upsert_tldr_cache(quiet_story.id, cached_key, "Cached TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        raise AssertionError("LLM must not run while cooling down")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    server.llm_limiter.on_429()
    try:
        resp = client.post(
            "/api/tldr-detail",
            json={"story_id": quiet_story.id, "force_refresh": True},
        )
    finally:
        server.llm_limiter.reset()

    assert resp.status_code == 200
    assert resp.get_json()["tldr"] == "Cached TLDR"
    assert resp.get_json()["cached"] is True
    assert resp.get_json()["retryable"] is True
    assert resp.get_json()["reason"] == "provider_cooldown"
    assert resp.get_json()["retry_after_seconds"] > 0


def test_tldr_tap_should_probe_gates() -> None:
    """Predicate unit test: young HN threads with comments probe; old,
    non-HN, comment-less, and timeless stories don't."""
    import server
    from pipeline import Config

    config = Config()
    now = time.time()
    base = Story(
        id=1,
        title="t",
        url="https://example.com/t",
        score=1,
        time=int(now - 3600),
        text_content="t",
        source="hn",
        comment_count=10,
        self_text="",
        top_comments="comments",
        article_body="",
    )

    assert server._tldr_tap_should_probe(base, config, now) is True
    assert (
        server._tldr_tap_should_probe(replace(base, source="rss_x"), config, now)
        is False
    )
    assert (
        server._tldr_tap_should_probe(replace(base, top_comments=""), config, now)
        is False
    )
    assert server._tldr_tap_should_probe(replace(base, time=0), config, now) is False
    assert (
        server._tldr_tap_should_probe(
            replace(base, time=int(now - 100 * 3600)), config, now
        )
        is False
    )


def test_flask_test_client_tldr_skips_refresh_for_quiet_recent_thread(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recent but low-velocity HN thread should NOT trigger a forced
    refresh -- only threads that look actively busy are worth the extra
    Algolia call."""
    import server

    _, db, _, handler, user = test_env
    now = time.time()
    quiet_story = Story(
        id=1721,
        title="Quiet recent story",
        url="https://example.com/quiet-recent",
        score=10,
        time=int(now - 2 * 3600),  # 2h old
        text_content="Quiet recent story. Body.",
        source="hn",
        comment_count=5,  # 2.5 comments/hour, below default floor + velocity
        comment_count_at_fetch=5,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(quiet_story)
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": quiet_story.id})

    assert resp.status_code == 200
    assert calls == []


def test_flask_test_client_tldr_skips_refresh_for_old_active_thread(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A high-velocity-looking thread that is older than
    tldr_refresh_recent_hours should NOT trigger a forced refresh."""
    import server

    _, db, _, handler, user = test_env
    handler.config = replace(handler.config, tldr_refresh_recent_hours=72.0)
    now = time.time()
    old_story = Story(
        id=1722,
        title="Old busy story",
        url="https://example.com/old-busy",
        score=200,
        time=int(now - 100 * 3600),  # 100h old, past the 72h window
        text_content="Old busy story. Body.",
        source="hn",
        comment_count=500,
        comment_count_at_fetch=500,
        self_text="",
        top_comments="Existing prewarmed comments.",
        article_body="Body.",
    )
    db.upsert_story(old_story)
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    calls: list[dict[str, Any]] = []

    async def mock_fetch_story(client_, sid, db_, *, force=False):
        calls.append({"sid": sid, "force": force})
        return None

    monkeypatch.setattr("pipeline.fetch_story", mock_fetch_story)

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = client.post("/api/tldr-detail", json={"story_id": old_story.id})

    assert resp.status_code == 200
    assert calls == []


def test_flask_test_client_tldr_stale_fallback_on_quota_denied(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache-key mismatch (e.g. article_body enriched after TLDR was
    cached) must not surface as a 429/error when quota is exhausted -- the
    stale-but-cached TLDR should be served instead."""
    import server

    _, db, _, handler, user = test_env
    handler.config = replace(
        handler.config,
        tldr_uncached_per_user_limit=1,
        tldr_uncached_per_user_window_seconds=3600,
        tldr_uncached_global_limit=100,
    )
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    for story_id in (1720, 1721):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Flask stale-fallback story {story_id}",
                url=f"https://example.com/flask-stale-fallback-{story_id}",
                score=10,
                time=1600000000,
                text_content="Story body.",
                source="hn",
                comment_count=0,
                self_text="",
                top_comments="",
                article_body="Body enriched after TLDR was cached.",
            )
        )
    stale_key = server._tldr_cache_key(
        title="Flask stale-fallback story 1721",
        self_text="",
        top_comments="",
        article_body="Body before enrichment.",
    )
    db.upsert_tldr_cache(1721, stale_key, "Stale summary from before enrichment")

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    first = client.post("/api/tldr-detail", json={"story_id": 1720})
    second = client.post("/api/tldr-detail", json={"story_id": 1721})

    assert first.status_code == 200
    assert first.get_json()["cached"] is False
    assert second.status_code == 200
    assert second.get_json() == {
        "ok": True,
        "tldr": "Stale summary from before enrichment",
        "cached": True,
        "stale": True,
    }


@pytest.mark.parametrize("error_status", [429, 402])
def test_flask_test_client_tldr_provider_error_degrades_gracefully(
    test_env: Any, monkeypatch: pytest.MonkeyPatch, error_status: int
) -> None:
    """Pin the tap path when the LLM provider refuses (rate limit or billing
    cap, e.g. a capped Mistral key returning 429 or 402): a stale-cached TLDR
    must be served either way; with nothing cached, both degrade to the
    cooldown countdown (402 seeds the limiter explicitly since a billing
    refusal carries no Retry-After semantics)."""
    import server

    _, db, _, handler, user = test_env
    handler.config = replace(
        handler.config,
        tldr_uncached_per_user_limit=100,
        tldr_uncached_per_user_window_seconds=3600,
        tldr_uncached_global_limit=100,
    )
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    for story_id in (1722, 1723):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Provider-error story {story_id}",
                url=f"https://example.com/provider-error-{story_id}",
                score=10,
                time=1600000000,
                text_content="Story body.",
                source="hn",
                comment_count=0,
                self_text="",
                top_comments="",
                article_body="",
            )
        )
    stale_key = server._tldr_cache_key(
        title="Provider-error story 1723",
        self_text="",
        top_comments="",
        article_body="Body before enrichment.",
    )
    db.upsert_tldr_cache(1723, stale_key, "Stale summary survives provider outage")
    # Current key differs from the stale one so the request misses cache.
    db.upsert_story(
        replace(
            db.get_story(1723),
            article_body="Body enriched after TLDR was cached.",
        )
    )

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        return server.TldrResult(
            kind="llm_error",
            error_status=error_status,
            error_text=f"mocked provider HTTP {error_status}",
        )

    async def mock_fetch_article_body_with_result(
        url: str,
    ) -> "server.ArticleFetchResult":
        # No outbound request: this test is about the provider refusal.
        return server.ArticleFetchResult(status=404, error="http_404")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)
    monkeypatch.setattr(
        server, "_fetch_article_body_with_result", mock_fetch_article_body_with_result
    )

    fresh = client.post("/api/tldr-detail", json={"story_id": 1722})
    stale = client.post("/api/tldr-detail", json={"story_id": 1723})

    assert stale.status_code == 200
    assert stale.get_json() == {
        "ok": True,
        "tldr": "Stale summary survives provider outage",
        "cached": True,
        "stale": True,
    }
    assert fresh.status_code == 429
    assert "cooling down" in fresh.get_json()["error"]
    if error_status == 402:
        # Billing refusal seeds the limiter: prefetch skips while capped.
        assert server.llm_limiter.retry_after_seconds > 0


@pytest.mark.parametrize("cacheable", [True, False])
async def test_prefetch_tldrs_for_ranked_regenerates_stale_beyond_top_combo(
    test_env: Any, monkeypatch: pytest.MonkeyPatch, cacheable: bool
) -> None:
    """Side B: the top-per-combo prefetch alone never reaches a story ranked
    below the cutoff, so a stale-key story there is stuck until it re-enters
    the top N. `stale_per_run` scans the remainder of the cold deck for
    cache-key mismatches and regenerates a bounded number of them."""
    import server as srv
    from pipeline import RankedStory

    _, db, _, _, _ = test_env

    stories = [
        Story(
            id=3100 + i,
            title=f"Stale scan story {i}",
            url=f"https://example.com/stale-scan-{i}",
            score=10,
            time=1600000000,
            text_content="body",
            source="hn",
            comment_count=0,
            self_text="",
            top_comments="",
            article_body="New body.",
        )
        for i in range(3)
    ]
    for s in stories:
        db.upsert_story(s)

    ranked = [
        RankedStory(story=s, score=1.0, best_match_title="", combo_keys="recent_hn")
        for s in stories
    ]

    # stories[1]: beyond the per_combo=1 cutoff, cached under a stale key
    # (article_body changed since it was cached) -> must be regenerated.
    stale_key = srv._tldr_cache_key(
        title=stories[1].title,
        self_text="",
        top_comments="",
        article_body="Old body before enrichment.",
    )
    db.upsert_tldr_cache(stories[1].id, stale_key, "Stale TLDR")

    # stories[2]: beyond the cutoff, cached under the *current* key ->
    # must NOT be regenerated even with the stale scan enabled.
    fresh_key = srv._tldr_cache_key(
        title=stories[2].title,
        self_text="",
        top_comments="",
        article_body="New body.",
    )
    db.upsert_tldr_cache(stories[2].id, fresh_key, "Fresh TLDR")

    calls: list[str] = []

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        calls.append(title)
        return srv.TldrResult(kind="ok", tldr=f"TLDR: {title}", cacheable=cacheable)

    monkeypatch.setattr(srv, "generate_detailed_tldr", mock_generate_detailed_tldr)
    monkeypatch.setattr(srv, "_PREFETCH_STAGGER_S", 0)

    generated = await srv._prefetch_tldrs_for_ranked(
        ranked, db, per_combo=1, stale_per_run=2, date_top_n=0
    )

    assert generated == (2 if cacheable else 0)
    assert sorted(calls) == sorted([stories[0].title, stories[1].title])
    if not cacheable:
        assert db.get_any_tldr_for_story(stories[0].id) is None
        assert db.get_any_tldr_for_story(stories[1].id) == "Stale TLDR"
    assert db.get_any_tldr_for_story(stories[2].id) == "Fresh TLDR"


async def test_prefetch_tldrs_for_ranked_covers_date_sorted_head(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Date-tab lane: the newest story by time is prefetched even when it
    sits below the per-combo cutoff and has no stale key."""
    import server as srv
    from pipeline import RankedStory

    _, db, _, _, _ = test_env

    old = Story(
        id=3120,
        title="Old combo-top story",
        url="https://example.com/date-old",
        score=10,
        time=1600000000,
        text_content="body",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Body.",
    )
    new = Story(
        id=3121,
        title="Newest story below cutoff",
        url="https://example.com/date-new",
        score=10,
        time=1700000000,
        text_content="body",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Body.",
    )
    for s in (old, new):
        db.upsert_story(s)
    ranked = [
        RankedStory(story=old, score=1.0, best_match_title="", combo_keys="recent_hn"),
        RankedStory(story=new, score=0.1, best_match_title="", combo_keys=""),
    ]

    calls: list[str] = []

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        calls.append(title)
        return srv.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(srv, "generate_detailed_tldr", mock_generate_detailed_tldr)
    monkeypatch.setattr(srv, "_PREFETCH_STAGGER_S", 0)

    generated = await srv._prefetch_tldrs_for_ranked(
        ranked, db, per_combo=1, stale_per_run=0, date_top_n=1
    )

    assert generated == 2
    assert sorted(calls) == sorted([old.title, new.title])


async def test_prefetch_tldrs_for_ranked_date_lane_dedupes_combo_picks(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newest story already picked by the combo pass is fetched once."""
    import server as srv
    from pipeline import RankedStory

    _, db, _, _, _ = test_env

    s = Story(
        id=3130,
        title="Newest combo-top story",
        url="https://example.com/date-dedupe",
        score=10,
        time=1700000000,
        text_content="body",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Body.",
    )
    db.upsert_story(s)
    ranked = [
        RankedStory(story=s, score=1.0, best_match_title="", combo_keys="recent_hn")
    ]

    calls: list[str] = []

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        calls.append(title)
        return srv.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(srv, "generate_detailed_tldr", mock_generate_detailed_tldr)
    monkeypatch.setattr(srv, "_PREFETCH_STAGGER_S", 0)

    generated = await srv._prefetch_tldrs_for_ranked(
        ranked, db, per_combo=1, stale_per_run=0, date_top_n=8
    )

    assert generated == 1
    assert calls == [s.title]


async def test_prefetch_tldrs_for_ranked_skips_run_during_provider_cooldown(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Background prefetch must not fire (or deepen) a provider ban; taps get
    first shot at recovered quota."""
    import server as srv
    from pipeline import RankedStory

    _, db, _, _, _ = test_env

    s = Story(
        id=3140,
        title="Cooldown skip story",
        url="https://example.com/cooldown-skip",
        score=10,
        time=1700000000,
        text_content="body",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Body.",
    )
    db.upsert_story(s)
    ranked = [
        RankedStory(story=s, score=1.0, best_match_title="", combo_keys="recent_hn")
    ]

    calls: list[str] = []

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        calls.append(title)
        return srv.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(srv, "generate_detailed_tldr", mock_generate_detailed_tldr)
    monkeypatch.setattr(srv, "_PREFETCH_STAGGER_S", 0)
    srv.llm_limiter.on_429()
    try:
        generated = await srv._prefetch_tldrs_for_ranked(
            ranked, db, per_combo=1, stale_per_run=0, date_top_n=1
        )
    finally:
        srv.llm_limiter.reset()

    assert generated == 0
    assert calls == []


async def test_prefetch_tldrs_for_ranked_logs_zero_outcome_with_candidates(
    test_env: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A silent generated=0 once hid a fail-fast storm (page-open burst
    tripped the ban, every prefetch fail-fasted, log showed nothing); the
    zero outcome must log candidates plus cooldown state."""
    import logging

    import server as srv
    from pipeline import RankedStory

    _, db, _, _, _ = test_env

    s = Story(
        id=3141,
        title="Already cached story",
        url="https://example.com/already-cached",
        score=10,
        time=1700000000,
        text_content="body",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Body.",
    )
    db.upsert_story(s)
    db.upsert_tldr_cache(
        s.id,
        srv._tldr_cache_key(
            title=s.title, self_text="", top_comments="", article_body="Body."
        ),
        "TLDR: cached",
    )
    ranked = [
        RankedStory(story=s, score=1.0, best_match_title="", combo_keys="recent_hn")
    ]

    monkeypatch.setattr(srv, "_PREFETCH_STAGGER_S", 0)
    with caplog.at_level(logging.INFO):
        generated = await srv._prefetch_tldrs_for_ranked(
            ranked, db, per_combo=1, stale_per_run=0, date_top_n=0
        )

    assert generated == 0
    assert "tldr_prefetch generated=0 candidates=1" in caplog.text


def test_shape_tldr_caps_blocks_per_section() -> None:
    """One section keeps at most four bullets and one subheading; the reader
    pane fits one screen only when the block count is bounded."""
    import server

    raw = (
        "### Article\n#### First\n#### Second\n"
        "- a\n- b\n- c\n- d\n- e\n\n"
        "### Discussion\n- 1\n- 2\n- 3\n- 4\n- 5\n"
    )
    shaped = server._shape_tldr(raw)
    assert shaped.count("- ") == 8
    assert shaped.count("####") == 1
    assert "- e" not in shaped
    assert "#### Second" not in shaped


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


def test_normalize_tldr_markdown_converts_star_bullets() -> None:
    """gemini-2.5-flash-lite emits `*` bullets; normalize to `-`."""
    import server

    normalized = server._normalize_tldr_markdown(
        "*   **Nitter** resumes service.\n* plain point"
    )
    assert normalized == "- **Nitter** resumes service.\n- plain point"


def test_normalize_tldr_markdown_bolds_single_marker_emphasis() -> None:
    """Single-marker emphasis is italic in CommonMark but bold in the
    dashboard; both clients must agree, and intra-word underscores and
    existing bold must be left alone."""
    import server

    normalized = server._normalize_tldr_markdown(
        "- pronouns (e.g., *him/her*, _his/hers_) stay **bold** and snake_case intact"
    )
    assert normalized == (
        "- pronouns (e.g., **him/her**, **his/hers**) stay **bold** and snake_case intact"
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

    resp = local_http.post(
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

    async def mock_fetch_article_body_with_result(url):
        raise AssertionError("Reddit comments pages should not be scraped as articles")

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return server.TldrResult(kind="ok", tldr=f"TLDR: {self_text} | {top_comments}")

    import server

    monkeypatch.setattr(
        server, "_fetch_reddit_rss_context", mock_fetch_reddit_rss_context
    )
    monkeypatch.setattr(
        server, "_fetch_article_body_with_result", mock_fetch_article_body_with_result
    )
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = local_http.post(
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
    async def mock_fetch_story(client, sid, database, *, force=False):
        story = database.get_story(sid)
        from dataclasses import replace

        updated = replace(
            story,
            top_comments="Fetched comments",
            text_content="Dynamic test. Fetched comments",
        )
        database.upsert_story(updated)
        return updated

    async def mock_fetch_article_body_with_result(url):
        return server.ArticleFetchResult(body="Fetched article body text", status=200)

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return server.TldrResult(
            kind="ok", tldr=f"TLDR: {title} | {top_comments} | {article_body}"
        )

    import server
    import pipeline

    monkeypatch.setattr(pipeline, "fetch_story", mock_fetch_story)
    monkeypatch.setattr(
        server, "_fetch_article_body_with_result", mock_fetch_article_body_with_result
    )
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    # Request TLDR
    resp = local_http.post(
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


@pytest.mark.parametrize(("source", "story_id"), [("bq_seed", 779), ("ch_seed", 771)])
def test_tldr_detail_hydrates_archive_seed_comments_on_demand(
    test_env, monkeypatch, source: str, story_id: int
):
    """Archive rows have no prewarmed comments; a TLDR tap fetches them."""
    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=story_id,
            title="Archive dynamic test",
            url=f"https://example.com/{source}-dynamic-test",
            score=100,
            time=1600000000,
            text_content="Archive dynamic test.",
            source=source,
            comment_count=5,
            discussion_url=f"https://news.ycombinator.com/item?id={story_id}",
        )
    )

    async def mock_fetch_story(client, sid, database, *, force=False):
        from dataclasses import replace

        updated = replace(database.get_story(sid), top_comments="Fetched comments")
        database.upsert_story(updated)
        return updated

    async def mock_fetch_article_body_with_result(url):
        return server.ArticleFetchResult(error="empty_extraction")

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title} | {top_comments}")

    import server
    import pipeline

    monkeypatch.setattr(pipeline, "fetch_story", mock_fetch_story)
    monkeypatch.setattr(
        server, "_fetch_article_body_with_result", mock_fetch_article_body_with_result
    )
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": story_id},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert "Fetched comments" in resp.json()["tldr"]
    assert db.get_story(story_id).top_comments == "Fetched comments"


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
        return server.TldrResult(
            kind="ok", tldr=f"cached-result-{calls}: {title} | {article_body}"
        )

    import server

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp1 = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 778},
        cookies={"hn_token": user.token},
    )
    resp2 = local_http.post(
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


def test_tldr_cached_response_bypasses_uncached_quota(test_env, monkeypatch) -> None:
    import server

    port, db, _, handler, user = test_env
    handler.config = Config(
        db_path=db.db_path,
        server_port=port,
        tldr_uncached_per_user_limit=1,
        tldr_uncached_per_user_window_seconds=3600,
        tldr_uncached_global_limit=100,
    )
    uncached_story = Story(
        id=780,
        title="Uncached quota consumer",
        url="https://example.com/uncached-quota",
        score=10,
        time=1600000000,
        text_content="Uncached quota consumer. Body.",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Body.",
    )
    cached_story = Story(
        id=781,
        title="Cached quota bypass",
        url="https://example.com/cached-quota",
        score=10,
        time=1600000000,
        text_content="Cached quota bypass. Cached body.",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Cached body.",
    )
    db.upsert_story(uncached_story)
    db.upsert_story(cached_story)
    cached_key = server._tldr_cache_key(
        title=cached_story.title,
        self_text="",
        top_comments="",
        article_body="Cached body.",
    )
    db.upsert_tldr_cache(cached_story.id, cached_key, "Already cached")

    calls = 0

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        nonlocal calls
        calls += 1
        return server.TldrResult(kind="ok", tldr=f"generated-{calls}: {title}")

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    first = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": uncached_story.id},
        cookies={"hn_token": user.token},
    )
    second = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": cached_story.id},
        cookies={"hn_token": user.token},
    )

    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert second.status_code == 200
    assert second.json() == {"ok": True, "tldr": "Already cached", "cached": True}
    assert calls == 1


def test_tldr_uncached_per_session_limit_blocks_generation(
    test_env, monkeypatch
) -> None:
    port, db, _, handler, user = test_env
    handler.config = Config(
        db_path=db.db_path,
        server_port=port,
        tldr_uncached_per_user_limit=1,
        tldr_uncached_per_user_window_seconds=3600,
        tldr_uncached_global_limit=100,
    )
    for story_id in (782, 783):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Per-user TLDR story {story_id}",
                url=f"https://example.com/per-user-tldr-{story_id}",
                score=10,
                time=1600000000,
                text_content="Story body.",
                source="hn",
                comment_count=0,
                self_text="",
                top_comments="",
                article_body="Story body.",
            )
        )

    calls: list[str] = []

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        calls.append(title)
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    import server

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    first = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 782},
        cookies={"hn_token": user.token},
    )
    second = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 783},
        cookies={"hn_token": user.token},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) > 0
    assert second.json()["retry_after"] == int(second.headers["Retry-After"])
    assert calls == ["Per-user TLDR story 782"]


def test_tldr_uncached_global_limit_blocks_second_session(
    test_env, monkeypatch
) -> None:
    port, db, _, handler, user = test_env
    other_user = db.create_user("other_tldr_user")
    handler.config = Config(
        db_path=db.db_path,
        server_port=port,
        tldr_uncached_per_user_limit=100,
        tldr_uncached_global_limit=1,
        tldr_uncached_global_window_seconds=3600,
    )
    for story_id in (784, 785):
        db.upsert_story(
            Story(
                id=story_id,
                title=f"Global TLDR story {story_id}",
                url=f"https://example.com/global-tldr-{story_id}",
                score=10,
                time=1600000000,
                text_content="Story body.",
                source="hn",
                comment_count=0,
                self_text="",
                top_comments="",
                article_body="Story body.",
            )
        )

    calls: list[str] = []

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        calls.append(title)
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    import server

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    first = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 784},
        cookies={"hn_token": user.token},
    )
    second = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 785},
        cookies={"hn_token": other_user.token},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert calls == ["Global TLDR story 784"]


def test_tldr_detail_uncached_requires_session(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sessionless clients get cached TLDRs only; never spend the LLM budget."""
    port, db, _, _, _ = test_env
    db.upsert_story(
        Story(
            id=786,
            title="Sessionless TLDR story",
            url="https://example.com/sessionless-tldr",
            score=10,
            time=1600000000,
            text_content="Story body.",
            source="hn",
            comment_count=0,
            self_text="",
            top_comments="",
            article_body="Story body.",
        )
    )
    calls: list[str] = []

    async def mock_generate_detailed_tldr(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        calls.append(title)
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    import server

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    for body in ({"story_id": 786}, {"story_id": 786, "force_refresh": True}):
        response = httpx.post(f"http://127.0.0.1:{port}/api/tldr-detail", json=body)
        assert response.status_code == 401
    assert calls == []

    db.upsert_tldr_cache(786, "old-key", "Old TLDR")
    stale = httpx.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 786, "force_refresh": True},
    )
    assert stale.status_code == 200
    assert stale.json()["tldr"] == "Old TLDR"
    assert calls == []


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
        return server.TldrResult(kind="no_content")

    import server

    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp1 = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 779},
        cookies={"hn_token": user.token},
    )
    resp2 = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 779},
        cookies={"hn_token": user.token},
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert call_count == 2  # both requests regenerated (no cache write)
    assert resp1.json()["cached"] is False
    assert resp2.json()["cached"] is False
    assert resp1.json()["empty"] is True  # clients skip, never render
    assert resp1.json()["retryable"] is True
    assert db.get_tldr_cache(779, "") is None  # no cache entry written


def test_tldr_partial_response_remains_retryable(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import server

    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=780,
            title="Partial story",
            url=None,
            score=5,
            time=1600000000,
            text_content="Body",
            article_body="Body",
        )
    )

    async def generate(*args: object, **kwargs: object) -> server.TldrResult:
        return server.TldrResult(
            kind="ok", tldr="### Article\n- Partial", cacheable=False
        )

    monkeypatch.setattr(server, "generate_detailed_tldr", generate)
    response = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 780},
        cookies={"hn_token": user.token},
    )
    assert response.status_code == 200
    assert response.json()["retryable"] is True
    assert db.get_any_tldr_for_story(780) is None


def test_tldr_markdown_neutralizes_html_and_unsafe_links() -> None:
    """LLM TLDR text lands in innerHTML: raw HTML must render as text and
    only http(s) links survive, while normal markdown still formats."""
    import json as _json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the TLDR markdown renderer")
    _, script = _read_template_and_static()
    start = script.index("    const TAGS=")
    end = script.index("    function styleTldrLabels(")
    cases = {
        "img": "- Point: <img src=x onerror=alert(1)>",
        "script": "Intro <script>alert(1)</script> & more",
        "js_link": "- [click](javascript:alert(1))",
        "spaced_js_link": "- [click]( javascript:alert(1))",
        "entity_js_link": "[x](&#106;avascript:alert(1))",
        "data_link": "[x](data:text/html,<b>hi</b>)",
        "md_image": "![a](https://evil.example/pixel.png)",
        "ok_link": "- **Key:** see [docs](https://example.com/a?b=1&c=2)",
        "code": "use `a < b && c` here",
    }
    harness = (
        script[start:end]
        + f"\nconst cases = {_json.dumps(cases)};\n"
        + "const out = {}; for (const [k, v] of Object.entries(cases)) "
        + "out[k] = parseSimpleMarkdown(v);\nconsole.log(JSON.stringify(out));\n"
    )
    completed = subprocess.run(
        [node, "-e", harness], check=True, capture_output=True, text=True, timeout=10
    )
    out: dict[str, str] = _json.loads(completed.stdout)

    for html in out.values():
        assert "<img" not in html
        assert "<script" not in html
        assert 'href="javascript' not in html.replace(" ", "")
        assert 'href="&amp;#106;' not in html
        assert 'href="data:' not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in out["img"]
    assert "&lt;script&gt;" in out["script"]
    assert "&amp; more" in out["script"]
    assert "<a>click</a>" in out["js_link"]
    assert "<strong>Key:</strong>" in out["ok_link"]
    assert (
        '<a href="https://example.com/a?b=1&amp;c=2" target="_blank" '
        'rel="noopener noreferrer nofollow">docs</a>'
    ) in out["ok_link"]
    assert "<code>a &lt; b &amp;&amp; c</code>" in out["code"]


def test_forced_tldr_refresh_serializes_requests() -> None:
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute browser refresh logic")
    _, script = _read_template_and_static()
    start = script.index("    async function openTldrDetail(")
    end = script.index("    const KEY_ACTIONS", start)
    harness = (
        r"""
const assert = require('node:assert/strict');
const content = {dataset: {}, style: {display: 'none'}};
const card = {dataset: {storyId: '1'}, querySelector: selector =>
  selector === '.tldr-detail-content' ? content : null};
const tldrCache = new Map([[1, 'old summary']]);
const tldrRetryAt = new Map([[1, Date.now() + 600000]]);
const apiPath = path => path;
let calls = 0, finish;
const fetch = () => {
  calls++;
  return new Promise(resolve => { finish = resolve; });
};
const response = {ok: true, json: async () => ({tldr: 'fresh summary'})};
const parseSimpleMarkdown = text => text;
const styleTldrLabels = () => {};
const enhanceTldrContent = () => {};
const ensureTldrRefreshButton = () => {};
"""
        + script[start:end]
        + r"""
(async () => {
  const first = openTldrDetail(card, {force: true});
  await openTldrDetail(card, {force: true});
  await openTldrDetail(card);
  assert.equal(calls, 1);
  finish(response);
  await first;
  assert.equal(content.dataset.loading, undefined);
  assert.equal(tldrCache.get(1), 'fresh summary');
  const next = openTldrDetail(card, {force: true});
  assert.equal(calls, 2);
  finish(response);
  await next;
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    )
    subprocess.run(
        [node, "-e", harness], check=True, capture_output=True, text=True, timeout=10
    )


def test_prefetch_follows_navigation_order() -> None:
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute browser queue logic")
    _, script = _read_template_and_static()
    start = script.index("    function prefetchUpcomingTldrs()")
    end = script.index("    function cardsForAge", start)
    # Execute the production function against queue states, including a
    # retained active card in the middle after a refill and wraparound.
    harness = (
        """
      const assert = require('node:assert/strict');
      const PREFETCH_COUNT = 4;
      let queue, activeCard, requested;
      const queuedCards = () => queue;
      const prefetchCards = cards => { requested = cards; };
    """
        + script[start:end]
        + """
      for (const [cards, active, expected] of [
        [[1,2,3,4,5,6], 1, [2,3,4,5]],
        [[1,2,3,4,5,6], 3, [4,5,6,1]],
        [[1,2,3,4,5,6], 6, [1,2,3,4]],
        [[1], 1, []], [[], null, []], [[1,2], null, [1,2]],
      ]) {
        queue = cards; activeCard = active;
        prefetchUpcomingTldrs();
        assert.deepEqual(requested, expected);
      }
    """
    )
    subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True)
    refill = script[
        script.index("    async function refillQueue(") : script.index(
            "    document.querySelectorAll('[data-fb]').forEach",
            script.index("    async function refillQueue("),
        )
    ]
    assert "else {\n        prefetchUpcomingTldrs();" in refill


def test_prefetch_cards_runs_sequentially_and_skips_detached() -> None:
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute browser queue logic")
    _, script = _read_template_and_static()
    start = script.index("    let tldrPrefetchChain")
    end = script.index("    function prefetchUpcomingTldrs()", start)
    # The page-open burst (active tap + 4 prefetches + server warm prefetch)
    # tripped the free token-rate limit within seconds, so prefetches must
    # run one-at-a-time behind the active tap, skipping detached cards.
    harness = (
        """
      const assert = require('node:assert/strict');
      const started = [];
      const resolvers = [];
      const openTldrDetail = (card) => {
        started.push(card.id);
        return new Promise((resolve) => resolvers.push(resolve));
      };
      const mk = (id, connected) => ({
        id, isConnected: connected, querySelector: () => null,
      });
      const tick = async (n) => {
        for (let i = 0; i < n; i++) {
          await new Promise((r) => setImmediate(r));
        }
      };
    """
        + script[start:end]
        + """
      (async () => {
        const a = mk('a', true), b = mk('b', true), gone = mk('gone', false);
        prefetchCards([a, b, gone]);
        await tick(5);
        assert.deepEqual(started, ['a']);
        resolvers[0]('ok-a');
        await tick(5);
        assert.deepEqual(started, ['a', 'b']);
        assert.equal(resolvers.length, 2);
        resolvers[1]('ok-b');
        await tick(5);
        assert.deepEqual(started, ['a', 'b']);
      })().then(() => process.exit(0), (e) => { console.error(e); process.exit(1); });
    """
    )
    subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True)


def test_maybe_cache_tldr_skips_salvaged_half(tmp_path: Path) -> None:
    """_maybe_cache_tldr persists complete TLDRs but never a salvaged half
    (single-row table: caching it would evict a previously complete TLDR)."""
    from database import Database

    import server

    db = Database(str(tmp_path / "cache_gate.db"))
    try:
        db.upsert_story(
            Story(
                id=781,
                title="Cache gate story",
                url=None,
                score=5,
                time=1600000000,
                text_content="x",
            )
        )
        assert (
            server._maybe_cache_tldr(
                db, 781, "key1", server.TldrResult(kind="ok", tldr="full")
            )
            is True
        )
        assert db.get_tldr_cache(781, "key1") == "full"

        assert (
            server._maybe_cache_tldr(
                db,
                781,
                "key2",
                server.TldrResult(
                    kind="ok", tldr="### Article\n- half", cacheable=False
                ),
            )
            is False
        )
        assert db.get_tldr_cache(781, "key2") is None
        assert db.get_any_tldr_for_story(781) == "full"  # complete row intact
    finally:
        db.close()


@pytest.mark.asyncio
async def test_generate_marks_single_half_salvage_uncacheable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed side → ok + cacheable=False; both failed → llm_error."""
    import server

    async def mock_call_llm_chat(
        *, api_key, base_url, model, prompt, max_tokens, extra=None
    ):
        if "Summarize the discussion" in prompt:
            return server.LlmChatResult(content="boom", ok=False, status=429)
        return server.LlmChatResult(content="- **Article** summary", ok=True)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Salvage test",
        self_text="Author text",
        top_comments="Comment text",
        article_body="Substantial article body. " * 30,
    )

    assert result.kind == "ok"
    assert result.cacheable is False
    assert result.tldr.startswith("### Article")


@pytest.mark.asyncio
async def test_generate_detailed_tldr_splits_article_and_comments(monkeypatch):
    import server

    calls = []

    async def mock_call_llm_chat(
        *, api_key, base_url, model, prompt, max_tokens, extra=None
    ):
        calls.append(prompt)
        if "Summarize the article" in prompt:
            return server.LlmChatResult(content="- **Article** summary", ok=True)
        if "Summarize the discussion" in prompt:
            return server.LlmChatResult(content="- **Discussion** summary", ok=True)
        return server.LlmChatResult(content="- **Fallback** summary", ok=True)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Split summary test",
        self_text="Author text",
        top_comments="Comment text",
        article_body="Substantial article body. " * 30,
    )

    assert len(calls) == 2
    assert "Article body" in calls[0]
    assert "Comments:" in calls[1]
    assert "Points:" not in calls[1]
    assert "Age hours:" not in calls[1]
    assert "### Article" in result.tldr
    assert "- **Article** summary" in result.tldr
    assert "### Discussion" in result.tldr
    assert "- **Discussion** summary" in result.tldr


async def test_generate_detailed_tldr_folds_thin_article_to_discussion_only(
    monkeypatch,
) -> None:
    """A thin article side (< ARTICLE_SECTION_MIN_CHARS) with rich comments
    must take the single discussion path: one LLM call, no stub Article
    half, full 1000-token budget for the comments."""
    import server

    calls = []

    async def mock_call_llm_chat(
        *, api_key, base_url, model, prompt, max_tokens, extra=None
    ):
        calls.append((prompt, max_tokens))
        return server.LlmChatResult(content="- **Discussion** summary", ok=True)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Thin article test",
        self_text="",
        top_comments="Rich comment text. " * 100,
        article_body="Tiny stub.",
    )

    assert len(calls) == 1
    prompt, max_tokens = calls[0]
    assert "Summarize the discussion" in prompt
    assert "Tiny stub" not in prompt
    assert max_tokens >= 1000
    assert "### Article" not in result.tldr
    assert "- **Discussion** summary" in result.tldr


@pytest.mark.parametrize(
    ("provider", "key_env", "model", "extra", "base_host"),
    [
        ("mistral", "MISTRAL_API_KEY", "mistral-small-latest", {}, "api.mistral.ai"),
        (
            "cerebras",
            "CEREBRAS_API_KEY",
            "gpt-oss-120b",
            {"reasoning_effort": "low"},
            "api.cerebras.ai",
        ),
        (
            "groq",
            "GROQ_API_KEY",
            "openai/gpt-oss-20b",
            {"reasoning_effort": "low"},
            "api.groq.com",
        ),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "meta-llama/llama-3.3-70b-instruct",
            {},
            "openrouter.ai",
        ),
        (
            "zen",
            "OPENCODE_ZEN_API_KEY",
            "ling-3.0-flash-fin-free",
            {},
            "opencode.ai",
        ),
        (
            "gemini",
            "GEMINI_API_KEY",
            "models/gemini-2.5-flash-lite",
            {"reasoning_effort": "none"},
            "generativelanguage.googleapis.com",
        ),
    ],
)
def test_llm_provider_config_table(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_env: str,
    model: str,
    extra: dict,
    base_host: str,
) -> None:
    """Every provider row maps env key + endpoint + default model + extras;
    LLM_MODEL overrides the default model for any provider."""
    import server

    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(key_env, "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    cfg = server._llm_provider_config()

    assert cfg.provider == provider
    assert cfg.api_key == "test-key"
    assert cfg.model == model
    assert cfg.extra == extra
    assert base_host in cfg.base_url

    monkeypatch.setenv("LLM_MODEL", "custom-model")
    assert server._llm_provider_config().model == "custom-model"
    if provider == "groq":
        assert server._llm_provider_config().extra == {}


def test_llm_provider_config_unknown_is_rejected(monkeypatch) -> None:
    import server

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    assert server._llm_provider_config().provider == "groq"

    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        server._llm_provider_config()


def test_provider_max_tokens_reserves_reasoning_headroom() -> None:
    import server

    cerebras = server.LlmProviderConfig(
        "cerebras", "key", "https://example.test", "model", {"reasoning_effort": "low"}
    )
    mistral = server.LlmProviderConfig(
        "mistral", "key", "https://example.test", "model", {}
    )
    zen = server.LlmProviderConfig("zen", "key", "https://example.test", "model", {})
    assert server._max_tokens_for_provider(cerebras, 900) == 1500
    assert server._max_tokens_for_provider(mistral, 900) == 900
    assert server._max_tokens_for_provider(zen, 900) == 8192
    groq = server.LlmProviderConfig(
        "groq",
        "key",
        "https://example.test",
        "openai/gpt-oss-20b",
        {"reasoning_effort": "low"},
    )
    assert server._max_tokens_for_provider(groq, 900) == 1500
    gospark = server.LlmProviderConfig(
        "gospark",
        "key",
        "https://opencode.ai/zen/go/v1/responses",
        "muse-spark-1.3-contributor",
        {"reasoning_effort": "low"},
    )
    # max_output_tokens covers reasoning + output, so gospark gets a larger
    # headroom than the chat-completions reasoning providers (reasoning runs
    # 350-1000 tokens at effort=low, with outliers past 1,600).
    assert server._max_tokens_for_provider(gospark, 450) == 2450


@pytest.mark.parametrize(
    ("content", "finish_reason", "valid"),
    [
        ("- one bullet", "stop", True),
        ("- one bullet", None, True),
        ("#### Heading only", "stop", False),
        ("- partial", "length", False),
        ("", "stop", False),
    ],
)
def test_llm_completion_validation(
    content: str, finish_reason: str | None, valid: bool
) -> None:
    import server

    assert server._valid_llm_completion(content, finish_reason) is valid


@pytest.mark.asyncio
async def test_generate_detailed_tldr_cerebras_passes_reasoning_effort_and_bumped_tokens(
    monkeypatch,
) -> None:
    """Cerebras's gpt-oss-120b is a reasoning model: without reasoning_effort
    and extra max_tokens headroom it burns the whole budget on hidden
    reasoning and returns no content (see WORKLOG 2026-07-10)."""
    import server

    calls = []

    async def mock_call_llm_chat(
        *, api_key, base_url, model, prompt, max_tokens, extra=None
    ):
        calls.append({"max_tokens": max_tokens, "extra": extra, "model": model})
        return server.LlmChatResult(content="- summary", ok=True)

    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    await server.generate_detailed_tldr(
        "Cerebras test",
        self_text="Author text",
        top_comments="Comment text",
        article_body="Substantial article body. " * 30,
    )

    assert len(calls) == 2
    for call in calls:
        assert call["model"] == "gpt-oss-120b"
        assert call["extra"] == {"reasoning_effort": "low"}
        assert call["max_tokens"] == 1050


def test_responses_text_extraction() -> None:
    """Responses payloads mix reasoning items with message items; only
    output_text parts must surface, concatenated in order."""
    import server

    assert server._responses_text({}) == ""
    assert server._responses_text({"output": "x"}) == ""
    payload = {
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "- first"},
                    {"type": "refusal", "refusal": "no"},
                    {"type": "output_text", "text": "\n- second"},
                ],
            },
        ]
    }
    assert server._responses_text(payload) == "- first\n- second"


def test_responses_error_prefers_provider_message() -> None:
    import server

    assert (
        server._responses_error({"error": {"message": "Upstream broke"}}, 500, "raw")
        == "Upstream broke"
    )
    assert "HTTP 400" in server._responses_error({}, 400, "raw")


@pytest.mark.asyncio
async def test_call_llm_for_config_dispatches_on_endpoint(monkeypatch) -> None:
    """chat/completions URLs go to _call_llm_chat; /responses URLs go to
    _call_llm_responses. The gospark row must take the responses path."""
    import server

    calls: list[str] = []

    async def fake_chat(**kwargs):
        calls.append("chat")
        return server.LlmChatResult(content="- c", ok=True)

    async def fake_responses(**kwargs):
        calls.append("responses")
        assert kwargs["extra"] == {"reasoning_effort": "low"}
        return server.LlmChatResult(content="- r", ok=True)

    monkeypatch.setattr(server, "_call_llm_chat", fake_chat)
    monkeypatch.setattr(server, "_call_llm_responses", fake_responses)

    chat_cfg = server.LlmProviderConfig(
        "mistral", "k", "https://api.mistral.ai/v1/chat/completions", "m", {}
    )
    go_cfg = server.LlmProviderConfig(
        "gospark",
        "k",
        "https://opencode.ai/zen/go/v1/responses",
        "muse-spark-1.3-contributor",
        {"reasoning_effort": "low"},
    )
    assert (
        await server._call_llm_for_config(chat_cfg, prompt="p", max_tokens=10)
    ).content == "- c"
    assert (
        await server._call_llm_for_config(go_cfg, prompt="p", max_tokens=10)
    ).content == "- r"
    assert calls == ["chat", "responses"]


@pytest.mark.asyncio
async def test_call_llm_responses_success_and_429(monkeypatch) -> None:
    """Responses caller: completed payload validates like chat output;
    incomplete (truncation) is rejected; a 429 surfaces status 429 for the
    cooldown path. Headers must carry the Go session identity + custom UA."""
    import server

    seen: dict[str, Any] = {}

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}
            self.text = "raw"

        def json(self):
            return self._payload

    def ok_payload():
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "- summary"}],
                }
            ],
            "usage": {"total_tokens": 100},
        }

    responses = [FakeResponse(200, ok_payload())]

    class FakeClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, base_url, *, headers, json):
            seen["headers"] = dict(headers)
            seen["payload"] = dict(json)
            return responses.pop(0)

    monkeypatch.setattr(server.httpx, "AsyncClient", FakeClient)

    async def allow_acquire(*, estimated_tokens=0):
        return True

    def noop_record(**kwargs):
        return None

    monkeypatch.setattr(server.llm_limiter, "acquire", allow_acquire)
    monkeypatch.setattr(server.llm_limiter, "record_response", noop_record)

    res = await server._call_llm_responses(
        api_key="k",
        base_url="https://opencode.ai/zen/go/v1/responses",
        model="muse-spark-1.3-contributor",
        prompt="Summarize.",
        max_tokens=450,
        extra={"reasoning_effort": "low"},
    )
    assert res.ok and res.content == "- summary"
    assert seen["headers"]["x-opencode-session"].startswith("hn-rewrite-tldr-")
    assert seen["headers"]["User-Agent"] == "hn-rewrite-tldr/1.0"
    assert seen["payload"]["reasoning"] == {"effort": "low"}
    assert "temperature" not in seen["payload"]

    # Truncated (incomplete status) must not validate.
    responses.append(FakeResponse(200, {**ok_payload(), "status": "incomplete"}))
    res = await server._call_llm_responses(
        api_key="k",
        base_url="https://example.test",
        model="m",
        prompt="p",
        max_tokens=10,
    )
    assert not res.ok

    # Provider 429 retried then surfaced with status for the cooldown path.
    responses.extend(
        [
            FakeResponse(429, {"error": {"message": "slow down"}}),
            FakeResponse(429, {"error": {"message": "slow down"}}),
            FakeResponse(429, {"error": {"message": "slow down"}}),
            FakeResponse(429, {"error": {"message": "slow down"}}),
        ]
    )
    res = await server._call_llm_responses(
        api_key="k",
        base_url="https://example.test",
        model="m",
        prompt="p",
        max_tokens=10,
    )
    assert not res.ok and res.status == 429
    assert "slow down" in res.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "article_chars,comment_chars",
    [(10_440, 0), (24_067, 0), (0, 24_000), (24_067, 13_000), (5_000, 24_000)],
)
async def test_generated_tldr_preserves_fixed_pane_allowance(
    monkeypatch: pytest.MonkeyPatch, article_chars: int, comment_chars: int
) -> None:
    import server

    async def fake_call(
        cfg: server.LlmProviderConfig,
        *,
        prompt: str,
        max_tokens: int,
        on_usage: server.LlmUsageRecorder | None = None,
    ) -> server.LlmChatResult:
        words = 120 if article_chars and comment_chars else 240
        assert f"aim for {words} words" in prompt
        return server.LlmChatResult(
            ok=True, content="\n".join(f"- Topic {i}" for i in range(1, 20))
        )

    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(server, "_call_llm_for_config", fake_call)
    result = await server.generate_detailed_tldr(
        "Newsletter",
        article_body="a" * article_chars,
        top_comments="c" * comment_chars,
    )
    assert result.kind == "ok"
    assert result.cacheable
    sections = [part for part in result.tldr.split("### ") if part.strip()]
    expected = [4, 4] if article_chars and comment_chars else [8]
    assert len(sections) == len(expected)
    for section, count in zip(sections, expected, strict=True):
        assert section.count("- Topic ") == count
        assert f"- Topic {count}" in section
        assert f"- Topic {count + 1}" not in section


@pytest.mark.asyncio
async def test_generate_detailed_tldr_combined_prompt_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both generation calls receive their fixed budget and emphasis instruction."""
    import server

    calls = []

    async def mock_call_llm_chat(
        *, api_key, base_url, model, prompt, max_tokens, extra=None
    ):
        calls.append(prompt)
        return server.LlmChatResult(content="- summary", ok=True)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    long_article = "x" * 5_000
    long_comments = "y" * 5_000

    await server.generate_detailed_tldr(
        "Split summary test",
        self_text=long_article,
        top_comments=long_comments,
        article_body="",
    )

    assert len(calls) == 2
    article_prompt, discussion_prompt = calls
    assert "3-4 bullets, aim for 120 words" in article_prompt
    assert "3-4 bullets, aim for 120 words" in discussion_prompt
    assert "Use **bold** key terms" in article_prompt
    assert "Use **bold** key terms in every content bullet" in discussion_prompt


@pytest.mark.asyncio
async def test_call_llm_chat_uses_limiter(monkeypatch):
    import server

    calls = []

    class FakeLimiter:
        async def acquire(self, *, estimated_tokens=0):
            assert estimated_tokens > 0
            calls.append(("acquire", None))
            return True

        def record_response(
            self, *, status, headers, reserved_tokens=0, used_tokens=None
        ):
            assert reserved_tokens > 0
            calls.append(("record_response", status, dict(headers)))

    class FakeResponse:
        status_code = 200
        headers = {"x-ratelimit-remaining-req-minute": "49"}
        text = '{"ok": true}'

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "- summary"}, "finish_reason": "stop"}
                ]
            }

    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, base_url, *, headers, json):
            calls.append(("post", base_url))
            return FakeResponse()

    monkeypatch.setattr(server, "llm_limiter", FakeLimiter())
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeClient)

    result = await server._call_llm_chat(
        api_key="test-key",
        base_url="https://example.test/chat",
        model="test-model",
        prompt="hello",
        max_tokens=10,
    )

    assert result.content == "- summary"
    assert result.ok is True
    assert calls == [
        ("acquire", None),
        ("post", "https://example.test/chat"),
        (
            "record_response",
            200,
            {"x-ratelimit-remaining-req-minute": "49"},
        ),
    ]


async def test_unified_fallback_omits_article_when_no_article_body(
    monkeypatch,
) -> None:
    """Discussion-only stories must not produce an ### Article or ### Story section."""
    import server

    calls = []

    async def mock_call_llm_chat(
        *, api_key, base_url, model, prompt, max_tokens, extra=None
    ):
        calls.append(prompt)
        return server.LlmChatResult(content="- **Discussion** summary", ok=True)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Story links to OpenAI",
        self_text="",
        top_comments="comment 1\ncomment 2",
        article_body="",
    )

    assert "### Article" not in result.tldr
    assert "### Story" not in result.tldr
    assert len(calls) == 1


async def test_generate_detailed_tldr_returns_stub_when_no_content(
    monkeypatch,
) -> None:
    """No article + no comments → short stub, zero LLM calls."""
    import server

    calls: list = []

    async def mock_call_llm_chat(
        *, api_key, base_url, model, prompt, max_tokens, extra=None
    ):
        calls.append(prompt)
        return "should not be called"

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setattr(server, "_call_llm_chat", mock_call_llm_chat)

    result = await server.generate_detailed_tldr(
        "Empty story",
        self_text="",
        top_comments="",
        article_body="",
    )

    assert calls == []
    assert result.kind == "no_content"


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
                    "commentCount": 39,
                    "baseScore": 132,
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
    assert ctx.comment_count == 39
    assert ctx.score == 132


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
            comment_count=39,
            score=132,
        )

    async def mock_fetch_article_body_with_result(url):
        raise AssertionError("LessWrong should not be scraped as articles")

    async def mock_generate_detailed_tldr(title, self_text, top_comments, article_body):
        return server.TldrResult(kind="ok", tldr=f"TLDR: {self_text} | {top_comments}")

    monkeypatch.setattr(
        server, "_fetch_lesswrong_context", mock_fetch_lesswrong_context
    )
    monkeypatch.setattr(
        server, "_fetch_article_body_with_result", mock_fetch_article_body_with_result
    )
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_detailed_tldr)

    resp = local_http.post(
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
    assert updated_story.comment_count == 39
    assert updated_story.score == 132


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


def test_dashboard_renders_user_vote_counts_zero_for_no_feedback(test_env):
    """Fresh user with no feedback → all three counts are 0."""
    port, db, regen_event, handler, user = test_env
    assert handler._render_dashboard_for_user(user) == SKELETON_HTML
    _wait_for_cache(handler, user, handler._dashboard_version(user.id), timeout=3.0)
    resp = local_http.get(
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
    _wait_for_cache(handler, user, handler._dashboard_version(user.id), timeout=3.0)
    resp = local_http.get(
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
    _wait_for_cache(handler, user, handler._dashboard_version(user.id), timeout=3.0)
    resp = local_http.get(
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
    resp = local_http.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": "test_token"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Loading your personalized dashboard" in resp.content
    assert b'meta http-equiv="refresh" content="1"' in resp.content


def _fake_render(ranked: list[RankedStory], *args: object, **kwargs: object) -> bytes:
    """Render stub that exposes what the handler asked for."""
    version = kwargs["dashboard_version"]
    current = kwargs["dashboard_latest_version"]
    return f"v={version}/{current} n={len(ranked)}".encode()


@pytest.fixture
def swr_handler(test_env, mock_embedder, monkeypatch: pytest.MonkeyPatch):
    _, db, _, _, user = test_env

    class SwrHandler(Handler):
        pass

    SwrHandler.config = Config(db_path=db.db_path, server_port=0)
    SwrHandler.db = db
    SwrHandler.embedder = mock_embedder
    SwrHandler._decks = {}
    SwrHandler._dashboard_versions = {}
    SwrHandler._cold_stories = []
    _reset_warm_state(SwrHandler)

    import pipeline

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "generate_dashboard_bytes", _fake_render)

    yield user, SwrHandler

    _cancel_warms(SwrHandler)


def _deck(version: int, built_at: float | None = None) -> DeckState:
    return DeckState([], time.time() if built_at is None else built_at, version)


def test_dashboard_stale_hit_renders_deck_as_stale_and_queues_warm(swr_handler):
    # A stale deck is still served immediately, but rendered with the live
    # version as data-current-version so the client's pageVer < currVer
    # check fires and it polls for the refreshed deck.
    user, h = swr_handler
    h._decks[user.id] = _deck(1)
    h._dashboard_versions[user.id] = 2  # current = generation 1 + 2 = 3
    calls: list[tuple[int, int, bool]] = []
    h._trigger_warm = classmethod(  # type: ignore[method-assign]
        lambda cls, warm_user, version, delay_s=0.0, expedite=True: calls.append(
            (warm_user.id, version, expedite)
        )
    )

    assert h._render_dashboard_for_user(user) == b"v=1/3 n=0"
    # Passive: must not cut short a queued vote-debounce warm.
    assert calls == [(user.id, 3, False)]


def test_dashboard_cache_hit_renders_current_without_warm(swr_handler):
    user, h = swr_handler
    h._dashboard_versions[user.id] = 2
    h._decks[user.id] = _deck(3)

    assert h._render_dashboard_for_user(user) == b"v=3/3 n=0"
    assert not _has_pending_warm(h)


def test_trigger_warm_coalesces_to_newest_pending_version(swr_handler):
    user, h = swr_handler
    h._dashboard_versions[user.id] = 41
    h._trigger_warm(user, version=42, delay_s=5.0)
    h._trigger_warm(user, version=42, delay_s=5.0)
    assert h._warm_scheduler().pending_version(user.id) == 42

    other = h.db.create_user("coalesce_other")
    h._trigger_warm(other, version=1, delay_s=5.0)
    h._dashboard_versions[other.id] = 1
    h._trigger_warm(other, version=2, delay_s=5.0)
    assert h._warm_scheduler().pending_version(other.id) == 2


def test_trigger_warm_skips_when_deck_is_fresh(swr_handler):
    user, h = swr_handler
    h._decks[user.id] = _deck(h._dashboard_version(user.id))
    h._trigger_warm(user, version=1)
    assert not _has_pending_warm(h)


def test_trigger_warm_never_requests_below_live_version(swr_handler):
    # A late request for an old version still builds the live one.
    user, h = swr_handler
    h._dashboard_versions[user.id] = 4
    h._trigger_warm(user, version=1, delay_s=5.0)
    assert h._warm_scheduler().pending_version(user.id) == 5


def test_evict_old_decks_keeps_newest(swr_handler):
    user, h = swr_handler
    h._MAX_CACHED_DECKS = 100
    for i in range(102):
        h._decks[i] = _deck(0, built_at=float(i))

    with h._dashboard_versions_guard:
        h._evict_old_decks_locked()

    assert len(h._decks) == 100
    assert 0 not in h._decks and 1 not in h._decks
    assert 99 in h._decks and 101 in h._decks


def test_pool_changed_stales_every_deck_and_queues_cached_users(swr_handler):
    # Regen/RSS refresh bumps the pool generation once: every user's live
    # version advances (including users with no vote counter and users with
    # no cached deck), and each cached user gets a refresh queued. A cached
    # deck for a user id no longer in the users table is skipped.
    user, h = swr_handler
    other = h.db.create_user("pool_other")
    h._dashboard_versions = {user.id: 2}
    h._decks = {user.id: _deck(3), other.id: _deck(1), 999999999: _deck(1)}
    rebuilt: list[bool] = []
    h._rebuild_cold_deck = classmethod(  # type: ignore[method-assign]
        lambda cls: rebuilt.append(True)
    )
    calls: list[tuple[int, int]] = []
    h._trigger_warm = classmethod(  # type: ignore[method-assign]
        lambda cls, warm_user, version, delay_s=0.0, **_: calls.append(
            (warm_user.id, version)
        )
    )
    before = {uid: h._dashboard_version(uid) for uid in (user.id, other.id, 12345)}

    h._pool_changed()

    assert rebuilt == [True]
    assert {uid: h._dashboard_version(uid) for uid in before} == {
        uid: v + 1 for uid, v in before.items()
    }
    assert sorted(calls) == sorted([(user.id, 4), (other.id, 2)])


def test_setFilter_preserves_sort_age_source_refresh_behavior() -> None:
    """Tab changes share setFilter while preserving refresh and filter rules."""
    _, static = _read_template_and_static()
    idx = static.index("function setFilter(")
    end = static.index("\n\n    applyGradient();", idx)
    body = static[idx:end]
    assert "scheduleDeckRefresh({ advance: true })" in body
    assert "orderForCurrentSort()" in body
    assert "showNextCard({ allowRefresh: false, excludeActive: true })" in body
    assert "excludeActive = false" in static
    assert "!excludeActive || card !== activeCard" in static
    assert body.count("focusActiveCard()") >= 3
    assert "matchesCurrentCombo(activeCard)" in body
    assert "filterName === 'sort' && value === 'popular'" in body
    assert "currentSource === 'non-hn'" in body
    assert "popularTab.disabled = (value === 'non-hn')" in body
    assert "currentSort = 'recommended'" in body
    assert "updateFilterTabs('sort', currentSort)" in body
    assert "scheduleIdleAgePrefetch()" not in body
    assert "scheduleIdleAgePrefetch" not in static
    assert "FILTERS" in static
    assert "refillQueued" not in body
    assert "refillWhenReady" not in body


def test_deck_actions_restore_native_focus_to_active_card() -> None:
    """Deck-changing actions share deferred, non-scrolling card focus."""
    template, inline_script = _read_template_and_static()
    focus_block = inline_script.split("function focusActiveCard()", 1)[1].split(
        "function setActiveCard", 1
    )[0]
    assert "activeCard?.isConnected" in focus_block
    assert "activeCard.focus({ preventScroll: true })" in focus_block
    assert "first-time-tip" in focus_block

    set_active_block = inline_script.split("function setActiveCard", 1)[1].split(
        "function updateVoteBar", 1
    )[0]
    assert "focusActiveCard()" in set_active_block

    submit_block = inline_script.split("function submitVote(", 1)[1].split(
        "function ", 1
    )[0]
    assert "scheduleVoteRefresh(data);\n          focusActiveCard();" in submit_block
    assert (
        "const preferred = nextQueuedSibling(card);\n"
        "          card.remove();\n"
        "          showNextCard({ preferred });\n"
        "          focusActiveCard();"
    ) in submit_block
    undo_block = inline_script.split("function undoLastVote()", 1)[1].split(
        "function ", 1
    )[0]
    assert "scheduleVoteRefresh(data);\n        focusActiveCard();" in undo_block

    key_actions = inline_script.split("const KEY_ACTIONS =", 1)[1].split(
        "document.addEventListener('keydown'", 1
    )[0]
    assert (
        "document.body.classList.toggle('fullscreen');\n        focusActiveCard();"
        in key_actions
    )
    assert "t: () => refreshTldr()" in key_actions
    assert "s: () => refreshDeck()" in key_actions
    assert "function refreshTldr(card)" in inline_script
    assert "openTldrDetail(card || activeCard, { force: true })" in inline_script
    assert "function refreshDeck()" in inline_script
    assert "queueRefill(false)" in inline_script
    assert "refreshTldr(card);" in inline_script
    assert ">t</span> re-summarize TLDR" in template
    assert ">s</span> refresh deck" in template
    for hint, label in (
        ("r", "sort recommended"),
        ("p", "sort popular"),
        ("x", "sort explore"),
        ("d", "sort date"),
        ("e", "age recent"),
        ("a", "age archive"),
    ):
        assert f">{hint}</span> {label}" in template
    key_action_buttons = inline_script.split(
        "document.querySelectorAll('[data-key-action]').forEach", 1
    )[1].split("async function fetchRefillDoc", 1)[0]
    assert (
        "document.body.classList.toggle('fullscreen');\n          focusActiveCard();"
        in key_action_buttons
    )
    # Side-rail open rows must open, not vote: runKeyAction routes them to
    # openStoryUrl instead of falling through to submitVote.
    assert "openStoryUrl('article')" in key_action_buttons
    assert "openStoryUrl('comments')" in key_action_buttons
    assert (
        "max-height: calc(100dvh - var(--vote-bar-height) - var(--page-gutter));"
        in template
    )

    refill_block = inline_script.split("async function refillQueue", 1)[1].split(
        "document.querySelectorAll('[data-fb]')", 1
    )[0]
    assert "document.activeElement === activeCard" in refill_block
    assert "activeCard.focus({ preventScroll: true })" in refill_block
    assert refill_block.index("orderForCurrentSort()") < refill_block.index(
        "activeCard.focus({ preventScroll: true })"
    )


def test_submitVote_advances_to_the_voted_cards_successor_not_the_deck_head() -> None:
    """Voting must not reset the viewer to the top of the stack: the
    successor is resolved from the voted card's DOM position before removal,
    and showNextCard only trusts it if it's still connected and still
    eligible (guards against a race with a concurrent refill/filter change).
    """
    _, inline_script = _read_template_and_static()

    submit_block = inline_script.split("function submitVote(", 1)[1].split(
        "function ", 1
    )[0]
    # Successor must be captured from `card` (the voted card) before it is
    # removed from the DOM, not from queuedCards() head-of-deck afterward.
    assert (
        "const preferred = nextQueuedSibling(card);\n          card.remove();"
    ) in submit_block

    show_next_block = inline_script.split("function showNextCard(", 1)[1].split(
        "function ", 1
    )[0]
    assert "preferred = null" in show_next_block
    assert "preferred.isConnected" in show_next_block
    assert "queue.includes(preferred)" in show_next_block
    # Falls back to the original head-of-deck pick when preferred is stale.
    assert (
        "queue.find(card => !excludeActive || card !== activeCard)" in show_next_block
    )

    next_sibling_block = inline_script.split("function nextQueuedSibling(", 1)[1].split(
        "function ", 1
    )[0]
    assert "nextElementSibling" in next_sibling_block
    assert "queue.includes(el)" in next_sibling_block  # capped view


def test_rendered_cards_carry_client_contract_attributes(test_env):
    """Served cards carry age, combo, source, position and version attributes."""
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
    _wait_for_cache(handler, user, handler._dashboard_version(user.id), timeout=3.0)
    resp = local_http.get(
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

    # Every served card carries the attributes the client filters, orders and
    # logs interactions by.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")
    stories_el = soup.select_one("#stories")
    assert stories_el is not None
    page_version = stories_el["data-dashboard-version"]
    cards = soup.select("article.story-card")
    assert [int(str(c["data-position"])) for c in cards] == list(range(len(cards)))
    for card in cards:
        combos = str(card["data-combo"]).split()
        age = "recent" if card["data-is-recent"] == "1" else "archive"
        assert f"{age}_mixed" in combos
        assert card["data-is-hn"] == "1"  # both fixtures are HN-family sources
        assert card["data-dashboard-version"] == page_version
        assert card["data-ranker-arm"] == "baseline"
        kinds = {a["data-event-kind"] for a in card.select("a[data-event-kind]")}
        assert "comments_open" in kinds


def test_deck_cards_returns_only_card_fragment(test_env) -> None:
    """The fragment must carry the same cards as the full page but none of
    the static shell (Pico CSS, custom CSS, inline JS) - that's the whole
    point of the slim refill endpoint."""
    port, db, regen_event, handler, user = test_env
    now = int(time.time())
    story_id = 6001
    db.upsert_story(
        Story(
            id=story_id,
            title="Deck cards fragment story",
            url=None,
            score=250,
            time=now - 3600,
            text_content="A story with content.",
            source="hn",
            comment_count=4,
        )
    )
    handler._render_dashboard_for_user(user)
    _wait_for_cache(handler, user, handler._dashboard_version(user.id), timeout=3.0)

    full = local_http.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    fragment_resp = local_http.get(
        f"http://127.0.0.1:{port}/api/deck-cards",
        cookies={"hn_token": user.token},
    )

    assert fragment_resp.status_code == 200
    assert fragment_resp.headers["content-type"].startswith("text/html")
    assert (
        fragment_resp.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
    )
    fragment = fragment_resp.text
    assert f'data-story-id="{story_id}"' in fragment
    assert "<style" not in fragment
    assert "<script" not in fragment
    assert len(fragment) < len(full.text) * 0.5


def test_deck_cards_triggers_warm_on_stale_cache(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # /api/deck-cards is the endpoint every in-tab refill (vote, filter tab
    # click) calls. Unlike GET / it used to never compare its cached
    # version to the live one, so an open tab could poll this endpoint
    # forever and always get the same stale deck (see WORKLOG 2026-08-28).
    # It must now self-heal the same way GET /'s stale_hit branch does:
    # serve the stale fragment immediately (SWR), but kick off a warm.
    port, _, _, handler, user = test_env
    cached_html = b'<!--cards:start--><div data-story-id="7001"></div><!--cards:end-->'
    import pipeline

    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", lambda *a, **kw: cached_html
    )
    handler._decks[user.id] = DeckState([], time.time(), 1)
    handler._dashboard_versions[user.id] = 2  # current = 3
    calls: list[int] = []
    monkeypatch.setattr(
        handler,
        "_trigger_warm",
        classmethod(
            lambda cls, warm_user, version, delay_s=0.0, **_: calls.append(version)
        ),
    )

    response = local_http.get(
        f"http://127.0.0.1:{port}/api/deck-cards",
        cookies={"hn_token": user.token},
    )

    assert response.status_code == 200
    assert 'data-story-id="7001"' in response.text
    assert calls == [3]


def test_deck_cards_does_not_warm_when_cache_is_current(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    port, _, _, handler, user = test_env
    cached_html = b'<!--cards:start--><div data-story-id="7001"></div><!--cards:end-->'
    import pipeline

    monkeypatch.setattr(
        pipeline, "generate_dashboard_bytes", lambda *a, **kw: cached_html
    )
    handler._decks[user.id] = DeckState([], time.time(), 3)
    handler._dashboard_versions[user.id] = 2  # current = 3
    calls: list[int] = []
    monkeypatch.setattr(
        handler,
        "_trigger_warm",
        classmethod(
            lambda cls, warm_user, version, delay_s=0.0, **_: calls.append(version)
        ),
    )

    response = local_http.get(
        f"http://127.0.0.1:{port}/api/deck-cards",
        cookies={"hn_token": user.token},
    )

    assert response.status_code == 200
    assert calls == []


def test_justext_rejects_sidebar_boilerplate() -> None:
    """jusText must classify navigation/sidebar <article> fragments as
    boilerplate and extract only the main content paragraph."""
    import server as srv
    import justext

    # Simulate a page where the actual article is a single <p> in an
    # unclassed <article>, surrounded by many classed <article> sidebar
    # widgets.
    html = """\
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<nav><a href="/">Home</a> <a href="/news">News</a> <a href="/about">About</a></nav>
<article>
  <h1>Climate Scientists Discover New Approach to Carbon Capture</h1>
  <p>A team of researchers at MIT has developed a novel electrochemical
  process that removes carbon dioxide from seawater at half the energy cost
  of existing methods. The technique, published this week in Nature, uses a
  bismuth-based electrode that selectively binds CO2 molecules even at the
  low concentrations found in ocean water. Scaling the process could help
  mitigate the 30% of anthropogenic CO2 currently absorbed by the oceans.</p>
  <p>Dr. Sarah Chen, lead author, noted that "the electrode material is
  abundant and the process operates at room temperature — this is not a
  lab curiosity, it is an engineering challenge now." The team is working
  with a spin-off company to build a pilot plant by 2028.</p>
</article>
<article class="column sidebar">
  <h2>MOST POPULAR</h2>
  <ul>
    <li><a href="/article/1">Tech Giant Lays Off 5000 Workers</a></li>
    <li><a href="/article/2">New Programming Language Gains Traction</a></li>
    <li><a href="/article/3">Mars Rover Discovers Ancient Lake Bed</a></li>
  </ul>
</article>
<article class="column sidebar">
  <h2>RELATED STORIES</h2>
  <ul>
    <li><a href="/article/4">Ocean Acidification Study Released</a></li>
    <li><a href="/article/5">Renewable Energy Hits Record Output</a></li>
  </ul>
</article>
<footer>Copyright 2026. All rights reserved. Contact us. Privacy Policy.</footer>
</body>
</html>"""

    text = srv._extract_with_justext(html)
    assert text is not None
    assert len(text) >= 500
    assert "Carbon Capture" in text
    assert "MOST POPULAR" not in text
    assert "RELATED STORIES" not in text
    assert "Tech Giant" not in text
    assert "Privacy Policy" not in text

    # Also verify jusText paragraph classification directly
    paragraphs = justext.justext(html, justext.get_stoplist("English"))
    good_count = sum(1 for p in paragraphs if not p.is_boilerplate)
    assert good_count >= 2  # the two main content paragraphs
    boilerplate_count = sum(1 for p in paragraphs if p.is_boilerplate)
    assert boilerplate_count > good_count  # sidebar dominates the page

    # Verify BS semantic also works as fallback
    text_bs = srv._extract_with_bs_semantic(html)
    assert text_bs is not None
    assert "Carbon Capture" in text_bs
    assert "MOST POPULAR" not in text_bs


def test_on_demand_tldr_records_fetch_failure(test_env, monkeypatch):
    """On-demand article fetch failure records in article_fetch_failures."""
    import server
    import time as time_mod

    port, db, _, _, user = test_env
    db.upsert_story(
        Story(
            id=1001,
            title="Failure test",
            url="https://example.com/failing",
            score=10,
            time=int(time_mod.time()) - 3600,
            text_content="Failure test.",
            source="hn",
            comment_count=0,
            discussion_url=None,
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    async def mock_fetch(url):
        return server.ArticleFetchResult(status=403, error="http_403")

    async def mock_generate_tldr(title, self_text, top_comments, article_body):
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title}")

    monkeypatch.setattr(server, "_fetch_article_body_with_result", mock_fetch)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_tldr)

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": 1001},
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200

    failure = db.get_article_fetch_failure(1001)
    assert failure is not None
    assert failure.last_status == 403
    assert failure.last_error == "http_403"
    assert failure.failure_count == 1


def test_on_demand_tldr_clears_failure_on_success(test_env, monkeypatch):
    """On-demand success clears prior failure record."""
    import server
    import time as time_mod

    port, db, _, _, user = test_env
    sid = 1002
    db.upsert_story(
        Story(
            id=sid,
            title="Success test",
            url="https://example.com/recovering",
            score=10,
            time=int(time_mod.time()) - 3600,
            text_content="Success test.",
            source="hn",
            comment_count=0,
            discussion_url=None,
            comment_count_at_fetch=0,
            self_text="",
            top_comments="",
            article_body="",
        )
    )

    # Pre-populate a failure record
    db.record_article_fetch_failure(
        sid, "https://example.com/recovering", status=503, error="http_503"
    )
    assert db.get_article_fetch_failure(sid) is not None

    async def mock_fetch(url):
        return server.ArticleFetchResult(
            body="Recovered article body content here",
            status=200,
        )

    async def mock_generate_tldr(title, self_text, top_comments, article_body):
        return server.TldrResult(kind="ok", tldr=f"TLDR: {title} | {article_body}")

    monkeypatch.setattr(server, "_fetch_article_body_with_result", mock_fetch)
    monkeypatch.setattr(server, "generate_detailed_tldr", mock_generate_tldr)

    resp = local_http.post(
        f"http://127.0.0.1:{port}/api/tldr-detail",
        json={"story_id": sid},
        cookies={"hn_token": user.token},
    )
    assert resp.status_code == 200

    # Failure record should be cleared
    assert db.get_article_fetch_failure(sid) is None

    # Article body should be stored
    updated = db.get_story(sid)
    assert updated is not None
    assert updated.article_body == "Recovered article body content here"
    # Embedding refresh is intentionally deferred to the warm/regen
    # article-fetch path (off the tap critical path); the tap persists
    # the body and clears the failure only.


def test_warm_background_task_dedupes_in_flight_ids(test_env, monkeypatch):
    """Overlapping warm tasks skip stories already in-flight."""
    import server as srv
    from pipeline import RankedStory, Config

    port, db, _, _, _ = test_env

    now = int(time.time())
    s1 = Story(
        id=3001,
        title="S1",
        url="https://example.com/s1",
        score=10,
        time=now - 3600,
        text_content="s1",
        source="hn",
    )
    s2 = Story(
        id=3002,
        title="S2",
        url="https://example.com/s2",
        score=10,
        time=now - 3600,
        text_content="s2",
        source="hn",
    )
    s3 = Story(
        id=3003,
        title="S3",
        url="https://example.com/s3",
        score=10,
        time=now - 3600,
        text_content="s3",
        source="hn",
    )
    for s in [s1, s2, s3]:
        db.upsert_story(s)

    ranked = [
        RankedStory(story=s, score=1.0, best_match_title="") for s in [s1, s2, s3]
    ]

    cfg = Config.load()

    # Mark s1 and s2 as in-flight, leave s3 free
    srv.Handler._article_fetch_in_flight = {3001, 3002}

    fetched_ids: list[int] = []

    async def noop_fetch(*args, **kwargs):
        fetched_ids.extend(s.id for s in kwargs["stories"])
        return {}

    import pipeline

    monkeypatch.setattr(pipeline, "fetch_and_cache_article_bodies", noop_fetch)

    srv.Handler._warm_background_tasks(
        ranked,
        db,
        MockEmbedder(),
        cfg,
        per_combo=0,
    )

    # Only s3 should have been added to in-flight during the task (then cleared)
    # s1 and s2 remain unchanged since they were already in-flight
    assert fetched_ids == [3003]
    assert 3001 in srv.Handler._article_fetch_in_flight
    assert 3002 in srv.Handler._article_fetch_in_flight
    assert 3003 not in srv.Handler._article_fetch_in_flight


def test_warm_background_article_fetch_failure_still_prefetches_tldrs(
    test_env, monkeypatch
):
    """Article fetch is best-effort; TLDR prefetch must still run on failure."""
    import pipeline
    import server as srv
    from pipeline import Config, RankedStory

    _port, db, _, _, _ = test_env
    story = Story(
        id=3010,
        title="Failure should not block TLDR",
        url="https://example.com/failure",
        score=10,
        time=int(time.time()) - 3600,
        text_content="failure should not block tldr",
        source="hn",
    )
    db.upsert_story(story)
    ranked = [RankedStory(story=story, score=1.0, best_match_title="")]
    srv.Handler._article_fetch_in_flight = set()

    async def failing_fetch(*args, **kwargs):
        raise RuntimeError("boom")

    prefetch_calls: list[list[int]] = []

    async def capture_prefetch(
        ranked_stories,
        database,
        per_combo,
        stale_per_run=0,
        date_top_n=0,
        stagger_s=None,
    ):
        prefetch_calls.append([rs.story.id for rs in ranked_stories])
        return 1

    monkeypatch.setattr(pipeline, "fetch_and_cache_article_bodies", failing_fetch)
    monkeypatch.setattr(srv, "_prefetch_tldrs_for_ranked", capture_prefetch)
    monkeypatch.setattr(srv.Handler, "_tldr_prefetch_gate", srv.BackgroundCadence())

    srv.Handler._warm_background_tasks(
        ranked,
        db,
        MockEmbedder(),
        Config(article_fetch_max_per_run=10),
        per_combo=1,
    )

    assert prefetch_calls == [[3010]]
    assert srv.Handler._article_fetch_in_flight == set()


def test_quiet_third_party_loggers_silences_httpx_and_trafilatura() -> None:
    """httpx INFO and trafilatura's benign "discarding data" WARNING made up
    ~88% of a week of journal volume in production, drowning out anything
    actionable during an incident (see WORKLOG 2026-08-27)."""
    import logging
    import server as srv

    httpx_logger = logging.getLogger("httpx")
    trafilatura_logger = logging.getLogger("trafilatura")
    orig_httpx_level = httpx_logger.level
    orig_trafilatura_level = trafilatura_logger.level
    try:
        httpx_logger.setLevel(logging.NOTSET)
        trafilatura_logger.setLevel(logging.NOTSET)

        srv._quiet_third_party_loggers()

        assert httpx_logger.level == logging.WARNING
        assert trafilatura_logger.level == logging.ERROR
    finally:
        httpx_logger.setLevel(orig_httpx_level)
        trafilatura_logger.setLevel(orig_trafilatura_level)


@pytest.mark.parametrize(
    ("rank_ms", "stage_sum_ms", "expected"),
    [
        (54_000.0, 12_000.0, True),  # monster warm, stages explain little
        (6_230_000.0, 900.0, True),  # 103-min stall, near-zero stages
        (9_000.0, 3_500.0, False),  # ordinary slow warm, below 20s floor
        (25_000.0, 20_000.0, False),  # expensive but accounted compute
        (25_000.0, 8_000.0, True),  # just over the 3x boundary
        (21_000.0, 7_000.0, False),  # exactly 3x is not starvation
    ],
)
def test_warm_is_starved_predicate(
    rank_ms: float, stage_sum_ms: float, expected: bool
) -> None:
    import server

    assert server._warm_is_starved(rank_ms, stage_sum_ms) is expected


def test_usage_int_parses_defensively() -> None:
    import server

    assert server._usage_int({"prompt_tokens": 10}, "prompt_tokens") == 10
    assert server._usage_int({"prompt_tokens": -1}, "prompt_tokens") is None
    assert server._usage_int({"prompt_tokens": "10"}, "prompt_tokens") is None
    assert server._usage_int({"other": 1}, "prompt_tokens") is None
    assert server._usage_int(None, "prompt_tokens") is None
    assert server._usage_int("usage", "prompt_tokens") is None


def test_note_llm_usage_swallows_recorder_errors() -> None:
    """Usage recording must never break TLDR serving: a failing recorder
    is a log line, not an exception."""
    import server

    def bad_recorder(provider, in_tok, out_tok, reasoning_tok):
        raise RuntimeError("db is down")

    result = server.LlmChatResult(content="- hi", ok=True)
    server._note_llm_usage(None, "mistral", result)
    server._note_llm_usage(bad_recorder, "mistral", result)


async def test_call_llm_for_config_records_chat_usage(monkeypatch) -> None:
    import server

    recorded = []

    class FakeLimiter:
        async def acquire(self, *, estimated_tokens=0):
            return True

        def record_response(self, **kwargs):
            pass

    class FakeResponse:
        status_code = 200
        headers = {}
        text = "{}"

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "- summary"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }

    class FakeClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, base_url, *, headers, json):
            return FakeResponse()

    monkeypatch.setattr(server, "llm_limiter", FakeLimiter())
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeClient)

    cfg = server.LlmProviderConfig(
        provider="mistral",
        api_key="test-key",
        base_url="https://api.mistral.ai/v1/chat/completions",
        model="mistral-small-latest",
        extra={},
    )
    result = await server._call_llm_for_config(
        cfg, prompt="hello", max_tokens=10, on_usage=lambda *a: recorded.append(a)
    )

    assert result.ok is True
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens) == (
        10,
        20,
        None,
    )
    assert recorded == [("mistral", 10, 20, None)]


async def test_call_llm_responses_records_reasoning_usage(monkeypatch) -> None:
    import server

    recorded = []

    class FakeLimiter:
        async def acquire(self, *, estimated_tokens=0):
            return True

        def record_response(self, **kwargs):
            pass

    class FakeResponse:
        status_code = 200
        headers = {}
        text = "{}"

        def json(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "- summary"}],
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "output_tokens_details": {"reasoning_tokens": 12},
                },
                "status": "completed",
            }

    class FakeClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, base_url, *, headers, json):
            return FakeResponse()

    monkeypatch.setattr(server, "llm_limiter", FakeLimiter())
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeClient)

    cfg = server.LlmProviderConfig(
        provider="gospark",
        api_key="test-key",
        base_url="https://opencode.ai/zen/go/v1/responses",
        model="muse-spark",
        extra={},
    )
    result = await server._call_llm_for_config(
        cfg, prompt="hello", max_tokens=100, on_usage=lambda *a: recorded.append(a)
    )

    assert result.ok is True
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens) == (
        100,
        50,
        12,
    )
    assert recorded == [("gospark", 100, 50, 12)]


def test_log_llm_spend_today_logs_estimate(caplog) -> None:
    """One regen = one spend line per provider, with a $ estimate for mistral."""
    import logging

    import server
    from database import Database

    db = Database(":memory:")
    try:
        db.record_llm_usage("mistral", 1000, 2000, None)
        with caplog.at_level(logging.INFO):
            server._log_llm_spend_today(db)
        assert any(
            "llm_spend_today" in r.message
            and "provider=mistral" in r.message
            and "est_usd=0.0007" in r.message
            for r in caplog.records
        )
        # Empty day: no lines, no crash.
        caplog.clear()
        empty_db = Database(":memory:")
        try:
            with caplog.at_level(logging.INFO):
                server._log_llm_spend_today(empty_db)
            assert not any("llm_spend_today" in r.message for r in caplog.records)
        finally:
            empty_db.close()
    finally:
        db.close()


async def test_generate_detailed_tldr_records_usage_via_global_recorder(
    monkeypatch,
) -> None:
    """generate_detailed_tldr must record token usage through the global
    recorder bound in main() (no signature change, existing mocks intact)."""
    import server
    from database import Database

    async def fake_call_llm_for_config(cfg, *, prompt, max_tokens, on_usage=None):
        result = server.LlmChatResult(
            content="- **Discussion** summary",
            ok=True,
            input_tokens=5,
            output_tokens=7,
        )
        server._note_llm_usage(on_usage, cfg.provider, result)
        return result

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setattr(server, "_call_llm_for_config", fake_call_llm_for_config)

    db = Database(":memory:")
    old = server._llm_usage_recorder
    server.set_llm_usage_recorder(db.record_llm_usage)
    try:
        result = await server.generate_detailed_tldr(
            "Story links to OpenAI",
            self_text="",
            top_comments="comment 1\ncomment 2",
            article_body="",
        )
        assert result.kind == "ok"
        import time as time_mod

        rows = db.get_llm_usage_day(time_mod.strftime("%Y-%m-%d"))
        assert rows == [
            {
                "provider": "mistral",
                "calls": 1,
                "input_tokens": 5,
                "output_tokens": 7,
                "reasoning_tokens": 0,
            }
        ]
    finally:
        server.set_llm_usage_recorder(old)
        db.close()


def _article_only_story(story_id: int) -> Story:
    return Story(
        id=story_id,
        title=f"Article story {story_id}",
        url=f"https://example.com/article-{story_id}",
        score=10,
        time=1600000000,
        text_content="Story body.",
        source="hn",
        comment_count=0,
        self_text="",
        top_comments="",
        article_body="Story body.",
    )


def test_profile_link_asks_before_replacing_another_live_session(
    test_env: Any,
) -> None:
    """Another site can link a visitor to /u/<its token>; a GET must not
    silently swap the visitor's existing profile. Only a same-origin POST
    switches; a device without a profile still switches in one click."""
    _, db, _, handler, user = test_env
    other = db.create_user("other_profile_token")
    app = create_app(handler)

    def client_with(token: str | None) -> Any:
        client = app.test_client()
        if token is not None:
            client.set_cookie("hn_token", token)
        return client

    ask = client_with(other.token).get(f"/u/{user.token}")
    assert ask.status_code == 200
    assert "Set-Cookie" not in ask.headers
    assert ask.headers["Referrer-Policy"] == "no-referrer"
    assert b'<form method="post">' in ask.data

    cross = client_with(other.token).post(
        f"/u/{user.token}", headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert cross.status_code == 403
    assert "Set-Cookie" not in cross.headers

    switch = client_with(other.token).post(
        f"/u/{user.token}", headers={"Sec-Fetch-Site": "same-origin"}
    )
    assert switch.status_code == 302
    assert switch.headers["Set-Cookie"].startswith(f"hn_token={user.token}")

    for token in (None, user.token, "stale-token-not-in-db"):
        direct = client_with(token).get(f"/u/{user.token}")
        assert direct.status_code == 302
        assert direct.headers["Set-Cookie"].startswith(f"hn_token={user.token}")


@pytest.mark.parametrize(
    ("proto", "secure"), [("https", True), ("http", False), (None, False)]
)
def test_session_cookie_is_secure_only_over_https(
    test_env: Any, proto: str | None, secure: bool
) -> None:
    _, _, _, handler, user = test_env
    app = create_app(handler)
    headers = {"X-Forwarded-Proto": proto} if proto else {}

    for path in ("/", f"/u/{user.token}"):
        cookie = app.test_client().get(path, headers=headers).headers["Set-Cookie"]
        assert cookie.startswith("hn_token=")
        assert ("; Secure" in cookie) is secure


@pytest.mark.parametrize("path", ["/api/feedback", "/api/tldr-detail"])
@pytest.mark.parametrize("body", ["[]", "1", '"story"', "null"])
def test_non_object_json_bodies_are_bad_requests(
    test_env: Any, path: str, body: str
) -> None:
    _, _, _, handler, user = test_env
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    resp = client.post(path, data=body, content_type="application/json")

    assert resp.status_code == 400


def test_vote_on_unknown_story_is_404_without_spending_quota(
    test_env: Any,
) -> None:
    _, db, _, handler, user = test_env
    handler.config = replace(handler.config, feedback_per_user_limit=1)
    db.upsert_story(_article_only_story(1901))
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)

    missing = client.post(
        "/api/feedback", json={"story_id": 99_999_991, "action": "up"}
    )
    real = client.post("/api/feedback", json={"story_id": 1901, "action": "up"})

    assert missing.status_code == 404
    assert real.status_code == 200


def test_tldr_generation_cap_serves_stale_or_busy_and_always_releases(
    test_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over the concurrency cap, a request gets the stale TLDR or a short
    429 without calling the LLM or spending quota; a held slot is released
    however the request ends, including when generation raises."""
    import server

    _, db, _, handler, user = test_env
    for story_id in (1902, 1903):
        db.upsert_story(_article_only_story(story_id))
    db.upsert_tldr_cache(1903, "old-key", "Old TLDR")
    client = create_app(handler).test_client()
    client.set_cookie("hn_token", user.token)
    llm_calls: list[str] = []

    async def fail_generate(
        title: str, self_text: str, top_comments: str, article_body: str
    ) -> "server.TldrResult":
        llm_calls.append(title)
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(server, "generate_detailed_tldr", fail_generate)

    handler.config = replace(
        handler.config,
        tldr_max_concurrent_generations=0,
        tldr_uncached_per_user_limit=1,
    )
    busy = client.post("/api/tldr-detail", json={"story_id": 1902})
    stale = client.post(
        "/api/tldr-detail", json={"story_id": 1903, "force_refresh": True}
    )
    assert busy.status_code == 429
    assert busy.headers["Retry-After"] == "5"
    assert stale.status_code == 200
    assert stale.get_json()["tldr"] == "Old TLDR"
    assert llm_calls == []

    handler.config = replace(handler.config, tldr_max_concurrent_generations=1)
    failed = client.post("/api/tldr-detail", json={"story_id": 1902})
    assert llm_calls == ["Article story 1902"]
    assert failed.status_code >= 400
    assert handler._tldr_generations == 0
