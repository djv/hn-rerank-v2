import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import httpx
import pytest
from http.server import ThreadingHTTPServer
from hypothesis import given, settings, strategies as st

from server import Handler
from pipeline import Config, Embedder
from database import Database, Story


@pytest.fixture
def test_env(tmp_path):
    db_file = tmp_path / "test_server.db"
    db = Database(str(db_file))

    # Create test user
    user = db.create_user("test_token")

    output_file = tmp_path / "public" / "index.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("<html>Test Dashboard</html>", encoding="utf-8")

    config = Config(
        db_path=str(db_file),
        output=str(output_file),
        server_port=0,
    )

    regen_event = threading.Event()

    class TestHandler(Handler):
        pass

    class MockEmbedder(Embedder):
        def __init__(self):
            pass

        def encode(self, texts: list[str], batch_size: int = 64) -> ...:
            import numpy as np

            return np.zeros((len(texts), 384), dtype=np.float32)

    TestHandler.config = config
    TestHandler.db = db
    TestHandler.embedder = MockEmbedder()
    TestHandler.regen_event = regen_event
    TestHandler._dashboard_cache = {}
    TestHandler._dashboard_versions = {}
    TestHandler._render_locks = {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield port, db, regen_event, output_file, user

    server.shutdown()
    db.close()


def test_token_redirect(test_env):
    port, _, _, _, user = test_env
    resp = httpx.get(f"http://127.0.0.1:{port}/u/{user.token}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"] == "../"
    assert "hn_token" in resp.headers.get("Set-Cookie", "")


def test_first_visit_redirect(test_env):
    port, _, _, _, _ = test_env
    resp = httpx.get(f"http://127.0.0.1:{port}/", follow_redirects=False)
    assert resp.status_code == 302
    location = resp.headers["Location"]
    assert location.startswith("u/")
    assert "hn_token" in resp.headers.get("Set-Cookie", "")


def test_dashboard_route_no_user_creates_token_and_redirects(test_env):
    port, db, _, _, _ = test_env
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
    port, _, _, _, user = test_env
    resp = httpx.get(
        f"http://127.0.0.1:{port}/",
        cookies={"hn_token": user.token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    # Dynamic rendering returns personalized dashboard, not static file
    assert resp.status_code == 200
    assert 'id="queue-status"' in resp.text
    assert 'data-mode="default"' in resp.text
    assert 'data-mode="popular"' in resp.text
    assert 'data-mode="explore"' in resp.text
    assert 'data-mode="date"' in resp.text
    assert 'data-key-action="undo"' in resp.text
    assert 'id="sort-toggle"' not in resp.text


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


def test_feedback_post_defers_refresh_when_queue_not_low(test_env):
    port, db, regen_event, _, user = test_env
    db.upsert_story(
        Story(
            id=1000,
            title="Deferred refresh story",
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
        json={"story_id": 1000, "action": "up", "queue_remaining": 8},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": False}
    assert len(db.get_all_feedback(user.id)) == 1
    assert not regen_event.is_set()


def test_feedback_post_does_not_refresh_from_queue_depth_alone(test_env):
    port, db, regen_event, _, user = test_env
    db.upsert_story(
        Story(
            id=1001,
            title="Low queue refresh story",
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
        json={"story_id": 1001, "action": "up", "queue_remaining": 4},
        cookies={"hn_token": user.token},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "ranking_refresh_queued": False}
    assert len(db.get_all_feedback(user.id)) == 1
    assert not regen_event.is_set()


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


def test_dashboard_cache_uses_feedback_versions(test_env, monkeypatch):
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = object()
    TestHandler._dashboard_cache = {}
    TestHandler._dashboard_versions = {}
    TestHandler._render_locks = {}

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

    html_v0 = TestHandler._render_dashboard_for_user(user)
    assert html_v0 == b"version=0"
    assert len(calls) == 1

    assert TestHandler._render_dashboard_for_user(user) == b"version=0"
    assert len(calls) == 1

    version = TestHandler._invalidate_dashboard_cache(user.id)
    assert version == 1
    html_v1 = TestHandler._render_dashboard_for_user(user)
    assert html_v1 == b"version=1"
    assert len(calls) == 2


def test_stale_warm_render_does_not_overwrite_current_cache(test_env, monkeypatch):
    _, db, _, _, user = test_env

    class TestHandler(Handler):
        pass

    TestHandler.config = Config(db_path=db.db_path, server_port=0)
    TestHandler.db = db
    TestHandler.embedder = object()
    TestHandler._dashboard_cache = {}
    TestHandler._dashboard_versions = {}
    TestHandler._render_locks = {}

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

    current_html = TestHandler._render_dashboard_for_user(
        user, expected_version=new_version
    )
    cache_key = f"dashboard_{user.id}"
    assert TestHandler._dashboard_cache[cache_key][2] == new_version

    stale_html = TestHandler._render_dashboard_for_user(
        user, expected_version=old_version
    )
    assert stale_html == current_html
    assert TestHandler._dashboard_cache[cache_key][2] == new_version


@given(
    operations=st.lists(
        st.sampled_from(["invalidate", "render_current", "render_stale"]),
        min_size=1,
        max_size=40,
    )
)
@settings(max_examples=60, deadline=None)
def test_dashboard_cache_version_invariant_property(operations):
    with TemporaryDirectory() as temp_dir:
        db = Database(str(Path(temp_dir) / "prop_server.db"))
        user = db.create_user("prop_user")

        class TestHandler(Handler):
            pass

        TestHandler.config = Config(db_path=db.db_path, server_port=0)
        TestHandler.db = db
        TestHandler.embedder = object()
        TestHandler._dashboard_cache = {}
        TestHandler._dashboard_versions = {}
        TestHandler._render_locks = {}

        def fake_fast_rerank_for_user(database, config, embedder, user_id):
            return []

        def fake_generate_dashboard_bytes(
            ranked, config, database, user_id, user_token
        ):
            return f"v={TestHandler._dashboard_version(user_id)}".encode()

        import pipeline

        old_rank = pipeline.fast_rerank_for_user
        old_render = pipeline.generate_dashboard_bytes
        pipeline.fast_rerank_for_user = fake_fast_rerank_for_user
        pipeline.generate_dashboard_bytes = fake_generate_dashboard_bytes
        try:
            for operation in operations:
                current_version = TestHandler._dashboard_version(user.id)
                if operation == "invalidate":
                    TestHandler._invalidate_dashboard_cache(user.id)
                elif operation == "render_current":
                    TestHandler._render_dashboard_for_user(user)
                else:
                    stale_version = max(0, current_version - 1)
                    TestHandler._render_dashboard_for_user(
                        user, expected_version=stale_version
                    )

                cache_key = f"dashboard_{user.id}"
                cached = TestHandler._dashboard_cache.get(cache_key)
                if cached is not None:
                    assert cached[2] == TestHandler._dashboard_version(user.id)
        finally:
            pipeline.fast_rerank_for_user = old_rank
            pipeline.generate_dashboard_bytes = old_render
            db.close()


def test_cors_headers(test_env):
    port, _, _, _, _ = test_env
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


def test_tldr_handler_returns_404_for_missing_story(test_env):
    port, _, _, _, user = test_env

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


@pytest.mark.asyncio
async def test_generate_detailed_tldr_splits_article_and_comments(monkeypatch):
    import server

    calls = []

    async def mock_call_llm_chat(*, api_key, base_url, model, prompt, max_tokens):
        calls.append(prompt)
        if "Summarize the article" in prompt:
            return "- **Article** summary"
        if "Summarize the Hacker News discussion" in prompt:
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
    assert "HN comments" in calls[1]
    assert "Points:" not in calls[1]
    assert "Comments:" not in calls[1]
    assert "Age hours:" not in calls[1]
    assert "### Article" in result
    assert "- **Article** summary" in result
    assert "### Discussion" in result
    assert "- **Discussion** summary" in result


def test_keydown_guard_excludes_buttons_and_anchors():
    """Regression: the global keydown handler in templates/index.html must not
    bail out when a <button> or <a> has focus, otherwise clicking a mode tab
    or vote button blocks the next ArrowUp/ArrowDown from registering.
    """
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "index.html"
    ).read_text(encoding="utf-8")
    assert "addEventListener('keydown'" in template, (
        "keydown handler not found in template"
    )

    # Locate the closest(...) call in the guard
    idx = template.index("closest?.(")
    end = template.index(")", idx)
    guard = template[idx : end + 1]

    assert "button" not in guard, "button should not block global shortcuts"
    assert "'a'" not in guard, "a should not block global shortcuts"
    assert "input" in guard, "input should still block"
    assert "textarea" in guard, "textarea should still block"
    assert "select" in guard, "select should still block"
    assert '[contenteditable="true"]' in guard, "contenteditable should still block"
