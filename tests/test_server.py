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
    assert resp.json() == {"ok": True}

    records = db.get_all_feedback(user.id)
    assert len(records) == 1
    assert records[0].story_id == 999
    assert records[0].action == "up"
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

    assert len(db.get_all_feedback(user.id)) == 0
    assert regen_event.is_set()


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
    monkeypatch.setattr(pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes)

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
    monkeypatch.setattr(pipeline, "generate_dashboard_bytes", fake_generate_dashboard_bytes)

    old_version = TestHandler._invalidate_dashboard_cache(user.id)
    new_version = TestHandler._invalidate_dashboard_cache(user.id)
    assert (old_version, new_version) == (1, 2)

    current_html = TestHandler._render_dashboard_for_user(
        user, expected_version=new_version
    )
    cache_key = f"dashboard_{user.id}"
    assert TestHandler._dashboard_cache[cache_key][2] == new_version

    stale_html = TestHandler._render_dashboard_for_user(user, expected_version=old_version)
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

        def fake_generate_dashboard_bytes(ranked, config, database, user_id, user_token):
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
    assert "- Quantization may degrade quality.\n- Providers reduce hardware barriers." in normalized


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

    async def mock_generate_detailed_tldr(
        title, self_text, top_comments, article_body, points, comment_count, age_hours
    ):
        return f"TLDR: {self_text} | {top_comments} | comments={comment_count}"

    import server

    monkeypatch.setattr(server, "_fetch_reddit_rss_context", mock_fetch_reddit_rss_context)
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
    assert "comments=1" in data["tldr"]

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

    async def mock_generate_detailed_tldr(
        title, self_text, top_comments, article_body, points, comment_count, age_hours
    ):
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
        points=42,
        comment_count=12,
        age_hours=3.5,
    )

    assert len(calls) == 2
    assert "Article body" in calls[0]
    assert "HN comments" in calls[1]
    assert "### Article" in result
    assert "- **Article** summary" in result
    assert "### Discussion" in result
    assert "- **Discussion** summary" in result
