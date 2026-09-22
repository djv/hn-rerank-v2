from __future__ import annotations

from dataclasses import replace

from database import Database, Story
from server import Handler, _tldr_cache_key, create_app


def test_cache_only_endpoint_handles_misses_and_invalidated_inputs() -> None:
    db = Database(":memory:")

    class Runtime(Handler):
        pass

    Runtime.db = db
    try:
        user = db.create_user("cache-reader")
        story = Story(-1, "Title", None, 0, 1, "Body", self_text="Body")
        db.upsert_story(story)
        client = create_app(Runtime).test_client()
        assert client.get("/api/tldr-cache/-1").status_code == 401
        client.set_cookie("hn_token", user.token)
        # No model/embedder/provider is configured: misses must remain pure reads.
        assert client.get("/api/tldr-cache/-1").status_code == 204
        assert client.get("/api/tldr-cache/999").status_code == 204
        key = _tldr_cache_key(
            title="Title", self_text="Body", top_comments="", article_body=""
        )
        db.upsert_tldr_cache(-1, key, "Cached summary")
        hit = client.get("/api/tldr-cache/-1")
        assert hit.status_code == 200
        assert hit.get_json()["tldr"] == "Cached summary"
        assert hit.headers["Cache-Control"] == "private, no-store"
        db.upsert_story(replace(story, self_text="Body with additional content"))
        assert client.get("/api/tldr-cache/-1").status_code == 204
        assert db.get_any_tldr_for_story(-1) == "Cached summary"
    finally:
        db.close()
