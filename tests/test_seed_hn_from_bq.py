import hashlib
import subprocess

import httpx
import numpy as np
import pytest

from database import Database, Story
from pipeline import story_embedding_text
from scripts import seed_hn_from_bq


class DummyEmbedder:
    def encode(self, texts: list[str], batch_size: int = 32):
        arr = np.zeros((len(texts), 384), dtype=np.float32)
        if len(texts):
            arr[:, 0] = 1.0
        return arr


def test_bq_row_maps_to_hn_story_with_composed_text():
    story = seed_hn_from_bq.story_from_bq_row(
        {
            "id": "123",
            "title": "<b>Useful story</b>",
            "url": "https://example.com/story",
            "text": "<p>Self text body.</p>",
            "score": "150",
            "descendants": "42",
            "created_at_i": "1760000000",
        }
    )

    assert story is not None
    assert story.id == 123
    assert story.source == "bq_seed"
    assert story.title == "Useful story"
    assert story.self_text == "Self text body."
    assert story.comment_count == 42
    assert story.text_content
    assert "Useful story" in story.text_content
    assert "Self text body" in story.text_content


def test_build_bq_query_accepts_months_min_score_and_limit():
    query = seed_hn_from_bq.build_bq_query(months=9, min_score=250, limit=500)

    assert "INTERVAL 9 MONTH" in query
    assert "score >= 250" in query
    assert query.rstrip().endswith("LIMIT 500")


def test_build_bq_query_validates_bounds():
    with pytest.raises(ValueError, match="months"):
        seed_hn_from_bq.build_bq_query(months=0)
    with pytest.raises(ValueError, match="min-score"):
        seed_hn_from_bq.build_bq_query(min_score=-1)


@pytest.mark.asyncio
async def test_seed_skips_feedback_story_without_update(monkeypatch):
    db = Database(":memory:")
    try:
        user = db.create_user("u")
        original = Story(
            id=123,
            title="Original",
            url="https://old.example",
            score=1,
            time=1,
            text_content="Original text",
            source="hn",
        )
        db.upsert_story(original)
        db.upsert_feedback(user.id, 123, "up")

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url):
                raise AssertionError("feedback-protected story should not hydrate")

        monkeypatch.setattr(seed_hn_from_bq.httpx, "AsyncClient", MockClient)
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated,
        ) = await seed_hn_from_bq.seed_rows(
            [
                {
                    "id": 123,
                    "title": "Replacement",
                    "url": "https://new.example",
                    "text": "Replacement text",
                    "score": 200,
                    "descendants": 10,
                    "created_at_i": 1760000000,
                }
            ],
            db=db,
            embedder=DummyEmbedder(),
        )

        assert inserted == 0
        assert skipped_feedback == 1
        assert skipped_existing == 0
        assert hydrated == 0
        assert db.get_story(123) == original
    finally:
        db.close()


@pytest.mark.asyncio
async def test_seed_skips_existing_story_without_feedback(monkeypatch):
    db = Database(":memory:")
    try:
        original = Story(
            id=321,
            title="Existing",
            url="https://old.example",
            score=11,
            time=11,
            text_content="Existing text",
            source="hn",
            comment_count=3,
        )
        db.upsert_story(original)

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url):
                raise AssertionError("existing story should not hydrate")

        monkeypatch.setattr(seed_hn_from_bq.httpx, "AsyncClient", MockClient)
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated,
        ) = await seed_hn_from_bq.seed_rows(
            [
                {
                    "id": 321,
                    "title": "Replacement",
                    "url": "https://new.example",
                    "text": "Replacement text",
                    "score": 200,
                    "descendants": 10,
                    "created_at_i": 1760000000,
                }
            ],
            db=db,
            embedder=DummyEmbedder(),
        )

        assert inserted == 0
        assert skipped_feedback == 0
        assert skipped_existing == 1
        assert hydrated == 0
        assert db.get_story(321) == original
    finally:
        db.close()


@pytest.mark.asyncio
async def test_comment_hydration_success_updates_embedding(monkeypatch):
    db = Database(":memory:")
    try:
        item = {
            "type": "story",
            "num_comments": 2,
            "children": [
                {
                    "type": "comment",
                    "text": "Substantive comment with enough words to pass filtering.",
                    "children": [],
                }
            ],
        }

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url):
                return httpx.Response(200, json=item)

        monkeypatch.setattr(seed_hn_from_bq.httpx, "AsyncClient", MockClient)
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated,
        ) = await seed_hn_from_bq.seed_rows(
            [
                {
                    "id": 456,
                    "title": "Hydrate me",
                    "url": "https://example.com/h",
                    "text": "",
                    "score": 120,
                    "descendants": 1,
                    "created_at_i": 1760000000,
                }
            ],
            db=db,
            embedder=DummyEmbedder(),
        )

        assert (inserted, skipped_feedback, skipped_existing, hydrated) == (1, 0, 0, 1)
        story = db.get_story(456)
        assert story.top_comments
        assert story.comment_count == 2
        assert story.comment_count_at_fetch == 2
        assert "Substantive comment" in story.text_content

        model_version = "all-MiniLM-L6-v2|mean|norm|256"
        text_hash = hashlib.sha256(
            story_embedding_text(story).encode("utf-8")
        ).hexdigest()
        assert db.get_embedding(456, model_version, text_hash) is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_comment_hydration_failure_preserves_bq_skeleton(monkeypatch):
    db = Database(":memory:")
    try:

        class MockClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url):
                return httpx.Response(503)

        monkeypatch.setattr(seed_hn_from_bq.httpx, "AsyncClient", MockClient)
        await seed_hn_from_bq.seed_rows(
            [
                {
                    "id": 789,
                    "title": "Skeleton",
                    "url": "https://example.com/s",
                    "text": "BQ self text",
                    "score": 130,
                    "descendants": 7,
                    "created_at_i": 1760000000,
                }
            ],
            db=db,
            embedder=DummyEmbedder(),
        )

        story = db.get_story(789)
        assert story.title == "Skeleton"
        assert story.self_text == "BQ self text"
        assert story.top_comments == ""
        assert story.comment_count == 7
        assert story.text_content
    finally:
        db.close()


def test_run_bq_query_passes_max_rows_to_cli(monkeypatch):
    captured = {}

    class FakeProc:
        stdout = "[]"
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    seed_hn_from_bq.run_bq_query(limit=1000, months=3, min_score=500)
    assert "--max_rows=1000" in captured["cmd"]


def test_run_bq_query_uses_default_max_rows_when_no_limit(monkeypatch):
    captured = {}

    class FakeProc:
        stdout = "[]"
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    seed_hn_from_bq.run_bq_query(months=3, min_score=500)
    assert "--max_rows=1000" in captured["cmd"]
