import hashlib
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from database import Database, Story
from pipeline import Config, Embedder, story_embedding_text
from scripts import seed_hn_from_clickhouse
from scripts._seed_common import (
    seed_rows,
    story_from_bq_row,
)


class DummyEmbedder(Embedder):
    def __init__(self) -> None:
        pass

    def encode(self, texts: list[str], batch_size: int | None = None):
        arr = np.zeros((len(texts), 384), dtype=np.float32)
        if len(texts):
            arr[:, 0] = 1.0
        return arr


def test_ch_row_maps_to_hn_story_with_composed_text():
    story = story_from_bq_row(
        {
            "id": "123",
            "title": "<b>Useful story</b>",
            "url": "https://example.com/story",
            "text": "<p>Self text body.</p>",
            "score": "150",
            "descendants": "42",
            "created_at_i": "1760000000",
        },
        source="ch_seed",
    )

    assert story is not None
    assert story.id == 123
    assert story.source == "ch_seed"
    assert story.title == "Useful story"
    assert story.self_text == "Self text body."
    assert story.comment_count == 42
    assert story.text_content
    assert "Useful story" in story.text_content
    assert "Self text body" in story.text_content


def test_build_ch_query_accepts_months_min_score_and_limit():
    query = seed_hn_from_clickhouse.build_ch_query(months=9, min_score=250, limit=500)

    assert "INTERVAL 9 MONTH" in query
    assert "AND score >= 250" in query
    assert query.rstrip().endswith("LIMIT 500")


def test_build_ch_query_validates_bounds():
    with pytest.raises(ValueError, match="months"):
        seed_hn_from_clickhouse.build_ch_query(months=0)
    with pytest.raises(ValueError, match="min-score"):
        seed_hn_from_clickhouse.build_ch_query(min_score=-1)


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

        def fail_bulk(*a, **kw):
            raise AssertionError("feedback-protected story should not hydrate")

        monkeypatch.setattr(
            "scripts._seed_common._ch_query_stories_with_comments", fail_bulk
        )
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated,
        ) = await seed_rows(
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
            source="ch_seed",
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

        def fail_bulk(*a, **kw):
            raise AssertionError("existing story should not hydrate")

        monkeypatch.setattr(
            "scripts._seed_common._ch_query_stories_with_comments", fail_bulk
        )
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated,
        ) = await seed_rows(
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
            source="ch_seed",
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
        ch_item = {
            "id": 456,
            "type": "story",
            "title": "Hydrate me",
            "url": "https://example.com/h",
            "story_text": "",
            "text": "",
            "num_comments": 2,
            "created_at_i": 1760000000,
            "points": 120,
            "children": [
                {
                    "id": 1001,
                    "type": "comment",
                    "text": "Substantive comment with enough words and length to pass the minimum comment length filtering.",
                    "children": [],
                }
            ],
        }

        def fake_bulk(story_ids, max_levels=5):
            return {456: ch_item}

        monkeypatch.setattr(
            "scripts._seed_common._ch_query_stories_with_comments", fake_bulk
        )
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated,
        ) = await seed_rows(
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
            source="ch_seed",
            db=db,
            embedder=DummyEmbedder(),
        )

        assert (inserted, skipped_feedback, skipped_existing, hydrated) == (1, 0, 0, 1)
        story = db.get_story(456)
        assert story is not None
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
async def test_comment_hydration_failure_preserves_ch_skeleton(monkeypatch):
    db = Database(":memory:")
    try:

        def fail_bulk(*a, **kw):
            raise RuntimeError("simulated CH outage")

        monkeypatch.setattr(
            "scripts._seed_common._ch_query_stories_with_comments", fail_bulk
        )
        await seed_rows(
            [
                {
                    "id": 789,
                    "title": "Skeleton",
                    "url": "https://example.com/s",
                    "text": "CH self text",
                    "score": 130,
                    "descendants": 7,
                    "created_at_i": 1760000000,
                }
            ],
            source="ch_seed",
            db=db,
            embedder=DummyEmbedder(),
        )

        story = db.get_story(789)
        assert story is not None
        assert story.title == "Skeleton"
        assert story.self_text == "CH self text"
        assert story.top_comments == ""
        assert story.comment_count == 7
        assert story.text_content
    finally:
        db.close()


def test_run_ch_query_uses_correct_endpoint(monkeypatch):
    captured = {}

    class MockResponse:
        def __init__(self, status_code=200, json_data=None):
            self.status_code = status_code
            self._json_data = json_data or []

        def raise_for_status(self):
            pass

        def json(self):
            return self._json_data

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["query"] = kwargs.get("content", "")
        return MockResponse(200, {"data": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    seed_hn_from_clickhouse.run_ch_query(limit=500, months=6, min_score=200)
    assert (
        captured["url"] == "https://play.clickhouse.com/?user=play&default_format=JSON"
    )
    assert "LIMIT 500" in captured["query"]
    assert "INTERVAL 6 MONTH" in captured["query"]
    assert "AND score >= 200" in captured["query"]


def test_dry_run_skips_config_and_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--dry-run writes rows to file and does not load Config or Database."""
    monkeypatch.setattr(
        seed_hn_from_clickhouse,
        "run_ch_query",
        lambda *a, **kw: [
            {
                "id": 1,
                "title": "X",
                "score": 100,
                "descendants": 5,
                "created_at_i": 1760000000,
            }
        ],
    )

    def fail_load(*a: object, **kw: object) -> None:
        raise RuntimeError("Config.load should not be called in dry-run")

    monkeypatch.setattr(Config, "load", fail_load)

    out = tmp_path / "ch_dryrun.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--dry-run", "--dry-run-output", str(out)])
    seed_hn_from_clickhouse.main()

    assert out.exists()
    lines = out.read_text().splitlines()
    meta = json.loads(lines[0])
    assert meta["_meta"]["source"] == "ch_seed"
    assert meta["_meta"]["rows"] == 1
    row = json.loads(lines[1])
    assert row["id"] == 1
