"""Strict HN hydration rejects malformed items before any write (review §1)."""

from __future__ import annotations

import time
from typing import Any, cast

import pytest

from database import Database, Story

SID = 4242


def _stored(db: Database) -> Story:
    row = Story(
        id=SID,
        title="Real title",
        url="https://example.com/real",
        score=100,
        time=int(time.time()) - 3600,
        text_content="Real title discussion",
        source="hn",
        comment_count=83,
        comment_count_at_fetch=80,
        top_comments="earlier top comments here",
        article_body="FULL BODY TEXT",
    )
    db.upsert_story(row)
    return row


class _Resp:
    def __init__(self, payload: Any) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _Client:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def get(self, url: str) -> _Resp:
        return _Resp(self._payload)


def _valid_item(now: int) -> dict[str, Any]:
    return {
        "id": SID,
        "type": "story",
        "title": "Real title",
        "url": "https://example.com/real",
        "points": 101,
        "num_comments": 120,
        "created_at_i": now,
        "story_text": "",
        "children": [
            {
                "type": "comment",
                "id": 1,
                "text": "A brand new top-level comment with more than sixty characters in it.",
                "children": [],
            }
        ],
    }


def _no_id_item(now: int) -> dict[str, Any]:
    """A shape-valid payload of unknown provenance: no id anywhere."""
    item = _valid_item(now)
    del item["id"]
    return item


STRICT_ID_CASES: dict[str, Any] = {
    # No usable identity: strict must reject even though the shape is valid.
    "missing-id": _no_id_item(1_700_000_000),
    "null-id": {**_no_id_item(1_700_000_000), "id": None},
    # An explicit null never falls through to a valid objectID.
    "null-id-with-valid-objectID": {
        **_no_id_item(1_700_000_000),
        "id": None,
        "objectID": SID,
    },
    "wrong-string-id": {**_no_id_item(1_700_000_000), "id": "not-a-number"},
    "mismatched-objectID": {**_no_id_item(1_700_000_000), "objectID": 9999},
}

VALID_ID_VARIANTS: dict[str, Any] = {
    "int-id": _valid_item(1_700_000_000),
    "string-id": {**_no_id_item(1_700_000_000), "id": str(SID)},
    "int-objectID": {**_no_id_item(1_700_000_000), "objectID": SID},
    "string-objectID": {**_no_id_item(1_700_000_000), "objectID": str(SID)},
}


MALFORMED: dict[str, Any] = {
    "bare-type-only": {"type": "story"},
    "mismatched-id": {
        "id": 9999,
        "type": "story",
        "title": "Some other story",
        "points": 5,
        "num_comments": 1,
        "created_at_i": 1_700_000_000,
        "children": [],
    },
    "empty-title": {
        "id": SID,
        "type": "story",
        "title": "   ",
        "points": 101,
        "num_comments": 120,
        "created_at_i": 1_700_000_000,
        "children": [],
    },
    "string-points": {
        "id": SID,
        "type": "story",
        "title": "Real title",
        "points": "lots",
        "num_comments": 120,
        "created_at_i": 1_700_000_000,
        "children": [],
    },
    "negative-count": {
        "id": SID,
        "type": "story",
        "title": "Real title",
        "points": 101,
        "num_comments": -3,
        "created_at_i": 1_700_000_000,
        "children": [],
    },
    "dict-children": {
        "id": SID,
        "type": "story",
        "title": "Real title",
        "points": 101,
        "num_comments": 120,
        "created_at_i": 1_700_000_000,
        "children": {"id": 1},
    },
    "non-dict-body": [1, 2, 3],
}


@pytest.mark.parametrize("payload", MALFORMED.values(), ids=list(MALFORMED))
async def test_strict_hydration_rejects_malformed_without_writing(
    payload: Any,
) -> None:
    """A malformed 200 item is a failure: strict sees None, the stored row
    keeps its title/URL/score/counts/body, and nothing is generated from it."""
    from pipeline.enrichment import fetch_story

    db = Database(":memory:")
    try:
        before = _stored(db)
        result = await fetch_story(
            cast(Any, _Client(payload)), SID, db, force=True, strict=True
        )
        assert result is None
        assert db.get_story(SID) == before
    finally:
        db.close()


@pytest.mark.parametrize("payload", MALFORMED.values(), ids=list(MALFORMED))
async def test_non_strict_hydration_keeps_cached_row_on_malformed(
    payload: Any,
) -> None:
    """Normal callers are preserved: a malformed item returns the cached row
    instead of a Story built from defaults."""
    from pipeline.enrichment import fetch_story

    db = Database(":memory:")
    try:
        before = _stored(db)
        result = await fetch_story(cast(Any, _Client(payload)), SID, db, force=True)
        assert result == before
        assert db.get_story(SID) == before
    finally:
        db.close()


async def test_strict_hydration_accepts_well_formed_item() -> None:
    """The valid path still hydrates, preserves the article body and writes."""
    from pipeline.enrichment import fetch_story

    db = Database(":memory:")
    try:
        _stored(db)
        now = int(time.time()) - 3600
        result = await fetch_story(
            cast(Any, _Client(_valid_item(now))), SID, db, force=True, strict=True
        )
        assert result is not None
        assert result.title == "Real title"
        assert result.score == 101
        assert result.article_body == "FULL BODY TEXT"
        assert "FULL BODY TEXT" in result.text_content
        stored = db.get_story(SID)
        assert stored is not None and stored.score == 101
    finally:
        db.close()


@pytest.mark.parametrize("payload", STRICT_ID_CASES.values(), ids=list(STRICT_ID_CASES))
async def test_strict_hydration_requires_matching_identity(payload: Any) -> None:
    """Strict verification needs the requested item: an id-shaped payload
    with a missing, null or wrong identity is a failure even when every
    other field is valid — nothing is written and nothing is generated."""
    from pipeline.enrichment import fetch_story

    db = Database(":memory:")
    try:
        before = _stored(db)
        result = await fetch_story(
            cast(Any, _Client(payload)), SID, db, force=True, strict=True
        )
        assert result is None
        assert db.get_story(SID) == before
    finally:
        db.close()


@pytest.mark.parametrize(
    "payload", VALID_ID_VARIANTS.values(), ids=list(VALID_ID_VARIANTS)
)
async def test_strict_hydration_accepts_matching_identity_variants(
    payload: Any,
) -> None:
    """Int, numeric-string and objectID identities that match still hydrate."""
    from pipeline.enrichment import fetch_story

    db = Database(":memory:")
    try:
        _stored(db)
        result = await fetch_story(
            cast(Any, _Client(payload)), SID, db, force=True, strict=True
        )
        assert result is not None
        assert result.title == "Real title"
        assert result.score == 101
        stored = db.get_story(SID)
        assert stored is not None and stored.score == 101
    finally:
        db.close()


@pytest.mark.parametrize("payload", [_no_id_item(1_700_000_000)])
async def test_non_strict_hydration_keeps_optional_id_compatibility(
    payload: Any,
) -> None:
    """Legacy callers are preserved: without strict, a shape-valid payload
    with no id field still hydrates and writes (only strict demands
    identity). Present counts are validated, not required — Algolia may
    derive the tree count, so absence must not fail."""
    from pipeline.enrichment import fetch_story

    db = Database(":memory:")
    try:
        _stored(db)
        result = await fetch_story(cast(Any, _Client(payload)), SID, db, force=True)
        assert result is not None
        assert result.title == "Real title"
        assert result.score == 101
        stored = db.get_story(SID)
        assert stored is not None and stored.score == 101
    finally:
        db.close()
