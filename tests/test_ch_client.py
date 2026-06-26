"""Tests for ch_client.py — uses mocks for CH HTTP responses (no live calls)."""

from __future__ import annotations

import time

import httpx
import pytest

import ch_client
from ch_client import (
    clear_cache,
    query_comments_bulk,
    query_live_window,
    query_single_story,
    query_stories_bulk,
    query_stories_with_comments,
)


class _MockResponse:
    def __init__(self, status_code: int, json_data: dict | list | None = None) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("POST", ch_client.CH_PLAYGROUND_URL),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict | list:
        return self._json_data


@pytest.fixture(autouse=True)
def _clear_cache_each_test() -> None:
    clear_cache()


# ---------- query_live_window ----------


def test_query_live_window_validates_args() -> None:
    with pytest.raises(ValueError, match="days"):
        query_live_window(days=0)
    with pytest.raises(ValueError, match="min_score"):
        query_live_window(days=1, min_score=-1)
    with pytest.raises(ValueError, match="limit"):
        query_live_window(days=1, limit=0)


def test_query_live_window_returns_algolia_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["query"] = kwargs.get("data", "")
        return _MockResponse(
            200,
            {
                "data": [
                    {
                        "id": 123,
                        "title": "Hello",
                        "url": "https://example.com",
                        "text": "self post",
                        "score": 50,
                        "descendants": 10,
                        "ts": 1760000000,
                    }
                ],
                "rows": 1,
            },
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    result = query_live_window(days=7, min_score=10, limit=100)

    assert captured["url"] == ch_client.CH_PLAYGROUND_URL
    assert "INTERVAL 7 DAY" in captured["query"]
    assert "score >= 10" in captured["query"]
    assert "LIMIT 100" in captured["query"]
    assert len(result) == 1
    item = result[0]
    assert item["id"] == 123
    assert item["type"] == "story"
    assert item["title"] == "Hello"
    assert item["url"] == "https://example.com"
    assert item["points"] == 50
    assert item["num_comments"] == 10
    assert item["created_at_i"] == 1760000000
    assert item["text"] == "self post"
    assert item["children"] == []


def test_query_live_window_handles_list_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CH can return either a dict {meta, data, rows} or a bare list."""

    def fake_post(url, **kwargs):
        return _MockResponse(
            200,
            [
                {
                    "id": 1,
                    "title": "T",
                    "url": None,
                    "text": "",
                    "score": 1,
                    "descendants": 0,
                    "ts": 100,
                }
            ],
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    result = query_live_window(days=1, min_score=1, limit=10)
    assert len(result) == 1


# ---------- query_stories_bulk ----------


def test_query_stories_bulk_empty() -> None:
    assert query_stories_bulk([]) == {}


def test_query_stories_bulk_returns_stories_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url, **kwargs):
        return _MockResponse(
            200,
            {
                "data": [
                    {
                        "id": 1,
                        "type": "story",
                        "title": "S1",
                        "url": "u1",
                        "text": "",
                        "score": 100,
                        "descendants": 5,
                        "ts": 1000,
                    },
                    {
                        "id": 2,
                        "type": "comment",  # filtered out
                        "title": "",
                        "url": None,
                        "text": "c",
                        "score": 0,
                        "descendants": 0,
                        "ts": 2000,
                    },
                ]
            },
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    result = query_stories_bulk([1, 2, 99])
    assert set(result.keys()) == {1}
    assert result[1]["type"] == "story"


# ---------- query_comments_bulk ----------


def test_query_comments_bulk_empty() -> None:
    assert query_comments_bulk([]) == {}


def test_query_comments_bulk_validates_levels() -> None:
    with pytest.raises(ValueError, match="max_levels"):
        query_comments_bulk([1], max_levels=0)


def test_query_comments_bulk_groups_by_story_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url, **kwargs):
        return _MockResponse(
            200,
            {
                "data": [
                    {
                        "id": 10,
                        "type": "comment",
                        "parent": 1,
                        "story_id": 1,
                        "ts": 100,
                        "text": "a",
                        "kids": [],
                    },
                    {
                        "id": 11,
                        "type": "comment",
                        "parent": 1,
                        "story_id": 1,
                        "ts": 200,
                        "text": "b",
                        "kids": [],
                    },
                    {
                        "id": 20,
                        "type": "comment",
                        "parent": 2,
                        "story_id": 2,
                        "ts": 300,
                        "text": "c",
                        "kids": [],
                    },
                ]
            },
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    result = query_comments_bulk([1, 2, 99], max_levels=2)
    assert len(result[1]) == 2
    assert len(result[2]) == 1
    assert result[99] == []  # not in payload
    assert {c["id"] for c in result[1]} == {10, 11}


# ---------- query_stories_with_comments ----------


def test_query_stories_with_comments_combines_stories_and_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if "SELECT" in kwargs.get("data", "") and "IN (" in kwargs.get("data", ""):
            # Could be either stories query or comments; just return both shapes
            return _MockResponse(
                200,
                {
                    "data": [
                        {
                            "id": 1,
                            "type": "story",
                            "title": "S1",
                            "url": "u",
                            "text": "",
                            "score": 100,
                            "descendants": 2,
                            "ts": 1000,
                        }
                    ]
                    + [
                        {
                            "id": 10,
                            "type": "comment",
                            "parent": 1,
                            "ts": 100,
                            "text": "comment1",
                            "kids": [],
                        }
                    ]
                },
            )
        return _MockResponse(200, {"data": []})

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    result = query_stories_with_comments([1], max_levels=2)

    assert 1 in result
    assert result[1]["type"] == "story"
    # The query combines via UNION ALL into one call; result has both
    # The story dict gets a `children` list (may be empty if comment row
    # had a non-matching parent during combine)
    assert "children" in result[1]


def test_query_stories_with_comments_empty() -> None:
    assert query_stories_with_comments([]) == {}


def test_query_stories_with_comments_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        return _MockResponse(
            200,
            {
                "data": [
                    {
                        "id": 1,
                        "type": "story",
                        "title": "S",
                        "url": None,
                        "text": "",
                        "score": 10,
                        "descendants": 0,
                        "ts": 100,
                    }
                ]
            },
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    r1 = query_stories_with_comments([1])
    r2 = query_stories_with_comments([1])
    assert r1 is r2  # same dict from cache
    # Two HTTP calls: one for stories, one for comments (cache miss both)
    # Second invocation should be 0 HTTP calls (cache hit)
    assert call_count["n"] == 2


# ---------- query_single_story ----------


def test_query_single_story_validates_id() -> None:
    with pytest.raises(ValueError, match="story_id"):
        query_single_story(0)


def test_query_single_story_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url, **kwargs):
        return _MockResponse(200, {"data": []})

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    assert query_single_story(999) is None


def test_query_single_story_returns_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url, **kwargs):
        return _MockResponse(
            200,
            {
                "data": [
                    {
                        "id": 1,
                        "type": "story",
                        "title": "T",
                        "url": "u",
                        "text": "",
                        "score": 5,
                        "descendants": 0,
                        "ts": 100,
                    }
                ]
            },
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    item = query_single_story(1)
    assert item is not None
    assert item["id"] == 1
    assert "children" in item


# ---------- Cache behavior ----------


def test_cache_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        return _MockResponse(
            200,
            {
                "data": [
                    {
                        "id": 1,
                        "type": "story",
                        "title": "S",
                        "url": None,
                        "text": "",
                        "score": 10,
                        "descendants": 0,
                        "ts": 100,
                    }
                ]
            },
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    # First call: cache miss
    query_stories_with_comments([1])
    assert call_count["n"] == 2  # stories + comments

    # Manually expire cache by adjusting the timestamp
    for key in ch_client._cache:
        ts, val = ch_client._cache[key]
        ch_client._cache[key] = (ts - 10000, val)

    # Second call: should re-fetch
    query_stories_with_comments([1])
    assert call_count["n"] == 4  # 2 more HTTP calls


def test_cache_lru_eviction() -> None:
    """Insert > MAX entries; oldest should be evicted."""
    # Pre-fill cache with timestamps in the past so they don't TTL out
    for i in range(ch_client._CACHE_MAX_ENTRIES + 5):
        ch_client._cache[("bulk", (i,), 5)] = (time.time() - i, "v")

    # Trigger a put
    ch_client._cache_put(("bulk", (9999,), 5), "new")
    # Cap should hold
    assert len(ch_client._cache) <= ch_client._CACHE_MAX_ENTRIES


def test_cache_clear() -> None:
    ch_client._cache[("foo",)] = (time.time(), "bar")
    ch_client.clear_cache()
    assert ch_client._cache == {}


def test_cache_key_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same story_ids in different order should hit the same cache entry."""
    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        return _MockResponse(
            200,
            {
                "data": [
                    {
                        "id": 1,
                        "type": "story",
                        "title": "S",
                        "url": None,
                        "text": "",
                        "score": 10,
                        "descendants": 0,
                        "ts": 100,
                    }
                ]
            },
        )

    monkeypatch.setattr(ch_client.httpx, "post", fake_post)
    query_stories_with_comments([1, 2])
    query_stories_with_comments([2, 1])
    # Same key (sorted), so should be cache hit on the second
    assert call_count["n"] == 2  # 2 calls for first (stories+comments), 0 for second
