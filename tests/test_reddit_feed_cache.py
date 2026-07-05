"""Tests for the Reddit topfeed RSS response cache."""

from __future__ import annotations

import time as time_module

from database import Story
from reddit_feed_cache import cache


def _story(id: int = 1, source: str = "rss_reddit_test") -> Story:
    return Story(
        id=id,
        title=f"Story {id}",
        url=f"http://example.com/{id}",
        score=id * 10,
        time=1_000_000,
        text_content=f"content {id}",
        source=source,
        comment_count=None,
        discussion_url=None,
    )


def test_get_returns_none_on_empty() -> None:
    assert cache.get("http://example.com/feed") is None


def test_set_then_get() -> None:
    cache.set("http://example.com/feed", [_story(1)])
    result = cache.get("http://example.com/feed")
    assert result is not None
    assert len(result) == 1
    assert result[0].id == 1


def test_returns_none_after_ttl(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time_module, "time", lambda: now[0])
    url = "http://example.com/feed"

    cache.set(url, [_story(1)])
    assert cache.get(url) is not None

    now[0] += cache.TTL_SECONDS + 1.0
    assert cache.get(url) is None


def test_set_overwrites_and_refreshes_ttl(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time_module, "time", lambda: now[0])
    url = "http://example.com/feed"

    cache.set(url, [_story(1)])
    cache.set(url, [_story(2)])

    result = cache.get(url)
    assert result is not None
    assert result[0].id == 2

    now[0] += cache.TTL_SECONDS - 1.0
    assert cache.get(url) is not None


def test_different_urls_independent() -> None:
    cache.set("http://a", [_story(1)])
    cache.set("http://b", [_story(2)])
    a = cache.get("http://a")
    b = cache.get("http://b")
    assert a is not None and b is not None
    assert a[0].id == 1
    assert b[0].id == 2


def test_reset_clears_all() -> None:
    cache.set("http://a", [_story(1)])
    cache.set("http://b", [_story(2)])
    cache.reset()
    assert cache.get("http://a") is None
    assert cache.get("http://b") is None


def test_stats_counts() -> None:
    cache.reset()
    assert cache.stats()["misses"] == 0
    assert cache.stats()["hits"] == 0

    cache.get("http://miss")
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 0

    cache.set("http://hit", [_story(1)])
    cache.get("http://hit")
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1

    s = cache.stats()
    assert s["size"] == 1
    assert s["max_entries"] == 100
    assert s["ttl_seconds"] == 14400


def test_returns_copy() -> None:
    stories = [_story(1)]
    cache.set("http://feed", stories)
    stories.append(_story(2))
    cached = cache.get("http://feed")
    assert cached is not None and len(cached) == 1


def test_eviction_at_max_entries() -> None:
    cache.reset()
    for i in range(cache.MAX_ENTRIES + 5):
        cache.set(f"http://feed_{i}", [_story(i)])
    assert cache.stats()["size"] <= cache.MAX_ENTRIES
    assert cache.get("http://feed_0") is None


def test_expired_entry_removed_on_get(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(time_module, "time", lambda: now[0])
    url = "http://feed"

    cache.set(url, [_story(1)])
    assert cache.stats()["size"] == 1

    now[0] += cache.TTL_SECONDS + 1.0
    cache.get(url)
    assert cache.stats()["size"] == 0


def test_stats_reset() -> None:
    cache.reset()
    cache.get("http://miss")
    assert cache.stats()["misses"] == 1
    cache.reset()
    assert cache.stats()["misses"] == 0
