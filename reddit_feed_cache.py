"""Shared in-memory Reddit topfeed RSS response cache.

Both `pipeline.py:fetch_rss_feeds` and any future Reddit feed fetcher use
the module-level `cache` singleton to avoid re-fetching subreddit topfeeds
within the TTL window. The cache is in-memory only — lost on server restart,
which is fine since the 2h TTL means minimal warm-up overhead.

Cache hit skips the rate limiter entirely (no HTTP request).
Cache miss triggers the normal fetch + limiter flow, then stores the result.
"""

from __future__ import annotations

import logging
import time

from database import Story

logger = logging.getLogger(__name__)

type CacheStats = dict[str, int | float]


class RedditFeedCache:
    """In-memory cache for parsed Reddit topfeed RSS responses."""

    def __init__(self) -> None:
        self.TTL_SECONDS: float = 7200.0
        self.MAX_ENTRIES: int = 100
        self.reset()

    def reset(self) -> None:
        self._cache: dict[str, tuple[float, list[Story]]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, feed_url: str) -> list[Story] | None:
        now = time.time()
        entry = self._cache.get(feed_url)
        if entry is None:
            self._misses += 1
            logger.debug("reddit_feed_cache miss feed=%s", feed_url)
            return None
        cached_at, stories = entry
        if now - cached_at > self.TTL_SECONDS:
            del self._cache[feed_url]
            self._misses += 1
            logger.debug(
                "reddit_feed_cache expired feed=%s age=%.0fs", feed_url, now - cached_at
            )
            return None
        self._hits += 1
        logger.debug(
            "reddit_feed_cache hit feed=%s age=%.0fs", feed_url, now - cached_at
        )
        return list(stories)

    def set(self, feed_url: str, stories: list[Story]) -> None:
        self._cache[feed_url] = (time.time(), list(stories))
        if len(self._cache) > self.MAX_ENTRIES:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
            logger.debug("reddit_feed_cache evicted feed=%s", oldest_key)

    def stats(self) -> CacheStats:
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "max_entries": self.MAX_ENTRIES,
            "ttl_seconds": self.TTL_SECONDS,
        }


cache: RedditFeedCache = RedditFeedCache()
