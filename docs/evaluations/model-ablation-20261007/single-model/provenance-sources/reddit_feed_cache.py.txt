"""Shared in-memory Reddit topfeed RSS response cache.

Both `pipeline.py:fetch_rss_feeds` and any future Reddit feed fetcher use
the module-level `cache` singleton to avoid re-fetching subreddit topfeeds
within the TTL window. The cache is in-memory only — lost on server restart,
which is fine since the 4h TTL means minimal warm-up overhead.

Cache hit skips the rate limiter entirely (no HTTP request).
Cache miss triggers the normal fetch + limiter flow, then stores the result.
With the 304 conditional-GET path, a cache hit sends If-None-Match /
If-Modified-Since and a 304 response reuses the cached body without
consuming rate-limit budget (per Reddit's published policy).
"""

from __future__ import annotations

import logging
import threading
import time

from cachetools import TTLCache

from database import Story

logger = logging.getLogger(__name__)

type CacheStats = dict[str, int | float]


class RedditFeedCache:
    """In-memory cache for parsed Reddit topfeed RSS responses."""

    def __init__(self) -> None:
        self.TTL_SECONDS: float = 14400.0
        self.MAX_ENTRIES: int = 100
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._cache: TTLCache[str, list[Story]] = TTLCache(
                maxsize=self.MAX_ENTRIES,
                ttl=self.TTL_SECONDS,
                timer=lambda: time.time(),
            )
            self._hits: int = 0
            self._misses: int = 0

    def get(self, feed_url: str) -> list[Story] | None:
        with self._lock:
            stories = self._cache.get(feed_url)
            if stories is None:
                self._misses += 1
                logger.debug("reddit_feed_cache miss feed=%s", feed_url)
                return None
            self._hits += 1
            logger.debug("reddit_feed_cache hit feed=%s", feed_url)
            return list(stories)

    def set(self, feed_url: str, stories: list[Story]) -> None:
        with self._lock:
            self._cache[feed_url] = list(stories)

    def stats(self) -> CacheStats:
        with self._lock:
            self._cache.expire()
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "max_entries": self.MAX_ENTRIES,
                "ttl_seconds": self.TTL_SECONDS,
            }


cache: RedditFeedCache = RedditFeedCache()
