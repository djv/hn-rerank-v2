"""Shared Reddit RSS rate-limit gate singleton.

Both `pipeline.py:fetch_rss_feeds` (subreddit topfeeds) and
`server.py:_fetch_reddit_rss_context` (per-story comments RSS) call the
module-level `limiter` before each Reddit request. The limiter enforces:

  - 2s spacing between requests (Reddit's ~1 req/2s unauth IP limit)
  - Exponential backoff on 429 (2, 4, 8, 16, 32, 60s, capped at 60s)
  - Honors `Retry-After` header when present
  - Circuit opens after `MAX_CONSECUTIVE_429` (default 3) — remaining
    Reddit feeds this cycle are skipped; next cycle the limiter resets

State persists across regen cycles (cumulative backoff is the point);
server restart re-initializes the singleton to a fresh state.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RedditRateLimiter:
    """Single-bucket rate limiter for all reddit.com requests from one IP."""

    def __init__(self) -> None:
        self.INTER_REQUEST_DELAY: float = 2.0
        self.MAX_CONSECUTIVE_429: int = 3
        self.BACKOFF: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
        self.reset()

    def reset(self) -> None:
        self._next_allowed_at: float = 0.0
        self._consecutive_429: int = 0

    @property
    def circuit_open(self) -> bool:
        return self._consecutive_429 >= self.MAX_CONSECUTIVE_429

    async def acquire(self) -> bool:
        if self.circuit_open:
            return False
        now = time.monotonic()
        wait = self._next_allowed_at - now
        if wait > 0:
            await asyncio.sleep(wait)
        return True

    def on_429(self, retry_after: float | None = None) -> None:
        self._consecutive_429 += 1
        if retry_after is not None and retry_after > 0:
            delay = retry_after
        else:
            idx = min(self._consecutive_429 - 1, len(self.BACKOFF) - 1)
            delay = self.BACKOFF[idx]
        self._next_allowed_at = time.monotonic() + delay
        logger.warning(
            "reddit_limiter 429 consecutive=%d next_delay=%.1fs",
            self._consecutive_429,
            delay,
        )

    def on_success(self) -> None:
        self._consecutive_429 = 0
        self._next_allowed_at = time.monotonic() + self.INTER_REQUEST_DELAY


limiter: RedditRateLimiter = RedditRateLimiter()
