"""Shared Reddit RSS rate-limit gate singleton.

Both `pipeline.py:fetch_rss_feeds` (subreddit topfeeds) and
`server.py:_fetch_reddit_rss_context` (per-story comments RSS) call the
module-level `limiter` before each Reddit request. The limiter enforces:

  - 2s spacing between requests (Reddit's ~1 req/2s unauth IP limit),
    with uniform ±JITTER_SECONDS jitter to break robotic timing
  - Exponential backoff on 429 (2, 4, 8, 16, 32, 60s, capped at 60s)
  - Honors `Retry-After` header when present
  - Circuit opens after `MAX_CONSECUTIVE_429` (default 3) consecutive
    429s; one probe request is admitted per `CIRCUIT_COOLDOWN` (default
    300s) — if the probe succeeds the circuit closes, if it 429s the
    cooldown resets

State persists across regen cycles (cumulative backoff is the point);
server restart re-initializes the singleton to a fresh state.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time

logger = logging.getLogger(__name__)


class RedditRateLimiter:
    """Single-bucket rate limiter for all reddit.com requests from one IP.

    Thread-safe: state mutations are guarded by a lock so the queue
    worker thread (which drives scheduled fetches) and the HTTP request
    threads (which drive on-demand fetches) can both call this class
    without races. The lock is released during asyncio.sleep in
    acquire() so other threads can proceed.
    """

    def __init__(self) -> None:
        self.INTER_REQUEST_DELAY: float = 2.0
        self.JITTER_SECONDS: float = 0.5
        self.MAX_CONSECUTIVE_429: int = 3
        self.CIRCUIT_COOLDOWN: float = 300.0
        self.BACKOFF: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._next_allowed_at: float = 0.0
            self._consecutive_429: int = 0
            self._circuit_opened_at: float = 0.0
            self._probing: bool = False

    @property
    def circuit_open(self) -> bool:
        with self._lock:
            return self._consecutive_429 >= self.MAX_CONSECUTIVE_429

    async def acquire(self) -> bool:
        """Reserve a rate-limit slot for the next Reddit request.

        Inside the lock, atomically computes this caller's slot time
        (``max(now, _next_allowed_at)``) and reserves the FOLLOWING slot
        by bumping ``_next_allowed_at = slot + delay``. The next caller
        to enter the lock will see the bumped value and stagger itself
        correctly, even if it's in a different OS thread (queue worker
        vs HTTP handler).

        Previously ``_next_allowed_at`` was advanced only in
        ``on_success``/``on_429`` (after the HTTP response), so two
        concurrent ``acquire()`` callers both saw the same stale value
        and fired HTTP simultaneously. See WORKLOG 2026-06-28
        "Limiter concurrency race fix" for the full analysis.
        """
        with self._lock:
            if self._consecutive_429 >= self.MAX_CONSECUTIVE_429:
                if self._probing:
                    return False
                now = time.monotonic()
                if now - self._circuit_opened_at < self.CIRCUIT_COOLDOWN:
                    return False
                self._probing = True
                logger.info(
                    "reddit_limiter half-open probe admitted (cooldown=%.1fs elapsed)",
                    now - self._circuit_opened_at,
                )
            now = time.monotonic()
            delay = self.INTER_REQUEST_DELAY + random.uniform(
                -self.JITTER_SECONDS, self.JITTER_SECONDS
            )
            delay = max(0.0, delay)
            slot = max(now, self._next_allowed_at)
            self._next_allowed_at = slot + delay
            wait = slot - now
        if wait > 0:
            await asyncio.sleep(wait)
        return True

    def on_429(
        self,
        retry_after: float | None = None,
        *,
        rate_limit_reset: float | None = None,
    ) -> None:
        with self._lock:
            prev = self._consecutive_429
            self._consecutive_429 += 1
            if rate_limit_reset is not None and rate_limit_reset > 0:
                # Reddit's x-ratelimit-reset header gives the server's actual
                # remaining window until the rate-limit budget refills. Prefer
                # it over the BACKOFF table when present.
                delay = min(rate_limit_reset, 120.0)
            elif retry_after is not None and retry_after > 0:
                delay = retry_after
            else:
                idx = min(self._consecutive_429 - 1, len(self.BACKOFF) - 1)
                delay = self.BACKOFF[idx]
            # ``max(_next_allowed_at, now + delay)`` — never earlier than
            # what ``acquire()`` already reserved. A successful prior
            # acquire may have set the next slot to a time < now + delay;
            # the 429 backoff can only push it further out, never pull it
            # back. This protects callers who are mid-``asyncio.sleep``
            # against invalidation.
            now = time.monotonic()
            self._next_allowed_at = max(self._next_allowed_at, now + delay)
            if prev < self.MAX_CONSECUTIVE_429 <= self._consecutive_429:
                self._circuit_opened_at = now
                self._probing = False
            elif self._probing:
                self._circuit_opened_at = now
                self._probing = False
        logger.warning(
            "reddit_limiter 429 consecutive=%d next_delay=%.1fs",
            self._consecutive_429,
            delay,
        )

    def on_success(self) -> None:
        """Record a successful request and close the circuit if it was probing.

        Note: ``_next_allowed_at`` is NOT advanced here. The slot
        reservation is made in :meth:`acquire` (inside the lock) so
        concurrent callers are staggered correctly *before* the HTTP
        response is known. This method only resets the circuit state
        (``_consecutive_429``, ``_circuit_opened_at``, ``_probing``).
        """
        with self._lock:
            was_open = self._consecutive_429 >= self.MAX_CONSECUTIVE_429
            self._consecutive_429 = 0
            self._circuit_opened_at = 0.0
            self._probing = False
        if was_open:
            logger.info("reddit_limiter circuit closed after successful probe")


limiter: RedditRateLimiter = RedditRateLimiter()
