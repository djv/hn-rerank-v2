"""Shared LLM provider rate-limit cooldown gate.

All LLM HTTP calls share one module-level limiter so a 429 from one
background prefetch or on-demand TLDR request slows every caller instead
of only the coroutine that happened to receive the response.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class LlmRateLimiter:
    """Thread-safe shared cooldown for LLM 429 responses."""

    BACKOFF: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 30.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._next_allowed_at: float = 0.0
            self._consecutive_429: int = 0
            self._last_remaining_req_minute: str | None = None

    async def acquire(self) -> bool:
        """Wait until the shared 429 cooldown has expired."""
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed_at - now)
        if wait > 0:
            await asyncio.sleep(wait)
        return True

    def record_response(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
    ) -> None:
        remaining = headers.get("x-ratelimit-remaining-req-minute")
        if status == 429:
            self.on_429(remaining_req_minute=remaining)
        elif 200 <= status < 300:
            self.on_success(remaining_req_minute=remaining)

    def on_429(self, *, remaining_req_minute: str | None = None) -> None:
        with self._lock:
            self._consecutive_429 += 1
            idx = min(self._consecutive_429 - 1, len(self.BACKOFF) - 1)
            delay = self.BACKOFF[idx]
            now = time.monotonic()
            self._next_allowed_at = max(self._next_allowed_at, now + delay)
            self._last_remaining_req_minute = remaining_req_minute
            consecutive = self._consecutive_429
        logger.warning(
            "llm_limiter 429 consecutive=%d next_delay=%.1fs remaining_req_minute=%s",
            consecutive,
            delay,
            remaining_req_minute,
        )

    def on_success(self, *, remaining_req_minute: str | None = None) -> None:
        with self._lock:
            self._consecutive_429 = 0
            self._last_remaining_req_minute = remaining_req_minute


limiter: LlmRateLimiter = LlmRateLimiter()
