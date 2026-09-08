"""Shared LLM provider rate-limit cooldown gate.

All LLM HTTP calls share one module-level limiter so a 429 from one
background prefetch or on-demand TLDR request slows every caller instead
of only the coroutine that happened to receive the response.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class LlmRateLimiter:
    """Thread-safe shared cooldown for LLM 429 responses."""

    BACKOFF: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 30.0)
    # Providers can demand very long waits (Groq has sent 660s after bulk
    # prefetch tripped the free tier). Honoring that verbatim dead-ends the
    # UI for the full duration, so cap the adopted delay; the consecutive-429
    # backoff re-extends it if the provider is still rejecting.
    MAX_PROVIDER_RETRY_AFTER_S: float = 120.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._next_allowed_at: float = 0.0
            self._consecutive_429: int = 0
            self._last_remaining_req_minute: str | None = None
            self._tokens: float | None = None
            self._token_limit = 0.0
            self._tokens_at = time.monotonic()

    def _refill(self, now: float) -> None:
        if self._tokens is not None:
            self._tokens = min(
                self._token_limit,
                self._tokens + max(0, now - self._tokens_at) * self._token_limit / 60,
            )
        self._tokens_at = now

    @property
    def retry_after_seconds(self) -> int:
        """Remaining provider cooldown, suitable for an HTTP Retry-After."""
        with self._lock:
            return max(0, math.ceil(self._next_allowed_at - time.monotonic()))

    async def acquire(self, *, estimated_tokens: int = 0) -> bool:
        """Reserve estimated tokens atomically; long cooldowns fail fast.

        Groq reports its minute token bucket in response headers. Reserving
        before sending prevents concurrent article/discussion and prefetch
        calls from all spending the same remaining quota. Estimates are
        conservative; provider 429s remain authoritative.
        """
        deadline = time.monotonic() + 30
        while True:
            with self._lock:
                now = time.monotonic()
                self._refill(now)
                wait = max(0.0, self._next_allowed_at - now)
                if self._tokens is not None and estimated_tokens:
                    # A character-based estimate is not a tokenizer. Let the
                    # provider decide whether a large prompt fits its limit.
                    estimated_tokens = min(estimated_tokens, int(self._token_limit))
                    wait = max(
                        wait, (estimated_tokens - self._tokens) * 60 / self._token_limit
                    )
                if wait <= 0:
                    if self._tokens is not None:
                        self._tokens -= estimated_tokens
                    return True
                if now + wait > deadline:
                    return False
            await asyncio.sleep(wait)

    def record_response(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        reserved_tokens: int = 0,
        used_tokens: int | None = None,
    ) -> None:
        remaining = headers.get("x-ratelimit-remaining-req-minute")
        if status == 429:
            self.on_429(remaining_req_minute=remaining)
        elif 200 <= status < 300:
            self.on_success(remaining_req_minute=remaining)
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens is not None and used_tokens is not None:
                self._tokens = min(
                    self._token_limit,
                    self._tokens
                    + min(reserved_tokens, self._token_limit)
                    - used_tokens,
                )
            try:
                limit = float(headers.get("x-ratelimit-limit-tokens", ""))
                tokens = float(headers.get("x-ratelimit-remaining-tokens", ""))
                if math.isfinite(limit) and limit > 0 and math.isfinite(tokens):
                    self._token_limit = limit
                    # Never replenish reservations held by other in-flight calls.
                    self._tokens = max(
                        0,
                        min(
                            limit,
                            tokens,
                            self._tokens if self._tokens is not None else limit,
                        ),
                    )
            except ValueError:
                pass
            if status == 429:
                try:
                    delay = float(headers.get("retry-after", ""))
                    if math.isfinite(delay) and delay >= 0:
                        if delay > self.MAX_PROVIDER_RETRY_AFTER_S:
                            logger.warning(
                                "llm_limiter capping provider retry-after %.0fs to %.0fs",
                                delay,
                                self.MAX_PROVIDER_RETRY_AFTER_S,
                            )
                            delay = self.MAX_PROVIDER_RETRY_AFTER_S
                        self._next_allowed_at = max(self._next_allowed_at, now + delay)
                except ValueError:
                    pass

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
