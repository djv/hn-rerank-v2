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
from collections.abc import Callable, Mapping
from typing import Literal

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

    async def acquire(
        self, *, estimated_tokens: int = 0, wait_allowed: bool = True
    ) -> bool:
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
                if not wait_allowed or now + wait > deadline:
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

# Fallback providers get their own cooldown: the primary's 429 (e.g. a spent
# Go plan) must not block the call that routes around it. The primary keeps
# the shared `limiter` above.
_fallback_limiters: dict[str, LlmRateLimiter] = {}
_fallback_lock = threading.Lock()


def fallback_limiter(provider: str) -> LlmRateLimiter:
    with _fallback_lock:
        return _fallback_limiters.setdefault(provider, LlmRateLimiter())


def reset_fallback_limiters() -> None:
    with _fallback_lock:
        _fallback_limiters.clear()


class PrimaryCircuitBreaker:
    """One recovery probe across threads/loops; stale in-flight replies are ignored."""

    BACKOFF = (120.0, 300.0, 900.0, 3600.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self.reset()

    def reset(self, clock: Callable[[], float] = time.monotonic) -> None:
        with self._lock:
            self._clock = clock
            self._state: Literal["closed", "open", "half-open"] = "closed"
            self._until = 0.0
            self._opens = 0
            self._probe = False
            self._generation += 1

    @property
    def retry_after_seconds(self) -> int:
        with self._lock:
            if self._state != "open":
                return 0
            return max(0, math.ceil(self._until - self._clock()))

    def acquire(self) -> int | None:
        with self._lock:
            if self._state == "open":
                if self._clock() < self._until:
                    return None
                self._state = "half-open"
                logger.info("llm primary breaker half-open probe")
            if self._state == "half-open":
                if self._probe:
                    return None
                self._probe = True
            return self._generation

    def complete(
        self,
        generation: int,
        *,
        success: bool,
        health_failure: str = "",
        retry_after_s: float = 0.0,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                # A concurrent half may report a longer outage after the first
                # failure opened this generation. Extend it without escalating.
                if (
                    generation == self._generation - 1
                    and self._state == "open"
                    and health_failure
                ):
                    self._until = max(self._until, self._clock() + retry_after_s)
                return
            self._probe = False
            if success:
                if self._state != "closed":
                    logger.info("llm primary breaker closed")
                    self._generation += 1
                self._state = "closed"
                self._opens = 0
            elif health_failure:
                delay = max(
                    retry_after_s,
                    self.BACKOFF[min(self._opens, len(self.BACKOFF) - 1)],
                )
                self._opens += 1
                self._until = self._clock() + delay
                self._state = "open"
                self._generation += 1
                logger.warning(
                    "llm primary breaker opened reason=%s duration=%.0fs",
                    health_failure,
                    delay,
                )


primary_breaker = PrimaryCircuitBreaker()


def reset_primary_breaker(clock: Callable[[], float] = time.monotonic) -> None:
    primary_breaker.reset(clock)
