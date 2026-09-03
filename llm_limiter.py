"""Shared LLM provider rate-limit and concurrency admission gate.

All LLM HTTP calls share one module-level limiter so a 429 from one
background prefetch or on-demand TLDR request slows every caller instead
of only the coroutine that happened to receive the response.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import random
import threading
import time
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class Priority(enum.Enum):
    """Relative importance of an LLM call for admission under `llm_limiter`.

    FOREGROUND is a live user request (tldr-detail on-demand generation).
    BACKGROUND is prefetch/warm work. When `requests_per_minute` is
    configured, BACKGROUND acquires yield their share of the budget back to
    FOREGROUND once available tokens drop below
    `background_reserve_fraction` of capacity — see WORKLOG 2026-09-03
    follow-up.
    """

    FOREGROUND = "foreground"
    BACKGROUND = "background"


class LlmRateLimiter:
    """Thread-safe shared admission gate for outbound LLM calls.

    Mechanisms, all independently optional and each a no-op at its default
    (matching pre-2026-09 behavior when left unconfigured):

      - `_next_allowed_at` / `BACKOFF`: cooldown pushed forward on 429
        (original mechanism).
      - `_next_slot_at` / `min_spacing_seconds`: minimum gap enforced
        between any two acquires, so concurrent callers (e.g. the
        article/discussion pair in generate_detailed_tldr) don't land on
        the provider in the same instant (2026-09-03 fix).
      - `requests_per_minute`: a token bucket capping the sustained request
        rate to the provider's real budget. Implemented as continuous
        refill up to capacity, not fixed-interval pacing, so a quiet period
        can still absorb a small burst.
      - `max_concurrency`: a process-wide in-flight cap using a plain
        counter behind `_lock`, deliberately NOT an `asyncio.Semaphore` --
        Semaphore is bound to one event loop, and this limiter is shared
        across many (one per Flask worker thread's own `asyncio.run`),
        which is exactly why the per-loop `asyncio.Semaphore(2)` in
        `_prefetch_tldrs_for_ranked` failed to bound real concurrency when
        multiple warm cycles overlapped (see WORKLOG 2026-09-03 follow-up).

    `acquire()` computes the 429/spacing/token wait analytically under the
    lock (a single `asyncio.sleep`, like the original code), then -- only
    if `max_concurrency` is set -- polls in a short loop for a free
    concurrency slot, since that wait depends on when another in-flight
    call finishes and can't be predicted in advance. Callers that rely on
    `max_concurrency` must call `release()` when their call finishes
    (success, failure, or exception) -- `release()` is a harmless no-op if
    `max_concurrency` was never configured.
    """

    BACKOFF: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 30.0)
    BACKOFF_JITTER_FRACTION = 0.2
    _CONCURRENCY_POLL_INTERVAL = 0.05

    def __init__(
        self,
        min_spacing_seconds: float = 0.0,
        requests_per_minute: float | None = None,
        max_concurrency: int | None = None,
        background_reserve_fraction: float = 0.0,
        default_deadline_seconds: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.min_spacing_seconds = min_spacing_seconds
        self.requests_per_minute = requests_per_minute
        self.max_concurrency = max_concurrency
        self.background_reserve_fraction = background_reserve_fraction
        self.default_deadline_seconds = default_deadline_seconds
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._next_allowed_at: float = 0.0
            self._next_slot_at: float = 0.0
            self._consecutive_429: int = 0
            self._last_remaining_req_minute: str | None = None
            self._tokens: float = float(self.requests_per_minute or 0.0)
            self._last_refill: float = time.monotonic()
            self._in_flight: int = 0

    def _refill_locked(self, now: float) -> None:
        if not self.requests_per_minute:
            return
        elapsed = max(0.0, now - self._last_refill)
        rate = self.requests_per_minute / 60.0
        self._tokens = min(self.requests_per_minute, self._tokens + elapsed * rate)
        self._last_refill = now

    async def acquire(
        self,
        priority: Priority = Priority.FOREGROUND,
        deadline: float | None = None,
    ) -> bool:
        """Wait for this caller's reserved slot (cooldown + spacing + token
        budget), then, if `max_concurrency` is configured, for a free
        in-flight slot. Returns False instead of waiting past `deadline`
        (falling back to `default_deadline_seconds` when omitted and set).
        """
        eff_deadline = deadline
        if eff_deadline is None and self.default_deadline_seconds is not None:
            eff_deadline = time.monotonic() + self.default_deadline_seconds

        with self._lock:
            now = time.monotonic()
            self._refill_locked(now)
            slot = max(now, self._next_allowed_at, self._next_slot_at)
            token_wait = 0.0
            if self.requests_per_minute:
                reserve = (
                    self.requests_per_minute * self.background_reserve_fraction
                    if priority is Priority.BACKGROUND
                    else 0.0
                )
                available = self._tokens - reserve
                if available < 1.0:
                    rate = self.requests_per_minute / 60.0
                    token_wait = (1.0 - available) / rate
                self._tokens = max(0.0, self._tokens - 1.0)
            wait = max(0.0, slot - now, token_wait)
            self._next_slot_at = slot + self.min_spacing_seconds
        if wait > 0:
            if eff_deadline is not None and now + wait > eff_deadline:
                return False
            await asyncio.sleep(wait)

        if self.max_concurrency is not None:
            while True:
                with self._lock:
                    if self._in_flight < self.max_concurrency:
                        self._in_flight += 1
                        return True
                if eff_deadline is not None and time.monotonic() >= eff_deadline:
                    return False
                await asyncio.sleep(self._CONCURRENCY_POLL_INTERVAL)
        return True

    def release(self) -> None:
        """Release a concurrency slot claimed by `acquire()`. Safe to call
        even when `max_concurrency` was never configured."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

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

    def _clamp_tokens_to_header_locked(self, remaining_req_minute: str | None) -> None:
        """Trust the provider's own remaining-budget header over our local
        token count when it reports less than we think we have -- our
        bucket is an estimate, the header is ground truth."""
        if remaining_req_minute is None or not self.requests_per_minute:
            return
        try:
            remaining_n = float(remaining_req_minute)
        except ValueError:
            return
        self._tokens = min(self._tokens, remaining_n)

    def on_429(self, *, remaining_req_minute: str | None = None) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_at:
                # Already inside an active cooldown: this 429 almost
                # certainly came from a call that was already in flight
                # before the cooldown took effect (a burst of concurrent
                # callers all rejected together -- see WORKLOG 2026-09-03
                # follow-up), not new evidence that backoff needs to
                # escalate further. Re-escalating here is what turned one
                # incident into an instant jump to the 30s ceiling.
                delay = self._next_allowed_at - now
            else:
                self._consecutive_429 += 1
                idx = min(self._consecutive_429 - 1, len(self.BACKOFF) - 1)
                base_delay = self.BACKOFF[idx]
                jitter = random.uniform(0, base_delay * self.BACKOFF_JITTER_FRACTION)
                delay = base_delay + jitter
                self._next_allowed_at = now + delay
                self._next_slot_at = max(self._next_slot_at, self._next_allowed_at)
            consecutive = self._consecutive_429
            self._last_remaining_req_minute = remaining_req_minute
            self._clamp_tokens_to_header_locked(remaining_req_minute)
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
            self._clamp_tokens_to_header_locked(remaining_req_minute)


limiter: LlmRateLimiter = LlmRateLimiter()
