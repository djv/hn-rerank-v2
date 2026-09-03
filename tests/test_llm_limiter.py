"""Tests for llm_limiter.py shared LLM 429 cooldown."""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import llm_limiter
from llm_limiter import LlmRateLimiter, Priority


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: List[float] = []


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleep_recorder() -> SleepRecorder:
    return SleepRecorder()


@pytest.fixture
def limiter(
    monkeypatch: pytest.MonkeyPatch,
    fake_clock: FakeClock,
    sleep_recorder: SleepRecorder,
) -> LlmRateLimiter:
    limiter = LlmRateLimiter()
    limiter.reset()
    monkeypatch.setattr(llm_limiter.time, "monotonic", fake_clock.monotonic)
    # Neutralize backoff jitter (added 2026-09-03 to avoid waiters waking in
    # lockstep) so the exact-value assertions below stay deterministic;
    # jitter itself is covered by test_on_429_backoff_adds_jitter.
    monkeypatch.setattr(llm_limiter.random, "uniform", lambda a, b: 0.0)

    async def fake_sleep(delay: float) -> None:
        sleep_recorder.calls.append(delay)

    monkeypatch.setattr(llm_limiter.asyncio, "sleep", fake_sleep)
    return limiter


@pytest.mark.asyncio
async def test_acquire_returns_true_when_no_backoff(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    assert await limiter.acquire() is True
    assert sleep_recorder.calls == []


@pytest.mark.asyncio
async def test_acquire_blocks_after_429(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.record_response(status=429, headers={})

    assert await limiter.acquire() is True

    assert sleep_recorder.calls == [2.0]


@pytest.mark.asyncio
async def test_acquire_unblocks_after_backoff_expires(
    limiter: LlmRateLimiter,
    fake_clock: FakeClock,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.record_response(status=429, headers={})
    fake_clock.advance(2.0)

    assert await limiter.acquire() is True

    assert sleep_recorder.calls == []


@pytest.mark.asyncio
async def test_concurrent_acquires_all_blocked_by_429(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.record_response(status=429, headers={})

    await llm_limiter.asyncio.gather(limiter.acquire(), limiter.acquire())

    assert sleep_recorder.calls == [2.0, 2.0]


def test_backoff_table_progression(
    limiter: LlmRateLimiter,
    fake_clock: FakeClock,
) -> None:
    for expected_delay in [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]:
        limiter.record_response(status=429, headers={})
        assert limiter._next_allowed_at == pytest.approx(
            fake_clock.now + expected_delay
        )
        fake_clock.advance(expected_delay)


def test_record_success_resets_consecutive_429(
    limiter: LlmRateLimiter, fake_clock: FakeClock
) -> None:
    limiter.record_response(status=429, headers={})
    fake_clock.advance(2.0)  # let the first cooldown expire: a new incident
    limiter.record_response(status=429, headers={})
    assert limiter._consecutive_429 == 2

    limiter.record_response(status=200, headers={})

    assert limiter._consecutive_429 == 0


def test_backoff_cannot_pull_forward(
    limiter: LlmRateLimiter,
    fake_clock: FakeClock,
) -> None:
    limiter._next_allowed_at = fake_clock.now + 20.0

    limiter.record_response(status=429, headers={})

    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 20.0)


def test_record_response_tracks_remaining_header(limiter: LlmRateLimiter) -> None:
    limiter.record_response(
        status=200,
        headers={"x-ratelimit-remaining-req-minute": "42"},
    )

    assert limiter._last_remaining_req_minute == "42"


# --- min_spacing_seconds (2026-09-03: article/discussion self-inflicted 429s) ---


@pytest.mark.asyncio
async def test_sequential_acquires_are_spaced_apart(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.min_spacing_seconds = 5.0

    await limiter.acquire()  # first call: no reservation yet, wait=0, no sleep
    await limiter.acquire()  # second call: must wait out the reserved slot

    assert sleep_recorder.calls == [5.0]


@pytest.mark.asyncio
async def test_concurrent_acquires_are_spaced_apart(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.min_spacing_seconds = 3.0

    await llm_limiter.asyncio.gather(
        limiter.acquire(), limiter.acquire(), limiter.acquire()
    )

    assert sleep_recorder.calls == [3.0, 6.0]


@pytest.mark.asyncio
async def test_zero_spacing_preserves_prior_no_wait_behavior(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    assert limiter.min_spacing_seconds == 0.0

    await llm_limiter.asyncio.gather(limiter.acquire(), limiter.acquire())

    assert sleep_recorder.calls == []


@given(
    n=st.integers(min_value=1, max_value=8),
    spacing=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
    ).filter(lambda s: s == 0.0 or s >= 1e-6),
)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_acquire_reserves_monotonically_spaced_slots(
    n: int,
    spacing: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N concurrent acquire() calls reserve slots >= spacing apart.

    Uses a frozen fake clock and a no-op mocked asyncio.sleep so the
    reserved-slot arithmetic is checked directly, without any real waiting.
    """
    fake_clock = FakeClock()
    monkeypatch.setattr(llm_limiter.time, "monotonic", fake_clock.monotonic)
    recorded: List[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(llm_limiter.asyncio, "sleep", fake_sleep)

    limiter = LlmRateLimiter(min_spacing_seconds=spacing)

    async def run_all() -> None:
        await asyncio.gather(*(limiter.acquire() for _ in range(n)))

    asyncio.run(run_all())

    if spacing <= 0:
        assert recorded == []
    else:
        expected = [i * spacing for i in range(1, n)]
        assert recorded == pytest.approx(expected)


# --- concurrent-burst 429 accounting fix (2026-09-03 follow-up) ---


def test_burst_429_during_active_cooldown_does_not_reescalate(
    limiter: LlmRateLimiter,
    fake_clock: FakeClock,
) -> None:
    """A wave of near-simultaneous 429s (e.g. 10 concurrent callers all
    rejected together) must count as one incident, not escalate the
    backoff to its ceiling in one shot -- this is exactly what turned a
    single rate-limit hit into a 30s stall in the reported outage."""
    limiter.record_response(status=429, headers={})
    assert limiter._consecutive_429 == 1
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 2.0)

    # Nine more 429s arrive in the same instant, before the cooldown set
    # above has expired.
    for _ in range(9):
        limiter.record_response(status=429, headers={})

    assert limiter._consecutive_429 == 1
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 2.0)


def test_429_after_cooldown_expires_does_escalate(
    limiter: LlmRateLimiter,
    fake_clock: FakeClock,
) -> None:
    limiter.record_response(status=429, headers={})
    fake_clock.advance(2.0)
    limiter.record_response(status=429, headers={})

    assert limiter._consecutive_429 == 2
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 4.0)


def test_on_429_backoff_adds_jitter(
    monkeypatch: pytest.MonkeyPatch,
    fake_clock: FakeClock,
) -> None:
    monkeypatch.setattr(llm_limiter.time, "monotonic", fake_clock.monotonic)
    monkeypatch.setattr(llm_limiter.random, "uniform", lambda a, b: b)

    limiter = LlmRateLimiter()
    limiter.record_response(status=429, headers={})

    # base delay 2.0 + max jitter (20% of 2.0)
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 2.4)


# --- requests_per_minute token bucket ---


def test_token_bucket_allows_burst_up_to_capacity(
    fake_clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_limiter.time, "monotonic", fake_clock.monotonic)
    recorded: List[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(llm_limiter.asyncio, "sleep", fake_sleep)

    limiter = LlmRateLimiter(requests_per_minute=3)

    async def run_all() -> None:
        await asyncio.gather(*(limiter.acquire() for _ in range(3)))

    asyncio.run(run_all())

    assert recorded == []  # full capacity available, no waiting needed


def test_token_bucket_waits_once_capacity_exhausted(
    fake_clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_limiter.time, "monotonic", fake_clock.monotonic)
    recorded: List[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(llm_limiter.asyncio, "sleep", fake_sleep)

    limiter = LlmRateLimiter(requests_per_minute=60)  # 1 token/sec refill

    async def run_all() -> None:
        await asyncio.gather(*(limiter.acquire() for _ in range(61)))

    asyncio.run(run_all())

    # 60 acquires drain the full bucket instantly; the 61st needs ~1s more.
    assert recorded == pytest.approx([1.0])


def test_background_priority_yields_reserved_budget_to_foreground(
    fake_clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_limiter.time, "monotonic", fake_clock.monotonic)
    recorded: List[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(llm_limiter.asyncio, "sleep", fake_sleep)

    limiter = LlmRateLimiter(
        requests_per_minute=10, background_reserve_fraction=0.5
    )

    async def run_all() -> None:
        # Drain down to the 5-token reserve floor with background traffic.
        for _ in range(5):
            await limiter.acquire(priority=Priority.BACKGROUND)
        # The 6th background acquire must wait rather than dip into the
        # reserve; a foreground acquire in its place would not.
        await limiter.acquire(priority=Priority.BACKGROUND)

    asyncio.run(run_all())

    assert recorded == pytest.approx([6.0])  # (1 - 0) token deficit / (10/60) rate


@pytest.mark.asyncio
async def test_foreground_priority_ignores_background_reserve(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.requests_per_minute = 10
    limiter.background_reserve_fraction = 0.5
    limiter._tokens = 5.0  # exactly at the reserve floor

    assert await limiter.acquire(priority=Priority.FOREGROUND) is True

    assert sleep_recorder.calls == []


def test_record_response_clamps_tokens_to_reported_remaining(
    fake_clock: FakeClock,
) -> None:
    limiter = LlmRateLimiter(requests_per_minute=60)
    limiter._tokens = 60.0

    limiter.record_response(
        status=200, headers={"x-ratelimit-remaining-req-minute": "0"}
    )

    assert limiter._tokens == 0.0


# --- max_concurrency process-wide cap ---


@pytest.mark.asyncio
async def test_max_concurrency_caps_in_flight_acquires() -> None:
    limiter = LlmRateLimiter(max_concurrency=2)
    in_flight = 0
    max_seen = 0

    async def worker() -> None:
        nonlocal in_flight, max_seen
        await limiter.acquire()
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        limiter.release()

    await asyncio.gather(*(worker() for _ in range(5)))

    assert max_seen == 2


@pytest.mark.asyncio
async def test_release_without_acquire_is_a_safe_noop() -> None:
    limiter = LlmRateLimiter(max_concurrency=1)
    limiter.release()  # must not go negative / raise
    assert await limiter.acquire() is True


# --- deadline ---


@pytest.mark.asyncio
async def test_acquire_returns_false_past_deadline(
    limiter: LlmRateLimiter,
    fake_clock: FakeClock,
) -> None:
    limiter.record_response(status=429, headers={})  # next_allowed_at = +2.0

    ok = await limiter.acquire(deadline=fake_clock.now + 1.0)

    assert ok is False


@pytest.mark.asyncio
async def test_acquire_within_deadline_still_waits(
    limiter: LlmRateLimiter,
    fake_clock: FakeClock,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.record_response(status=429, headers={})  # next_allowed_at = +2.0

    ok = await limiter.acquire(deadline=fake_clock.now + 5.0)

    assert ok is True
    assert sleep_recorder.calls == [2.0]


@pytest.mark.asyncio
async def test_default_deadline_seconds_applies_when_no_explicit_deadline(
    fake_clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_limiter.time, "monotonic", fake_clock.monotonic)

    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(llm_limiter.asyncio, "sleep", fake_sleep)

    limiter = LlmRateLimiter(default_deadline_seconds=1.0)
    limiter.record_response(status=429, headers={})  # next_allowed_at = +2.0

    assert await limiter.acquire() is False


# --- property: rate and concurrency invariants hold for any schedule ---


@given(
    n=st.integers(min_value=1, max_value=20),
    rpm=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=20, deadline=None)
def test_token_bucket_never_exceeds_requests_per_minute_in_any_window(
    n: int, rpm: int
) -> None:
    """For any number of immediate concurrent callers, the number granted
    without waiting never exceeds the configured per-minute capacity."""

    async def run() -> list[bool]:
        limiter = LlmRateLimiter(requests_per_minute=rpm)
        granted_immediately = []
        for _ in range(n):
            with limiter._lock:
                now = time.monotonic()
                limiter._refill_locked(now)
                would_wait = limiter._tokens < 1.0
                limiter._tokens = max(0.0, limiter._tokens - 1.0)
            granted_immediately.append(not would_wait)
        return granted_immediately

    granted = asyncio.run(run())
    assert sum(granted) <= rpm


@given(n=st.integers(min_value=1, max_value=10))
@settings(max_examples=15, deadline=None)
def test_max_concurrency_never_exceeded_under_load(n: int) -> None:
    async def run() -> int:
        limiter = LlmRateLimiter(max_concurrency=3)
        in_flight = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal in_flight, max_seen
            await limiter.acquire()
            async with lock:
                in_flight += 1
                max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0)
            async with lock:
                in_flight -= 1
            limiter.release()

        await asyncio.gather(*(worker() for _ in range(n)))
        return max_seen

    max_seen = asyncio.run(run())
    assert max_seen <= 3
