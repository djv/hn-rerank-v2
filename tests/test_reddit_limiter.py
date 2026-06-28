"""Tests for reddit_limiter.py — shared Reddit RSS rate-limit gate.

Uses a fake monotonic clock and a recording-but-no-op sleep so the tests
are deterministic and fast.
"""

from __future__ import annotations

from typing import List

import pytest

from reddit_limiter import RedditRateLimiter


class FakeClock:
    """Fake monotonic clock for deterministic time control."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.calls: int = 0

    def monotonic(self) -> float:
        self.calls += 1
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class SleepRecorder:
    """Records asyncio.sleep calls without actually sleeping."""

    def __init__(self) -> None:
        self.calls: List[float] = []

    async def sleep(self, delay: float) -> None:
        self.calls.append(delay)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleep_recorder() -> SleepRecorder:
    return SleepRecorder()


@pytest.fixture
def limiter(
    monkeypatch, fake_clock: FakeClock, sleep_recorder: SleepRecorder
) -> RedditRateLimiter:
    rl = RedditRateLimiter()
    rl.reset()
    monkeypatch.setattr("reddit_limiter.time.monotonic", fake_clock.monotonic)
    # Patch asyncio.sleep inside the reddit_limiter module
    import reddit_limiter

    async def fake_sleep(delay: float) -> None:
        sleep_recorder.calls.append(delay)
        fake_clock.advance(delay)

    monkeypatch.setattr(reddit_limiter.asyncio, "sleep", fake_sleep)
    return rl


@pytest.mark.asyncio
async def test_acquire_no_wait_initially(
    limiter: RedditRateLimiter, sleep_recorder: SleepRecorder
) -> None:
    assert await limiter.acquire() is True
    assert sleep_recorder.calls == []


@pytest.mark.asyncio
async def test_acquire_waits_inter_request_delay_after_on_success(
    limiter: RedditRateLimiter, fake_clock: FakeClock, sleep_recorder: SleepRecorder
) -> None:
    limiter.on_success()
    assert sleep_recorder.calls == []
    # _next_allowed_at = fake_clock.now + INTER_REQUEST_DELAY = 1002.0
    assert await limiter.acquire() is True
    # Should have slept (1002.0 - 1000.0) = 2.0s
    assert sleep_recorder.calls == [2.0]


@pytest.mark.asyncio
async def test_on_429_uses_backoff_table(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    for expected_delay, idx in [
        (2.0, 1),
        (4.0, 2),
        (8.0, 3),
        (16.0, 4),
        (32.0, 5),
        (60.0, 6),
    ]:
        limiter._consecutive_429 = idx - 1
        limiter.on_429()
        # _next_allowed_at = now + backoff[idx-1]
        assert limiter._next_allowed_at == pytest.approx(
            fake_clock.now + expected_delay
        )


def test_on_429_caps_at_60s(limiter: RedditRateLimiter, fake_clock: FakeClock) -> None:
    # 7th consecutive 429 should still cap at 60s
    limiter._consecutive_429 = 6
    limiter.on_429()
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 60.0)
    # 20th consecutive 429 should still cap at 60s
    limiter._consecutive_429 = 19
    limiter.on_429()
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 60.0)


def test_on_429_honors_retry_after(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    limiter.on_429(retry_after=7.5)
    # retry_after > 0 overrides backoff
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 7.5)
    assert limiter._consecutive_429 == 1


def test_on_429_ignores_zero_retry_after(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    limiter.on_429(retry_after=0.0)
    # Falls back to BACKOFF[0] = 2.0
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 2.0)
    assert limiter._consecutive_429 == 1


def test_on_429_ignores_none_retry_after(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    limiter.on_429(retry_after=None)
    assert limiter._next_allowed_at == pytest.approx(fake_clock.now + 2.0)


@pytest.mark.asyncio
async def test_circuit_opens_after_max_consecutive_429(
    limiter: RedditRateLimiter,
) -> None:
    assert limiter.circuit_open is False
    limiter.on_429()
    limiter.on_429()
    assert limiter.circuit_open is False
    limiter.on_429()
    assert limiter.circuit_open is True
    # Acquire returns False fast (no sleep) when circuit is open
    assert await limiter.acquire() is False


def test_on_success_resets_counter(limiter: RedditRateLimiter) -> None:
    limiter.on_429()
    limiter.on_429()
    assert limiter._consecutive_429 == 2
    limiter.on_success()
    assert limiter._consecutive_429 == 0


def test_reset_clears_state(limiter: RedditRateLimiter, fake_clock: FakeClock) -> None:
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    assert limiter.circuit_open is True
    limiter.reset()
    assert limiter.circuit_open is False
    assert limiter._next_allowed_at == 0.0
    assert limiter._consecutive_429 == 0


@pytest.mark.asyncio
async def test_circuit_reopens_after_reset(limiter: RedditRateLimiter) -> None:
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    assert await limiter.acquire() is False
    limiter.reset()
    assert await limiter.acquire() is True


@pytest.mark.asyncio
async def test_acquire_with_circuit_open_skips_sleep(
    limiter: RedditRateLimiter, sleep_recorder: SleepRecorder
) -> None:
    """When circuit is open, acquire returns False without sleeping."""
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    sleep_recorder.calls.clear()
    assert await limiter.acquire() is False
    assert sleep_recorder.calls == []


@pytest.mark.asyncio
async def test_sequential_calls_enforce_2s_spacing(
    limiter: RedditRateLimiter, fake_clock: FakeClock, sleep_recorder: SleepRecorder
) -> None:
    """Three sequential acquire+on_success cycles should each wait 2s (after the first)."""
    for _ in range(3):
        assert await limiter.acquire() is True
        limiter.on_success()
    # First acquire is free (next_allowed_at=0). Each subsequent acquire
    # blocks for INTER_REQUEST_DELAY (2.0) because on_success bumped
    # _next_allowed_at to now+2.0. So 3 cycles = 2 sleeps total.
    assert sleep_recorder.calls == [2.0, 2.0]
    assert fake_clock.now == pytest.approx(1000.0 + 4.0)


def test_instance_attributes_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests can override the limiter constants via attribute assignment."""
    rl = RedditRateLimiter()
    rl.INTER_REQUEST_DELAY = 0.0
    rl.MAX_CONSECUTIVE_429 = 1
    assert rl.circuit_open is False
    rl.on_429()
    assert rl.circuit_open is True
    rl.reset()
    rl.MAX_CONSECUTIVE_429 = 3
    assert rl.circuit_open is False
