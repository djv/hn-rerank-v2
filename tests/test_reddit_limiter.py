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
    limiter.JITTER_SECONDS = 0.0  # deterministic
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
    limiter.JITTER_SECONDS = 0.0  # deterministic for this test
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


def test_jitter_stays_within_bounds(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """Jitter keeps on_success delays in [INTER-JITTER, INTER+JITTER]."""
    limiter.JITTER_SECONDS = 0.5
    for _ in range(1000):
        limiter._next_allowed_at = 0.0
        t_before = fake_clock.now
        limiter.on_success()
        delay = limiter._next_allowed_at - t_before
        assert 1.5 <= delay <= 2.5, f"delay {delay} outside [1.5, 2.5]"
    # No assertion on exact distribution; just that the bounds hold


def test_jitter_zero_is_deterministic(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """With JITTER_SECONDS=0, on_success uses the exact INTER_REQUEST_DELAY."""
    limiter.JITTER_SECONDS = 0.0
    for _ in range(10):
        limiter._next_allowed_at = 0.0
        t_before = fake_clock.now
        limiter.on_success()
        assert limiter._next_allowed_at - t_before == pytest.approx(2.0)


def test_jitter_does_not_affect_429_backoff(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """Jitter applies only to on_success; 429 backoff is exactly the table value."""
    limiter.JITTER_SECONDS = 0.5
    t_before = fake_clock.now
    limiter.on_429(retry_after=None)
    delay = limiter._next_allowed_at - t_before
    # First 429 → BACKOFF[0] = 2.0, no jitter
    assert delay == pytest.approx(2.0)

    t_before = fake_clock.now
    limiter.on_429(retry_after=None)
    delay = limiter._next_allowed_at - t_before
    # Second 429 → BACKOFF[1] = 4.0, no jitter
    assert delay == pytest.approx(4.0)


def test_reset_clears_half_open_state(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """reset() must zero _circuit_opened_at and clear the _probing flag."""
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    assert limiter.circuit_open is True
    assert limiter._circuit_opened_at == pytest.approx(fake_clock.now)
    fake_clock.advance(1000.0)
    limiter._probing = True
    limiter.reset()
    assert limiter._circuit_opened_at == 0.0
    assert limiter._probing is False
    assert limiter.circuit_open is False


@pytest.mark.asyncio
async def test_circuit_half_opens_after_cooldown(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """After CIRCUIT_COOLDOWN elapses, acquire admits one probe request."""
    limiter.CIRCUIT_COOLDOWN = 300.0
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    assert limiter.circuit_open is True
    assert await limiter.acquire() is False

    fake_clock.advance(299.0)
    assert await limiter.acquire() is False

    fake_clock.advance(2.0)
    assert await limiter.acquire() is True
    assert limiter._probing is True
    assert limiter.circuit_open is True

    # Probe already in flight; subsequent callers during the probe are rejected
    assert await limiter.acquire() is False


@pytest.mark.asyncio
async def test_probe_success_closes_circuit(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """A successful on_success() during half-open probe closes the circuit."""
    limiter.CIRCUIT_COOLDOWN = 300.0
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    fake_clock.advance(300.0)
    assert await limiter.acquire() is True

    limiter.on_success()
    assert limiter.circuit_open is False
    assert limiter._circuit_opened_at == 0.0
    assert limiter._probing is False

    # Normal operation resumes: 2s spacing
    assert await limiter.acquire() is True
    fake_clock.advance(2.0)
    assert await limiter.acquire() is True


@pytest.mark.asyncio
async def test_probe_failure_resets_cooldown(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """A 429 during half-open probe re-opens the circuit and resets the cooldown."""
    limiter.CIRCUIT_COOLDOWN = 300.0
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    opened_at = fake_clock.now
    fake_clock.advance(300.0)
    assert await limiter.acquire() is True

    limiter.on_429()
    assert limiter.circuit_open is True
    assert limiter._circuit_opened_at == pytest.approx(fake_clock.now)
    assert limiter._circuit_opened_at > opened_at
    assert limiter._probing is False

    # Must wait another full cooldown for the next probe
    fake_clock.advance(299.0)
    assert await limiter.acquire() is False
    fake_clock.advance(2.0)
    assert await limiter.acquire() is True


def test_on_429_records_open_time_only_on_transition(
    limiter: RedditRateLimiter, fake_clock: FakeClock
) -> None:
    """_circuit_opened_at is set once when the circuit transitions closed -> open,
    not on every subsequent 429 while already open."""
    limiter.on_429()
    limiter.on_429()
    assert limiter.circuit_open is False
    assert limiter._circuit_opened_at == 0.0

    limiter.on_429()
    assert limiter.circuit_open is True
    t_open = fake_clock.now
    assert limiter._circuit_opened_at == pytest.approx(t_open)

    fake_clock.advance(10.0)
    limiter.on_429()
    assert limiter._circuit_opened_at == pytest.approx(t_open)
