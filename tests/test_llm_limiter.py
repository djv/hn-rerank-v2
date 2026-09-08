"""Tests for llm_limiter.py shared LLM 429 cooldown."""

from __future__ import annotations

from typing import List

import pytest

import llm_limiter
from llm_limiter import LlmRateLimiter


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

    async def fake_sleep(delay: float) -> None:
        sleep_recorder.calls.append(delay)
        fake_clock.advance(delay)

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

    assert sleep_recorder.calls == [2.0]


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


def test_record_success_resets_consecutive_429(limiter: LlmRateLimiter) -> None:
    limiter.record_response(status=429, headers={})
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


@pytest.mark.asyncio
async def test_groq_reservations_prevent_double_spending(
    limiter: LlmRateLimiter, sleep_recorder: SleepRecorder
) -> None:
    limiter.record_response(
        status=200,
        headers={
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "4000",
        },
    )
    assert await limiter.acquire(estimated_tokens=3000)
    assert await limiter.acquire(estimated_tokens=3000)
    assert sleep_recorder.calls == [15.0]


@pytest.mark.asyncio
async def test_long_provider_retry_after_fails_fast(
    limiter: LlmRateLimiter, sleep_recorder: SleepRecorder
) -> None:
    limiter.record_response(status=429, headers={"retry-after": "3600"})
    assert not await limiter.acquire()
    assert sleep_recorder.calls == []


@pytest.mark.asyncio
async def test_provider_retry_after_is_capped(limiter: LlmRateLimiter) -> None:
    """A huge provider directive (Groq sent 660s) must not dead-end the UI;
    the consecutive-429 backoff re-extends if rejections continue."""
    limiter.record_response(status=429, headers={"retry-after": "660"})
    assert limiter.retry_after_seconds <= int(LlmRateLimiter.MAX_PROVIDER_RETRY_AFTER_S)
    assert limiter.retry_after_seconds > 0


@pytest.mark.asyncio
async def test_retry_after_overrides_short_backoff(
    limiter: LlmRateLimiter, sleep_recorder: SleepRecorder
) -> None:
    limiter.record_response(status=429, headers={"retry-after": "12.5"})
    assert await limiter.acquire()
    assert sleep_recorder.calls == [12.5]


@pytest.mark.asyncio
async def test_late_response_cannot_restore_reserved_tokens(
    limiter: LlmRateLimiter, sleep_recorder: SleepRecorder
) -> None:
    headers = {
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "4000",
    }
    limiter.record_response(status=200, headers=headers)
    assert await limiter.acquire(estimated_tokens=3000)
    limiter.record_response(status=200, headers=headers)
    assert await limiter.acquire(estimated_tokens=3000)
    assert sleep_recorder.calls == [15.0]


@pytest.mark.parametrize("value", ["garbage", "nan", "inf", "-1"])
@pytest.mark.asyncio
async def test_invalid_quota_headers_do_not_poison_limiter(
    limiter: LlmRateLimiter, value: str
) -> None:
    limiter.record_response(
        status=200,
        headers={
            "x-ratelimit-limit-tokens": value,
            "x-ratelimit-remaining-tokens": value,
        },
    )
    assert await limiter.acquire(estimated_tokens=100)


@pytest.mark.asyncio
async def test_unused_completion_budget_is_returned(
    limiter: LlmRateLimiter,
    sleep_recorder: SleepRecorder,
) -> None:
    limiter.record_response(
        status=200,
        headers={
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "6000",
        },
    )
    assert await limiter.acquire(estimated_tokens=5000)
    limiter.record_response(
        status=200,
        headers={
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "5000",
        },
        reserved_tokens=5000,
        used_tokens=1000,
    )
    assert await limiter.acquire(estimated_tokens=5000)
    assert sleep_recorder.calls == []


@pytest.mark.asyncio
async def test_large_estimate_can_use_full_provider_bucket(
    limiter: LlmRateLimiter,
) -> None:
    limiter.record_response(
        status=200,
        headers={
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "8000",
        },
    )
    assert await limiter.acquire(estimated_tokens=9000)
    assert limiter._tokens == 0
