"""Tests for the scheduled Reddit fetch queue."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine

import pytest

from reddit_fetch_queue import RedditFetchQueue

CoroFactory = Callable[[], Coroutine[Any, Any, None]]


def _noop() -> CoroFactory:
    """Build a no-op coroutine factory for tests."""

    async def factory() -> None:
        pass

    return factory


def _marker(marker: list[str], label: str) -> CoroFactory:
    """Build a coroutine factory that records its execution."""

    async def factory() -> None:
        marker.append(label)

    return factory


def _sleeper(duration: float) -> CoroFactory:
    """Build a coroutine factory that sleeps for `duration` seconds."""

    async def factory() -> None:
        await asyncio.sleep(duration)

    return factory


def test_enqueue_zero_tasks_is_noop() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.enqueue_spread(0, 0.0, "topfeed", [])
    assert q.wait_until_empty(timeout=0.1) is True
    assert q.stats()["pending"] == 0


def test_enqueue_mismatched_n_raises() -> None:
    q = RedditFetchQueue()
    with pytest.raises(ValueError, match="n="):
        q.enqueue_spread(3, 0.0, "topfeed", [_noop()])
    q.reset()


def test_wait_until_empty_drains_after_completion() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.01
    q.SPREAD_WINDOW_TOPFEEDS = 0.05
    q.SPREAD_WINDOW_PREWARM = 0.05
    marker: list[str] = []
    q.enqueue_spread(
        3,
        time.monotonic(),
        "topfeed",
        [
            _marker(marker, "a"),
            _marker(marker, "b"),
            _marker(marker, "c"),
        ],
    )
    drained = q.wait_until_empty(timeout=2.0)
    assert drained is True
    assert sorted(marker) == ["a", "b", "c"]
    assert q.stats()["pending"] == 0
    assert q.stats()["completed"] == 3


def test_wait_until_empty_timeout_returns_false() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.01
    q.SPREAD_WINDOW_TOPFEEDS = 10.0  # tasks scheduled 10s apart
    marker: list[str] = []
    q.enqueue_spread(
        5, time.monotonic(), "topfeed", [_marker(marker, str(i)) for i in range(5)]
    )
    drained = q.wait_until_empty(timeout=0.2)
    assert drained is False
    assert q.stats()["pending"] > 0


def test_tasks_run_in_target_time_order() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.01
    q.SPREAD_WINDOW_TOPFEEDS = 0.1  # 5 tasks spread over 100ms
    marker: list[str] = []
    base = time.monotonic() + 0.1  # start 100ms in the future
    q.enqueue_spread(5, base, "topfeed", [_marker(marker, str(i)) for i in range(5)])
    assert q.wait_until_empty(timeout=2.0) is True
    # Tasks 0..4 were scheduled at base+0, base+20ms, ..., base+80ms
    # Order should be 0,1,2,3,4
    assert marker == ["0", "1", "2", "3", "4"]


def test_failed_task_does_not_break_queue() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.01
    q.SPREAD_WINDOW_TOPFEEDS = 0.05

    def boom() -> CoroFactory:
        async def factory() -> None:
            raise RuntimeError("kaboom")

        return factory

    marker: list[str] = []
    q.enqueue_spread(
        3,
        time.monotonic(),
        "topfeed",
        [
            _marker(marker, "a"),
            boom(),
            _marker(marker, "c"),
        ],
    )
    assert q.wait_until_empty(timeout=2.0) is True
    assert sorted(marker) == ["a", "c"]
    assert q.stats()["failed"] == 1
    assert q.stats()["completed"] == 2


def test_slow_task_blocks_subsequent_tasks() -> None:
    """The worker is single-threaded; a slow task delays the next."""
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.01
    q.SPREAD_WINDOW_TOPFEEDS = 0.05
    started: list[float] = []
    base = time.monotonic()

    def timed(label: str) -> CoroFactory:
        async def factory() -> None:
            started.append(time.monotonic() - base)

        return factory

    q.enqueue_spread(
        2,
        time.monotonic(),
        "topfeed",
        [
            _sleeper(0.1),
            timed("second"),
        ],
    )
    assert q.wait_until_empty(timeout=2.0) is True
    # First task sleeps 100ms; second should start ~100ms after enqueue
    assert started[0] >= 0.1


def test_enqueue_spread_distributes_evenly() -> None:
    """20 tasks across 1s should be spread 50ms apart."""
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.001
    q.SPREAD_WINDOW_TOPFEEDS = 1.0
    starts: list[float] = []
    base = time.monotonic()

    def timed() -> CoroFactory:
        async def factory() -> None:
            starts.append(time.monotonic() - base)

        return factory

    q.enqueue_spread(20, base, "topfeed", [timed() for _ in range(20)])
    assert q.wait_until_empty(timeout=2.0) is True
    # Stride is 1.0 / 20 = 0.05s
    # First task runs immediately, last at ~0.95s
    assert len(starts) == 20
    assert starts[-1] - starts[0] >= 0.8  # wide spread
    # Median should be ~0.5s
    assert 0.4 <= starts[9] <= 0.6


def test_reset_clears_pending_and_signals_idle() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.01
    q.SPREAD_WINDOW_TOPFEEDS = 60.0
    q.enqueue_spread(
        10, time.monotonic() + 30.0, "topfeed", [_noop() for _ in range(10)]
    )
    assert q.stats()["pending"] == 10
    assert q._idle_event.is_set() is False
    q.reset()
    assert q.stats()["pending"] == 0
    assert q._idle_event.is_set() is True


def test_shutdown_stops_worker() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.shutdown(timeout=1.0)
    assert q._stop.is_set() is True
    assert q._thread.is_alive() is False


def test_enqueue_all_reddit_fetches_interleaves() -> None:
    """enqueue_all_reddit_fetches alternates topfeed/prewarm factories on a
    single shared window. With equal-length lists, the run order is
    [t0, p0, t1, p1, ...]. The combined stride is min_stride_seconds,
    so each task is `min_stride` apart.
    """
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.001
    q.MIN_FETCH_SPACING = 0.01
    marker: list[str] = []
    topfeed = [_marker(marker, f"t{i}") for i in range(3)]
    prewarm = [_marker(marker, f"p{i}") for i in range(3)]
    q.enqueue_all_reddit_fetches(topfeed, prewarm, min_stride_seconds=0.01)
    assert q.wait_until_empty(timeout=2.0) is True
    # With min_stride=0.01s and 6 tasks, the limiter (2s+jitter
    # override below) doesn't delay; the queue runs in target order.
    assert marker == ["t0", "p0", "t1", "p1", "t2", "p2"]
    assert q.stats()["pending"] == 0
    assert q.stats()["completed"] == 6


def test_enqueue_all_reddit_fetches_uneven_lengths() -> None:
    """When the lists are uneven, the longer one fills the tail.

    2 topfeed + 4 prewarm → [t0, p0, t1, p1, p2, p3]
    """
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.001
    q.MIN_FETCH_SPACING = 0.01
    marker: list[str] = []
    topfeed = [_marker(marker, f"t{i}") for i in range(2)]
    prewarm = [_marker(marker, f"p{i}") for i in range(4)]
    q.enqueue_all_reddit_fetches(topfeed, prewarm, min_stride_seconds=0.01)
    assert q.wait_until_empty(timeout=2.0) is True
    assert marker == ["t0", "p0", "t1", "p1", "p2", "p3"]


def test_enqueue_all_reddit_fetches_empty_inputs_noop() -> None:
    q = RedditFetchQueue()
    q.reset()
    q.enqueue_all_reddit_fetches([], [])
    assert q.wait_until_empty(timeout=0.1) is True
    assert q.stats()["pending"] == 0


def test_enqueue_all_reddit_fetches_uses_class_default_when_no_min_stride() -> None:
    """Falls back to the class-level MIN_FETCH_SPACING when caller omits
    the kwarg. For this test we set the class attr to 0.01 so the
    tasks run in 60ms total.
    """
    q = RedditFetchQueue()
    q.reset()
    q.POLL_INTERVAL = 0.001
    orig = RedditFetchQueue.MIN_FETCH_SPACING
    RedditFetchQueue.MIN_FETCH_SPACING = 0.01
    try:
        q.MIN_FETCH_SPACING = 0.01
        marker: list[str] = []
        topfeed = [_marker(marker, f"t{i}") for i in range(2)]
        prewarm = [_marker(marker, f"p{i}") for i in range(2)]
        q.enqueue_all_reddit_fetches(topfeed, prewarm)
        assert q.wait_until_empty(timeout=2.0) is True
        assert marker == ["t0", "p0", "t1", "p1"]
    finally:
        RedditFetchQueue.MIN_FETCH_SPACING = orig
