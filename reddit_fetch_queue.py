"""Background queue for spreading Reddit fetch tasks across time.

When the regen runs, the pipeline would historically burst 41 subreddit
topfeed requests and 46 per-story prewarm requests at the limiter's
2-second cadence (87 fetches in ~3 min). Reddit's anti-abuse flags
bursts of that shape — see the 2026-06-28 incident where 37 consecutive
429s in 3 seconds knocked the limiter into a 12-hour stuck-open state.

This module replaces the burst with a scheduled queue: a single daemon
worker thread pops tasks whose `target_at` has passed, runs them via
`asyncio.run`, and signals an idle event when the heap drains. Callers
enqueue with `enqueue_spread(n, base_at, kind, factories)` to spread
N tasks evenly over a kind-specific window (10 min for topfeeds, 15 min
for prewarm by default), then `wait_until_empty(timeout=...)` to block
until the spread completes.

State machine:

  - `enqueue_spread` clears `_idle_event` and pushes N tasks with
    evenly-spaced target timestamps.
  - The worker loop pops tasks whose `target_at <= now` and runs them
    one at a time (synchronous from the caller's perspective).
  - When the heap is empty, the worker sets `_idle_event` and sleeps
    for `POLL_INTERVAL` before re-checking.
  - `wait_until_empty` returns when `_idle_event` is set (or the
    timeout elapses).

The queue does NOT reset between regen cycles — tasks from a previous
regen that haven't run yet are kept. In practice the 3h regen interval
is much longer than the 25 min combined spread window, so the queue
drains well before the next regen.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import threading
import time
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

CoroFactory = Callable[[], Coroutine[Any, Any, None]]


class _Task:
    __slots__ = ("target_at", "task_id", "factory")

    def __init__(self, target_at: float, task_id: str, factory: CoroFactory) -> None:
        self.target_at = target_at
        self.task_id = task_id
        self.factory = factory

    def __lt__(self, other: _Task) -> bool:
        return self.target_at < other.target_at


class RedditFetchQueue:
    """Threadsafe scheduled-queue of Reddit fetch coroutine factories."""

    SPREAD_WINDOW_TOPFEEDS: float = 2100.0
    SPREAD_WINDOW_PREWARM: float = 2100.0
    POLL_INTERVAL: float = 0.01
    WORKER_JOIN_TIMEOUT: float = 5.0

    def __init__(self) -> None:
        self._heap: list[_Task] = []
        self._lock = threading.Lock()
        self._idle_event = threading.Event()
        self._idle_event.set()  # initially empty
        self._stop = threading.Event()
        self._completed = 0
        self._failed = 0
        self._thread = threading.Thread(
            target=self._worker, name="reddit-fetch-queue", daemon=True
        )
        self._thread.start()

    def enqueue_spread(
        self,
        n: int,
        base_at: float,
        kind: str,
        factories: list[CoroFactory],
        *,
        window_seconds: float | None = None,
    ) -> None:
        """Schedule N tasks evenly over the kind-specific spread window.

        Args:
            n: number of tasks to schedule (must equal len(factories)).
            base_at: monotonic time (e.g. `time.monotonic()`) for the
                first task's target. Subsequent tasks are staggered
                `window / n` seconds apart.
            kind: "topfeed" or "prewarm" (used for logging + default window).
            factories: list of no-arg callables returning awaitables.
            window_seconds: override the spread window (in seconds). If
                None, falls back to the class default for the given kind.
        """
        if n == 0:
            return
        if n != len(factories):
            raise ValueError(f"n={n} != len(factories)={len(factories)}")
        if window_seconds is None:
            window_seconds = (
                self.SPREAD_WINDOW_TOPFEEDS
                if kind == "topfeed"
                else self.SPREAD_WINDOW_PREWARM
            )
        stride = window_seconds / n
        with self._lock:
            self._idle_event.clear()
            for i, factory in enumerate(factories):
                heapq.heappush(
                    self._heap,
                    _Task(base_at + i * stride, f"{kind}-{i}", factory),
                )
        logger.info(
            "reddit_fetch_queue enqueued %d %s tasks over %.0fs (stride=%.1fs)",
            n,
            kind,
            window_seconds,
            stride,
        )

    def wait_until_empty(self, timeout: float | None = None) -> bool:
        """Block until the heap drains. Returns True if drained, False on timeout."""
        return self._idle_event.wait(timeout=timeout)

    def _pop_ready(self, now: float) -> _Task | None:
        with self._lock:
            if not self._heap:
                self._idle_event.set()
                return None
            task = self._heap[0]
            if task.target_at <= now:
                heapq.heappop(self._heap)
                return task
            return None

    def _worker(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            task = self._pop_ready(now)
            if task is None:
                time.sleep(self.POLL_INTERVAL)
                continue
            try:
                asyncio.run(task.factory())
                self._completed += 1
            except Exception:
                self._failed += 1
                logger.exception("reddit_fetch_queue task %s failed", task.task_id)

    def shutdown(self, timeout: float = WORKER_JOIN_TIMEOUT) -> None:
        """Stop the worker thread. For tests and graceful shutdown only."""
        self._stop.set()
        self._thread.join(timeout=timeout)

    def reset(self) -> None:
        """Clear the heap and signal idle. For tests. Does not stop the worker."""
        with self._lock:
            self._heap.clear()
            self._idle_event.set()
            self._completed = 0
            self._failed = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "pending": len(self._heap),
                "completed": self._completed,
                "failed": self._failed,
            }


queue: RedditFetchQueue = RedditFetchQueue()
