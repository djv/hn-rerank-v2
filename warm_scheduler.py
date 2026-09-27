"""Coalescing per-user job queue drained by a bounded worker pool.

Replaces the dashboard's per-user debounce timers, render locks and
in-flight bookkeeping with one invariant: for each key there is at most one
pending job (holding the newest version wanted) and at most one running job,
and at most ``workers`` jobs run at once overall.

``request(key, version, delay_s)`` records that ``version`` is wanted no
earlier than ``now + delay_s``. Requests coalesce: the pending version only
ever rises; a positive delay restarts the wait (debounce), a zero delay makes
the job runnable now unless ``expedite=False``, which leaves an already
pending job's wait alone (passive "this deck is stale" requests must not
cut short a vote debounce). A request that arrives while the key's job is running
stays pending and runs after it, so the newest version always gets built;
one for a version the running job already covers is dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K")
P = TypeVar("P")


@dataclass
class _Pending(Generic[P]):
    payload: P
    version: int
    not_before: float


class WarmScheduler(Generic[K, P]):
    def __init__(
        self,
        run: Callable[[P, int], None],
        *,
        workers: int = 2,
        clock: Callable[[], float] = time.monotonic,
        name: str = "warm",
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self._run = run
        self._workers = workers
        self._clock = clock
        self._name = name
        self._cond = threading.Condition()
        self._pending: dict[K, _Pending[P]] = {}
        # key -> version of the job currently running for it
        self._running: dict[K, int] = {}
        self._threads: list[threading.Thread] = []

    def request(
        self,
        key: K,
        payload: P,
        version: int,
        delay_s: float = 0.0,
        *,
        expedite: bool = True,
    ) -> None:
        now = self._clock()
        with self._cond:
            pending = self._pending.get(key)
            if pending is None and self._running.get(key, version - 1) >= version:
                return  # the running job already builds this version
            if pending is None:
                self._pending[key] = _Pending(payload, version, now + delay_s)
            else:
                pending.payload = payload
                pending.version = max(pending.version, version)
                if delay_s > 0:
                    pending.not_before = now + delay_s
                elif expedite:
                    pending.not_before = min(pending.not_before, now)
            self._ensure_workers_locked()
            self._cond.notify_all()

    def pending_version(self, key: K) -> int | None:
        with self._cond:
            pending = self._pending.get(key)
            return None if pending is None else pending.version

    def clear_pending(self) -> None:
        """Drop queued jobs; running jobs finish normally."""
        with self._cond:
            self._pending.clear()
            self._cond.notify_all()

    def busy(self) -> bool:
        with self._cond:
            return bool(self._pending or self._running)

    def wait_idle(self, timeout_s: float) -> bool:
        """Block until nothing is pending or running; False on timeout."""
        deadline = self._clock() + timeout_s
        with self._cond:
            while self._pending or self._running:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                self._cond.wait(min(remaining, 0.05))
            return True

    def _ensure_workers_locked(self) -> None:
        self._threads = [t for t in self._threads if t.is_alive()]
        while len(self._threads) < self._workers:
            thread = threading.Thread(
                target=self._worker,
                name=f"{self._name}-{len(self._threads)}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _take_ready_locked(self) -> tuple[K, _Pending[P]] | None:
        now = self._clock()
        ready = [
            (job.not_before, key)
            for key, job in self._pending.items()
            if key not in self._running and job.not_before <= now
        ]
        if not ready:
            return None
        _, key = min(ready, key=lambda item: item[0])
        job = self._pending.pop(key)
        self._running[key] = job.version
        return key, job

    def _next_wait_locked(self) -> float | None:
        waits = [
            job.not_before - self._clock()
            for key, job in self._pending.items()
            if key not in self._running
        ]
        return max(0.0, min(waits)) if waits else None

    def _worker(self) -> None:
        while True:
            with self._cond:
                taken = self._take_ready_locked()
                while taken is None:
                    self._cond.wait(self._next_wait_locked())
                    taken = self._take_ready_locked()
            key, job = taken
            try:
                self._run(job.payload, job.version)
            except Exception:
                logging.exception(
                    "%s job failed key=%s version=%s", self._name, key, job.version
                )
            finally:
                with self._cond:
                    self._running.pop(key, None)
                    self._cond.notify_all()
