"""Tests for the coalescing per-key warm scheduler."""

from __future__ import annotations

import threading
import time

from warm_scheduler import WarmScheduler


def test_requests_coalesce_to_newest_version() -> None:
    ran: list[tuple[str, int]] = []
    gate = threading.Event()
    blocker_started = threading.Event()

    def run(payload: str, version: int) -> None:
        if payload == "blocker":
            blocker_started.set()
            gate.wait(2)
        ran.append((payload, version))

    sched: WarmScheduler[int, str] = WarmScheduler(run, workers=1)
    # Keep the only worker busy on another key so key 1's requests pile up.
    sched.request(0, "blocker", 1)
    assert blocker_started.wait(2)
    for version in (1, 3, 2):
        sched.request(1, "user", version)
    gate.set()
    assert sched.wait_idle(2)
    assert [v for p, v in ran if p == "user"] == [3]


def test_one_job_per_key_and_newer_request_runs_after() -> None:
    started = threading.Event()
    release = threading.Event()
    active: set[int] = set()
    overlaps: list[int] = []
    ran: list[int] = []

    def run(key: int, version: int) -> None:
        if key in active:
            overlaps.append(key)
        active.add(key)
        started.set()
        release.wait(2)
        ran.append(version)
        active.discard(key)

    sched: WarmScheduler[int, int] = WarmScheduler(run, workers=4)
    sched.request(7, 7, 1)
    assert started.wait(2)
    sched.request(7, 7, 2)  # arrives mid-run: must wait, then run
    time.sleep(0.05)
    release.set()
    assert sched.wait_idle(2)
    assert ran == [1, 2]
    assert overlaps == []


def test_worker_bound_limits_concurrency() -> None:
    lock = threading.Lock()
    live = 0
    peak = 0

    def run(key: int, version: int) -> None:
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.03)
        with lock:
            live -= 1

    sched: WarmScheduler[int, int] = WarmScheduler(run, workers=2)
    for key in range(8):
        sched.request(key, key, 1)
    assert sched.wait_idle(3)
    assert peak == 2


def test_positive_delay_debounces_and_zero_delay_runs_now() -> None:
    ran: list[float] = []
    sched: WarmScheduler[int, int] = WarmScheduler(
        lambda p, v: ran.append(time.monotonic()), workers=1
    )
    t0 = time.monotonic()
    sched.request(1, 1, 1, delay_s=0.3)
    time.sleep(0.1)
    sched.request(1, 1, 2, delay_s=0.3)  # restarts the idle wait
    assert sched.pending_version(1) == 2
    assert sched.wait_idle(2)
    assert ran and ran[0] - t0 >= 0.35

    ran.clear()
    t1 = time.monotonic()
    sched.request(2, 2, 1, delay_s=5.0)
    sched.request(2, 2, 1, delay_s=0.0)  # an immediate request cuts the wait
    assert sched.wait_idle(2)
    assert ran and ran[0] - t1 < 1.0


def test_failed_job_is_logged_and_worker_survives() -> None:
    ran: list[int] = []

    def run(key: int, version: int) -> None:
        if key == 1:
            raise RuntimeError("boom")
        ran.append(key)

    sched: WarmScheduler[int, int] = WarmScheduler(run, workers=1)
    sched.request(1, 1, 1)
    sched.request(2, 2, 1)
    assert sched.wait_idle(2)
    assert ran == [2]
    assert not sched.busy()


def test_clear_pending_drops_queued_jobs() -> None:
    ran: list[int] = []
    sched: WarmScheduler[int, int] = WarmScheduler(
        lambda p, v: ran.append(v), workers=1
    )
    sched.request(1, 1, 5, delay_s=10.0)
    assert sched.busy()
    sched.clear_pending()
    assert sched.wait_idle(1)
    assert ran == []
