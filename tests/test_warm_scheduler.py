"""Tests for the coalescing per-key warm scheduler."""

from __future__ import annotations

import random
import threading
import time

from hypothesis import given, settings, strategies as st

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


@settings(max_examples=40, deadline=None)
@given(
    requests=st.lists(
        st.tuples(st.integers(0, 3), st.integers(1, 20), st.sampled_from([0.0, 0.01])),
        min_size=1,
        max_size=25,
    ),
    workers=st.integers(1, 3),
    seed=st.integers(0, 2**16),
)
def test_scheduler_invariants_hold_for_any_request_sequence(
    requests: list[tuple[int, int, float]], workers: int, seed: int
) -> None:
    """For any interleaving of non-decreasing per-key requests: jobs for one
    key never overlap, a key's built versions never go backwards, the newest requested version is always
    built, and no more than `workers` jobs run at once. (A version may be
    built again if it is requested after its job finished; skipping
    finished work is the caller's cache check, not the scheduler's.)"""
    rng = random.Random(seed)
    lock = threading.Lock()
    running: set[int] = set()
    overlaps: list[int] = []
    peak = 0
    built: dict[int, list[int]] = {}

    def run(key: int, version: int) -> None:
        nonlocal peak
        with lock:
            if key in running:
                overlaps.append(key)
            running.add(key)
            peak = max(peak, len(running))
        time.sleep(rng.choice([0.0, 0.001, 0.003]))
        with lock:
            built.setdefault(key, []).append(version)
            running.discard(key)

    # The handler only ever requests max(requested, live version), so each
    # key's request stream is non-decreasing; model that.
    highest: dict[int, int] = {}
    stream = []
    for key, version, delay in requests:
        highest[key] = max(highest.get(key, 0), version)
        stream.append((key, highest[key], delay))
    requests = stream

    sched: WarmScheduler[int, int] = WarmScheduler(run, workers=workers)
    for key, version, delay in requests:
        sched.request(key, key, version, delay_s=delay)
        if rng.random() < 0.3:
            time.sleep(0.001)
    assert sched.wait_idle(5)

    assert overlaps == []
    assert peak <= workers
    for key in {k for k, _, _ in requests}:
        versions = built[key]
        assert versions == sorted(versions)
        assert versions[-1] == max(v for k, v, _ in requests if k == key)


def test_request_for_the_running_version_is_not_queued_again() -> None:
    """Readiness polls re-request the in-flight version every few hundred ms;
    they must not queue a duplicate job behind it."""
    started = threading.Event()
    release = threading.Event()
    ran: list[int] = []

    def run(key: int, version: int) -> None:
        started.set()
        release.wait(2)
        ran.append(version)

    sched: WarmScheduler[int, int] = WarmScheduler(run, workers=1)
    sched.request(1, 1, 5)
    assert started.wait(2)
    for _ in range(3):
        sched.request(1, 1, 5)
        sched.request(1, 1, 4)
    assert sched.pending_version(1) is None
    sched.request(1, 1, 6)  # newer: must run after the current job
    release.set()
    assert sched.wait_idle(2)
    assert ran == [5, 6]
