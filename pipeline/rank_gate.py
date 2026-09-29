"""Let a user-facing rerank run without background embedding work beside it.

On the 4-core VPS a rerank after a vote took 8-17 s longer (tier2 and
dedup stages, normally under 1 s) while background article fetches and
regen prewarm were embedding stories in the same process (2026-09-29).
Reranks mark themselves active; background loops call ``wait_idle`` between
small units of work and resume when no rerank is running.

Never call ``wait_idle`` from code a rerank itself runs: it would wait on
itself until the timeout.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

# Upper bound on one wait, so a stuck counter only delays background work.
MAX_WAIT_S = 120.0

_active = 0
_cond = threading.Condition()


@contextmanager
def ranking() -> Iterator[None]:
    """Mark a rerank as running for the duration of the block."""
    global _active
    with _cond:
        _active += 1
    try:
        yield
    finally:
        with _cond:
            _active -= 1
            _cond.notify_all()


def is_ranking() -> bool:
    with _cond:
        return _active > 0


def wait_idle(max_wait_s: float = MAX_WAIT_S) -> float:
    """Block until no rerank is running (or *max_wait_s* passes); returns
    the seconds waited."""
    start = time.monotonic()
    with _cond:
        _cond.wait_for(lambda: _active == 0, timeout=max_wait_s)
    return time.monotonic() - start


async def wait_idle_async(max_wait_s: float = MAX_WAIT_S) -> float:
    if not is_ranking():
        return 0.0
    return await asyncio.to_thread(wait_idle, max_wait_s)
