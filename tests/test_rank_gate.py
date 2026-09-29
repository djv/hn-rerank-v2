from __future__ import annotations

import asyncio
import threading
import time

from pipeline import rank_gate


def test_wait_idle_blocks_until_the_rerank_ends() -> None:
    entered = threading.Event()
    release = threading.Event()

    def rerank() -> None:
        with rank_gate.ranking():
            entered.set()
            release.wait(5)

    t = threading.Thread(target=rerank)
    t.start()
    entered.wait(5)
    assert rank_gate.is_ranking()

    waited: list[float] = []
    waiter = threading.Thread(target=lambda: waited.append(rank_gate.wait_idle(5)))
    waiter.start()
    time.sleep(0.1)
    assert waiter.is_alive()

    release.set()
    t.join(5)
    waiter.join(5)
    assert not rank_gate.is_ranking()
    assert 0.05 < waited[0] < 5


def test_wait_idle_gives_up_after_max_wait() -> None:
    with rank_gate.ranking():
        assert rank_gate.wait_idle(0.05) >= 0.05
    assert not rank_gate.is_ranking()


def test_wait_idle_async_is_free_when_idle() -> None:
    assert asyncio.run(rank_gate.wait_idle_async()) == 0.0
