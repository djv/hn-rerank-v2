from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from hypothesis import given, strategies as st

from background_cadence import BackgroundCadence


@given(
    interval=st.integers(1, 10000),
    steps=st.lists(
        st.tuples(st.integers(0, 10000), st.booleans()), min_size=1, max_size=50
    ),
)
def test_arbitrary_triggers_respect_cadence_and_single_flight(
    interval: int, steps: list[tuple[int, bool]]
) -> None:
    gate = BackgroundCadence()
    now = 0
    last: int | None = None
    running = False
    for elapsed, finish in steps:
        now += elapsed
        if finish:
            gate.finish()
            running = False
        expected = not running and (last is None or now - last >= interval)
        assert gate.claim(now, interval) == expected
        if expected:
            last, running = now, True


def test_simultaneous_claims_have_exactly_one_winner() -> None:
    gate = BackgroundCadence()
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(lambda _: gate.claim(0, 10), range(32))) == 1
    gate.finish()
    assert not gate.claim(9, 10)
    assert gate.claim(10, 10)
