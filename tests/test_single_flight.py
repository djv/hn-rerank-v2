from __future__ import annotations

import threading

from hypothesis import given, settings, strategies as st

from single_flight import SingleFlight


@given(
    flights=st.dictionaries(
        st.integers(-5, 5),
        st.tuples(st.integers(0, 4), st.none() | st.text(max_size=5)),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=40, deadline=None)
def test_followers_share_their_leaders_result(
    flights: dict[int, tuple[int, str | None]],
) -> None:
    """Per key: one leader; everyone who joins before it lands gets exactly
    its result (None when it failed), never another key's; once landed, the
    next caller leads a new flight."""
    table: SingleFlight[int, str] = SingleFlight()
    leaders = {}
    for key in flights:
        flight, leading = table.join_or_lead(key)
        assert leading
        leaders[key] = flight
        assert table.try_lead(key) is None

    joined = threading.Semaphore(0)
    results: dict[tuple[int, int], tuple[bool, str | None]] = {}

    def follow(key: int, index: int) -> None:
        flight, leading = table.join_or_lead(key)
        joined.release()
        results[key, index] = (leading, flight.wait(5.0))

    threads = [
        threading.Thread(target=follow, args=(key, index))
        for key, (followers, _) in flights.items()
        for index in range(followers)
    ]
    for thread in threads:
        thread.start()
    for _ in threads:
        assert joined.acquire(timeout=5.0)
    for key, (_, value) in flights.items():
        table.land(key, leaders[key], value)
    for thread in threads:
        thread.join(5.0)

    assert results == {
        (key, index): (False, value)
        for key, (followers, value) in flights.items()
        for index in range(followers)
    }
    for key in flights:
        assert not table.in_flight(key)
        assert table.join_or_lead(key)[1]


def test_a_follower_stops_waiting_at_its_timeout() -> None:
    table: SingleFlight[str, str] = SingleFlight()
    table.join_or_lead("story")
    flight, leading = table.join_or_lead("story")
    assert not leading
    assert flight.wait(0.01) is None


def test_landing_a_superseded_flight_leaves_the_new_one() -> None:
    table: SingleFlight[str, str] = SingleFlight()
    old, _ = table.join_or_lead("story")
    table.land("story", old, "a")
    new, leading = table.join_or_lead("story")
    assert leading
    table.land("story", old, "late")
    assert table.in_flight("story")
    table.land("story", new, "b")
    assert new.wait(0) == "b"
