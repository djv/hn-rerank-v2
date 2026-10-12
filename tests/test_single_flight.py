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


def test_captured_ordinary_keeps_its_forced_identity() -> None:
    """Freshness is read off the captured flight, not the key: an old
    ordinary flight still reports ordinary after a fresh leader takes the
    key, while the table reports the new flight as forced."""
    table: SingleFlight[int, str] = SingleFlight()
    old, leading = table.join_or_lead(7)
    assert leading
    assert old.forced is False
    table.land(7, old, "ordinary-result")
    new, leading = table.join_or_lead(7, forced=True)
    assert leading
    assert new.forced is True
    assert old.forced is False
    assert old.wait(0) == "ordinary-result"
    assert table.led_forced(7) is True
    table.land(7, new, "fresh")
    assert new.wait(0) == "fresh"
    assert old.forced is False


def test_followup_is_shared_even_after_it_lands() -> None:
    """All forced waiters of one ordinary flight share a single follow-up,
    including a waiter that asks after the follow-up already landed."""
    table: SingleFlight[int, str] = SingleFlight()
    ordinary, leading = table.join_or_lead(7)
    assert leading
    table.land(7, ordinary, "ordinary-result")
    first, leading = table.followup_or_lead(ordinary, 7)
    assert leading and first.forced is True
    table.land(7, first, "fresh")
    second, leading = table.followup_or_lead(ordinary, 7)
    assert not leading
    assert second is first
    assert second.wait(0) == "fresh"


def test_followup_refuses_parallel_work_on_ordinary_key() -> None:
    """A forced follow-up never starts beside an ordinary flight: when the
    key holds an ordinary (the waiter's own still-running flight, or a
    newer one), the caller gets that non-forced flight back and must
    report unfinished rather than spend again."""
    table: SingleFlight[int, str] = SingleFlight()
    ordinary, leading = table.join_or_lead(7)
    assert leading
    same, leading = table.followup_or_lead(ordinary, 7)
    assert not leading and same is ordinary and not same.forced
    table.land(7, ordinary, "ordinary-result")
    newer, leading = table.join_or_lead(7)
    assert leading and not newer.forced
    blocked, leading = table.followup_or_lead(ordinary, 7)
    assert not leading and blocked is newer and not blocked.forced


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
