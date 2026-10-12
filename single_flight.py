"""One in-flight computation per key; later callers share its result.

Threads only: a leader runs the work and lands the result, followers block
on `Flight.wait`. Used for TLDR generation, so two taps (or a tap and the
warm prefetch) on the same story make one LLM call, not two.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Flight(Generic[V]):
    def __init__(self, *, forced: bool = False) -> None:
        self._done = threading.Event()
        self._value: V | None = None
        # Bound at creation under the table lock and never mutated: a
        # joiner reads freshness off its own captured flight, not off a
        # later lookup of the key (which may already hold a new flight).
        self.forced = forced
        # The single bounded fresh follow-up for this (ordinary) flight,
        # set once under the table lock by `followup_or_lead` and kept
        # even after completion so late waiters still share it. Dies with
        # this object: no process-long history.
        self._followup: Flight[V] | None = None

    def wait(self, timeout_s: float) -> V | None:
        """The leader's result; None if it failed or took longer than this."""
        if not self._done.wait(timeout_s):
            return None
        return self._value


class SingleFlight(Generic[K, V]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[K, tuple[Flight[V], bool]] = {}

    def join_or_lead(self, key: K, *, forced: bool = False) -> tuple[Flight[V], bool]:
        """The key's flight, and whether the caller leads it (and so must
        `land` it, even on failure).

        `forced` is bound to the flight object atomically at creation: a
        joiner reads `flight.forced` off its own captured flight, never a
        later lookup of the key (which may already hold a new flight).
        """
        with self._lock:
            entry = self._flights.get(key)
            if entry is not None:
                return entry[0], False
            flight: Flight[V] = Flight(forced=forced)
            self._flights[key] = (flight, forced)
            return flight, True

    def led_forced(self, key: K) -> bool:
        """Whether the key's current flight was forced-led (False if none).

        Kept for compatibility; new code reads `flight.forced` off the
        captured flight instead, which cannot race with a newer flight
        for the same key.
        """
        with self._lock:
            entry = self._flights.get(key)
            return entry is not None and entry[0].forced

    def followup_or_lead(self, ordinary: Flight[V], key: K) -> tuple[Flight[V], bool]:
        """The single bounded fresh follow-up for an ordinary flight.

        Every forced waiter of the same *ordinary* flight shares one
        follow-up, even a waiter that wakes after the follow-up already
        landed: the reference lives on the ordinary flight, not in the
        key table. Returns the follow-up and whether the caller leads it
        (and so must `land` it).

        When the caller does not lead, check `flight.forced`: a
        non-forced return means the key is held by an ordinary flight
        (the waiter's own, still running after a timed-out wait, or a
        newer one) — start no parallel work and report unfinished
        instead. A forced return is joined with `wait`, finished or not.
        """
        with self._lock:
            existing = ordinary._followup
            if existing is not None:
                return existing, False
            entry = self._flights.get(key)
            if entry is not None:
                current = entry[0]
                if current.forced:
                    # A fresh flight already leads (a faster forced
                    # waiter's follow-up, or a fresh tap): share it.
                    ordinary._followup = current
                return current, False
            flight: Flight[V] = Flight(forced=True)
            ordinary._followup = flight
            self._flights[key] = (flight, True)
            return flight, True

    def try_lead(self, key: K) -> Flight[V] | None:
        """A new flight to lead, or None when one is already in the air."""
        flight, leading = self.join_or_lead(key)
        return flight if leading else None

    def land(self, key: K, flight: Flight[V], value: V | None) -> None:
        """Publish the leader's result (None = failed) and release the key."""
        flight._value = value
        with self._lock:
            current = self._flights.get(key)
            if current is not None and current[0] is flight:
                del self._flights[key]
        flight._done.set()

    def in_flight(self, key: K) -> bool:
        with self._lock:
            return key in self._flights
