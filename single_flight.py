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
    def __init__(self) -> None:
        self._done = threading.Event()
        self._value: V | None = None

    def wait(self, timeout_s: float) -> V | None:
        """The leader's result; None if it failed or took longer than this."""
        if not self._done.wait(timeout_s):
            return None
        return self._value


class SingleFlight(Generic[K, V]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[K, Flight[V]] = {}

    def join_or_lead(self, key: K) -> tuple[Flight[V], bool]:
        """The key's flight, and whether the caller leads it (and so must
        `land` it, even on failure)."""
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None:
                return flight, False
            flight = self._flights[key] = Flight()
            return flight, True

    def try_lead(self, key: K) -> Flight[V] | None:
        """A new flight to lead, or None when one is already in the air."""
        flight, leading = self.join_or_lead(key)
        return flight if leading else None

    def land(self, key: K, flight: Flight[V], value: V | None) -> None:
        """Publish the leader's result (None = failed) and release the key."""
        flight._value = value
        with self._lock:
            if self._flights.get(key) is flight:
                del self._flights[key]
        flight._done.set()

    def in_flight(self, key: K) -> bool:
        with self._lock:
            return key in self._flights
