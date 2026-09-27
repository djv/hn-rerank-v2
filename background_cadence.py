"""Single-flight, start-to-start cadence for discretionary background work."""

from __future__ import annotations

import threading


class BackgroundCadence:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_started: float | None = None
        self._running = False

    def claim(self, now: float, interval: float) -> bool:
        if interval <= 0:
            raise ValueError("interval must be positive")
        with self._lock:
            if self._running or (
                self._last_started is not None and now - self._last_started < interval
            ):
                return False
            self._last_started = now
            self._running = True
            return True

    def finish(self) -> None:
        with self._lock:
            self._running = False
