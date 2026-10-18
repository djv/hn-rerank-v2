"""Reading-time (dwell) segments for the reader's interaction events."""

from __future__ import annotations

from dataclasses import dataclass, field

# Server bound for duration_ms (server.py _parse_interaction_event).
MAX_DWELL_MS = 86_400_000


@dataclass(frozen=True)
class DwellContext:
    """What a dwell event needs: the selected story and its dimensions."""

    story_id: int
    position: int
    dashboard_version: int
    sort_mode: str
    window: str
    badges: tuple[str, ...] = field(default_factory=tuple)
    surface: str = "views"
    queue_position: int | None = None


@dataclass(frozen=True)
class Dwell:
    ctx: DwellContext
    duration_ms: int


# Input idle longer than this ends a reading segment. A terminal left open
# keeps focus, so this is the only bound on away time; 90 s covers reading a
# summary page, and 300 s logged breaks as reading (FINDINGS 2026-10-13).
IDLE_LIMIT_S = 90.0


class DwellTracker:
    """One open reading segment at most, with an injected monotonic clock.

    Every method takes the current monotonic time in seconds. A segment's
    end is ``min(now, last_input + idle_limit_s)`` where ``last_input`` is
    the later of the segment start and the last ``touch``. Dwells shorter
    than ``min_s`` are dropped (None), matching the 1 s impression rule.
    Durations are integer milliseconds, clamped to 0..86,400,000.
    """

    def __init__(self, idle_limit_s: float = IDLE_LIMIT_S, min_s: float = 1.0) -> None:
        self._idle_limit_s = idle_limit_s
        self._min_s = min_s
        self._ctx: DwellContext | None = None
        self._start: float | None = None
        self._last_input: float | None = None

    @property
    def current(self) -> DwellContext | None:
        """The open segment's context, or None when nothing is open."""
        return self._ctx

    def _finish(self, ctx: DwellContext, start: float, end: float) -> Dwell | None:
        duration_ms = int((end - start) * 1000)
        duration_ms = max(duration_ms, 0)
        duration_ms = min(duration_ms, MAX_DWELL_MS)
        if (end - start) < self._min_s:
            return None
        return Dwell(ctx, duration_ms)

    def start(self, ctx: DwellContext, now: float) -> Dwell | None:
        """Stop the open segment (if any) at *now* and open *ctx* at *now*."""
        old: Dwell | None = None
        if self._ctx is not None:
            assert self._start is not None and self._last_input is not None
            end = min(now, self._last_input + self._idle_limit_s)
            old = self._finish(self._ctx, self._start, end)
        self._ctx = ctx
        self._start = now
        self._last_input = now
        return old

    def touch(self, now: float) -> Dwell | None:
        """Record user input at *now*.

        Past the idle limit without input, the open segment closes at
        ``last_input + idle_limit_s`` and a new segment for the same
        context opens at *now* (no new impression); returns the old dwell.
        """
        if self._ctx is None:
            return None
        assert self._start is not None and self._last_input is not None
        if now < self._last_input:
            return None
        if now - self._last_input > self._idle_limit_s:
            end = self._last_input + self._idle_limit_s
            old = self._finish(self._ctx, self._start, end)
            self._start = now
            self._last_input = now
            return old
        self._last_input = now
        return None

    def stop(self, now: float) -> Dwell | None:
        """Close the open segment at *now* (idle-capped); None when empty."""
        if self._ctx is None:
            return None
        assert self._start is not None and self._last_input is not None
        ctx, start, last_input = self._ctx, self._start, self._last_input
        self._ctx = None
        self._start = None
        self._last_input = None
        end = min(now, last_input + self._idle_limit_s)
        return self._finish(ctx, start, end)
