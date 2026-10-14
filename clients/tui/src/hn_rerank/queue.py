"""Client-only queue composition, preserving each card's served origin."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import UTC, datetime

from .models import Feed, View, Window


@dataclass(frozen=True)
class QueueEntry:
    story_id: int
    origin: View
    served_rank: int
    served_version: int


def default_sort(now: datetime | None = None) -> str:
    """Alternate startup sorts by UTC ordinal day (inject now for tests)."""
    day = (now if now is not None else datetime.now(UTC)).astimezone(UTC)
    return "queue" if day.toordinal() % 2 else "recommended"


def queue_entries(
    feed: Feed, excluded: set[int] | None = None, restored: set[int] | None = None
) -> list[QueueEntry]:
    """Mix Recommended, Explore, Recommended, Popular throughout each round.

    Missing stories and duplicates are skipped. For multiple badge kinds,
    the earliest group wins; every group's internal order is served order.
    """
    stories = {story.id: story for story in feed.stories}
    entries: list[QueueEntry] = []
    seen: set[int] = set(excluded or ())
    restored_ids = restored or set()

    def served_order(view: View) -> list[int]:
        return [sid for sid in feed.orders.get(view, []) if sid not in restored_ids]

    def append(sid: int, view: View, rank: int) -> bool:
        if sid not in stories or sid in seen:
            return False
        seen.add(sid)
        entries.append(QueueEntry(sid, view, rank, feed.version))
        return True

    for rank, sid in enumerate(served_order("recommended")):
        append(sid, "recommended", rank)
    groups: list[list[tuple[int, int]]] = [[], [], [], []]
    kinds = ("discover", "uncertain", "interest")
    for rank, sid in enumerate(served_order("explore")):
        if sid not in stories:
            continue
        badges = {badge.kind for badge in stories[sid].badge_details}
        group = next((i for i, kind in enumerate(kinds) if kind in badges), 3)
        groups[group].append((rank, sid))
    for group in groups:
        for rank, sid in group:
            append(sid, "explore", rank)
    popular = 0
    for rank, sid in enumerate(served_order("popular")):
        if append(sid, "popular", rank):
            popular += 1
            if popular == 5:
                break
    lanes = {
        view: deque(entry for entry in entries if entry.origin == view)
        for view in ("recommended", "explore", "popular")
    }
    mixed: list[QueueEntry] = []
    while any(lanes.values()):
        for view in ("recommended", "explore", "recommended", "popular"):
            if lanes[view]:
                mixed.append(lanes[view].popleft())
    return mixed


def build_round(
    rounds: list[list[QueueEntry]],
    feed: Feed,
    excluded: set[int],
    restored: set[int] | None = None,
) -> list[list[QueueEntry]]:
    """Append one nonempty round, preserving every earlier slot and origin."""
    seen = {entry.story_id for batch in rounds for entry in batch}
    batch = queue_entries(feed, excluded | seen, restored)
    return [*rounds, batch] if batch else list(rounds)


def in_window(posted: int, window: Window, now: float) -> bool:
    """Nested recent windows; archive cards never age out of their queue."""
    if window == "archive" or posted <= 0:
        return True  # Unknown post times never age out.
    seconds = {"12h": 43200, "1d": 86400, "1w": 604800, "1m": 2592000}
    return max(now - posted, 0) <= seconds[window]
