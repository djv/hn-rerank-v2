"""Client-only queue composition, preserving each card's served origin."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .models import Feed, View


@dataclass(frozen=True)
class QueueEntry:
    story_id: int
    origin: View
    served_rank: int


def default_sort(now: datetime | None = None) -> str:
    """Alternate startup sorts by UTC ordinal day (inject now for tests)."""
    day = (now if now is not None else datetime.now(UTC)).astimezone(UTC)
    return "queue" if day.toordinal() % 2 else "recommended"


def queue_entries(feed: Feed) -> list[QueueEntry]:
    """Recommended, badge-grouped Explore, then five additional Popular cards.

    Missing stories and duplicates are skipped. For multiple badge kinds,
    the earliest group wins; every group's internal order is served order.
    """
    stories = {story.id: story for story in feed.stories}
    entries: list[QueueEntry] = []
    seen: set[int] = set()

    def append(sid: int, view: View, rank: int) -> bool:
        if sid not in stories or sid in seen:
            return False
        seen.add(sid)
        entries.append(QueueEntry(sid, view, rank))
        return True

    for rank, sid in enumerate(feed.orders.get("recommended", [])):
        append(sid, "recommended", rank)
    groups: list[list[tuple[int, int]]] = [[], [], [], []]
    kinds = ("discover", "uncertain", "interest")
    for rank, sid in enumerate(feed.orders.get("explore", [])):
        if sid not in stories:
            continue
        badges = {badge.kind for badge in stories[sid].badge_details}
        group = next((i for i, kind in enumerate(kinds) if kind in badges), 3)
        groups[group].append((rank, sid))
    for group in groups:
        for rank, sid in group:
            append(sid, "explore", rank)
    popular = 0
    for rank, sid in enumerate(feed.orders.get("popular", [])):
        if append(sid, "popular", rank):
            popular += 1
            if popular == 5:
                break
    return entries


def extend_queue(snapshot: list[QueueEntry], feed: Feed) -> list[QueueEntry]:
    """Keep all original slots, appending only newly served Recommended cards."""
    entries = list(snapshot)
    seen = {entry.story_id for entry in snapshot}
    available = {story.id for story in feed.stories}
    for rank, sid in enumerate(feed.orders.get("recommended", [])):
        if sid in available and sid not in seen:
            entries.append(QueueEntry(sid, "recommended", rank))
            seen.add(sid)
    return entries
