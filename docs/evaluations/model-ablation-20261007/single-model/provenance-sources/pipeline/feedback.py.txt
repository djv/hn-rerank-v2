"""Canonical training rows without modifying saved feedback."""

from __future__ import annotations

from database import Story
from dedup import normalize_url


def deduplicate_feedback(
    stories: list[Story], labels: list[int], times: list[float]
) -> tuple[list[Story], list[int], list[float]]:
    """Keep the latest vote per normalized URL; missing URLs use story ID.

    Conflicting cross-post votes resolve by timestamp then ID, deterministically.
    This is URL deduplication, not semantic duplicate detection.
    """
    if not (len(stories) == len(labels) == len(times)):
        raise ValueError("Feedback arrays must be aligned")
    latest: dict[str, int] = {}
    for i, story in enumerate(stories):
        key = str(normalize_url(story.url) or f"story:{story.id}")
        old = latest.get(key)
        if old is None or (times[i], story.id) > (times[old], stories[old].id):
            latest[key] = i
    keep = sorted(latest.values())
    return (
        [stories[i] for i in keep],
        [labels[i] for i in keep],
        [times[i] for i in keep],
    )
