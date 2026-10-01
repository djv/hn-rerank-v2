"""Properties of fetched, stored, and live comment counts on a TLDR tap."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from unittest.mock import patch

from hypothesis import example, given, settings, strategies as st

import pipeline
import server
from database import Database, Story
from pipeline import Config, LiveCounts


class TapGrowthMismatch(AssertionError):
    """The refresh decision compared with a newer stored count instead of fetched."""


@dataclass(frozen=True)
class ThreadCounts:
    fetched: int
    stored: int
    live: int


@st.composite
def thread_counts(draw: st.DrawFn) -> ThreadCounts:
    """Stored growth can precede, equal, or lag the current Firebase count."""
    fetched = draw(st.integers(min_value=1, max_value=1_000))
    stored = fetched + draw(st.integers(min_value=1, max_value=1_000))
    live = draw(st.integers(min_value=fetched, max_value=stored + 1_000))
    return ThreadCounts(fetched, stored, live)


@settings(max_examples=40, deadline=None)
@given(counts=thread_counts())
@example(counts=ThreadCounts(fetched=1, stored=2, live=2))
def test_tap_growth_compares_live_with_fetched_comments(counts: ThreadCounts) -> None:
    """Already recorded growth still requires refreshing an older summary.

    Exercise both production helpers; fake only Firebase's response. The
    persisted count must stay monotonic, and the summary's fetched marker
    must stay unchanged until real comment hydration happens.
    """
    db = Database(":memory:")
    story = Story(
        id=1732,
        title="Young thread with previously fetched comments",
        url=None,
        score=50,
        time=int(time.time() - 3_600),
        text_content="Previously fetched comments",
        comment_count=counts.stored,
        comment_count_at_fetch=counts.fetched,
        top_comments="Previously fetched comments",
    )
    config = Config(tldr_refresh_recent_hours=24, tldr_tap_probe_timeout_seconds=1)

    async def live_items(
        stories: Sequence[Story], timeout_s: float
    ) -> dict[int, LiveCounts]:
        assert [candidate.id for candidate in stories] == [story.id]
        assert timeout_s == config.tldr_tap_probe_timeout_seconds
        return {story.id: LiveCounts(score=story.score, descendants=counts.live)}

    try:
        db.upsert_story(story)
        with patch.object(pipeline, "_probe_live_items", live_items):
            healed, grew = server._tldr_tap_probe_growth(db, config, story)
        persisted = db.get_story(story.id)
        assert persisted is not None
        assert healed.comment_count == max(counts.stored, counts.live)
        assert persisted.comment_count == max(counts.stored, counts.live)
        assert persisted.comment_count_at_fetch == counts.fetched
        assert persisted.top_comments == story.top_comments
        if grew != (counts.live > counts.fetched):
            raise TapGrowthMismatch(f"Refresh decision {grew} for {counts!r}")
    finally:
        db.close()
