from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from hypothesis import given, settings, strategies as st

from database import Database, Story
from pipeline.enrichment import _merge_source_context, prewarm_lesswrong_stories
from pipeline.ranking import compose_story_text
from server import LessWrongContext


@given(
    initial_count=st.integers(1, 1000),
    increments=st.lists(st.integers(1, 500), min_size=1, max_size=8),
    body_size=st.integers(20, 200),
    comment_size=st.integers(20, 200),
)
def test_source_merge_is_idempotent_and_resists_late_metadata(
    initial_count: int, increments: list[int], body_size: int, comment_size: int
) -> None:
    """Growing counts must survive old responses without shrinking rich text."""
    story = Story(
        id=-1,
        title="Thread",
        url="https://example.org/thread",
        score=10,
        time=1,
        text_content="",
        self_text="b" * body_size,
        top_comments="c" * comment_size,
        comment_count=initial_count,
        comment_count_at_fetch=initial_count,
    )
    highest = initial_count
    for increment in increments:
        highest += increment
        context = LessWrongContext(
            self_text="b",
            top_comments="c",
            comment_count=highest,
            score=20,
        )
        fresh = _merge_source_context(story, context, None, prefer_longer_comments=True)
        assert fresh.comment_count == highest
        assert fresh.comment_count_at_fetch == highest
        assert fresh.self_text == story.self_text
        assert fresh.top_comments == story.top_comments
        assert fresh.text_content == compose_story_text(
            story.title, story.self_text, story.top_comments, ""
        )
        assert (
            _merge_source_context(fresh, context, None, prefer_longer_comments=True)
            == fresh
        )
        delayed = replace(context, comment_count=initial_count, score=1)
        assert (
            _merge_source_context(fresh, delayed, None, prefer_longer_comments=True)
            == fresh
        )
        story = fresh


@given(
    initial_count=st.integers(1, 1000),
    growth=st.integers(1, 500),
    body_size=st.integers(20, 200),
    comment_size=st.integers(20, 200),
    same_text=st.booleans(),
)
@settings(deadline=None)
def test_lesswrong_metadata_refresh_reaches_db_without_richer_text(
    initial_count: int,
    growth: int,
    body_size: int,
    comment_size: int,
    same_text: bool,
) -> None:
    """Exercise the actual prewarm gate, persistence, replay and delayed fetch."""
    body, comments = "b" * body_size, "c" * comment_size
    story = Story(
        id=-1,
        title="Thread",
        url="https://www.lesswrong.com/posts/abc123/thread",
        source="rss_lesswrong_com",
        score=10,
        time=1,
        self_text=body,
        top_comments=comments,
        text_content=compose_story_text("Thread", body, comments, ""),
        comment_count=initial_count,
        comment_count_at_fetch=initial_count,
    )
    context = LessWrongContext(
        self_text=body if same_text else "b",
        top_comments=comments if same_text else "c",
        comment_count=initial_count + growth,
        score=11,
    )

    async def fetch_context(post_id: str) -> LessWrongContext:
        assert post_id == "abc123"
        return context

    db = Database(":memory:")
    try:
        db.upsert_story(story)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("server._fetch_lesswrong_context", fetch_context)
            assert asyncio.run(prewarm_lesswrong_stories([-1], db)) == 1
            fresh = db.get_story(-1)
            assert fresh is not None
            assert fresh.comment_count == initial_count + growth
            assert fresh.comment_count_at_fetch == initial_count + growth
            assert fresh.score == 11
            assert fresh.text_content == story.text_content
            assert fresh.self_text == body
            assert fresh.top_comments == comments
            assert asyncio.run(prewarm_lesswrong_stories([-1], db)) == 0
            context = replace(context, comment_count=initial_count, score=1)
            assert asyncio.run(prewarm_lesswrong_stories([-1], db)) == 0
            assert db.get_story(-1) == fresh
    finally:
        db.close()
