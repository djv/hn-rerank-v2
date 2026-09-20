from __future__ import annotations

import asyncio
from dataclasses import replace

import httpx
import pytest
from hypothesis import given, settings, strategies as st

import pipeline
from database import Action, Database, Story, StoryIdentityConflict
from pipeline.enrichment import fetch_story, fetch_rss_feeds


@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(["refresh", "ingest", "heal"]),
            st.integers(1, 100),
            st.integers(0, 2000),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(deadline=None)
def test_comment_snapshot_survives_operation_sequences(
    operations: list[tuple[str, int, int]],
) -> None:
    db = Database(":memory:")
    initial = Story(
        id=1,
        title="Thread",
        url=None,
        score=1,
        time=1,
        text_content=pipeline.compose_story_text("Thread", comments="old"),
        top_comments="old",
        comment_count=1,
        comment_count_at_fetch=1,
    )
    snapshot = (initial.top_comments, 1)
    highest_live = 1
    previous_snapshots = [initial]
    try:
        db.upsert_story(initial)
        for index, (operation, size, count) in enumerate(operations, start=2):
            if operation == "refresh":
                comment = f"revision {index} " + "x" * size
                incoming = replace(
                    initial,
                    top_comments=comment,
                    text_content=pipeline.compose_story_text(
                        "Thread", comments=comment
                    ),
                    comment_count=count,
                    comment_count_at_fetch=index,
                )
                db.upsert_story(incoming, comments_authoritative=True)
                snapshot = (comment, index)
                previous_snapshots.append(incoming)
            elif operation == "ingest":
                # Select any previously observed snapshot, not just the initial row.
                incoming = previous_snapshots[size % len(previous_snapshots)]
                count = incoming.comment_count or 0
                db.upsert_story(incoming)
            else:
                current = db.get_story(1)
                assert current is not None
                db.upsert_story(replace(current, comment_count=count))
            highest_live = max(highest_live, count)
            current = db.get_story(1)
            assert current is not None
            assert (current.top_comments, current.comment_count_at_fetch) == snapshot
            assert current.comment_count == highest_live
            assert current.text_content == pipeline.compose_story_text(
                "Thread", comments=snapshot[0]
            )
    finally:
        db.close()


@pytest.mark.parametrize("prewarm", [False, True])
@given(
    old_size=st.integers(20, 100),
    new_size=st.integers(20, 100),
    fetched_count=st.integers(1, 1000),
    live_count=st.integers(1, 2000),
)
@settings(deadline=None)
def test_successful_refresh_persists_comment_snapshot(
    prewarm: bool, old_size: int, new_size: int, fetched_count: int, live_count: int
) -> None:
    db = Database(":memory:")
    try:
        old = Story(
            id=1,
            title="Thread",
            url="https://example.com/story",
            score=1,
            time=1,
            text_content="Thread old discussion",
            source="hn",
            top_comments="old " * old_size,
            comment_count=live_count,
            comment_count_at_fetch=fetched_count - 1,
            article_body="Preserved article",
        )
        db.upsert_story(old)

        async def refresh() -> Story | None:
            transport = httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "type": "story",
                        "title": "Thread",
                        "url": old.url,
                        "num_comments": fetched_count,
                        "children": [
                            {
                                "id": 2,
                                "type": "comment",
                                "text": "new " * new_size,
                                "children": [],
                            }
                        ],
                    },
                )
            )
            async with httpx.AsyncClient(transport=transport) as client:
                return await fetch_story(client, 1, db, force=True)

        if prewarm:
            from unittest.mock import patch

            with patch(
                "ch_client.query_stories_with_comments",
                return_value={
                    1: {
                        "num_comments": fetched_count,
                        "children": [
                            {
                                "id": 2,
                                "type": "comment",
                                "text": "new " * new_size,
                                "children": [],
                            }
                        ],
                    }
                },
            ):
                assert pipeline.prewarm_top_stories([1], db) == 1
            fresh = db.get_story(1)
        else:
            fresh = asyncio.run(refresh())
        expected_comments = ("new " * new_size).rstrip()
        assert fresh is not None
        assert fresh.top_comments == expected_comments
        stored = db.get_story(1)
        assert stored is not None
        assert (stored.top_comments, stored.comment_count_at_fetch) == (
            expected_comments,
            fetched_count,
        )
        assert stored.comment_count == max(live_count, fetched_count)
        assert fresh == stored
        assert stored.text_content == pipeline.compose_story_text(
            stored.title, stored.self_text, expected_comments, old.article_body
        )
        assert stored.article_body == old.article_body
        # Re-ingestion of an older snapshot must not undo a successful refresh.
        db.upsert_story(old)
        again = db.get_story(1)
        assert again is not None
        assert (again.top_comments, again.comment_count_at_fetch) == (
            expected_comments,
            fetched_count,
        )
    finally:
        db.close()


@given(status=st.sampled_from([403, 404, 429, 500, 503]), size=st.integers(20, 100))
@settings(deadline=None)
def test_failed_refresh_preserves_snapshot(status: int, size: int) -> None:
    db = Database(":memory:")
    try:
        old = Story(
            id=1,
            title="Thread",
            url="https://example.com/story",
            score=1,
            time=1,
            text_content="Thread",
            top_comments="old " * size,
            comment_count=20,
            comment_count_at_fetch=20,
        )
        db.upsert_story(old)

        async def refresh() -> Story | None:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(status))
            ) as client:
                return await fetch_story(client, 1, db, force=True)

        assert asyncio.run(refresh()) == old
        assert db.get_story(1) == old
    finally:
        db.close()


@given(sid=st.integers(-(2**31), -1), action=st.sampled_from(["up", "down", "neutral"]))
@settings(deadline=None)
def test_rss_identity_conflict_preserves_story_and_feedback(
    sid: int, action: Action
) -> None:
    db = Database(":memory:")
    try:
        original = Story(
            id=sid,
            title="Original",
            url="https://example.com/first",
            score=0,
            time=1,
            text_content="Original",
            source="rss_test",
        )
        db.upsert_story(original)
        user = db.create_user("property-user")
        db.upsert_feedback(user.id, sid, action)
        before = db.get_feedback_for_training(user.id)
        with pytest.raises(StoryIdentityConflict):
            db.upsert_story(
                replace(original, title="Different", url="https://example.com/second")
            )
        assert db.get_story(sid) == original
        assert db.get_feedback_for_training(user.id) == before
    finally:
        db.close()


@given(reverse=st.booleans(), title=st.text(alphabet="abcdef", min_size=1, max_size=30))
@settings(deadline=None)
def test_rss_ingestion_skips_real_hash_collision(reverse: bool, title: str) -> None:
    # Real collision in the production URL hash; do not duplicate its implementation.
    urls = [
        "https://example.com/rss-collision-probe/10971",
        "https://example.com/rss-collision-probe/62542",
    ]
    if reverse:
        urls.reverse()
    urls.append("https://example.com/unrelated-story")
    entries = "".join(
        f"<item><title>{title}</title><link>{url}</link></item>" for url in urls
    )
    feed = f'<rss version="2.0"><channel>{entries}</channel></rss>'

    async def fake_fetch(
        *args: object, **kwargs: object
    ) -> tuple[int, bytes, dict[str, str]]:
        return 200, feed.encode(), {}

    db = Database(":memory:")
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("http_fetch.fetch_with_urllib_fallback", fake_fetch)
            stories = asyncio.run(
                fetch_rss_feeds(
                    ["https://example.com/feed.xml"],
                    10,
                    30,
                    set(),
                    db,
                )
            )
            assert [s.url for s in stories] == [urls[0], urls[2]]
            first = db.get_story(-505376979)
            assert first is not None and first.url == urls[0]
            # Repeated ingestion preserves the winner and stable IDs.
            repeated = asyncio.run(
                fetch_rss_feeds(
                    ["https://example.com/feed.xml"],
                    10,
                    30,
                    set(),
                    db,
                )
            )
            assert [(s.id, s.url) for s in repeated] == [(s.id, s.url) for s in stories]
    finally:
        db.close()


@given(sid=st.integers(-(2**31), -10))
@settings(deadline=None)
def test_reddit_collision_excluded_from_snapshot_and_prewarm(sid: int) -> None:
    from reddit_feed_cache import cache

    feed = "https://www.reddit.com/r/test/top/.rss"
    db = Database(":memory:")
    original = Story(
        id=sid,
        title="Original",
        url="https://example.com/first",
        score=0,
        time=1,
        text_content="Original",
        source="rss_test",
    )
    conflict = replace(original, url="https://reddit.com/r/test/comments/second")
    safe = replace(conflict, id=-2, url="https://reddit.com/r/test/comments/safe")
    prewarmed: list[int] = []

    def topfeeds(*args: object) -> tuple[list, list[str]]:
        cache.set(feed, [conflict, safe])
        return [], [feed]

    def prewarm(ids: list[int], database: Database) -> tuple[list, list[int]]:
        prewarmed.extend(ids)
        return [], []

    try:
        db.upsert_story(original)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pipeline, "build_reddit_topfeed_factories", topfeeds)
            patch.setattr(pipeline, "build_reddit_prewarm_factories", prewarm)
            config = replace(
                pipeline.Config(), rss=pipeline.RssConfig(enabled=True, feeds=(feed,))
            )
            result = pipeline.refresh_reddit_candidates(config, db, None)
        assert db.get_story(sid) == original
        assert result.changed_stories == 1
        assert prewarmed == [-2]
        state = db.get_reddit_feed_state(feed)
        assert state is not None
        assert state.item_count == 1
        with db.conn() as conn:
            ids = conn.execute(
                "SELECT story_id FROM reddit_feed_items WHERE feed_url = ?", (feed,)
            ).fetchall()
        assert ids == [(-2,)]
    finally:
        cache.reset()
        db.close()
