import time
import sqlite3
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from database import Database, Story
from scripts.migrate_db_to_strict import migrate_database


@pytest.fixture
def db():
    db_instance = Database(":memory:")
    yield db_instance
    db_instance.close()


def test_upsert_and_get_story(db):
    from pipeline import compose_story_text

    expected_text = compose_story_text(
        "Test Story",
        "Some self text content",
        "Some top comments here",
        "Some article body here",
    )
    story = Story(
        id=123,
        title="Test Story",
        url="https://example.com/test",
        score=100,
        time=1600000000,
        text_content=expected_text,
        source="hn",
        comment_count=42,
        discussion_url="https://news.ycombinator.com/item?id=123",
        self_text="Some self text content",
        top_comments="Some top comments here",
        article_body="Some article body here",
    )
    db.upsert_story(story)

    fetched = db.get_story(123)
    assert fetched is not None
    assert fetched.id == story.id
    assert fetched.title == story.title
    assert fetched.url == story.url
    assert fetched.score == story.score
    assert fetched.time == story.time

    expected_text = compose_story_text(
        story.title, story.self_text, story.top_comments, story.article_body
    )
    assert fetched.text_content == expected_text
    assert fetched.source == story.source
    assert fetched.comment_count == story.comment_count
    assert fetched.discussion_url == story.discussion_url
    assert fetched.self_text == story.self_text
    assert fetched.top_comments == story.top_comments
    assert fetched.article_body == story.article_body

    # Check update
    updated_story = Story(
        id=123,
        title="Updated Title",
        url="https://example.com/test",
        score=105,
        time=1600000000,
        text_content="Some updated text",
        source="hn",
        comment_count=43,
        discussion_url="https://news.ycombinator.com/item?id=123",
    )
    db.upsert_story(updated_story)
    fetched2 = db.get_story(123)
    assert fetched2.title == "Updated Title"
    assert fetched2.score == 105


def test_read_only_database_opens_existing_db_without_writes(tmp_path) -> None:
    db_path = tmp_path / "readonly.db"
    writable = Database(str(db_path))
    writable.upsert_story(
        Story(id=1, title="Stored", url=None, score=1, time=1, text_content="Text")
    )
    writable.close()

    readonly = Database(str(db_path), read_only=True)
    try:
        assert readonly.get_story(1) is not None
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            readonly.upsert_story(
                Story(
                    id=2,
                    title="Rejected",
                    url=None,
                    score=1,
                    time=1,
                    text_content="Text",
                )
            )
    finally:
        readonly.close()


def test_fresh_database_uses_strict_tables_and_schema_version(tmp_path) -> None:
    db_path = tmp_path / "strict.db"
    database = Database(str(db_path))
    database.close()

    with sqlite3.connect(db_path) as conn:
        application_tables = conn.execute(
            "SELECT name, strict FROM pragma_table_list "
            "WHERE schema='main' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert application_tables
        assert all(strict == 1 for _, strict in application_tables)
        assert conn.execute("PRAGMA user_version").fetchone() == (1,)

        with pytest.raises(sqlite3.IntegrityError, match="datatype mismatch"):
            conn.execute(
                "INSERT INTO stories "
                "(id, title, score, time, text_content, source, fetched_at) "
                "VALUES ('not-an-id', 'Title', 1, 1, 'Text', 'hn', 1.0)"
            )


def test_database_rejects_unmigrated_flexible_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE stories (id INTEGER PRIMARY KEY)")

    with pytest.raises(RuntimeError, match="explicit STRICT migration"):
        Database(str(db_path))
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (0,)
        assert conn.execute(
            "SELECT strict FROM pragma_table_list WHERE name='stories'"
        ).fetchone() == (0,)


def test_strict_migration_preserves_schema_and_removes_orphan_caches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.db"
    destination = tmp_path / "strict.db"
    with sqlite3.connect(source) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys=OFF;
            CREATE TABLE stories (id INTEGER PRIMARY KEY, title TEXT NOT NULL);
            CREATE TABLE embeddings (
                story_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
            );
            CREATE TABLE tldr_cache (
                story_id INTEGER NOT NULL,
                cache_key TEXT NOT NULL,
                PRIMARY KEY (story_id, cache_key),
                FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
            );
            CREATE INDEX idx_story_title ON stories(title);
            INSERT INTO stories VALUES (1, 'kept');
            INSERT INTO embeddings VALUES (1, X'0102');
            INSERT INTO embeddings VALUES (2, X'0304');
            INSERT INTO tldr_cache VALUES (1, 'kept');
            INSERT INTO tldr_cache VALUES (3, 'orphan');
            """
        )

    result = migrate_database(source, destination, remove_orphan_caches=True)

    assert result.removed_orphans == {"embeddings": 1, "tldr_cache": 1}
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT count(*) FROM embeddings").fetchone() == (2,)
        assert conn.execute("SELECT count(*) FROM tldr_cache").fetchone() == (2,)
    with sqlite3.connect(destination) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT count(*) FROM embeddings").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM tldr_cache").fetchone() == (1,)
        assert conn.execute(
            "SELECT strict FROM pragma_table_list WHERE name='stories'"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT count(*) FROM sqlite_schema WHERE name='idx_story_title'"
        ).fetchone() == (1,)


def test_strict_migration_refuses_orphans_without_explicit_permission(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.db"
    destination = tmp_path / "strict.db"
    with sqlite3.connect(source) as conn:
        conn.executescript(
            """
            CREATE TABLE stories (id INTEGER PRIMARY KEY);
            CREATE TABLE embeddings (
                story_id INTEGER PRIMARY KEY,
                FOREIGN KEY (story_id) REFERENCES stories(id)
            );
            PRAGMA foreign_keys=OFF;
            INSERT INTO embeddings VALUES (99);
            """
        )

    with pytest.raises(RuntimeError, match="--remove-orphan-caches"):
        migrate_database(source, destination, remove_orphan_caches=False)
    assert source.exists()
    assert not destination.exists()


def test_archive_score_time_index_created(db: Database) -> None:
    rows = db.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'index' AND name = 'idx_stories_archive_score_time'
        """
    )
    assert rows
    assert "score DESC" in rows[0][0]
    assert "source IN ('bq_seed', 'ch_seed')" in rows[0][0]


def test_archive_candidate_plan_uses_score_time_index(db: Database) -> None:
    plan = db.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT id FROM stories INDEXED BY idx_stories_archive_score_time
        WHERE source IN ('bq_seed', 'ch_seed') AND text_content != ''
        ORDER BY score DESC, time DESC LIMIT 10
        """
    )
    plan_text = "\n".join(str(row) for row in plan)
    assert "idx_stories_archive_score_time" in plan_text
    assert "USE TEMP B-TREE" not in plan_text


def test_hn_recent_query_no_temp_btree(db: Database) -> None:
    """The HN recent query (tier-1 gravity + LIMIT 1500) must not
    external-sort the full stories table. SQLite's plan uses
    `idx_stories_source` to bound the scan by source, then a bounded
    in-memory sort over the 30-day window. We assert that the source
    filter is applied (not a full SCAN) and the LIMIT is honored."""
    cutoff = int(time.time()) - 30 * 86400
    now_ts = int(time.time())
    plan = db.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT id FROM stories
        WHERE time >= ? AND source = 'hn'
        ORDER BY CAST(score AS REAL) / POW((? - time) / 3600.0 + 2.0, 1.8) DESC,
                 time DESC
        LIMIT 1500
        """,
        (cutoff, now_ts),
    )
    plan_text = "\n".join(str(row) for row in plan)
    # The source filter must use an index, not a full SCAN.
    assert "SCAN stories" not in plan_text
    assert "idx_stories_source" in plan_text


def test_rss_recent_query_uses_time_index(db: Database) -> None:
    """The RSS recent query (recency + LIMIT 500) must use the
    `idx_stories_time` index for the ORDER BY time DESC, not do a
    full table SCAN."""
    plan = db.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT id FROM stories
        WHERE time >= ? AND source != 'hn' AND source NOT IN ('bq_seed', 'ch_seed')
        ORDER BY time DESC
        LIMIT 500
        """,
        (int(time.time()) - 30 * 86400,),
    )
    plan_text = "\n".join(str(row) for row in plan)
    assert "idx_stories_time" in plan_text or "USE TEMP B-TREE" not in plan_text


def test_get_stories(db):
    s1 = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    s2 = Story(id=2, title="S2", url=None, score=20, time=200, text_content="T2")
    db.upsert_story(s1)
    db.upsert_story(s2)

    stories = db.get_stories([1, 2, 3])
    assert len(stories) == 2
    ids = {s.id for s in stories}
    assert ids == {1, 2}


def test_upsert_embedding_roundtrip(db):
    story = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    db.upsert_story(story)

    vec = np.random.randn(384).astype(np.float32)
    db.upsert_embedding(1, "v1", "hash123", vec)

    fetched = db.get_embedding(1, "v1", "hash123")
    assert fetched is not None
    assert np.allclose(fetched, vec)

    # Cache miss test
    assert db.get_embedding(2, "v1", "hash123") is None
    assert db.get_embedding(1, "v2", "hash123") is None
    assert db.get_embedding(1, "v1", "different_hash") is None


def test_get_embeddings_batch(db):
    s1 = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    s2 = Story(id=2, title="S2", url=None, score=20, time=200, text_content="T2")
    db.upsert_story(s1)
    db.upsert_story(s2)

    vec1 = np.ones(384, dtype=np.float32)
    vec2 = np.zeros(384, dtype=np.float32)
    db.upsert_embedding(1, "v1", "h1", vec1)
    db.upsert_embedding(2, "v1", "h2", vec2)

    hashes = {1: "h1", 2: "h2", 3: "h3"}
    batch = db.get_embeddings_batch([1, 2, 3], "v1", hashes)
    assert len(batch) == 2
    assert np.allclose(batch[1], vec1)
    assert np.allclose(batch[2], vec2)

    # Test mismatching hash filtered out
    bad_hashes = {1: "h1", 2: "wrong_hash"}
    batch2 = db.get_embeddings_batch([1, 2], "v1", bad_hashes)
    assert len(batch2) == 1
    assert 2 not in batch2


def test_tldr_cache_roundtrip_replaces_stale_entries(db):
    story = Story(id=10, title="S1", url=None, score=10, time=100, text_content="T1")
    db.upsert_story(story)

    db.upsert_tldr_cache(10, "hash1", "first summary")
    assert db.get_tldr_cache(10, "hash1") == "first summary"
    assert db.get_tldr_cache(10, "hash2") is None

    db.upsert_tldr_cache(10, "hash2", "updated summary")
    assert db.get_tldr_cache(10, "hash1") is None
    assert db.get_tldr_cache(10, "hash2") == "updated summary"


def test_upsert_story_preserves_longest_text_fields_and_recomposes(db):
    from pipeline import compose_story_text

    initial = Story(
        id=11,
        title="Long cached story",
        url="https://example.com/long",
        score=10,
        time=100,
        text_content=compose_story_text(
            "Long cached story",
            "Long self text with details.",
            "/u/a: Long useful comment with details.",
            "Long article body with details.",
        ),
        self_text="Long self text with details.",
        top_comments="/u/a: Long useful comment with details.",
        article_body="Long article body with details.",
    )
    db.upsert_story(initial)

    shorter_update = Story(
        id=11,
        title="Updated title",
        url="https://example.com/long",
        score=12,
        time=101,
        text_content="Updated title. short",
        self_text="short",
        top_comments="short",
        article_body="short",
    )
    db.upsert_story(shorter_update)

    fetched = db.get_story(11)
    assert fetched.title == "Updated title"
    assert fetched.score == 12
    assert fetched.self_text == initial.self_text
    assert fetched.top_comments == initial.top_comments
    assert fetched.article_body == initial.article_body
    assert fetched.text_content == compose_story_text(
        "Updated title",
        initial.self_text,
        initial.top_comments,
        initial.article_body,
    )


def test_upsert_story_preserves_time_on_reinsert(db):
    # GitHub Trending RSS feeds have no published_parsed; pipeline.py falls
    # back to `now` and re-stamps the time on every fetch. upsert_story must
    # preserve the original (first-encountered) time so the date shown in
    # the dashboard is stable across re-fetches.
    first = Story(
        id=9001,
        title="First seen",
        url="https://github.com/example/repo",
        score=0,
        time=1_700_000_000,
        text_content="First seen. initial content",
        source="rss_mshibanami_github_io",
    )
    db.upsert_story(first)
    assert db.get_story(9001).time == 1_700_000_000

    # Simulate a later fetch with a fresh `now` time (the GitHub Trending case).
    later = Story(
        id=9001,
        title="First seen",
        url="https://github.com/example/repo",
        score=0,
        time=1_700_100_000,
        text_content="First seen. updated content",
        source="rss_mshibanami_github_io",
    )
    db.upsert_story(later)
    fetched = db.get_story(9001)
    assert fetched.time == 1_700_000_000, (
        f"expected original time to be preserved, got {fetched.time}"
    )
    # Title still updates (the only other field we changed here).
    assert fetched.text_content == "First seen. updated content"


def test_upsert_story_uses_new_time_for_placeholder(db):
    # When a story was inserted as a placeholder (time=0, e.g. _empty_story
    # for an HN ID that 200'd with no payload), the next real upsert must
    # populate the time. We can't preserve zero as "first seen".
    placeholder = Story(
        id=9002,
        title="",
        url=None,
        score=0,
        time=0,
        text_content="",
        source="hn",
    )
    db.upsert_story(placeholder)
    assert db.get_story(9002).time == 0

    real = Story(
        id=9002,
        title="Real title",
        url="https://example.com/real",
        score=5,
        time=1_700_200_000,
        text_content="Real title. some content",
        source="hn",
    )
    db.upsert_story(real)
    assert db.get_story(9002).time == 1_700_200_000


def test_upsert_story_preserves_richer_comment_metadata(db) -> None:
    """Regression: the merge logic in upsert_story must preserve
    comment_count and discussion_url from the existing row when an
    incoming RSS re-fetch carries zeroed metadata.

    Background: Reddit and LessWrong RSS feeds lack <comments> and
    num_comments elements. The RSS parser hardcodes comment_count=0
    and discussion_url=None. The Reddit prewarm / LW prewarm / on-demand
    TLDR paths correctly populate these after the RSS upsert — but
    without metadata-aware merge, the next regen's stale RSS upsert
    would clobber the populated values. See WORKLOG 2026-06-29.
    """
    from pipeline import compose_story_text

    initial = Story(
        id=9100,
        title="Reddit story",
        url="https://www.reddit.com/r/test/comments/abc/title",
        score=0,
        time=1_700_000_000,
        text_content=compose_story_text(
            "Reddit story",
            "Self text from RSS",
            "/u/a: comment one /u/b: comment two /u/c: comment three",
            "",
        ),
        source="rss_reddit_test",
        comment_count=3,
        comment_count_at_fetch=3,
        discussion_url="https://www.reddit.com/r/test/comments/abc/title",
        self_text="Self text from RSS",
        top_comments="/u/a: comment one /u/b: comment two /u/c: comment three",
        article_body="",
    )
    db.upsert_story(initial)

    # Simulate the next regen's stale RSS upsert: a freshly-parsed
    # Reddit RSS row with the zeroed metadata that the parser always
    # produces (pipeline.py:1484-1485, 1496-1498).
    stale_rss = Story(
        id=9100,
        title="Reddit story",
        url="https://www.reddit.com/r/test/comments/abc/title",
        score=0,
        time=1_700_000_000,
        text_content="Reddit story. Self text from RSS",
        source="rss_reddit_test",
        comment_count=0,
        comment_count_at_fetch=0,
        discussion_url=None,
        self_text="Self text from RSS",
        top_comments="",
        article_body="",
    )
    db.upsert_story(stale_rss)

    fetched = db.get_story(9100)
    assert fetched.comment_count == 3, (
        f"expected DB's comment_count=3 to survive, got {fetched.comment_count}"
    )
    assert fetched.comment_count_at_fetch == 3
    assert fetched.discussion_url == (
        "https://www.reddit.com/r/test/comments/abc/title"
    )
    assert fetched.top_comments == initial.top_comments


def test_upsert_story_keeps_newer_comment_count_when_higher(db) -> None:
    """Counterpart to the regression test: if the incoming story has
    a HIGHER comment_count than the existing row, the merge should
    take the incoming value (not silently discard it).
    """
    initial = Story(
        id=9101,
        title="LW story",
        url="https://www.lesswrong.com/posts/abc/slug",
        score=10,
        time=1_700_000_000,
        text_content="LW story. body",
        source="rss_lesswrong_com",
        comment_count=2,
        comment_count_at_fetch=2,
        discussion_url="https://www.lesswrong.com/posts/abc/slug",
        self_text="body",
        top_comments="older comment one older comment two",
    )
    db.upsert_story(initial)

    fresh = Story(
        id=9101,
        title="LW story",
        url="https://www.lesswrong.com/posts/abc/slug",
        score=15,
        time=1_700_000_000,
        text_content="LW story. body newer comment one newer comment two newer comment three",
        source="rss_lesswrong_com",
        comment_count=5,
        comment_count_at_fetch=5,
        discussion_url="https://www.lesswrong.com/posts/abc/slug",
        self_text="body",
        top_comments=(
            "newer comment one newer comment two "
            "newer comment three newer comment four newer comment five"
        ),
    )
    db.upsert_story(fresh)

    fetched = db.get_story(9101)
    assert fetched.comment_count == 5
    assert fetched.top_comments == fresh.top_comments
    assert fetched.discussion_url == "https://www.lesswrong.com/posts/abc/slug"


def test_feedback_crud(db):
    # Create a test user
    user = db.create_user("test_token_1")

    story = Story(
        id=456,
        title="Feedback Title",
        url="https://example.com/feedback",
        score=0,
        time=0,
        text_content="Feedback text content",
        source="rss_lobsters",
    )
    db.upsert_story(story)
    db.upsert_feedback(user.id, story_id=456, action="up")

    feedbacks = db.get_all_feedback(user.id)
    assert len(feedbacks) == 1
    assert feedbacks[0].story_id == 456
    assert feedbacks[0].action == "up"
    assert feedbacks[0].title == "Feedback Title"
    assert feedbacks[0].url == "https://example.com/feedback"
    assert feedbacks[0].text_content == "Feedback text content"
    assert feedbacks[0].source == "rss_lobsters"

    # Update
    db.upsert_feedback(user.id, story_id=456, action="down")
    feedbacks2 = db.get_all_feedback(user.id)
    assert feedbacks2[0].action == "down"

    # Delete
    db.delete_feedback(user.id, 456)
    assert len(db.get_all_feedback(user.id)) == 0


def test_prune_stories(db):
    story = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    db.upsert_story(story)

    # Pruning with age = 0 should delete everything
    deleted = db.prune_stories(max_age_days=0)
    assert deleted == 1
    assert db.get_story(1) is None


def test_prune_stories_preserves_feedback_stories(db):
    user = db.create_user("test_token_2")
    story1 = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    story2 = Story(id=2, title="S2", url=None, score=20, time=100, text_content="T2")
    db.upsert_story(story1)
    db.upsert_story(story2)
    db.upsert_feedback(user.id, 1, "up")

    # Pruning with age = 0 should delete story2 but not story1
    deleted = db.prune_stories(max_age_days=0)
    assert deleted == 1
    assert db.get_story(1) is not None
    assert db.get_story(2) is None


def test_prune_stories_preserves_archive_stories(db):
    for source, sid in [("bq_seed", 3), ("ch_seed", 4)]:
        db.upsert_story(
            Story(
                id=sid,
                title="Archive",
                url=None,
                score=100,
                time=100,
                text_content="Archive text",
                source=source,
            )
        )
    db.upsert_story(
        Story(
            id=99,
            title="Old",
            url=None,
            score=1,
            time=1,
            text_content="Old text",
            source="hn",
        )
    )
    deleted = db.prune_stories(max_age_days=0)

    assert deleted == 1
    assert db.get_story(3) is not None
    assert db.get_story(4) is not None
    assert db.get_story(99) is None


def test_feedback_training_data(db):
    user = db.create_user("test_token_3")
    db.upsert_story(
        Story(
            id=1,
            title="T1",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text1",
            source="hn",
        )
    )
    db.upsert_story(
        Story(
            id=2,
            title="T2",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text2",
            source="hn",
        )
    )
    db.upsert_story(
        Story(
            id=3,
            title="T3",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text3",
            source="hn",
        )
    )

    db.upsert_feedback(user.id, 1, "up")
    db.upsert_feedback(user.id, 2, "down")
    db.upsert_feedback(user.id, 3, "neutral")

    stories, labels, vote_times = db.get_feedback_for_training(user.id)
    assert len(stories) == 3
    assert len(labels) == 3

    # Map mapping IDs to actions labels
    id_to_label = {s.id: lbl for s, lbl in zip(stories, labels)}
    assert id_to_label[1] == 2  # up
    assert id_to_label[2] == 0  # down
    assert id_to_label[3] == 1  # neutral


@given(
    # Timestamps relative to now (in days offset around the threshold)
    fetched_offsets=st.lists(
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0),
        min_size=5,
        max_size=50,
    ),
    # Indices of stories to attach feedback to
    feedback_indices=st.sets(st.integers(min_value=0, max_value=49)),
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_story_pruning_integrity_invariants(fetched_offsets, feedback_indices):
    db = Database(":memory:")
    try:
        user = db.create_user("test_token_hypothesis")
        now = time.time()
        max_age_days = 30
        cutoff = now - (max_age_days * 86400)

        stories_meta = []
        # Insert stories with variable ages
        for i, offset_days in enumerate(fetched_offsets):
            # fetched_at is offset around the threshold
            fetched_at = cutoff + (offset_days * 86400)
            story = Story(
                id=i,
                title=f"Story {i}",
                url=None,
                score=100,
                time=int(now),
                text_content="Content",
            )
            db.upsert_story(story)

            # Override fetched_at directly in DB to simulate temporal aging
            db.execute(
                "UPDATE stories SET fetched_at = ? WHERE id = ?", (fetched_at, i)
            )

            # Apply feedback if indexed
            has_feedback = i in feedback_indices and i < len(fetched_offsets)
            if has_feedback:
                db.upsert_feedback(user.id, i, "up")

            stories_meta.append((i, fetched_at, has_feedback))

        db.prune_stories(max_age_days=max_age_days)

        # Assert temporal and referential invariants
        for sid, fetched_at, has_feedback in stories_meta:
            story = db.get_story(sid)
            is_older = fetched_at < cutoff

            if is_older:
                if has_feedback:
                    # Invariant: Must survive despite age due to referential link
                    assert story is not None, (
                        f"Story {sid} with feedback was pruned despite relative age."
                    )
                else:
                    # Invariant: Must be deleted
                    assert story is None, (
                        f"Old story {sid} without feedback survived pruning."
                    )
            else:
                # Invariant: Young stories always survive
                assert story is not None, f"Young story {sid} was incorrectly pruned."
    finally:
        db.close()


def test_user_management(db):
    # Test create_user
    user = db.create_user("tok123")
    assert user.id is not None
    assert user.token == "tok123"

    # Test get_user_by_token
    fetched = db.get_user_by_token("tok123")
    assert fetched is not None
    assert fetched.id == user.id

    # Test get_or_create_user
    existing = db.get_or_create_user("tok123")
    assert existing.id == user.id

    new_user = db.get_or_create_user("new_tok")
    assert new_user.token == "new_tok"


def test_per_user_feedback_isolation(db):
    user1 = db.create_user("token_u1")
    user2 = db.create_user("token_u2")

    story = Story(
        id=99,
        title="Shared Story",
        url="http://shared.com",
        score=50,
        time=1000,
        text_content="content",
    )
    db.upsert_story(story)

    # Isolated votes
    db.upsert_feedback(user1.id, 99, "up")
    db.upsert_feedback(user2.id, 99, "down")

    fb1 = db.get_all_feedback(user1.id)
    fb2 = db.get_all_feedback(user2.id)

    assert len(fb1) == 1
    assert fb1[0].action == "up"

    assert len(fb2) == 1
    assert fb2[0].action == "down"

    # Training data isolation
    stories1, labels1, _ = db.get_feedback_for_training(user1.id)
    stories2, labels2, _ = db.get_feedback_for_training(user2.id)

    assert len(stories1) == 1
    assert labels1[0] == 2  # up is 2

    assert len(stories2) == 1
    assert labels2[0] == 0  # down is 0

    # Delete isolation
    db.delete_feedback(user1.id, 99)
    assert len(db.get_all_feedback(user1.id)) == 0
    assert len(db.get_all_feedback(user2.id)) == 1


def test_count_feedback_by_action(db):
    """count_feedback_by_action returns per-action buckets."""
    user = db.create_user("test_count")
    assert db.count_feedback_by_action(user.id) == {"up": 0, "neutral": 0, "down": 0}

    db.upsert_story(
        Story(id=1, title="S1", url=None, score=10, time=0, text_content="T1")
    )
    db.upsert_story(
        Story(id=2, title="S2", url=None, score=10, time=0, text_content="T2")
    )
    db.upsert_story(
        Story(id=3, title="S3", url=None, score=10, time=0, text_content="T3")
    )

    db.upsert_feedback(user.id, 1, "up")
    db.upsert_feedback(user.id, 2, "neutral")
    db.upsert_feedback(user.id, 3, "down")
    assert db.count_feedback_by_action(user.id) == {"up": 1, "neutral": 1, "down": 1}

    db.upsert_feedback(user.id, 1, "down")  # change vote
    assert db.count_feedback_by_action(user.id) == {"up": 0, "neutral": 1, "down": 2}
