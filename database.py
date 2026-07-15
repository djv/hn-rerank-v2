from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Generator, NamedTuple, TypeAlias
from contextlib import contextmanager
import queue
import numpy as np
from numpy.typing import NDArray


Action: TypeAlias = Literal["up", "neutral", "down"]
HnDupeStatus: TypeAlias = Literal["canonical", "no_match", "retry"]
InteractionEventType: TypeAlias = Literal[
    "impression", "article_open", "comments_open", "dwell"
]
STRICT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Story:
    id: int
    title: str
    url: str | None
    score: int
    time: int
    text_content: str
    source: str = "hn"
    comment_count: int | None = None
    discussion_url: str | None = None
    comment_count_at_fetch: int = 0
    self_text: str = ""
    top_comments: str = ""
    article_body: str = ""


@dataclass(frozen=True)
class SeedStoryState:
    """Small story projection used to classify archive reconciliation rows."""

    id: int
    source: str
    has_top_comments: bool


@dataclass(frozen=True)
class FeedbackRecord:
    story_id: int
    action: Action
    title: str
    url: str | None
    text_content: str
    source: str
    updated_at: float


@dataclass(frozen=True)
class User:
    id: int
    token: str
    created_at: float


@dataclass(frozen=True)
class RankPerfSample:
    """One warm rerank's perf trace, ready to persist. `fields` is the full
    RankTrace.to_log_fields() dict (dynamic stage set) stored as JSON; the
    other columns are the always-queryable dimensions pulled out of it."""

    recorded_at: float
    user_id: int
    version: int
    rank_total_ms: float
    html_ms: float
    candidates: int
    feedback_total: int
    model_cache: str
    stories: int
    fields: dict[str, int | float | str]


@dataclass(frozen=True)
class InteractionEvent:
    """One explicit client interaction, normalized at the HTTP boundary."""

    event_id: str
    client_session_id: str
    user_id: int
    story_id: int
    event_type: InteractionEventType
    dashboard_version: int
    position: int
    sort_mode: str
    age_filter: str
    source_filter: str
    ranker_arm: str
    occurred_at: float
    duration_ms: int | None = None


class InteractionInsertResult(NamedTuple):
    """Outcome of an interaction-event batch insert."""

    inserted: int
    duplicates: int
    unknown: int


@dataclass(frozen=True)
class HnDupeResolution:
    source_story_id: int
    canonical_story_id: int | None
    status: HnDupeStatus
    checked_at: float
    next_check_at: float
    failure_count: int
    last_error: str


@dataclass(frozen=True)
class RedditFeedState:
    feed_url: str
    last_attempt_at: float
    last_success_at: float
    failure_count: int
    next_retry_at: float
    last_error: str
    item_count: int


class Database:
    def __init__(self, path: str = "hn_rewrite.db", *, read_only: bool = False) -> None:
        self.read_only = read_only
        db_path = Path(path)
        if not read_only:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._pool = queue.Queue()
        pool_size = 1 if self.db_path == ":memory:" else 5
        for _ in range(pool_size):
            if read_only:
                uri = f"file:{db_path.resolve()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._pool.put(conn)
        if not read_only:
            try:
                self._assert_schema_compatible()
                self._create_tables()
            except Exception:
                self.close()
                raise

    @contextmanager
    def conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def _assert_schema_compatible(self) -> None:
        with self.conn() as conn:
            tables = conn.execute(
                "SELECT name, strict FROM pragma_table_list "
                "WHERE schema='main' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if not tables:
                return
            non_strict = [str(name) for name, strict in tables if strict != 1]
            version_row = conn.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0]) if version_row else 0
            if non_strict or version != STRICT_SCHEMA_VERSION:
                raise RuntimeError(
                    "Database schema requires explicit STRICT migration; run "
                    "`uv run python scripts/migrate_db_to_strict.py` while the "
                    f"service is stopped (version={version}, non_strict={non_strict})"
                )

    def _create_tables(self) -> None:
        with self.conn() as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stories (
                        id             INTEGER PRIMARY KEY,
                        title          TEXT NOT NULL,
                        url            TEXT,
                        score          INTEGER NOT NULL DEFAULT 0,
                        time           INTEGER NOT NULL DEFAULT 0,
                        text_content   TEXT NOT NULL DEFAULT '',
                        source         TEXT NOT NULL DEFAULT 'hn',
                        comment_count  INTEGER,
                        discussion_url TEXT,
                        fetched_at     REAL NOT NULL,
                        comment_count_at_fetch INTEGER NOT NULL DEFAULT 0,
                        self_text      TEXT NOT NULL DEFAULT '',
                        top_comments   TEXT NOT NULL DEFAULT '',
                        article_body   TEXT NOT NULL DEFAULT ''
                    ) STRICT
                """)
                cursor = conn.execute("PRAGMA table_info(stories)")
                columns = {row[1] for row in cursor.fetchall()}
                if "comment_count_at_fetch" not in columns:
                    conn.execute(
                        "ALTER TABLE stories ADD COLUMN comment_count_at_fetch INTEGER NOT NULL DEFAULT 0"
                    )
                for col in ("self_text", "top_comments", "article_body"):
                    if col not in columns:
                        conn.execute(
                            f"ALTER TABLE stories ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                        )

                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stories_time ON stories(time)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stories_source ON stories(source)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stories_archive_score_time "
                    "ON stories(score DESC, time DESC) "
                    "WHERE source IN ('bq_seed', 'ch_seed') AND text_content != ''"
                )

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        story_id      INTEGER PRIMARY KEY,
                        model_version TEXT NOT NULL,
                        text_hash     TEXT NOT NULL DEFAULT '',
                        embedding     BLOB NOT NULL,
                        FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                    ) STRICT
                """)
                cursor = conn.execute("PRAGMA table_info(embeddings)")
                emb_columns = {row[1] for row in cursor.fetchall()}
                if "text_hash" not in emb_columns:
                    conn.execute(
                        "ALTER TABLE embeddings ADD COLUMN text_hash TEXT NOT NULL DEFAULT ''"
                    )

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tldr_cache (
                        story_id    INTEGER NOT NULL,
                        cache_key   TEXT NOT NULL,
                        tldr        TEXT NOT NULL,
                        created_at  REAL NOT NULL,
                        PRIMARY KEY (story_id, cache_key),
                        FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                    ) STRICT
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_tldr_cache_story ON tldr_cache(story_id)"
                )

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS article_fetch_failures (
                        story_id       INTEGER PRIMARY KEY,
                        url            TEXT NOT NULL,
                        failure_count  INTEGER NOT NULL DEFAULT 0,
                        last_status    INTEGER,
                        last_error     TEXT NOT NULL DEFAULT '',
                        permanent      INTEGER NOT NULL DEFAULT 0,
                        next_retry_at  REAL NOT NULL DEFAULT 0,
                        updated_at     REAL NOT NULL,
                        FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                    ) STRICT
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_article_fetch_failures_url "
                    "ON article_fetch_failures(url)"
                )

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rank_perf (
                        id              INTEGER PRIMARY KEY,
                        recorded_at     REAL NOT NULL,
                        user_id         INTEGER NOT NULL,
                        version         INTEGER NOT NULL,
                        rank_total_ms   REAL NOT NULL,
                        html_ms         REAL NOT NULL,
                        candidates      INTEGER NOT NULL,
                        feedback_total  INTEGER NOT NULL,
                        model_cache     TEXT NOT NULL,
                        stories         INTEGER NOT NULL,
                        fields_json     TEXT NOT NULL
                    ) STRICT
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rank_perf_recorded_at "
                    "ON rank_perf(recorded_at)"
                )

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS interaction_events (
                        event_id           TEXT PRIMARY KEY,
                        client_session_id  TEXT NOT NULL,
                        user_id             INTEGER NOT NULL,
                        story_id            INTEGER NOT NULL,
                        event_type         TEXT NOT NULL CHECK(
                            event_type IN ('impression', 'article_open',
                                           'comments_open', 'dwell')
                        ),
                        dashboard_version  INTEGER NOT NULL CHECK(dashboard_version >= 0),
                        position            INTEGER NOT NULL CHECK(position >= 0),
                        sort_mode           TEXT NOT NULL,
                        age_filter          TEXT NOT NULL,
                        source_filter       TEXT NOT NULL,
                        ranker_arm          TEXT NOT NULL,
                        occurred_at         REAL NOT NULL,
                        duration_ms         INTEGER CHECK(
                            duration_ms IS NULL OR duration_ms >= 0
                        ),
                        received_at         REAL NOT NULL
                    ) STRICT
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interaction_events_user_time "
                    "ON interaction_events(user_id, occurred_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interaction_events_story_time "
                    "ON interaction_events(story_id, occurred_at)"
                )

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS hn_dupe_resolutions (
                        source_story_id INTEGER PRIMARY KEY,
                        canonical_story_id INTEGER,
                        status TEXT NOT NULL CHECK(status IN ('canonical', 'no_match', 'retry')),
                        checked_at REAL NOT NULL,
                        next_check_at REAL NOT NULL,
                        failure_count INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT NOT NULL DEFAULT '',
                        FOREIGN KEY (source_story_id) REFERENCES stories(id) ON DELETE CASCADE,
                        FOREIGN KEY (canonical_story_id) REFERENCES stories(id) ON DELETE SET NULL
                    ) STRICT
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_hn_dupe_resolutions_due "
                    "ON hn_dupe_resolutions(next_check_at)"
                )

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reddit_feed_state (
                        feed_url TEXT PRIMARY KEY,
                        last_attempt_at REAL NOT NULL,
                        last_success_at REAL NOT NULL DEFAULT 0,
                        failure_count INTEGER NOT NULL DEFAULT 0,
                        next_retry_at REAL NOT NULL DEFAULT 0,
                        last_error TEXT NOT NULL DEFAULT '',
                        item_count INTEGER NOT NULL DEFAULT 0
                    ) STRICT
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reddit_feed_items (
                        feed_url TEXT NOT NULL,
                        story_id INTEGER NOT NULL,
                        position INTEGER NOT NULL,
                        observed_at REAL NOT NULL,
                        PRIMARY KEY (feed_url, story_id),
                        FOREIGN KEY (feed_url) REFERENCES reddit_feed_state(feed_url)
                            ON DELETE CASCADE,
                        FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                    ) STRICT
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reddit_circuit_state (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        consecutive_429 INTEGER NOT NULL,
                        retry_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    ) STRICT
                """)

                # Run migration of article_cache to stories table if article_cache exists
                tbl_cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='article_cache'"
                )
                if tbl_cursor.fetchone():
                    logging.info("Migrating article_cache records to stories table...")
                    conn.execute("""
                        UPDATE stories SET article_body = (
                          SELECT article_text FROM article_cache 
                          WHERE article_cache.story_id = stories.id
                        ) WHERE EXISTS (
                          SELECT 1 FROM article_cache WHERE article_cache.story_id = stories.id
                        )
                    """)
                    conn.execute("DROP TABLE article_cache")

                # Users table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id          INTEGER PRIMARY KEY,
                        token       TEXT UNIQUE NOT NULL,
                        created_at  REAL NOT NULL
                    ) STRICT
                """)

                # Multi-user feedback migration
                tbl_cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
                )
                if tbl_cursor.fetchone():
                    cursor = conn.execute("PRAGMA table_info(feedback)")
                    fb_columns = {row[1] for row in cursor.fetchall()}
                    if "user_id" not in fb_columns:
                        logging.info("Migrating feedback table to multi-user schema...")
                        conn.execute("""
                            CREATE TABLE feedback_new (
                                user_id     INTEGER NOT NULL,
                                story_id    INTEGER NOT NULL,
                                action      TEXT NOT NULL CHECK(action IN ('up', 'neutral', 'down')),
                                updated_at  REAL NOT NULL,
                                PRIMARY KEY (user_id, story_id),
                                FOREIGN KEY (story_id) REFERENCES stories(id)
                            ) STRICT
                        """)
                        conn.execute(
                            "INSERT OR IGNORE INTO users (id, token, created_at) VALUES (1, 'default', ?)",
                            (time.time(),),
                        )
                        conn.execute("""
                            INSERT INTO feedback_new (user_id, story_id, action, updated_at)
                            SELECT 1, story_id, action, updated_at FROM feedback
                        """)
                        conn.execute("DROP TABLE feedback")
                        conn.execute("ALTER TABLE feedback_new RENAME TO feedback")
                    elif "title" in fb_columns:
                        # Legacy migration from denormalized feedback
                        logging.info(
                            "Migrating feedback table schema to normalized version..."
                        )
                        conn.execute("""
                            CREATE TABLE feedback_new (
                                user_id     INTEGER NOT NULL DEFAULT 1,
                                story_id    INTEGER NOT NULL,
                                action      TEXT NOT NULL CHECK(action IN ('up', 'neutral', 'down')),
                                updated_at  REAL NOT NULL,
                                PRIMARY KEY (user_id, story_id),
                                FOREIGN KEY (story_id) REFERENCES stories(id)
                            ) STRICT
                        """)
                        conn.execute(
                            "INSERT OR IGNORE INTO users (id, token, created_at) VALUES (1, 'default', ?)",
                            (time.time(),),
                        )
                        conn.execute("""
                            INSERT OR IGNORE INTO feedback_new (user_id, story_id, action, updated_at)
                            SELECT 1, story_id, action, updated_at FROM feedback
                        """)
                        conn.execute("DROP TABLE feedback")
                        conn.execute("ALTER TABLE feedback_new RENAME TO feedback")
                else:
                    conn.execute(
                        "INSERT OR IGNORE INTO users (id, token, created_at) VALUES (1, 'default', ?)",
                        (time.time(),),
                    )
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS feedback (
                            user_id     INTEGER NOT NULL,
                            story_id    INTEGER NOT NULL,
                            action      TEXT NOT NULL CHECK(action IN ('up', 'neutral', 'down')),
                            updated_at  REAL NOT NULL,
                            PRIMARY KEY (user_id, story_id),
                            FOREIGN KEY (story_id) REFERENCES stories(id)
                        ) STRICT
                    """)

                conn.execute(f"PRAGMA user_version={STRICT_SCHEMA_VERSION}")

    def close(self) -> None:
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break

    # Stories
    def upsert_story(self, story: Story) -> None:
        with self.conn() as conn:
            # Check if the story already exists and has longer cached content
            cursor = conn.execute(
                "SELECT self_text, top_comments, article_body, "
                "comment_count, comment_count_at_fetch, discussion_url "
                "FROM stories WHERE id = ?",
                (story.id,),
            )
            row = cursor.fetchone()
            if row:
                db_self, db_comments, db_body = row[0] or "", row[1] or "", row[2] or ""

                # Keep the longest available version of each text field
                final_self = (
                    story.self_text if len(story.self_text) >= len(db_self) else db_self
                )
                final_comments = (
                    story.top_comments
                    if len(story.top_comments) >= len(db_comments)
                    else db_comments
                )
                final_body = (
                    story.article_body
                    if len(story.article_body) >= len(db_body)
                    else db_body
                )

                # Preserve richer comment_count and discussion_url from the
                # database. Re-ingestion from RSS can produce a story with
                # comment_count=0 and discussion_url=None even when the
                # prewarm path or on-demand TLDR has since populated those
                # fields; an unconditional UPSERT would clobber them on
                # every regen cycle.
                final_comment_count = (
                    story.comment_count
                    if (story.comment_count or 0) >= (row[3] or 0)
                    else row[3]
                )
                final_ccaf = max(story.comment_count_at_fetch or 0, row[4] or 0)
                final_discussion_url = story.discussion_url or row[5]

                # Recompose or merge metadata if any field changed
                recomposed = (
                    final_self != story.self_text
                    or final_comments != story.top_comments
                    or final_body != story.article_body
                )
                metadata_changed = (
                    final_comment_count != story.comment_count
                    or final_discussion_url != story.discussion_url
                    or final_ccaf != story.comment_count_at_fetch
                )
                if recomposed or metadata_changed:
                    new_text = story.text_content
                    if recomposed:
                        from pipeline import compose_story_text

                        new_text = compose_story_text(
                            story.title, final_self, final_comments, final_body
                        )
                    story = replace(
                        story,
                        self_text=final_self,
                        top_comments=final_comments,
                        article_body=final_body,
                        text_content=new_text,
                        comment_count=final_comment_count,
                        comment_count_at_fetch=final_ccaf,
                        discussion_url=final_discussion_url,
                    )

            with conn:
                conn.execute(
                    """
                    INSERT INTO stories (
                        id, title, url, score, time, text_content, source,
                        comment_count, discussion_url, fetched_at,
                        comment_count_at_fetch, self_text, top_comments,
                        article_body
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        url=excluded.url,
                        score=excluded.score,
                        time = CASE
                            WHEN stories.time > 0 THEN stories.time
                            WHEN excluded.time > 0 THEN excluded.time
                            ELSE 0
                        END,
                        text_content=excluded.text_content,
                        source=excluded.source,
                        comment_count=excluded.comment_count,
                        discussion_url=excluded.discussion_url,
                        fetched_at=excluded.fetched_at,
                        comment_count_at_fetch=excluded.comment_count_at_fetch,
                        self_text=excluded.self_text,
                        top_comments=excluded.top_comments,
                        article_body=excluded.article_body
                    """,
                    (
                        story.id,
                        story.title,
                        story.url,
                        story.score,
                        story.time,
                        story.text_content,
                        story.source,
                        story.comment_count,
                        story.discussion_url,
                        time.time(),
                        story.comment_count_at_fetch,
                        story.self_text,
                        story.top_comments,
                        story.article_body,
                    ),
                )

    @staticmethod
    def _row_to_story(row: tuple) -> Story:
        return Story(
            id=row[0],
            title=row[1],
            url=row[2],
            score=row[3],
            time=row[4],
            text_content=row[5] or "",
            source=row[6],
            comment_count=row[7],
            discussion_url=row[8],
            comment_count_at_fetch=row[9],
            self_text=row[10] or "",
            top_comments=row[11] or "",
            article_body=row[12] or "",
        )

    def get_story(self, story_id: int) -> Story | None:
        with self.conn() as conn:
            cursor = conn.execute(
                """
                SELECT id, title, url, score, time, text_content, source, comment_count, discussion_url,
                       comment_count_at_fetch, self_text, top_comments, article_body
                FROM stories WHERE id = ?
                """,
                (story_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_story(row)

    def get_stories(self, ids: list[int]) -> list[Story]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT id, title, url, score, time, text_content, source, comment_count, discussion_url,
                   comment_count_at_fetch, self_text, top_comments, article_body
            FROM stories WHERE id IN ({placeholders})
        """
        with self.conn() as conn:
            cursor = conn.execute(query, ids)
            return [self._row_to_story(row) for row in cursor.fetchall()]

    def get_seed_story_states(self, ids: list[int]) -> dict[int, SeedStoryState]:
        """Return only the fields archive reconciliation needs for ``ids``."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.conn() as conn:
            rows = conn.execute(
                f"SELECT id, source, top_comments != '' FROM stories "
                f"WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return {
            int(row[0]): SeedStoryState(
                id=int(row[0]), source=str(row[1]), has_top_comments=bool(row[2])
            )
            for row in rows
        }

    # HN explicit duplicate canonicalization cache
    @staticmethod
    def _row_to_hn_dupe_resolution(row: tuple) -> HnDupeResolution:
        return HnDupeResolution(
            source_story_id=int(row[0]), canonical_story_id=row[1], status=row[2],
            checked_at=float(row[3]), next_check_at=float(row[4]),
            failure_count=int(row[5]), last_error=str(row[6] or ""),
        )

    def get_hn_dupe_resolutions(
        self, source_story_ids: list[int], *, now: float | None = None
    ) -> dict[int, HnDupeResolution]:
        if not source_story_ids:
            return {}
        now = time.time() if now is None else now
        placeholders = ",".join("?" for _ in source_story_ids)
        with self.conn() as conn:
            rows = conn.execute(
                f"SELECT source_story_id, canonical_story_id, status, checked_at, "
                f"next_check_at, failure_count, last_error FROM hn_dupe_resolutions "
                f"WHERE source_story_id IN ({placeholders}) AND next_check_at > ?",
                [*source_story_ids, now],
            ).fetchall()
        return {row[0]: self._row_to_hn_dupe_resolution(row) for row in rows}

    def get_due_hn_dupe_candidate_ids(
        self, candidate_ids: list[int], *, limit: int = 250, now: float | None = None
    ) -> list[int]:
        """Select current low-comment HN candidates fairly: unseen, then oldest due."""
        if not candidate_ids or limit <= 0:
            return []
        now = time.time() if now is None else now
        placeholders = ",".join("?" for _ in candidate_ids)
        with self.conn() as conn:
            rows = conn.execute(
                f"SELECT s.id FROM stories s LEFT JOIN hn_dupe_resolutions r "
                f"ON r.source_story_id = s.id WHERE s.id IN ({placeholders}) "
                "AND s.source = 'hn' AND COALESCE(s.comment_count, s.comment_count_at_fetch, 0) <= 8 "
                "AND (r.source_story_id IS NULL OR r.next_check_at <= ?) "
                "ORDER BY CASE WHEN r.source_story_id IS NULL THEN 0 ELSE 1 END, "
                "COALESCE(r.next_check_at, 0), s.id LIMIT ?",
                [*candidate_ids, now, limit],
            ).fetchall()
        return [int(row[0]) for row in rows]

    def upsert_hn_dupe_resolution(
        self, resolution: HnDupeResolution
    ) -> None:
        with self.conn() as conn:
            with conn:
                conn.execute(
                    "INSERT INTO hn_dupe_resolutions (source_story_id, canonical_story_id, status, "
                    "checked_at, next_check_at, failure_count, last_error) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(source_story_id) DO UPDATE SET canonical_story_id=excluded.canonical_story_id, "
                    "status=excluded.status, checked_at=excluded.checked_at, next_check_at=excluded.next_check_at, "
                    "failure_count=excluded.failure_count, last_error=excluded.last_error",
                    (resolution.source_story_id, resolution.canonical_story_id, resolution.status,
                     resolution.checked_at, resolution.next_check_at, resolution.failure_count,
                     resolution.last_error[:500]),
                )

    def prune_stories(self, max_age_days: int = 60) -> int:
        cutoff = time.time() - (max_age_days * 86400)
        with self.conn() as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM stories WHERE fetched_at < ? "
                    "AND source NOT IN ('bq_seed', 'ch_seed') "
                    "AND id NOT IN (SELECT story_id FROM feedback)",
                    (cutoff,),
                )
                return cursor.rowcount

    def upsert_embedding(
        self,
        story_id: int,
        model_version: str,
        text_hash: str,
        vec: NDArray[np.float32],
    ) -> None:
        blob = vec.astype(np.float32).tobytes()
        with self.conn() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO embeddings (story_id, model_version, text_hash, embedding)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(story_id) DO UPDATE SET
                        model_version=excluded.model_version,
                        text_hash=excluded.text_hash,
                        embedding=excluded.embedding
                    """,
                    (story_id, model_version, text_hash, blob),
                )

    def get_embedding(
        self, story_id: int, model_version: str, text_hash: str
    ) -> NDArray[np.float32] | None:
        with self.conn() as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE story_id = ? AND model_version = ? AND text_hash = ?",
                (story_id, model_version, text_hash),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return np.frombuffer(row[0], dtype=np.float32)

    def get_embeddings_batch(
        self, ids: list[int], model_version: str, hashes: dict[int, str]
    ) -> dict[int, NDArray[np.float32]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT story_id, text_hash, embedding FROM embeddings
            WHERE model_version = ? AND story_id IN ({placeholders})
        """
        params = [model_version] + ids
        with self.conn() as conn:
            cursor = conn.execute(query, params)
            res = {}
            for row in cursor.fetchall():
                sid, h, blob = row[0], row[1], row[2]
                if hashes.get(sid) == h:
                    res[sid] = np.frombuffer(blob, dtype=np.float32)
            return res

    # TLDR cache
    def get_tldr_cache(self, story_id: int, cache_key: str) -> str | None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT tldr FROM tldr_cache WHERE story_id = ? AND cache_key = ?",
                (story_id, cache_key),
            ).fetchone()
            return row[0] if row else None

    def get_any_tldr_for_story(self, story_id: int) -> str | None:
        """Return the cached TLDR for a story regardless of cache key.

        `upsert_tldr_cache` deletes any prior row before inserting, so at
        most one row exists per story_id; this is a stale-tolerant fallback
        for when story content changed after the TLDR was generated (see
        `_handle_flask_tldr_detail` fallback in server.py).
        """
        with self.conn() as conn:
            row = conn.execute(
                "SELECT tldr FROM tldr_cache WHERE story_id = ?",
                (story_id,),
            ).fetchone()
            return row[0] if row else None

    def get_tldr_cache_keys(self, story_ids: list[int]) -> dict[int, str]:
        """Bulk lookup of cache_key for stories that have a cached TLDR."""
        if not story_ids:
            return {}
        with self.conn() as conn:
            placeholders = ",".join("?" for _ in story_ids)
            rows = conn.execute(
                f"SELECT story_id, cache_key FROM tldr_cache WHERE story_id IN ({placeholders})",
                story_ids,
            ).fetchall()
            return {row[0]: row[1] for row in rows}

    def upsert_tldr_cache(self, story_id: int, cache_key: str, tldr: str) -> None:
        with self.conn() as conn:
            with conn:
                conn.execute("DELETE FROM tldr_cache WHERE story_id = ?", (story_id,))
                conn.execute(
                    """
                    INSERT INTO tldr_cache (story_id, cache_key, tldr, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (story_id, cache_key, tldr, time.time()),
                )

    # Feedback
    def upsert_feedback(
        self,
        user_id: int,
        story_id: int,
        action: Action,
    ) -> None:
        with self.conn() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO feedback (user_id, story_id, action, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, story_id) DO UPDATE SET
                        action=excluded.action,
                        updated_at=excluded.updated_at
                    """,
                    (user_id, story_id, action, time.time()),
                )

    def get_all_feedback(self, user_id: int | None = None) -> list[FeedbackRecord]:
        with self.conn() as conn:
            if user_id is not None:
                cursor = conn.execute(
                    """
                    SELECT f.story_id, f.action, s.title, s.url, s.text_content, s.source, f.updated_at
                    FROM feedback f
                    LEFT JOIN stories s ON s.id = f.story_id
                    WHERE f.user_id = ?
                    ORDER BY f.updated_at DESC
                    """,
                    (user_id,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT f.story_id, f.action, s.title, s.url, s.text_content, s.source, f.updated_at
                    FROM feedback f
                    LEFT JOIN stories s ON s.id = f.story_id
                    ORDER BY f.updated_at DESC
                    """
                )
            return [
                FeedbackRecord(
                    story_id=row[0],
                    action=row[1],
                    title=row[2] or "",
                    url=row[3],
                    text_content=row[4] or "",
                    source=row[5] or "hn",
                    updated_at=row[6],
                )
                for row in cursor.fetchall()
            ]

    def get_feedback_stories(
        self, user_id: int, actions: tuple[str, ...]
    ) -> list[Story]:
        """Return Story objects for feedback rows matching *actions*.

        Score is forced to -1 so these stories always lose same-source
        tiebreaks in dedup — they participate only as suppressors, never
        as survivors.
        """
        if not actions:
            return []
        placeholders = ",".join("?" for _ in actions)
        with self.conn() as conn:
            cursor = conn.execute(
                f"SELECT DISTINCT s.id, s.title, s.url, s.time, "
                f"       s.text_content, s.source "
                f"FROM stories s "
                f"JOIN feedback f ON f.story_id = s.id "
                f"WHERE f.user_id = ? AND f.action IN ({placeholders}) "
                f"ORDER BY s.id",
                (user_id, *actions),
            )
            return [
                Story(
                    id=row[0],
                    title=row[1] or "",
                    url=row[2],
                    score=-1,  # always lose same-source tiebreaks
                    time=row[3] or 0,
                    text_content=row[4] or "",
                    source=row[5] or "hn",
                )
                for row in cursor.fetchall()
            ]

    def count_feedback_by_action(self, user_id: int) -> dict[str, int]:
        """Return per-action counts for a user. Unknown actions are ignored."""
        rows = self.execute(
            "SELECT action, COUNT(*) FROM feedback WHERE user_id = ? GROUP BY action",
            (user_id,),
        )
        counts: dict[str, int] = {"up": 0, "neutral": 0, "down": 0}
        for action, n in rows:
            if action in counts:
                counts[action] = n
        return counts

    def delete_feedback(self, user_id: int, story_id: int) -> bool:
        """Delete the feedback row for (user_id, story_id).

        Returns True if a row was deleted, False if none existed.
        """
        with self.conn() as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM feedback WHERE user_id = ? AND story_id = ?",
                    (user_id, story_id),
                )
                return cursor.rowcount > 0

    def get_feedback_for_training(
        self, user_id: int | None = None
    ) -> tuple[list[Story], list[int], list[float]]:
        with self.conn() as conn:
            if user_id is not None:
                cursor = conn.execute(
                    """
                    SELECT f.story_id, f.action, s.title, s.url, s.text_content, s.source,
                           f.updated_at, COALESCE(s.score, 0), COALESCE(s.time, 0),
                           COALESCE(s.comment_count, 0), COALESCE(s.self_text, ''),
                           COALESCE(s.top_comments, ''), COALESCE(s.article_body, '')
                    FROM feedback f
                    LEFT JOIN stories s ON s.id = f.story_id
                    WHERE f.user_id = ?
                    """,
                    (user_id,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT f.story_id, f.action, s.title, s.url, s.text_content, s.source,
                           f.updated_at, COALESCE(s.score, 0), COALESCE(s.time, 0),
                           COALESCE(s.comment_count, 0), COALESCE(s.self_text, ''),
                           COALESCE(s.top_comments, ''), COALESCE(s.article_body, '')
                    FROM feedback f
                    LEFT JOIN stories s ON s.id = f.story_id
                    """
                )
            stories: list[Story] = []
            labels: list[int] = []
            vote_times: list[float] = []
            action_to_label = {"down": 0, "neutral": 1, "up": 2}
            for (
                story_id,
                action,
                title,
                url,
                text_content,
                source,
                updated_at,
                score,
                story_time,
                comment_count,
                self_text,
                top_comments,
                article_body,
            ) in cursor.fetchall():
                if action not in action_to_label:
                    continue

                stories.append(
                    Story(
                        id=story_id,
                        title=title or "",
                        url=url,
                        score=score,
                        time=story_time,
                        text_content=text_content or "",
                        source=source or "hn",
                        comment_count=comment_count,
                        self_text=self_text,
                        top_comments=top_comments,
                        article_body=article_body,
                    )
                )
                labels.append(action_to_label[action])
                vote_times.append(updated_at)
            return stories, labels, vote_times

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self.conn() as conn:
            with conn:
                cursor = conn.execute(sql, params)
                return cursor.fetchall()

    def record_reddit_feed_success(
        self, feed_url: str, story_ids: list[int], now_ts: float
    ) -> None:
        """Atomically replace one feed's ordered successful snapshot."""
        with self.conn() as conn:
            with conn:
                conn.execute(
                    "INSERT INTO reddit_feed_state "
                    "(feed_url, last_attempt_at, last_success_at, failure_count, "
                    "next_retry_at, last_error, item_count) VALUES (?, ?, ?, 0, 0, '', ?) "
                    "ON CONFLICT(feed_url) DO UPDATE SET last_attempt_at=excluded.last_attempt_at, "
                    "last_success_at=excluded.last_success_at, failure_count=0, "
                    "next_retry_at=0, last_error='', item_count=excluded.item_count",
                    (feed_url, now_ts, now_ts, len(story_ids)),
                )
                conn.execute(
                    "DELETE FROM reddit_feed_items WHERE feed_url = ?", (feed_url,)
                )
                conn.executemany(
                    "INSERT INTO reddit_feed_items "
                    "(feed_url, story_id, position, observed_at) VALUES (?, ?, ?, ?)",
                    [
                        (feed_url, story_id, position, now_ts)
                        for position, story_id in enumerate(story_ids)
                    ],
                )

    def record_reddit_feed_failure(
        self, feed_url: str, error: str, now_ts: float
    ) -> None:
        with self.conn() as conn:
            with conn:
                row = conn.execute(
                    "SELECT failure_count FROM reddit_feed_state WHERE feed_url = ?",
                    (feed_url,),
                ).fetchone()
                failures = (int(row[0]) if row else 0) + 1
                retry = now_ts + min(300.0 * (2 ** (failures - 1)), 14400.0)
                conn.execute(
                    "INSERT INTO reddit_feed_state "
                    "(feed_url, last_attempt_at, failure_count, next_retry_at, last_error) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(feed_url) DO UPDATE SET "
                    "last_attempt_at=excluded.last_attempt_at, failure_count=excluded.failure_count, "
                    "next_retry_at=excluded.next_retry_at, last_error=excluded.last_error",
                    (feed_url, now_ts, failures, retry, error[:500]),
                )

    def get_reddit_feed_state(self, feed_url: str) -> RedditFeedState | None:
        rows = self.execute(
            "SELECT feed_url, last_attempt_at, last_success_at, failure_count, "
            "next_retry_at, last_error, item_count FROM reddit_feed_state WHERE feed_url = ?",
            (feed_url,),
        )
        return RedditFeedState(*rows[0]) if rows else None

    def save_reddit_circuit_state(
        self, consecutive_429: int, retry_at: float, now_ts: float
    ) -> None:
        self.execute(
            "INSERT INTO reddit_circuit_state "
            "(singleton, consecutive_429, retry_at, updated_at) VALUES (1, ?, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET consecutive_429=excluded.consecutive_429, "
            "retry_at=excluded.retry_at, updated_at=excluded.updated_at",
            (consecutive_429, retry_at, now_ts),
        )

    def get_reddit_circuit_state(self) -> tuple[int, float] | None:
        rows = self.execute(
            "SELECT consecutive_429, retry_at FROM reddit_circuit_state WHERE singleton = 1"
        )
        return (int(rows[0][0]), float(rows[0][1])) if rows else None

    # Rank perf telemetry
    def insert_rank_perf(self, sample: RankPerfSample) -> None:
        with self.conn() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO rank_perf (
                        recorded_at, user_id, version, rank_total_ms, html_ms,
                        candidates, feedback_total, model_cache, stories, fields_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample.recorded_at,
                        sample.user_id,
                        sample.version,
                        sample.rank_total_ms,
                        sample.html_ms,
                        sample.candidates,
                        sample.feedback_total,
                        sample.model_cache,
                        sample.stories,
                        json.dumps(sample.fields),
                    ),
                )

    def insert_interaction_events(
        self, events: list[InteractionEvent]
    ) -> InteractionInsertResult:
        """Insert a batch of events and return ``(inserted, duplicates, unknown)``.

        Event IDs are the idempotency key.  Story IDs are checked only for
        new events; replaying an event remains a no-op even if its story has
        since been pruned from the story corpus.  New events referencing a
        story ID absent from ``stories`` are skipped and counted as
        ``unknown`` — they never fail the rest of the batch.
        """
        if not events:
            return InteractionInsertResult(0, 0, 0)
        with self.conn() as conn:
            with conn:
                event_ids = [event.event_id for event in events]
                placeholders = ",".join("?" for _ in event_ids)
                existing_rows = conn.execute(
                    "SELECT event_id FROM interaction_events "
                    f"WHERE event_id IN ({placeholders})",
                    event_ids,
                ).fetchall()
                existing_ids = {str(row[0]) for row in existing_rows}
                new_story_ids = {
                    event.story_id
                    for event in events
                    if event.event_id not in existing_ids
                }
                unknown_story_ids: set[int] = set()
                if new_story_ids:
                    story_placeholders = ",".join("?" for _ in new_story_ids)
                    story_rows = conn.execute(
                        "SELECT id FROM stories "
                        f"WHERE id IN ({story_placeholders})",
                        tuple(new_story_ids),
                    ).fetchall()
                    known_story_ids = {int(row[0]) for row in story_rows}
                    unknown_story_ids = new_story_ids - known_story_ids

                received_at = time.time()
                inserted = 0
                unknown = 0
                for event in events:
                    if (
                        event.event_id not in existing_ids
                        and event.story_id in unknown_story_ids
                    ):
                        unknown += 1
                        continue
                    cursor = conn.execute(
                        """
                        INSERT INTO interaction_events (
                            event_id, client_session_id, user_id, story_id,
                            event_type, dashboard_version, position, sort_mode,
                            age_filter, source_filter, ranker_arm, occurred_at,
                            duration_ms, received_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(event_id) DO NOTHING
                        """,
                        (
                            event.event_id,
                            event.client_session_id,
                            event.user_id,
                            event.story_id,
                            event.event_type,
                            event.dashboard_version,
                            event.position,
                            event.sort_mode,
                            event.age_filter,
                            event.source_filter,
                            event.ranker_arm,
                            event.occurred_at,
                            event.duration_ms,
                            received_at,
                        ),
                    )
                    inserted += cursor.rowcount
                return InteractionInsertResult(
                    inserted, len(events) - inserted - unknown, unknown
                )

    # Article fetch failure memory
    def get_article_fetch_failure(self, story_id: int) -> dict | None:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT story_id, url, failure_count, last_status, last_error,
                       permanent, next_retry_at, updated_at
                FROM article_fetch_failures
                WHERE story_id = ?
                """,
                (story_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "story_id": row[0],
                "url": row[1],
                "failure_count": row[2],
                "last_status": row[3],
                "last_error": row[4],
                "permanent": bool(row[5]),
                "next_retry_at": row[6],
                "updated_at": row[7],
            }

    def record_article_fetch_failure(
        self,
        story_id: int,
        url: str,
        *,
        status: int | None = None,
        error: str | None = None,
        permanent: bool = False,
        next_retry_at: float | None = None,
    ) -> None:
        now = time.time()
        if next_retry_at is None:
            next_retry_at = now
        with self.conn() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO article_fetch_failures (
                        story_id, url, failure_count, last_status, last_error,
                        permanent, next_retry_at, updated_at
                    )
                    VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT(story_id) DO UPDATE SET
                        url=excluded.url,
                        failure_count=article_fetch_failures.failure_count + 1,
                        last_status=excluded.last_status,
                        last_error=excluded.last_error,
                        permanent=excluded.permanent,
                        next_retry_at=excluded.next_retry_at,
                        updated_at=excluded.updated_at
                    """,
                    (
                        story_id,
                        url,
                        status,
                        (error or "")[:500],
                        1 if permanent else 0,
                        next_retry_at,
                        now,
                    ),
                )

    def clear_article_fetch_failure(self, story_id: int) -> None:
        with self.conn() as conn:
            with conn:
                conn.execute(
                    "DELETE FROM article_fetch_failures WHERE story_id = ?",
                    (story_id,),
                )

    # Users
    def create_user(self, token: str) -> User:
        with self.conn() as conn:
            with conn:
                now = time.time()
                conn.execute(
                    "INSERT INTO users (token, created_at) VALUES (?, ?)",
                    (token, now),
                )
                row = conn.execute(
                    "SELECT id FROM users WHERE token = ?", (token,)
                ).fetchone()
                return User(id=row[0], token=token, created_at=now)

    def get_user_by_token(self, token: str) -> User | None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT id, token, created_at FROM users WHERE token = ?", (token,)
            ).fetchone()
            if row:
                return User(id=row[0], token=row[1], created_at=row[2])
            return None

    def get_or_create_user(self, token: str) -> User:
        user = self.get_user_by_token(token)
        if user:
            return user
        return self.create_user(token)
