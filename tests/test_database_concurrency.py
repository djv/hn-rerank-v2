"""Concurrent story preservation regression found during project review."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch


from database import Database, Story
from pipeline import compose_story_text

if TYPE_CHECKING:
    # Match sqlite3's complete binding protocol in this transparent wrapper.
    from sqlite3 import _Parameters


class ConcurrentSnapshotLoss(AssertionError):
    """An older writer discarded a newer stored snapshot."""


def test_routine_upsert_preserves_a_concurrent_authoritative_snapshot(
    tmp_path: Path,
) -> None:
    """An older ingestion write cannot undo comments a refresh just fetched.

    Pause the routine writer immediately after reading its preservation
    snapshot, then let a second pooled SQLite connection save richer data.
    If the snapshot is transaction-protected, the fresh writer starts before
    releasing the stale one. Otherwise it must commit before that release;
    a timeout is a setup failure, never evidence that the race was fixed.
    """
    read_or_done = threading.Event()
    resume = threading.Event()
    fresh_done = threading.Event()
    fresh_attempted = threading.Event()
    protected_snapshot: list[bool] = []
    errors: list[BaseException] = []

    class InterleavingCursor(sqlite3.Cursor):
        def fetchone(self) -> tuple[object, ...] | None:
            row = super().fetchone()
            if (
                threading.current_thread().name == "stale-story-writer"
                and self.description is not None
                and self.description[0][0] == "self_text"
            ):
                protected_snapshot.append(self.connection.in_transaction)
                read_or_done.set()
                if not resume.wait(5):
                    raise RuntimeError("The stale writer was not resumed")
            return row

    class InterleavingConnection(sqlite3.Connection):
        def execute(
            self,
            sql: str,
            parameters: _Parameters = (),
            /,
        ) -> sqlite3.Cursor:
            if threading.current_thread().name == "fresh-story-writer":
                fresh_attempted.set()
            return self.cursor(factory=InterleavingCursor).execute(sql, parameters)

    with patch(
        "database.sqlite3.connect",
        partial(sqlite3.connect, factory=InterleavingConnection),
    ):
        db = Database(str(tmp_path / "concurrent-story.sqlite"))

    stale = Story(
        id=99991,
        title="Concurrent story",
        url=None,
        score=10,
        time=1,
        text_content="Old text",
        comment_count=1,
        comment_count_at_fetch=1,
        self_text="Old post",
        top_comments="Old comment",
        article_body="Old body",
    )
    fresh = replace(
        stale,
        self_text="A richer self post fetched during the refresh",
        top_comments="Fresh comments fetched just now, much longer",
        article_body="A richer article body fetched during the refresh",
        comment_count=100,
        comment_count_at_fetch=100,
    )
    fresh = replace(
        fresh,
        text_content=compose_story_text(
            fresh.title, fresh.self_text, fresh.top_comments, fresh.article_body
        ),
    )

    def stale_write() -> None:
        try:
            db.upsert_story(stale)
        except BaseException as exc:
            errors.append(exc)
        finally:
            read_or_done.set()

    def fresh_write() -> None:
        try:
            db.upsert_story(fresh, comments_authoritative=True)
        except BaseException as exc:
            errors.append(exc)
        finally:
            fresh_done.set()

    stale_thread = threading.Thread(target=stale_write, name="stale-story-writer")
    fresh_thread = threading.Thread(target=fresh_write, name="fresh-story-writer")
    try:
        db.upsert_story(stale)
        stale_thread.start()
        if not read_or_done.wait(5):
            raise RuntimeError("The stale writer did not reach its read")
        assert len(protected_snapshot) == 1, errors
        fresh_thread.start()
        if protected_snapshot[0]:
            if not fresh_attempted.wait(5):
                raise RuntimeError("The fresh writer did not attempt its transaction")
        elif not fresh_done.wait(5):
            raise RuntimeError("The fresh writer did not commit before stale release")
        resume.set()
        stale_thread.join(5)
        fresh_thread.join(5)
        if stale_thread.is_alive() or fresh_thread.is_alive():
            raise RuntimeError("A concurrent story writer did not finish")
        if errors:
            raise errors[0]
        stored = db.get_story(stale.id)
        assert stored is not None
        observed = (
            stored.self_text,
            stored.top_comments,
            stored.article_body,
            stored.comment_count,
            stored.comment_count_at_fetch,
            stored.text_content,
        )
        expected = (
            fresh.self_text,
            fresh.top_comments,
            fresh.article_body,
            fresh.comment_count,
            fresh.comment_count_at_fetch,
            fresh.text_content,
        )
        if observed != expected:
            raise ConcurrentSnapshotLoss(f"Expected {expected!r}, stored {observed!r}")
    finally:
        resume.set()
        if stale_thread.ident is not None:
            stale_thread.join(5)
        if fresh_thread.ident is not None:
            fresh_thread.join(5)
        if not stale_thread.is_alive() and not fresh_thread.is_alive():
            db.close()
