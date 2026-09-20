#!/usr/bin/env python3
"""Add the interaction ledger to an existing STRICT hn-rewrite database."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import STRICT_SCHEMA_VERSION


def _require_stopped_service() -> None:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "hn_rewrite.service"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() == "active":
        raise RuntimeError("hn_rewrite.service must be stopped before migration")


def _backup_database(source: Path, backup: Path) -> None:
    if backup.exists():
        raise FileExistsError(f"Backup path already exists: {backup}")
    try:
        with (
            sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as source_conn,
            sqlite3.connect(backup) as backup_conn,
        ):
            source_conn.backup(backup_conn)
            result = backup_conn.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise RuntimeError(f"Backup integrity check failed: {result}")
    except Exception:
        backup.unlink(missing_ok=True)
        raise


def migrate_database(source: Path) -> Path:
    """Back up and migrate ``source`` in place, returning the backup path."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = source.with_name(f"{source.name}.pre_interaction_events_{timestamp}")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"Source integrity check failed: {integrity}")
        version_row = conn.execute("PRAGMA user_version").fetchone()
        version = int(version_row[0]) if version_row else 0
        if version not in {STRICT_SCHEMA_VERSION - 1, STRICT_SCHEMA_VERSION}:
            raise RuntimeError(
                f"Expected schema version {STRICT_SCHEMA_VERSION - 1} or "
                f"{STRICT_SCHEMA_VERSION}, found {version}"
            )

    _backup_database(source, backup)
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            conn.execute(
                """
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
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_events_user_time "
                "ON interaction_events(user_id, occurred_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_events_story_time "
                "ON interaction_events(story_id, occurred_at)"
            )
            conn.execute(f"PRAGMA user_version={STRICT_SCHEMA_VERSION}")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"Migrated database integrity check failed: {integrity}")
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, nargs="?", default=Path("hn_rewrite.db"))
    args = parser.parse_args()
    _require_stopped_service()
    backup = migrate_database(args.db)
    print(f"Migrated {args.db} to schema version {STRICT_SCHEMA_VERSION}")
    print(f"Retained backup at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
