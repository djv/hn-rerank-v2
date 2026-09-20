#!/usr/bin/env python3
"""Copy an hn-rewrite SQLite database into a fully STRICT replacement."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import STRICT_SCHEMA_VERSION  # noqa: E402


ORPHAN_CACHE_TABLES = ("embeddings", "tldr_cache")


@dataclass(frozen=True)
class MigrationResult:
    source_counts: dict[str, int]
    destination_counts: dict[str, int]
    removed_orphans: dict[str, int]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _strict_create_sql(sql: str) -> str:
    if re.search(r"\bSTRICT\s*$", sql, flags=re.IGNORECASE):
        return sql
    converted, count = re.subn(r"\)\s*$", ") STRICT", sql, count=1)
    if count != 1:
        raise ValueError(f"Cannot convert table definition to STRICT: {sql}")
    return converted


def _application_tables(conn: sqlite3.Connection, schema: str = "main") -> list[str]:
    rows = conn.execute(
        f"SELECT name FROM {_quote(schema)}.sqlite_schema "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _table_count(conn: sqlite3.Connection, table: str, schema: str = "main") -> int:
    row = conn.execute(
        f"SELECT count(*) FROM {_quote(schema)}.{_quote(table)}"
    ).fetchone()
    assert row is not None
    return int(row[0])


def _orphan_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(
        f"SELECT count(*) FROM source.{_quote(table)} AS cache "
        "LEFT JOIN source.stories AS story ON story.id = cache.story_id "
        "WHERE story.id IS NULL"
    ).fetchone()
    assert row is not None
    return int(row[0])


def migrate_database(
    source: Path,
    destination: Path,
    *,
    remove_orphan_caches: bool,
) -> MigrationResult:
    """Build and validate ``destination`` without modifying ``source``."""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source database does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if source == destination:
        raise ValueError("Source and destination must be different paths")

    source_uri = f"file:{source}?mode=ro"
    source_conn = sqlite3.connect(source_uri, uri=True)
    try:
        integrity = source_conn.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"Source integrity check failed: {integrity}")
        table_rows = source_conn.execute(
            "SELECT name, sql FROM sqlite_schema "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        source_counts = {
            str(name): _table_count(source_conn, str(name)) for name, _ in table_rows
        }
        schema_objects = source_conn.execute(
            "SELECT type, name, sql FROM sqlite_schema "
            "WHERE type IN ('index', 'trigger', 'view') AND sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'view' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, name"
        ).fetchall()
    finally:
        source_conn.close()

    destination.parent.mkdir(parents=True, exist_ok=True)
    dest_conn = sqlite3.connect(destination)
    try:
        dest_conn.execute("PRAGMA journal_mode=DELETE")
        dest_conn.execute("PRAGMA foreign_keys=OFF")
        dest_conn.execute("ATTACH DATABASE ? AS source", (source_uri,))
        removed_orphans: dict[str, int] = {}
        try:
            with dest_conn:
                for name, sql in table_rows:
                    if sql is None:
                        raise RuntimeError(f"Missing CREATE SQL for table {name}")
                    dest_conn.execute(_strict_create_sql(str(sql)))

                for name, _ in table_rows:
                    table = str(name)
                    columns = [
                        str(row[1])
                        for row in dest_conn.execute(
                            f"PRAGMA main.table_info({_quote(table)})"
                        ).fetchall()
                    ]
                    column_sql = ", ".join(_quote(column) for column in columns)
                    where_sql = ""
                    if table in ORPHAN_CACHE_TABLES:
                        orphan_count = _orphan_count(dest_conn, table)
                        removed_orphans[table] = orphan_count
                        if orphan_count and not remove_orphan_caches:
                            raise RuntimeError(
                                f"{table} contains {orphan_count} orphan rows; "
                                "rerun with --remove-orphan-caches"
                            )
                        where_sql = (
                            " WHERE EXISTS (SELECT 1 FROM source.stories "
                            f"WHERE source.stories.id = source.{_quote(table)}.story_id)"
                        )
                    dest_conn.execute(
                        f"INSERT INTO main.{_quote(table)} ({column_sql}) "
                        f"SELECT {column_sql} FROM source.{_quote(table)}{where_sql}"
                    )

                for _, _, sql in schema_objects:
                    assert sql is not None
                    dest_conn.execute(str(sql))
                dest_conn.execute(f"PRAGMA user_version={STRICT_SCHEMA_VERSION}")
        finally:
            dest_conn.execute("DETACH DATABASE source")

        dest_conn.execute("PRAGMA foreign_keys=ON")
        integrity = dest_conn.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError(f"Destination integrity check failed: {integrity}")
        foreign_key_rows = dest_conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_rows:
            raise RuntimeError(
                f"Destination has {len(foreign_key_rows)} foreign-key violations"
            )
        non_strict = [
            str(row[1])
            for row in dest_conn.execute("PRAGMA table_list").fetchall()
            if row[1] not in {"sqlite_schema", "sqlite_temp_schema"} and row[5] != 1
        ]
        if non_strict:
            raise RuntimeError(f"Destination has non-STRICT tables: {non_strict}")

        destination_counts = {
            table: _table_count(dest_conn, table) for table in source_counts
        }
        for table, source_count in source_counts.items():
            expected = source_count - removed_orphans.get(table, 0)
            if destination_counts[table] != expected:
                raise RuntimeError(
                    f"Row-count mismatch for {table}: "
                    f"expected {expected}, got {destination_counts[table]}"
                )
        dest_conn.execute("VACUUM")
    except Exception:
        dest_conn.close()
        destination.unlink(missing_ok=True)
        raise
    else:
        dest_conn.close()

    return MigrationResult(source_counts, destination_counts, removed_orphans)


def activate_database(source: Path, destination: Path) -> Path:
    """Atomically install a validated replacement and retain the original."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = source.with_name(f"{source.name}.pre_strict_{timestamp}")
    if backup.exists():
        raise FileExistsError(f"Backup path already exists: {backup}")
    os.replace(source, backup)
    try:
        os.replace(destination, source)
    except Exception:
        os.replace(backup, source)
        raise
    return backup


def _require_stopped_service() -> None:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "hn_rewrite.service"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() == "active":
        raise RuntimeError("hn_rewrite.service must be stopped before migration")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--remove-orphan-caches", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    _require_stopped_service()
    result = migrate_database(
        args.source,
        args.destination,
        remove_orphan_caches=args.remove_orphan_caches,
    )
    print(f"Created validated STRICT database: {args.destination}")
    for table in sorted(result.source_counts):
        removed = result.removed_orphans.get(table, 0)
        print(
            f"{table}: {result.source_counts[table]} -> "
            f"{result.destination_counts[table]} (removed orphans: {removed})"
        )
    if args.activate:
        backup = activate_database(args.source, args.destination)
        print(f"Activated {args.source}; retained original at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
