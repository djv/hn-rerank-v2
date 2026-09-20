"""Drop tables that no longer have any code reference (dead schema residue).

Refuses to run if any target table is non-empty -- this script is only safe
for tables nothing writes to anymore. Dry-run by default; pass --apply to
actually drop.

Usage:
    uv run python scripts/drop_dead_tables.py            # dry run
    uv run python scripts/drop_dead_tables.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402

# Verified via `rg` across all *.py: none of database.py::_create_tables,
# server.py, pipeline/*, or scripts/* read or write these tables. They are
# residue from a superseded schema.
DEAD_TABLES = ("reading_events", "user_signals", "muted_channels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually drop the tables")
    args = parser.parse_args()

    db = Database()
    with db.conn() as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        counts: dict[str, int] = {}
        for table in DEAD_TABLES:
            if table not in existing:
                print(f"{table}: not present, skipping")
                continue
            counts[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608 -- fixed allowlist above
            ).fetchone()[0]
            print(f"{table}: {counts[table]} rows")

        non_empty = [t for t, n in counts.items() if n > 0]
        if non_empty:
            print(
                f"\nRefusing to drop non-empty table(s): {non_empty}. "
                "This script only drops tables with zero rows."
            )
            sys.exit(1)

        targets = list(counts)
        if not targets:
            print("\nNothing to drop.")
            return

        if not args.apply:
            print(f"\nDry run. Would drop: {targets}. Re-run with --apply.")
            return

        with conn:
            for table in targets:
                conn.execute(f"DROP TABLE {table}")  # noqa: S608 -- fixed allowlist above
        print(f"\nDropped: {targets}")

        (integrity,) = conn.execute("PRAGMA integrity_check").fetchone()
        print(f"PRAGMA integrity_check: {integrity}")
        if integrity != "ok":
            print("WARNING: integrity check did not return 'ok'.")
            sys.exit(1)


if __name__ == "__main__":
    main()
