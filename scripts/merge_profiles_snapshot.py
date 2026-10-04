#!/usr/bin/env python3
"""Copy a DB snapshot and add one user whose votes are two profiles merged.

For offline evaluation only: a person who started a new profile (e.g. user
151 after user 1) can be evaluated as if both vote histories were one. The
merged user gets every vote of --new-user plus the --base-user votes on
stories the new profile never voted on (the new profile's vote wins), each
with its original timestamp. ``--variant ID:YYYY-MM-DD`` (repeatable) adds
more merged users to the same copy, each keeping only the --base-user votes
cast on or after that date (``ID`` alone keeps all of them); a ``:live``
suffix also drops votes on stories older than 30 days when voted (the
archive deck). The source is opened read-only; only the new
--output file is written, and it must not exist yet.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


def merge(
    source: Path,
    output: Path,
    base_user: int,
    new_user: int,
    variants: dict[int, tuple[float, bool]],
) -> dict[int, int]:
    """Write *output* with one merged user per ``variants`` entry (merged
    user id -> earliest base-user vote time kept, live stories only); vote
    counts."""
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite")
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(output)
    try:
        src.backup(dst)
    finally:
        src.close()
    with dst:
        for merged_user, (since, live) in variants.items():
            _add_merged_user(dst, base_user, new_user, merged_user, since, live)
    counts = {
        merged_user: int(
            dst.execute(
                "SELECT COUNT(*) FROM feedback WHERE user_id = ?", (merged_user,)
            ).fetchone()[0]
        )
        for merged_user in variants
    }
    dst.close()
    return counts


def _add_merged_user(
    dst: sqlite3.Connection,
    base_user: int,
    new_user: int,
    merged_user: int,
    since: float,
    live: bool,
) -> None:
    if dst.execute("SELECT 1 FROM users WHERE id = ?", (merged_user,)).fetchone():
        raise SystemExit(f"user {merged_user} already exists in the source")
    columns = {row[1] for row in dst.execute("PRAGMA table_info(users)")}
    values = {
        "id": merged_user,
        "token": f"merged-{base_user}-{new_user}-{merged_user}",
        "created_at": time.time(),
    }
    if "username" in columns:
        values["username"] = f"merged-{base_user}-{new_user}"
    dst.execute(
        f"INSERT INTO users ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})",
        tuple(values.values()),
    )
    dst.execute(
        """INSERT INTO feedback (user_id, story_id, action, updated_at)
           SELECT ?, story_id, action, updated_at FROM feedback WHERE user_id = ?""",
        (merged_user, new_user),
    )
    dst.execute(
        """INSERT INTO feedback (user_id, story_id, action, updated_at)
           SELECT ?, f.story_id, f.action, f.updated_at
           FROM feedback f JOIN stories s ON s.id = f.story_id
           WHERE f.user_id = ? AND f.updated_at >= ?
             AND (? = 0 OR f.updated_at - s.time <= 30 * 86400)
             AND f.story_id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)""",
        (merged_user, base_user, since, int(live), new_user),
    )


def _parse_variant(spec: str) -> tuple[int, tuple[float, bool]]:
    user, *rest = spec.split(":")
    live = "live" in rest
    days = [part for part in rest if part and part != "live"]
    since = (
        datetime.strptime(days[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        if days
        else 0.0
    )
    return int(user), (since, live)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-user", type=int, required=True)
    parser.add_argument("--new-user", type=int, required=True)
    parser.add_argument("--merged-user", type=int, default=900_001)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="ID[:YYYY-MM-DD][:live]",
        help="merged user keeping base votes from that UTC date, optionally live stories only (repeatable)",
    )
    args = parser.parse_args()
    variants = dict(_parse_variant(spec) for spec in args.variant) or {
        args.merged_user: (0.0, False)
    }
    counts = merge(args.source, args.output, args.base_user, args.new_user, variants)
    for user, count in counts.items():
        print(f"wrote {args.output}: user {user} has {count} votes")


if __name__ == "__main__":
    main()
