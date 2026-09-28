#!/usr/bin/env python3
"""Copy a DB snapshot and add one user whose votes are two profiles merged.

For offline evaluation only: a person who started a new profile (e.g. user
151 after user 1) can be evaluated as if both vote histories were one. The
merged user gets every vote of --new-user plus the --base-user votes on
stories the new profile never voted on (the new profile's vote wins), each
with its original timestamp. The source is opened read-only; only the new
--output file is written, and it must not exist yet.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def merge(
    source: Path, output: Path, base_user: int, new_user: int, merged_user: int
) -> int:
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite")
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(output)
    try:
        src.backup(dst)
    finally:
        src.close()
    with dst:
        if dst.execute("SELECT 1 FROM users WHERE id = ?", (merged_user,)).fetchone():
            raise SystemExit(f"user {merged_user} already exists in {source}")
        columns = {row[1] for row in dst.execute("PRAGMA table_info(users)")}
        values = {
            "id": merged_user,
            "token": f"merged-{base_user}-{new_user}",
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
               SELECT ?, story_id, action, updated_at FROM feedback
               WHERE user_id = ?
                 AND story_id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)""",
            (merged_user, base_user, new_user),
        )
    count = dst.execute(
        "SELECT COUNT(*) FROM feedback WHERE user_id = ?", (merged_user,)
    ).fetchone()[0]
    dst.close()
    return int(count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-user", type=int, required=True)
    parser.add_argument("--new-user", type=int, required=True)
    parser.add_argument("--merged-user", type=int, default=900_001)
    args = parser.parse_args()
    count = merge(
        args.source, args.output, args.base_user, args.new_user, args.merged_user
    )
    print(f"wrote {args.output}: user {args.merged_user} has {count} votes")


if __name__ == "__main__":
    main()
