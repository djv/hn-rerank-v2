#!/usr/bin/env python3
"""Compare two users' votes: are the older profile's votes like the new one's?

For deciding whether (and which of) an old profile's votes should train a new
profile. Per user, and per month of the older user's votes: label shares,
HN share, share of archive stories (older than 30 days when voted), median
story age when voted, text length, top sources. Then agreement on stories
both voted on. Reads the DB read-only.

    uv run python scripts/compare_profiles.py --db SNAPSHOT --old 1 --new 151
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import is_hn_source  # noqa: E402

ACTIONS = ("up", "neutral", "down")


@dataclass(frozen=True)
class Vote:
    story_id: int
    action: str
    voted_at: float
    story_time: float
    source: str
    text_len: int


def load_votes(conn: sqlite3.Connection, user_id: int) -> list[Vote]:
    rows = conn.execute(
        """SELECT f.story_id, f.action, f.updated_at, s.time, s.source,
                  length(coalesce(s.text_content, ''))
           FROM feedback f JOIN stories s ON s.id = f.story_id
           WHERE f.user_id = ?""",
        (user_id,),
    ).fetchall()
    return [
        Vote(int(r[0]), r[1], float(r[2]), float(r[3]), r[4], int(r[5])) for r in rows
    ]


def summary(label: str, votes: list[Vote]) -> str:
    if not votes:
        return f"{label:18} (no votes)"
    n = len(votes)
    counts = Counter(v.action for v in votes)
    shares = " ".join(f"{a} {counts[a] / n:4.0%}" for a in ACTIONS)
    hn = sum(is_hn_source(v.source) for v in votes) / n
    ages = np.array([(v.voted_at - v.story_time) / 86400 for v in votes])
    archive = float((ages > 30).mean())
    length = int(np.median([v.text_len for v in votes]))
    top = ", ".join(
        f"{s} {c / n:.0%}" for s, c in Counter(v.source for v in votes).most_common(3)
    )
    return (
        f"{label:18} n={n:5}  {shares}  HN {hn:4.0%}  archive {archive:4.0%}  "
        f"age {np.median(ages):6.1f}d  len {length:5}  top: {top}"
    )


def up_share_by(votes: list[Vote], key: str) -> dict[str, tuple[int, float]]:
    groups: dict[str, list[Vote]] = {}
    for v in votes:
        if key == "hn":
            name = "HN" if is_hn_source(v.source) else "other"
        else:
            name = "archive" if v.voted_at - v.story_time > 30 * 86400 else "live"
        groups.setdefault(name, []).append(v)
    return {
        name: (len(g), sum(v.action == "up" for v in g) / len(g))
        for name, g in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--old", type=int, required=True)
    parser.add_argument("--new", type=int, required=True)
    args = parser.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    old, new = load_votes(conn, args.old), load_votes(conn, args.new)
    conn.close()

    print(summary(f"user {args.old}", old))
    months: dict[str, list[Vote]] = {}
    for v in old:
        month = datetime.fromtimestamp(v.voted_at, timezone.utc).strftime("%Y-%m")
        months.setdefault(month, []).append(v)
    for month, votes in sorted(months.items()):
        print(summary(f"  {month}", votes))
    print(summary(f"user {args.new}", new))
    for label, votes in ((f"user {args.old}", old), (f"user {args.new}", new)):
        for key in ("hn", "age"):
            parts = "  ".join(
                f"{name} n={n} up {share:.0%}"
                for name, (n, share) in up_share_by(votes, key).items()
            )
            print(f"{label:18} up share by {key}: {parts}")

    old_by_story = {v.story_id: v.action for v in old}
    both = [
        (old_by_story[v.story_id], v.action) for v in new if v.story_id in old_by_story
    ]
    agree = sum(a == b for a, b in both)
    print(f"voted by both: {len(both)}, same label {agree / max(len(both), 1):.0%}")
    matrix = Counter(both)
    print("old -> new   " + "  ".join(f"{b:>7}" for b in ACTIONS))
    for a in ACTIONS:
        print(f"{a:>10}   " + "  ".join(f"{matrix[(a, b)]:7}" for b in ACTIONS))


if __name__ == "__main__":
    main()
