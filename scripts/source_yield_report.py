#!/usr/bin/env python3
"""Per-source yield: stories fetched vs shown vs voted, to judge feeds.

For each ``stories.source`` it prints stories fetched in the window, how
many (user, story) pairs were shown (``impression`` events), opened, and
voted, and the share of shown pairs that were upvoted. HN is the baseline
to compare feeds against. Read-only: opens the DB with ``mode=ro``.

Usage:
    uv run python scripts/source_yield_report.py
    uv run python scripts/source_yield_report.py --since 2026-09-27 --user-id 1
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config  # noqa: E402

_QUERY = """
WITH
fetched AS (
    SELECT source, COUNT(*) AS n FROM stories WHERE time >= :since GROUP BY 1
),
shown AS (
    SELECT DISTINCT e.user_id, e.story_id, s.source
    FROM interaction_events e JOIN stories s ON s.id = e.story_id
    WHERE e.event_type = 'impression' AND e.occurred_at >= :since
      AND (:user_id IS NULL OR e.user_id = :user_id)
),
opened AS (
    SELECT DISTINCT e.user_id, e.story_id
    FROM interaction_events e
    WHERE e.event_type IN ('article_open', 'comments_open')
      AND e.occurred_at >= :since
),
votes AS (
    SELECT f.user_id, f.story_id, f.action, s.source
    FROM feedback f JOIN stories s ON s.id = f.story_id
    WHERE f.updated_at >= :since AND (:user_id IS NULL OR f.user_id = :user_id)
),
per_shown AS (
    SELECT sh.source,
           COUNT(*) AS shown,
           SUM(o.story_id IS NOT NULL) AS opened,
           SUM(v.action IS NOT NULL) AS voted,
           SUM(v.action = 'up') AS up_shown
    FROM shown sh
    LEFT JOIN opened o USING (user_id, story_id)
    LEFT JOIN votes v USING (user_id, story_id)
    GROUP BY 1
),
per_vote AS (
    SELECT source, SUM(action = 'up') AS up, SUM(action = 'neutral') AS neutral,
           SUM(action = 'down') AS down
    FROM votes GROUP BY 1
)
SELECT src.source,
       COALESCE(fetched.n, 0), COALESCE(ps.shown, 0), COALESCE(ps.opened, 0),
       COALESCE(ps.voted, 0), COALESCE(ps.up_shown, 0),
       COALESCE(pv.up, 0), COALESCE(pv.neutral, 0), COALESCE(pv.down, 0)
FROM (SELECT source FROM fetched UNION SELECT source FROM per_shown
      UNION SELECT source FROM per_vote) src
LEFT JOIN fetched ON fetched.source = src.source
LEFT JOIN per_shown ps ON ps.source = src.source
LEFT JOIN per_vote pv ON pv.source = src.source
"""


@dataclass(frozen=True)
class SourceYield:
    source: str
    fetched: int
    shown: int
    opened: int
    voted: int
    up_shown: int
    up: int
    neutral: int
    down: int

    @property
    def up_rate(self) -> float | None:
        """Share of shown stories that were upvoted."""
        return self.up_shown / self.shown if self.shown else None


def load_yields(
    conn: sqlite3.Connection, since: float, user_id: int | None
) -> list[SourceYield]:
    rows = conn.execute(_QUERY, {"since": since, "user_id": user_id}).fetchall()
    yields = [
        SourceYield(str(row[0]), *(int(value) for value in row[1:])) for row in rows
    ]
    yields.sort(key=lambda y: (y.shown, y.fetched), reverse=True)
    return yields


def _pct(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.0%}"


def render(yields: list[SourceYield], min_shown: int) -> str:
    header = (
        f"{'source':<40} {'fetched':>7} {'shown':>6} {'opened':>6} "
        f"{'voted':>6} {'up':>5} {'neu':>5} {'down':>5} {'up/shown':>8}"
    )
    lines = [header, "-" * len(header)]
    for y in yields:
        if y.shown < min_shown and y.fetched == 0:
            continue
        flag = "  <- low" if y.shown >= min_shown and (y.up_rate or 0) < 0.1 else ""
        lines.append(
            f"{y.source:<40} {y.fetched:>7} {y.shown:>6} {y.opened:>6} "
            f"{y.voted:>6} {y.up:>5} {y.neutral:>5} {y.down:>5} "
            f"{_pct(y.up_rate):>8}{flag}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-source yield: fetched vs shown vs voted."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--db", help="DB path (default: config db_path)")
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--days", type=float, default=30.0, help="window (days)")
    window.add_argument("--since", help="window start, YYYY-MM-DD (UTC)")
    parser.add_argument("--user-id", type=int, help="one user (default: all)")
    parser.add_argument(
        "--min-shown", type=int, default=10, help="flag sources below 10%% up/shown"
    )
    args = parser.parse_args()

    if args.since:
        since = (
            datetime.strptime(args.since, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    else:
        since = time.time() - args.days * 86400
    db_path = Path(args.db or Config.load(args.config).db_path).resolve()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        print(render(load_yields(conn, since, args.user_id), args.min_shown))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
