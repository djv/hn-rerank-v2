#!/usr/bin/env python3
"""Per-badge and per-feed yield: of the stories shown, how many were voted up.

Each (user, story) pair counts once, with the feed (``sort_mode``) and badge
kinds of its first impression in the window. A story carrying two badges
counts under both. Read-only: opens the DB with ``mode=ro``.

Usage:
    uv run python scripts/badge_yield_report.py --user-id 151 --since 2026-09-30
    uv run python scripts/badge_yield_report.py --db snapshot.db --user-id 151
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config  # noqa: E402

_QUERY = """
WITH first_shown AS (
    SELECT user_id, story_id, MIN(occurred_at) AS at
    FROM interaction_events
    WHERE event_type = 'impression' AND occurred_at >= :since
      AND (:user_id IS NULL OR user_id = :user_id)
    GROUP BY 1, 2
)
SELECT e.sort_mode, e.badges, s.source, f.action
FROM first_shown fs
JOIN interaction_events e
  ON e.user_id = fs.user_id AND e.story_id = fs.story_id
 AND e.occurred_at = fs.at AND e.event_type = 'impression'
JOIN stories s ON s.id = fs.story_id
LEFT JOIN feedback f
  ON f.user_id = fs.user_id AND f.story_id = fs.story_id
 AND f.updated_at >= :since
GROUP BY fs.user_id, fs.story_id
"""


@dataclass
class Yield:
    shown: int = 0
    up: int = 0
    neutral: int = 0
    down: int = 0

    def add(self, action: str | None) -> None:
        self.shown += 1
        if action == "up":
            self.up += 1
        elif action == "neutral":
            self.neutral += 1
        elif action == "down":
            self.down += 1

    @property
    def voted(self) -> int:
        return self.up + self.neutral + self.down


@dataclass
class Report:
    by_feed: dict[str, Yield] = field(default_factory=lambda: defaultdict(Yield))
    by_badge: dict[str, Yield] = field(default_factory=lambda: defaultdict(Yield))
    by_source: dict[str, Yield] = field(default_factory=lambda: defaultdict(Yield))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_report(conn: sqlite3.Connection, since: float, user_id: int | None) -> Report:
    report = Report()
    for sort_mode, badges, source, action in conn.execute(
        _QUERY, {"since": since, "user_id": user_id}
    ):
        report.by_feed[str(sort_mode)].add(action)
        kinds = [kind for kind in str(badges).split(",") if kind] or ["(none)"]
        for kind in kinds:
            report.by_badge[kind].add(action)
        family = "hn" if str(source) in {"hn", "hn_archive"} else "other"
        report.by_source[family].add(action)
    return report


def render_table(title: str, rows: dict[str, Yield]) -> str:
    header = (
        f"{title:<14} {'shown':>6} {'voted':>6} {'up':>4} {'neu':>4} "
        f"{'down':>5} {'up/shown':>8} {'95% CI':>12} {'down/voted':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, y in sorted(rows.items(), key=lambda item: -item[1].shown):
        low, high = wilson(y.up, y.shown)
        rate = f"{y.up / y.shown:.1%}" if y.shown else "-"
        down = f"{y.down / y.voted:.0%}" if y.voted else "-"
        lines.append(
            f"{name:<14} {y.shown:>6} {y.voted:>6} {y.up:>4} {y.neutral:>4} "
            f"{y.down:>5} {rate:>8} {f'{low:.0%}-{high:.0%}':>12} {down:>10}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-badge and per-feed yield of shown stories."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--db", help="DB path (default: config db_path)")
    parser.add_argument("--since", default="2026-09-30", help="YYYY-MM-DD (UTC)")
    parser.add_argument("--user-id", type=int, help="one user (default: all)")
    args = parser.parse_args()

    db_path = args.db or Config.load(args.config).db_path
    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        report = load_report(conn, since, args.user_id)
    finally:
        conn.close()
    print(render_table("feed", report.by_feed))
    print()
    print(render_table("badge", report.by_badge))
    print()
    print(render_table("source", report.by_source))


if __name__ == "__main__":
    main()
