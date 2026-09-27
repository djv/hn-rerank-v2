#!/usr/bin/env python3
"""Rename pre-June-2026 story source labels to the current ``rss_*`` names.

Early RSS rows were stored as ``tildes``, ``digg``, ``reddit_programming``,
bare ``rss`` and so on. ``source_category_onehot`` puts most of those in no
category at all (and ``reddit_*`` under generic RSS rather than Reddit), so
the ranker trains on wrong source features for those votes. This maps each
legacy label to the name ``_rss_source_name`` gives its feed today.

Only ``stories.source`` changes; ids, text, feedback and embeddings stay.
Dry run by default; ``--apply`` writes in one transaction.

Usage:
    uv run python scripts/relabel_legacy_sources.py
    uv run python scripts/relabel_legacy_sources.py --apply
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config  # noqa: E402

# Labels whose feed is known; the article URL points elsewhere.
FIXED: dict[str, str] = {
    "tildes": "rss_tildes_net",
    "slashdot": "rss_slashdot_org",
    "rss_rss_slashdot_org": "rss_slashdot_org",
    "digg": "rss_digg_com",
    "github_trending": "rss_mshibanami_github_io",
    "haskell_discourse": "rss_discourse_haskell_org",
    "lesswrong": "rss_lesswrong_com",
}
# Labels resolved per row from the story URL.
BY_URL = ("rss", "rss_reddit_com")
_LEGACY_REDDIT = re.compile(r"^reddit_([a-z0-9_]+)$")
_SUBREDDIT = re.compile(r"^/r/([^/]+)/", re.IGNORECASE)


def new_label(source: str, url: str | None) -> str | None:
    """The current label for a legacy row, or None to leave it as is."""
    if source in FIXED:
        return FIXED[source]
    if match := _LEGACY_REDDIT.match(source):
        return f"rss_reddit_{match.group(1)}"
    if source not in BY_URL or not url:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host in {"reddit.com", "old.reddit.com"}:
        sub = _SUBREDDIT.match(parsed.path)
        if sub is None:
            return None
        return "rss_reddit_" + re.sub(r"[^a-z0-9_]+", "_", sub.group(1).lower())
    if source == "rss_reddit_com" or not host:
        return None
    return "rss_" + host.replace(".", "_")


def plan(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    legacy = [*FIXED, *BY_URL]
    marks = ",".join("?" for _ in legacy)
    rows = conn.execute(
        f"SELECT id, source, url FROM stories WHERE source IN ({marks}) "
        "OR source GLOB 'reddit_*'",
        legacy,
    ).fetchall()
    changes: list[tuple[int, str, str]] = []
    for story_id, source, url in rows:
        label = new_label(str(source), url)
        if label is not None and label != source:
            changes.append((int(story_id), str(source), label))
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename legacy story source labels to current rss_* names."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--db", help="DB path (default: config db_path)")
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    db_path = Path(args.db or Config.load(args.config).db_path).resolve()
    conn = sqlite3.connect(db_path)
    try:
        changes = plan(conn)
        for (old, new), n in sorted(
            Counter((old, new) for _, old, new in changes).items()
        ):
            print(f"{n:5d}  {old} -> {new}")
        print(f"{len(changes)} stories to relabel")
        if args.apply and changes:
            with conn:
                conn.executemany(
                    "UPDATE stories SET source = ? WHERE id = ? AND source = ?",
                    [(new, sid, old) for sid, old, new in changes],
                )
            print("applied")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
