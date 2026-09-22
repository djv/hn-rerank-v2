#!/usr/bin/env python3
"""Read-only publication feedback audit; no model loads or API requests."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit


def hostname(url: str | None) -> str | None:
    """Conservative host identity for an audit, not a publisher classifier."""
    try:
        parsed = urlsplit(url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        return parsed.hostname.encode("idna").decode().lower().removeprefix("www.")
    except (ValueError, UnicodeError):
        return None


@dataclass(frozen=True)
class Vote:
    story_id: int
    title: str
    url: str
    source: str
    action: str
    updated_at: float


@dataclass(frozen=True)
class Audit:
    story_id: int
    title: str
    host: str | None
    source: str
    training_counts: dict[str, int]
    publication_counts: dict[str, int]
    publication_votes: list[Vote]
    duplicate_urls: dict[str, list[int]]
    text_chars: int
    self_text_chars: int
    article_chars: int
    text_prefix: str


def audit(path: Path, user_id: int, story_id: int) -> Audit:
    """Inspect saved votes and stored embedding input without modifying SQLite."""
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        story = conn.execute(
            "SELECT id,title,url,source,text_content,self_text,article_body "
            "FROM stories WHERE id=?",
            (story_id,),
        ).fetchone()
        if story is None:
            raise ValueError(f"Story {story_id} not found")
        if (
            conn.execute("SELECT id FROM users WHERE id=?", (user_id,)).fetchone()
            is None
        ):
            raise ValueError(f"User {user_id} not found")
        rows = conn.execute(
            "SELECT s.id,s.title,s.url,s.source,f.action,f.updated_at "
            "FROM feedback f LEFT JOIN stories s ON s.id=f.story_id "
            "WHERE f.user_id=? ORDER BY f.updated_at",
            (user_id,),
        ).fetchall()
        host = hostname(story["url"])
        votes = [
            Vote(
                int(r["id"]),
                r["title"],
                r["url"],
                r["source"],
                r["action"],
                float(r["updated_at"]),
            )
            for r in rows
            if host is not None and hostname(r["url"]) == host
        ]
        urls: dict[str, list[int]] = {}
        for vote in votes:
            # Exact URL audit only: does not claim complete semantic deduplication.
            urls.setdefault(vote.url, []).append(vote.story_id)
        return Audit(
            story_id,
            story["title"],
            host,
            story["source"],
            dict(Counter(r["action"] for r in rows)),
            dict(Counter(v.action for v in votes)),
            votes,
            {url: ids for url, ids in urls.items() if len(ids) > 1},
            len(story["text_content"] or ""),
            len(story["self_text"] or ""),
            len(story["article_body"] or ""),
            (story["text_content"] or "")[:400],
        )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--story-id", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(asdict(audit(args.db, args.user_id, args.story_id)), indent=2))


if __name__ == "__main__":
    main()
