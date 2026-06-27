"""One-shot backfill: set self_text from text_content for RSS stories where self_text is empty.

Idempotent — safe to re-run (WHERE self_text = '' guard).
No destructive operations of any kind.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database
from pipeline import clean_text


def _strip_title_prefix(text_content: str, title: str) -> str | None:
    """Strip the title prefix from text_content.

    The old pipeline composed text_content as f'{title}. {snippet}' using
    the raw title (not cleaned). Returns the body text on success, or None
    if no prefix matches (meaning text_content is just the title stub).
    """
    for candidate in (title, clean_text(title)):
        prefix = f"{candidate}. "
        if text_content.startswith(prefix):
            return text_content[len(prefix) :]
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db = Database()

    with db._conn() as conn:
        rows = conn.execute(
            "SELECT id, title, text_content FROM stories "
            "WHERE source LIKE 'rss_%' AND self_text = '' AND text_content != ''"
        ).fetchall()

    # Use a dedicated connection for the UPDATE so we don't modify the
    # pooled connection's row_factory.
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        updated = 0
        warned = 0
        for story_id, title, text_content in rows:
            body = _strip_title_prefix(text_content, title)
            if body is not None:
                new_self_text = body
            else:
                new_self_text = text_content
                logging.warning(
                    "story_id=%s: title prefix not found in text_content; "
                    "using full text_content as best-effort fallback",
                    story_id,
                )
                warned += 1

            cursor = conn.execute(
                "UPDATE stories SET self_text = ? WHERE id = ? AND self_text = ''",
                (new_self_text, story_id),
            )
            if cursor.rowcount:
                updated += 1

        conn.commit()

    print(f"Backfill complete: {updated} updated, {warned} warned, {len(rows)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
