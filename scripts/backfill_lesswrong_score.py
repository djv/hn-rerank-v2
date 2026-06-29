"""One-shot backfill: refresh ``score`` and ``comment_count`` for LessWrong
RSS stories from the LessWrong GraphQL endpoint.

Background: prior to 2026-06-29 the LW prewarm stored
``comment_count = len(comments fetched)`` (capped at 20) and never asked
the GraphQL endpoint for ``baseScore``, so ``score`` stayed at the
RSS-default 0. This script:

- Selects all rows with ``source = 'rss_lesswrong_com'``
- Calls ``_fetch_lesswrong_context(post_id)`` for each
- Updates only ``score`` and ``comment_count`` (does NOT touch
  ``self_text`` / ``top_comments`` / ``text_content`` — those are stable
  and re-embedding would waste compute)
- Uses ``max(existing, new)`` to avoid regressing values from a
  fresher run

Idempotent. Non-destructive (UPDATE only, no DELETE, no DROP).
Respects the AGENTS.md DB safety rules.

Run: ``uv run python scripts/backfill_lesswrong_score.py``
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database
from server import _extract_lesswrong_post_id, _fetch_lesswrong_context


async def _refresh_one(post_id: str) -> tuple[int, int] | None:
    """Fetch LW context for one post; return (score, comment_count) or None on failure."""
    ctx = await _fetch_lesswrong_context(post_id)
    if ctx is None:
        return None
    return ctx.score, ctx.comment_count


async def main_async() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db = Database()

    with db.conn() as conn:
        rows = conn.execute(
            "SELECT id, url, score, comment_count, comment_count_at_fetch "
            "FROM stories WHERE source = 'rss_lesswrong_com' AND url IS NOT NULL"
        ).fetchall()

    if not rows:
        print("No rss_lesswrong_com rows found; nothing to backfill.")
        return 0

    logging.info("Found %d LessWrong rows; refreshing score + comment_count", len(rows))

    with sqlite3.connect(db.db_path) as conn:
        conn.execute("PRAGMA busy_timeout=5000")
        updated = 0
        skipped = 0
        failed = 0
        for (
            story_id,
            url,
            current_score,
            current_comment_count,
            current_comment_count_at_fetch,
        ) in rows:
            post_id = _extract_lesswrong_post_id(url)
            if not post_id:
                logging.warning(
                    "story_id=%s: could not extract post_id from %r", story_id, url
                )
                skipped += 1
                continue

            result = await _refresh_one(post_id)
            if result is None:
                logging.warning(
                    "story_id=%s: GraphQL fetch failed for %s", story_id, post_id
                )
                failed += 1
                continue

            new_score, new_comment_count = result
            new_score = max(int(current_score or 0), new_score)
            new_comment_count = max(int(current_comment_count or 0), new_comment_count)
            new_comment_count_at_fetch = max(
                int(current_comment_count_at_fetch or 0), new_comment_count
            )

            if (
                new_score == int(current_score or 0)
                and new_comment_count == int(current_comment_count or 0)
                and new_comment_count_at_fetch
                == int(current_comment_count_at_fetch or 0)
            ):
                skipped += 1
                continue

            cursor = conn.execute(
                "UPDATE stories SET score = ?, comment_count = ?, "
                "comment_count_at_fetch = ? WHERE id = ?",
                (new_score, new_comment_count, new_comment_count_at_fetch, story_id),
            )
            if cursor.rowcount:
                updated += 1
                logging.info(
                    "story_id=%s: score %s -> %s, comment_count %s -> %s, "
                    "comment_count_at_fetch %s -> %s",
                    story_id,
                    current_score,
                    new_score,
                    current_comment_count,
                    new_comment_count,
                    current_comment_count_at_fetch,
                    new_comment_count_at_fetch,
                )

        conn.commit()

    print(
        f"Backfill complete: {updated} updated, {skipped} skipped, {failed} failed, "
        f"{len(rows)} total"
    )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
