"""Backfill discussion_url and comment_count for RSS stories whose
top_comments are populated but those fields were wiped by stale-cache
overwrites on every regen cycle.

Root cause: every regen re-runs `fetch_rss_feeds` (and Phase 1.5 for
Reddit), which calls `db.upsert_story(story)` with the freshly-parsed
story object — that object has `comment_count=0` and
`discussion_url=None` because RSS feeds don't carry these fields (Reddit
RSS has no <comments> element; LW feed has no comment metadata). The
prewarm paths (and the on-demand TLDR path) then correctly populate
`top_comments`, `comment_count`, and `discussion_url` on the story
row — but the very next regen's stale RSS upsert would clobber them,
because the merge logic in `upsert_story` only preserved text fields
(`self_text`, `top_comments`, `article_body`).

`upsert_story` is now also metadata-preserving, so the wipe shouldn't
recur; this script repairs the rows that were already clobbered.

For each affected row:
- `discussion_url` is set to the story's URL (Reddit and LW URLs are
  already the discussion URL).
- `comment_count` is recomputed locally from the existing `top_comments`
  text — Reddit prefixes each comment with `/u/<author>:`; LW is fetched
  raw text. We count non-empty blocks separated by the same delimiters
  the prewarm uses. For LW, comment_count can't be recovered perfectly
  from the cached top_comments (LW doesn't include author prefixes), so
  we fall back to a heuristic: count the number of paragraphs separated
  by `\n\n` if the structure is visible, else fall back to 1 (the LW
  prewarm may have returned 0 if GraphQL reported no comments even
  though the response body included text — treat that as "at least 1").
- `comment_count_at_fetch` is bumped to the new count if higher.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402

DB_PATH = ROOT / "hn_rewrite.db"

_REDDIT_AUTHOR_PREFIX = "/u/"


def _count_reddit_comments(top_comments: str) -> int:
    """Reddit prewarm prefixes each comment with `/u/<author>:`."""
    if not top_comments:
        return 0
    return top_comments.count(_REDDIT_AUTHOR_PREFIX) or 1


def _count_lesswrong_comments(top_comments: str) -> int:
    """LW prewarm joins comments with a single space (no prefix). The
    cached text is a single blob, so we can't perfectly recover the
    comment count from the cache. The LW GraphQL response may include
    text but report `commentCount=0` (e.g. a deleted post with stale
    comments), so we fall back to 1 if there's any text at all.
    """
    if not top_comments:
        return 0
    return 1


def main() -> None:
    db = Database(str(DB_PATH))
    with db.conn() as conn:
        rows = conn.execute(
            """
            SELECT id, url, source, comment_count, comment_count_at_fetch,
                   discussion_url, top_comments
            FROM stories
            WHERE (source LIKE 'rss_reddit_%' OR source = 'rss_lesswrong_com')
              AND top_comments != ''
              AND (discussion_url IS NULL OR comment_count IS NULL OR comment_count = 0)
            """
        ).fetchall()

    if not rows:
        print("No RSS stories need backfilling.")
        return

    print(f"Backfilling {len(rows)} RSS stories...")
    fixed = 0
    with db.conn() as conn:
        with conn:
            for row in rows:
                sid, url, source, comment_count, ccaf, disc_url, top_comments = row
                new_disc = disc_url or url
                if source.startswith("rss_reddit_"):
                    new_count = _count_reddit_comments(top_comments)
                elif source == "rss_lesswrong_com":
                    new_count = _count_lesswrong_comments(top_comments)
                else:
                    new_count = 1
                new_ccaf = max(ccaf or 0, new_count)
                conn.execute(
                    """
                    UPDATE stories
                    SET discussion_url = ?,
                        comment_count = ?,
                        comment_count_at_fetch = ?
                    WHERE id = ? AND (
                        discussion_url IS NULL
                        OR comment_count IS NULL
                        OR comment_count = 0
                    )
                    """,
                    (new_disc, new_count, new_ccaf, sid),
                )
                fixed += 1

    print(f"Backfilled {fixed} stories.")


if __name__ == "__main__":
    main()
