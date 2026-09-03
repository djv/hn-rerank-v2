"""Purge tldr_cache rows poisoned by a salvaged article-only/discussion-only
half that was cached as if it were a complete TLDR.

Root cause (see WORKLOG.md, 2026-09-03): `generate_detailed_tldr` fires the
article and discussion prompts concurrently; before that fix, a 429 on
either one made the surviving half get cached under `TldrResult(kind="ok")`
indistinguishable from a full result. `upsert_tldr_cache` deletes any prior
row before inserting, so a later salvage could also destroy a previously
complete summary. The fix (limiter spacing + `TldrResult.partial`) stops new
poisoning; this script repairs rows already stuck, e.g.
https://news.ycombinator.com/item?id=49541519.

A row is "poisoned" only when it is *currently live* -- its stored
`cache_key` matches what `server._tldr_cache_key` computes from the story's
*current* fields (i.e. it's exactly what `/api/tldr-detail` would look up
today, not a stale leftover from before the story was enriched) -- and its
`tldr` covers exactly one of the `### Article` / `### Discussion` section
headers despite the story currently having material for both halves. This
deliberately excludes stale rows whose cache_key no longer matches: those
already regenerate on next view via the normal cache-miss path, and in the
meantime `get_any_tldr_for_story`'s stale fallback may still be serving
them usefully (e.g. under quota denial) -- indiscriminately deleting every
single-section row would erase that safety net for stories that were
legitimately comments-only or article-only when they were first generated.

Never touches `stories` or `feedback`; purged rows simply regenerate on the
next `/api/tldr-detail` request or prefetch pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config  # noqa: E402
from server import SELF_TEXT_PROMPT_MIN_CHARS, _tldr_cache_key  # noqa: E402

_ARTICLE_MARKER = "### Article"
_DISCUSSION_MARKER = "### Discussion"


def _has_article_section(self_text: str, article_body: str) -> bool:
    """Mirrors generate_detailed_tldr's article_section condition exactly
    (server.py) -- a short self_text alone (below SELF_TEXT_PROMPT_MIN_CHARS,
    e.g. a typical Ask HN body) does NOT count, so those stories correctly
    take the single-source discussion-only path and must not be flagged."""
    return bool(self_text and len(self_text) >= SELF_TEXT_PROMPT_MIN_CHARS) or bool(
        article_body
    )


def find_poisoned_rows(db: Database) -> list[tuple[int, str, str]]:
    """Returns (story_id, cache_key, title) for currently-live tldr_cache
    rows salvaged to only one of the two sections despite the story now
    having material for both -- i.e. generate_detailed_tldr would take the
    dual-prompt branch today, but the cached result covers only one half."""
    candidates = db.execute(
        """
        SELECT t.story_id, t.cache_key, t.tldr, s.title,
               s.self_text, s.top_comments, s.article_body
        FROM tldr_cache t
        JOIN stories s ON s.id = t.story_id
        WHERE s.top_comments != ''
          AND ((t.tldr LIKE '%' || ? || '%') != (t.tldr LIKE '%' || ? || '%'))
        """,
        (_ARTICLE_MARKER, _DISCUSSION_MARKER),
    )

    poisoned: list[tuple[int, str, str]] = []
    for story_id, cache_key, tldr, title, self_text, top_comments, article_body in candidates:
        self_text = self_text or ""
        article_body = article_body or ""
        if not _has_article_section(self_text, article_body):
            continue  # legitimately discussion-only, not a salvage
        current_key = _tldr_cache_key(
            title=title,
            self_text=self_text,
            top_comments=top_comments or "",
            article_body=article_body,
        )
        if cache_key == current_key:
            poisoned.append((int(story_id), str(cache_key), str(title)))
    return poisoned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the poisoned rows. Default is dry-run (list only).",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    parser.add_argument(
        "--config", default="config.toml", help="Path to config.toml (default: %(default)s)."
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    db = Database(config.db_path)

    poisoned = find_poisoned_rows(db)
    if not poisoned:
        print("No poisoned tldr_cache rows found.")
        return

    print(f"Found {len(poisoned)} poisoned tldr_cache row(s):")
    for story_id, _cache_key, title in poisoned[:20]:
        print(f"  id={story_id}  {title[:70]}")
    if len(poisoned) > 20:
        print(f"  ... ({len(poisoned) - 20} more)")

    if not args.apply:
        print("\nDry run — pass --apply to delete these rows.")
        return

    if not args.yes:
        ans = input(f"\nDelete {len(poisoned)} tldr_cache row(s)? [y/N] ")
        if ans.lower() != "y":
            print("Aborted.")
            sys.exit(1)

    with db.conn() as conn:
        with conn:
            cursor = conn.executemany(
                "DELETE FROM tldr_cache WHERE story_id = ? AND cache_key = ?",
                [(story_id, cache_key) for story_id, cache_key, _title in poisoned],
            )
    print(f"Deleted {cursor.rowcount} tldr_cache row(s). They will regenerate on next view.")


if __name__ == "__main__":
    main()
