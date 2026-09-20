"""Backfill top_comments for HN stories that missed comment prewarm.

Root cause (see WORKLOG.md, 2026-07-26): `_build_comments_bulk_query`
joined against the full comments table once per tree level, which
reliably exceeded play.clickhouse.com's query memory limit
(Code: 241, MEMORY_LIMIT_EXCEEDED) from 2026-07-23 onward. Every regen's
`prewarm_top_stories` call failed outright, so `stories.top_comments`
stayed empty for every HN story fetched after that point — and because
the `/api/tldr-detail` refresh gate itself required `top_comments` to be
non-empty (fixed separately in server.py), those stories could never
recover on their own. `ch_client.query_comments_bulk` now walks the
`kids` arrays level-by-level instead of joining, which fixes new
fetches; this script repairs the stories already stuck.

Reuses `pipeline.enrichment.prewarm_top_stories` — the same function the
live regen path calls — so comment selection, text composition, and
embedding recomputation stay identical to production behavior.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config, Embedder  # noqa: E402
from pipeline.enrichment import prewarm_top_stories  # noqa: E402

_CHUNK_SIZE = 100


def find_stuck_hn_story_ids(db: Database, limit: int | None) -> list[int]:
    """HN stories with comments on HN but nothing in top_comments."""
    query = """
        SELECT id FROM stories
        WHERE source = 'hn' AND comment_count > 0 AND top_comments = ''
        ORDER BY score DESC
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return [int(r[0]) for r in db.execute(query)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max stories to backfill."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List affected stories without calling ClickHouse or writing to the DB.",
    )
    parser.add_argument(
        "--config", default="config.toml", help="Path to config.toml (default: %(default)s)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = Config.load(args.config)
    db = Database(config.db_path)

    story_ids = find_stuck_hn_story_ids(db, args.limit)
    if not story_ids:
        print("No stuck HN stories found.")
        return

    print(f"Found {len(story_ids)} HN stories with comment_count > 0 and empty top_comments.")

    if args.dry_run:
        preview = ", ".join(str(sid) for sid in story_ids[:20])
        more = f", ... ({len(story_ids) - 20} more)" if len(story_ids) > 20 else ""
        print(f"Dry run — would backfill: {preview}{more}")
        return

    embedder = Embedder(
        config.onnx_model_dir,
        model_version=config.embedding_model_version,
        max_tokens=config.embedding_max_tokens,
        batch_size=config.embedding_batch_size,
        ort_variant=config.embedding_ort_variant,
    )

    total_updated = 0
    for i in range(0, len(story_ids), _CHUNK_SIZE):
        chunk = story_ids[i : i + _CHUNK_SIZE]
        updated = prewarm_top_stories(chunk, db, embedder)
        total_updated += updated
        print(f"  chunk {i // _CHUNK_SIZE + 1}: backfilled {updated}/{len(chunk)}")

    print(f"Backfilled {total_updated}/{len(story_ids)} stories.")


if __name__ == "__main__":
    main()
