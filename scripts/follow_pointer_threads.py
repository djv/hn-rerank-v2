"""Swap pointer-thread notes for the comments of the thread they point to.

A story whose whole discussion is a short "Comments moved to item?id=N" note
(`pipeline.hn_dupes.pointer_thread_target`) got a TLDR invented from that
note (story 32148318, 38507672; WORKLOG.md 2026-09-29). Taps and prefetch now
follow the link one story at a time; this does all of them at once, with the
same `server._follow_pointer_thread`, and embeds the new text here so the
server's next candidate-pool rebuild does not re-embed them under its lock.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database import Database, Story  # noqa: E402
from pipeline import Config, Embedder, get_or_compute_embeddings  # noqa: E402
from pipeline.hn_dupes import POINTER_THREAD_MAX_CHARS  # noqa: E402

_CONCURRENCY = 4


def find_pointer_stories(db: Database) -> list[Story]:
    """HN-family stories whose stored comments are still a pointer note."""
    import server

    rows = db.execute(
        "SELECT id FROM stories WHERE source IN ('hn', 'ch_seed', 'bq_seed') "
        "AND length(top_comments) BETWEEN 1 AND ? AND top_comments LIKE '%item?id=%' "
        "ORDER BY score DESC",
        (POINTER_THREAD_MAX_CHARS,),
    )
    stories = db.get_stories([int(r[0]) for r in rows])
    return [s for s in stories if server._is_unfollowed_pointer(s)]


async def follow_all(db: Database, stories: list[Story]) -> list[Story]:
    import server

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def one(story: Story) -> Story | None:
        async with sem:
            followed = await server._follow_pointer_thread(db, story)
        return followed if followed.top_comments != story.top_comments else None

    results = await asyncio.gather(*(one(s) for s in stories))
    return [s for s in results if s is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pointer stories without fetching threads or writing the DB.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to config.toml (default: %(default)s).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = Config.load(args.config)
    db = Database(config.db_path)

    stories = find_pointer_stories(db)
    print(f"Found {len(stories)} pointer-thread stories.")
    if args.dry_run or not stories:
        for s in stories[:20]:
            print(f"  {s.id} [{s.source}] {s.title[:60]}")
        return

    followed = asyncio.run(follow_all(db, stories))
    print(f"Followed {len(followed)}/{len(stories)} (others: target empty or missing).")
    if not followed:
        return

    embedder = Embedder(
        config.onnx_model_dir,
        model_version=config.embedding_model_version,
        max_tokens=config.embedding_max_tokens,
        batch_size=config.embedding_batch_size,
        ort_variant=config.embedding_ort_variant,
    )
    get_or_compute_embeddings(followed, embedder, db)
    print(f"Embedded {len(followed)} updated stories.")


if __name__ == "__main__":
    main()
