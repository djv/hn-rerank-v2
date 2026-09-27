#!/usr/bin/env python3
"""Fetch article text for recent RSS snippet rows, draining the regen backlog.

Uses the same selection and fetcher as regen (``select_rss_article_prewarm``
+ ``fetch_and_cache_article_bodies``): SSRF-guarded fetches, shared failure
memory, re-embedding of changed rows. Regen does 30 per run; this does the
rest in chunks. Writes only ``stories.article_body``/``text_content``,
embeddings and ``article_fetch_failures``.

Usage:
    uv run python scripts/backfill_rss_articles.py --dry-run
    uv run python scripts/backfill_rss_articles.py --max 2000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import (  # noqa: E402
    Config,
    Embedder,
    fetch_and_cache_article_bodies,
    select_rss_article_prewarm,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch article text for recent RSS snippet rows."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--max", type=int, default=2000, help="stories in total")
    parser.add_argument("--chunk", type=int, default=50, help="stories per batch")
    parser.add_argument("--dry-run", action="store_true", help="only count")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = Config.load(args.config)
    db = Database(config.db_path)
    try:
        cutoff = int(time.time()) - config.article_fetch_max_age_days * 86400
        candidates = db.get_stories(
            [
                int(row[0])
                for row in db.execute(
                    "SELECT id FROM stories WHERE source LIKE 'rss\\_%' ESCAPE '\\' "
                    "AND time >= ? AND article_body = ''",
                    (cutoff,),
                )
            ]
        )
        targets = select_rss_article_prewarm(
            candidates,
            db,
            max_per_run=args.max,
            max_age_days=config.article_fetch_max_age_days,
        )
        logging.info("backfill_rss_articles: %d eligible", len(targets))
        if args.dry_run or not targets:
            return
        embedder = Embedder(
            config.onnx_model_dir,
            model_version=config.embedding_model_version,
            max_tokens=config.embedding_max_tokens,
            batch_size=config.embedding_batch_size,
            ort_variant=config.embedding_ort_variant,
        )
        done = 0
        for start in range(0, len(targets), args.chunk):
            chunk = targets[start : start + args.chunk]
            fetched = asyncio.run(
                fetch_and_cache_article_bodies(
                    db=db,
                    embedder=embedder,
                    stories=chunk,
                    concurrency=config.article_fetch_concurrency,
                )
            )
            done += len(fetched)
            logging.info(
                "backfill_rss_articles: %d/%d fetched so far",
                done,
                start + len(chunk),
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
