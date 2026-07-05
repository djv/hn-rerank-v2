from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database
from pipeline import Config, Embedder

from scripts._seed_common import (
    _write_dryrun,
    seed_rows,
)

BQ_QUERY_TEMPLATE = """
SELECT
  id,
  title,
  url,
  text,
  score,
  descendants,
  UNIX_SECONDS(timestamp) AS created_at_i
FROM `bigquery-public-data.hacker_news.full`
WHERE type = 'story'
  AND timestamp >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL {months} MONTH))
  AND score >= {min_score}
  AND (deleted IS NULL OR deleted = FALSE)
  AND (dead IS NULL OR dead = FALSE)
ORDER BY score DESC, timestamp DESC
"""


def build_bq_query(
    months: int = 3,
    min_score: int = 100,
    limit: int | None = None,
) -> str:
    if months <= 0:
        raise ValueError("--months must be a positive integer")
    if min_score < 0:
        raise ValueError("--min-score must be a non-negative integer")
    query = BQ_QUERY_TEMPLATE.format(months=int(months), min_score=int(min_score))
    if limit is not None:
        query = f"{query}\nLIMIT {int(limit)}"
    return query


def run_bq_query(
    limit: int | None = None,
    months: int = 3,
    min_score: int = 100,
) -> list[dict[str, Any]]:
    query = build_bq_query(months=months, min_score=min_score, limit=limit)
    max_rows = max(int(limit) if limit is not None else 1000, 1000)
    proc = subprocess.run(
        [
            "bq",
            "--format=json",
            "query",
            f"--max_rows={max_rows}",
            "--use_legacy_sql=false",
            query,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout or "[]")
    if not isinstance(data, list):
        raise ValueError("bq returned a non-list JSON payload")
    return data


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed recent high-score HN archive stories from BigQuery."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--min-score", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch rows and write to JSONL, skip DB and Algolia",
    )
    parser.add_argument(
        "--dry-run-output",
        type=Path,
        default=None,
        help="Output path for dry-run JSONL",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = run_bq_query(
        args.limit,
        months=args.months,
        min_score=args.min_score,
    )
    logging.info("bq rows=%s", len(rows))

    if args.dry_run:
        out = args.dry_run_output or Path(f"bq_seed_dryrun_{int(time.time())}.jsonl")
        _write_dryrun(
            rows,
            out,
            query=build_bq_query(args.months, args.min_score, args.limit),
            source="bq_seed",
        )
        return

    config = Config.load(args.config)
    db = Database(config.db_path)

    try:
        embedder = Embedder(
            config.onnx_model_dir,
            batch_size=config.embedding_batch_size,
            ort_variant=config.embedding_ort_variant,
        )
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated_comments,
        ) = await seed_rows(
            rows,
            source="bq_seed",
            db=db,
            embedder=embedder,
            concurrency=args.concurrency,
        )
        logging.info(
            "seeded=%s skipped_feedback=%s skipped_existing=%s hydrated_comments=%s",
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated_comments,
        )
    finally:
        db.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
