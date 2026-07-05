from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database
from pipeline import CH_ARCHIVE_SOURCE, Config, Embedder

from scripts._seed_common import _write_dryrun, seed_rows

CH_PLAYGROUND_URL = "https://play.clickhouse.com/?user=play&default_format=JSON"

CH_QUERY_TEMPLATE = """
SELECT
    id,
    title,
    url,
    text,
    score,
    descendants,
    toUnixTimestamp(time) AS created_at_i
FROM hackernews_history FINAL
WHERE type = 'story'
  AND time >= now() - INTERVAL {months} MONTH
  AND deleted = 0
  AND dead = 0
  AND score >= {min_score}
ORDER BY score DESC, time DESC
"""


def build_ch_query(
    months: int = 3,
    min_score: int = 100,
    limit: int | None = None,
) -> str:
    if months <= 0:
        raise ValueError("--months must be a positive integer")
    if min_score < 0:
        raise ValueError("--min-score must be a non-negative integer")
    query = CH_QUERY_TEMPLATE.format(months=int(months), min_score=int(min_score))
    if limit is not None:
        query = f"{query}\nLIMIT {int(limit)}"
    return query


def run_ch_query(
    limit: int | None = None,
    months: int = 3,
    min_score: int = 100,
) -> list[dict[str, Any]]:
    query = build_ch_query(months=months, min_score=min_score, limit=limit)
    resp = httpx.post(
        CH_PLAYGROUND_URL,
        content=query,
        timeout=30.0,
    )
    resp.raise_for_status()
    payload: Any = resp.json()
    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
    elif isinstance(payload, list):
        data = payload
    else:
        raise ValueError("ClickHouse returned an unexpected JSON payload shape")
    return data


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed recent high-score HN archive stories from ClickHouse Playground."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--min-score", type=int, default=200)
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

    rows = run_ch_query(
        args.limit,
        months=args.months,
        min_score=args.min_score,
    )
    logging.info("ch rows=%s", len(rows))

    if args.dry_run:
        out = args.dry_run_output or Path(f"ch_seed_dryrun_{int(time.time())}.jsonl")
        _write_dryrun(
            rows,
            out,
            query=build_ch_query(args.months, args.min_score, args.limit),
            source=CH_ARCHIVE_SOURCE,
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
            source=CH_ARCHIVE_SOURCE,
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
