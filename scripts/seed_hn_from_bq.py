from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, Story
from pipeline import (
    BQ_ARCHIVE_SOURCE,
    Config,
    Embedder,
    clean_text,
    compose_story_text,
    get_or_compute_embeddings,
    _extract_comments_recursive,
    _select_top_comments,
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


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def story_from_bq_row(row: dict[str, Any]) -> Story | None:
    sid = _coerce_int(row.get("id"))
    title = clean_text(str(row.get("title") or ""))
    if sid <= 0 or not title:
        return None

    self_text = clean_text(str(row.get("text") or ""))
    text_content = compose_story_text(title, self_text)
    if not text_content:
        return None

    return Story(
        id=sid,
        title=title,
        url=row.get("url") or None,
        score=_coerce_int(row.get("score")),
        time=_coerce_int(row.get("created_at_i")),
        text_content=text_content,
        source=BQ_ARCHIVE_SOURCE,
        comment_count=_coerce_int(row.get("descendants")),
        discussion_url=f"https://news.ycombinator.com/item?id={sid}",
        comment_count_at_fetch=0,
        self_text=self_text,
        top_comments="",
        article_body="",
    )


async def hydrate_comments_from_algolia(
    client: httpx.AsyncClient,
    story: Story,
) -> Story:
    try:
        resp = await client.get(f"https://hn.algolia.com/api/v1/items/{story.id}")
        if resp.status_code != 200:
            return story
        item = resp.json()
    except Exception:
        return story

    if not item or item.get("type") != "story":
        return story

    children = item.get("children") or []
    all_comments = _extract_comments_recursive(children)
    selected = _select_top_comments(all_comments)
    top_comments = " ".join(c["text"] for c in selected)[:10000]
    comment_count = item.get("num_comments")
    story_text = clean_text(str(item.get("story_text") or item.get("text") or ""))
    self_text = (
        story_text if len(story_text) > len(story.self_text) else story.self_text
    )
    text_content = compose_story_text(
        story.title,
        self_text,
        top_comments,
        story.article_body,
    )
    if not text_content:
        return story

    return replace(
        story,
        self_text=self_text,
        top_comments=top_comments,
        text_content=text_content,
        comment_count=_coerce_int(comment_count, story.comment_count or 0),
        comment_count_at_fetch=_coerce_int(comment_count, story.comment_count or 0),
    )


def feedback_story_ids(db: Database) -> set[int]:
    rows = db.execute("SELECT DISTINCT story_id FROM feedback")
    return {int(row[0]) for row in rows}


def existing_story_ids(db: Database) -> set[int]:
    rows = db.execute("SELECT id FROM stories")
    return {int(row[0]) for row in rows}


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


async def seed_rows(
    rows: list[dict[str, Any]],
    *,
    db: Database,
    embedder: Embedder,
    concurrency: int = 10,
) -> tuple[int, int, int, int]:
    protected_ids = feedback_story_ids(db)
    existing_ids = existing_story_ids(db)
    skeletons: list[Story] = []
    skipped_feedback = 0
    skipped_existing = 0
    for row in rows:
        story = story_from_bq_row(row)
        if story is None:
            continue
        if story.id in protected_ids:
            skipped_feedback += 1
            continue
        if story.id in existing_ids:
            skipped_existing += 1
            continue
        skeletons.append(story)

    sem = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(timeout=20.0) as client:

        async def hydrate(story: Story) -> Story:
            async with sem:
                return await hydrate_comments_from_algolia(client, story)

        hydrated = await asyncio.gather(*(hydrate(story) for story in skeletons))

    for story in hydrated:
        db.upsert_story(story)

    get_or_compute_embeddings(hydrated, embedder, db)
    hydrated_comments = sum(1 for story in hydrated if story.top_comments)
    return len(hydrated), skipped_feedback, skipped_existing, hydrated_comments


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed recent high-score HN archive stories from BigQuery."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--min-score", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = Config.load(args.config)
    db = Database(config.db_path)
    try:
        rows = run_bq_query(
            args.limit,
            months=args.months,
            min_score=args.min_score,
        )
        logging.info("bq rows=%s", len(rows))
        embedder = Embedder(config.onnx_model_dir)
        (
            inserted,
            skipped_feedback,
            skipped_existing,
            hydrated_comments,
        ) = await seed_rows(
            rows,
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
