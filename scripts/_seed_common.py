from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_client import query_stories_with_comments as _ch_query_stories_with_comments
from database import Database, Story
from pipeline import (
    BQ_ARCHIVE_SOURCE,
    Embedder,
    clean_text,
    compose_story_text,
    get_or_compute_embeddings,
    _extract_comments_recursive,
    _select_top_comments,
)


def _write_dryrun(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    query: str,
    source: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        meta = {
            "_meta": {
                "source": source,
                "query": query,
                "rows": len(rows),
                "generated_at": int(time.time()),
            }
        }
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logging.info("Dry-run wrote %s rows to %s", len(rows), output_path)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def story_from_bq_row(
    row: dict[str, Any], *, source: str = BQ_ARCHIVE_SOURCE
) -> Story | None:
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
        source=source,
        comment_count=_coerce_int(row.get("descendants")),
        discussion_url=f"https://news.ycombinator.com/item?id={sid}",
        comment_count_at_fetch=0,
        self_text=self_text,
        top_comments="",
        article_body="",
    )


def _apply_ch_comments_to_story(story: Story, item: dict[str, Any]) -> Story:
    """Apply the comments + score from a CH item dict to a skeleton Story."""
    comment_count = _coerce_int(item.get("num_comments"), story.comment_count or 0)
    children = item.get("children") or []
    all_comments = _extract_comments_recursive(children)
    selected = _select_top_comments(all_comments)
    top_comments = " ".join(c["text"] for c in selected)[:10000]

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
        comment_count=comment_count,
        comment_count_at_fetch=comment_count,
    )


def feedback_story_ids(db: Database) -> set[int]:
    rows = db.execute("SELECT DISTINCT story_id FROM feedback")
    return {int(row[0]) for row in rows}


def existing_story_ids(db: Database) -> set[int]:
    rows = db.execute("SELECT id FROM stories")
    return {int(row[0]) for row in rows}


async def seed_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    db: Database,
    embedder: Embedder,
    concurrency: int = 10,
) -> tuple[int, int, int, int]:
    """Skeleton -> bulk-CH-hydrate -> upsert.

    Replaces the previous per-story parallel Algolia hydration. The CH bulk
    path is one query for the entire skeleton set (vs N parallel Algolia
    calls), and the comment-selection logic is identical to the previous
    implementation (re-uses _extract_comments_recursive / _select_top_comments).
    """
    del concurrency  # bulk path is single-query; concurrency no longer used
    protected_ids = feedback_story_ids(db)
    existing_ids = existing_story_ids(db)
    skeletons: list[Story] = []
    skipped_feedback = 0
    skipped_existing = 0
    for row in rows:
        story = story_from_bq_row(row, source=source)
        if story is None:
            continue
        if story.id in protected_ids:
            skipped_feedback += 1
            continue
        if story.id in existing_ids:
            skipped_existing += 1
            continue
        skeletons.append(story)

    if not skeletons:
        return 0, skipped_feedback, skipped_existing, 0

    skeleton_ids = [s.id for s in skeletons]
    try:
        ch_items = _ch_query_stories_with_comments(skeleton_ids, max_levels=5)
    except Exception as exc:
        logging.warning(
            "bulk CH hydration failed (%r); falling back to skeleton-only", exc
        )
        ch_items = {}

    hydrated: list[Story] = []
    hydrated_count = 0
    for skeleton in skeletons:
        item = ch_items.get(skeleton.id)
        if item is None:
            hydrated.append(skeleton)
            continue
        story = _apply_ch_comments_to_story(skeleton, item)
        hydrated.append(story)
        if story.top_comments:
            hydrated_count += 1

    for story in hydrated:
        db.upsert_story(story)

    get_or_compute_embeddings(hydrated, embedder, db)
    return len(hydrated), skipped_feedback, skipped_existing, hydrated_count
