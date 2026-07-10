from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Iterator
from typing import Any, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_client import (
    clear_cache as _clear_ch_cache,
    query_stories_with_comments as _ch_query_stories_with_comments,
)
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


@dataclass(frozen=True)
class SeedResult:
    """Outcome counters for one archive seed or reconciliation run."""

    inserted: int = 0
    refreshed: int = 0
    promoted: int = 0
    skipped_feedback: int = 0
    skipped_existing: int = 0
    hydrated_comments: int = 0

    def __iter__(self) -> Iterator[int]:
        """Preserve the legacy four-counter unpacking contract for BQ callers."""
        yield self.inserted
        yield self.skipped_feedback
        yield self.skipped_existing
        yield self.hydrated_comments


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


T = TypeVar("T")


def _chunked(items: list[T], size: int) -> list[list[T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


async def seed_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    db: Database,
    embedder: Embedder,
    concurrency: int = 10,
    reconcile: bool = False,
    live_window_days: int = 30,
    now_ts: int | None = None,
    hydration_batch_size: int = 200,
) -> SeedResult:
    """Seed archive stories, optionally reconciling safe existing HN rows.

    The default is pure backfill: feedback and existing IDs are untouched.
    Reconciliation refreshes only rows owned by this archive source or HN
    rows. Recent HN rows retain ``source='hn'`` so they remain in the live
    gravity-ranked candidate leg; aged qualifying HN rows are promoted to the
    archive source. Other source labels (notably ``bq_seed``) are never
    changed.
    """
    del concurrency  # bulk path is explicitly batched; concurrency is legacy CLI API
    if live_window_days <= 0:
        raise ValueError("live_window_days must be positive")
    if hydration_batch_size <= 0:
        raise ValueError("hydration_batch_size must be positive")

    protected_ids = feedback_story_ids(db)
    row_stories = [
        story
        for row in rows
        if (story := story_from_bq_row(row, source=source)) is not None
    ]
    existing_by_id = db.get_seed_story_states([story.id for story in row_stories])
    live_cutoff = int(now_ts if now_ts is not None else time.time()) - (
        live_window_days * 86400
    )
    to_upsert: list[Story] = []
    needs_hydration: list[Story] = []
    result = SeedResult()

    for story in row_stories:
        if story.id in protected_ids:
            result = replace(result, skipped_feedback=result.skipped_feedback + 1)
            continue
        existing = existing_by_id.get(story.id)
        if existing is None:
            target_source = (
                "hn" if reconcile and story.time >= live_cutoff else source
            )
            new_story = replace(story, source=target_source)
            to_upsert.append(new_story)
            needs_hydration.append(new_story)
            continue
        if not reconcile or existing.source not in {"hn", source}:
            result = replace(result, skipped_existing=result.skipped_existing + 1)
            continue

        target_source = (
            "hn" if existing.source == "hn" and story.time >= live_cutoff else source
        )
        reconciled = replace(story, source=target_source)
        to_upsert.append(reconciled)
        if target_source == source and existing.source == "hn":
            result = replace(result, promoted=result.promoted + 1)
        else:
            result = replace(result, refreshed=result.refreshed + 1)
        if not existing.has_top_comments:
            needs_hydration.append(reconciled)

    if not to_upsert:
        return result

    hydrated_count = 0
    hydration_by_id = {story.id: story for story in needs_hydration}
    for story_batch in _chunked(to_upsert, hydration_batch_size):
        hydrated_by_id: dict[int, Story] = {}
        batch_hydration_ids = [
            story.id for story in story_batch if story.id in hydration_by_id
        ]
        if batch_hydration_ids:
            try:
                ch_items = _ch_query_stories_with_comments(
                    batch_hydration_ids, max_levels=5
                )
                for story_id in batch_hydration_ids:
                    item = ch_items.get(story_id)
                    if item is None:
                        continue
                    hydrated = _apply_ch_comments_to_story(
                        hydration_by_id[story_id], item
                    )
                    hydrated_by_id[story_id] = hydrated
                    if hydrated.top_comments:
                        hydrated_count += 1
            except Exception as exc:
                logging.warning(
                    "bulk CH hydration failed for %d stories (%r); using skeletons",
                    len(batch_hydration_ids),
                    exc,
                )
            finally:
                _clear_ch_cache()

        final_stories = [
            hydrated_by_id.get(story.id, story) for story in story_batch
        ]
        for story in final_stories:
            db.upsert_story(story)

        # ``upsert_story`` preserves longer cached fields. Re-read only this
        # bounded batch before hashing, preventing a seed run from retaining
        # thousands of full comment bodies and embeddings at once.
        stored_stories = db.get_stories([story.id for story in final_stories])
        get_or_compute_embeddings(stored_stories, embedder, db)
    return replace(
        result,
        inserted=sum(1 for story in to_upsert if story.id not in existing_by_id),
        hydrated_comments=hydrated_count,
    )
