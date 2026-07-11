from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, Story
from pipeline import CH_ARCHIVE_SOURCE, Config, Embedder, get_or_compute_embeddings
from scripts._seed_common import _apply_ch_comments_to_story
from ch_client import query_stories_with_comments

STORY_COLS = (
    "id, title, url, score, time, text_content, source, "
    "comment_count, discussion_url, comment_count_at_fetch, "
    "self_text, top_comments, article_body"
)


def rows_to_stories(rows: list[tuple]) -> list[Story]:
    out: list[Story] = []
    for r in rows:
        out.append(
            Story(
                id=int(r[0]),
                title=str(r[1] or ""),
                url=str(r[2]) if r[2] else None,
                score=int(r[3] or 0),
                time=int(r[4] or 0),
                text_content=str(r[5] or ""),
                source=str(r[6] or ""),
                comment_count=int(r[7] or 0),
                discussion_url=str(r[8] or ""),
                comment_count_at_fetch=int(r[9] or 0),
                self_text=str(r[10] or ""),
                top_comments=str(r[11] or ""),
                article_body=str(r[12] or ""),
            )
        )
    return out


def get_skeleton_stories(db: Database, source: str) -> list[Story]:
    rows = db.execute(
        f"SELECT {STORY_COLS} FROM stories "
        "WHERE source = ? AND (top_comments IS NULL OR top_comments = '')",
        (source,),
    )
    return rows_to_stories(rows)


def get_unembedded_story_ids(db: Database) -> set[int]:
    rows = db.execute(
        "SELECT s.id FROM stories s "
        "LEFT JOIN embeddings e ON e.story_id = s.id "
        "WHERE e.story_id IS NULL"
    )
    return {int(r[0]) for r in rows}


def batch_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = Config.load("config.toml")
    db = Database(config.db_path)
    embedder = Embedder(
        config.onnx_model_dir,
        model_version=config.embedding_model_version,
        max_tokens=config.embedding_max_tokens,
        batch_size=config.embedding_batch_size,
        ort_variant=config.embedding_ort_variant,
    )

    # --- Step 1: Hydrate comments for skeleton ch_seed stories ---
    skeletons = get_skeleton_stories(db, CH_ARCHIVE_SOURCE)
    logging.info("skeleton stories to hydrate: %s", len(skeletons))

    hydrated = 0
    failed = 0

    for batch_idx, batch in enumerate(batch_list(skeletons, 500)):
        ids = [s.id for s in batch]
        ch_items: dict = {}
        for attempt in range(3):
            try:
                ch_items = query_stories_with_comments(ids, max_levels=5)
                break
            except Exception as exc:
                logging.warning(
                    "batch %s attempt %s/%s failed: %r",
                    batch_idx,
                    attempt + 1,
                    3,
                    exc,
                )
                if attempt == 0:
                    ids = ids[: len(ids) // 2]  # halve the batch
        else:
            # All 3 attempts exhausted; try individual fallback
            ch_items = {}
            for story in batch:
                try:
                    item = query_stories_with_comments([story.id], max_levels=5)
                    if story.id in item:
                        ch_items[story.id] = item[story.id]
                except Exception:
                    pass

        for skeleton in batch:
            item = ch_items.get(skeleton.id)
            if item is not None:
                story = _apply_ch_comments_to_story(skeleton, item)
                db.upsert_story(story)
                if story.top_comments:
                    hydrated += 1
                else:
                    failed += 1
            else:
                failed += 1

        if (batch_idx + 1) % 10 == 0:
            n_batches = (len(skeletons) + 499) // 500
            logging.info(
                "  hydrating: batch %s/%s (ok=%s failed=%s)",
                batch_idx + 1,
                n_batches,
                hydrated,
                failed,
            )

    logging.info("comment hydration done: ok=%s failed=%s", hydrated, failed)

    # --- Step 2: Compute missing embeddings ---
    unembedded_ids = get_unembedded_story_ids(db)
    logging.info("stories missing embeddings: %s", len(unembedded_ids))

    computed = 0
    for batch in batch_list(list(unembedded_ids), 500):
        ph = ",".join("?" for _ in batch)
        rows = db.execute(
            f"SELECT {STORY_COLS} FROM stories WHERE id IN ({ph})", tuple(batch)
        )
        stories = rows_to_stories(rows)
        get_or_compute_embeddings(stories, embedder, db)
        computed += len(stories)
        if computed % 5000 == 0:
            logging.info("  embedding: %s/%s", computed, len(unembedded_ids))

    logging.info("embedding done: computed %s stories", computed)

    total = db.execute("SELECT COUNT(*) FROM stories")[0][0]
    embedded = db.execute("SELECT COUNT(DISTINCT story_id) FROM embeddings")[0][0]
    with_comments = db.execute(
        "SELECT COUNT(*) FROM stories WHERE top_comments IS NOT NULL AND top_comments != ''"
    )[0][0]
    logging.info(
        "final: stories=%s embedded=%s with_comments=%s", total, embedded, with_comments
    )


if __name__ == "__main__":
    asyncio.run(main())
