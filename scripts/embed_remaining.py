from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, Story
from pipeline import Config, Embedder, get_or_compute_embeddings

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = Config.load("config.toml")
    db = Database(config.db_path)
    embedder = Embedder(
        config.onnx_model_dir,
        batch_size=config.embedding_batch_size,
        ort_variant=config.embedding_ort_variant,
    )

    rows = db.execute(
        "SELECT s.id FROM stories s "
        "LEFT JOIN embeddings e ON e.story_id = s.id "
        "WHERE e.story_id IS NULL"
    )
    unembedded_ids = [int(r[0]) for r in rows]
    logging.info("stories missing embeddings: %s", len(unembedded_ids))

    computed = 0
    for i in range(0, len(unembedded_ids), 500):
        batch = unembedded_ids[i : i + 500]
        ph = ",".join("?" for _ in batch)
        rows = db.execute(
            f"SELECT {STORY_COLS} FROM stories WHERE id IN ({ph})", tuple(batch)
        )
        stories = rows_to_stories(rows)
        get_or_compute_embeddings(stories, embedder, db)
        computed += len(stories)
        logging.info("  embedding: %s/%s", computed, len(unembedded_ids))

    total = db.execute("SELECT COUNT(*) FROM stories")[0][0]
    embedded = db.execute("SELECT COUNT(DISTINCT story_id) FROM embeddings")[0][0]
    logging.info("done: stories=%s embedded=%s", total, embedded)


if __name__ == "__main__":
    main()
