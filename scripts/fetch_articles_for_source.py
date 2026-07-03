from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, Story
from pipeline import (
    Config,
    Embedder,
    compose_story_text,
    get_or_compute_embeddings,
    _urllib_fetch,
)
from server import _extract_article_body

ARTICLE_BODY_CHAR_LIMIT = 15_000


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


_COLS = (
    "id, title, url, score, time, text_content, source, "
    "comment_count, discussion_url, comment_count_at_fetch, "
    "self_text, top_comments, article_body"
)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = Config.load("config.toml")
    embedder = Embedder(
        config.onnx_model_dir,
        batch_size=config.embedding_batch_size,
        ort_variant=config.embedding_ort_variant,
    )

    sources = sys.argv[1:] or [
        "rss_theskepticalcardiologist_substack_com",
        "rss_erictopol_substack_com",
    ]

    db = Database()

    for source in sources:
        rows = db.execute(
            f"SELECT {_COLS} FROM stories WHERE source = ? "
            "AND (article_body IS NULL OR article_body = '') AND url IS NOT NULL",
            (source,),
        )
        if not rows:
            logging.info("%s: no stories missing article_body", source)
            continue

        stories = rows_to_stories(rows)
        logging.info("%s: %d stories to fetch", source, len(stories))

        ok = 0
        for s in stories:
            status, body = _urllib_fetch(
                s.url or "", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            )
            if status != 200:
                logging.warning("  id=%s fetch %d", s.id, status)
                continue

            article_body = _extract_article_body(body)
            if not article_body:
                logging.warning("  id=%s no extraction", s.id)
                continue
            article_body = article_body[:ARTICLE_BODY_CHAR_LIMIT]

            new_text = compose_story_text(
                s.title, s.self_text, s.top_comments, article_body
            )
            updated = Story(
                id=s.id,
                title=s.title,
                url=s.url,
                score=s.score,
                time=s.time,
                text_content=new_text,
                source=s.source,
                comment_count=s.comment_count,
                discussion_url=s.discussion_url,
                comment_count_at_fetch=s.comment_count_at_fetch,
                self_text=s.self_text,
                top_comments=s.top_comments,
                article_body=article_body,
            )
            db.upsert_story(updated)
            ok += 1
            logging.info(
                "  id=%s title=%s body=%db", s.id, s.title[:50], len(article_body)
            )

        logging.info("%s: %d/%d done", source, ok, len(stories))

        # recompute embeddings for any stories whose text changed
        updated_rows = db.execute(
            f"SELECT {_COLS} FROM stories WHERE source = ?",
            (source,),
        )
        updated_stories = rows_to_stories(updated_rows)
        get_or_compute_embeddings(updated_stories, embedder, db)
        logging.info("%s: embeddings verified", source)

    logging.info("done")


if __name__ == "__main__":
    asyncio.run(main())
