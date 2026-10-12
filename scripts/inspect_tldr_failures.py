"""Inspect stored public story inputs and optionally replay their TLDR calls.

The database is opened in read-only mode. Replays use the configured provider
and report completion metadata without writing summaries or loading an encoder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from database import Story


def read_story(db_path: Path, story_id: int) -> Story:
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT id, title, url, score, time, source, self_text, top_comments, "
            "article_body FROM stories WHERE id = ?",
            (story_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Story {story_id} is absent")
    return Story(
        id=int(row[0]),
        title=str(row[1]),
        url=row[2],
        score=int(row[3]),
        time=int(row[4]),
        text_content="",
        source=str(row[5]),
        self_text=row[6] or "",
        top_comments=row[7] or "",
        article_body=row[8] or "",
    )


async def inspect_story(story: Story, *, replay: bool) -> dict[str, object]:
    report: dict[str, object] = {
        "story_id": story.id,
        "title": story.title,
        "url": story.url,
        "source": story.source,
        "self_chars": len(story.self_text),
        "comment_chars": len(story.top_comments),
        "article_chars": len(story.article_body),
        "article_prefix": story.article_body[:32],
    }
    if not replay:
        return report
    calls: list[dict[str, object]] = []
    original = server._call_llm_once

    async def observed_call(
        cfg: server.LlmProviderConfig,
        *,
        prompt: str,
        max_tokens: int,
        on_usage: server.LlmUsageRecorder | None = None,
    ) -> server.LlmChatResult:
        result = await original(
            cfg, prompt=prompt, max_tokens=max_tokens, on_usage=on_usage
        )
        calls.append(
            {
                "section": "discussion" if "Comments:" in prompt else "article",
                "ok": result.ok,
                "status": result.status,
                "finish_reason": result.finish_reason,
                "max_tokens": max_tokens,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "chars": len(result.content),
                "sample": result.content[:160] if not result.ok else "",
            }
        )
        return result

    with patch.object(server, "_call_llm_once", side_effect=observed_call):
        result = await server.generate_detailed_tldr(
            story.title, story.self_text, story.top_comments, story.article_body
        )
    report.update(kind=result.kind, cacheable=result.cacheable, calls=calls)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--story-id", type=int, action="append", required=True)
    parser.add_argument("--replay", action="store_true", help="Call the live provider")
    args = parser.parse_args()
    if args.replay:
        server.load_env()
    for story_id in args.story_id:
        story = read_story(args.db, story_id)
        print(json.dumps(asyncio.run(inspect_story(story, replay=args.replay))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
