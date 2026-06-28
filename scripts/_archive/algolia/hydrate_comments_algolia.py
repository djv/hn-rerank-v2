"""Archived Algolia per-story parallel hydration (2026-06-26).

Replaced by ch_client.query_stories_with_comments (one CH bulk query instead
of N parallel Algolia calls). Kept here as a fallback if CH becomes
unavailable.

To restore the old behavior in scripts/_seed_common.py, import this module
instead of using ch_client.
"""

from __future__ import annotations

from dataclasses import replace

import httpx

from database import Story
from pipeline import (
    clean_text,
    compose_story_text,
    _extract_comments_recursive,
    _select_top_comments,
)
from scripts._seed_common import _coerce_int


async def hydrate_comments_from_algolia(
    client: httpx.AsyncClient,
    story: Story,
) -> Story:
    """One Algolia items call per story. Used to be the archive seed hydration
    path; replaced by CH bulk on 2026-06-26.
    """
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
