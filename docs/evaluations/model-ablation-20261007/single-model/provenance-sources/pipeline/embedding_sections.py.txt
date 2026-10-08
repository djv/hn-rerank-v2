"""Offline full-body representation experiment; not the production encoder."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from database import Story
from pipeline.ranking import clean_text, compose_story_text, story_embedding_text


SECTION_LAYOUT_VERSION = "full-body-char-chunks-v1"


def section_texts(story: Story, *, chunk_chars: int = 2400) -> list[str]:
    """Cover the whole available body with bounded chunks, repeating the title.

    Prefer article over RSS self text to avoid counting the opening twice.
    Fixed character chunks work on flattened stored bodies without headings.
    This is a coverage baseline, not semantic section detection. Tokenizer
    truncation must still be audited on languages with high token/char ratios.
    """
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    body = (story.article_body or story.self_text or story.text_content).strip()
    title = story.title.strip()
    if not body:
        return [title]
    return [
        f"{title}\n\n{body[i : i + chunk_chars]}"
        for i in range(0, len(body), chunk_chars)
    ]


def pool_sections(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Equal-weight chunk mean, L2-normalized; never standard-scale embeddings."""
    if vectors.ndim != 2 or len(vectors) == 0 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite nonempty section vectors")
    pooled = vectors.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm <= 1e-8:
        raise ValueError("Section mean has zero norm")
    return np.asarray(pooled / norm, dtype=np.float32)


# Offline per-section embedding experiment (2026-09-28): one vector per part
# of the story, each with its own token budget, compared alone and
# concatenated. "body" is title + self text + article, i.e. production's
# text without comments.
StorySection = Literal["full", "title", "self", "article", "comments", "body"]
STORY_SECTIONS: tuple[StorySection, ...] = (
    "full",
    "title",
    "self",
    "article",
    "comments",
    "body",
)


def story_section_text(story: Story, section: StorySection) -> str:
    """The text embedded for one section, led by the title for context.

    "full" is production's embedding text.
    A story without that section embeds its title alone, so every story gets
    a unit vector and a missing section reads as "nothing beyond the title".
    """
    title = clean_text(story.title)
    if section == "full":
        return story_embedding_text(story)
    if section == "title":
        return title
    if section == "body":
        return compose_story_text(story.title, story.self_text, "", story.article_body)
    raw = {
        "self": story.self_text,
        "article": story.article_body,
        "comments": story.top_comments,
    }[section]
    body = clean_text(raw)
    return f"{title}. {body}" if body else title
