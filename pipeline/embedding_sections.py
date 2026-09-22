"""Offline full-body representation experiment; not the production encoder."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from database import Story


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
