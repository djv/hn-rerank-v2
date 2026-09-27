from __future__ import annotations

import numpy as np
import pytest

from database import Story
from pipeline.embedding_sections import pool_sections, section_texts


def test_sections_cover_every_character_without_duplicating_rss() -> None:
    body = "opening " * 600 + "TAIL TOPIC"
    story = Story(
        1,
        "Newsletter",
        None,
        0,
        0,
        "legacy",
        article_body=body,
        self_text="duplicated RSS opening",
    )
    chunks = section_texts(story, chunk_chars=1000)
    assert "".join(chunk.split("\n\n", 1)[1] for chunk in chunks) == body
    assert "TAIL TOPIC" in chunks[-1]
    assert all("duplicated RSS" not in chunk for chunk in chunks)
    assert len(chunks) > 1


def test_pool_is_unit_norm_and_rejects_invalid_data() -> None:
    result = pool_sections(np.array([[1, 0], [0, 1]], dtype=np.float32))
    assert np.linalg.norm(result) == pytest.approx(1)
    np.testing.assert_allclose(result, [2**-0.5, 2**-0.5])
    with pytest.raises(ValueError):
        pool_sections(np.zeros((1, 384), dtype=np.float32))
