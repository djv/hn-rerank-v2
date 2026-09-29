from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from database import Database, Story
from pipeline import Config, ModelConfig
from pipeline import Embedder, linear_blend
from pipeline.ranking import _score_and_rank

UP_WORDS = "compiler kernel rust database systems performance"
DOWN_WORDS = "celebrity gossip football fashion horoscope royal"


def _seed(db: Database, seed: int = 0) -> int:
    """40 votes whose text separates the classes; embeddings are pure noise."""
    user = db.create_user(f"blend-{seed}")
    for i in range(40):
        up = i % 2 == 0
        words = UP_WORDS if up else DOWN_WORDS
        story = Story(
            id=1000 + i,
            title=f"{words.split()[i % 6]} story {i}",
            url=f"https://{'up' if up else 'down'}.example/{i}",
            score=10,
            time=1_600_000_000 + i,
            text_content=f"{words} {words.split()[(i + 1) % 6]} item {i}",
        )
        db.upsert_story(story)
        action: Literal["up", "down"] = "up" if up else "down"
        db.upsert_feedback(user.id, story.id, action)
    return user.id


def _candidates() -> list[Story]:
    return [
        Story(
            id=1,
            title="gossip",
            url="https://down.example/x",
            score=10,
            time=1_600_000_000,
            text_content=f"{DOWN_WORDS} royal football",
        ),
        Story(
            id=2,
            title="compiler",
            url="https://up.example/x",
            score=10,
            time=1_600_000_000,
            text_content=f"{UP_WORDS} rust kernel",
        ),
    ]


class _Embedder(Embedder):
    """Noise embeddings: text-independent signal must come from TF-IDF."""

    def __init__(self) -> None:
        self.batch_size = 32
        self.max_tokens = 4096

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(len(texts))
        vecs = rng.standard_normal((len(texts), 384)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _rank(db: Database, user_id: int, config: Config) -> dict[int, float]:
    rng = np.random.default_rng(99)
    embs = rng.standard_normal((2, 384)).astype(np.float32)
    ranked = _score_and_rank(
        _candidates(),
        embs,
        db,
        config,
        _Embedder(),
        user_id=user_id,
    )
    return {r.story.id: r.score for r in ranked}


def _config(enabled: bool, *, dense: float = 0.2, tfidf: float = 0.3) -> Config:
    return Config(
        model=ModelConfig(
            min_up_for_svm=2,
            min_down_for_svm=2,
            linear_blend_enabled=enabled,
            linear_blend_dense_weight=dense,
            linear_blend_tfidf_weight=tfidf,
        )
    )


def test_tfidf_alone_ranks_by_text_when_embeddings_are_noise() -> None:
    db = Database(":memory:")
    try:
        user_id = _seed(db)
        scores = _rank(db, user_id, _config(True, dense=0.0, tfidf=1.0))
        assert scores[2] > scores[1]
    finally:
        db.close()


def test_blend_fits_once_per_feedback_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    real = linear_blend.fit_linear_blend

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(linear_blend, "fit_linear_blend", counting)
    db = Database(":memory:")
    try:
        user_id = _seed(db, seed=1)
        first = _rank(db, user_id, _config(True))
        second = _rank(db, user_id, _config(True))
        assert len(calls) == 1
        assert first == second
    finally:
        db.close()


def test_blend_fit_failure_falls_back_to_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = Database(":memory:")
    try:
        user_id = _seed(db, seed=2)
        off = _rank(db, user_id, _config(False))

        def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("no vocabulary")

        monkeypatch.setattr(linear_blend, "fit_linear_blend", boom)
        assert _rank(db, user_id, _config(True)) == off
    finally:
        db.close()


def test_flag_off_leaves_scores_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("linear blend must not run when disabled")

    monkeypatch.setattr(linear_blend, "fit_linear_blend", boom)
    db = Database(":memory:")
    try:
        user_id = _seed(db, seed=3)
        _rank(db, user_id, _config(False))
    finally:
        db.close()


def test_percentile_scores_share_average_rank_on_ties() -> None:
    got = linear_blend.percentile_scores(np.array([1.0, 1.0, 3.0, 2.0]))
    assert got.tolist() == pytest.approx([1 / 6, 1 / 6, 1.0, 2 / 3])
    assert linear_blend.percentile_scores(np.array([5.0])).tolist() == [1.0]


def test_count_rows_cached_per_story_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    linear_blend._ROWS.clear()
    stories = _candidates()
    first = linear_blend.count_rows(stories)
    calls = []
    real = linear_blend._HASHER.transform

    def counting(texts):  # type: ignore[no-untyped-def]
        calls.append(len(texts))
        return real(texts)

    monkeypatch.setattr(linear_blend._HASHER, "transform", counting)
    assert (linear_blend.count_rows(stories) != first).nnz == 0
    assert calls == []
    edited = [stories[0].__class__(**{**stories[0].__dict__, "title": "new title"})]
    linear_blend.count_rows(edited)
    assert calls == [1]
