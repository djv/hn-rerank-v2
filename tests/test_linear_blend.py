from __future__ import annotations

from typing import Literal, cast

from dataclasses import replace

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


def _config(
    enabled: bool,
    *,
    dense: float = 0.2,
    tfidf: float = 0.3,
    ramp: bool = False,
    gate: int = 2,
) -> Config:
    return Config(
        model=ModelConfig(
            min_up_for_svm=gate,
            min_down_for_svm=gate,
            linear_blend_enabled=enabled,
            linear_blend_dense_weight=dense,
            linear_blend_tfidf_weight=tfidf,
            linear_blend_ramp=ramp,
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
    edited = [replace(stories[0], title="new title")]
    linear_blend.count_rows(edited)
    assert calls == [1]


def test_warm_start_reaches_the_cold_fit_in_fewer_iterations() -> None:
    """A refit started from the previous vote set's model lands on the same
    solution (the problem is convex) but needs fewer solver iterations."""
    rng = np.random.default_rng(3)
    stories, labels = [], []
    for i in range(120):
        up = i % 3 != 0
        words = (UP_WORDS if up else DOWN_WORDS).split()
        picked = rng.choice(words, size=4)
        stories.append(
            Story(
                id=5000 + i,
                title=" ".join(picked),
                url=f"https://s{i % 7}.example/{i}",
                score=1,
                time=1,
                text_content=" ".join(rng.choice(words + ["misc", "news"], size=30)),
            )
        )
        labels.append(2 if up else 0)
    dense = rng.standard_normal((120, 16)).astype(np.float32)
    weights = linear_blend.balanced_weights(labels)

    def fit(n: int, warm: linear_blend.LinearBlendModels | None = None):
        return linear_blend.fit_linear_blend(
            dense[:n],
            labels[:n],
            weights[:n],
            stories[:n],
            labels[:n],
            dense_c=0.1,
            tfidf_c=4.0,
            warm=warm,
        )

    previous = fit(119)
    cold = fit(120)
    warm = fit(120, warm=previous)

    x = cold.idf.transform(linear_blend.count_rows(stories)[:, cold.keep])
    np.testing.assert_allclose(
        linear_blend.up_minus_down(warm.tfidf, x),
        linear_blend.up_minus_down(cold.tfidf, x),
        atol=0.02,
    )
    assert warm.tfidf.n_iter_[0] < cold.tfidf.n_iter_[0]
    assert warm.dense.n_iter_[0] <= cold.dense.n_iter_[0]


def test_blend_ramps_in_with_the_svm_tier() -> None:
    """At the 20 up / 20 down gate the SVM tier has no weight, so neither has
    the blend: scores match the blend switched off. Past the gate the blend
    moves the ranking (TF-IDF alone separates these candidates)."""
    db = Database(":memory:")
    try:
        user_id = _seed(db, seed=4)
        at_gate = _rank(
            db, user_id, _config(True, tfidf=1.0, dense=0.0, ramp=True, gate=20)
        )
        assert at_gate == _rank(db, user_id, _config(False, gate=20))
        past_gate = _rank(db, user_id, _config(True, tfidf=1.0, dense=0.0, ramp=True))
        assert past_gate != _rank(db, user_id, _config(False))
    finally:
        db.close()


def test_cache_keeps_one_fit_per_user_and_drops_warm_starts_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cachetools import LRUCache

    monkeypatch.setattr(linear_blend, "_CACHE", LRUCache(maxsize=10_000))
    monkeypatch.setattr(linear_blend, "_LATEST", {})
    # Stand-ins: the cache never looks inside a fit.
    fits = {
        name: cast(linear_blend.LinearBlendModels, object())
        for name in ("a1", "a2", "b1", "c1")
    }

    def put(user: int, sig: str) -> None:
        linear_blend.set_cached((user, sig, 1), fits[sig], 2)

    put(1, "a1")
    put(1, "a2")
    assert list(linear_blend._CACHE) == [(1, "a2", 1)]
    assert linear_blend.latest(1) is fits["a2"]
    put(2, "b1")
    put(3, "c1")  # over maxsize 2: user 1, least recently used, leaves
    assert set(linear_blend._CACHE) == {(2, "b1", 1), (3, "c1", 1)}
    assert linear_blend.latest(1) is None
