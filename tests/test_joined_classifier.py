from __future__ import annotations

import time
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from database import Database, Story
from pipeline import Config, Embedder, ModelConfig
from pipeline.joined_classifier import JoinedFeatures, JoinedLogistic
from pipeline.ranking import RankTrace, _score_and_rank
from scripts.eval_one_classifier import ClassifierSpec, OneClassifier
from scripts.eval_single_preference_model import fit_lexical_inputs

WORDS = (
    "compiler kernel rust database systems",
    "gossip football fashion horoscope royal",
    "garden recipe travel museum weather",
)
EMB_DIM = 6
OFFLINE_FEATURES: dict[JoinedFeatures, Literal["all", "embedding_words"]] = {
    "all": "all",
    "no_metadata": "embedding_words",
}


def _stories(rng: np.random.Generator, labels: list[int], start: int) -> list[Story]:
    out = []
    for i, label in enumerate(labels):
        words = WORDS[label].split()
        picked = rng.choice(words, size=3)
        out.append(
            Story(
                id=start + i,
                title=" ".join(picked),
                url=f"https://site{label}.example/{i}",
                score=1,
                time=1,
                text_content=" ".join(rng.choice(words, size=5)),
            )
        )
    return out


@pytest.mark.parametrize("features", ["all", "no_metadata"])
@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_matches_the_offline_adapter_scores(
    features: JoinedFeatures, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    labels = list(rng.permutation(np.repeat([0, 1, 2], 15)))
    training = _stories(rng, labels, 0)
    candidates = _stories(rng, list(rng.integers(0, 3, 20)), 500)
    x_train = rng.normal(size=(len(labels), EMB_DIM + 3))
    x_cand = rng.normal(size=(len(candidates), EMB_DIM + 3))
    weights = rng.uniform(0.5, 2.0, len(labels))

    live = JoinedLogistic(
        c=4.0,
        features=features,
        embedding_dim=EMB_DIM,
        embedding_weight=16.0,
        numeric_scale=np.sqrt(0.1 / 4),
        word_scale=1.0,
    ).fit(x_train, labels, sample_weight=weights, stories=training)
    offline = OneClassifier(
        c=4.0,
        gamma=0.03,
        chunk_size=8,
        spec=ClassifierSpec(
            "logistic",
            features=OFFLINE_FEATURES[features],
            score="up_down",
            embedding_weight=16.0,
            numeric_scale=float(np.sqrt(0.1 / 4)),
        ),
        lexical=fit_lexical_inputs(training, labels, candidates),
        metadata_columns=3,
    )
    offline.fit(x_train, labels, sample_weight=weights)
    decision = offline.decision_function(x_cand)

    probs = live.predict_proba(x_cand, candidates)
    np.testing.assert_array_equal(live.classes_, [0, 1, 2])
    np.testing.assert_array_equal(probs, offline.last_probabilities)
    # The adapter's up column round-trips through log/exp: equal to rounding.
    np.testing.assert_allclose(
        probs[:, 2] - probs[:, 0], decision[:, 2], rtol=0, atol=1e-15
    )


def test_rejects_placeholder_rows_without_stories() -> None:
    model = JoinedLogistic(
        c=4.0,
        features="all",
        embedding_dim=2,
        embedding_weight=1.0,
        numeric_scale=1.0,
        word_scale=1.0,
    )
    stories = _stories(np.random.default_rng(0), [0, 2, 0, 2], 0)
    with pytest.raises(ValueError, match="one story per training row"):
        model.fit(
            np.zeros((5, 4)),
            [0, 2, 0, 2, 1],
            sample_weight=np.ones(5),
            stories=stories,
        )


class _NoiseEmbedder(Embedder):
    def __init__(self) -> None:
        self.batch_size = 32
        self.max_tokens = 4096

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(len(texts))
        vecs = rng.standard_normal((len(texts), 384)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _seed(db: Database, labels: list[int]) -> int:
    user = db.create_user("joined")
    rng = np.random.default_rng(7)
    action: dict[int, Literal["down", "neutral", "up"]] = {
        0: "down",
        1: "neutral",
        2: "up",
    }
    for story, label in zip(_stories(rng, labels, 1000), labels, strict=True):
        db.upsert_story(story)
        db.upsert_feedback(user.id, story.id, action[label])
    return user.id


def _rank(
    labels: list[int], features: JoinedFeatures
) -> tuple[dict[int, float], dict[int, tuple[float, float, float]], RankTrace]:
    db = Database(":memory:")
    try:
        user_id = _seed(db, labels)
        config = Config(
            model=ModelConfig(
                min_up_for_svm=2,
                min_down_for_svm=2,
                # Full weight on the classifier tier from the first votes.
                tier2_blend_window=1,
                tier3_blend_window=1,
                classifier="joined_logistic",
                joined_features=features,
            )
        )
        candidates = [
            # Story 1 reads like the upvotes (WORDS[2]), story 2 like the
            # downvotes (WORDS[0]); the embeddings are noise.
            Story(1, "garden recipe travel", "", 10, 1, "museum weather garden"),
            Story(2, "rust kernel compiler", "", 10, 1, "systems database rust"),
        ]
        embs = np.random.default_rng(9).standard_normal((2, 384)).astype(np.float32)
        trace = RankTrace()
        ranked = _score_and_rank(
            candidates, embs, db, config, _NoiseEmbedder(), user_id, trace=trace
        )
        return (
            {r.story.id: r.score for r in ranked},
            {
                r.story.id: (r.prob_down or 0, r.prob_neutral or 0, r.prob_up or 0)
                for r in ranked
            },
            trace,
        )
    finally:
        db.close()


@pytest.mark.parametrize("features", ["all", "no_metadata"])
def test_live_ranking_uses_the_joined_words(features: JoinedFeatures) -> None:
    labels = [0, 1, 2] * 14
    scores, probs, trace = _rank(labels, features)
    assert trace.labels["model_cache"] == "miss"
    assert "svm_fit" not in trace.labels
    assert scores[1] > scores[2]
    for p in probs.values():
        assert sum(p) == pytest.approx(1.0)
    assert probs[1][2] > probs[1][0]


def test_live_ranking_falls_back_without_a_neutral_vote() -> None:
    scores, probs, trace = _rank([0, 2] * 20, "all")
    assert trace.labels["svm_fit"] == "error"
    assert set(scores) == {1, 2}
    assert all(p == (0, 0, 0) for p in probs.values())


def test_joined_classifier_cannot_run_inside_the_linear_blend() -> None:
    with pytest.raises(ValueError, match="replaces the linear blend"):
        Config(
            model=ModelConfig(classifier="joined_logistic", linear_blend_enabled=True)
        )


@pytest.mark.parametrize(
    ("labels", "interleaved"), [([0, 1, 2] * 14, True), ([0, 2] * 20, False)]
)
def test_rerank_interleaves_only_when_every_challenger_fits(
    labels: list[int], interleaved: bool
) -> None:
    from pipeline.interleave import challenger_configs
    from pipeline.ranking import rerank_candidates

    db = Database(":memory:")
    try:
        user_id = _seed(db, labels)
        config = Config(
            interleave_user_ids=(user_id,),
            model=ModelConfig(min_up_for_svm=2, min_down_for_svm=2),
        )
        now = int(time.time())
        rng = np.random.default_rng(3)
        candidates = [
            replace(s, time=now - 600 * i)
            for i, s in enumerate(_stories(rng, list(rng.integers(0, 3, 24)), 5000))
        ]
        for s in candidates:
            db.upsert_story(s)
        embs = rng.standard_normal((len(candidates), 384)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        trace = RankTrace()
        deck = rerank_candidates(
            db,
            config,
            _NoiseEmbedder(),
            candidates,
            embs,
            user_id=user_id,
            trace=trace,
            challengers=challenger_configs(config, user_id),
            rng=np.random.default_rng(4),
        )
        arms = {r.arm for r in deck.window("1d").recommended}
        if interleaved:
            assert arms == {"production", "joined_all", "joined_no_metadata"}
            assert trace.labels["interleave"] == "joined_all,joined_no_metadata"
            assert "challenger_joined_all" in trace.timings_ms
        else:
            assert arms == {""}
            assert trace.labels["interleave"] == "off"
    finally:
        db.close()


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_warm_start_reaches_the_same_model_in_fewer_iterations(seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = list(rng.permutation(np.repeat([0, 1, 2], 30)))
    stories = _stories(rng, labels, 0)
    x = rng.normal(size=(len(labels), EMB_DIM + 3))
    candidates = _stories(rng, list(rng.integers(0, 3, 15)), 500)
    x_cand = rng.normal(size=(len(candidates), EMB_DIM + 3))

    def model() -> JoinedLogistic:
        return JoinedLogistic(
            c=4.0,
            features="all",
            embedding_dim=EMB_DIM,
            embedding_weight=16.0,
            numeric_scale=0.158,
            word_scale=1.0,
        )

    # The previous fit lacks the newest vote, as after a swipe.
    previous = model().fit(
        x[:-1],
        labels[:-1],
        sample_weight=np.ones(len(labels) - 1),
        stories=stories[:-1],
    )
    weights = np.ones(len(labels))
    cold = model().fit(x, labels, sample_weight=weights, stories=stories)
    warm = model().fit(x, labels, sample_weight=weights, stories=stories, warm=previous)

    assert warm.estimator.n_iter_[0] <= cold.estimator.n_iter_[0]
    np.testing.assert_allclose(
        warm.predict_proba(x_cand, candidates),
        cold.predict_proba(x_cand, candidates),
        # lbfgs stops at tol=1e-4 from either start: equal to that tolerance.
        atol=5e-3,
    )
