"""Synthetic Teach checks: no DB, embedding model, or external requests."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st
from sklearn.metrics import roc_auc_score

from database import Story
from scripts.eval_ranker_variants import FoldData
from scripts.simulate_teach import (
    RULES,
    Pair,
    Rule,
    metrics,
    paired_bootstrap,
    select,
    simulate,
    summarize,
)


def fake_fold(size: int = 12) -> FoldData:
    stories = [Story(i, str(i), None, 1, 1, "") for i in range(size + 3)]
    embeddings = np.tile(np.array([[1.0, 0.0]]), (size + 3, 1))
    empty = np.empty((0, 0))
    return FoldData(
        candidates=stories[3:],
        cand_emb=embeddings[3:],
        train_stories=stories[:3],
        test_stories=stories[3:],
        test_actions=np.arange(size) % 3,
        train_vote_times=np.arange(3, dtype=float),
        test_vote_times=np.arange(3, size + 3, dtype=float),
        x_train_base=empty,
        x_cand_base=empty,
        y_train=np.arange(3),
        train_emb=embeddings[:3],
        tier2_scores=np.zeros(size),
    )


def fake_scorer(fold: FoldData) -> tuple[np.ndarray, np.ndarray | None]:
    return (
        np.array([s.id / 100 for s in fold.candidates]),
        np.tile(np.array([[0.2, 0.3, 0.5]]), (len(fold.candidates), 1)),
    )


@pytest.mark.parametrize("rule", RULES)
@given(
    size=st.integers(1, 30),
    count=st.integers(0, 40),
    seed=st.integers(0, 2**32 - 1),
)
def test_selection_invariants(rule: Rule, size: int, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.random(size)
    probs = rng.dirichlet(np.ones(3), size=size)
    embeddings = np.column_stack([np.cos(scores), np.sin(scores)])
    picks = select(
        rule,
        scores,
        probs,
        embeddings,
        np.array([[1.0, 0.0]]),
        np.array([0, 1, 2]),
        count,
        np.random.default_rng(seed),
    )
    assert len(picks) == min(size, count)
    assert len(set(picks)) == len(picks)
    assert all(0 <= p < size for p in picks)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [("entropy", 1), ("up_margin", 2), ("near_cutoff", 2), ("novel", 3)],
)
def test_rule_definitions(rule: Rule, expected: int) -> None:
    scores = np.array([0.0, 0.2, 0.8, 1.0])
    probs = np.array(
        [[0.01, 0.01, 0.98], [1 / 3] * 3, [0.2, 0.29, 0.51], [0.9, 0.05, 0.05]]
    )
    embeddings = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])
    picks = select(
        rule,
        scores,
        probs,
        embeddings,
        np.array([[1.0, 0.0]]),
        np.array([2, 0, 0]),
        1,
        np.random.default_rng(0),
    )
    assert picks.tolist() == [expected]


@given(seed=st.integers(0, 2**32 - 1))
def test_mix_reproducible(seed: int) -> None:
    fold = fake_fold(24)
    a = simulate(fold, 1, "mix", seed, [3, 7, 12], 4, fake_scorer)
    b = simulate(fold, 1, "mix", seed, [3, 7, 12], 4, fake_scorer)
    assert a == b


@pytest.mark.parametrize("rule", RULES)
def test_rounds_cumulative_and_isolated(rule: Rule) -> None:
    fold = fake_fold()
    evaluation_ids = {s.id for s in fold.test_stories[6:]}
    training_sizes: list[int] = []
    training_times: list[np.ndarray] = []

    def scorer(current: FoldData) -> tuple[np.ndarray, np.ndarray | None]:
        assert not evaluation_ids.intersection(s.id for s in current.train_stories)
        assert not {s.id for s in current.candidates}.intersection(
            s.id for s in current.train_stories
        )
        assert current.runtime_db is None
        training_sizes.append(len(current.train_stories))
        training_times.append(current.train_vote_times.copy())
        return fake_scorer(current)

    run = simulate(fold, 1, rule, 42, [3, 5, 10], 2, scorer)
    assert [c.labels_added for c in run.checkpoints] == [0, 3, 5, 6]
    assert [c.attained for c in run.checkpoints] == [True, True, True, False]
    assert training_sizes == [3, 3, 5, 6, 6, 8, 8, 9]
    assert len(set(run.picked_ids)) == 6
    assert not evaluation_ids.intersection(run.picked_ids)
    assert all(set(c.story_ids) == evaluation_ids for c in run.checkpoints)
    assert len(fold.train_stories) == 3
    assert set(training_times[-1]) == set(range(9))


@pytest.mark.parametrize("rule", ["entropy", "up_margin"])
def test_missing_probabilities_fail(rule: Rule) -> None:
    with pytest.raises(RuntimeError, match="three-class probabilities"):
        select(
            rule,
            np.ones(2),
            None,
            np.eye(2),
            np.eye(2),
            np.array([0, 2]),
            1,
            np.random.default_rng(0),
        )
    with pytest.raises(RuntimeError, match="three-class probabilities"):
        select(
            "mix",
            np.ones(2),
            None,
            np.eye(2),
            np.eye(2),
            np.array([0, 2]),
            1,
            np.random.default_rng(0),
            rule,
        )


def test_metrics_hand_example() -> None:
    labels = np.array([2, 0, 1, 2])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    result = metrics(labels, scores)
    assert result.auc == roc_auc_score(labels == 2, scores) == 0.75
    assert result.precision_at_20 == 0.5
    assert result.up_rate_top_decile == 1.0
    long_labels = np.array([2] * 10 + [0] * 15)
    long_scores = np.arange(25, dtype=float)
    assert metrics(long_labels, long_scores).precision_at_20 == 5 / 20
    assert metrics(long_labels, long_scores).up_rate_top_decile == 0
    assert metrics(np.zeros(4), scores).auc is None


def test_identical_bootstrap_zero_width() -> None:
    labels = np.array([0, 1, 2, 2, 0, 2])
    scores = np.array([0.1, 0.3, 0.9, 0.8, 0.2, 0.4])
    result = paired_bootstrap([Pair(labels, [scores], [scores])])
    assert all(
        (ci.difference, ci.low, ci.high) == (0.0, 0.0, 0.0) for ci in result.values()
    )


def test_bootstrap_paired_and_seed_averaged() -> None:
    labels = np.array([0, 2, 0, 2])
    good = np.array([0.0, 1.0, 0.2, 0.8])
    bad = -good
    result = paired_bootstrap([Pair(labels, [good, bad], [bad, good])], resamples=10)
    assert all(
        (ci.difference, ci.low, ci.high) == (0.0, 0.0, 0.0) for ci in result.values()
    )


def test_summary_excludes_unattainable_budgets() -> None:
    fold = fake_fold()
    runs = [simulate(fold, 1, "random", s, [10], 2, fake_scorer) for s in range(2)]
    summaries = summarize(runs, [10], ["random"], 2)
    assert summaries[0].attained_folds == 1
    assert summaries[1].attained_folds == 0
    assert summaries[1].mean["auc"] is None
