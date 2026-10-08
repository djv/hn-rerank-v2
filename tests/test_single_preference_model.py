from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
from unittest.mock import patch

from database import Story
from pipeline import Config, ranking
from pipeline.ranking import PrecomputedRbfSVC
from scripts.eval_single_preference_model import (
    KernelSpec,
    SingleKernelSVC,
    restrict_feedback,
    run_evaluation,
)
from scripts import eval_ranker_variants as evaluator


def test_zero_linear_share_reproduces_production_rbf() -> None:
    rng = np.random.default_rng(1843)
    x = rng.normal(size=(36, 9))
    y = [0, 1, 2] * 12
    weights = np.ones(36)
    base = PrecomputedRbfSVC(c=4.0, gamma=0.03, chunk_size=5)
    other = SingleKernelSVC(
        c=4.0, gamma=0.03, chunk_size=7, spec=KernelSpec("hybrid", 0.0)
    )
    base.fit(x, y, sample_weight=weights)
    other.fit(x, y, sample_weight=weights)
    candidates = rng.normal(size=(13, 9))
    np.testing.assert_allclose(
        base.decision_function(candidates),
        other.decision_function(candidates),
        atol=1e-10,
    )


def test_hybrid_kernel_is_symmetric_psd_and_uses_training_scale() -> None:
    rng = np.random.default_rng(914)
    x = rng.normal(size=(30, 8))
    original = x.copy()
    model = SingleKernelSVC(
        c=1.0, gamma=0.01, chunk_size=4, spec=KernelSpec("hybrid", 0.3)
    )
    model.fit(x, [0, 1, 2] * 10, sample_weight=np.ones(30))
    kernel = model.kernel_matrix(x, x)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10
    assert np.diag(kernel).mean() == pytest.approx(1.0)
    scale = model.linear_scale
    candidates = rng.normal(size=(11, 8)) * 100
    chunked = model.decision_function(candidates)
    direct = model._svc.decision_function(model.kernel_matrix(candidates, x))
    np.testing.assert_allclose(chunked, direct)
    assert model.linear_scale == scale
    np.testing.assert_array_equal(x, original)


def test_feedback_cohort_retains_alignment_without_changing_rows() -> None:
    stories = [
        Story(id=i, title=str(i), url="", score=1, time=1, text_content="")
        for i in range(3)
    ]
    feedback = (stories, [2, 0, 1], [11.0, 22.0, 33.0])
    kept = restrict_feedback(feedback, frozenset({0, 2}))
    assert [story.id for story in kept[0]] == [0, 2]
    assert kept[1:] == ([2, 1], [11.0, 33.0])
    assert len(feedback[0]) == 3


@pytest.mark.parametrize("share", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_linear_share_is_rejected(share: float) -> None:
    with pytest.raises(ValueError, match="linear_share"):
        KernelSpec("hybrid", share)


def test_experimental_kernel_routes_only_to_single_model_variant() -> None:
    original_factory = ranking.PrecomputedRbfSVC
    config = Config()
    baseline = replace(
        config,
        model=replace(
            config.model, linear_blend_enabled=True, svm_precomputed_enabled=True
        ),
    )
    single = replace(
        baseline, model=replace(baseline.model, linear_blend_enabled=False)
    )
    factories = []

    def score(
        fold: evaluator.FoldData, config: Config, source_db: object = None
    ) -> tuple[np.ndarray, None]:
        factories.append(ranking.PrecomputedRbfSVC)
        return np.array([0.5]), None

    def evaluate(argv: list[str]) -> None:
        fold = evaluator.FoldData(
            candidates=[],
            cand_emb=np.empty((0, 2)),
            train_stories=[],
            test_stories=[],
            test_actions=np.empty(0),
            train_vote_times=np.empty(0),
            x_train_base=np.empty((0, 2)),
            x_cand_base=np.empty((0, 2)),
            y_train=np.empty(0),
            tier2_scores=np.empty(0),
        )
        evaluator._production_scores(fold, baseline)
        evaluator._production_scores(fold, single)

    with (
        patch.object(evaluator, "_production_scores", score),
        patch.object(evaluator, "main", evaluate),
    ):
        run_evaluation([], KernelSpec("hybrid", 0.6))
    assert factories[0] is original_factory
    experimental = factories[1](c=4, gamma=0.03, chunk_size=4)
    assert isinstance(experimental, SingleKernelSVC)
    assert experimental.spec == KernelSpec("hybrid", 0.6)
    assert ranking.PrecomputedRbfSVC is original_factory


def test_cohort_with_external_snapshot_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        run_evaluation(
            ["--embeddings-file=data.npz"], KernelSpec("rbf"), frozenset({1})
        )


def test_continuous_pairwise_margins_prefer_correct_cluster() -> None:
    rng = np.random.default_rng(912)
    centers = np.array([[-5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    training = np.concatenate(
        [center + rng.normal(0, 0.15, (12, 2)) for center in centers]
    )
    model = SingleKernelSVC(
        c=4, gamma=0.03, chunk_size=2, spec=KernelSpec("rbf", score_form="margin")
    )
    model.fit(training, [0] * 12 + [1] * 12 + [2] * 12, sample_weight=np.ones(36))
    scores = model.decision_function(centers)
    np.testing.assert_array_equal(scores.argmax(axis=1), [0, 1, 2])
    assert scores[2, 2] > scores[1, 2] > scores[0, 2]
    np.testing.assert_allclose(scores.sum(axis=1), 0, atol=1e-12)


def test_anchored_hybrid_preserves_rbf_at_zero_linear_strength() -> None:
    rng = np.random.default_rng(172)
    training = rng.normal(size=(30, 8))
    base = PrecomputedRbfSVC(c=16, gamma=0.01, chunk_size=4)
    anchored = SingleKernelSVC(
        c=16, gamma=0.01, chunk_size=3, spec=KernelSpec("anchored", 0)
    )
    labels = [0, 1, 2] * 10
    base.fit(training, labels, sample_weight=np.ones(30))
    anchored.fit(training, labels, sample_weight=np.ones(30))
    candidates = rng.normal(size=(11, 8))
    np.testing.assert_allclose(
        base.decision_function(candidates), anchored.decision_function(candidates)
    )


def word_story(sid: int, title: str) -> Story:
    return Story(id=sid, title=title, url="", score=1, time=1, text_content="")


def test_candidate_words_cannot_change_training_idf_or_columns() -> None:
    from scripts.eval_single_preference_model import fit_lexical_inputs

    training = [
        word_story(i, "apple orchard" if i < 3 else "rocket orbit") for i in range(6)
    ]
    labels = [0] * 3 + [2] * 3
    first = fit_lexical_inputs(training, labels, [word_story(20, "apple orchard")])
    second = fit_lexical_inputs(
        training, labels, [word_story(20, "unseen quasar " * 30)]
    )
    np.testing.assert_array_equal(first.kept_columns, second.kept_columns)
    np.testing.assert_array_equal(first.idf, second.idf)
    np.testing.assert_array_equal(first.training.toarray(), second.training.toarray())
    assert (first.candidates != second.candidates).nnz > 0


def test_one_classifier_uses_words_when_semantics_are_identical() -> None:
    from scripts.eval_single_preference_model import fit_lexical_inputs

    training = [
        word_story(i, ["apple orchard", "rocket orbit", "violin concert"][i // 6])
        for i in range(18)
    ]
    candidates = [
        word_story(30 + i, title)
        for i, title in enumerate(["apple orchard", "rocket orbit", "violin concert"])
    ]
    labels = [0] * 6 + [1] * 6 + [2] * 6
    lexical = fit_lexical_inputs(training, labels, candidates)
    model = SingleKernelSVC(
        c=4, gamma=0.03, chunk_size=2, spec=KernelSpec("rbf", word_c=4), lexical=lexical
    )
    model.fit(np.zeros((18, 3)), labels, sample_weight=np.ones(18))
    scores = model.decision_function(np.zeros((3, 3)))
    np.testing.assert_array_equal(scores.argmax(axis=1), [0, 1, 2])
    with pytest.raises(ValueError, match="candidate lexical"):
        model.decision_function(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="align"):
        model.fit(np.zeros((18, 3)), list(reversed(labels)), sample_weight=np.ones(18))


@pytest.mark.parametrize("word_c", [-0.1, float("nan"), float("inf")])
def test_invalid_word_strength_is_rejected(word_c: float) -> None:
    with pytest.raises(ValueError, match="word_c"):
        KernelSpec("rbf", word_c=word_c)


def test_word_kernel_uses_serving_database_order_and_keeps_baseline() -> None:
    from database import Database

    db = Database(":memory:")
    actual = [
        word_story(i, "shared apple" if i < 2 else "shared rocket") for i in range(4)
    ]
    labels = [0, 0, 2, 2]
    candidates = [word_story(10, "shared apple"), word_story(11, "shared rocket")]
    fold = evaluator.FoldData(
        candidates=candidates,
        cand_emb=np.zeros((2, 2)),
        train_stories=list(reversed(actual)),
        test_stories=candidates,
        test_actions=np.array([0, 2]),
        train_vote_times=np.ones(4),
        x_train_base=np.zeros((4, 2)),
        x_cand_base=np.zeros((2, 2)),
        y_train=np.array(list(reversed(labels))),
        tier2_scores=np.zeros(2),
        runtime_db=db,
    )
    defaults = Config()
    base = replace(
        defaults,
        model=replace(
            defaults.model, linear_blend_enabled=True, svm_precomputed_enabled=True
        ),
    )
    single = replace(
        base,
        model=replace(
            base.model, linear_blend_enabled=False, deduplicate_training_feedback=False
        ),
    )
    original_factory = ranking.PrecomputedRbfSVC
    routed = []

    def score(
        fold: evaluator.FoldData, config: Config, source_db: Database | None = None
    ) -> tuple[np.ndarray, None]:
        assert fold.runtime_db is db
        if config.model.linear_blend_enabled:
            assert ranking.PrecomputedRbfSVC is original_factory
        else:
            model = ranking.PrecomputedRbfSVC(c=4, gamma=0.03, chunk_size=2)
            assert isinstance(model, SingleKernelSVC)
            assert model.lexical is not None
            assert model.lexical.training_ids == tuple(s.id for s in actual)
            assert model.lexical.labels == tuple(labels)
            assert (
                tuple(s.id for s in db.get_feedback_for_training()[0])
                == model.lexical.training_ids
            )
            assert (
                tuple(s.id for s in db.get_feedback_for_training()[0])
                == model.lexical.training_ids
            )
            routed.append(model)
        return np.zeros(2), None

    def evaluate(argv: list[str]) -> None:
        evaluator._production_scores(fold, base)
        evaluator._production_scores(fold, single)

    try:
        with (
            patch.object(
                Database,
                "get_feedback_for_training",
                side_effect=[
                    (actual, labels, [1.0] * 4),
                    (list(reversed(actual)), list(reversed(labels)), [1.0] * 4),
                ],
            ),
            patch.object(evaluator, "_production_scores", score),
            patch.object(evaluator, "main", evaluate),
        ):
            run_evaluation([], KernelSpec("rbf", word_c=1))
        assert len(routed) == 1
        assert ranking.PrecomputedRbfSVC is original_factory
    finally:
        db.close()
