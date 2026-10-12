from __future__ import annotations

import numpy as np
import pytest
from scripts.eval_one_classifier import ClassifierSpec, OneClassifier
from scripts.eval_single_preference_model import (
    fit_lexical_inputs,
    KernelSpec,
    SingleKernelSVC,
)
from database import Story


@pytest.mark.parametrize("family", ["logistic", "linear_svc", "histogram", "mlp"])
def test_one_classifier_learns_classes_from_joined_inputs(family: str) -> None:
    from typing import cast
    from scripts.eval_one_classifier import Family

    rng = np.random.default_rng(20261007)
    labels = np.repeat([0, 1, 2], 40).tolist()
    training = [
        Story(
            id=i,
            title=["apple orchard", "rocket orbit", "violin concert"][y],
            url="",
            score=1,
            time=1,
            text_content="",
        )
        for i, y in enumerate(labels)
    ]
    candidates = [
        Story(id=200 + i, title=title, url="", score=1, time=1, text_content="")
        for i, title in enumerate(["apple orchard", "rocket orbit", "violin concert"])
    ]
    centers = np.eye(3) * 5
    features = np.hstack(
        [centers[labels] + rng.normal(0, 0.1, (120, 3)), np.zeros((120, 2))]
    )
    original = features.copy()
    lexical = fit_lexical_inputs(training, labels, candidates)
    model = OneClassifier(
        c=4,
        gamma=0.03,
        chunk_size=2,
        spec=ClassifierSpec(cast(Family, family)),
        lexical=lexical,
        metadata_columns=2,
    )
    model.fit(features, labels, sample_weight=np.ones(120))
    inputs = np.hstack([centers, np.zeros((3, 2))])
    scores = model.decision_function(inputs)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    np.testing.assert_array_equal(scores.argmax(axis=1), [0, 1, 2])
    np.testing.assert_array_equal(features, original)
    if family != "linear_svc":
        np.testing.assert_allclose(np.exp(scores).sum(axis=1), 1)
    if model.pca is not None:
        before = model.pca.components_.copy()
        model.decision_function(inputs * 100)
        np.testing.assert_array_equal(model.pca.components_, before)


def test_embedding_block_weight_preserves_metadata_and_input() -> None:
    x = np.arange(24, dtype=float).reshape(4, 6)
    original = x.copy()
    model = SingleKernelSVC(
        c=4,
        gamma=0.03,
        chunk_size=2,
        spec=KernelSpec("linear", embedding_weight=4),
        metadata_columns=2,
    )
    weighted = model.weight_embeddings(x)
    np.testing.assert_array_equal(weighted[:, :4], x[:, :4] * 2)
    np.testing.assert_array_equal(weighted[:, 4:], x[:, 4:])
    np.testing.assert_array_equal(x, original)


def test_title_overlap_audit_handles_case_and_empty_titles() -> None:
    from scripts.eval_single_preference_model import title_overlap_ids

    def story(sid: int, title: str) -> Story:
        return Story(id=sid, title=title, url="", score=1, time=1, text_content="")

    training = [story(1, "Some   Story"), story(2, "")]
    candidates = [story(3, "some story"), story(4, ""), story(5, "different story")]
    assert title_overlap_ids(training, candidates) == (3,)


@pytest.mark.parametrize("deduplicate", [False, True])
def test_up_down_rank_keeps_actual_probabilities_through_serving_scorer(
    deduplicate: bool,
) -> None:
    from dataclasses import replace
    from unittest.mock import patch
    from pipeline import Config
    from scripts import eval_one_classifier as one
    from scripts import eval_ranker_variants as evaluator

    labels = [0, 1, 2] * 12
    centers = np.eye(3, dtype=np.float32)
    training = [
        Story(
            id=i + 1,
            title=["apple orchard", "rocket orbit", "violin concert"][y],
            url=f"https://example.com/{i}",
            score=1,
            time=1,
            text_content="",
        )
        for i, y in enumerate(labels)
    ]
    training.append(replace(training[0], id=90, title="rocket orbit"))
    labels.append(1)
    candidates = [
        Story(
            id=100 + i,
            title=title,
            url=f"https://example.com/c{i}",
            score=1,
            time=1,
            text_content="",
        )
        for i, title in enumerate(["apple orchard", "rocket orbit", "violin concert"])
    ]
    fold = evaluator.FoldData(
        candidates=candidates,
        cand_emb=centers,
        train_stories=training,
        test_stories=candidates,
        test_actions=np.array([0, 1, 2]),
        train_vote_times=np.ones(len(labels)),
        x_train_base=centers[labels],
        x_cand_base=centers,
        y_train=np.array(labels),
        tier2_scores=np.zeros(3),
        train_emb=centers[labels],
    )
    defaults = Config()
    config = replace(
        defaults,
        model=replace(
            defaults.model,
            linear_blend_enabled=False,
            svm_precomputed_enabled=True,
            side_embedding_enabled=False,
            deduplicate_training_feedback=deduplicate,
            min_up_for_svm=10,
            min_down_for_svm=10,
            tier2_blend_window=1,
            tier3_blend_window=1,
        ),
    )
    models: list[OneClassifier] = []

    class TrackingClassifier(OneClassifier):
        def fit(
            self, features: np.ndarray, labels: list[int], *, sample_weight: np.ndarray
        ) -> TrackingClassifier:
            super().fit(features, labels, sample_weight=sample_weight)
            models.append(self)
            return self

    blended_config = replace(
        config, model=replace(config.model, linear_blend_enabled=True)
    )
    control_scores, control_probabilities = evaluator._production_scores(
        fold, blended_config
    )

    def evaluate(argv: list[str]) -> None:
        baseline_scores, baseline_probabilities = evaluator._production_scores(
            fold, blended_config
        )
        np.testing.assert_array_equal(baseline_scores, control_scores)
        np.testing.assert_array_equal(baseline_probabilities, control_probabilities)
        assert not models  # The production control keeps its own estimator.
        scores, probabilities = evaluator._production_scores(fold, config)
        assert probabilities is not None
        assert models[-1].last_probabilities is not None
        np.testing.assert_allclose(
            probabilities, models[-1].last_probabilities, atol=1e-7
        )
        assert models[-1].last_decision is not None
        np.testing.assert_allclose(
            models[-1].last_decision[:, 2],
            probabilities[:, 2] - probabilities[:, 0],
            atol=1e-7,
        )
        from pipeline.ranking import _minmax01

        np.testing.assert_allclose(
            scores,
            _minmax01(probabilities[:, 2] - probabilities[:, 0]),
            atol=1e-7,
        )
        assert models[-1].lexical is not None
        if deduplicate:
            assert 1 not in models[-1].lexical.training_ids
        else:
            assert 1 in models[-1].lexical.training_ids
        assert 90 in models[-1].lexical.training_ids

    with (
        patch.object(one, "OneClassifier", TrackingClassifier),
        patch.object(evaluator, "main", evaluate),
    ):
        one.run_evaluation([], ClassifierSpec("logistic", score="up_down"))
    assert len(models) == 1


def test_vector_cache_preserves_values_and_isolates_mutations() -> None:
    from scripts.eval_single_preference_model import memoized_vector_math

    calls: list[int] = []

    def operation(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        calls.append(1)
        return x[indices].sum(axis=1)

    cached = memoized_vector_math(operation)
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    first = cached(x, np.array([0, 2]))
    expected = first.copy()
    first[:] = -100
    np.testing.assert_array_equal(cached(x.copy(), np.array([0, 2])), expected)
    assert len(calls) == 1
    changed = x.copy()
    changed[0, 0] = 100
    np.testing.assert_array_equal(
        cached(changed, np.array([0, 2])), operation(changed, np.array([0, 2]))
    )
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("feature_set", "numeric_row"),
    [
        ("all", [6, 12, 18, 24, 33, 36]),
        ("numeric", [6, 12, 18, 24, 33, 36]),
        ("embedding_words", [6, 12, 18, 24]),
        ("metadata_words", [33, 36]),
        ("words", []),
    ],
)
def test_feature_subsets_keep_only_the_requested_blocks(
    feature_set: str, numeric_row: list[int]
) -> None:
    from typing import cast
    from scipy.sparse import csr_matrix
    from scripts.eval_one_classifier import FeatureSet

    stories = [
        Story(id=i, title="apple orchard", url="", score=1, time=1, text_content="")
        for i in range(6)
    ]
    lexical = fit_lexical_inputs(stories, [0, 1, 2] * 2, stories)
    model = OneClassifier(
        c=4,
        gamma=0.03,
        chunk_size=2,
        spec=ClassifierSpec(
            "logistic",
            features=cast(FeatureSet, feature_set),
            embedding_weight=4,
            numeric_scale=3,
            word_scale=2,
        ),
        lexical=lexical,
        metadata_columns=2,
    )
    features = np.tile([1, 2, 3, 4, 11, 12], (6, 1)).astype(float)
    matrix = model.inputs(features, training=True)
    assert isinstance(matrix, csr_matrix)
    actual = matrix.toarray()
    numeric = np.tile(numeric_row, (6, 1))
    expected = (
        numeric
        if feature_set == "numeric"
        else np.hstack([numeric, lexical.training.toarray() * 2])
    )
    np.testing.assert_allclose(actual, expected)


def test_metadata_scale_touches_only_the_metadata_block() -> None:
    from scipy.sparse import csr_matrix

    stories = [
        Story(id=i, title="apple orchard", url="", score=1, time=1, text_content="")
        for i in range(6)
    ]
    lexical = fit_lexical_inputs(stories, [0, 1, 2] * 2, stories)
    features = np.tile([1, 2, 3, 4, 11, 12], (6, 1)).astype(float)

    def inputs(metadata_scale: float) -> np.ndarray:
        model = OneClassifier(
            c=4,
            gamma=0.03,
            chunk_size=2,
            spec=ClassifierSpec(
                "logistic",
                embedding_weight=4,
                numeric_scale=3,
                metadata_scale=metadata_scale,
            ),
            lexical=lexical,
            metadata_columns=2,
        )
        matrix = model.inputs(features, training=True)
        assert isinstance(matrix, csr_matrix)
        return matrix.toarray()

    full, half = inputs(1.0), inputs(0.5)
    np.testing.assert_array_equal(half[:, :4], full[:, :4])
    np.testing.assert_allclose(half[:, 4:6], full[:, 4:6] / 2)
    np.testing.assert_array_equal(half[:, 6:], full[:, 6:])
    with pytest.raises(ValueError, match="block weights"):
        ClassifierSpec("logistic", metadata_scale=0)


def test_missing_class_placeholder_fails_before_fitting() -> None:
    stories = [
        Story(id=i, title="apple orchard", url="", score=1, time=1, text_content="")
        for i in range(6)
    ]
    labels = [0, 2] * 3
    model = OneClassifier(
        c=4,
        gamma=0.03,
        chunk_size=2,
        spec=ClassifierSpec("logistic"),
        lexical=fit_lexical_inputs(stories, labels, stories),
        metadata_columns=2,
    )
    with pytest.raises(ValueError, match="lexical labels do not align"):
        model.fit(np.zeros((7, 6)), labels + [1], sample_weight=np.ones(7))
