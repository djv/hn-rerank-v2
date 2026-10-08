#!/usr/bin/env python3
"""Offline classifier/feature bakeoff through the unchanged production evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, Story
from pipeline import Config, ranking
from scripts import eval_ranker_variants as evaluator
from scripts.eval_single_preference_model import (
    LexicalInputs,
    fit_lexical_inputs,
    read_cohort,
    restrict_feedback,
    reused_vector_math,
    title_overlap_ids,
)

Family = Literal["logistic", "linear_svc", "histogram", "mlp"]
FeatureSet = Literal["all", "embedding_words", "numeric", "words", "metadata_words"]
Estimator = (
    LogisticRegression | LinearSVC | HistGradientBoostingClassifier | MLPClassifier
)
Matrix = NDArray[np.float64] | csr_matrix


@dataclass(frozen=True)
class ClassifierSpec:
    family: Family
    features: FeatureSet = "all"
    word_scale: float = 1.0
    max_leaf_nodes: int = 15
    l2: float = 1.0
    hidden: int = 32
    score: Literal["up", "up_down"] = "up"
    embedding_weight: float = 1.0
    numeric_scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.word_scale) or self.word_scale <= 0:
            raise ValueError("word_scale must be positive and finite")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be nonnegative and finite")
        if self.max_leaf_nodes < 2 or self.hidden < 1:
            raise ValueError("model size must be positive")
        if not all(
            np.isfinite(v) and v > 0
            for v in (self.embedding_weight, self.numeric_scale)
        ):
            raise ValueError("block weights must be positive and finite")


class OneClassifier(ranking.PrecomputedRbfSVC):
    """One supervised fit; training-only projections for dense nonlinear models."""

    def __init__(
        self,
        *,
        c: float,
        gamma: float,
        chunk_size: int,
        spec: ClassifierSpec,
        lexical: LexicalInputs | None,
        metadata_columns: int,
    ) -> None:
        self.spec = spec
        self.lexical = lexical
        self.metadata_columns = metadata_columns
        self.pca: PCA | None = None
        self.svd: TruncatedSVD | None = None
        self.last_probabilities: NDArray[np.float64] | None = None
        self.last_decision: NDArray[np.float64] | None = None
        self.estimator: Estimator
        if spec.family == "logistic":
            self.estimator = LogisticRegression(C=c, max_iter=2000, random_state=0)
        elif spec.family == "linear_svc":
            self.estimator = LinearSVC(C=c, dual="auto", max_iter=5000, random_state=0)
        elif spec.family == "histogram":
            self.estimator = HistGradientBoostingClassifier(
                max_leaf_nodes=spec.max_leaf_nodes,
                max_iter=150,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=spec.l2,
                early_stopping=False,
                random_state=0,
            )
        else:
            self.estimator = MLPClassifier(
                hidden_layer_sizes=(spec.hidden,),
                alpha=1.0 / c,
                max_iter=300,
                early_stopping=False,
                random_state=0,
            )

    @property
    def classes_(self) -> NDArray[np.int64]:
        return np.asarray(self.estimator.classes_, dtype=np.int64)

    def inputs(self, features: NDArray[np.float64], *, training: bool) -> Matrix:
        features = np.asarray(features, dtype=np.float64)
        if features.shape[1] <= self.metadata_columns:
            raise ValueError("feature layout lacks embedding columns")
        embeddings = features[:, : -self.metadata_columns] * np.sqrt(
            self.spec.embedding_weight
        )
        metadata = features[:, -self.metadata_columns :]
        dense_model = self.spec.family in {"histogram", "mlp"}
        numeric_parts: list[NDArray[np.float64]] = []
        if self.spec.features in {"all", "embedding_words", "numeric"}:
            if dense_model:
                if training:
                    self.pca = PCA(
                        n_components=min(64, len(features) - 1, embeddings.shape[1]),
                        svd_solver="randomized",
                        random_state=0,
                    )
                    embeddings = self.pca.fit_transform(embeddings)
                else:
                    assert self.pca is not None
                    embeddings = self.pca.transform(embeddings)
            numeric_parts.append(embeddings)
        if self.spec.features in {"all", "numeric", "metadata_words"}:
            numeric_parts.append(metadata)
        numeric = (
            np.hstack(numeric_parts) if numeric_parts else np.empty((len(features), 0))
        )
        numeric = numeric * self.spec.numeric_scale
        if self.spec.features == "numeric":
            return numeric if dense_model else csr_matrix(numeric)
        if self.lexical is None:
            raise ValueError("word features require fitted lexical inputs")
        words = self.lexical.training if training else self.lexical.candidates
        if len(features) != words.shape[0]:
            raise ValueError("lexical and numeric rows do not align")
        if dense_model:
            if training:
                self.svd = TruncatedSVD(
                    n_components=min(32, words.shape[1] - 1, words.shape[0] - 1),
                    random_state=0,
                )
                projected = self.svd.fit_transform(words)
            else:
                assert self.svd is not None
                projected = self.svd.transform(words)
            return np.hstack([numeric, projected * self.spec.word_scale])
        return hstack([csr_matrix(numeric), words * self.spec.word_scale], format="csr")

    def fit(
        self,
        features: NDArray[np.float64],
        labels: list[int],
        *,
        sample_weight: NDArray[np.float64],
    ) -> OneClassifier:
        if self.lexical is not None and tuple(labels) != self.lexical.labels:
            raise ValueError("lexical labels do not align with classifier")
        self.estimator.fit(
            self.inputs(features, training=True), labels, sample_weight=sample_weight
        )
        return self

    def decision_function(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        inputs = self.inputs(features, training=False)
        if isinstance(self.estimator, LinearSVC):
            decision = np.asarray(
                self.estimator.decision_function(inputs), dtype=np.float64
            )
        else:
            self.last_probabilities = np.asarray(
                self.estimator.predict_proba(inputs), dtype=np.float64
            )
            decision = np.log(np.clip(self.last_probabilities, 1e-12, 1.0))
        if self.spec.score == "up_down":
            if isinstance(self.estimator, LinearSVC):
                raise ValueError("up_down score requires probability classifier")
            classes = list(self.classes_)
            probabilities = np.exp(decision)
            # Serving consumes the up column for ranking. Other columns stay
            # log-probabilities. The scoped serving mapper below preserves the
            # estimator's actual probabilities for entropy and diagnostic fields.
            decision[:, classes.index(2)] = (
                probabilities[:, classes.index(2)] - probabilities[:, classes.index(0)]
            )
        self.last_decision = decision
        return decision


@dataclass(frozen=True)
class ScoringTiming:
    baseline: bool
    c: float
    seconds: float
    candidate_ids: tuple[int, ...]
    title_overlap_ids: tuple[int, ...]


def run_evaluation(
    argv: Sequence[str], spec: ClassifierSpec, cohort: frozenset[int] | None = None
) -> None:
    if cohort is not None and any(a.startswith("--embeddings-file") for a in argv):
        raise ValueError("cohort cannot be combined with --embeddings-file")
    original_score = evaluator._production_scores
    original_feedback = Database.get_feedback_for_training
    original_softmax = ranking._softmax_rows
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    timings: list[ScoringTiming] = []

    def score(
        fold: evaluator.FoldData, config: Config, source_db: Database | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        started = time.perf_counter()
        baseline = config.model.linear_blend_enabled
        try:
            if baseline:
                return original_score(fold, config, source_db)
            if not config.model.svm_precomputed_enabled:
                raise ValueError("classifier adapter requires precomputed route")
            with evaluator._fold_database(fold, config, source_db) as db:
                frozen_rows = db.get_feedback_for_training()
                rows = frozen_rows
                if config.model.deduplicate_training_feedback:
                    from pipeline.feedback import deduplicate_feedback

                    rows = deduplicate_feedback(*rows)
                lexical = (
                    fit_lexical_inputs(rows[0], rows[1], fold.candidates)
                    if spec.features != "numeric"
                    else None
                )
                metadata_columns = (
                    10
                    + 4 * int(config.model.publication_affinity_enabled)
                    + 2 * int(config.model.engagement_features_enabled)
                )
                reader = Database.get_feedback_for_training
                built: list[OneClassifier] = []

                def build(*, c: float, gamma: float, chunk_size: int) -> OneClassifier:
                    model = OneClassifier(
                        c=c,
                        gamma=gamma,
                        chunk_size=chunk_size,
                        spec=spec,
                        lexical=lexical,
                        metadata_columns=metadata_columns,
                    )
                    built.append(model)
                    return model

                def probabilities(decision: np.ndarray) -> np.ndarray:
                    if (
                        built
                        and built[-1].last_decision is decision
                        and built[-1].last_probabilities is not None
                    ):
                        return built[-1].last_probabilities.copy()
                    return original_softmax(decision)

                def ordered_feedback(
                    current: Database, user_id: int | None = None
                ) -> tuple[list[Story], list[int], list[float]]:
                    if current is db and user_id is None:
                        return (
                            list(frozen_rows[0]),
                            list(frozen_rows[1]),
                            list(frozen_rows[2]),
                        )
                    return reader(current, user_id=user_id)

                with (
                    patch.object(
                        Database, "get_feedback_for_training", ordered_feedback
                    ),
                    patch.object(
                        ranking,
                        "PrecomputedRbfSVC",
                        build,
                    ),
                    patch.object(ranking, "_softmax_rows", probabilities),
                ):
                    return original_score(
                        replace(fold, runtime_db=db), config, source_db
                    )
        finally:
            timings.append(
                ScoringTiming(
                    baseline,
                    config.model.svm_c,
                    time.perf_counter() - started,
                    tuple(s.id for s in fold.candidates),
                    title_overlap_ids(fold.train_stories, fold.candidates),
                )
            )

    def feedback(
        db: Database, user_id: int | None = None
    ) -> tuple[list[Story], list[int], list[float]]:
        rows = original_feedback(db, user_id=user_id)
        return restrict_feedback(rows, cohort) if cohort is not None else rows

    with (
        patch.object(evaluator, "_production_scores", side_effect=score),
        patch.object(Database, "get_feedback_for_training", feedback),
        reused_vector_math(),
    ):
        evaluator.main(list(argv))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    output, _ = parser.parse_known_args(argv)
    if output.output is not None:
        report = json.loads(output.output.read_text())
        report["one_classifier_experiment"] = {
            "spec": asdict(spec),
            "driver_sha256": digest,
            "cohort_ids": sorted(cohort) if cohort is not None else None,
            "scoring_timings": [asdict(t) for t in timings],
            "vector_math_cache": True,
        }
        output.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--family",
        choices=("logistic", "linear_svc", "histogram", "mlp"),
        required=True,
    )
    parser.add_argument(
        "--features",
        choices=("all", "embedding_words", "numeric", "words", "metadata_words"),
        default="all",
    )
    parser.add_argument("--word-scale", type=float, default=1.0)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--score", choices=("up", "up_down"), default="up")
    parser.add_argument("--feedback-cohort", type=Path)
    parser.add_argument("--embedding-weight", type=float, default=1.0)
    parser.add_argument("--numeric-scale", type=float, default=1.0)
    parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    spec = ClassifierSpec(
        args.family,
        args.features,
        args.word_scale,
        args.max_leaf_nodes,
        args.l2,
        args.hidden,
        args.score,
        args.embedding_weight,
        args.numeric_scale,
    )
    evaluation_args = (
        args.evaluation_args[1:]
        if args.evaluation_args[:1] == ["--"]
        else args.evaluation_args
    )
    if not evaluation_args:
        parser.error("canonical evaluator arguments required after --")
    run_evaluation(
        evaluation_args,
        spec,
        read_cohort(args.feedback_cohort) if args.feedback_cohort else None,
    )


if __name__ == "__main__":
    main()
