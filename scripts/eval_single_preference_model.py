#!/usr/bin/env python3
"""Evaluate one SVM's linear/nonlinear capacity through the canonical harness.

Production scorers retain their original RBF kernel. Only variants with the
dense preference contribution disabled receive the experimental kernel.
The optional feedback cohort filters reads in memory; no database rows change.
Pass canonical evaluator arguments after ``--``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar
from unittest.mock import patch

import numpy as np
from cachetools import LRUCache
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, Story
from pipeline import Config
from pipeline import linear_blend, ranking
from scripts import eval_ranker_variants as evaluator

P = ParamSpec("P")
R = TypeVar("R")


def vector_argument_key(value: object) -> str:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            f"{array.shape}:{array.dtype.str}:"
            + hashlib.blake2b(memoryview(array), digest_size=16).hexdigest()
        )
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    if isinstance(value, tuple):
        return repr(tuple(vector_argument_key(v) for v in value))
    raise TypeError(f"Unsupported vector-cache argument: {type(value).__name__}")


def memoized_vector_math(function: Callable[P, R]) -> Callable[P, R]:
    """Reuse exact fold-local vector inputs; never reuse transformed labels."""
    cache: LRUCache[tuple[str, ...], R] = LRUCache(maxsize=32)

    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        key = tuple(vector_argument_key(v) for v in args) + tuple(
            f"{k}={vector_argument_key(v)}" for k, v in sorted(kwargs.items())
        )
        if key in cache:
            return copy.deepcopy(cache[key])
        result = function(*args, **kwargs)
        cache[key] = copy.deepcopy(result)
        return result

    return call


@contextmanager
def reused_vector_math() -> Iterator[None]:
    # Neighbor/cluster features depend on the exact input vectors and indices,
    # not on the downstream classifier. Bounded result-only caching avoids
    # repeating their expensive arithmetic for every challenger in a fold.
    with ExitStack() as stack:
        for name in (
            "_knn_mean_and_max",
            "_loocv_knn_features",
            "_positive_cluster_centers",
            "_similarity_to_positive_cluster_centers",
            "_chunked_max_dot",
        ):
            stack.enter_context(
                patch.object(
                    ranking, name, memoized_vector_math(getattr(ranking, name))
                )
            )
        yield


@dataclass(frozen=True)
class KernelSpec:
    family: Literal["rbf", "hybrid", "linear", "anchored"]
    linear_share: float = 0.3
    score_form: Literal["ovr", "margin"] = "ovr"
    word_c: float = 0.0
    embedding_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.linear_share <= 1.0:
            raise ValueError("linear_share must be finite and in [0, 1]")
        if not np.isfinite(self.word_c) or self.word_c < 0:
            raise ValueError("word_c must be finite and nonnegative")
        if not np.isfinite(self.embedding_weight) or self.embedding_weight <= 0:
            raise ValueError("embedding_weight must be positive and finite")


@dataclass(frozen=True)
class LexicalInputs:
    training: csr_matrix
    candidates: csr_matrix
    training_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    labels: tuple[int, ...]
    kept_columns: NDArray[np.bool_]
    idf: NDArray[np.float64]


@dataclass(frozen=True)
class KernelAudit:
    embedding_mean_squared_norm: float
    metadata_mean_squared_norm: float
    linear_scale: float
    word_mean_squared_norm: float | None


def title_overlap_ids(
    training: Sequence[Story], candidates: Sequence[Story]
) -> tuple[int, ...]:
    def normalized(title: str) -> str:
        return " ".join(ranking.clean_text(title).casefold().split())

    titles = {normalized(s.title) for s in training} - {""}
    return tuple(s.id for s in candidates if normalized(s.title) in titles)


def fit_lexical_inputs(
    training: Sequence[Story], labels: Sequence[int], candidates: Sequence[Story]
) -> LexicalInputs:
    """Production's hashed 1-2 grams, training-only min-df and IDF, no classifier."""
    if len(training) != len(labels):
        raise ValueError("lexical training stories and labels do not align")
    counts = linear_blend.count_rows(training)
    keep = np.asarray((counts > 0).sum(axis=0)).ravel() >= 2
    if not keep.any():
        raise ValueError("lexical kernel has no repeated training terms")
    idf = TfidfTransformer(sublinear_tf=True)
    train_words = idf.fit_transform(counts[:, keep]).tocsr()
    candidate_words = idf.transform(
        linear_blend.count_rows(candidates)[:, keep]
    ).tocsr()
    return LexicalInputs(
        training=train_words,
        candidates=candidate_words,
        training_ids=tuple(s.id for s in training),
        candidate_ids=tuple(s.id for s in candidates),
        labels=tuple(labels),
        kept_columns=keep,
        idf=np.asarray(idf.idf_, dtype=np.float64),
    )


class SingleKernelSVC(ranking.PrecomputedRbfSVC):
    """One classifier, with an additive linear/RBF kernel and bounded inference.

    The linear kernel's average training diagonal is normalized to one.
    Its scale is learned from training only; input embeddings are not scaled.
    Production has already scaled only metadata columns before this boundary.
    """

    def __init__(
        self,
        *,
        c: float,
        gamma: float,
        chunk_size: int,
        spec: KernelSpec,
        lexical: LexicalInputs | None = None,
        metadata_columns: int = 0,
    ) -> None:
        super().__init__(c=c, gamma=gamma, chunk_size=chunk_size)
        self.spec = spec
        self.lexical = lexical
        self.metadata_columns = metadata_columns
        self.audit: KernelAudit | None = None
        if spec.word_c > 0 and lexical is None:
            raise ValueError("positive word_c requires fitted lexical inputs")
        self.linear_scale = 1.0
        if spec.score_form == "margin":
            self._svc.decision_function_shape = "ovo"

    def kernel_matrix(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        share = 1.0 if self.spec.family == "linear" else self.spec.linear_share
        if self.spec.family == "rbf":
            share = 0.0
        if share == 0.0:
            return rbf_kernel(x, y, gamma=self.gamma)
        linear = (x @ y.T) / self.linear_scale
        if self.spec.family == "anchored":
            # Preserve the RBF's C; linear_share specifies the effective
            # linear C, namely C times the raw linear kernel coefficient.
            return rbf_kernel(x, y, gamma=self.gamma) + (share / self._svc.C) * (
                x @ y.T
            )
        if share == 1.0:
            return linear
        return (1.0 - share) * rbf_kernel(x, y, gamma=self.gamma) + share * linear

    def fit(
        self,
        features: NDArray[np.float64],
        labels: list[int],
        *,
        sample_weight: NDArray[np.float64],
    ) -> SingleKernelSVC:
        self._training_features = self.weight_embeddings(features)
        self.linear_scale = max(
            float(np.square(self._training_features).sum(axis=1).mean()), 1e-12
        )
        if self.metadata_columns:
            self.audit = KernelAudit(
                float(
                    np.square(self._training_features[:, : -self.metadata_columns])
                    .sum(axis=1)
                    .mean()
                ),
                float(
                    np.square(self._training_features[:, -self.metadata_columns :])
                    .sum(axis=1)
                    .mean()
                ),
                self.linear_scale,
                float(
                    np.asarray(
                        self.lexical.training.multiply(self.lexical.training).sum(
                            axis=1
                        )
                    ).mean()
                )
                if self.lexical is not None
                else None,
            )
        kernel = self.kernel_matrix(self._training_features, self._training_features)
        if self.spec.word_c > 0:
            assert self.lexical is not None
            if (
                len(features) != self.lexical.training.shape[0]
                or tuple(labels) != self.lexical.labels
            ):
                raise ValueError(
                    "lexical rows or labels do not align with SVM training"
                )
            kernel += (self.spec.word_c / self._svc.C) * (
                self.lexical.training @ self.lexical.training.T
            ).toarray()
        self._svc.fit(kernel, labels, sample_weight=sample_weight)
        return self

    def weight_embeddings(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        weighted = np.array(features, dtype=np.float64, copy=True)
        if self.spec.embedding_weight != 1:
            if not 0 < self.metadata_columns < weighted.shape[1]:
                raise ValueError(
                    "embedding weighting requires explicit metadata layout"
                )
            weighted[:, : -self.metadata_columns] *= np.sqrt(self.spec.embedding_weight)
        return weighted

    def decision_function(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._training_features is None:
            raise RuntimeError("fit must run before decision_function")
        features = self.weight_embeddings(features)
        if self.spec.word_c > 0:
            assert self.lexical is not None
            if len(features) != self.lexical.candidates.shape[0]:
                raise ValueError("candidate lexical rows do not align")
        chunks = []
        for start in range(0, len(features), self.chunk_size):
            kernel = self.kernel_matrix(
                features[start : start + self.chunk_size], self._training_features
            )
            if self.spec.word_c > 0:
                assert self.lexical is not None
                kernel += (self.spec.word_c / self._svc.C) * (
                    self.lexical.candidates[start : start + self.chunk_size]
                    @ self.lexical.training.T
                ).toarray()
            decision = self._svc.decision_function(kernel)
            if self.spec.score_form == "margin" and len(self.classes_) > 2:
                # Multiclass OVO margins are positive for the first class
                # in each pair. Sum signed margins without OVR vote tiers.
                continuous = np.zeros((len(kernel), len(self.classes_)))
                column = 0
                for i in range(len(self.classes_)):
                    for j in range(i + 1, len(self.classes_)):
                        continuous[:, i] += decision[:, column]
                        continuous[:, j] -= decision[:, column]
                        column += 1
                decision = continuous
            chunks.append(decision)
        if not chunks:
            width = len(self.classes_) if len(self.classes_) > 2 else 1
            return np.empty((0, width), dtype=np.float64)
        return np.concatenate(chunks, axis=0)


def restrict_feedback(
    feedback: tuple[list[Story], list[int], list[float]], cohort: frozenset[int]
) -> tuple[list[Story], list[int], list[float]]:
    rows = [
        (story, label, timestamp)
        for story, label, timestamp in zip(*feedback, strict=True)
        if story.id in cohort
    ]
    return (
        [story for story, _, _ in rows],
        [label for _, label, _ in rows],
        [timestamp for _, _, timestamp in rows],
    )


def read_cohort(path: Path) -> frozenset[int]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("feedback cohort must be a nonempty JSON array of integer IDs")
    ids: set[int] = set()
    for sid in data:
        if isinstance(sid, bool) or not isinstance(sid, int):
            raise ValueError("feedback cohort must contain only integer IDs")
        ids.add(sid)
    return frozenset(ids)


def run_evaluation(
    argv: Sequence[str], spec: KernelSpec, cohort: frozenset[int] | None = None
) -> None:
    if cohort is not None and any(
        arg == "--embeddings-file" or arg.startswith("--embeddings-file=")
        for arg in argv
    ):
        raise ValueError("feedback cohort cannot be combined with --embeddings-file")
    driver_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_score = evaluator._production_scores
    original_feedback = Database.get_feedback_for_training
    fitted: list[SingleKernelSVC] = []
    title_audits: list[dict[str, tuple[int, ...]]] = []

    def build(
        *,
        c: float,
        gamma: float,
        chunk_size: int,
        lexical: LexicalInputs | None = None,
        metadata_columns: int,
    ) -> SingleKernelSVC:
        model = SingleKernelSVC(
            c=c,
            gamma=gamma,
            chunk_size=chunk_size,
            spec=spec,
            lexical=lexical,
            metadata_columns=metadata_columns,
        )
        fitted.append(model)
        return model

    def score(
        fold: evaluator.FoldData, config: Config, source_db: Database | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        no_dense = (
            not config.model.linear_blend_enabled
            or config.model.linear_blend_dense_weight == 0.0
        )
        if not no_dense or (
            spec.family == "rbf"
            and spec.score_form == "ovr"
            and spec.word_c == 0
            and spec.embedding_weight == 1
        ):
            return original_score(fold, config, source_db)
        if not config.model.svm_precomputed_enabled:
            raise ValueError(
                "experimental kernels require svm_precomputed_enabled=true"
            )
        metadata_columns = (
            10
            + 4 * int(config.model.publication_affinity_enabled)
            + 2 * int(config.model.engagement_features_enabled)
        )
        title_audits.append(
            {
                "candidate_ids": tuple(s.id for s in fold.candidates),
                "overlap_ids": title_overlap_ids(fold.train_stories, fold.candidates),
            }
        )
        if spec.word_c > 0:
            if config.model.linear_blend_enabled:
                raise ValueError("word kernel requires linear_blend_enabled=false")
            # Use the serving DB's row order, not the fold's original order.
            # Reuse that exact DB for the scorer, so lexical/numeric rows align.
            with evaluator._fold_database(fold, config, source_db) as db:
                frozen_rows = db.get_feedback_for_training()
                rows = frozen_rows
                if config.model.deduplicate_training_feedback:
                    from pipeline.feedback import deduplicate_feedback

                    rows = deduplicate_feedback(*rows)
                lexical = fit_lexical_inputs(rows[0], rows[1], fold.candidates)
                reader = Database.get_feedback_for_training

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

                # Freeze the exact row sequence across lexical and numeric reads;
                # correctness does not rely on SQLite repeating an unordered SELECT.
                with (
                    patch.object(
                        Database, "get_feedback_for_training", ordered_feedback
                    ),
                    patch.object(
                        ranking,
                        "PrecomputedRbfSVC",
                        partial(
                            build,
                            lexical=lexical,
                            metadata_columns=metadata_columns,
                        ),
                    ),
                ):
                    return original_score(
                        replace(fold, runtime_db=db), config, source_db
                    )
        with patch.object(
            ranking,
            "PrecomputedRbfSVC",
            partial(build, metadata_columns=metadata_columns),
        ):
            return original_score(fold, config, source_db)

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

    # Keep the kernel and exact cohort identity with explicitly named reports.
    output_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    output_parser.add_argument("--output", type=Path)
    output_args, _ = output_parser.parse_known_args(list(argv))
    if output_args.output is not None:
        report = json.loads(output_args.output.read_text())
        cohort_ids = sorted(cohort) if cohort is not None else None
        report["single_model_experiment"] = {
            "kernel": asdict(spec),
            "cohort_ids": cohort_ids,
            "cohort_sha256": hashlib.sha256(json.dumps(cohort_ids).encode()).hexdigest()
            if cohort_ids is not None
            else None,
            "driver_sha256": driver_digest,
            "fit_audits": [asdict(m.audit) for m in fitted if m.audit is not None],
            "title_audits": title_audits,
            "vector_math_cache": True,
        }
        output_args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--family", choices=("rbf", "hybrid", "linear", "anchored"), default="rbf"
    )
    parser.add_argument("--linear-share", type=float, default=0.3)
    parser.add_argument("--score-form", choices=("ovr", "margin"), default="ovr")
    parser.add_argument("--word-c", type=float, default=0.0)
    parser.add_argument("--embedding-weight", type=float, default=1.0)
    parser.add_argument("--feedback-cohort", type=Path)
    parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    spec = KernelSpec(
        args.family,
        args.linear_share,
        args.score_form,
        args.word_c,
        args.embedding_weight,
    )
    cohort = read_cohort(args.feedback_cohort) if args.feedback_cohort else None
    evaluator_args = args.evaluation_args
    if evaluator_args[:1] == ["--"]:
        evaluator_args = evaluator_args[1:]
    if not evaluator_args:
        parser.error("canonical evaluator arguments are required after --")
    print(
        f"single-kernel experiment: {spec}; cohort={len(cohort) if cohort else 'all'}",
        flush=True,
    )
    run_evaluation(evaluator_args, spec, cohort)


if __name__ == "__main__":
    main()
