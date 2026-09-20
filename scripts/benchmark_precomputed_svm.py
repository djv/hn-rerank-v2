#!/usr/bin/env python3
"""Compare production RBF SVC inference with an exact precomputed kernel.

The script runs the normal production feature pipeline against a read-only
SQLite connection, intercepting only SVC construction. Both models receive the
same scaled training and candidate matrices, labels, and sample weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import (  # noqa: E402
    Config,
    Embedder,
    RankTrace,
    fast_rerank_for_user,
    load_production_candidate_stories,
)
from scripts.benchmark_rank_cold_cache import (  # noqa: E402
    _heaviest_user_id,
    _missing_embedding_count,
)
import pipeline.ranking as ranking  # noqa: E402


RESULT: dict[str, int | float | bool] = {}


class BenchmarkSVC:
    """Drop-in SVC proxy that benchmarks an exact precomputed RBF model."""

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("kernel") != "rbf":
            raise ValueError("benchmark requires the production RBF kernel")
        self._gamma = float(kwargs["gamma"])
        self._regular = SVC(**kwargs)
        precomputed_kwargs = dict(kwargs)
        precomputed_kwargs["kernel"] = "precomputed"
        precomputed_kwargs.pop("gamma", None)
        self._precomputed = SVC(**precomputed_kwargs)
        self._train_x: NDArray[np.float64] | None = None

    @property
    def classes_(self) -> NDArray[np.int64]:
        return self._regular.classes_

    def fit(
        self,
        x: NDArray[np.float64],
        y: list[int],
        *,
        sample_weight: NDArray[np.float64],
    ) -> BenchmarkSVC:
        self._train_x = np.asarray(x)
        started = time.perf_counter()
        self._regular.fit(x, y, sample_weight=sample_weight)
        RESULT["regular_fit_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        train_kernel = rbf_kernel(x, x, gamma=self._gamma)
        RESULT["train_kernel_ms"] = (time.perf_counter() - started) * 1000
        RESULT["train_kernel_mib"] = train_kernel.nbytes / (1024 * 1024)

        started = time.perf_counter()
        self._precomputed.fit(train_kernel, y, sample_weight=sample_weight)
        RESULT["precomputed_fit_ms"] = (time.perf_counter() - started) * 1000
        return self

    def decision_function(
        self, x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._train_x is None:
            raise RuntimeError("fit must run before decision_function")

        started = time.perf_counter()
        regular = self._regular.decision_function(x)
        RESULT["regular_decision_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        candidate_kernel = rbf_kernel(x, self._train_x, gamma=self._gamma)
        RESULT["candidate_kernel_ms"] = (time.perf_counter() - started) * 1000
        RESULT["candidate_kernel_mib"] = candidate_kernel.nbytes / (1024 * 1024)

        started = time.perf_counter()
        precomputed = self._precomputed.decision_function(candidate_kernel)
        RESULT["precomputed_decision_ms"] = (time.perf_counter() - started) * 1000

        difference = np.abs(regular - precomputed)
        RESULT["max_abs_difference"] = float(difference.max(initial=0.0))
        RESULT["allclose_rtol_1e-7_atol_1e-9"] = bool(
            np.allclose(regular, precomputed, rtol=1e-7, atol=1e-9)
        )
        up_index = list(self.classes_).index(2)
        regular_up = regular if regular.ndim == 1 else regular[:, up_index]
        precomputed_up = (
            precomputed if precomputed.ndim == 1 else precomputed[:, up_index]
        )
        regular_top = np.argsort(-regular_up, kind="stable")[:40]
        precomputed_top = np.argsort(-precomputed_up, kind="stable")[:40]
        RESULT["top_40_exact"] = bool(np.array_equal(regular_top, precomputed_top))
        RESULT["candidates"] = len(x)
        RESULT["training_rows"] = len(self._train_x)
        return regular


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--db", default="hn_rewrite.db")
    parser.add_argument("--user-id", type=int)
    args = parser.parse_args()

    config = Config.load(args.config)
    db = Database(args.db, read_only=True)
    original_svc = ranking.SVC
    try:
        user_id = args.user_id or _heaviest_user_id(db)
        candidates = load_production_candidate_stories(
            db, config, user_id=user_id, exclude_feedback=True
        )
        feedback_stories, _labels, _vote_times = db.get_feedback_for_training(user_id)
        preflight = {
            "candidates": len(candidates),
            "feedback": len(feedback_stories),
            "candidate_missing_embeddings": _missing_embedding_count(
                db, candidates, config.embedding_model_version
            ),
            "feedback_missing_embeddings": _missing_embedding_count(
                db, feedback_stories, config.embedding_model_version
            ),
        }
        if preflight["candidate_missing_embeddings"] or preflight[
            "feedback_missing_embeddings"
        ]:
            raise SystemExit(f"Missing cached embeddings; refusing writes: {preflight}")

        embedder = Embedder(
            config.onnx_model_dir,
            model_version=config.embedding_model_version,
            max_tokens=config.embedding_max_tokens,
            batch_size=config.embedding_batch_size,
            ort_variant=config.embedding_ort_variant,
        )
        setattr(ranking, "SVC", BenchmarkSVC)
        trace = RankTrace()
        fast_rerank_for_user(db, config, embedder, user_id, trace=trace)
        output = {"user_id": user_id, "preflight": preflight, **RESULT}
        print(json.dumps(output, indent=2, sort_keys=True))
    finally:
        setattr(ranking, "SVC", original_svc)
        db.close()


if __name__ == "__main__":
    main()
