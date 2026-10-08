"""One logistic regression in place of the RBF SVM and the linear blend
(`ModelConfig.classifier = "joined_logistic"`).

Inputs are the SVM's scaled training rows, with the embedding block weighted
by ``sqrt(embedding_weight)`` and everything numeric multiplied by
``numeric_scale``, joined to the stories' hashed TF-IDF words (the word
model's vocabulary rule: terms in at least two training stories). Ranking
uses P(up) - P(down). Offline (docs/evaluations/model-ablation-20261007/
FEATURE-REMOVALS.md): "all" is shortlist candidate #2, "no_metadata" #4.
This matches the offline adapter in scripts/eval_one_classifier.py, which
tests/test_joined_classifier.py checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

from database import Story

from .linear_blend import count_rows

JoinedFeatures = Literal["all", "no_metadata"]


class JoinedLogistic:
    def __init__(
        self,
        *,
        c: float,
        features: JoinedFeatures,
        embedding_dim: int,
        embedding_weight: float,
        numeric_scale: float,
        word_scale: float,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.features = features
        self.embedding_dim = embedding_dim
        self.embedding_weight = embedding_weight
        self.numeric_scale = numeric_scale
        self.word_scale = word_scale
        self.estimator = LogisticRegression(C=c, max_iter=2000, random_state=0)
        self._keep: NDArray[np.bool_] | None = None
        self._idf: TfidfTransformer | None = None

    @property
    def classes_(self) -> NDArray[np.int64]:
        return np.asarray(self.estimator.classes_, dtype=np.int64)

    def _inputs(self, features: NDArray[np.floating], words: csr_matrix) -> csr_matrix:
        features = np.asarray(features, dtype=np.float64)
        if features.shape[1] <= self.embedding_dim:
            raise ValueError("feature rows lack metadata columns")
        if features.shape[0] != words.shape[0]:
            raise ValueError("numeric and word rows do not align")
        blocks = [features[:, : self.embedding_dim] * np.sqrt(self.embedding_weight)]
        if self.features == "all":
            blocks.append(features[:, self.embedding_dim :])
        numeric = np.hstack(blocks) * self.numeric_scale
        return hstack([csr_matrix(numeric), words * self.word_scale], format="csr")

    def _words(self, stories: Sequence[Story]) -> csr_matrix:
        assert self._keep is not None and self._idf is not None
        return self._idf.transform(count_rows(stories)[:, self._keep]).tocsr()

    def fit(
        self,
        features: NDArray[np.floating],
        labels: Sequence[int],
        *,
        sample_weight: NDArray[np.float64],
        stories: Sequence[Story],
    ) -> JoinedLogistic:
        """*stories* are the voted stories behind *features*, one per row;
        the SVM's zero placeholder rows for absent classes have none, so a
        vote set missing a class is rejected here (callers fall back)."""
        if len(stories) != len(labels):
            raise ValueError("joined classifier needs one story per training row")
        counts = count_rows(stories)
        keep = np.asarray((counts > 0).sum(axis=0)).ravel() >= 2
        if not keep.any():
            raise ValueError("no word appears in two training stories")
        idf = TfidfTransformer(sublinear_tf=True)
        words = idf.fit_transform(counts[:, keep]).tocsr()
        self._keep, self._idf = keep, idf
        self.estimator.fit(
            self._inputs(features, words), list(labels), sample_weight=sample_weight
        )
        return self

    def predict_proba(
        self, features: NDArray[np.floating], stories: Sequence[Story]
    ) -> NDArray[np.float64]:
        if self._keep is None:
            raise RuntimeError("fit must run before predict_proba")
        return np.asarray(
            self.estimator.predict_proba(self._inputs(features, self._words(stories))),
            dtype=np.float64,
        )
