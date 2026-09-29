"""Opt-in blend of the production score with a dense logistic regression and
a TF-IDF logistic regression (`linear_blend_enabled`).

Offline (FINDINGS.md, 2026-09-29): 0.5 production + 0.2 dense LR + 0.3 TF-IDF,
each as an average-rank percentile, beat production on user 1's votes.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
from cachetools import LRUCache
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from database import Story

TFIDF_TEXT_CHARS = 5000
UP, DOWN = 2, 0


def percentile_scores(scores: NDArray[np.floating]) -> NDArray[np.float32]:
    """Ranks scaled to [0, 1]; equal scores share their average rank."""
    if len(scores) <= 1:
        return np.ones(len(scores), dtype=np.float32)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average = (ends - counts + ends - 1) / 2.0
    return (average[inverse] / (len(scores) - 1)).astype(np.float32)


def story_domain(story: Story) -> str:
    return (urlparse(story.url or "").hostname or "").removeprefix("www.")


def tfidf_text(story: Story) -> str:
    """Domain and source tokens, title, and the start of the text content."""
    domain = story_domain(story).replace(".", "_") or story.source
    return (
        f"dom_{domain} src_{story.source} {story.title} "
        f"{story.text_content[:TFIDF_TEXT_CHARS]}"
    )


def make_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        stop_words="english",
    )


def balanced_weights(labels: Sequence[int]) -> NDArray[np.float64]:
    counts = {label: labels.count(label) for label in set(labels)}
    return np.array([len(labels) / (len(counts) * counts[label]) for label in labels])


def up_minus_down(clf: LogisticRegression, features: object) -> NDArray[np.float32]:
    probs = clf.predict_proba(features)
    classes = list(clf.classes_)
    return (probs[:, classes.index(UP)] - probs[:, classes.index(DOWN)]).astype(
        np.float32
    )


@dataclass(frozen=True)
class LinearBlendModels:
    """Fitted per feedback signature; cached next to the SVM."""

    dense: LogisticRegression
    vectorizer: TfidfVectorizer
    tfidf: LogisticRegression


def fit_linear_blend(
    dense_features: NDArray[np.float32],
    dense_labels: Sequence[int],
    dense_weights: NDArray[np.float64],
    stories: Sequence[Story],
    story_labels: Sequence[int],
    *,
    dense_c: float,
    tfidf_c: float,
) -> LinearBlendModels:
    """``dense_*`` are the SVM's scaled training rows (with any zero rows for
    absent classes); ``stories``/``story_labels`` are the real votes only."""
    dense = LogisticRegression(C=dense_c, solver="lbfgs", max_iter=2000, random_state=0)
    dense.fit(dense_features, dense_labels, sample_weight=dense_weights)
    vectorizer = make_tfidf_vectorizer()
    x_train = vectorizer.fit_transform([tfidf_text(s) for s in stories])
    tfidf = LogisticRegression(C=tfidf_c, max_iter=3000, random_state=0)
    tfidf.fit(x_train, story_labels, sample_weight=balanced_weights(list(story_labels)))
    return LinearBlendModels(dense=dense, vectorizer=vectorizer, tfidf=tfidf)


def blend_scores(
    models: LinearBlendModels,
    production: NDArray[np.floating],
    dense_candidates: NDArray[np.float32],
    candidates: Sequence[Story],
    *,
    dense_weight: float,
    tfidf_weight: float,
) -> NDArray[np.float32]:
    """Percentile-rank blend; production keeps the remaining weight."""
    x_cand = models.vectorizer.transform([tfidf_text(s) for s in candidates])
    return (
        (1.0 - dense_weight - tfidf_weight) * percentile_scores(production)
        + dense_weight
        * percentile_scores(up_minus_down(models.dense, dense_candidates))
        + tfidf_weight * percentile_scores(up_minus_down(models.tfidf, x_cand))
    ).astype(np.float32)


_CACHE: LRUCache[tuple[int, str, int], LinearBlendModels] = LRUCache(maxsize=10_000)
_LOCK = threading.Lock()


def get_cached(key: tuple[int, str, int]) -> LinearBlendModels | None:
    with _LOCK:
        return _CACHE.get(key)


def set_cached(
    key: tuple[int, str, int], models: LinearBlendModels, maxsize: int
) -> None:
    with _LOCK:
        _CACHE[key] = models
        while len(_CACHE) > maxsize:
            _CACHE.popitem()
