#!/usr/bin/env python3
"""Leakage-safe 30-day evaluator for personalized ranker variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from typing import Any
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

# DL imports are deferred; only loaded if a DL variant is requested.
# This lets the harness run all sklearn / SVM / cluster variants without
# pulling in the ~700MB torch dependency.
if _TORCH_AVAILABLE:
    from pipeline_dl import fit_attention_mlp, predict_attention_mlp
    from pipeline_dl_t0 import (
        fit_attention_mlp_t0,
        predict_attention_mlp_t0,
    )

from database import Database, Story
from pipeline import (
    Config,
    Embedder,
    ModelConfig,
    RankedStory,
    _knn_similarity,
    _positive_cluster_centers,
    clean_text,
    mmr_filter,
)

MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|256"
SELF_FIELD_CHAR_LIMIT = 6000
ARTICLE_FIELD_CHAR_LIMIT = 4000
COMMENT_FIELD_CHAR_LIMIT = 6000


@dataclass(frozen=True)
class FoldData:
    candidates: list[Story]
    cand_emb: np.ndarray
    cand_field_emb: np.ndarray
    cand_field_parts: np.ndarray
    train_stories: list[Story]
    test_stories: list[Story]
    test_actions: np.ndarray
    train_vote_times: np.ndarray
    x_train_base: np.ndarray
    x_cand_base: np.ndarray
    x_train_field: np.ndarray
    x_cand_field: np.ndarray
    x_train_field_sims: np.ndarray
    x_cand_field_sims: np.ndarray
    x_train_textsplit: np.ndarray
    x_cand_textsplit: np.ndarray
    y_train: np.ndarray
    tier2_scores: np.ndarray


def _db_sha256(db_path: str) -> str:
    return hashlib.sha256(Path(db_path).read_bytes()).hexdigest()[:16]


def _load_recent_candidates(
    db: Database, cutoff_ts: int
) -> tuple[list[Story], np.ndarray]:
    rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, "
        "       comment_count, discussion_url, comment_count_at_fetch, "
        "       self_text, top_comments, article_body "
        "FROM stories WHERE time >= ? AND text_content != ''",
        (cutoff_ts,),
    )
    stories = [Database._row_to_story(row) for row in rows]
    hashes = {
        s.id: hashlib.sha256(s.text_content.encode("utf-8")).hexdigest()
        for s in stories
    }
    cached = db.get_embeddings_batch([s.id for s in stories], MODEL_VERSION, hashes)
    embeddings = np.array(
        [cached.get(s.id, np.zeros(384, dtype=np.float32)) for s in stories],
        dtype=np.float32,
    )
    return stories, embeddings


def _field_embeddings_by_field(
    stories: list[Story], embedder: Embedder, batch_size: int = 64
) -> np.ndarray:
    if not stories:
        return np.empty((0, 4, 384), dtype=np.float32)

    field_texts = [
        [
            clean_text(s.title or ""),
            clean_text(s.self_text or "")[:SELF_FIELD_CHAR_LIMIT],
            clean_text(s.article_body or "")[:ARTICLE_FIELD_CHAR_LIMIT],
            clean_text(s.top_comments or "")[:COMMENT_FIELD_CHAR_LIMIT],
        ]
        for s in stories
    ]
    parts = np.zeros((len(stories), 4, 384), dtype=np.float32)
    for field_idx in range(4):
        texts = [fields[field_idx] for fields in field_texts]
        non_empty = [i for i, text in enumerate(texts) if text.strip()]
        if not non_empty:
            continue
        non_empty_texts = [texts[i] for i in non_empty]
        embs = []
        for i in range(0, len(non_empty_texts), batch_size):
            embs.append(embedder.encode(non_empty_texts[i : i + batch_size]))
        parts[np.array(non_empty, dtype=int), field_idx, :] = np.concatenate(
            embs, axis=0
        ).astype(np.float32)
    return parts


def _average_field_embeddings(field_parts: np.ndarray) -> np.ndarray:
    if len(field_parts) == 0:
        return np.empty((0, 384), dtype=np.float32)
    present = (np.linalg.norm(field_parts, axis=2) > 0).astype(np.float32)
    total = field_parts.sum(axis=1)
    counts = present.sum(axis=1, keepdims=True)
    averaged = total / np.maximum(counts, 1.0)
    norms = np.linalg.norm(averaged, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (averaged / norms).astype(np.float32)


def _field_level_embeddings(
    stories: list[Story], embedder: Embedder, batch_size: int = 64
) -> np.ndarray:
    return _average_field_embeddings(
        _field_embeddings_by_field(stories, embedder, batch_size)
    )


def _field_similarity_feature_matrix(
    query_parts: np.ndarray,
    train_parts: np.ndarray,
    y_train: np.ndarray,
    *,
    k: int,
    loocv: bool,
) -> np.ndarray:
    if len(query_parts) == 0:
        return np.empty((0, 16), dtype=np.float32)

    features = np.zeros((len(query_parts), 16), dtype=np.float32)
    query_present = np.linalg.norm(query_parts, axis=2) > 0
    train_present = np.linalg.norm(train_parts, axis=2) > 0

    for field_idx in range(4):
        query_emb = query_parts[:, field_idx, :]
        train_emb = train_parts[:, field_idx, :]
        present_query = query_present[:, field_idx]

        sim_up = np.zeros(len(query_parts), dtype=np.float32)
        sim_down = np.zeros(len(query_parts), dtype=np.float32)
        closest_up = np.zeros(len(query_parts), dtype=np.float32)
        closest_down = np.zeros(len(query_parts), dtype=np.float32)

        for label, sim_out, closest_out in (
            (2, sim_up, closest_up),
            (0, sim_down, closest_down),
        ):
            ref_mask = (y_train == label) & train_present[:, field_idx]
            ref_emb = train_emb[ref_mask]
            if len(ref_emb) == 0:
                continue

            ref_indices = np.where(ref_mask)[0]
            mat = query_emb @ ref_emb.T
            closest_mat = mat.copy()
            if loocv and len(query_parts) == len(train_parts):
                for col, row in enumerate(ref_indices):
                    if row < len(query_parts):
                        mat[row, col] = -2.0
                        closest_mat[row, col] = -1.0

            for row in np.where(present_query)[0]:
                exclude = 1 if loocv and row in ref_indices else 0
                n_available = len(ref_emb) - exclude
                if n_available <= 0:
                    continue
                k_use = min(k, n_available)
                sim_out[row] = float(np.sort(mat[row])[-k_use:].mean())
                closest_out[row] = float(np.max(closest_mat[row]))

        col = field_idx * 4
        for offset, raw in enumerate((sim_up, sim_down, closest_up, closest_down)):
            values = _normalize_sims(raw)
            values[~present_query] = 0.0
            features[:, col + offset] = values
    return features.astype(np.float32)


def _normalize_log_lengths(lengths: np.ndarray) -> np.ndarray:
    return np.clip(np.log1p(np.maximum(lengths, 0)), 0, 12.0) / 12.0


def _normalize_sims(sims: np.ndarray) -> np.ndarray:
    return (np.clip(sims, -1.0, 1.0) + 1.0) / 2.0


def _feature_matrix(
    embeddings: np.ndarray,
    stories: list[Story],
    sim_up: np.ndarray,
    sim_down: np.ndarray,
    closest_up: np.ndarray,
    closest_down: np.ndarray,
    *,
    textsplit: bool,
) -> np.ndarray:
    if textsplit:
        text_meta = np.column_stack(
            [
                _normalize_log_lengths(np.array([len(s.title) for s in stories])),
                _normalize_log_lengths(np.array([len(s.self_text) for s in stories])),
                _normalize_log_lengths(
                    np.array([len(s.article_body) for s in stories])
                ),
                _normalize_log_lengths(
                    np.array([len(s.top_comments) for s in stories])
                ),
            ]
        )
    else:
        text_meta = _normalize_log_lengths(
            np.array([len(s.text_content) for s in stories])
        )[:, None]

    sim_meta = np.column_stack(
        [
            _normalize_sims(sim_up),
            _normalize_sims(sim_down),
            _normalize_sims(closest_up),
            _normalize_sims(closest_down),
        ]
    )
    return np.concatenate([embeddings, text_meta, sim_meta], axis=1).astype(np.float32)


def _fit_scale(
    train_x: np.ndarray, cand_x: np.ndarray, emb_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_meta = np.clip(scaler.fit_transform(train_x[:, emb_dim:]), -2.5, 2.5)
    cand_meta = np.clip(scaler.transform(cand_x[:, emb_dim:]), -2.5, 2.5)
    return (
        np.hstack([train_x[:, :emb_dim], train_meta]),
        np.hstack([cand_x[:, :emb_dim], cand_meta]),
    )


def _loocv_similarity_features(
    train_emb: np.ndarray, y_train: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up_mask = y_train == 2
    down_mask = y_train == 0
    up_emb = train_emb[up_mask]
    down_emb = train_emb[down_mask]
    sim_up = np.zeros(len(train_emb), dtype=np.float32)
    sim_down = np.zeros(len(train_emb), dtype=np.float32)

    if len(up_emb):
        up_indices = np.where(up_mask)[0]
        mat = train_emb @ up_emb.T
        if len(up_emb) > 1:
            for col, row in enumerate(up_indices):
                mat[row, col] = -2.0
        for row in range(len(train_emb)):
            exclude = 1 if row in up_indices else 0
            k_use = min(k, max(1, len(up_emb) - exclude))
            sim_up[row] = np.sort(mat[row])[-k_use:].mean()
        clean = train_emb @ up_emb.T
        if len(up_emb) > 1:
            for col, row in enumerate(up_indices):
                clean[row, col] = -1.0
        closest_up = np.max(clean, axis=1)
    else:
        closest_up = np.zeros(len(train_emb), dtype=np.float32)

    if len(down_emb):
        down_indices = np.where(down_mask)[0]
        mat = train_emb @ down_emb.T
        if len(down_emb) > 1:
            for col, row in enumerate(down_indices):
                mat[row, col] = -2.0
        for row in range(len(train_emb)):
            exclude = 1 if row in down_indices else 0
            k_use = min(k, max(1, len(down_emb) - exclude))
            sim_down[row] = np.sort(mat[row])[-k_use:].mean()
        clean = train_emb @ down_emb.T
        if len(down_emb) > 1:
            for col, row in enumerate(down_indices):
                clean[row, col] = -1.0
        closest_down = np.max(clean, axis=1)
    else:
        closest_down = np.zeros(len(train_emb), dtype=np.float32)

    return sim_up, sim_down, closest_up, closest_down


def _candidate_similarity_features(
    cand_emb: np.ndarray, train_emb: np.ndarray, y_train: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up_emb = train_emb[y_train == 2]
    down_emb = train_emb[y_train == 0]
    sim_up = _knn_similarity(cand_emb, up_emb, k)
    sim_down = _knn_similarity(cand_emb, down_emb, k)
    closest_up = (
        np.max(cand_emb @ up_emb.T, axis=1)
        if len(up_emb)
        else np.zeros(len(cand_emb), dtype=np.float32)
    )
    closest_down = (
        np.max(cand_emb @ down_emb.T, axis=1)
        if len(down_emb)
        else np.zeros(len(cand_emb), dtype=np.float32)
    )
    return sim_up, sim_down, closest_up, closest_down


def _tier2_scores(
    cand_emb: np.ndarray, train_emb: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    up_emb = train_emb[y_train == 2]
    down_emb = train_emb[y_train == 0]
    up_centroid = (
        up_emb.mean(axis=0) if len(up_emb) else np.zeros(384, dtype=np.float32)
    )
    down_centroid = (
        down_emb.mean(axis=0) if len(down_emb) else np.zeros(384, dtype=np.float32)
    )
    scores = cand_emb @ up_centroid - cand_emb @ down_centroid
    return ((scores - scores.min()) / (scores.max() - scores.min() + 1e-8)).astype(
        np.float32
    )


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(y)
    return np.array([len(y) / (len(counts) * counts[label]) for label in y])


def _recency_decay(
    now_ts: float, vote_times: np.ndarray, half_life_days: float
) -> np.ndarray:
    return np.exp(-np.log(2.0) * (now_ts - vote_times) / (half_life_days * 86400.0))


def _percentile_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float32)
    if len(scores) <= 1:
        return np.ones(len(scores), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(scores), dtype=np.float32)
    return ranks


def _minmax_scores(scores: np.ndarray) -> np.ndarray:
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    return ((scores - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _story_domain(story: Story) -> str:
    if not story.url:
        return ""
    host = urlparse(story.url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _story_source(story: Story) -> str:
    return (story.source or "").lower()


def _source_domain_feature_matrix(
    query_stories: list[Story],
    train_stories: list[Story],
    y_train: np.ndarray,
    *,
    loocv: bool,
) -> np.ndarray:
    source_counts: Counter[str] = Counter()
    source_up: Counter[str] = Counter()
    source_down: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    domain_up: Counter[str] = Counter()
    domain_down: Counter[str] = Counter()

    for story, label in zip(train_stories, y_train, strict=True):
        source = _story_source(story)
        domain = _story_domain(story)
        source_counts[source] += 1
        domain_counts[domain] += 1
        if label == 2:
            source_up[source] += 1
            domain_up[domain] += 1
        elif label == 0:
            source_down[source] += 1
            domain_down[domain] += 1

    features = np.zeros((len(query_stories), 6), dtype=np.float32)
    for row, story in enumerate(query_stories):
        source = _story_source(story)
        domain = _story_domain(story)
        source_total = source_counts[source]
        source_pos = source_up[source]
        source_neg = source_down[source]
        domain_total = domain_counts[domain]
        domain_pos = domain_up[domain]
        domain_neg = domain_down[domain]

        if loocv and row < len(train_stories):
            label = int(y_train[row])
            if _story_source(train_stories[row]) == source:
                source_total -= 1
                if label == 2:
                    source_pos -= 1
                elif label == 0:
                    source_neg -= 1
            if _story_domain(train_stories[row]) == domain:
                domain_total -= 1
                if label == 2:
                    domain_pos -= 1
                elif label == 0:
                    domain_neg -= 1

        features[row] = [
            (source_pos + 0.5) / (source_total + 1.5),
            (source_neg + 0.5) / (source_total + 1.5),
            min(math.log1p(max(source_total, 0)) / 6.0, 1.0),
            (domain_pos + 0.5) / (domain_total + 1.5),
            (domain_neg + 0.5) / (domain_total + 1.5),
            min(math.log1p(max(domain_total, 0)) / 6.0, 1.0),
        ]
    return features


def _select_matrices(
    fold: FoldData, *, feature_source: str, textsplit: bool
) -> tuple[np.ndarray, np.ndarray]:
    if feature_source == "field":
        return fold.x_train_field, fold.x_cand_field
    if textsplit:
        return fold.x_train_textsplit, fold.x_cand_textsplit
    return fold.x_train_base, fold.x_cand_base


def _scores_prob_3class(
    fold: FoldData, config: Config, *, textsplit: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    train_x, cand_x = _select_matrices(fold, feature_source="base", textsplit=textsplit)
    x_train, x_cand = _fit_scale(train_x, cand_x, fold.cand_emb.shape[1])
    missing = sorted({0, 1, 2} - set(fold.y_train))
    y = fold.y_train
    weights = _balanced_weights(y)
    if missing:
        x_train = np.vstack([x_train, np.zeros((len(missing), x_train.shape[1]))])
        y = np.concatenate([y, np.array(missing)])
        weights = np.concatenate([weights, np.full(len(missing), 1e-6)])
    svm = SVC(
        C=config.model.svm_c,
        kernel=config.model.svm_kernel,
        gamma=config.model.svm_gamma,
        random_state=0,
        decision_function_shape="ovr",
        probability=True,
    )
    svm.fit(x_train, y, sample_weight=weights)
    probs = svm.predict_proba(x_cand)
    classes = list(svm.classes_)
    scores = (
        probs[:, classes.index(2)]
        + config.model.neutral_weight * probs[:, classes.index(1)]
    )
    return scores.astype(np.float32), probs


def _scores_margin_3class(
    fold: FoldData,
    config: Config,
    *,
    textsplit: bool,
    mode: str,
    half_life_days: float | None = None,
    hard_negative_multiplier: float | None = None,
    label_weight_multipliers: dict[int, float] | None = None,
) -> tuple[np.ndarray, None]:
    train_x, cand_x = _select_matrices(fold, feature_source="base", textsplit=textsplit)
    x_train, x_cand = _fit_scale(train_x, cand_x, fold.cand_emb.shape[1])
    y = fold.y_train
    weights = _balanced_weights(y)
    if label_weight_multipliers is not None:
        weights = weights * np.array(
            [label_weight_multipliers.get(int(label), 1.0) for label in y],
            dtype=np.float64,
        )
    if half_life_days is not None:
        weights = weights * _recency_decay(
            time.time(), fold.train_vote_times, half_life_days
        )
    if hard_negative_multiplier is not None:
        down_mask = y == 0
        up_emb = fold.cand_emb[:0]
        train_emb = train_x[:, : fold.cand_emb.shape[1]]
        if np.any(y == 2):
            up_emb = train_emb[y == 2]
        if np.any(down_mask) and len(up_emb):
            hard = np.max(train_emb[down_mask] @ up_emb.T, axis=1)
            hard = _normalize_sims(hard)
            weights[down_mask] *= 1.0 + hard_negative_multiplier * hard
    svm = SVC(
        C=config.model.svm_c,
        kernel=config.model.svm_kernel,
        gamma=config.model.svm_gamma,
        random_state=0,
        decision_function_shape="ovr",
    )
    svm.fit(x_train, y, sample_weight=weights)
    decision = svm.decision_function(x_cand)
    classes = list(svm.classes_)
    if decision.ndim == 1:
        up_sign = 1.0 if classes[-1] == 2 else -1.0
        scores = up_sign * decision
    elif mode == "up":
        scores = decision[:, classes.index(2)]
    elif mode == "up_neutral":
        scores = (
            decision[:, classes.index(2)]
            + config.model.neutral_weight * decision[:, classes.index(1)]
        )
    elif mode == "up_minus_down":
        scores = decision[:, classes.index(2)] - decision[:, classes.index(0)]
    else:
        raise ValueError(f"Unknown margin mode: {mode}")
    return scores.astype(np.float32), None


def _prepare_linear_model_inputs(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x, cand_x = _select_matrices(fold, feature_source="base", textsplit=False)
    x_train, x_cand = _fit_scale(train_x, cand_x, fold.cand_emb.shape[1])
    y = fold.y_train
    weights = _balanced_weights(y)
    return x_train, x_cand, y, weights


def _scores_linear_svc_up(fold: FoldData, config: Config) -> tuple[np.ndarray, None]:
    x_train, x_cand, y, weights = _prepare_linear_model_inputs(fold, config)
    clf = LinearSVC(
        C=config.model.svm_c,
        dual="auto",
        random_state=0,
        max_iter=5000,
    )
    clf.fit(x_train, y, sample_weight=weights)
    decision = clf.decision_function(x_cand)
    classes = list(clf.classes_)
    if decision.ndim == 1:
        up_sign = 1.0 if classes[-1] == 2 else -1.0
        scores = up_sign * decision
    else:
        scores = decision[:, classes.index(2)]
    return scores.astype(np.float32), None


def _scores_logreg_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    x_train, x_cand, y, weights = _prepare_linear_model_inputs(fold, config)
    clf = LogisticRegression(
        C=config.model.svm_c,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    clf.fit(x_train, y, sample_weight=weights)
    probs = clf.predict_proba(x_cand)
    classes = list(clf.classes_)
    scores = probs[:, classes.index(2)]
    return scores.astype(np.float32), probs


def _scores_sgd_log_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    x_train, x_cand, y, weights = _prepare_linear_model_inputs(fold, config)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        max_iter=2000,
        tol=1e-3,
        random_state=0,
    )
    clf.fit(x_train, y, sample_weight=weights)
    probs = clf.predict_proba(x_cand)
    classes = list(clf.classes_)  # type: ignore  # sklearn SGDClassifier.classes_ not recognized by ty after fit
    scores = probs[:, classes.index(2)]
    return scores.astype(np.float32), probs


def _scores_mlp_up(
    fold: FoldData,
    config: Config,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_train, x_cand, y, weights = _prepare_linear_model_inputs(fold, config)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=min(64, len(y)),
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=0,
    )
    clf.fit(x_train, y, sample_weight=weights)
    probs = clf.predict_proba(x_cand)
    classes = list(clf.classes_)
    scores = probs[:, classes.index(2)]
    return scores.astype(np.float32), probs


def _scores_attention_mlp_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T1: multi-head, wide MLP, cosine sims (full Tier 1)."""
    return _scores_attention_mlp_v1(fold, config, extra_dim=5, hidden_dim=256)


def _scores_attention_mlp_v1_nocos_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T1 without cosine-sim features."""
    return _scores_attention_mlp_v1(fold, config, extra_dim=0, hidden_dim=256)


def _scores_attention_mlp_v1_h64_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T1 with hidden_dim=64 (same as T0)."""
    return _scores_attention_mlp_v1(fold, config, extra_dim=5, hidden_dim=64)


def _scores_attention_mlp_v1(
    fold: FoldData, config: Config, *, extra_dim: int, hidden_dim: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Configurable T1 scorer.

    Args:
        extra_dim: number of cosine-sim features (0 to disable)
        hidden_dim: MLP hidden layer width
    """
    return _scores_attention_mlp_v2(
        fold,
        config,
        extra_dim=extra_dim,
        hidden_dim=hidden_dim,
        use_meta_per_class=False,
        use_ranking=False,
        use_mixup=False,
    )


# Tier 2 wrappers
def _scores_attention_mlp_v2_rank_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T1 + ranking loss only."""
    return _scores_attention_mlp_v2(
        fold,
        config,
        extra_dim=0,
        hidden_dim=256,
        use_meta_per_class=False,
        use_ranking=True,
        use_mixup=False,
    )


def _scores_attention_mlp_v2_meta_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T1 + per-class meta only."""
    return _scores_attention_mlp_v2(
        fold,
        config,
        extra_dim=0,
        hidden_dim=256,
        use_meta_per_class=True,
        use_ranking=False,
        use_mixup=False,
    )


def _scores_attention_mlp_v2_mixup_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T1 + mixup only."""
    return _scores_attention_mlp_v2(
        fold,
        config,
        extra_dim=0,
        hidden_dim=256,
        use_meta_per_class=False,
        use_ranking=False,
        use_mixup=True,
    )


def _scores_attention_mlp_v2_all_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T1 + all Tier 2 features (ranking + meta + mixup)."""
    return _scores_attention_mlp_v2(
        fold,
        config,
        extra_dim=0,
        hidden_dim=256,
        use_meta_per_class=True,
        use_ranking=True,
        use_mixup=True,
    )


def _scores_attention_mlp_v2(
    fold: FoldData,
    config: Config,
    *,
    extra_dim: int,
    hidden_dim: int,
    use_meta_per_class: bool,
    use_ranking: bool,
    use_mixup: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Configurable T2 scorer.

    Args:
        extra_dim: number of cosine-sim features (0 to disable)
        hidden_dim: MLP hidden layer width
        use_meta_per_class: concatenate per-class mean meta
        use_ranking: enable pairwise ranking loss
        use_mixup: enable mixup augmentation
    """
    emb_dim = fold.cand_emb.shape[1]

    train_x, cand_x = _select_matrices(fold, feature_source="base", textsplit=False)

    try:
        scaler = StandardScaler()
        train_meta = np.clip(scaler.fit_transform(train_x[:, emb_dim:]), -2.5, 2.5)
        cand_meta = np.clip(scaler.transform(cand_x[:, emb_dim:]), -2.5, 2.5)
    except Exception:
        train_meta = train_x[:, emb_dim:].copy()
        cand_meta = cand_x[:, emb_dim:].copy()

    train_emb = train_x[:, :emb_dim].astype(np.float32)
    cand_emb = cand_x[:, :emb_dim].astype(np.float32)
    train_labels = fold.y_train

    train_extra = None
    cand_extra = None
    if extra_dim > 0:
        k = config.model.knn_k
        n_clusters = config.model.positive_cluster_k
        up_emb = train_emb[train_labels == 2]
        down_emb = train_emb[train_labels == 0]

        def _cosine_sims(query: np.ndarray) -> np.ndarray:
            N = len(query)
            out = np.zeros((N, 5), dtype=np.float32)
            if len(up_emb):
                out[:, 0] = _knn_similarity(query, up_emb, k=min(k, len(up_emb)))
                out[:, 1] = np.max(query @ up_emb.T, axis=1)
            if len(down_emb):
                out[:, 2] = _knn_similarity(query, down_emb, k=min(k, len(down_emb)))
                out[:, 3] = np.max(query @ down_emb.T, axis=1)
            if len(up_emb) >= n_clusters:
                centers = _positive_cluster_centers(up_emb, n_clusters)
                out[:, 4] = np.max(query @ centers.T, axis=1)
            return np.clip(out, -1.0, 1.0)

        train_extra = _cosine_sims(train_emb).astype(np.float32)
        cand_extra = _cosine_sims(cand_emb).astype(np.float32)

    train_meta_per_class = None
    cand_meta_per_class = None
    if use_meta_per_class:
        up_meta = train_meta[train_labels == 2].mean(axis=0, keepdims=True)
        down_meta = train_meta[train_labels == 0].mean(axis=0, keepdims=True)
        meta_pc = np.concatenate([up_meta, down_meta], axis=1).astype(np.float32)
        train_meta_per_class = np.tile(meta_pc, (len(train_emb), 1))
        cand_meta_per_class = np.tile(meta_pc, (len(cand_emb), 1))

    kw: dict = dict(
        train_extra=train_extra,
        hidden_dim=hidden_dim,
        n_epochs=100,
        lr=1e-3,
        patience=15,
        val_frac=0.2,
        seed=0,
    )
    if use_ranking:
        kw["ranking_lambda"] = 0.5
        kw["ranking_margin"] = 0.5
        kw["ranking_pairs"] = 256
    if use_mixup:
        kw["mixup_alpha"] = 0.4

    model = fit_attention_mlp(
        train_emb,
        train_labels,
        train_meta.astype(np.float32),
        train_meta_per_class=train_meta_per_class,
        **kw,
    )

    if model is None:
        return np.full(len(cand_emb), 0.5, dtype=np.float32), None

    scores, probs = predict_attention_mlp(
        model,
        cand_emb,
        cand_meta.astype(np.float32),
        train_emb,
        train_labels,
        cand_extra=cand_extra,
        cand_meta_per_class=cand_meta_per_class,
        batch_size=128,
    )
    return scores.astype(np.float32), probs.astype(np.float32)


def _rank_ascending(scores: np.ndarray) -> np.ndarray:
    """Rank items ascending (1=worst, N=best)."""
    from scipy.stats import rankdata

    return rankdata(scores, method="average").astype(np.float32)


def _scores_blend_up(
    fold: FoldData, config: Config, alpha: float, *, kind: str
) -> tuple[np.ndarray, None]:
    """Blend SVM + MLP scores.

    Args:
        alpha: weight for SVM score (0 = pure MLP, 1 = pure SVM)
        kind: 'score' for raw-score blend, 'rank' for rank blend
    """
    svm_scores, _ = _scores_margin_3class(fold, config, textsplit=False, mode="up")
    mlp_scores, _ = _scores_attention_mlp_v2_meta_up(fold, config)
    svm_scores = svm_scores.astype(np.float32)
    mlp_scores = mlp_scores.astype(np.float32)

    if kind == "rank":
        svm_ranks = _rank_ascending(svm_scores)
        mlp_ranks = _rank_ascending(mlp_scores)
        blended = alpha * svm_ranks + (1 - alpha) * mlp_ranks
    else:  # score blend (default)
        blended = alpha * svm_scores + (1 - alpha) * mlp_scores

    return blended.astype(np.float32), None


def _scores_attention_mlp_t0_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T0 baseline: single-head, narrow MLP, no cosine sims."""
    emb_dim = fold.cand_emb.shape[1]

    train_x, cand_x = _select_matrices(fold, feature_source="base", textsplit=False)

    try:
        scaler = StandardScaler()
        train_meta = np.clip(scaler.fit_transform(train_x[:, emb_dim:]), -2.5, 2.5)
        cand_meta = np.clip(scaler.transform(cand_x[:, emb_dim:]), -2.5, 2.5)
    except Exception:
        train_meta = train_x[:, emb_dim:].copy()
        cand_meta = cand_x[:, emb_dim:].copy()

    train_emb = train_x[:, :emb_dim].astype(np.float32)
    cand_emb = cand_x[:, :emb_dim].astype(np.float32)
    train_labels = fold.y_train

    model = fit_attention_mlp_t0(
        train_emb,
        train_labels,
        train_meta.astype(np.float32),
        n_epochs=100,
        lr=1e-3,
        patience=15,
        val_frac=0.2,
        seed=0,
    )

    if model is None:
        return np.full(len(cand_emb), 0.5, dtype=np.float32), None

    scores, probs = predict_attention_mlp_t0(
        model,
        cand_emb,
        cand_meta.astype(np.float32),
        train_emb,
        train_labels,
        batch_size=128,
    )
    return scores.astype(np.float32), probs.astype(np.float32)


def _scores_attention_mlp_t0_cos_up(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray | None]:
    """T0 baseline + cosine sim features."""
    emb_dim = fold.cand_emb.shape[1]

    train_x, cand_x = _select_matrices(fold, feature_source="base", textsplit=False)

    try:
        scaler = StandardScaler()
        train_meta = np.clip(scaler.fit_transform(train_x[:, emb_dim:]), -2.5, 2.5)
        cand_meta = np.clip(scaler.transform(cand_x[:, emb_dim:]), -2.5, 2.5)
    except Exception:
        train_meta = train_x[:, emb_dim:].copy()
        cand_meta = cand_x[:, emb_dim:].copy()

    train_emb = train_x[:, :emb_dim].astype(np.float32)
    cand_emb = cand_x[:, :emb_dim].astype(np.float32)
    train_labels = fold.y_train

    k = config.model.knn_k
    n_clusters = config.model.positive_cluster_k
    up_emb = train_emb[train_labels == 2]
    down_emb = train_emb[train_labels == 0]

    def _cosine_sims(query: np.ndarray) -> np.ndarray:
        N = len(query)
        out = np.zeros((N, 5), dtype=np.float32)
        if len(up_emb):
            out[:, 0] = _knn_similarity(query, up_emb, k=min(k, len(up_emb)))
            out[:, 1] = np.max(query @ up_emb.T, axis=1)
        if len(down_emb):
            out[:, 2] = _knn_similarity(query, down_emb, k=min(k, len(down_emb)))
            out[:, 3] = np.max(query @ down_emb.T, axis=1)
        if len(up_emb) >= n_clusters:
            centers = _positive_cluster_centers(up_emb, n_clusters)
            out[:, 4] = np.max(query @ centers.T, axis=1)
        return np.clip(out, -1.0, 1.0)

    train_extra = _cosine_sims(train_emb).astype(np.float32)
    cand_extra = _cosine_sims(cand_emb).astype(np.float32)

    model = fit_attention_mlp_t0(
        train_emb,
        train_labels,
        train_meta.astype(np.float32),
        train_extra=train_extra,
        n_epochs=100,
        lr=1e-3,
        patience=15,
        val_frac=0.2,
        seed=0,
    )

    if model is None:
        return np.full(len(cand_emb), 0.5, dtype=np.float32), None

    scores, probs = predict_attention_mlp_t0(
        model,
        cand_emb,
        cand_meta.astype(np.float32),
        train_emb,
        train_labels,
        cand_extra=cand_extra,
        batch_size=128,
    )
    return scores.astype(np.float32), probs.astype(np.float32)


def _scores_margin_source_domain(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, None]:
    train_extra = _source_domain_feature_matrix(
        fold.train_stories, fold.train_stories, fold.y_train, loocv=True
    )
    cand_extra = _source_domain_feature_matrix(
        fold.candidates, fold.train_stories, fold.y_train, loocv=False
    )
    train_x = np.hstack([fold.x_train_base, train_extra]).astype(np.float32)
    cand_x = np.hstack([fold.x_cand_base, cand_extra]).astype(np.float32)
    x_train, x_cand = _fit_scale(train_x, cand_x, fold.cand_emb.shape[1])
    weights = _balanced_weights(fold.y_train)
    svm = SVC(
        C=config.model.svm_c,
        kernel=config.model.svm_kernel,
        gamma=config.model.svm_gamma,
        random_state=0,
        decision_function_shape="ovr",
    )
    svm.fit(x_train, fold.y_train, sample_weight=weights)
    decision = svm.decision_function(x_cand)
    classes = list(svm.classes_)
    if decision.ndim == 1:
        up_sign = 1.0 if classes[-1] == 2 else -1.0
        scores = up_sign * decision
    else:
        scores = decision[:, classes.index(2)]
    return scores.astype(np.float32), None


def _scores_margin_rank_calibrated(
    fold: FoldData, config: Config, *, tier2_weight: float
) -> tuple[np.ndarray, None]:
    margin, _ = _scores_margin_3class(fold, config, textsplit=False, mode="up")
    calibrated = _percentile_scores(margin)
    tier2 = _percentile_scores(fold.tier2_scores)
    return ((1.0 - tier2_weight) * calibrated + tier2_weight * tier2).astype(
        np.float32
    ), None


def _scores_positive_clusters(
    fold: FoldData, *, n_clusters: int
) -> tuple[np.ndarray, None]:
    up_emb = fold.x_train_base[fold.y_train == 2, : fold.cand_emb.shape[1]]
    down_emb = fold.x_train_base[fold.y_train == 0, : fold.cand_emb.shape[1]]
    if len(up_emb) == 0:
        return fold.tier2_scores.copy(), None
    if len(up_emb) <= n_clusters:
        centers = up_emb
    else:
        # Deterministic farthest-first centers avoids adding another tuning surface.
        centers = [up_emb[0]]
        sims_to_centers = up_emb @ centers[0]
        while len(centers) < n_clusters:
            next_idx = int(np.argmin(sims_to_centers))
            centers.append(up_emb[next_idx])
            sims_to_centers = np.maximum(sims_to_centers, up_emb @ centers[-1])
        centers = np.vstack(centers)
    pos_score = np.max(fold.cand_emb @ centers.T, axis=1)
    if len(down_emb):
        down_score = np.max(fold.cand_emb @ down_emb.T, axis=1)
    else:
        down_score = np.zeros(len(fold.cand_emb), dtype=np.float32)
    return _minmax_scores(pos_score - down_score), None


def _farthest_first_centroids(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    if len(embeddings) <= n_clusters:
        return embeddings.copy()
    centers = [embeddings[0]]
    sims_to_centers = embeddings @ centers[0]
    while len(centers) < n_clusters:
        next_idx = int(np.argmin(sims_to_centers))
        centers.append(embeddings[next_idx])
        sims_to_centers = np.maximum(sims_to_centers, embeddings @ centers[-1])
    return np.vstack(centers).astype(np.float32)


def _kmeans_centroids(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    if len(embeddings) <= n_clusters:
        return embeddings.copy()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    km.fit(embeddings)
    centers = km.cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return centers / np.clip(norms, a_min=1e-12, a_max=None)


def _positive_cluster_centroids(
    embeddings: np.ndarray, *, n_clusters: int, method: str
) -> np.ndarray:
    if method == "ff":
        return _farthest_first_centroids(embeddings, n_clusters)
    if method == "kmeans":
        return _kmeans_centroids(embeddings, n_clusters)
    raise ValueError(f"Unknown cluster method: {method}")


def _closest_positive_cluster_feature(
    query_emb: np.ndarray,
    up_emb: np.ndarray,
    *,
    n_clusters: int,
    method: str,
) -> np.ndarray:
    if len(up_emb) == 0:
        return np.zeros(len(query_emb), dtype=np.float32)
    centers = _positive_cluster_centroids(
        up_emb, n_clusters=min(n_clusters, len(up_emb)), method=method
    )
    return _normalize_sims(np.max(query_emb @ centers.T, axis=1)).astype(np.float32)


def _cluster_similarity_features(
    query_emb: np.ndarray,
    train_emb: np.ndarray,
    y_train: np.ndarray,
    *,
    n_clusters: int,
    method: str,
) -> dict[str, np.ndarray]:
    up_emb = train_emb[y_train == 2]
    down_emb = train_emb[y_train == 0]
    features: dict[str, np.ndarray] = {
        "pos_cluster": np.zeros(len(query_emb), dtype=np.float32),
        "neg_cluster": np.zeros(len(query_emb), dtype=np.float32),
        "cluster_margin": np.zeros(len(query_emb), dtype=np.float32),
        "pos_cluster_entropy": np.zeros(len(query_emb), dtype=np.float32),
        "centroid_residual": np.zeros(len(query_emb), dtype=np.float32),
        "down_up_interaction": np.zeros(len(query_emb), dtype=np.float32),
    }

    if len(up_emb):
        pos_centers = _positive_cluster_centroids(
            up_emb, n_clusters=min(n_clusters, len(up_emb)), method=method
        )
        pos_sims = query_emb @ pos_centers.T
        pos_cluster = _normalize_sims(np.max(pos_sims, axis=1))
        features["pos_cluster"] = pos_cluster.astype(np.float32)
        logits = pos_sims * 10.0
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)
        entropy = -np.sum(probs * np.log2(np.clip(probs, 1e-12, None)), axis=1)
        denom = math.log2(pos_sims.shape[1]) if pos_sims.shape[1] > 1 else 1.0
        features["pos_cluster_entropy"] = (entropy / denom).astype(np.float32)
        up_centroid = up_emb.mean(axis=0)
        up_centroid = up_centroid / max(float(np.linalg.norm(up_centroid)), 1e-12)
        centroid_sim = _normalize_sims(query_emb @ up_centroid)
        features["centroid_residual"] = (pos_cluster - centroid_sim).astype(np.float32)

    if len(down_emb):
        neg_centers = _positive_cluster_centroids(
            down_emb, n_clusters=min(n_clusters, len(down_emb)), method=method
        )
        neg_cluster = _normalize_sims(np.max(query_emb @ neg_centers.T, axis=1))
        features["neg_cluster"] = neg_cluster.astype(np.float32)

    features["cluster_margin"] = (
        features["pos_cluster"] - features["neg_cluster"]
    ).astype(np.float32)
    features["down_up_interaction"] = (
        features["pos_cluster"] * features["neg_cluster"]
    ).astype(np.float32)
    return features


def _scores_margin_with_cluster_features(
    fold: FoldData,
    config: Config,
    *,
    n_clusters: int,
    method: str,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, None]:
    train_emb = fold.x_train_base[:, : fold.cand_emb.shape[1]]
    train_features = _cluster_similarity_features(
        train_emb,
        train_emb,
        fold.y_train,
        n_clusters=n_clusters,
        method=method,
    )
    cand_features = _cluster_similarity_features(
        fold.cand_emb,
        train_emb,
        fold.y_train,
        n_clusters=n_clusters,
        method=method,
    )
    train_extra = np.column_stack([train_features[name] for name in feature_names])
    cand_extra = np.column_stack([cand_features[name] for name in feature_names])
    train_x = np.hstack([fold.x_train_base, train_extra]).astype(np.float32)
    cand_x = np.hstack([fold.x_cand_base, cand_extra]).astype(np.float32)
    x_train, x_cand = _fit_scale(train_x, cand_x, fold.cand_emb.shape[1])
    weights = _balanced_weights(fold.y_train)
    svm = SVC(
        C=config.model.svm_c,
        kernel=config.model.svm_kernel,
        gamma=config.model.svm_gamma,
        random_state=0,
        decision_function_shape="ovr",
    )
    svm.fit(x_train, fold.y_train, sample_weight=weights)
    decision = svm.decision_function(x_cand)
    classes = list(svm.classes_)
    if decision.ndim == 1:
        up_sign = 1.0 if classes[-1] == 2 else -1.0
        scores = up_sign * decision
    else:
        scores = decision[:, classes.index(2)]
    return scores.astype(np.float32), None


def _scores_margin_3class_field(
    fold: FoldData, config: Config, *, half_life_days: float | None = None
) -> tuple[np.ndarray, None]:
    x_train, x_cand = _select_matrices(fold, feature_source="field", textsplit=False)
    x_train, x_cand = _fit_scale(x_train, x_cand, fold.cand_field_emb.shape[1])
    y = fold.y_train
    weights = _balanced_weights(y)
    if half_life_days is not None:
        weights = weights * _recency_decay(
            time.time(), fold.train_vote_times, half_life_days
        )
    svm = SVC(
        C=config.model.svm_c,
        kernel=config.model.svm_kernel,
        gamma=config.model.svm_gamma,
        random_state=0,
        decision_function_shape="ovr",
    )
    svm.fit(x_train, y, sample_weight=weights)
    decision = svm.decision_function(x_cand)
    classes = list(svm.classes_)
    if decision.ndim == 1:
        up_sign = 1.0 if classes[-1] == 2 else -1.0
        scores = up_sign * decision
    else:
        scores = decision[:, classes.index(2)]
    return scores.astype(np.float32), None


def _scores_margin_3class_field_sims(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, None]:
    x_train, x_cand = _fit_scale(
        fold.x_train_field_sims, fold.x_cand_field_sims, fold.cand_emb.shape[1]
    )
    weights = _balanced_weights(fold.y_train)
    svm = SVC(
        C=config.model.svm_c,
        kernel=config.model.svm_kernel,
        gamma=config.model.svm_gamma,
        random_state=0,
        decision_function_shape="ovr",
    )
    svm.fit(x_train, fold.y_train, sample_weight=weights)
    decision = svm.decision_function(x_cand)
    classes = list(svm.classes_)
    if decision.ndim == 1:
        up_sign = 1.0 if classes[-1] == 2 else -1.0
        scores = up_sign * decision
    else:
        scores = decision[:, classes.index(2)]
    return scores.astype(np.float32), None


def _scores_binary_margin(
    fold: FoldData,
    config: Config,
    *,
    textsplit: bool,
    probability: bool,
    half_life_days: float | None = None,
) -> tuple[np.ndarray, None]:
    keep = fold.y_train != 1
    train_x, cand_x = _select_matrices(fold, feature_source="base", textsplit=textsplit)
    x_train, x_cand = _fit_scale(train_x[keep], cand_x, fold.cand_emb.shape[1])
    y = fold.y_train[keep]
    weights = _balanced_weights(y)
    if half_life_days is not None:
        weights = weights * _recency_decay(
            time.time(), fold.train_vote_times[keep], half_life_days
        )
    svm_kwargs = {
        "C": config.model.svm_c,
        "kernel": config.model.svm_kernel,
        "gamma": config.model.svm_gamma,
        "random_state": 0,
    }
    if probability:
        svm_kwargs["probability"] = True
    svm = SVC(**svm_kwargs)
    svm.fit(x_train, y, sample_weight=weights)
    classes = list(svm.classes_)
    if probability:
        probs = svm.predict_proba(x_cand)
        return probs[:, classes.index(2)].astype(np.float32), None
    decision = svm.decision_function(x_cand)
    up_sign = 1.0 if classes[-1] == 2 else -1.0
    return (up_sign * decision).astype(np.float32), None


def _scores_pairwise(
    fold: FoldData, *, textsplit: bool, strict_up_only: bool = False
) -> tuple[np.ndarray, None]:
    rel = np.array([{0: 0.0, 1: 0.2, 2: 1.0}[int(y)] for y in fold.y_train])
    train_raw, cand_raw = _select_matrices(
        fold, feature_source="base", textsplit=textsplit
    )
    train_x, cand_x = _fit_scale(train_raw, cand_raw, fold.cand_emb.shape[1])
    rng = np.random.default_rng(0)
    pairs: list[np.ndarray] = []
    labels: list[int] = []
    max_pairs_per_order = 60000
    pair_orders = (
        ((1.0, 0.2), (1.0, 0.0))
        if strict_up_only
        else ((1.0, 0.2), (1.0, 0.0), (0.2, 0.0))
    )
    for hi_rel, lo_rel in pair_orders:
        hi = np.where(rel == hi_rel)[0]
        lo = np.where(rel == lo_rel)[0]
        if not len(hi) or not len(lo):
            continue
        pair_idx = np.array(np.meshgrid(hi, lo)).T.reshape(-1, 2)
        if len(pair_idx) > max_pairs_per_order:
            pair_idx = pair_idx[
                rng.choice(len(pair_idx), max_pairs_per_order, replace=False)
            ]
        diffs = train_x[pair_idx[:, 0]] - train_x[pair_idx[:, 1]]
        pairs.append(diffs)
        labels.extend([1] * len(diffs))
        pairs.append(-diffs)
        labels.extend([0] * len(diffs))

    if not pairs:
        return fold.tier2_scores.copy(), None

    x_pairs = np.vstack(pairs)
    y_pairs = np.array(labels)
    order = rng.permutation(len(y_pairs))
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        penalty="l2",
        max_iter=2000,
        tol=1e-4,
        random_state=0,
        class_weight="balanced",
    )
    clf.fit(x_pairs[order], y_pairs[order])
    return clf.decision_function(cand_x).astype(np.float32), None


def _metrics(
    scores: np.ndarray,
    fold: FoldData,
    config: Config,
    probs: np.ndarray | None = None,
) -> dict:
    order = np.argsort(-scores)
    ranked = [
        RankedStory(
            story=fold.candidates[i], score=float(scores[i]), best_match_title=""
        )
        for i in order
    ]
    emb_map = {s.id: fold.cand_emb[i] for i, s in enumerate(fold.candidates)}
    rel_map = {0: 0.0, 1: 0.2, 2: 1.0}
    test_rel = np.array([rel_map[int(action)] for action in fold.test_actions])
    ideal = test_rel.tolist()

    def compute(rank_map: dict[int, int]) -> dict:
        rel_by_pos = {
            rank_map[s.id]: test_rel[i]
            for i, s in enumerate(fold.test_stories)
            if s.id in rank_map
        }

        def ndcg(k: int) -> float:
            dcg = sum(r / math.log2(p + 2) for p, r in rel_by_pos.items() if p < k)
            idcg = sum(
                r / math.log2(i + 2)
                for i, r in enumerate(sorted(ideal, reverse=True)[:k])
            )
            return dcg / idcg if idcg > 0 else 0.0

        up_positions = sorted(
            rank_map[s.id]
            for i, s in enumerate(fold.test_stories)
            if fold.test_actions[i] == 2 and s.id in rank_map
        )
        n_up = int((fold.test_actions == 2).sum())
        ap = (
            sum((idx + 1) / (pos + 1) for idx, pos in enumerate(up_positions)) / n_up
            if n_up and up_positions
            else 0.0
        )
        up_by_pos = set(up_positions)
        down_positions = {
            rank_map[s.id]
            for i, s in enumerate(fold.test_stories)
            if fold.test_actions[i] == 0 and s.id in rank_map
        }
        return {
            "ndcg_at_100": ndcg(100),
            "ndcg_at_40": ndcg(40),
            "ndcg_at_200": ndcg(200),
            "map": ap,
            "precision_at_40": sum(1 for p in up_by_pos if p < 40) / 40.0,
            "downvote_rate_at_40": sum(1 for p in down_positions if p < 40) / 40.0,
            "hit_at_100": sum(1 for p in rel_by_pos if p < 100)
            / max(len(fold.test_stories), 1),
        }

    raw_rank_map = {fold.candidates[idx].id: pos for pos, idx in enumerate(order)}
    top = mmr_filter(
        ranked,
        emb_map,
        threshold=config.model.diversity_threshold,
        limit=config.count,
    )
    mmr_rank_map = {item.story.id: pos for pos, item in enumerate(top)}
    up_ids = {
        s.id for i, s in enumerate(fold.test_stories) if fold.test_actions[i] == 2
    }
    up_ranks = [
        pos for pos, idx in enumerate(order) if fold.candidates[idx].id in up_ids
    ]
    ranks = {
        "median_rank": float(np.median(up_ranks)) if up_ranks else 0.0,
        "p25_rank": float(np.percentile(up_ranks, 25)) if up_ranks else 0.0,
        "p75_rank": float(np.percentile(up_ranks, 75)) if up_ranks else 0.0,
    }
    raw = compute(raw_rank_map) | ranks
    mmr = compute(mmr_rank_map) | ranks
    if probs is not None:
        id_to_idx = {s.id: i for i, s in enumerate(fold.candidates)}
        p_up = []
        y_up = []
        for i, story in enumerate(fold.test_stories):
            if story.id in id_to_idx:
                p_up.append(probs[id_to_idx[story.id], 2])
                y_up.append(1.0 if fold.test_actions[i] == 2 else 0.0)
        raw["brier_up"] = float(np.mean((np.array(p_up) - np.array(y_up)) ** 2))
        mmr["brier_up"] = raw["brier_up"]
    else:
        raw["brier_up"] = 0.0
        mmr["brier_up"] = 0.0
    return {"raw": raw, "mmr": mmr}


def _make_fold(
    candidates: list[Story],
    cand_emb: np.ndarray,
    cand_field_emb: np.ndarray,
    cand_field_parts: np.ndarray,
    fb_stories: list[Story],
    fb_to_cand: np.ndarray,
    fb_field_emb: np.ndarray,
    fb_field_parts: np.ndarray,
    fb_vote_times: np.ndarray,
    y: np.ndarray,
    valid_positions: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    config: Config,
    *,
    needs_field: bool,
) -> FoldData:
    train_story_indices = valid_positions[train_pos]
    test_story_indices = valid_positions[test_pos]
    train_ids = {fb_stories[idx].id for idx in train_story_indices}
    cand_mask = np.array([s.id not in train_ids for s in candidates])
    fold_candidates = [s for i, s in enumerate(candidates) if cand_mask[i]]
    fold_cand_emb = cand_emb[cand_mask]
    fold_cand_field_emb = cand_field_emb[cand_mask]
    fold_cand_field_parts = cand_field_parts[cand_mask]

    train_emb = cand_emb[fb_to_cand[train_story_indices]]
    train_field_emb = fb_field_emb[train_story_indices] if needs_field else train_emb
    train_field_parts = (
        fb_field_parts[train_story_indices]
        if needs_field
        else np.empty((0, 4, 384), dtype=np.float32)
    )
    y_train = y[train_pos]
    train_stories = [fb_stories[idx] for idx in train_story_indices]
    test_stories = [fb_stories[idx] for idx in test_story_indices]
    test_actions = y[test_pos]
    train_vote_times = fb_vote_times[train_pos]

    train_sim_up, train_sim_down, train_closest_up, train_closest_down = (
        _loocv_similarity_features(train_emb, y_train, config.model.knn_k)
    )
    cand_sim_up, cand_sim_down, cand_closest_up, cand_closest_down = (
        _candidate_similarity_features(
            fold_cand_emb, train_emb, y_train, config.model.knn_k
        )
    )
    if needs_field:
        (
            train_field_sim_up,
            train_field_sim_down,
            train_field_closest_up,
            train_field_closest_down,
        ) = _loocv_similarity_features(train_field_emb, y_train, config.model.knn_k)
        (
            cand_field_sim_up,
            cand_field_sim_down,
            cand_field_closest_up,
            cand_field_closest_down,
        ) = _candidate_similarity_features(
            fold_cand_field_emb, train_field_emb, y_train, config.model.knn_k
        )
        x_train_field = _feature_matrix(
            train_field_emb,
            train_stories,
            train_field_sim_up,
            train_field_sim_down,
            train_field_closest_up,
            train_field_closest_down,
            textsplit=False,
        )
        x_cand_field = _feature_matrix(
            fold_cand_field_emb,
            fold_candidates,
            cand_field_sim_up,
            cand_field_sim_down,
            cand_field_closest_up,
            cand_field_closest_down,
            textsplit=False,
        )
        train_field_sim_features = _field_similarity_feature_matrix(
            train_field_parts,
            train_field_parts,
            y_train,
            k=config.model.knn_k,
            loocv=True,
        )
        cand_field_sim_features = _field_similarity_feature_matrix(
            fold_cand_field_parts,
            train_field_parts,
            y_train,
            k=config.model.knn_k,
            loocv=False,
        )
        train_base_meta = _feature_matrix(
            train_emb,
            train_stories,
            train_sim_up,
            train_sim_down,
            train_closest_up,
            train_closest_down,
            textsplit=True,
        )[:, train_emb.shape[1] :]
        cand_base_meta = _feature_matrix(
            fold_cand_emb,
            fold_candidates,
            cand_sim_up,
            cand_sim_down,
            cand_closest_up,
            cand_closest_down,
            textsplit=True,
        )[:, fold_cand_emb.shape[1] :]
        x_train_field_sims = np.hstack(
            [train_emb, train_base_meta, train_field_sim_features]
        ).astype(np.float32)
        x_cand_field_sims = np.hstack(
            [fold_cand_emb, cand_base_meta, cand_field_sim_features]
        ).astype(np.float32)
    else:
        x_train_field = np.empty((0, 0), dtype=np.float32)
        x_cand_field = np.empty((0, 0), dtype=np.float32)
        x_train_field_sims = np.empty((0, 0), dtype=np.float32)
        x_cand_field_sims = np.empty((0, 0), dtype=np.float32)
    return FoldData(
        candidates=fold_candidates,
        cand_emb=fold_cand_emb,
        cand_field_emb=fold_cand_field_emb,
        cand_field_parts=fold_cand_field_parts,
        train_stories=train_stories,
        test_stories=test_stories,
        test_actions=test_actions,
        train_vote_times=train_vote_times,
        x_train_base=_feature_matrix(
            train_emb,
            train_stories,
            train_sim_up,
            train_sim_down,
            train_closest_up,
            train_closest_down,
            textsplit=False,
        ),
        x_cand_base=_feature_matrix(
            fold_cand_emb,
            fold_candidates,
            cand_sim_up,
            cand_sim_down,
            cand_closest_up,
            cand_closest_down,
            textsplit=False,
        ),
        x_train_field=x_train_field,
        x_cand_field=x_cand_field,
        x_train_field_sims=x_train_field_sims,
        x_cand_field_sims=x_cand_field_sims,
        x_train_textsplit=_feature_matrix(
            train_emb,
            train_stories,
            train_sim_up,
            train_sim_down,
            train_closest_up,
            train_closest_down,
            textsplit=True,
        ),
        x_cand_textsplit=_feature_matrix(
            fold_cand_emb,
            fold_candidates,
            cand_sim_up,
            cand_sim_down,
            cand_closest_up,
            cand_closest_down,
            textsplit=True,
        ),
        y_train=y_train,
        tier2_scores=_tier2_scores(fold_cand_emb, train_emb, y_train),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", default="eval_ranker_variants.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--window-days",
        type=int,
        help="Override the candidate story age window for evaluation only.",
    )
    parser.add_argument("--svm-c", type=float)
    parser.add_argument("--svm-gamma", type=float)
    parser.add_argument(
        "--variants",
        help="Comma-separated variant names to run. Defaults to all variants.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Sample at most this many candidate stories after loading the eval window.",
    )
    parser.add_argument(
        "--max-feedback-per-class",
        type=int,
        help="Sample at most this many valid feedback stories per label class.",
    )
    parser.add_argument(
        "--leak-check",
        action="store_true",
        help=(
            "After running the normal variant suite, run it again with "
            "y (labels) shuffled (fixed seed). A trustworthy harness should "
            "see shuffled NDCG@40 drop to random baseline (~0.10 for "
            "n_test=80/n_cand=7000). High shuffled values indicate data "
            "leakage in the offline harness."
        ),
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.svm_c is not None or args.svm_gamma is not None:
        model = replace(
            config.model,
            svm_c=args.svm_c if args.svm_c is not None else config.model.svm_c,
            svm_gamma=(
                args.svm_gamma if args.svm_gamma is not None else config.model.svm_gamma
            ),
        )
        config = replace(config, model=ModelConfig(**model.__dict__))
    db = Database(config.db_path)
    user = db.get_user_by_token("default")
    if user is None:
        raise RuntimeError("Missing default user token")

    requested = (
        [name.strip() for name in args.variants.split(",") if name.strip()]
        if args.variants
        else []
    )
    needs_field = (not requested) or any(
        name.startswith("field_") for name in requested
    )
    embedder = Embedder(config.onnx_model_dir) if needs_field else None

    window_days = args.window_days if args.window_days is not None else config.days
    cutoff_ts = int(time.time()) - window_days * 86400
    candidates, cand_emb = _load_recent_candidates(db, cutoff_ts)
    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training(user_id=user.id)
    all_y = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories])
    valid_mask = fb_to_cand >= 0

    if args.max_feedback_per_class is not None:
        rng = np.random.default_rng(0)
        keep_feedback_positions = []
        for label in (0, 1, 2):
            positions = np.where(valid_mask & (all_y == label))[0]
            if len(positions) > args.max_feedback_per_class:
                positions = rng.choice(
                    positions, size=args.max_feedback_per_class, replace=False
                )
            keep_feedback_positions.extend(int(pos) for pos in positions)
        keep_feedback_positions = np.array(sorted(keep_feedback_positions), dtype=int)
        fb_stories = [fb_stories[i] for i in keep_feedback_positions]
        all_y = all_y[keep_feedback_positions]
        fb_vote_times = fb_vote_times[keep_feedback_positions]
        fb_to_cand = fb_to_cand[keep_feedback_positions]
        valid_mask = fb_to_cand >= 0

    if args.max_candidates is not None and len(candidates) > args.max_candidates:
        rng = np.random.default_rng(1)
        required_ids = {s.id for s in fb_stories if s.id in cand_id_to_idx}
        required_indices = {cand_id_to_idx[sid] for sid in required_ids}
        if len(required_indices) > args.max_candidates:
            raise RuntimeError(
                "--max-candidates is smaller than the sampled valid feedback set"
            )
        remaining_slots = args.max_candidates - len(required_indices)
        optional_indices = np.array(
            [i for i in range(len(candidates)) if i not in required_indices],
            dtype=int,
        )
        if remaining_slots < len(optional_indices):
            optional_indices = rng.choice(
                optional_indices, size=remaining_slots, replace=False
            )
        keep_candidate_indices = np.array(
            sorted(set(required_indices) | {int(i) for i in optional_indices}),
            dtype=int,
        )
        candidates = [candidates[i] for i in keep_candidate_indices]
        cand_emb = cand_emb[keep_candidate_indices]
        cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
        fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories])
        valid_mask = fb_to_cand >= 0

    cand_field_parts = (
        _field_embeddings_by_field(candidates, embedder)
        if needs_field and embedder is not None
        else np.empty((len(candidates), 4, cand_emb.shape[1]), dtype=np.float32)
    )
    fb_field_parts = np.empty((len(fb_stories), 4, cand_emb.shape[1]), dtype=np.float32)
    if needs_field and embedder is not None:
        fb_field_parts.fill(0.0)
        valid_fb_positions = np.where(valid_mask)[0]
        fb_field_parts[valid_fb_positions] = cand_field_parts[
            fb_to_cand[valid_fb_positions]
        ]
    else:
        fb_field_parts = np.empty(
            (len(fb_stories), 4, cand_emb.shape[1]), dtype=np.float32
        )
    cand_field_emb = _average_field_embeddings(cand_field_parts)
    fb_field_emb = _average_field_embeddings(fb_field_parts)
    valid_positions = np.where(valid_mask)[0]
    y = all_y[valid_mask]
    fb_vote_times = fb_vote_times[valid_mask]
    label_names = {0: "down", 1: "neutral", 2: "up"}
    candidate_recall = {}
    for label in (0, 1, 2):
        total = int((all_y == label).sum())
        present = int(((all_y == label) & valid_mask).sum())
        candidate_recall[label_names[label]] = {
            "present": present,
            "total": total,
            "recall": present / total if total else 0.0,
        }

    print(
        f"user={user.token} candidates={len(candidates)} "
        f"valid_feedback={len(y)} labels={Counter(y)}"
    )
    print(f"candidate_recall={candidate_recall}")
    if len(set(y)) < 3:
        raise RuntimeError("Need all three labels for stratified evaluation")

    variants = {
        "legacy_prob_3class": lambda fold: _scores_prob_3class(
            fold, config, textsplit=False
        ),
        "prob_3class_textsplit": lambda fold: _scores_prob_3class(
            fold, config, textsplit=True
        ),
        "margin3_up": lambda fold: _scores_margin_3class(
            fold, config, textsplit=False, mode="up"
        ),
        "linear_svc_up": lambda fold: _scores_linear_svc_up(fold, config),
        "logreg_up": lambda fold: _scores_logreg_up(fold, config),
        "sgd_log_up": lambda fold: _scores_sgd_log_up(fold, config),
        "mlp_32_a1e-3": lambda fold: _scores_mlp_up(fold, config, (32,), 1e-3),
        "mlp_64_a1e-3": lambda fold: _scores_mlp_up(fold, config, (64,), 1e-3),
        "mlp_64_16_a1e-3": lambda fold: _scores_mlp_up(fold, config, (64, 16), 1e-3),
        "source_domain_margin3_up": lambda fold: _scores_margin_source_domain(
            fold, config
        ),
        "rank_calibrated_tier2_10": lambda fold: _scores_margin_rank_calibrated(
            fold, config, tier2_weight=0.10
        ),
        "rank_calibrated_tier2_20": lambda fold: _scores_margin_rank_calibrated(
            fold, config, tier2_weight=0.20
        ),
        "rank_calibrated_tier2_30": lambda fold: _scores_margin_rank_calibrated(
            fold, config, tier2_weight=0.30
        ),
        "field_margin3_up": lambda fold: _scores_margin_3class_field(fold, config),
        "field_sims_margin3_up": lambda fold: _scores_margin_3class_field_sims(
            fold, config
        ),
        "margin3_up_recency30d": lambda fold: _scores_margin_3class(
            fold, config, textsplit=False, mode="up", half_life_days=30.0
        ),
        "field_margin3_up_recency30d": lambda fold: _scores_margin_3class_field(
            fold, config, half_life_days=30.0
        ),
        "margin3_up_recency90d": lambda fold: _scores_margin_3class(
            fold, config, textsplit=False, mode="up", half_life_days=90.0
        ),
        "margin3_up_minus_down": lambda fold: _scores_margin_3class(
            fold, config, textsplit=False, mode="up_minus_down"
        ),
        "positive_clusters_4": lambda fold: _scores_positive_clusters(
            fold, n_clusters=4
        ),
        "positive_clusters_8": lambda fold: _scores_positive_clusters(
            fold, n_clusters=8
        ),
        "pos_cluster_feat_ff4": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=4, method="ff", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_ff8": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=8, method="ff", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_ff12": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=12, method="ff", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans4": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=4, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans2": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=2, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans3": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=3, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans5": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=5, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans6": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=6, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans8": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=8, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans12": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=12, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans16": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=16, method="kmeans", feature_names=("pos_cluster",)
        ),
        "pos_cluster_feat_kmeans24": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=24, method="kmeans", feature_names=("pos_cluster",)
        ),
        "neg_cluster_feat_kmeans4": lambda fold: _scores_margin_with_cluster_features(
            fold, config, n_clusters=4, method="kmeans", feature_names=("neg_cluster",)
        ),
        "cluster_margin_feat_kmeans4": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=4,
            method="kmeans",
            feature_names=("cluster_margin",),
        ),
        "pos_cluster_entropy_kmeans4": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=4,
            method="kmeans",
            feature_names=("pos_cluster_entropy",),
        ),
        "centroid_residual_kmeans4": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=4,
            method="kmeans",
            feature_names=("centroid_residual",),
        ),
        "down_up_interaction_kmeans4": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=4,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans2": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=2,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans3": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=3,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans5": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=5,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans6": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=6,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans8": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=8,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans12": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=12,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans16": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=16,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "down_up_interaction_kmeans24": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=24,
            method="kmeans",
            feature_names=("down_up_interaction",),
        ),
        "cluster_combo_kmeans4": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=4,
            method="kmeans",
            feature_names=(
                "pos_cluster",
                "neg_cluster",
                "cluster_margin",
                "pos_cluster_entropy",
                "centroid_residual",
                "down_up_interaction",
            ),
        ),
        "cluster_combo_kmeans2": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=2,
            method="kmeans",
            feature_names=(
                "pos_cluster",
                "neg_cluster",
                "cluster_margin",
                "pos_cluster_entropy",
                "centroid_residual",
                "down_up_interaction",
            ),
        ),
        "cluster_combo_kmeans8": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=8,
            method="kmeans",
            feature_names=(
                "pos_cluster",
                "neg_cluster",
                "cluster_margin",
                "pos_cluster_entropy",
                "centroid_residual",
                "down_up_interaction",
            ),
        ),
        "cluster_combo_kmeans12": lambda fold: _scores_margin_with_cluster_features(
            fold,
            config,
            n_clusters=12,
            method="kmeans",
            feature_names=(
                "pos_cluster",
                "neg_cluster",
                "cluster_margin",
                "pos_cluster_entropy",
                "centroid_residual",
                "down_up_interaction",
            ),
        ),
        "margin3_up_hardneg_2": lambda fold: _scores_margin_3class(
            fold,
            config,
            textsplit=False,
            mode="up",
            hard_negative_multiplier=2.0,
        ),
        "margin3_up_hardneg_4": lambda fold: _scores_margin_3class(
            fold,
            config,
            textsplit=False,
            mode="up",
            hard_negative_multiplier=4.0,
        ),
        "margin3_weight_neutral05": lambda fold: _scores_margin_3class(
            fold,
            config,
            textsplit=False,
            mode="up",
            label_weight_multipliers={1: 0.5},
        ),
        "margin3_weight_neutral025": lambda fold: _scores_margin_3class(
            fold,
            config,
            textsplit=False,
            mode="up",
            label_weight_multipliers={1: 0.25},
        ),
        "margin3_weight_down15_neutral05": lambda fold: _scores_margin_3class(
            fold,
            config,
            textsplit=False,
            mode="up",
            label_weight_multipliers={0: 1.5, 1: 0.5},
        ),
        "margin3_textsplit_up": lambda fold: _scores_margin_3class(
            fold, config, textsplit=True, mode="up"
        ),
        "binary_margin_no_neutral": lambda fold: _scores_binary_margin(
            fold, config, textsplit=False, probability=False
        ),
        "binary_margin_no_neutral_recency30d": lambda fold: _scores_binary_margin(
            fold, config, textsplit=False, probability=False, half_life_days=30.0
        ),
        "binary_prob_no_neutral": lambda fold: _scores_binary_margin(
            fold, config, textsplit=False, probability=True
        ),
        "binary_margin_textsplit": lambda fold: _scores_binary_margin(
            fold, config, textsplit=True, probability=False
        ),
        "binary_prob_textsplit": lambda fold: _scores_binary_margin(
            fold, config, textsplit=True, probability=True
        ),
        "pairwise_base": lambda fold: _scores_pairwise(fold, textsplit=False),
        "pairwise_up_only": lambda fold: _scores_pairwise(
            fold, textsplit=False, strict_up_only=True
        ),
        "pairwise_textsplit": lambda fold: _scores_pairwise(fold, textsplit=True),
        "tier2_centroid": lambda fold: (fold.tier2_scores.copy(), None),
    }
    # Register DL variants only if torch is available. This lets the
    # harness run all 65 non-DL variants on a stock install; DL/blend
    # variants are opt-in via `uv sync --group dl-experiment`.
    if _TORCH_AVAILABLE:
        variants.update(
            {
                "attention_mlp": lambda fold: _scores_attention_mlp_up(fold, config),
                "attn_mlp_t0": lambda fold: _scores_attention_mlp_t0_up(fold, config),
                "attn_mlp_t0_cos": lambda fold: _scores_attention_mlp_t0_cos_up(
                    fold, config
                ),
                "attn_mlp_v1_nocos": lambda fold: _scores_attention_mlp_v1_nocos_up(
                    fold, config
                ),
                "attn_mlp_v1_h64": lambda fold: _scores_attention_mlp_v1_h64_up(
                    fold, config
                ),
                "attn_mlp_v2_rank": lambda fold: _scores_attention_mlp_v2_rank_up(
                    fold, config
                ),
                "attn_mlp_v2_meta": lambda fold: _scores_attention_mlp_v2_meta_up(
                    fold, config
                ),
                "attn_mlp_v2_mixup": lambda fold: _scores_attention_mlp_v2_mixup_up(
                    fold, config
                ),
                "attn_mlp_v2_all": lambda fold: _scores_attention_mlp_v2_all_up(
                    fold, config
                ),
                "blend_score_10": lambda fold: _scores_blend_up(
                    fold, config, 0.10, kind="score"
                ),
                "blend_score_25": lambda fold: _scores_blend_up(
                    fold, config, 0.25, kind="score"
                ),
                "blend_score_50": lambda fold: _scores_blend_up(
                    fold, config, 0.50, kind="score"
                ),
                "blend_score_75": lambda fold: _scores_blend_up(
                    fold, config, 0.75, kind="score"
                ),
                "blend_score_90": lambda fold: _scores_blend_up(
                    fold, config, 0.90, kind="score"
                ),
                "blend_rank_10": lambda fold: _scores_blend_up(
                    fold, config, 0.10, kind="rank"
                ),
                "blend_rank_25": lambda fold: _scores_blend_up(
                    fold, config, 0.25, kind="rank"
                ),
                "blend_rank_50": lambda fold: _scores_blend_up(
                    fold, config, 0.50, kind="rank"
                ),
                "blend_rank_75": lambda fold: _scores_blend_up(
                    fold, config, 0.75, kind="rank"
                ),
                "blend_rank_90": lambda fold: _scores_blend_up(
                    fold, config, 0.90, kind="rank"
                ),
            }
        )
    if requested:
        missing = sorted(set(requested) - set(variants))
        if missing:
            if any(
                m.startswith(("attention_mlp", "attn_mlp", "blend_")) for m in missing
            ):
                raise RuntimeError(
                    f"Requested DL/blend variants not available: {', '.join(missing)}. "
                    "Install torch with `uv sync --group dl-experiment`."
                )
            raise ValueError(f"Unknown variants: {', '.join(missing)}")
        variants = {name: variants[name] for name in requested}

    def _run_variants(y_label: np.ndarray, label: str = "") -> dict[str, list[dict]]:
        """Run the variant suite on a given label vector.

        label is a prefix prepended to per-fold progress lines (used by
        the leak check pass to distinguish its output from the main run).
        """
        results: dict[str, list[dict]] = {name: [] for name in variants}
        splits = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=0
        ).split(np.zeros((len(y_label), 1)), y_label)
        for fold_no, (train_pos, test_pos) in enumerate(splits, start=1):
            fold = _make_fold(
                candidates,
                cand_emb,
                cand_field_emb,
                cand_field_parts,
                fb_stories,
                fb_to_cand,
                fb_field_emb,
                fb_field_parts,
                fb_vote_times,
                y_label,
                valid_positions,
                train_pos,
                test_pos,
                config,
                needs_field=needs_field,
            )
            for name, scorer in variants.items():
                scores, probs = scorer(fold)
                results[name].append(_metrics(scores, fold, config, probs))
            print(f"{label}fold {fold_no}/{args.folds} done")
        return results

    results = _run_variants(y)

    metric_keys = [
        "ndcg_at_100",
        "ndcg_at_40",
        "ndcg_at_200",
        "map",
        "precision_at_40",
        "downvote_rate_at_40",
        "hit_at_100",
        "median_rank",
        "p25_rank",
        "p75_rank",
        "brier_up",
    ]
    report: dict[str, Any] = {
        "config": {
            "split": f"{args.folds}-fold-stratified",
            "window_days": window_days,
            "user_token": user.token,
            "user_id": user.id,
            "n_candidates": len(candidates),
            "n_feedback_valid": len(y),
            "labels": {str(k): int(v) for k, v in Counter(y).items()},
            "candidate_recall": candidate_recall,
            "svm_c": config.model.svm_c,
            "svm_gamma": config.model.svm_gamma,
            "knn_k": config.model.knn_k,
            "positive_cluster_k": config.model.positive_cluster_k,
            "mmr_threshold": config.model.diversity_threshold,
            "mmr_limit": config.count,
            "db_sha256": _db_sha256(config.db_path),
        },
        "variants": {},
    }
    for name, rows in results.items():
        report["variants"][name] = {
            "mean": {
                side: {
                    key: float(np.mean([row[side][key] for row in rows]))
                    for key in metric_keys
                }
                for side in ("raw", "mmr")
            },
            "std": {
                side: {
                    key: float(np.std([row[side][key] for row in rows]))
                    for key in metric_keys
                }
                for side in ("raw", "mmr")
            },
            "per_fold": rows,
        }

    if args.leak_check:
        print(f"\n=== Leak check: shuffling y (n={len(y)}, seed=0) ===")
        y_shuffled = np.random.default_rng(0).permutation(y)
        leak_results = _run_variants(y_shuffled, label="[leak-check] ")
        report["leak_check"] = {
            "config": {
                "y_seed": 0,
                "n_feedback_valid": len(y_shuffled),
                "labels": {str(k): int(v) for k, v in Counter(y_shuffled).items()},
            },
            "variants": {},
        }
        for name, rows in leak_results.items():
            report["leak_check"]["variants"][name] = {
                "mean": {
                    side: {
                        key: float(np.mean([row[side][key] for row in rows]))
                        for key in metric_keys
                    }
                    for side in ("raw", "mmr")
                },
                "std": {
                    side: {
                        key: float(np.std([row[side][key] for row in rows]))
                        for key in metric_keys
                    }
                    for side in ("raw", "mmr")
                },
                "per_fold": rows,
            }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")
    for name, data in report["variants"].items():
        raw = data["mean"]["raw"]
        mmr = data["mean"]["mmr"]
        print(
            f"{name:26s} raw_ndcg100={raw['ndcg_at_100']:.3f} "
            f"raw_ndcg40={raw['ndcg_at_40']:.3f} "
            f"mmr_ndcg100={mmr['ndcg_at_100']:.3f} "
            f"raw_map={raw['map']:.3f} mmr_map={mmr['map']:.3f} "
            f"median={raw['median_rank']:.1f}"
        )

    if args.leak_check and "leak_check" in report:
        print(f"\n{'=' * 80}")
        print(
            f"{'Variant':26s} {'normal raw40':>14s} {'shuffled raw40':>16s} {'ratio':>8s}"
        )
        print("-" * 80)
        for name in report["variants"]:
            if name not in report["leak_check"]["variants"]:
                continue
            normal_n = report["variants"][name]["mean"]["raw"]["ndcg_at_40"]
            # ty loses dict nesting through the long subscript chain; Any cast
            # at the variants level lets it find the ndcg_at_40 leaf.
            leak_variants: Any = report["leak_check"]["variants"]
            shuffled_n = leak_variants[name]["mean"]["raw"]["ndcg_at_40"]
            ratio = shuffled_n / normal_n if normal_n > 1e-9 else float("inf")
            print(f"{name:26s} {normal_n:14.4f} {shuffled_n:16.4f} {ratio:8.2f}")
            if ratio > 0.5:
                print(
                    f"{'':26s} WARNING: shuffled/raw ratio > 0.5, "
                    f"possible data leakage in harness"
                )
        print("=" * 80)


if __name__ == "__main__":
    main()
