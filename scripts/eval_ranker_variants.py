#!/usr/bin/env python3
"""Canonical current-snapshot retrospective ranking diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import sqlite3
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import time
from collections import Counter
from typing import Any
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from database import Database, Story
from dedup import normalize_url
from pipeline import (
    Config,
    Embedder,
    ModelConfig,
    RankedStory,
    _knn_similarity,
    load_production_candidate_stories,
    mmr_filter,
    story_embedding_text,
)


@dataclass(frozen=True)
class FoldData:
    candidates: list[Story]
    cand_emb: np.ndarray
    train_stories: list[Story]
    test_stories: list[Story]
    test_actions: np.ndarray
    train_vote_times: np.ndarray
    x_train_base: np.ndarray
    x_cand_base: np.ndarray
    y_train: np.ndarray
    tier2_scores: np.ndarray
    train_emb: np.ndarray | None = None
    runtime_db: Database | None = None
    similarities: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldSplit:
    fold_no: int
    train_pos: np.ndarray
    test_pos: np.ndarray


def _db_sha256(db_path: str) -> str:
    with Path(db_path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def frozen_database(path: str):
    """SQLite backup includes committed WAL pages; both reader handles are read-only."""
    with tempfile.TemporaryDirectory(prefix="rank-eval-") as directory:
        target = Path(directory) / "snapshot.db"
        source = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        digest = _db_sha256(str(target))
        db = Database(str(target), read_only=True)
        try:
            yield db, digest
        finally:
            db.close()


def _snapshot_context(path: str | Path) -> tuple[Config, float, str]:
    from pipeline.config import RssConfig

    with np.load(path, allow_pickle=False) as data:
        required = {
            "config_json",
            "evaluation_time",
            "database_snapshot",
            "database_sha256",
            "user_id",
        }
        if not required <= set(data.files):
            raise ValueError(
                "Snapshot lacks frozen configuration/time/database; regenerate bakeoff"
            )
        raw = json.loads(str(data["config_json"].item()))
        model = raw.pop("model")
        model["dedup_exclude_actions"] = tuple(model["dedup_exclude_actions"])
        rss = raw.pop("rss")
        rss["feeds"] = tuple(rss["feeds"])
        config = Config(**raw, model=ModelConfig(**model), rss=RssConfig(**rss))
        now = float(data["evaluation_time"].item())
        database_path = str(data["database_snapshot"].item())
        digest = str(data["database_sha256"].item())
    if not math.isfinite(now) or _db_sha256(database_path) != digest:
        raise ValueError("Frozen snapshot timestamp or database hash is invalid")
    return config, now, database_path


def _validated_embeddings(
    stories: list[Story], cached: dict[int, np.ndarray]
) -> np.ndarray:
    # Production is 384-d; replay embeddings from other models may differ, but
    # every vector must share one dimension.
    dims = {v.shape for v in cached.values()}
    dim = dims.pop()[0] if len(dims) == 1 else 384
    invalid = [
        s.id
        for s in stories
        if s.id not in cached
        or cached[s.id].shape != (dim,)
        or not np.isfinite(cached[s.id]).all()
        or not np.isclose(np.linalg.norm(cached[s.id]), 1.0, atol=1e-3)
    ]
    if invalid:
        raise ValueError(
            f"Embedding coverage: {len(stories) - len(invalid)}/{len(stories)} valid; "
            f"missing/invalid IDs (first 20): {invalid[:20]}"
        )
    return np.asarray([cached[s.id] for s in stories], dtype=np.float32).reshape(
        -1, dim
    )


class _FrozenEmbedder(Embedder):
    def __init__(self, model_version: str, embedding_dim: int = 384) -> None:
        self.model_version = model_version
        self.embedding_dim = embedding_dim

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        raise RuntimeError("Evaluation attempted to compute an uncaptured embedding")


@contextmanager
def _fold_database(fold: FoldData, config: Config, source_db: Database | None):
    if fold.runtime_db is not None:
        yield fold.runtime_db
        return
    db = Database(":memory:")
    try:
        stories = {s.id: s for s in fold.candidates + fold.train_stories}
        vectors = dict(zip([s.id for s in fold.candidates], fold.cand_emb, strict=True))
        train_emb = fold.train_emb
        if train_emb is None:
            train_emb = fold.x_train_base[:, : fold.cand_emb.shape[1]]
        vectors.update(zip([s.id for s in fold.train_stories], train_emb, strict=True))
        for story in stories.values():
            db.upsert_story(story)
            db.upsert_embedding(
                story.id,
                config.embedding_model_version,
                str(_embedding_text_hashes([story])[0]),
                vectors[story.id],
            )
        for story, label in zip(fold.train_stories, fold.y_train, strict=True):
            db.upsert_feedback(1, story.id, ("down", "neutral", "up")[int(label)])
        with db.conn() as conn:
            conn.executemany(
                "UPDATE feedback SET updated_at=? WHERE user_id=1 AND story_id=?",
                [
                    (float(t), s.id)
                    for s, t in zip(
                        fold.train_stories, fold.train_vote_times, strict=True
                    )
                ],
            )
            conn.commit()
        # Canonical resolution and target metadata come from the same read-only snapshot.
        with ExitStack() as stack:
            if source_db is not None:
                stack.enter_context(
                    patch.object(
                        db, "get_hn_dupe_resolutions", source_db.get_hn_dupe_resolutions
                    )
                )
                local_get_story = db.get_story
                stack.enter_context(
                    patch.object(
                        db,
                        "get_story",
                        lambda sid: local_get_story(sid) or source_db.get_story(sid),
                    )
                )
            yield db
    finally:
        db.close()


@contextmanager
def _reuse_preprocessing():
    """Retain only this fold's compatible scaling results and their input arrays."""
    prepared: dict[tuple[int, int, int], tuple] = {}
    fit_scale = _fit_scale

    def reuse_scale(train: np.ndarray, cand: np.ndarray, emb_dim: int) -> tuple:
        key = (id(train), id(cand), emb_dim)
        if key not in prepared:
            prepared[key] = (train, cand, fit_scale(train, cand, emb_dim))
        return prepared[key][2]

    with patch(__name__ + "._fit_scale", side_effect=reuse_scale):
        yield


def _production_scores(
    fold: FoldData, config: Config, source_db: Database | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    from pipeline.ranking import RankScoreContext, RankTrace, _score_and_rank

    if not fold.train_stories:
        from pipeline import cold_ranked_candidates

        ranked = cold_ranked_candidates(fold.candidates, int(time.time()))
        return np.array([r.score for r in ranked]), None

    trace = RankTrace()
    score_context = RankScoreContext()
    with _fold_database(fold, config, source_db) as db:
        ranked = _score_and_rank(
            fold.candidates,
            fold.cand_emb,
            db,
            config,
            _FrozenEmbedder(config.embedding_model_version, fold.cand_emb.shape[1]),
            trace=trace,
            score_context=score_context,
        )
    if trace.labels.get("svm_fit") == "error":
        raise RuntimeError(
            "Production fit failed; aborting fold instead of scoring fallback"
        )
    if trace.labels.get("svm_probs") == "error":
        raise RuntimeError("Production probability mapping failed; aborting fold")
    if (
        score_context.cand_closest_up is not None
        and score_context.cand_closest_down is not None
        and score_context.cand_closest_neutral is not None
    ):
        fold.similarities.update(
            {
                2: score_context.cand_closest_up,
                0: score_context.cand_closest_down,
                1: score_context.cand_closest_neutral,
            }
        )
    by_id = {r.story.id: r for r in ranked}
    probabilities = (
        np.array(
            [
                [by_id[s.id].prob_down, by_id[s.id].prob_neutral, by_id[s.id].prob_up]
                for s in fold.candidates
            ],
            dtype=np.float32,
        )
        if ranked and ranked[0].prob_up is not None
        else None
    )
    return np.array([by_id[s.id].score for s in fold.candidates]), probabilities


def _recommended(
    scores: np.ndarray,
    fold: FoldData,
    config: Config,
    probs: np.ndarray | None,
    source_db: Database | None,
) -> list[RankedStory]:
    from pipeline import finalize_ranked_deck
    from pipeline.ranking import assemble_ranked_deck
    from pipeline.hn_dupes import (
        _load_feedback_context,
        _matches_feedback,
        canonicalize_hn_dupes,
    )

    ranked = [
        RankedStory(
            story=fold.candidates[i],
            score=float(scores[i]),
            best_match_title="",
            prob_down=float(probs[i, 0]) if probs is not None else None,
            prob_neutral=float(probs[i, 1]) if probs is not None else None,
            prob_up=float(probs[i, 2]) if probs is not None else None,
        )
        for i in np.argsort(-scores, kind="stable")
    ]
    embedder = _FrozenEmbedder(config.embedding_model_version, fold.cand_emb.shape[1])
    with _fold_database(fold, config, source_db) as db:
        context = _load_feedback_context(
            db, user_id=1, actions=tuple(config.model.dedup_exclude_actions)
        )
        if not fold.train_stories:
            from pipeline import build_cold_deck

            deck = build_cold_deck(db, config, candidates=fold.candidates)
            return canonicalize_hn_dupes(
                deck,
                db,
                selected_limit=config.count,
                user_id=1,
                feedback_actions=tuple(config.model.dedup_exclude_actions),
            )
        from pipeline.ranking import RankScoreContext, _chunked_max_dot

        if not fold.similarities:
            train_emb = (
                fold.train_emb
                if fold.train_emb is not None
                else fold.x_train_base[:, : fold.cand_emb.shape[1]]
            )
            for label in (0, 1, 2):
                fold.similarities[label] = _chunked_max_dot(
                    fold.cand_emb, train_emb[fold.y_train == label]
                )

        score_context = RankScoreContext(
            cand_closest_up=fold.similarities[2],
            cand_closest_down=fold.similarities[0],
            cand_closest_neutral=fold.similarities[1],
        )
        deck = assemble_ranked_deck(
            ranked,
            fold.candidates,
            fold.cand_emb,
            db,
            config,
            embedder,
            user_id=1,
            score_context=score_context,
            is_feedback_match=lambda s: _matches_feedback(s, context),
        )
        return finalize_ranked_deck(
            deck, fold.candidates, fold.cand_emb, db, config, embedder, 1
        )


def _paired_differences(results: dict[str, list[dict]]) -> dict:
    reference = results["production"]
    differences = {}
    for name, rows in results.items():
        if name == "production":
            continue
        if len(rows) != len(reference):
            raise ValueError("Cannot pair incomplete folds")
        differences[name] = [
            {
                side: {
                    key: row[side][key] - baseline[side][key]
                    if row[side][key] is not None and baseline[side][key] is not None
                    else None
                    for key in row[side]
                }
                for side in row
            }
            for row, baseline in zip(rows, reference, strict=True)
        ]
    return _aggregate_results(differences)


def _embedding_text_hashes(stories: list[Story]) -> np.ndarray:
    return np.array(
        [
            hashlib.sha256(story_embedding_text(story).encode("utf-8")).hexdigest()
            for story in stories
        ],
        dtype="<U64",
    )


def _load_external_embeddings(path: str | Path, stories: list[Story]) -> np.ndarray:
    """Load a model-bakeoff embedding snapshot after validating its story rows."""
    with np.load(path, allow_pickle=False) as data:
        required = {"story_ids", "text_hashes", "embeddings"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Embedding snapshot missing required array(s): {', '.join(sorted(missing))}"
            )
        story_ids = np.asarray(data["story_ids"], dtype=np.int64)
        text_hashes = np.asarray(data["text_hashes"], dtype="<U64")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    expected_hashes = _embedding_text_hashes(stories)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(stories):
        raise ValueError(
            "Embedding snapshot shape does not match production candidates: "
            f"{embeddings.shape} for {len(stories)} stories"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding snapshot contains non-finite values")
    snapshot_keys = list(zip(story_ids.tolist(), text_hashes.tolist(), strict=True))
    expected_keys = list(
        zip([story.id for story in stories], expected_hashes.tolist(), strict=True)
    )
    if len(set(snapshot_keys)) != len(snapshot_keys):
        raise ValueError(
            "Embedding snapshot contains duplicate story IDs or text hashes"
        )
    if set(snapshot_keys) != set(expected_keys):
        raise ValueError(
            "Embedding snapshot story IDs or text hashes do not match production candidates"
        )
    vectors_by_key = dict(zip(snapshot_keys, embeddings, strict=True))
    ordered = np.array([vectors_by_key[key] for key in expected_keys], dtype=np.float32)
    return _validated_embeddings(
        stories, dict(zip([s.id for s in stories], ordered, strict=True))
    )


def _snapshot_candidate_stories(path: str | Path) -> list[Story]:
    """Load the frozen candidate rows stored alongside an embedding snapshot."""
    with np.load(path, allow_pickle=False) as data:
        if "stories_json" not in data.files:
            raise ValueError("Embedding snapshot missing required array: stories_json")
        raw_json = str(data["stories_json"].item())
    try:
        raw_stories = json.loads(raw_json)
        if not isinstance(raw_stories, list):
            raise ValueError("stories_json must contain a list")
        return [Story(**raw_story) for raw_story in raw_stories]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Embedding snapshot has invalid frozen story rows") from exc


def _snapshot_feedback(
    path: str | Path,
) -> tuple[list[Story], np.ndarray, np.ndarray]:
    """Load the feedback labels and times captured with an embedding snapshot."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "feedback_story_ids",
            "feedback_stories_json",
            "feedback_labels",
            "feedback_vote_times",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                "Embedding snapshot missing frozen feedback array(s): "
                + ", ".join(sorted(missing))
            )
        story_ids = np.asarray(data["feedback_story_ids"], dtype=np.int64)
        raw_labels = np.asarray(data["feedback_labels"])
        if raw_labels.ndim != 1 or not np.isin(raw_labels, [0, 1, 2]).all():
            raise ValueError(
                "Embedding snapshot frozen feedback labels must be 0, 1, or 2"
            )
        labels = raw_labels.astype(int)
        vote_times = np.asarray(data["feedback_vote_times"], dtype=np.float64)
        raw_json = str(data["feedback_stories_json"].item())
    try:
        raw_stories = json.loads(raw_json)
        if not isinstance(raw_stories, list):
            raise ValueError("feedback_stories_json must contain a list")
        stories = [Story(**raw_story) for raw_story in raw_stories]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Embedding snapshot has invalid frozen feedback rows") from exc
    if not (len(story_ids) == len(stories) == len(labels) == len(vote_times)):
        raise ValueError(
            "Embedding snapshot frozen feedback arrays have different lengths"
        )
    if not np.array_equal(story_ids, np.array([story.id for story in stories])):
        raise ValueError(
            "Embedding snapshot frozen feedback story IDs do not match rows"
        )
    if len(set(story_ids.tolist())) != len(story_ids):
        raise ValueError(
            "Embedding snapshot frozen feedback contains duplicate story IDs"
        )
    if not set(labels.tolist()) <= {0, 1, 2}:
        raise ValueError("Embedding snapshot frozen feedback labels must be 0, 1, or 2")
    if not np.isfinite(vote_times).all():
        raise ValueError("Embedding snapshot frozen feedback times are non-finite")
    return stories, labels, vote_times


def _load_production_candidates(
    db: Database,
    config: Config,
    user_id: int,
    embeddings_file: str | Path | None = None,
    *,
    max_candidates: int | None = None,
    required_story_ids: set[int] | None = None,
) -> tuple[list[Story], np.ndarray]:
    if embeddings_file is not None:
        stories = _snapshot_candidate_stories(embeddings_file)
        return stories, _load_external_embeddings(embeddings_file, stories)

    stories = load_production_candidate_stories(
        db,
        config,
        user_id=user_id,
        exclude_feedback=False,
    )
    indices = _candidate_indices_with_feedback(
        stories,
        max_candidates=max_candidates,
        required_story_ids=required_story_ids or set(),
    )
    stories = [stories[i] for i in indices]
    hashes = {
        s.id: text_hash
        for s, text_hash in zip(stories, _embedding_text_hashes(stories), strict=True)
    }
    cached = db.get_embeddings_batch(
        [s.id for s in stories], config.embedding_model_version, hashes
    )
    embeddings = _validated_embeddings(stories, cached)
    return stories, embeddings


def _candidate_indices_with_feedback(
    candidates: list[Story],
    *,
    max_candidates: int | None,
    required_story_ids: set[int],
) -> np.ndarray:
    """Choose a deterministic capped pool while retaining evaluable feedback."""
    if max_candidates is None or len(candidates) <= max_candidates:
        return np.arange(len(candidates), dtype=int)

    required_indices = {
        index
        for index, story in enumerate(candidates)
        if story.id in required_story_ids
    }
    if len(required_indices) > max_candidates:
        raise RuntimeError(
            "--max-candidates is smaller than the valid feedback candidate set"
        )
    optional_indices = np.array(
        sorted(
            (
                index
                for index in range(len(candidates))
                if index not in required_indices
            ),
            key=lambda index: candidates[index].id,
        ),
        dtype=int,
    )
    remaining_slots = max_candidates - len(required_indices)
    if remaining_slots < len(optional_indices):
        optional_indices = np.random.default_rng(1).choice(
            optional_indices, size=remaining_slots, replace=False
        )
    return np.array(
        sorted(required_indices | {int(index) for index in optional_indices}), dtype=int
    )


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
) -> np.ndarray:
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
    from pipeline.ranking import _loocv_knn_features

    features = []
    for label in (2, 0):
        indices = np.flatnonzero(y_train == label)
        features.append(_loocv_knn_features(train_emb, train_emb[indices], indices, k))
    return features[0][0], features[1][0], features[0][1], features[1][1]


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
    if not len(cand_emb):
        return np.empty(0, dtype=np.float32)
    up_emb = train_emb[y_train == 2]
    down_emb = train_emb[y_train == 0]
    up_centroid = (
        up_emb.mean(axis=0)
        if len(up_emb)
        else np.zeros(cand_emb.shape[1], dtype=np.float32)
    )
    down_centroid = (
        down_emb.mean(axis=0)
        if len(down_emb)
        else np.zeros(cand_emb.shape[1], dtype=np.float32)
    )
    scores = cand_emb @ up_centroid - cand_emb @ down_centroid
    return ((scores - scores.min()) / (scores.max() - scores.min() + 1e-8)).astype(
        np.float32
    )


def _scores_candidate_order(fold: FoldData) -> tuple[np.ndarray, None]:
    return (-np.arange(len(fold.candidates), dtype=np.float32), None)


def _scores_gravity(
    fold: FoldData, now_ts: float | None = None
) -> tuple[np.ndarray, None]:
    now = time.time() if now_ts is None else now_ts
    ages_hours = np.array(
        [max((now - story.time) / 3600.0, 1.0) for story in fold.candidates],
        dtype=np.float64,
    )
    hn_scores = np.array(
        [max(float(story.score), 0.0) for story in fold.candidates],
        dtype=np.float64,
    )
    recency_tiebreak = np.array(
        [float(story.time) for story in fold.candidates],
        dtype=np.float64,
    )
    if len(recency_tiebreak):
        recency_tiebreak = _percentile_scores(recency_tiebreak).astype(np.float64)
    return (hn_scores / np.power(ages_hours, 1.8) + recency_tiebreak * 1e-9).astype(
        np.float32
    ), None


def _scores_centroid_up_minus_down(fold: FoldData) -> tuple[np.ndarray, None]:
    return fold.tier2_scores.copy(), None


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


def _fit_svc_up_margin(
    train_raw: np.ndarray,
    cand_raw: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    config: Config,
    emb_dim: int,
) -> np.ndarray:
    """Shared RBF-margin fit for margin3_up and its additive ablations."""
    x_train, x_cand = _fit_scale(train_raw, cand_raw, emb_dim)
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
    return scores.astype(np.float32)


def _scores_margin_3class(
    fold: FoldData,
    config: Config,
    *,
    half_life_days: float | None = None,
) -> tuple[np.ndarray, None]:
    y = fold.y_train
    weights = _balanced_weights(y)
    if half_life_days is not None:
        weights = weights * _recency_decay(
            float(fold.train_vote_times.max()), fold.train_vote_times, half_life_days
        )
    scores = _fit_svc_up_margin(
        fold.x_train_base, fold.x_cand_base, y, weights, config, fold.cand_emb.shape[1]
    )
    return scores, None


def _scores_margin3_dwell(
    fold: FoldData,
    config: Config,
    source_db: Database,
    user_id: int,
    *,
    gain: float = 1.0,
    cap_ms: int = 120_000,
) -> tuple[np.ndarray, None]:
    """margin3_up with dwell-confidence sample weights (B3 experiment).

    Each train row keeps its balanced class weight, multiplied by
    1 + gain * capped_dwell / cap. Dwell is cut at the fold training
    cutoff (max train vote time), so no post-cutoff engagement leaks
    into training; stories without dwell keep factor 1.0.
    """
    y = fold.y_train
    weights = _balanced_weights(y)
    cutoff = float(fold.train_vote_times.max())
    dwell = source_db.get_capped_dwell_by_story(user_id, cap_ms, cutoff)
    if dwell:
        factors = np.array(
            [
                1.0 + gain * min(dwell.get(int(s.id), 0.0), float(cap_ms)) / cap_ms
                for s in fold.train_stories
            ]
        )
        weights = weights * factors
    scores = _fit_svc_up_margin(
        fold.x_train_base, fold.x_cand_base, y, weights, config, fold.cand_emb.shape[1]
    )
    return scores, None


def _ablation_extra_columns(
    fold: FoldData, config: Config, *, cluster: bool, source: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Production-extra feature columns appended to the base margin3 matrix.

    Mirrors serving: positive-cluster max-similarity over up-vote embeddings
    and the 4-binary source category stack. Scaling is left to _fit_scale,
    which standardizes metadata columns exactly like serving.
    """
    train_extras: list[np.ndarray] = []
    cand_extras: list[np.ndarray] = []
    emb_dim = fold.cand_emb.shape[1]
    train_emb = fold.x_train_base[:, :emb_dim]
    if cluster:
        from pipeline.ranking import _positive_cluster_similarity

        up_emb = train_emb[fold.y_train == 2]
        k = config.model.positive_cluster_k
        train_extras.append(_positive_cluster_similarity(train_emb, up_emb, k)[:, None])
        cand_extras.append(
            _positive_cluster_similarity(fold.cand_emb, up_emb, k)[:, None]
        )
    if source:
        from pipeline.ranking import source_category_stack

        train_extras.append(
            source_category_stack([s.source for s in fold.train_stories])
        )
        cand_extras.append(source_category_stack([s.source for s in fold.candidates]))
    zeros_train = np.empty((len(train_emb), 0), dtype=np.float32)
    zeros_cand = np.empty((len(fold.cand_emb), 0), dtype=np.float32)
    train_extra = (
        np.hstack(train_extras).astype(np.float32) if train_extras else zeros_train
    )
    cand_extra = (
        np.hstack(cand_extras).astype(np.float32) if cand_extras else zeros_cand
    )
    return (
        np.hstack([fold.x_train_base, train_extra]),
        np.hstack([fold.x_cand_base, cand_extra]),
    )


def _production_tier_blend(
    margin: np.ndarray, fold: FoldData, config: Config
) -> np.ndarray:
    """Blend a margin scorer with tier1/tier2 using serving's alpha schedule.

    Approximation note: tier1 uses the fold training cutoff as `now` (serving
    uses wall time) and tier2 is the evaluator's centroid score, which matches
    serving's formula on fold embeddings rather than independent feedback
    vectors. Order effects — all NDCG cares about — are preserved.
    """
    from pipeline.ranking import _minmax01

    cutoff = (
        float(fold.train_vote_times.max())
        if len(fold.train_vote_times)
        else float(time.time())
    )
    tier1 = np.array(
        [
            s.score / max((max((cutoff - s.time) / 3600.0, 0.0) + 2.0) ** 1.8, 0.1)
            for s in fold.candidates
        ],
        dtype=np.float32,
    )
    if tier1.max() > 0:
        tier1 = tier1 / tier1.max()
    y = fold.y_train
    n_up, n_down = int((y == 2).sum()), int((y == 0).sum())
    model = config.model
    alpha_2 = float(np.clip(len(y) / model.tier2_blend_window, 0.0, 1.0))
    blend_start = min(model.min_up_for_svm, model.min_down_for_svm)
    alpha_3 = float(
        np.clip((min(n_up, n_down) - blend_start) / model.tier3_blend_window, 0.0, 1.0)
    )
    t1_weight = 1.0 - alpha_2
    t2_weight = alpha_2 * (1.0 - alpha_3)
    t3_weight = alpha_2 * alpha_3
    return np.asarray(
        t1_weight * tier1
        + t2_weight * fold.tier2_scores
        + t3_weight * _minmax01(margin),
        dtype=np.float32,
    )


def _scores_margin3_plus(
    fold: FoldData, config: Config, *, cluster: bool, source: bool, tierblend: bool
) -> tuple[np.ndarray, None]:
    train_raw, cand_raw = _ablation_extra_columns(
        fold, config, cluster=cluster, source=source
    )
    margin = _fit_svc_up_margin(
        train_raw,
        cand_raw,
        fold.y_train,
        _balanced_weights(fold.y_train),
        config,
        fold.cand_emb.shape[1],
    )
    if tierblend:
        return _production_tier_blend(margin, fold, config), None
    return margin, None


def _prepare_linear_model_inputs(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_cand = _fit_scale(
        fold.x_train_base, fold.x_cand_base, fold.cand_emb.shape[1]
    )
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


def _scores_logreg(
    fold: FoldData,
    config: Config,
    *,
    c: float | None = None,
    target: str = "up",
    embedding_only: bool = False,
    binary: bool = False,
    half_life_days: float | None = None,
) -> tuple[np.ndarray, None]:
    """Logistic-regression family for the 2026-09-25 ranking study.

    target "up" scores P(up); "up_minus_down" scores P(up) - P(down) (needs
    the 3-class fit). binary fits up vs not-up. embedding_only drops the
    kNN/length meta columns. half_life_days decays older training votes.
    """
    x_train, x_cand, y, _ = _prepare_linear_model_inputs(fold, config)
    if embedding_only:
        emb_dim = fold.cand_emb.shape[1]
        x_train, x_cand = x_train[:, :emb_dim], x_cand[:, :emb_dim]
    if binary:
        y = (y == 2).astype(int)
    weights = _balanced_weights(y)
    if half_life_days is not None:
        weights = weights * _recency_decay(
            float(fold.train_vote_times.max()), fold.train_vote_times, half_life_days
        )
    clf = LogisticRegression(
        C=config.model.svm_c if c is None else c,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    clf.fit(x_train, y, sample_weight=weights)
    probs = clf.predict_proba(x_cand)
    classes = list(clf.classes_)
    up = probs[:, classes.index(1 if binary else 2)]
    if target == "up_minus_down":
        if binary:
            raise ValueError("up_minus_down needs the 3-class fit")
        up = up - probs[:, classes.index(0)]
    return up.astype(np.float32), None


def _scores_production_up_minus_down(
    fold: FoldData, config: Config, source_db: Database | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Production features and SVM, scored softmax(up) - softmax(down)."""
    _, probs = _production_scores(fold, config, source_db)
    if probs is None:
        raise RuntimeError("Production returned no class probabilities")
    return (probs[:, 2] - probs[:, 0]).astype(np.float32), probs


def _scores_knn_up_minus_down(
    fold: FoldData, config: Config
) -> tuple[np.ndarray, None]:
    """Mean top-k similarity to upvotes minus to downvotes; no model fit."""
    if fold.train_emb is None:
        raise ValueError("kNN scoring needs training embeddings")
    sim_up, sim_down, _, _ = _candidate_similarity_features(
        fold.cand_emb, fold.train_emb, fold.y_train, config.model.knn_k
    )
    return (sim_up - sim_down).astype(np.float32), None


def _load_replay_embeddings(
    path: Path, expected_hashes: dict[int, Any]
) -> dict[int, np.ndarray]:
    """Vectors from an encode_replay_embeddings.py file, keyed by story ID.

    Every expected story must be present with the same embedding-text hash,
    so a model comparison never silently mixes in stale or missing texts.
    """
    with np.load(path, allow_pickle=False) as data:
        ids = [int(i) for i in data["story_ids"]]
        hashes = dict(zip(ids, (str(h) for h in data["text_hashes"]), strict=True))
        vectors = dict(zip(ids, data["embeddings"], strict=True))
    missing = [sid for sid in expected_hashes if sid not in vectors]
    stale = [
        sid
        for sid, digest in expected_hashes.items()
        if sid in hashes and hashes[sid] != str(digest)
    ]
    if missing or stale:
        raise ValueError(
            f"Replay embeddings do not match: {len(missing)} missing, "
            f"{len(stale)} with changed text"
        )
    return {sid: vectors[sid] for sid in expected_hashes}


def _load_concatenated_replay(
    paths: list[Path], expected_hashes: dict[int, Any]
) -> dict[int, np.ndarray]:
    """One or more replay files joined per story, each part scaled 1/sqrt(k)."""
    parts = [_load_replay_embeddings(path, expected_hashes) for path in paths]
    scale = np.float32(1.0 / np.sqrt(len(parts)))
    return {
        sid: np.concatenate([part[sid] for part in parts]).astype(np.float32) * scale
        for sid in expected_hashes
    }


def _parse_model_overrides(spec: str) -> dict[str, float | int | bool]:
    """``svm_c=2;knn_k=20`` -> ModelConfig overrides, typed by the field's default.

    Used by ``prod[...]`` variant names to hill-climb production settings.
    """
    defaults = asdict(ModelConfig())
    overrides: dict[str, float | int | bool] = {}
    for item in filter(None, (part.strip() for part in spec.split(";"))):
        key, sep, raw = item.partition("=")
        if not sep or key not in defaults:
            raise ValueError(f"Unknown ModelConfig override: {item!r}")
        default = defaults[key]
        if isinstance(default, bool):
            overrides[key] = raw.lower() in {"1", "true", "yes"}
        elif isinstance(default, int):
            overrides[key] = int(raw)
        elif isinstance(default, float) or key == "svm_gamma":
            overrides[key] = float(raw)
        else:
            raise ValueError(f"ModelConfig field {key!r} is not numeric/bool")
    return overrides


def _weighted_rank_blend(
    scores: dict[str, np.ndarray], *, lr_weight: float, knn_weight: float
) -> np.ndarray:
    """Percentile-rank blend; production gets 1 - lr_weight - knn_weight."""
    weights = {
        "production": 1.0 - lr_weight - knn_weight,
        "logreg": lr_weight,
        "knn": knn_weight,
    }
    if min(weights.values()) < 0:
        raise ValueError(f"Blend weights must be non-negative: {weights}")
    return sum(
        (w * _percentile_scores(scores[key]) for key, w in weights.items() if w),
        start=np.zeros(len(scores["production"]), dtype=np.float32),
    ).astype(np.float32)


def _rank_average(*scores: np.ndarray) -> np.ndarray:
    return np.mean([_percentile_scores(s) for s in scores], axis=0).astype(np.float32)


def _metrics(
    scores: np.ndarray,
    fold: FoldData,
    config: Config,
    probs: np.ndarray | None = None,
    *,
    source_db: Database | None = None,
    calibration_available: bool = False,
) -> dict:
    """Recovery of known feedback; unjudged cards have unknown relevance."""
    if scores.shape != (len(fold.candidates),) or not np.isfinite(scores).all():
        raise ValueError("Scorer returned invalid scores")
    order = np.argsort(-scores, kind="stable")
    judged = {
        s.id: int(a) for s, a in zip(fold.test_stories, fold.test_actions, strict=True)
    }
    candidate_ids = {s.id for s in fold.candidates}

    def compute(ids: list[int], eligible: set[int]) -> dict:
        positives = {
            sid for sid, label in judged.items() if label == 2 and sid in eligible
        }
        ranks = [i for i, sid in enumerate(ids) if sid in positives]
        n_up = len(positives)
        result: dict[str, float | int | None] = {
            "returned_cards": len(ids),
            "eligible_positives": n_up,
            "not_returned_eligible_positives": n_up - len(ranks),
            "excluded_positives": sum(a == 2 for a in judged.values()) - n_up,
            "judged_cards": sum(sid in judged for sid in ids),
            "judged_coverage": sum(sid in judged for sid in ids) / len(ids)
            if ids
            else None,
            "known_neutral_cards": sum(judged.get(sid) == 1 for sid in ids),
            "known_downvote_cards": sum(judged.get(sid) == 0 for sid in ids),
            "map": sum((i + 1) / (p + 1) for i, p in enumerate(ranks)) / n_up
            if n_up
            else None,
            "median_rank": float(np.median(ranks)) if ranks else None,
            "p25_rank": float(np.percentile(ranks, 25)) if ranks else None,
            "p75_rank": float(np.percentile(ranks, 75)) if ranks else None,
            "brier_up": None,
        }
        # Prevalence-free pairwise ranking quality over every judged card:
        # the share of (upvote, other) pairs ordered upvote-first. 0.5 is
        # random. Unlike NDCG@k it uses the whole list, so it is far less
        # noisy on ~700-card folds.
        for other_name, others in (("rest", (0, 1)), ("down", (0,))):
            ups_seen = pairs = n_other = 0
            for sid in ids:
                label = judged.get(sid)
                if sid in positives:
                    ups_seen += 1
                elif label in others:
                    pairs += ups_seen
                    n_other += 1
            result[f"auc_up_vs_{other_name}"] = (
                pairs / (n_up * n_other) if n_up and n_other else None
            )
        for k in (10, 12, 40, 100, 200):
            ideal = sum(1 / math.log2(i + 2) for i in range(min(n_up, k)))
            result[f"random_expected_ndcg_at_{k}"] = (
                (n_up / len(eligible))
                * sum(1 / math.log2(i + 2) for i in range(min(k, len(ids))))
                / ideal
                if ideal and eligible
                else None
            )
            result[f"ndcg_at_{k}"] = (
                sum(1 / math.log2(p + 2) for p in ranks if p < k) / ideal
                if ideal
                else None
            )
            result[f"up_recall_at_{k}"] = (
                sum(p < k for p in ranks) / n_up if n_up else None
            )
            result[f"hit_at_{k}"] = (
                sum(sid in judged for sid in ids[:k]) / len(judged) if judged else None
            )
        returned40 = len(ids[:40])
        result["known_upvote_fraction_at_40"] = (
            sum(p < 40 for p in ranks) / returned40 if returned40 else None
        )
        result["known_downvote_fraction_at_40"] = (
            sum(judged.get(sid) == 0 for sid in ids[:40]) / returned40
            if returned40
            else None
        )
        if calibration_available and probs is not None:
            index = {s.id: i for i, s in enumerate(fold.candidates)}
            errors = [
                (float(probs[index[sid], 2]) - float(judged[sid] == 2)) ** 2
                for sid in ids
                if sid in judged and sid in index
            ]
            result["brier_up"] = float(np.mean(errors)) if errors else None
        return result

    if probs is not None and (
        probs.shape != (len(fold.candidates), 3) or not np.isfinite(probs).all()
    ):
        raise ValueError("Invalid scorer probability matrix")
    raw_ids = [fold.candidates[i].id for i in order]
    output = {"raw": compute(raw_ids, candidate_ids)}
    if config.model.enable_mmr:
        ranked = [
            RankedStory(
                story=fold.candidates[i], score=float(scores[i]), best_match_title=""
            )
            for i in order
        ]
        top = mmr_filter(
            ranked,
            dict(zip([s.id for s in fold.candidates], fold.cand_emb, strict=True)),
            threshold=config.model.diversity_threshold,
            limit=config.count,
        )
        output["mmr"] = compute([r.story.id for r in top], candidate_ids)
    if fold.cand_emb.shape[1] != 384:
        # Deck assembly runs production dedup, which is 384-d only; replay
        # embeddings from other models are compared on the raw ranking.
        return output
    deck = _recommended(scores, fold, config, probs, source_db)
    from pipeline.config import is_hn_source

    cutoff = int(time.time()) - 30 * 86400
    for age in ("recent", "archive"):
        for source in ("mixed", "hn", "non-hn"):
            key = f"{age}_{source}"
            eligible = {
                s.id
                for s in fold.candidates + [r.story for r in deck]
                if (s.time >= cutoff) == (age == "recent")
                and (source == "mixed" or is_hn_source(s.source) == (source == "hn"))
            }
            output[f"recommended_{key}"] = compute(
                [r.story.id for r in deck if key in r.combo_keys.split()], eligible
            )
    return output


def _make_fold(
    candidates: list[Story],
    cand_emb: np.ndarray,
    fb_stories: list[Story],
    fb_to_cand: np.ndarray,
    fb_vote_times: np.ndarray,
    y: np.ndarray,
    valid_positions: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    config: Config,
    *,
    feedback_embeddings: np.ndarray | None = None,
    needs_experimental: bool = True,
    judged_only: bool = False,
) -> FoldData:
    train_story_indices = valid_positions[train_pos]
    test_story_indices = valid_positions[test_pos]

    def group_key(story: Story) -> str:
        return str(normalize_url(story.url) or f"story:{story.id}")

    # Keep the first held-out vote per article; never evaluate a cross-post of
    # an article already present in training. Identity matches production URL
    # normalization, including tracking-parameter removal.
    train_groups = {group_key(fb_stories[idx]) for idx in train_story_indices}
    seen_groups = set(train_groups)
    keep_test = []
    for position in sorted(
        test_pos, key=lambda p: (fb_vote_times[p], fb_stories[valid_positions[p]].id)
    ):
        key = group_key(fb_stories[valid_positions[position]])
        if key not in seen_groups:
            keep_test.append(position)
            seen_groups.add(key)
    test_pos = np.asarray(keep_test, dtype=int)
    test_story_indices = valid_positions[test_pos]
    test_ids = {fb_stories[idx].id for idx in test_story_indices}
    # Deduplicate candidate articles too, preferring the held-out story ID so
    # an equivalent unjudged cross-post cannot hide a judged item.
    retained: set[int] = set()
    seen_candidates = set(train_groups)
    for item in sorted(candidates, key=lambda s: (s.id not in test_ids, s.id)):
        key = group_key(item)
        if key not in seen_candidates and (not judged_only or item.id in test_ids):
            retained.add(item.id)
            seen_candidates.add(key)
    cand_mask = np.array([s.id in retained for s in candidates], dtype=bool)
    fold_candidates = [s for i, s in enumerate(candidates) if cand_mask[i]]
    fold_cand_emb = cand_emb[cand_mask]

    train_emb = (
        feedback_embeddings[train_story_indices]
        if feedback_embeddings is not None
        else cand_emb[fb_to_cand[train_story_indices]]
    )
    y_train = y[train_pos]
    train_stories = [fb_stories[idx] for idx in train_story_indices]
    test_stories = [fb_stories[idx] for idx in test_story_indices]
    test_actions = y[test_pos]
    train_vote_times = fb_vote_times[train_pos]

    if not needs_experimental:
        empty = np.empty((0, 0), dtype=np.float32)
        return FoldData(
            candidates=fold_candidates,
            cand_emb=fold_cand_emb,
            train_stories=train_stories,
            test_stories=test_stories,
            test_actions=test_actions,
            train_vote_times=train_vote_times,
            x_train_base=empty,
            x_cand_base=empty,
            y_train=y_train,
            train_emb=train_emb,
            tier2_scores=_tier2_scores(fold_cand_emb, train_emb, y_train),
        )

    train_sim_up, train_sim_down, train_closest_up, train_closest_down = (
        _loocv_similarity_features(train_emb, y_train, config.model.knn_k)
    )
    cand_sim_up, cand_sim_down, cand_closest_up, cand_closest_down = (
        _candidate_similarity_features(
            fold_cand_emb, train_emb, y_train, config.model.knn_k
        )
    )
    return FoldData(
        train_emb=train_emb,
        candidates=fold_candidates,
        cand_emb=fold_cand_emb,
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
        ),
        x_cand_base=_feature_matrix(
            fold_cand_emb,
            fold_candidates,
            cand_sim_up,
            cand_sim_down,
            cand_closest_up,
            cand_closest_down,
        ),
        y_train=y_train,
        tier2_scores=_tier2_scores(fold_cand_emb, train_emb, y_train),
    )


def _temporal_splits(
    y: np.ndarray,
    vote_times: np.ndarray,
    *,
    folds: int,
    initial_train_frac: float = 0.5,
) -> list[FoldSplit]:
    if folds < 1:
        raise ValueError("--folds must be at least 1")
    if y.ndim != 1 or vote_times.shape != y.shape:
        raise ValueError("Labels and timestamps must be aligned vectors")
    if len(y) < 2:
        raise RuntimeError(
            "Need at least 2 valid feedback rows for temporal evaluation"
        )
    if not np.isfinite(vote_times).all() or not 0 < initial_train_frac < 1:
        raise ValueError("Finite timestamps and 0 < initial_train_frac < 1 required")
    groups = np.unique(vote_times)
    if len(groups) < folds + 1:
        raise ValueError("Need at least folds + 1 distinct timestamp groups")
    initial = max(1, min(int(len(groups) * initial_train_frac), len(groups) - folds))
    blocks = np.array_split(groups[initial:], folds)
    splits = [
        FoldSplit(
            i,
            np.flatnonzero(vote_times < block[0]),
            np.flatnonzero(np.isin(vote_times, block)),
        )
        for i, block in enumerate(blocks, 1)
    ]
    return splits


def _variant_requires_all_labels(name: str) -> bool:
    all_label_prefixes = ("margin3", "linear_svc", "logreg", "ensemble", "prodlr")
    return name.startswith(all_label_prefixes)


def _required_labels_for_variants(variant_names: list[str]) -> set[int]:
    labels = (
        set()
        if all(
            name == "production" or name.startswith("svm_") for name in variant_names
        )
        else {0, 2}
    )
    if any(_variant_requires_all_labels(name) for name in variant_names):
        labels.add(1)
    return labels


def _validate_splits(
    splits: list[FoldSplit],
    y: np.ndarray,
    *,
    split_mode: str,
    required_train_labels: set[int],
) -> None:
    if not splits:
        raise RuntimeError(f"No folds produced for {split_mode} split")
    for split in splits:
        train_labels = set(int(label) for label in y[split.train_pos])
        missing = sorted(required_train_labels - train_labels)
        if missing:
            raise RuntimeError(
                f"{split_mode} fold {split.fold_no} training labels are missing "
                f"{missing}; need labels {sorted(required_train_labels)} for requested "
                "variants. Use --split stratified for shuffled diagnostics, request "
                "variants with looser label requirements, or add more feedback history."
            )


def _aggregate_results(
    rows_by_name: dict[str, list[dict]],
) -> dict[str, dict[str, Any]]:
    aggregated = {}
    for name, rows in rows_by_name.items():
        if not rows:
            raise ValueError(f"No completed folds for {name}")
        result: dict[str, Any] = {
            "mean": {},
            "std": {},
            "defined_folds": {},
            "per_fold": rows,
        }
        for side in rows[0]:
            result["mean"][side] = {}
            result["std"][side] = {}
            result["defined_folds"][side] = {}
            for key in rows[0][side]:
                values = [row[side][key] for row in rows if row[side][key] is not None]
                result["mean"][side][key] = float(np.mean(values)) if values else None
                result["std"][side][key] = float(np.std(values)) if values else None
                result["defined_folds"][side][key] = len(values)
        aggregated[name] = result
    return aggregated


def main(argv: list[str] | None = None) -> None:
    with ExitStack() as stack:
        _main(argv, stack)


def _main(argv: list[str] | None, stack: ExitStack) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file; defaults to config.toml")
    parser.add_argument("--output", default="eval_ranker_variants.json")
    parser.add_argument(
        "--embeddings-file",
        help=(
            "Optional .npz snapshot with story_ids, text_hashes, and embeddings. "
            "Used by the embedding-model bakeoff without writing to the live DB."
        ),
    )
    parser.add_argument(
        "--replay-embeddings",
        type=Path,
        action="append",
        help=(
            "heldout-feedback only: .npz from scripts/encode_replay_embeddings.py "
            "replacing stored embeddings for every feedback story (text-hash "
            "checked), to compare embedding models. Repeat to concatenate "
            "several models (each scaled by 1/sqrt(k), so vectors stay unit)."
        ),
    )
    parser.add_argument(
        "--embedding-label",
        help="Human-readable embedding specification recorded in the JSON report.",
    )
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--leak-seeds",
        type=int,
        default=5,
        help="Number of independent label permutations with --leak-check",
    )
    parser.add_argument(
        "--candidate-pool",
        choices=("current", "heldout-feedback"),
        default="current",
        help="Current production pool or judged-only retrospective replay of each held-out block; not comparable metrics.",
    )
    parser.add_argument(
        "--split",
        choices=("temporal", "stratified"),
        default="temporal",
        help=(
            "Evaluation split. temporal uses expanding-window chronological folds; "
            "stratified is rejected because its historical meaning cannot be preserved."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        help="Override the candidate story age window for evaluation only.",
    )
    parser.add_argument("--svm-c", type=float)
    parser.add_argument("--svm-gamma", type=float)
    parser.add_argument(
        "--variants",
        help="Comma-separated variant names to run. Defaults to production; production is always included.",
    )
    parser.add_argument(
        "--max-candidates",
        "--candidate-cap",
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
            "compare shuffled scores to the pool's relevance prevalence, "
            "not a fixed threshold (judged-only replay has many positives). "
            "Elevated shuffled scores require investigation or more seeds."
        ),
    )
    parser.add_argument(
        "--sweep-svm",
        action="store_true",
        help="Compare the named C/gamma grid against production",
    )
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="Evaluate the reserved latest 20 percent of timestamp groups",
    )
    parser.add_argument("--now", type=float, help="Frozen evaluation Unix timestamp")
    parser.add_argument(
        "--candidate-cap-seed",
        type=int,
        default=0,
        help="Legacy no-op; only zero is accepted",
    )
    args = parser.parse_args(argv)
    if args.candidate_cap_seed != 0:
        parser.error("--candidate-cap-seed was a no-op; only 0 is supported")
    for option in ("svm_c", "svm_gamma"):
        value = getattr(args, option)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{option.replace('_', '-')} must be finite and positive")
    for option in (
        "folds",
        "leak_seeds",
        "max_candidates",
        "max_feedback_per_class",
        "window_days",
    ):
        value = getattr(args, option)
        if value is not None and value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.split != "temporal":
        parser.error(
            "--split stratified is retired: chronological evaluation is required"
        )
    config = Config.load(args.config or "config.toml")
    if args.embeddings_file and (
        args.config is not None or args.window_days is not None
    ):
        parser.error(
            "Frozen embedding snapshots cannot be combined with --config or --window-days"
        )
    database_path = config.db_path
    frozen_now = None
    if args.embeddings_file:
        config, frozen_now, database_path = _snapshot_context(args.embeddings_file)
    now = (
        args.now
        if args.now is not None
        else frozen_now
        if frozen_now is not None
        else time.time()
    )
    if not math.isfinite(now):
        parser.error("--now must be finite")
    stack.enter_context(patch("time.time", return_value=now))
    production_config = config
    if args.svm_c is not None or args.svm_gamma is not None:
        model = replace(
            config.model,
            svm_c=args.svm_c if args.svm_c is not None else config.model.svm_c,
            svm_gamma=(
                args.svm_gamma if args.svm_gamma is not None else config.model.svm_gamma
            ),
        )
        config = replace(config, model=ModelConfig(**model.__dict__))
    db, snapshot_hash = stack.enter_context(frozen_database(database_path))
    user_id = args.user_id
    if args.embeddings_file:
        with np.load(args.embeddings_file, allow_pickle=False) as data:
            snapshot_user_id = int(data["user_id"].item())
        if user_id is not None and user_id != snapshot_user_id:
            parser.error("--user-id does not match the frozen feedback owner")
        user_id = snapshot_user_id
    user = (
        db.get_user_by_id(user_id)
        if user_id is not None
        else db.get_user_by_token("default")
    )
    if user is None:
        raise RuntimeError("Missing evaluation user")

    requested = (
        [name.strip() for name in args.variants.split(",") if name.strip()]
        if args.variants
        else ["production"]
    )
    if "production" not in requested:
        requested.insert(0, "production")

    window_days = args.window_days if args.window_days is not None else config.days
    eval_config = replace(config, days=window_days)
    if args.embeddings_file:
        fb_stories, all_y, fb_vote_times = _snapshot_feedback(args.embeddings_file)
    else:
        fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training(
            user_id=user.id
        )
        all_y = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    valid_mask = np.ones(len(fb_stories), dtype=bool)

    groups = np.unique(fb_vote_times)
    if len(groups) < 3:
        raise ValueError(
            "Need at least three timestamp groups before reserving confirmation"
        )
    confirmation_start = groups[max(1, int(math.floor(len(groups) * 0.8)))]
    if not np.isfinite(fb_vote_times).all() or not set(all_y.tolist()) <= {0, 1, 2}:
        raise ValueError("Invalid feedback labels or timestamps")

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
        valid_mask = np.ones(len(fb_stories), dtype=bool)

    if args.embeddings_file and args.max_candidates is not None:
        parser.error(
            "--max-candidates cannot change a frozen embedding snapshot; cap it at bakeoff creation"
        )
    if args.candidate_pool == "heldout-feedback":
        if args.embeddings_file or args.max_candidates:
            parser.error(
                "heldout-feedback cannot use embedding snapshots or candidate caps"
            )
        candidates = list(fb_stories)
        hashes = dict(
            zip(
                [s.id for s in candidates],
                _embedding_text_hashes(candidates),
                strict=True,
            )
        )
        cand_emb = _validated_embeddings(
            candidates,
            _load_concatenated_replay(args.replay_embeddings, hashes)
            if args.replay_embeddings
            else db.get_embeddings_batch(
                [s.id for s in candidates], config.embedding_model_version, hashes
            ),
        )
    else:
        if args.replay_embeddings:
            parser.error("--replay-embeddings needs --candidate-pool heldout-feedback")
        candidates, cand_emb = _load_production_candidates(
            db,
            eval_config,
            user.id,
            embeddings_file=args.embeddings_file,
            max_candidates=args.max_candidates,
            required_story_ids={s.id for s in fb_stories},
        )
    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories])

    if args.embeddings_file:
        with np.load(args.embeddings_file, allow_pickle=False) as data:
            if "feedback_embeddings" not in data.files:
                raise ValueError(
                    "Snapshot requires independent feedback_embeddings; regenerate bakeoff"
                )
            original_stories, _, _ = _snapshot_feedback(args.embeddings_file)
            vectors = dict(
                zip(
                    [s.id for s in original_stories],
                    data["feedback_embeddings"],
                    strict=True,
                )
            )
            feedback_embeddings = _validated_embeddings(fb_stories, vectors)
    else:
        hashes = dict(
            zip(
                [s.id for s in fb_stories],
                _embedding_text_hashes(fb_stories),
                strict=True,
            )
        )
        feedback_embeddings = _validated_embeddings(
            fb_stories,
            _load_concatenated_replay(args.replay_embeddings, hashes)
            if args.replay_embeddings
            else db.get_embeddings_batch(
                [s.id for s in fb_stories], config.embedding_model_version, hashes
            ),
        )
    if not args.confirmation:
        valid_mask &= fb_vote_times < confirmation_start

    valid_positions = np.where(valid_mask)[0]
    y = all_y[valid_mask]
    fb_vote_times = fb_vote_times[valid_mask]
    label_names = {0: "down", 1: "neutral", 2: "up"}
    candidate_recall = {}
    for label in (0, 1, 2):
        total = int((all_y == label).sum())
        present = int(((all_y == label) & (fb_to_cand >= 0)).sum())
        candidate_recall[label_names[label]] = {
            "present": present,
            "total": total,
            "recall": present / total if total else 0.0,
        }

    print(
        f"user_id={user.id} candidates={len(candidates)} "
        f"valid_feedback={len(y)} labels={Counter(y)}"
    )
    print(f"candidate_recall={candidate_recall}")

    variants = {
        "production": lambda fold: _production_scores(fold, production_config, db),
        "deduplicated": lambda fold: _production_scores(
            fold,
            replace(
                production_config,
                model=replace(
                    production_config.model, deduplicate_training_feedback=True
                ),
            ),
            db,
        ),
        "deduplicated_publication": lambda fold: _production_scores(
            fold,
            replace(
                production_config,
                model=replace(
                    production_config.model,
                    deduplicate_training_feedback=True,
                    publication_affinity_enabled=True,
                ),
            ),
            db,
        ),
        "publication_affinity": lambda fold: _production_scores(
            fold,
            replace(
                production_config,
                model=replace(
                    production_config.model, publication_affinity_enabled=True
                ),
            ),
            db,
        ),
        "margin3_up": lambda fold: _scores_margin_3class(fold, config),
        "linear_svc_up": lambda fold: _scores_linear_svc_up(fold, config),
        "logreg_up": lambda fold: _scores_logreg_up(fold, config),
        "margin3_up_recency30d": lambda fold: _scores_margin_3class(
            fold, config, half_life_days=30.0
        ),
        "margin3_dwell": lambda fold: _scores_margin3_dwell(fold, config, db, user.id),
        "margin3_plus_cluster": lambda fold: _scores_margin3_plus(
            fold, config, cluster=True, source=False, tierblend=False
        ),
        "margin3_plus_source": lambda fold: _scores_margin3_plus(
            fold, config, cluster=False, source=True, tierblend=False
        ),
        "margin3_plus_tierblend": lambda fold: _scores_margin3_plus(
            fold, config, cluster=False, source=False, tierblend=True
        ),
        "margin3_plus_all": lambda fold: _scores_margin3_plus(
            fold, config, cluster=True, source=True, tierblend=True
        ),
        "tier2_centroid": lambda fold: (fold.tier2_scores.copy(), None),
        # 2026-09-25 ranking study (see FINDINGS.md).
        "logreg_up_minus_down": lambda fold: _scores_logreg(
            fold, config, target="up_minus_down"
        ),
        "logreg_binary_up": lambda fold: _scores_logreg(fold, config, binary=True),
        "logreg_emb_only": lambda fold: _scores_logreg(
            fold, config, embedding_only=True
        ),
        "logreg_up_recency60d": lambda fold: _scores_logreg(
            fold, config, half_life_days=60.0
        ),
        "production_up_minus_down": lambda fold: _scores_production_up_minus_down(
            fold, production_config, db
        ),
        "knn_up_minus_down": lambda fold: _scores_knn_up_minus_down(fold, config),
        "ensemble_production_logreg": lambda fold: (
            _rank_average(
                _production_scores(fold, production_config, db)[0],
                _scores_logreg(fold, config)[0],
            ),
            None,
        ),
    }
    for c in (0.01, 0.03, 0.3, 1.0, 3.0):
        variants[f"logreg_up_c{c}"] = lambda fold, c=c: _scores_logreg(
            fold, config, c=c
        )
    if args.svm_c is not None or args.svm_gamma is not None:
        variants["svm_override"] = lambda fold: _production_scores(fold, config, db)
        requested.append("svm_override")
    if args.sweep_svm:
        for c in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            for gamma in (0.01, 0.02, 0.03, 0.05, 0.1):
                name = f"svm_c{c}_gamma{gamma}"
                tuned = replace(
                    production_config,
                    model=replace(production_config.model, svm_c=c, svm_gamma=gamma),
                )
                variants[name] = lambda fold, tuned=tuned: _production_scores(
                    fold, tuned, db
                )
                requested.append(name)
    # Hill-climbing names: prod[spec] = production with ModelConfig overrides;
    # produd[spec] scores it softmax(up) - softmax(down); prodlr[spec] rank-
    # averages it with logreg_up_minus_down.
    for name in requested:
        prefix, bracket, rest = name.partition("[")
        if not bracket or not rest.endswith("]"):
            continue
        spec = rest[:-1].split(";")
        blend = {
            key: float(value)
            for key, _, value in (item.partition("=") for item in spec)
            if key in ("lr_weight", "knn_weight")
        }
        model_spec = ";".join(
            item
            for item in spec
            if item.partition("=")[0] not in ("lr_weight", "knn_weight")
        )
        tuned = replace(
            production_config,
            model=replace(
                production_config.model, **_parse_model_overrides(model_spec)
            ),
        )
        if prefix == "prod":
            variants[name] = lambda fold, tuned=tuned: _production_scores(
                fold, tuned, db
            )
        elif prefix == "produd":
            variants[name] = lambda fold, tuned=tuned: _scores_production_up_minus_down(
                fold, tuned, db
            )
        elif prefix == "prodlr":
            variants[name] = lambda fold, tuned=tuned, blend=blend: (
                _weighted_rank_blend(
                    {
                        "production": _production_scores(fold, tuned, db)[0],
                        "logreg": _scores_logreg(fold, config, target="up_minus_down")[
                            0
                        ],
                        "knn": _scores_knn_up_minus_down(fold, config)[0],
                    },
                    lr_weight=blend.get("lr_weight", 0.5),
                    knn_weight=blend.get("knn_weight", 0.0),
                ),
                None,
            )
    if requested:
        missing = sorted(set(requested) - set(variants))
        if missing:
            raise ValueError(f"Unknown variants: {', '.join(missing)}")
        variants = {name: variants[name] for name in requested}

    required_train_labels = _required_labels_for_variants(list(variants))
    missing_global_labels = sorted(
        required_train_labels - set(int(label) for label in y)
    )
    if missing_global_labels:
        raise RuntimeError(
            f"Need labels {sorted(required_train_labels)} for requested variants; "
            f"valid feedback is missing {missing_global_labels}"
        )

    split_label = "temporal-expanding"
    splits = _temporal_splits(y, fb_vote_times, folds=args.folds)
    if args.confirmation:
        reserved = np.unique(fb_vote_times[fb_vote_times >= confirmation_start])
        if len(reserved) < args.folds:
            raise ValueError(
                "Confirmation has fewer timestamp groups than requested folds"
            )
        splits = [
            FoldSplit(
                i,
                np.flatnonzero(fb_vote_times < block[0]),
                np.flatnonzero(np.isin(fb_vote_times, block)),
            )
            for i, block in enumerate(np.array_split(reserved, args.folds), 1)
        ]
    _validate_splits(
        splits,
        y,
        split_mode=split_label,
        required_train_labels=required_train_labels,
    )

    baselines = {
        "random": lambda fold: (
            np.random.default_rng(0).random(len(fold.candidates)),
            None,
        ),
        "candidate_order": lambda fold: _scores_candidate_order(fold),
        "gravity": lambda fold: _scores_gravity(fold),
        "centroid_up_minus_down": lambda fold: _scores_centroid_up_minus_down(fold),
    }

    fold_audit: list[dict[str, int]] = []

    def _run_scorers(
        scorers: dict[str, Any],
        y_label: np.ndarray,
        label: str = "",
    ) -> dict[str, list[dict]]:
        """Run scorers on a given label vector.

        label is a prefix prepended to per-fold progress lines (used by
        the leak check pass to distinguish its output from the main run).
        """
        results: dict[str, list[dict]] = {name: [] for name in scorers}
        for split in splits:
            fold = _make_fold(
                candidates,
                cand_emb,
                fb_stories,
                fb_to_cand,
                fb_vote_times,
                y_label,
                valid_positions,
                split.train_pos,
                split.test_pos,
                config,
                feedback_embeddings=feedback_embeddings,
                judged_only=args.candidate_pool == "heldout-feedback",
                needs_experimental=any(
                    name != "production" and not name.startswith("svm_")
                    for name in variants
                ),
            )
            if not label:
                fold_audit.append(
                    {
                        "fold": split.fold_no,
                        "test_rows_before_group_isolation": len(split.test_pos),
                        "test_rows_after_group_isolation": len(fold.test_stories),
                        "candidate_rows_after_group_isolation": len(fold.candidates),
                    }
                )
            with _fold_database(fold, config, db) as fold_db:
                fold = replace(fold, runtime_db=fold_db)
                with _reuse_preprocessing():
                    for name, scorer in scorers.items():
                        scores, probs = (
                            scorer(fold) if fold.candidates else (np.empty(0), None)
                        )
                        results[name].append(
                            _metrics(
                                scores,
                                fold,
                                config,
                                probs,
                                source_db=db,
                                calibration_available=name == "logreg_up",
                            )
                        )
            print(f"{label}fold {split.fold_no}/{len(splits)} done")
        return results

    combined = _run_scorers(variants | baselines, y)
    results = {name: combined[name] for name in variants}
    baseline_results = {name: combined[name] for name in baselines}

    report: dict[str, Any] = {
        "schema_version": 4,
        "interpretation": "Retrospective recovery of held-out feedback using current stored content; not historical exposure or causal reading-quality evaluation. heldout-feedback is judged-only discrimination, with higher positive density: do not compare its metrics to current-pool results. URL groups are isolated from training and deduplicated in test/candidates; semantic duplicates may remain. Feedback updated_at approximates chronology. Confirmation is reusable historical evidence.",
        "relevance": {"up": 1, "neutral": 0, "down": 0},
        "variation": "std is fold variation, not uncertainty of a causal estimate",
        "config": {
            "split": split_label,
            "split_mode": args.split,
            "temporal_initial_train_frac": 0.5 if args.split == "temporal" else None,
            "candidate_loader": args.candidate_pool,
            "window_days": window_days,
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
            "db_sha256": snapshot_hash,
            "now": now,
            "embedding_snapshot_sha256": _db_sha256(args.embeddings_file)
            if args.embeddings_file
            else None,
            "n_feedback_input": len(fb_stories),
            "folds": [
                {
                    "fold": split.fold_no,
                    "training_rows": len(split.train_pos),
                    "test_rows": len(split.test_pos),
                    "training_cutoff": float(fb_vote_times[split.train_pos].max()),
                    "test_start": float(fb_vote_times[split.test_pos].min()),
                    "test_end": float(fb_vote_times[split.test_pos].max()),
                    "training_story_ids": [
                        fb_stories[valid_positions[i]].id for i in split.train_pos
                    ],
                    "test_story_ids": [
                        fb_stories[valid_positions[i]].id for i in split.test_pos
                    ],
                }
                for split in splits
            ],
            "frozen_config": asdict(config),
            "confirmation": args.confirmation,
            "confirmation_start": float(confirmation_start),
            "sampling": {
                "max_candidates": args.max_candidates,
                "max_feedback_per_class": args.max_feedback_per_class,
                "candidate_seed": 1,
                "feedback_seed": 0,
            },
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ),
            "dirty_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"])
            ).hexdigest(),
            "embedding_label": args.embedding_label,
            "replay_embeddings": [str(path) for path in args.replay_embeddings]
            if args.replay_embeddings
            else None,
            "embeddings_file": args.embeddings_file,
        },
        "variants": {},
        "baselines": {},
    }
    if args.split == "stratified":
        del report["config"]["temporal_initial_train_frac"]
    report["variants"] = _aggregate_results(results)
    report["baselines"] = _aggregate_results(baseline_results)
    report["group_isolation"] = fold_audit
    report["coverage_warnings"] = {
        name: {
            lane: [
                i + 1
                for i, row in enumerate(rows)
                if row[lane]["eligible_positives"] < 10
                or row[lane]["judged_cards"] < 20
            ]
            for lane in rows[0]
        }
        for name, rows in results.items()
    }
    # Listed folds lack even ten positives/twenty judged cards; absence of a
    # warning is not a statistical power guarantee. Never treat null as zero.

    if args.leak_check:
        permutations = []
        for seed in range(args.leak_seeds):
            print(f"\n=== Leak check: shuffling y (n={len(y)}, seed={seed}) ===")
            y_shuffled = np.random.default_rng(seed).permutation(y)
            _validate_splits(
                splits,
                y_shuffled,
                split_mode=f"{split_label} leak-check",
                required_train_labels=required_train_labels,
            )
            leak_combined = _run_scorers(
                variants | baselines, y_shuffled, label=f"[leak-check seed={seed}] "
            )
            permutations.append(
                {
                    "seed": seed,
                    "variants": _aggregate_results(
                        {name: leak_combined[name] for name in variants}
                    ),
                    "baselines": _aggregate_results(
                        {name: leak_combined[name] for name in baselines}
                    ),
                }
            )
        report["leak_check"] = {"permutations": permutations}

    report["paired_differences_against_production"] = _paired_differences(combined)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"wrote {args.output}")
    for name, data in report["variants"].items():
        print(name, json.dumps(data["mean"]["raw"]))


if __name__ == "__main__":
    main()
