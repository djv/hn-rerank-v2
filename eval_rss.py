"""RSS-only ranking evaluation. OFFLINE-ONLY: reads hn_rewrite.db exclusively.

LOOCV on non-HN feedback for user 1. Same code path as eval.py but metrics
filtered to non-HN test items. Reports per-source breakdown.

Writes eval_report_rss.json (not committed; separate from eval_report.json).
"""

from typing import Any

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from database import Database, Story
from legacy_features import _augment_features
from pipeline import (
    Config,
    RankedStory,
    _knn_similarity,
    is_hn_source,
    mmr_filter,
)

MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|256"
REPORT_PATH = Path(__file__).parent / "eval_report_rss.json"


def _db_sha256(db_path: str) -> str:
    return hashlib.sha256(Path(db_path).read_bytes()).hexdigest()[:16]


def _load_candidates(db: Database) -> tuple[list[Story], np.ndarray]:
    rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, "
        "       comment_count, discussion_url, comment_count_at_fetch, "
        "       self_text, top_comments, article_body "
        "FROM stories WHERE text_content != ''"
    )
    stories = [Database._row_to_story(row) for row in rows]
    story_hashes = {
        s.id: hashlib.sha256(s.text_content.encode("utf-8")).hexdigest()
        for s in stories
    }
    cached = db.get_embeddings_batch(
        [s.id for s in stories], MODEL_VERSION, story_hashes
    )
    embeddings = np.array(
        [cached.get(s.id, np.zeros(384, dtype=np.float32)) for s in stories],
        dtype=np.float32,
    )
    return stories, embeddings


def _compute_metrics(
    rank_map: dict[int, int],
    test_stories: list[Story],
    test_actions: np.ndarray,
    test_rel: np.ndarray,
    all_test_rels: list[float],
    brier_up: float = 0.0,
) -> dict:
    rel_by_pos = {}
    for i, ts in enumerate(test_stories):
        if ts.id in rank_map:
            rel_by_pos[rank_map[ts.id]] = test_rel[i]

    def _ndcg(rel_by_pos: dict[int, float], all_rels: list[float], k: int) -> float:
        dcg = sum(r / math.log2(p + 2) for p, r in rel_by_pos.items() if p < k)
        ideal = sorted(all_rels, reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    R = sum(1 for a in test_actions if a == 2)
    upvoted_positions = sorted(
        rank_map[ts.id]
        for i, ts in enumerate(test_stories)
        if test_actions[i] == 2 and ts.id in rank_map
    )
    if R > 0 and upvoted_positions:
        ap = (
            sum(
                (rank_idx + 1) / (pos + 1)
                for rank_idx, pos in enumerate(upvoted_positions)
            )
            / R
        )
    else:
        ap = 0.0

    upvoted_rrs = [
        1.0 / (rank_map[ts.id] + 1)
        for i, ts in enumerate(test_stories)
        if test_actions[i] == 2 and ts.id in rank_map
    ]
    mrr = float(np.mean(upvoted_rrs)) if upvoted_rrs else 0.0

    return {
        "ndcg_at_100": _ndcg(rel_by_pos, all_test_rels, 100),
        "ndcg_at_200": _ndcg(rel_by_pos, all_test_rels, 200),
        "ndcg_at_500": _ndcg(rel_by_pos, all_test_rels, 500),
        "hit_at_100": sum(1 for p in rel_by_pos if p < 100) / max(len(test_stories), 1),
        "hit_at_200": sum(1 for p in rel_by_pos if p < 200) / max(len(test_stories), 1),
        "hit_at_500": sum(1 for p in rel_by_pos if p < 500) / max(len(test_stories), 1),
        "map": ap,
        "mrr": mrr,
        "brier_up": brier_up,
    }


def _evaluate_fold(
    probs: np.ndarray,
    candidates: list[Story],
    cand_emb: np.ndarray,
    test_stories: list[Story],
    test_actions: np.ndarray,
    cand_scores: np.ndarray,
    formula: str,
    neutral_weight: float = 0.0,
    mmr_threshold: float = 0.50,
    mmr_limit: int = 40,
    cand_sim_up: np.ndarray | None = None,
    cand_sim_down: np.ndarray | None = None,
    cand_closest_up: np.ndarray | None = None,
    cand_closest_down: np.ndarray | None = None,
) -> dict:
    if formula == "current":
        scores = probs[:, 2] + neutral_weight * probs[:, 1]
    elif formula == "up_only":
        scores = probs[:, 2]
    elif formula == "hn_baseline":
        scores = cand_scores.astype(np.float32)
    elif formula == "centroid_diff":
        if cand_sim_up is None or cand_sim_down is None:
            raise ValueError("centroid_diff requires cand_sim_up and cand_sim_down")
        scores = cand_sim_up - cand_sim_down
    elif formula == "knn_diff":
        if cand_sim_up is None or cand_sim_down is None:
            raise ValueError("knn_diff requires cand_sim_up and cand_sim_down")
        scores = cand_sim_up - cand_sim_down
    elif formula == "closest_diff":
        if cand_closest_up is None or cand_closest_down is None:
            raise ValueError(
                "closest_diff requires cand_closest_up and cand_closest_down"
            )
        scores = cand_closest_up - cand_closest_down
    else:
        raise ValueError(f"Unknown formula: {formula}")

    order = np.argsort(-scores)
    ranked = [
        RankedStory(story=candidates[i], score=float(scores[i]), best_match_title="")
        for i in order
    ]
    emb_map = {candidates[i].id: cand_emb[i] for i in range(len(candidates))}

    rel_map = {0: 0.0, 1: 0.2, 2: 1.0}
    test_rel = np.array([rel_map[int(a)] for a in test_actions])
    all_test_rels = test_rel.tolist()

    cand_id_to_idx = {c.id: i for i, c in enumerate(candidates)}
    test_probs_up = []
    test_binary_up = []
    for i, ts in enumerate(test_stories):
        if ts.id in cand_id_to_idx:
            test_probs_up.append(probs[cand_id_to_idx[ts.id], 2])
            test_binary_up.append(1.0 if test_actions[i] == 2 else 0.0)
    brier_up = (
        float(np.mean((np.array(test_probs_up) - np.array(test_binary_up)) ** 2))
        if test_probs_up
        else 0.0
    )

    top40 = mmr_filter(ranked, emb_map, threshold=mmr_threshold, limit=mmr_limit)
    mmr_rank_map = {rs.story.id: pos for pos, rs in enumerate(top40)}
    mmr_metrics = _compute_metrics(
        mmr_rank_map, test_stories, test_actions, test_rel, all_test_rels, brier_up
    )

    raw_ranked = ranked
    raw_rank_map = {rs.story.id: pos for pos, rs in enumerate(raw_ranked)}
    raw_metrics = _compute_metrics(
        raw_rank_map, test_stories, test_actions, test_rel, all_test_rels, brier_up
    )

    test_upvote_ids = {s.id for i, s in enumerate(test_stories) if test_actions[i] == 2}
    upvote_ranks = [
        pos for pos, idx in enumerate(order) if candidates[idx].id in test_upvote_ids
    ]
    median_rank = float(np.median(upvote_ranks)) if upvote_ranks else 0.0
    p25_rank = float(np.percentile(upvote_ranks, 25)) if upvote_ranks else 0.0
    p75_rank = float(np.percentile(upvote_ranks, 75)) if upvote_ranks else 0.0

    mmr_metrics["median_rank"] = median_rank
    mmr_metrics["p25_rank"] = p25_rank
    mmr_metrics["p75_rank"] = p75_rank
    raw_metrics["median_rank"] = median_rank
    raw_metrics["p25_rank"] = p25_rank
    raw_metrics["p75_rank"] = p75_rank

    return {"mmr": mmr_metrics, "raw": raw_metrics}


def main() -> None:
    config = Config.load()
    db = Database(config.db_path)
    user = db.get_user_by_token("default")
    if user is None:
        raise RuntimeError("Missing default user token")
    now = time.time()

    # Feedback
    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training(user_id=user.id)
    fb_labels = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    print(f"Total feedback: {len(fb_stories)} ({Counter(fb_labels)})")

    # Candidates
    candidates, cand_emb = _load_candidates(db)
    print(f"Candidates: {len(candidates)}")

    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories], dtype=int)
    valid = fb_to_cand >= 0
    if not valid.all():
        print(
            f"Warning: {(~valid).sum()} feedback stories missing from candidates; excluded."
        )

    rss_mask = np.array([s.source != "hn" for s in fb_stories])
    rss_valid = valid & rss_mask
    rss_indices = np.where(rss_valid)[0]
    n_rss = len(rss_indices)
    up_rss = int(((fb_labels == 2) & rss_valid).sum())
    down_rss = int(((fb_labels == 0) & rss_valid).sum())
    neutral_rss = int(((fb_labels == 1) & rss_valid).sum())

    print(
        f"Non-HN feedback: {n_rss} ({up_rss} up, {down_rss} down, {neutral_rss} neutral)"
    )
    n_cand_rss = sum(1 for s in candidates if s.source != "hn")
    print(f"Non-HN candidates: {n_cand_rss}")
    print()

    if n_rss == 0:
        print("No non-HN feedback — nothing to eval.")
        return

    # Pre-compute candidate features (shared across folds)
    cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
    cand_text_lengths = np.array([len(s.text_content) for s in candidates])
    cand_ages_arr = np.array([now - max(s.time, 1) for s in candidates])
    cand_scores_arr = np.array([s.score for s in candidates])
    cand_quality_arr = cand_scores_arr / (np.maximum(cand_ages_arr / 3600.0, 0) + 1)

    # Feedback metadata (age = vote_time - story.time)
    fb_emb = cand_emb[fb_to_cand[valid]]
    fb_scores_arr = np.array([s.score for s in fb_stories])[valid]
    fb_ages_arr = np.array(
        [float(vt) - max(s.time, 1) for vt, s in zip(fb_vote_times, fb_stories)]
    )[valid]
    fb_comment_counts_arr = np.array([s.comment_count or 0 for s in fb_stories])[valid]
    fb_text_lengths_arr = np.array([len(s.text_content) for s in fb_stories])[valid]
    fb_quality_arr = fb_scores_arr / (np.maximum(fb_ages_arr / 3600.0, 0) + 1)
    fb_stories_valid = [s for i, s in enumerate(fb_stories) if valid[i]]

    cand_scores_array = np.array([s.score for s in candidates], dtype=np.float64)
    y = fb_labels[valid]
    k = config.model.knn_k

    formulas = [
        "current",
        "up_only",
        "hn_baseline",
        "centroid_diff",
        "knn_diff",
        "closest_diff",
    ]
    results: dict[str, list[dict]] = {f: [] for f in formulas}

    # 5-fold stratified CV on RSS items
    valid_indices = np.where(valid)[0]
    rss_positions = np.where(np.isin(valid_indices, rss_indices))[0]

    rss_y = fb_labels[rss_indices]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(skf.split(np.zeros((len(rss_y), 1)), rss_y))

    for fold_idx, (_, rss_test_pos_in_fold) in enumerate(folds):
        rss_test_global = rss_indices[rss_test_pos_in_fold]
        rss_test_in_valid = rss_positions[rss_test_pos_in_fold]

        train_mask = np.ones(len(valid_indices), dtype=bool)
        train_mask[rss_test_in_valid] = False
        train_pos = np.where(train_mask)[0]

        # Build training data
        fb_train_emb = fb_emb[train_pos]
        y_train = y[train_pos]

        fb_train_scores = fb_scores_arr[train_pos]
        fb_train_ages = fb_ages_arr[train_pos]
        fb_train_comments = fb_comment_counts_arr[train_pos]
        fb_train_textlens = fb_text_lengths_arr[train_pos]
        fb_train_quality = fb_quality_arr[train_pos]
        fb_train_stories = [fb_stories_valid[idx] for idx in train_pos]
        fb_train_age_hours = fb_train_ages / 3600.0
        fb_train_safe_h = np.maximum(fb_train_age_hours, 0.1)
        fb_train_score_vel = fb_train_scores / fb_train_safe_h
        fb_train_comment_vel = fb_train_comments / fb_train_safe_h

        train_ids = {fb_stories_valid[idx].id for idx in train_pos}
        cand_mask = np.array([s.id not in train_ids for s in candidates])
        fold_candidates = [s for idx, s in enumerate(candidates) if cand_mask[idx]]
        fold_cand_emb = cand_emb[cand_mask]
        fold_cand_scores = cand_scores_arr[cand_mask]
        fold_cand_ages = cand_ages_arr[cand_mask]
        fold_cand_comments = cand_comment_counts[cand_mask]
        fold_cand_textlens = cand_text_lengths[cand_mask]
        fold_cand_quality = cand_quality_arr[cand_mask]
        fold_cand_age_hours = fold_cand_ages / 3600.0
        fold_cand_safe_h = np.maximum(fold_cand_age_hours, 0.1)
        fold_cand_score_vel = fold_cand_scores / fold_cand_safe_h
        fold_cand_comment_vel = fold_cand_comments / fold_cand_safe_h
        fold_cand_scores_array = cand_scores_array[cand_mask]

        up_mask_train = y_train == 2
        down_mask_train = y_train == 0
        fb_up_train = fb_train_emb[up_mask_train]
        fb_down_train = fb_train_emb[down_mask_train]
        n_up_train = int(up_mask_train.sum())
        n_down_train = int(down_mask_train.sum())

        cand_sim_up = _knn_similarity(fold_cand_emb, fb_up_train, k)
        cand_sim_down = _knn_similarity(fold_cand_emb, fb_down_train, k)
        cand_closest_up = (
            np.max(fold_cand_emb @ fb_up_train.T, axis=1)
            if n_up_train
            else np.zeros(len(fold_candidates))
        )
        cand_closest_down = (
            np.max(fold_cand_emb @ fb_down_train.T, axis=1)
            if n_down_train
            else np.zeros(len(fold_candidates))
        )

        # LOOCV k-NN for training
        fb_sim_up = np.zeros(len(fb_train_emb), dtype=np.float32)
        fb_sim_down = np.zeros(len(fb_train_emb), dtype=np.float32)

        if n_up_train > 0:
            up_indices = np.where(up_mask_train)[0]
            sim_up_mat = fb_train_emb @ fb_up_train.T
            if n_up_train > 1:
                for i, tp in enumerate(up_indices):
                    sim_up_mat[tp, i] = -2.0
            k_eff = min(k, n_up_train)
            for i in range(len(fb_train_emb)):
                sims = sim_up_mat[i]
                exclude = 1 if i in up_indices else 0
                n_available = max(1, n_up_train - exclude)
                k_use = min(k_eff, n_available)
                topk = np.sort(sims)[-k_use:]
                fb_sim_up[i] = topk.mean()
            sim_up_mat_clean = fb_train_emb @ fb_up_train.T
            if n_up_train > 1:
                for i, tp in enumerate(up_indices):
                    sim_up_mat_clean[tp, i] = -1.0
            fb_closest_up = np.max(sim_up_mat_clean, axis=1)
        else:
            fb_closest_up = np.zeros(len(fb_train_emb))

        if n_down_train > 0:
            down_indices = np.where(down_mask_train)[0]
            sim_down_mat = fb_train_emb @ fb_down_train.T
            if n_down_train > 1:
                for i, tp in enumerate(down_indices):
                    sim_down_mat[tp, i] = -2.0
            k_eff = min(k, n_down_train)
            for i in range(len(fb_train_emb)):
                sims = sim_down_mat[i]
                exclude = 1 if i in down_indices else 0
                n_available = max(1, n_down_train - exclude)
                k_use = min(k_eff, n_available)
                topk = np.sort(sims)[-k_use:]
                fb_sim_down[i] = topk.mean()
            sim_down_mat_clean = fb_train_emb @ fb_down_train.T
            if n_down_train > 1:
                for i, tp in enumerate(down_indices):
                    sim_down_mat_clean[tp, i] = -1.0
            fb_closest_down = np.max(sim_down_mat_clean, axis=1)
        else:
            fb_closest_down = np.zeros(len(fb_train_emb))

        fb_train_csr_ratio = fb_train_comments / np.maximum(fb_train_scores, 1)
        fb_train_csr = np.clip(np.log1p(fb_train_csr_ratio), 0, 3.0) / 3.0

        fold_cand_csr_ratio = fold_cand_comments / np.maximum(fold_cand_scores, 1)
        fold_cand_csr = np.clip(np.log1p(fold_cand_csr_ratio), 0, 3.0) / 3.0

        fb_train_is_hn = np.array(
            [1.0 if is_hn_source(s.source) else 0.0 for s in fb_train_stories]
        )

        fold_cand_is_hn = np.array(
            [1.0 if is_hn_source(s.source) else 0.0 for s in fold_candidates]
        )

        X_train = _augment_features(
            fb_train_emb,
            fb_train_scores,
            fb_train_ages,
            comment_counts=fb_train_comments,
            text_lengths=fb_train_textlens,
            hn_quality=fb_train_quality,
            score_velocity=fb_train_score_vel,
            comment_velocity=fb_train_comment_vel,
            sim_to_upvoted=fb_sim_up,
            sim_to_downvoted=fb_sim_down,
            closest_upvoted=fb_closest_up,
            closest_downvoted=fb_closest_down,
            comment_score_ratio=fb_train_csr,
            is_hn=fb_train_is_hn,
        )
        X_cand = _augment_features(
            fold_cand_emb,
            fold_cand_scores,
            fold_cand_ages,
            comment_counts=fold_cand_comments,
            text_lengths=fold_cand_textlens,
            hn_quality=fold_cand_quality,
            score_velocity=fold_cand_score_vel,
            comment_velocity=fold_cand_comment_vel,
            sim_to_upvoted=cand_sim_up,
            sim_to_downvoted=cand_sim_down,
            closest_upvoted=cand_closest_up,
            closest_downvoted=cand_closest_down,
            comment_score_ratio=fold_cand_csr,
            is_hn=fold_cand_is_hn,
        )

        counts = Counter(y_train)
        weights = np.array(
            [len(y_train) / (3 * counts[c]) for c in y_train], dtype=np.float64
        )

        emb_dim = cand_emb.shape[1]
        scaler = StandardScaler()
        X_train_meta_scaled = np.clip(
            scaler.fit_transform(X_train[:, emb_dim:]), -2.5, 2.5
        )
        X_cand_meta_scaled = np.clip(scaler.transform(X_cand[:, emb_dim:]), -2.5, 2.5)

        X_train_scaled = np.hstack([X_train[:, :emb_dim], X_train_meta_scaled])
        X_cand_scaled = np.hstack([X_cand[:, :emb_dim], X_cand_meta_scaled])

        svm = SVC(
            C=config.model.svm_c,
            kernel=config.model.svm_kernel,
            gamma=config.model.svm_gamma,
            random_state=0,
            decision_function_shape="ovr",
            probability=True,
        )
        svm.fit(X_train_scaled, y_train, sample_weight=weights)
        probs = svm.predict_proba(X_cand_scaled)

        # Test: RSS test items for this fold
        test_stories = [fb_stories[gi] for gi in rss_test_global]
        test_actions = fb_labels[rss_test_global]

        for formula in formulas:
            results[formula].append(
                _evaluate_fold(
                    probs,
                    fold_candidates,
                    fold_cand_emb,
                    test_stories,
                    test_actions,
                    fold_cand_scores_array,
                    formula,
                    neutral_weight=config.model.neutral_weight,
                    mmr_threshold=config.model.diversity_threshold,
                    cand_sim_up=cand_sim_up,
                    cand_sim_down=cand_sim_down,
                    cand_closest_up=cand_closest_up,
                    cand_closest_down=cand_closest_down,
                )
            )

        print(f"Fold {fold_idx + 1}/5 done ({len(rss_test_global)} test items)")

    # Aggregate
    metric_keys = (
        "ndcg_at_100",
        "ndcg_at_200",
        "ndcg_at_500",
        "hit_at_100",
        "hit_at_200",
        "hit_at_500",
        "map",
        "mrr",
        "brier_up",
        "median_rank",
        "p25_rank",
        "p75_rank",
    )

    report: dict[str, Any] = {
        "config": {
            "split": "5-fold-stratified-rss",
            "user_token": user.token,
            "user_id": user.id,
            "n_feedback_total": int(len(fb_labels)),
            "n_feedback_rss": int(n_rss),
            "n_feedback_rss_up": int(up_rss),
            "n_feedback_rss_down": int(down_rss),
            "n_feedback_rss_neutral": int(neutral_rss),
            "n_candidates": int(len(candidates)),
            "n_candidates_rss": int(n_cand_rss),
            "rss_source_filter": "source != 'hn'",
            "n_folds": int(n_rss),
            "mmr_threshold": config.model.diversity_threshold,
            "mmr_limit": 40,
            "relevance_grade": "up=1, neutral=0.2, down=0",
            "db_sha256": _db_sha256(config.db_path),
        },
        "formulas": {
            f: {
                "mean": {
                    "mmr": {
                        k: float(np.mean([r["mmr"][k] for r in rs]))
                        for k in metric_keys
                    },
                    "raw": {
                        k: float(np.mean([r["raw"][k] for r in rs]))
                        for k in metric_keys
                    },
                },
                "std": {
                    "mmr": {
                        k: float(np.std([r["mmr"][k] for r in rs])) for k in metric_keys
                    },
                    "raw": {
                        k: float(np.std([r["raw"][k] for r in rs])) for k in metric_keys
                    },
                },
                "per_fold": rs,
            }
            for f, rs in results.items()
        },
        "per_source": {},
    }

    # Per-source breakdown: build from fold-level results
    # For each source, find which test items belong to it and which fold they were in
    source_to_folds: dict[str, list[int]] = {}
    source_to_action: dict[str, list[int]] = {}
    rss_test_global_by_fold = [rss_indices[fold_test_pos] for _, fold_test_pos in folds]
    for fold_idx, test_globals in enumerate(rss_test_global_by_fold):
        for gi in test_globals:
            src = fb_stories[gi].source
            if src == "hn":
                continue
            source_to_folds.setdefault(src, []).append(fold_idx)
            source_to_action.setdefault(src, []).append(int(fb_labels[gi]))
    for source in sorted(source_to_folds.keys()):
        n_test = len(source_to_folds[source])
        if n_test < 5:
            continue
        actions = source_to_action[source]
        n_up = sum(1 for a in actions if a == 2)
        source_formulas = {}
        for f, rs in results.items():
            source_fold_indices = list(set(source_to_folds[source]))
            source_rs = [rs[fi] for fi in source_fold_indices]
            source_formulas[f] = {
                "mean": {
                    "mmr": {
                        k: float(np.mean([r["mmr"][k] for r in source_rs]))
                        for k in metric_keys
                    },
                    "raw": {
                        k: float(np.mean([r["raw"][k] for r in source_rs]))
                        for k in metric_keys
                    },
                },
            }
        report["per_source"][source] = {
            "n_test": n_test,
            "n_up": n_up,
            "formulas": source_formulas,
        }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWritten {REPORT_PATH}")

    print("\n=== Non-HN eval metrics (5-fold stratified) ===")
    for metric in metric_keys:
        print(f"\n{metric}:")
        for f, data in report["formulas"].items():  # type: ignore[union-attr]
            for variant in ("mmr", "raw"):
                m, s = data["mean"][variant][metric], data["std"][variant][metric]  # type: ignore
                print(f"  {f:15s} {variant:4s}  {m:.4f} ± {s:.4f}")

    print("\n=== Per-source breakdown (current MMR) ===")
    for source, data in sorted(report["per_source"].items()):
        n = data["n_test"]
        fd = data["formulas"]
        if "current" in fd:
            ndcg = fd["current"]["mean"]["mmr"]["ndcg_at_100"]
            map_val = fd["current"]["mean"]["mmr"]["map"]
            mrr = fd["current"]["mean"]["mmr"]["mrr"]
            print(
                f"  {source:30s}  n={n:3d}  ndcg@100={ndcg:.3f}  map={map_val:.3f}  mrr={mrr:.3f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
