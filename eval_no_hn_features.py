"""Eval with HN-specific features removed.

Keeps only source-common features: embeddings (384), text_length,
and 4 k-NN similarity features (sim_to_upvoted, sim_to_downvoted,
closest_upvoted, closest_downvoted) = 389-dim.

Removed: scores, comment_counts, hn_quality, comment_score_ratio,
score_velocity, comment_velocity, is_hn.

Compares against full-feature baseline in eval_report.json.
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
from pipeline import (
    Config,
    RankedStory,
    _augment_features,
    _knn_similarity,
    mmr_filter,
)

MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|256"
REPORT_PATH = Path(__file__).parent / "eval_no_hn_features.json"
BASELINE_PATH = Path(__file__).parent / "eval_report.json"


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

    def _ndcg(
        rel_by_pos: dict[int, float], all_test_rels: list[float], k: int
    ) -> float:
        dcg = sum(r / math.log2(p + 2) for p, r in rel_by_pos.items() if p < k)
        ideal = sorted(all_test_rels, reverse=True)[:k]
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

    return {
        "ndcg_at_100": _ndcg(rel_by_pos, all_test_rels, 100),
        "ndcg_at_200": _ndcg(rel_by_pos, all_test_rels, 200),
        "ndcg_at_500": _ndcg(rel_by_pos, all_test_rels, 500),
        "hit_at_100": sum(1 for p in rel_by_pos if p < 100) / max(len(test_stories), 1),
        "hit_at_200": sum(1 for p in rel_by_pos if p < 200) / max(len(test_stories), 1),
        "hit_at_500": sum(1 for p in rel_by_pos if p < 500) / max(len(test_stories), 1),
        "map": ap,
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
) -> dict:
    if formula == "current":
        scores = probs[:, 2] + neutral_weight * probs[:, 1]
    elif formula == "hn_baseline":
        scores = cand_scores.astype(np.float32)
    elif formula == "knn_diff":
        raise ValueError("knn_diff not computed here")
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
    if test_probs_up:
        brier_up = float(
            np.mean((np.array(test_probs_up) - np.array(test_binary_up)) ** 2)
        )
    else:
        brier_up = 0.0

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

    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training(user_id=user.id)
    fb_labels = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    print(f"Feedback: {len(fb_stories)} rows ({Counter(fb_labels)})")

    candidates, cand_emb = _load_candidates(db)
    print(f"Candidates: {len(candidates)}")

    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories], dtype=int)
    valid = fb_to_cand >= 0
    if not valid.all():
        print(
            f"Warning: {(~valid).sum()} feedback stories missing from candidates; excluded."
        )

    now = time.time()
    cand_text_lengths = np.array([len(s.text_content) for s in candidates])
    cand_ages_arr = np.array([now - max(s.time, 1) for s in candidates])
    cand_scores_arr = np.array([s.score for s in candidates])

    fb_emb = cand_emb[fb_to_cand[valid]]
    fb_scores_arr = np.array([s.score for s in fb_stories])[valid]
    fb_ages_arr = np.array(
        [float(vt) - max(s.time, 1) for vt, s in zip(fb_vote_times, fb_stories)]
    )[valid]
    fb_text_lengths_arr = np.array([len(s.text_content) for s in fb_stories])[valid]

    cand_scores_array = np.array([s.score for s in candidates], dtype=np.float64)
    y = fb_labels[valid]

    formulas = ["current", "hn_baseline"]
    results: dict[str, list[dict]] = {f: [] for f in formulas}

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
            np.zeros((len(y), 1)), y
        )
    )

    for fold_idx, (train_pos, test_pos) in enumerate(folds):
        fb_train_emb = fb_emb[train_pos]
        y_train = y[train_pos]

        fb_train_scores = fb_scores_arr[train_pos]
        fb_train_ages = fb_ages_arr[train_pos]
        fb_train_textlens = fb_text_lengths_arr[train_pos]

        train_ids = {fb_stories[idx].id for idx in np.where(valid)[0][train_pos]}
        cand_mask = np.array([s.id not in train_ids for s in candidates])
        fold_candidates = [s for idx, s in enumerate(candidates) if cand_mask[idx]]
        fold_cand_emb = cand_emb[cand_mask]
        fold_cand_scores = cand_scores_arr[cand_mask]
        fold_cand_ages = cand_ages_arr[cand_mask]
        fold_cand_textlens = cand_text_lengths[cand_mask]
        fold_cand_scores_array = cand_scores_array[cand_mask]

        up_mask = y_train == 2
        down_mask = y_train == 0
        fb_up_train = fb_train_emb[up_mask]
        fb_down_train = fb_train_emb[down_mask]
        n_up = up_mask.sum()
        n_down = down_mask.sum()

        k = config.model.knn_k

        cand_sim_up = _knn_similarity(fold_cand_emb, fb_up_train, k)
        cand_sim_down = _knn_similarity(fold_cand_emb, fb_down_train, k)
        cand_closest_up = (
            np.max(fold_cand_emb @ fb_up_train.T, axis=1)
            if n_up
            else np.zeros(len(fold_candidates))
        )
        cand_closest_down = (
            np.max(fold_cand_emb @ fb_down_train.T, axis=1)
            if n_down
            else np.zeros(len(fold_candidates))
        )

        fb_sim_up = np.zeros(len(fb_train_emb), dtype=np.float32)
        fb_sim_down = np.zeros(len(fb_train_emb), dtype=np.float32)

        if n_up > 0:
            up_indices = np.where(up_mask)[0]
            sim_up_mat = fb_train_emb @ fb_up_train.T
            if n_up > 1:
                for i, tp in enumerate(up_indices):
                    sim_up_mat[tp, i] = -2.0
            k_eff = min(k, n_up)
            for i in range(len(fb_train_emb)):
                sims = sim_up_mat[i]
                exclude = 1 if i in up_indices else 0
                n_available = max(1, n_up - exclude)
                k_use = min(k_eff, n_available)
                topk = np.sort(sims)[-k_use:]
                fb_sim_up[i] = topk.mean()
            sim_up_mat_clean = fb_train_emb @ fb_up_train.T
            if n_up > 1:
                for i, tp in enumerate(up_indices):
                    sim_up_mat_clean[tp, i] = -1.0
            fb_closest_up = np.max(sim_up_mat_clean, axis=1)
        else:
            fb_closest_up = np.zeros(len(fb_train_emb))

        if n_down > 0:
            down_indices = np.where(down_mask)[0]
            sim_down_mat = fb_train_emb @ fb_down_train.T
            if n_down > 1:
                for i, tp in enumerate(down_indices):
                    sim_down_mat[tp, i] = -2.0
            k_eff = min(k, n_down)
            for i in range(len(fb_train_emb)):
                sims = sim_down_mat[i]
                exclude = 1 if i in down_indices else 0
                n_available = max(1, n_down - exclude)
                k_use = min(k_eff, n_available)
                topk = np.sort(sims)[-k_use:]
                fb_sim_down[i] = topk.mean()
            sim_down_mat_clean = fb_train_emb @ fb_down_train.T
            if n_down > 1:
                for i, tp in enumerate(down_indices):
                    sim_down_mat_clean[tp, i] = -1.0
            fb_closest_down = np.max(sim_down_mat_clean, axis=1)
        else:
            fb_closest_down = np.zeros(len(fb_train_emb))

        # Build features WITHOUT HN-specific fields.
        # Pass HN-specific fields as None so _augment_features skips them.
        # scores is required positional; we drop it post-hoc.
        X_train = _augment_features(
            fb_train_emb,
            fb_train_scores,
            fb_train_ages,
            comment_counts=None,
            text_lengths=fb_train_textlens,
            hn_quality=None,
            score_velocity=None,
            comment_velocity=None,
            sim_to_upvoted=fb_sim_up,
            sim_to_downvoted=fb_sim_down,
            closest_upvoted=fb_closest_up,
            closest_downvoted=fb_closest_down,
            comment_score_ratio=None,
            is_hn=None,
        )
        X_cand = _augment_features(
            fold_cand_emb,
            fold_cand_scores,
            fold_cand_ages,
            comment_counts=None,
            text_lengths=fold_cand_textlens,
            hn_quality=None,
            score_velocity=None,
            comment_velocity=None,
            sim_to_upvoted=cand_sim_up,
            sim_to_downvoted=cand_sim_down,
            closest_upvoted=cand_closest_up,
            closest_downvoted=cand_closest_down,
            comment_score_ratio=None,
            is_hn=None,
        )

        emb_dim = cand_emb.shape[1]

        # Drop scores column (index emb_dim in the 390-dim output).
        # Remaining meta: [text_length, sim_up, sim_down, close_up, close_down] — 5 cols.
        X_train = np.delete(X_train, emb_dim, axis=1)
        X_cand = np.delete(X_cand, emb_dim, axis=1)

        counts = Counter(y_train)
        weights = np.array(
            [len(y_train) / (3 * counts[c]) for c in y_train], dtype=np.float64
        )

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

        test_stories = [
            fb_stories[valid_idx] for valid_idx in np.where(valid)[0][test_pos]
        ]
        test_actions = y[test_pos]

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
                )
            )

        print(f"Fold {fold_idx + 1}/5 done")

    metric_keys = (
        "ndcg_at_100",
        "ndcg_at_200",
        "ndcg_at_500",
        "hit_at_100",
        "hit_at_200",
        "hit_at_500",
        "map",
        "brier_up",
        "median_rank",
        "p25_rank",
        "p75_rank",
    )

    report: dict[str, Any] = {
        "config": {
            "split": "5-fold-stratified",
            "user_token": user.token,
            "user_id": user.id,
            "n_feedback": int(len(fb_labels)),
            "n_candidates": int(len(candidates)),
            "n_folds": 5,
            "mmr_threshold": config.model.diversity_threshold,
            "mmr_limit": 40,
            "relevance_grade": "up=1, neutral=0.2, down=0",
            "db_sha256": _db_sha256(config.db_path),
            "feature_set": "common-only (embeddings + text_length + 4 sim)",
            "removed": "scores, comment_counts, hn_quality, comment_score_ratio, score_velocity, comment_velocity, is_hn",
            "n_features": int(X_train_scaled.shape[1]),
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
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWritten {REPORT_PATH}")

    # Side-by-side comparison with baseline
    if BASELINE_PATH.exists():
        base = json.loads(BASELINE_PATH.read_text())
    else:
        base = None

    for metric in ("ndcg_at_100", "ndcg_at_200", "map", "brier_up", "median_rank"):
        print(f"\n{metric} (MMR):")
        print(f"  {'formula':14s} {'no-HN':>8s} {'full':>8s} {'Δ':>8s}")
        print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 8}")
        for f in formulas:
            no_hn_val = report["formulas"][f]["mean"]["mmr"][metric]  # type: ignore
            full_val = (
                (
                    base["formulas"]
                    .get(f, {})
                    .get("mean", {})
                    .get("mmr", {})
                    .get(metric, None)
                )
                if base
                else None
            )
            if full_val is not None:
                delta = no_hn_val - full_val
                print(f"  {f:14s} {no_hn_val:>8.4f} {full_val:>8.4f} {delta:>+8.4f}")
            else:
                print(f"  {f:14s} {no_hn_val:>8.4f} {'N/A':>8s} {'N/A':>8s}")
        print(f"\n  {metric} (RAW):")
        print(f"  {'formula':14s} {'no-HN':>8s} {'full':>8s} {'Δ':>8s}")
        print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 8}")
        for f in formulas:
            no_hn_val = report["formulas"][f]["mean"]["raw"][metric]  # type: ignore
            full_val = (
                (
                    base["formulas"]
                    .get(f, {})
                    .get("mean", {})
                    .get("raw", {})
                    .get(metric, None)
                )
                if base
                else None
            )
            if full_val is not None:
                delta = no_hn_val - full_val
                print(f"  {f:14s} {no_hn_val:>8.4f} {full_val:>8.4f} {delta:>+8.4f}")
            else:
                print(f"  {f:14s} {no_hn_val:>8.4f} {'N/A':>8s} {'N/A':>8s}")


if __name__ == "__main__":
    main()
