"""Ranking evaluation. OFFLINE-ONLY: reads hn_rewrite.db exclusively.

Compares 3 score formulas via 5-fold stratified CV:
  current:     raw up-margin + 0.5 * raw neutral-margin   [production]
  up_only:     raw up-margin
  hn_baseline: raw HN points (no SVM)

Ranks by the SVM's raw one-vs-rest decision margin on the up class
(matches `pipeline.py:1761-1772` production path; `probability=False`
avoids the deprecated Platt-scaling path). Softmax over the 3-class
decision matrix is used to derive a soft P(up) for the brier_up
calibration metric, matching what production does for UI entropy.

Personalization features are computed per-fold (LOOCV self-exclusion)
to avoid train-test leakage. MMR and raw (pre-MMR) metrics both reported.

Writes eval_report.json (committed to git for tracking).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from database import Database, Story
    from legacy_features import _augment_features  # noqa: F401  (deprecated 2026-06-28; use pipeline._svm_personalization_features)
    from pipeline import (
        Config,
        Embedder,
        RankedStory,
        mmr_filter,
        rerank_candidates,
        story_embedding_text,
    )

MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|256"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


def _db_sha256(db_path: str) -> str:
    return hashlib.sha256(Path(db_path).read_bytes()).hexdigest()[:16]


def _load_candidates(db: Database) -> tuple[list[Story], np.ndarray]:
    """Read all non-negative-cached stories + their embeddings."""
    from database import Database as _Database

    rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, "
        "       comment_count, discussion_url, comment_count_at_fetch, "
        "       self_text, top_comments, article_body "
        "FROM stories WHERE text_content != ''"
    )
    stories = [_Database._row_to_story(row) for row in rows]
    import hashlib

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
    k_values: tuple[int, ...] = (40,),
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
        **{f"ndcg_at_{k}": _ndcg(rel_by_pos, all_test_rels, k) for k in k_values},
        **{
            f"hit_at_{k}": sum(1 for p in rel_by_pos if p < k)
            / max(len(test_stories), 1)
            for k in k_values
        },
        "map": ap,
        "brier_up": brier_up,
    }


def _evaluate_fold(
    decision: np.ndarray,
    probs: np.ndarray,
    up_idx: int,
    candidates: list[Story],
    cand_emb: np.ndarray,
    test_stories: list[Story],
    test_actions: np.ndarray,
    cand_scores: np.ndarray,
    formula: str,
    neutral_weight: float = 0.0,
    mmr_threshold: float = 0.50,
    mmr_limit: int = 40,
    decision_strip: np.ndarray | None = None,
    probs_strip: np.ndarray | None = None,
    cand_sim_up: np.ndarray | None = None,
    cand_sim_down: np.ndarray | None = None,
) -> dict:
    if formula == "current":
        if neutral_weight != 0.0:
            class_order = list(range(decision.shape[1]))
            if 1 in class_order:
                scores = (
                    decision[:, up_idx]
                    + neutral_weight * decision[:, class_order.index(1)]
                )
            else:
                scores = decision[:, up_idx]
        else:
            scores = decision[:, up_idx]
    elif formula == "up_only":
        scores = decision[:, up_idx]
    elif formula == "hn_baseline":
        scores = cand_scores.astype(np.float32)
    elif formula == "strip_hn":
        if decision_strip is None:
            raise ValueError("strip_hn requires decision_strip")
        scores = decision_strip[:, up_idx]
    elif formula == "knn_diff":
        if cand_sim_up is None or cand_sim_down is None:
            raise ValueError("knn_diff requires candidate similarity arrays")
        scores = cand_sim_up - cand_sim_down
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

    # Brier score: calibration of P(up) against actual upvote outcomes
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

    # MMR-based ranking (production path)
    top40 = mmr_filter(ranked, emb_map, threshold=mmr_threshold, limit=mmr_limit)
    mmr_rank_map = {rs.story.id: pos for pos, rs in enumerate(top40)}
    mmr_metrics = _compute_metrics(
        mmr_rank_map,
        test_stories,
        test_actions,
        test_rel,
        all_test_rels,
        brier_up,
        k_values=(40,),
    )

    # Raw ranking (no MMR, diagnostic) — full ranking for meaningful MAP
    raw_ranked = ranked
    raw_rank_map = {rs.story.id: pos for pos, rs in enumerate(raw_ranked)}
    raw_metrics = _compute_metrics(
        raw_rank_map,
        test_stories,
        test_actions,
        test_rel,
        all_test_rels,
        brier_up,
        k_values=(40,),
    )

    # Calculate rank statistics of upvoted test stories in overall ranked candidates
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

    # Per-source NDCG: filter test_stories by source and recompute metrics on
    # the source-filtered subset, against the same rank_map. This gives a real
    # per-source NDCG (not a fold-averaged proxy).
    source_to_test_idx: dict[str, list[int]] = {}
    for i, ts in enumerate(test_stories):
        source_to_test_idx.setdefault(ts.source, []).append(i)

    per_source: dict[str, dict] = {}
    for source, indices in source_to_test_idx.items():
        n_test = len(indices)
        if n_test < 5:
            continue
        n_up = sum(1 for i in indices if test_actions[i] == 2)
        src_stories = [test_stories[i] for i in indices]
        src_actions = test_actions[indices]
        src_rel = test_rel[indices]
        src_all_rels = src_rel.tolist()
        # brier_up is a fold-level calibration metric; not meaningful per-source.
        # Pop the per-call default (0.0) so it doesn't show up as a fake 0.0 in
        # the per-source breakdown.
        src_mmr = _compute_metrics(
            mmr_rank_map,
            src_stories,
            src_actions,
            src_rel,
            src_all_rels,
            brier_up=0.0,
            k_values=(40,),
        )
        src_mmr.pop("brier_up", None)
        src_raw = _compute_metrics(
            raw_rank_map,
            src_stories,
            src_actions,
            src_rel,
            src_all_rels,
            brier_up=0.0,
            k_values=(40,),
        )
        src_raw.pop("brier_up", None)
        # Per-source rank percentiles of upvoted test items
        src_upvote_ranks_raw = sorted(
            raw_rank_map[test_stories[i].id]
            for i in indices
            if test_actions[i] == 2 and test_stories[i].id in raw_rank_map
        )
        src_raw["median_rank"] = (
            float(np.median(src_upvote_ranks_raw)) if src_upvote_ranks_raw else 0.0
        )
        src_raw["p25_rank"] = (
            float(np.percentile(src_upvote_ranks_raw, 25))
            if src_upvote_ranks_raw
            else 0.0
        )
        src_raw["p75_rank"] = (
            float(np.percentile(src_upvote_ranks_raw, 75))
            if src_upvote_ranks_raw
            else 0.0
        )
        src_upvote_ranks_mmr = sorted(
            mmr_rank_map[test_stories[i].id]
            for i in indices
            if test_actions[i] == 2 and test_stories[i].id in mmr_rank_map
        )
        src_mmr["median_rank"] = (
            float(np.median(src_upvote_ranks_mmr)) if src_upvote_ranks_mmr else 0.0
        )
        src_mmr["p25_rank"] = (
            float(np.percentile(src_upvote_ranks_mmr, 25))
            if src_upvote_ranks_mmr
            else 0.0
        )
        src_mmr["p75_rank"] = (
            float(np.percentile(src_upvote_ranks_mmr, 75))
            if src_upvote_ranks_mmr
            else 0.0
        )
        per_source[source] = {
            "n_test": n_test,
            "n_up": n_up,
            "mmr": src_mmr,
            "raw": src_raw,
        }

    return {"mmr": mmr_metrics, "raw": raw_metrics, "per_source": per_source}


def _compute_final_queue_metrics(
    fold_candidates: list[Story],
    fold_cand_emb: np.ndarray,
    fb_train_stories: list[Story],
    y_train: np.ndarray,
    cand_emb: np.ndarray,
    cand_id_to_idx: dict[int, int],
    config: Config,
    embedder: Embedder,
    fold_idx: int,
    test_stories: list[Story],
    test_actions: np.ndarray,
) -> dict:
    """Run the production pipeline (rerank_candidates) per fold and measure
    NDCG@40 on the final queue (post-MMR, post-discovery-passes, post-sort).

    Returns {"mmr": metrics, "per_source": {source: ...}}.
    Returns empty dict on failure.
    """
    db = Database(":memory:")
    uid = 1000 + fold_idx

    sid_to_emb: dict[int, np.ndarray] = {}
    for s, emb in zip(fold_candidates, fold_cand_emb):
        sid_to_emb[s.id] = emb
    for s in fb_train_stories:
        idx = cand_id_to_idx.get(s.id)
        if idx is not None:
            sid_to_emb[s.id] = cand_emb[idx]

    all_stories: dict[int, Story] = {s.id: s for s in fold_candidates}
    for s in fb_train_stories:
        all_stories[s.id] = s

    for s in all_stories.values():
        db.upsert_story(s)

    for sid, emb in sid_to_emb.items():
        s = all_stories[sid]
        text = story_embedding_text(s)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        db.upsert_embedding(sid, MODEL_VERSION, text_hash, emb)

    for s, a in zip(fb_train_stories, y_train):
        action = "down" if int(a) == 0 else "neutral" if int(a) == 1 else "up"
        db.upsert_feedback(uid, s.id, action)

    try:
        final = rerank_candidates(
            db, config, embedder, fold_candidates, fold_cand_emb, user_id=uid
        )
    except Exception:
        # Final queue unavailable (e.g. embedder model dir missing, backend error)
        db.close()
        return {}

    if not final:
        db.close()
        return {}

    final_top40 = final[:40]
    final_rank_map = {r.story.id: pos for pos, r in enumerate(final_top40)}

    rel_map = {0: 0.0, 1: 0.2, 2: 1.0}
    test_rel = np.array([rel_map[int(a)] for a in test_actions])
    all_test_rels = test_rel.tolist()

    final_cand_id_to_idx = {r.story.id: i for i, r in enumerate(final)}
    test_probs_up: list[float] = []
    test_binary_up: list[float] = []
    for i, ts in enumerate(test_stories):
        fi = final_cand_id_to_idx.get(ts.id)
        if fi is not None:
            p = final[fi].prob_up
            if p is not None:
                test_probs_up.append(p)
                test_binary_up.append(1.0 if test_actions[i] == 2 else 0.0)
    brier_up = (
        float(np.mean((np.array(test_probs_up) - np.array(test_binary_up)) ** 2))
        if test_probs_up
        else 0.0
    )

    metrics = _compute_metrics(
        final_rank_map,
        test_stories,
        test_actions,
        test_rel,
        all_test_rels,
        brier_up,
        k_values=(40,),
    )

    # Rank percentiles across the full final queue (not just top-40)
    test_upvote_ids = {s.id for i, s in enumerate(test_stories) if test_actions[i] == 2}
    upvote_ranks = [pos for pos, r in enumerate(final) if r.story.id in test_upvote_ids]
    metrics["median_rank"] = float(np.median(upvote_ranks)) if upvote_ranks else 0.0
    metrics["p25_rank"] = (
        float(np.percentile(upvote_ranks, 25)) if upvote_ranks else 0.0
    )
    metrics["p75_rank"] = (
        float(np.percentile(upvote_ranks, 75)) if upvote_ranks else 0.0
    )

    source_to_test_idx: dict[str, list[int]] = {}
    for i, ts in enumerate(test_stories):
        source_to_test_idx.setdefault(ts.source, []).append(i)

    per_source: dict[str, dict] = {}
    for source, indices in source_to_test_idx.items():
        n_test = len(indices)
        if n_test < 5:
            continue
        n_up = sum(1 for i in indices if test_actions[i] == 2)
        src_stories = [test_stories[i] for i in indices]
        src_actions = test_actions[indices]
        src_rel = test_rel[indices]
        src_all_rels = src_rel.tolist()
        src_mmr = _compute_metrics(
            final_rank_map,
            src_stories,
            src_actions,
            src_rel,
            src_all_rels,
            brier_up=0.0,
            k_values=(40,),
        )
        # brier_up is fold-level only; drop the per-call default to keep
        # per-source output honest.
        src_mmr.pop("brier_up", None)
        src_upvote_ranks = sorted(
            final_rank_map[test_stories[i].id]
            for i in indices
            if test_actions[i] == 2 and test_stories[i].id in final_rank_map
        )
        src_mmr["median_rank"] = (
            float(np.median(src_upvote_ranks)) if src_upvote_ranks else 0.0
        )
        src_mmr["p25_rank"] = (
            float(np.percentile(src_upvote_ranks, 25)) if src_upvote_ranks else 0.0
        )
        src_mmr["p75_rank"] = (
            float(np.percentile(src_upvote_ranks, 75)) if src_upvote_ranks else 0.0
        )
        per_source[source] = {"n_test": n_test, "n_up": n_up, "mmr": src_mmr}

    db.close()
    return {"mmr": metrics, "per_source": per_source}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="5-fold offline eval. Pass --candidate-cap N to subsample "
        "candidates for an apples-to-apples comparison against a prior run."
    )
    parser.add_argument(
        "--candidate-cap",
        type=int,
        default=None,
        help="Subsample candidates to this many stories (random, fixed seed).",
    )
    parser.add_argument(
        "--candidate-cap-seed",
        type=int,
        default=0,
        help="Random seed for --candidate-cap subsampling (default 0).",
    )
    parser.add_argument(
        "--exclude-sources",
        nargs="*",
        default=None,
        help="Source names to drop from the candidate pool "
        "(e.g., --exclude-sources ch_seed bq_seed to measure on a "
        "non-archive pool).",
    )
    args = parser.parse_args()

    # Heavy imports deferred until after parse_args() so `eval.py --help`
    # doesn't pay the ~2s transformers+onnxruntime cold-start cost.
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from database import Database
    from pipeline import (
        Config,
        Embedder,
        _knn_similarity,
        _softmax_rows,
        _svm_personalization_features,
    )

    config = Config.load()
    db = Database(config.db_path)
    user = db.get_user_by_token("default")
    if user is None:
        raise RuntimeError("Missing default user token")

    try:
        embedder: Embedder | None = Embedder("onnx_model")
    except Exception as exc:
        print(f"Warning: Embedder not available ({exc}); skipping final_queue metrics.")
        embedder = None

    # Feedback
    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training(user_id=user.id)
    fb_labels = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    print(f"Feedback: {len(fb_stories)} rows ({Counter(fb_labels)})")

    # Candidates
    candidates, cand_emb = _load_candidates(db)
    print(f"Candidates: {len(candidates)}")
    if args.exclude_sources:
        excluded = set(args.exclude_sources)
        before = len(candidates)
        keep_idx = np.array([s.source not in excluded for s in candidates], dtype=bool)
        candidates = [s for s, k in zip(candidates, keep_idx) if k]
        cand_emb = cand_emb[keep_idx]
        print(f"  Excluded sources {sorted(excluded)}: {before} -> {len(candidates)}")
    if args.candidate_cap is not None and len(candidates) > args.candidate_cap:
        rng = np.random.default_rng(args.candidate_cap_seed)
        keep_idx = np.sort(
            rng.choice(len(candidates), size=args.candidate_cap, replace=False)
        )
        candidates = [candidates[i] for i in keep_idx]
        cand_emb = cand_emb[keep_idx]
        print(
            f"  Subsampled to {len(candidates)} candidates "
            f"(seed={args.candidate_cap_seed}) for apples-to-apples comparison"
        )

    # Map feedback stories → candidate indices
    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories], dtype=int)
    valid = fb_to_cand >= 0
    if not valid.all():
        print(
            f"Warning: {(~valid).sum()} feedback stories missing from candidates; excluded."
        )

    # Candidate features (age = now - story.time)
    cand_text_lengths = np.array([len(s.text_content) for s in candidates])

    # Feedback metadata (age = vote_time - story.time)
    fb_emb = cand_emb[fb_to_cand[valid]]
    fb_text_lengths_arr = np.array([len(s.text_content) for s in fb_stories])[valid]

    cand_scores_array = np.array([s.score for s in candidates], dtype=np.float64)
    y = fb_labels[valid]

    formulas = ["current", "up_only", "strip_hn", "hn_baseline", "knn_diff"]
    results: dict[str, list[dict]] = {f: [] for f in formulas}
    final_queue_results: list[dict] = []
    # Accumulated binary upvote labels across all folds; used to compute the
    # class-prior brier baseline (brier_const = p*(1-p)) at the end.
    all_y_binary: list[np.ndarray] = []

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
            np.zeros((len(y), 1)), y
        )
    )

    for fold_idx, (train_pos, test_pos) in enumerate(folds):
        # Slice training data for this fold
        fb_train_emb = fb_emb[train_pos]
        y_train = y[train_pos]

        fb_train_textlens = fb_text_lengths_arr[train_pos]
        fb_train_stories = [fb_stories[idx] for idx in np.where(valid)[0][train_pos]]

        # Exclude training story IDs from candidate pool to prevent leakage
        train_ids = {fb_stories[idx].id for idx in np.where(valid)[0][train_pos]}
        cand_mask = np.array([s.id not in train_ids for s in candidates])
        fold_candidates = [s for idx, s in enumerate(candidates) if cand_mask[idx]]
        fold_cand_emb = cand_emb[cand_mask]
        fold_cand_textlens = cand_text_lengths[cand_mask]
        fold_cand_scores_array = cand_scores_array[cand_mask]

        # Per-fold personalization with LOOCV self-exclusion
        up_mask = y_train == 2
        down_mask = y_train == 0
        fb_up_train = fb_train_emb[up_mask]
        fb_down_train = fb_train_emb[down_mask]
        n_up = up_mask.sum()
        n_down = down_mask.sum()

        # k-NN similarity (replaces global mean)
        k = config.model.knn_k

        # Candidate features (no self issue)
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

        # Train features: k-NN with LOOCV self-exclusion
        fb_sim_up = np.zeros(len(fb_train_emb), dtype=np.float32)
        fb_sim_down = np.zeros(len(fb_train_emb), dtype=np.float32)

        if n_up > 0:
            up_indices = np.where(up_mask)[0]
            sim_up_mat = fb_train_emb @ fb_up_train.T
            if n_up > 1:
                for i, tp in enumerate(up_indices):
                    sim_up_mat[tp, i] = -2.0  # exclude self
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

        from pipeline import source_category_stack

        fb_train_source = source_category_stack([s.source for s in fb_train_stories])
        fold_cand_source = source_category_stack([s.source for s in fold_candidates])

        # Build features for this fold (production 394-d feature set;
        # matches pipeline._svm_personalization_features used at runtime)
        X_train = _svm_personalization_features(
            fb_train_emb,
            text_lengths=fb_train_textlens,
            sim_to_upvoted=fb_sim_up,
            sim_to_downvoted=fb_sim_down,
            closest_upvoted=fb_closest_up,
            closest_downvoted=fb_closest_down,
            positive_cluster_similarity=None,
            is_hn_live=fb_train_source[:, 0],
            is_archive=fb_train_source[:, 1],
            is_reddit=fb_train_source[:, 2],
            is_rss=fb_train_source[:, 3],
        )
        X_cand = _svm_personalization_features(
            fold_cand_emb,
            text_lengths=fold_cand_textlens,
            sim_to_upvoted=cand_sim_up,
            sim_to_downvoted=cand_sim_down,
            closest_upvoted=cand_closest_up,
            closest_downvoted=cand_closest_down,
            positive_cluster_similarity=None,
            is_hn_live=fold_cand_source[:, 0],
            is_archive=fold_cand_source[:, 1],
            is_reddit=fold_cand_source[:, 2],
            is_rss=fold_cand_source[:, 3],
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
            probability=False,
        )
        svm.fit(X_train_scaled, y_train, sample_weight=weights)
        # Production path (`pipeline.py:1761-1772`): rank by raw one-vs-rest
        # up-margin from `decision_function`. `probability=True` would add a
        # 5-fold internal Platt-scaling step and is the deprecated path.
        decision = svm.decision_function(X_cand_scaled)
        class_order = list(svm.classes_)
        up_idx = class_order.index(2)
        # Softmax(decision) provides soft P(up) for the brier_up calibration
        # metric, matching the UI's entropy convention. Not a true calibrated
        # probability.
        probs = _softmax_rows(decision)

        # SVM with HN-specific features zeroed (production 394-d layout:
        # strips text_length + the 4 source-category dummies)
        X_train_strip = X_train_scaled.copy()
        X_cand_strip = X_cand_scaled.copy()
        # meta cols after emb_dim (production 394-d feature set):
        # 0=text_length, 1=sim_up, 2=sim_down, 3=closest_up, 4=closest_down,
        # 5=positive_cluster_similarity, 6=is_hn_live, 7=is_archive,
        # 8=is_reddit, 9=is_rss
        strip = [
            emb_dim + 0,  # text_length
            emb_dim + 6,  # is_hn_live
            emb_dim + 7,  # is_archive
            emb_dim + 8,  # is_reddit
            emb_dim + 9,  # is_rss
        ]
        X_train_strip[:, strip] = 0.0
        X_cand_strip[:, strip] = 0.0
        svm_s = SVC(
            C=config.model.svm_c,
            kernel=config.model.svm_kernel,
            gamma=config.model.svm_gamma,
            random_state=0,
            decision_function_shape="ovr",
            probability=False,
        )
        svm_s.fit(X_train_strip, y_train, sample_weight=weights)
        decision_strip = svm_s.decision_function(X_cand_strip)
        probs_strip = _softmax_rows(decision_strip)

        # Test fold: map test positions back to stories
        test_stories = [
            fb_stories[valid_idx] for valid_idx in np.where(valid)[0][test_pos]
        ]
        test_actions = y[test_pos]
        all_y_binary.append((test_actions == 2).astype(np.float32))

        for formula in formulas:
            results[formula].append(
                _evaluate_fold(
                    decision,
                    probs,
                    up_idx,
                    fold_candidates,
                    fold_cand_emb,
                    test_stories,
                    test_actions,
                    fold_cand_scores_array,
                    formula,
                    neutral_weight=config.model.neutral_weight,
                    mmr_threshold=config.model.diversity_threshold,
                    decision_strip=decision_strip,
                    probs_strip=probs_strip,
                    cand_sim_up=cand_sim_up,
                    cand_sim_down=cand_sim_down,
                )
            )

        # Final queue: production pipeline (tier blend + MMR/slice +
        # 13 discovery passes + enrichment + sort).
        # Single metric per fold (not per-formula).
        # Skipped when embedder failed to load (model dir missing); the
        # main 5-formula report is unaffected.
        if embedder is not None:
            fq = _compute_final_queue_metrics(
                fold_candidates=fold_candidates,
                fold_cand_emb=fold_cand_emb,
                fb_train_stories=fb_train_stories,
                y_train=y_train,
                cand_emb=cand_emb,
                cand_id_to_idx=cand_id_to_idx,
                config=config,
                embedder=embedder,
                fold_idx=fold_idx,
                test_stories=test_stories,
                test_actions=test_actions,
            )
            final_queue_results.append(fq)
        else:
            final_queue_results.append({})

        print(
            f"Fold {fold_idx + 1}/5 done  "
            f"n_test={len(test_stories)}  "
            f"n_up={int((test_actions == 2).sum())}  "
            f"n_neutral={int((test_actions == 1).sum())}  "
            f"n_down={int((test_actions == 0).sum())}"
        )

    # Aggregate
    metric_keys = (
        "ndcg_at_40",
        "hit_at_40",
        "map",
        "brier_up",
        "median_rank",
        "p25_rank",
        "p75_rank",
    )
    # Per-source breakdown: brier_up is fold-level only (calibration is
    # measured against the full test set, not per-source). Drop it from the
    # per-source metric set so aggregation doesn't try to read a missing key.
    per_source_metric_keys = tuple(k for k in metric_keys if k != "brier_up")

    report = {
        "config": {
            "split": "5-fold-stratified",
            "random_state": 0,
            "user_token": user.token,
            "user_id": user.id,
            "n_feedback": int(len(fb_labels)),
            "n_candidates": int(len(candidates)),
            "n_folds": 5,
            "mmr_threshold": config.model.diversity_threshold,
            "mmr_limit": 40,
            "relevance_grade": "up=1, neutral=0.2, down=0",
            "db_sha256": _db_sha256(config.db_path),
            "candidate_cap": args.candidate_cap,
            "candidate_cap_seed": args.candidate_cap_seed,
            "exclude_sources": args.exclude_sources,
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

    # Per-source aggregation across folds.
    # n_test/n_up are fold-level (depend on test_stories only, identical across
    # formulas within a fold) — take from the "current" formula's fold results
    # to avoid 5x overcounting.
    source_n_test: dict[str, int] = {}
    source_n_up: dict[str, int] = {}
    for fold_result in results["current"]:
        for source, ps in fold_result.get("per_source", {}).items():
            source_n_test[source] = source_n_test.get(source, 0) + ps["n_test"]
            source_n_up[source] = source_n_up.get(source, 0) + ps["n_up"]

    per_source_aggregated: dict[str, dict] = {}
    for source in sorted(source_n_test.keys(), key=lambda s: -source_n_test[s]):
        fold_ps_per_formula: dict[str, list[dict]] = {f: [] for f in formulas}
        for f, rs in results.items():
            for fold_result in rs:
                ps = fold_result.get("per_source", {}).get(source)
                if ps is None:
                    continue
                fold_ps_per_formula[f].append(ps)
        if not any(fold_ps_per_formula.values()):
            continue
        source_formulas: dict[str, dict] = {}
        for f, fold_ps_list in fold_ps_per_formula.items():
            if not fold_ps_list:
                continue
            source_formulas[f] = {
                "mean": {
                    "mmr": {
                        k: float(np.mean([ps["mmr"][k] for ps in fold_ps_list]))
                        for k in per_source_metric_keys
                    },
                    "raw": {
                        k: float(np.mean([ps["raw"][k] for ps in fold_ps_list]))
                        for k in per_source_metric_keys
                    },
                },
                "std": {
                    "mmr": {
                        k: float(np.std([ps["mmr"][k] for ps in fold_ps_list]))
                        for k in per_source_metric_keys
                    },
                    "raw": {
                        k: float(np.std([ps["raw"][k] for ps in fold_ps_list]))
                        for k in per_source_metric_keys
                    },
                },
            }
        per_source_aggregated[source] = {
            "n_test": source_n_test[source],
            "n_up": source_n_up[source],
            "formulas": source_formulas,
        }

    report["per_source"] = per_source_aggregated

    # Aggregate final queue metrics across folds.
    # A fold may have an empty `fr` dict if the production pipeline failed
    # (e.g. ONNX model missing) — filter those out before aggregating so
    # the report still writes for the 5 main formula folds even if all
    # final_queue folds failed.
    valid_fq = [fr for fr in final_queue_results if fr]
    if valid_fq:
        fq_mean_mmr = {
            k: float(np.mean([fr["mmr"][k] for fr in valid_fq])) for k in metric_keys
        }
        fq_std_mmr = {
            k: float(np.std([fr["mmr"][k] for fr in valid_fq])) for k in metric_keys
        }
        source_n_test_fq: dict[str, int] = {}
        source_n_up_fq: dict[str, int] = {}
        for fr in valid_fq:
            for source, ps in fr.get("per_source", {}).items():
                source_n_test_fq[source] = (
                    source_n_test_fq.get(source, 0) + ps["n_test"]
                )
                source_n_up_fq[source] = source_n_up_fq.get(source, 0) + ps["n_up"]
        per_source_fq: dict[str, dict] = {}
        for source in sorted(
            source_n_test_fq.keys(), key=lambda s: -source_n_test_fq[s]
        ):
            fold_ps_list = [
                fr["per_source"][source]
                for fr in valid_fq
                if source in fr.get("per_source", {})
            ]
            if not fold_ps_list:
                continue
            ps_mean_mmr = {
                k: float(np.mean([pso["mmr"][k] for pso in fold_ps_list]))
                for k in per_source_metric_keys
            }
            ps_std_mmr = {
                k: float(np.std([pso["mmr"][k] for pso in fold_ps_list]))
                for k in per_source_metric_keys
            }
            per_source_fq[source] = {
                "n_test": source_n_test_fq[source],
                "n_up": source_n_up_fq[source],
                "mean": {"mmr": ps_mean_mmr},
                "std": {"mmr": ps_std_mmr},
            }
        report["final_queue"] = {
            "mean": {"mmr": fq_mean_mmr},
            "std": {"mmr": fq_std_mmr},
            "per_fold": final_queue_results,
            "per_source": per_source_fq,
        }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWritten {REPORT_PATH}")

    # Brier baseline: best constant predictor is the class prior p, giving
    # brier_const = p * (1 - p). A well-calibrated model should beat this.
    if all_y_binary:
        y_all = np.concatenate(all_y_binary)
        p = float(y_all.mean())
        brier_const = p * (1.0 - p)
        print(
            f"\nClass prior: p(up) = {p:.4f}  |  "
            f"brier_const (constant predictor) = {brier_const:.4f}"
        )

    for metric in metric_keys:
        print(f"\n{metric} by formula (mean ± std):")
        for f, data in report["formulas"].items():  # type: ignore[union-attr]
            for variant in ("mmr", "raw"):
                m, s = data["mean"][variant][metric], data["std"][variant][metric]  # type: ignore
                print(f"  {f:12s} {variant:4s} {m:.3f} ± {s:.3f}")

    print("\n=== Per-source breakdown (current formula, ndcg_at_40 + hit_at_40) ===")
    print(
        f"  {'source':<30s} {'n_test':>6s} {'n_up':>5s}  "
        f"{'mmr40':>6s} ± {'mmr40_std':<6s}  "
        f"{'raw40':>6s} ± {'raw40_std':<6s}"
    )
    for source, data in report["per_source"].items():
        cur = data["formulas"].get("current", {})
        mmr_mean = cur.get("mean", {}).get("mmr", {}).get("ndcg_at_40")
        mmr_std = cur.get("std", {}).get("mmr", {}).get("ndcg_at_40")
        raw_mean = cur.get("mean", {}).get("raw", {}).get("ndcg_at_40")
        raw_std = cur.get("std", {}).get("raw", {}).get("ndcg_at_40")
        if mmr_mean is None:
            continue
        print(
            f"  {source:<30s} {data['n_test']:>6d} {data['n_up']:>5d}  "
            f"{mmr_mean:>6.3f} ± {mmr_std:<6.3f}  "
            f"{raw_mean:>6.3f} ± {raw_std:<6.3f}"
        )

    if "final_queue" in report:
        fq: Any = report["final_queue"]
        print("\n=== Final queue (production pipeline, top 40) ===")
        for metric in metric_keys:
            m = fq["mean"]["mmr"][metric]
            s = fq["std"]["mmr"][metric]
            print(f"  {metric:15s} mmr {m:.3f} ± {s:.3f}")
        per_src: Any = fq.get("per_source", {})
        if per_src:
            print("\n  Per-source breakdown (ndcg_at_40):")
            for source, data in per_src.items():
                m = data["mean"]["mmr"]["ndcg_at_40"]
                s = data["std"]["mmr"]["ndcg_at_40"]
                n_test = data["n_test"]
                n_up = data["n_up"]
                print(f"  {source:<30s} {n_test:>6d} {n_up:>5d}  {m:>6.3f} ± {s:<6.3f}")


if __name__ == "__main__":
    main()
