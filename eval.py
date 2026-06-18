"""Ranking evaluation with distractor injection. OFFLINE-ONLY.

Compares 3 score formulas via 5-fold stratified CV:
  soft:        P(up) + 0.5 * P(neutral)          [production]
  up_only:     P(up)
  hn_baseline: raw HN points (no SVM)

Each fold ranks the test fold stories AMONG ~3000 non-feedback distractor
stories (stories in the DB the user never voted on).  This mirrors the
production retrieval task: "find upvotable stories in a feed of ~3000."
"""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pipeline import _build_model, _make_labels_binary, _model_predict_up

from database import Database, Story
from pipeline import (
    Config,
    RankedStory,
    _augment_features,
    mmr_filter,
)

MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|256"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


def _db_sha256(db_path: str) -> str:
    return hashlib.sha256(Path(db_path).read_bytes()).hexdigest()[:16]


def _compute_upvote_metrics(
    ranked: list[RankedStory],
    test_stories: list[Story],
    test_actions: np.ndarray,
    full_ranked_order_ids: list[int],
) -> dict:
    story_to_action = {ts.id: int(act) for ts, act in zip(test_stories, test_actions)}
    ranked_actions = [story_to_action.get(rs.story.id, 0) for rs in ranked]
    total_upvotes = sum(1 for act in test_actions if act == 2)

    def _prec_rec_ndcg(k: int) -> tuple[float, float, float]:
        slice_actions = ranked_actions[:k]
        up_count = sum(1 for act in slice_actions if act == 2)
        precision = up_count / k if k > 0 else 0.0
        recall = up_count / total_upvotes if total_upvotes > 0 else 0.0

        dcg = sum(
            1.0 / math.log2(p + 2) for p, act in enumerate(slice_actions) if act == 2
        )
        ideal_count = min(k, total_upvotes)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return precision, recall, ndcg

    p5, r5, n5 = _prec_rec_ndcg(5)
    p10, r10, n10 = _prec_rec_ndcg(10)
    p20, r20, n20 = _prec_rec_ndcg(20)
    p40, r40, n40 = _prec_rec_ndcg(40)

    mrr = 0.0
    for pos, act in enumerate(ranked_actions):
        if act == 2:
            mrr = 1.0 / (pos + 1)
            break

    upvote_ranks = [
        pos
        for pos, sid in enumerate(full_ranked_order_ids)
        if story_to_action.get(sid, 0) == 2
    ]
    median_rank = float(np.median(upvote_ranks)) if upvote_ranks else 0.0

    return {
        "ndcg_at_5": n5,
        "ndcg_at_10": n10,
        "ndcg_at_20": n20,
        "ndcg_at_40": n40,
        "precision_at_5": p5,
        "precision_at_10": p10,
        "precision_at_20": p20,
        "precision_at_40": p40,
        "recall_at_10": r10,
        "recall_at_40": r40,
        "mrr": mrr,
        "median_rank": median_rank,
    }


def _evaluate_fold(
    scores: np.ndarray,
    all_stories: list[Story],
    all_emb: np.ndarray,
    test_stories: list[Story],
    test_actions: np.ndarray,
    mmr_threshold: float = 0.85,
    mmr_limit: int = 40,
) -> dict:
    order = np.argsort(-scores)
    ranked = [
        RankedStory(story=all_stories[i], score=float(scores[i]), best_match_title="")
        for i in order
    ]
    full_ranked_order_ids = [all_stories[i].id for i in order]
    emb_map = {all_stories[i].id: all_emb[i] for i in range(len(all_stories))}

    top40 = mmr_filter(ranked, emb_map, threshold=mmr_threshold, limit=mmr_limit)
    return _compute_upvote_metrics(
        top40, test_stories, test_actions, full_ranked_order_ids
    )


def main() -> None:
    config = Config.load()
    db = Database(config.db_path)

    # ------------------------------------------------------------------ #
    # 1. Load feedback stories
    # ------------------------------------------------------------------ #
    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training()
    fb_labels = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    print(f"Feedback: {len(fb_stories)} rows ({Counter(fb_labels)})")

    # Load embeddings for feedback stories
    cached = db.get_embeddings_batch([s.id for s in fb_stories], MODEL_VERSION)
    fb_emb = np.array(
        [cached.get(s.id, np.zeros(384, dtype=np.float32)) for s in fb_stories],
        dtype=np.float32,
    )

    valid = np.array([s.id in cached for s in fb_stories], dtype=bool)
    if not valid.all():
        print(
            f"Warning: {(~valid).sum()} feedback stories missing cached embeddings; excluded."
        )
        fb_stories = [fb_stories[i] for i in range(len(fb_stories)) if valid[i]]
        fb_labels = fb_labels[valid]
        fb_vote_times = fb_vote_times[valid]
        fb_emb = fb_emb[valid]

    # Pre-calculate metadata features
    fb_scores = np.array([s.score for s in fb_stories])
    fb_ages = np.array(
        [float(vt) - max(s.time, 1) for vt, s in zip(fb_vote_times, fb_stories)]
    )

    # ------------------------------------------------------------------ #
    # 2. Load distractor stories (non-feedback stories with embeddings)
    # ------------------------------------------------------------------ #
    fb_ids = {s.id for s in fb_stories}
    cursor = db.conn.execute(
        """
        SELECT s.id, s.title, s.url, s.score, s.time, s.text_content, s.source,
               s.comment_count, s.discussion_url
        FROM stories s
        INNER JOIN embeddings e ON e.story_id = s.id
        WHERE s.text_content != ''
          AND e.model_version = ?
          AND s.id NOT IN (SELECT f.story_id FROM feedback f)
        """,
        (MODEL_VERSION,),
    )
    dist_stories = [
        Story(
            id=row[0],
            title=row[1],
            url=row[2],
            score=row[3],
            time=row[4],
            text_content=row[5],
            source=row[6],
            comment_count=row[7],
            discussion_url=row[8],
        )
        for row in cursor.fetchall()
        if row[0] not in fb_ids
    ]
    print(f"Distractors: {len(dist_stories)} stories")

    if not dist_stories:
        print("ERROR: No distractor stories found. Cannot evaluate.")
        db.close()
        return

    dist_cached = db.get_embeddings_batch([s.id for s in dist_stories], MODEL_VERSION)
    dist_emb = np.array([dist_cached[s.id] for s in dist_stories], dtype=np.float32)

    now_ts = time.time()
    dist_scores = np.array([s.score for s in dist_stories], dtype=np.float32)
    dist_ages = np.array([now_ts - s.time for s in dist_stories], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 3. Cross-validation
    # ------------------------------------------------------------------ #
    y = fb_labels
    formulas = ["soft", "up_only", "hn_baseline"]
    results: dict[str, list[dict]] = {f: [] for f in formulas}

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
            np.zeros((len(y), 1)), y
        )
    )

    for fold_idx, (train_pos, test_pos) in enumerate(folds):
        # --- Train fold ---
        fb_train_emb = fb_emb[train_pos]
        y_train = y[train_pos]

        fb_train_scores = fb_scores[train_pos]
        fb_train_ages = fb_ages[train_pos]

        up_mask = y_train == 2
        down_mask = y_train == 0
        fb_up_train = fb_train_emb[up_mask]
        fb_down_train = fb_train_emb[down_mask]
        n_up = up_mask.sum()
        n_down = down_mask.sum()

        mean_up = fb_up_train.mean(axis=0) if n_up else np.zeros(384, dtype=np.float32)
        mean_down = (
            fb_down_train.mean(axis=0) if n_down else np.zeros(384, dtype=np.float32)
        )

        # Train personalization features via LOOCV
        fb_sim_up = fb_train_emb @ mean_up
        fb_sim_down = fb_train_emb @ mean_down

        if n_up > 0:
            if n_up > 1:
                fb_sim_up[up_mask] = (n_up * fb_sim_up[up_mask] - 1.0) / (n_up - 1)
            else:
                fb_sim_up[up_mask] = 0.0

            sim_up_mat = fb_train_emb @ fb_up_train.T
            if n_up > 1:
                train_up_positions = np.where(up_mask)[0]
                for i, tp in enumerate(train_up_positions):
                    sim_up_mat[tp, i] = -1.0
            fb_closest_up = np.max(sim_up_mat, axis=1)
        else:
            fb_closest_up = np.zeros(len(fb_train_emb))

        if n_down > 0:
            if n_down > 1:
                fb_sim_down[down_mask] = (n_down * fb_sim_down[down_mask] - 1.0) / (
                    n_down - 1
                )
            else:
                fb_sim_down[down_mask] = 0.0

            sim_down_mat = fb_train_emb @ fb_down_train.T
            if n_down > 1:
                train_down_positions = np.where(down_mask)[0]
                for i, tp in enumerate(train_down_positions):
                    sim_down_mat[tp, i] = -1.0
            fb_closest_down = np.max(sim_down_mat, axis=1)
        else:
            fb_closest_down = np.zeros(len(fb_train_emb))

        # --- Build train features ---
        X_train = _augment_features(
            fb_train_emb,
            fb_train_scores,
            fb_train_ages,
            sim_to_upvoted=fb_sim_up,
            sim_to_downvoted=fb_sim_down,
            closest_upvoted=fb_closest_up,
            closest_downvoted=fb_closest_down,
            use_raw_embeddings=config.model.use_raw_embeddings,
        )

        counts = Counter(y_train)
        weights = np.array(
            [len(y_train) / (3 * counts[c]) for c in y_train], dtype=np.float64
        )

        emb_dim = fb_emb.shape[1] if config.model.use_raw_embeddings else 0
        scaler = StandardScaler()
        X_train_meta_scaled = scaler.fit_transform(X_train[:, emb_dim:])

        X_train_scaled = np.hstack([X_train[:, :emb_dim], X_train_meta_scaled])

        model_name = config.model.model_name
        model = _build_model(model_name, config.model)
        if model_name == "svm":
            model.fit(X_train_scaled, y_train, sample_weight=weights)
        elif model_name == "lgbm_rank":
            model.fit(X_train_scaled, y_train, group=[len(X_train_scaled)])
        else:
            model.fit(X_train_scaled, _make_labels_binary(y_train))

        # --- Test fold ---
        fb_test_emb = fb_emb[test_pos]
        y_test = y[test_pos]
        fb_test_scores = fb_scores[test_pos]
        fb_test_ages = fb_ages[test_pos]
        test_stories = [fb_stories[i] for i in test_pos]

        # Test personalization
        test_sim_up = fb_test_emb @ mean_up
        test_sim_down = fb_test_emb @ mean_down
        test_closest_up = (
            np.max(fb_test_emb @ fb_up_train.T, axis=1)
            if n_up
            else np.zeros(len(fb_test_emb))
        )
        test_closest_down = (
            np.max(fb_test_emb @ fb_down_train.T, axis=1)
            if n_down
            else np.zeros(len(fb_test_emb))
        )

        # --- Distractor personalization ---
        dist_sim_up = dist_emb @ mean_up
        dist_sim_down = dist_emb @ mean_down
        dist_closest_up = (
            np.max(dist_emb @ fb_up_train.T, axis=1)
            if n_up
            else np.zeros(len(dist_emb))
        )
        dist_closest_down = (
            np.max(dist_emb @ fb_down_train.T, axis=1)
            if n_down
            else np.zeros(len(dist_emb))
        )

        # --- Build all candidate features ---
        X_test = _augment_features(
            fb_test_emb,
            fb_test_scores,
            fb_test_ages,
            sim_to_upvoted=test_sim_up,
            sim_to_downvoted=test_sim_down,
            closest_upvoted=test_closest_up,
            closest_downvoted=test_closest_down,
            use_raw_embeddings=config.model.use_raw_embeddings,
        )
        X_dist = _augment_features(
            dist_emb,
            dist_scores,
            dist_ages,
            sim_to_upvoted=dist_sim_up,
            sim_to_downvoted=dist_sim_down,
            closest_upvoted=dist_closest_up,
            closest_downvoted=dist_closest_down,
            use_raw_embeddings=config.model.use_raw_embeddings,
        )

        X_all = np.concatenate([X_test, X_dist], axis=0)
        X_all_meta_scaled = scaler.transform(X_all[:, emb_dim:])
        X_all_scaled = np.hstack([X_all[:, :emb_dim], X_all_meta_scaled])

        scores_model = _model_predict_up(model, model_name, X_all_scaled)

        all_stories = test_stories + dist_stories
        all_emb = np.concatenate([fb_test_emb, dist_emb], axis=0)

        for formula in formulas:
            if formula in ("soft", "up_only"):
                scores = scores_model
            elif formula == "hn_baseline":
                scores = np.concatenate([fb_test_scores, dist_scores])

            results[formula].append(
                _evaluate_fold(
                    scores.astype(np.float32),
                    all_stories,
                    all_emb,
                    test_stories,
                    y_test,
                )
            )

        print(f"Fold {fold_idx + 1}/5 done")

    # ------------------------------------------------------------------ #
    # 4. Aggregate & report
    # ------------------------------------------------------------------ #
    metric_keys = (
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_20",
        "ndcg_at_40",
        "precision_at_5",
        "precision_at_10",
        "precision_at_20",
        "precision_at_40",
        "recall_at_10",
        "recall_at_40",
        "mrr",
        "median_rank",
    )

    config_block = {
        "split": "5-fold-stratified",
        "random_state": 0,
        "n_feedback": int(len(fb_labels)),
        "n_distractors": int(len(dist_stories)),
        "n_folds": 5,
        "mmr_threshold": 0.85,
        "mmr_limit": 40,
        "relevance_grade": "up=1, neutral=0, down=0",
        "db_sha256": _db_sha256(config.db_path),
    }

    mean_dict: dict[str, dict[str, float]] = {}
    std_dict: dict[str, dict[str, float]] = {}
    formulas_block: dict[str, dict[str, object]] = {}
    for f, fold_results in results.items():
        fm = {k: float(np.mean([r[k] for r in fold_results])) for k in metric_keys}
        fs = {k: float(np.std([r[k] for r in fold_results])) for k in metric_keys}
        mean_dict[f] = fm
        std_dict[f] = fs
        formulas_block[f] = {"mean": fm, "std": fs, "per_fold": fold_results}

    report = {"config": config_block, "formulas": formulas_block}

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWritten {REPORT_PATH}")

    for metric in (
        "ndcg_at_10",
        "precision_at_10",
        "precision_at_40",
        "recall_at_40",
        "mrr",
        "median_rank",
    ):
        print(f"\n{metric} by formula (mean ± std):")
        for f in formulas:
            m, s = mean_dict[f][metric], std_dict[f][metric]
            print(f"  {f:12s} {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()
