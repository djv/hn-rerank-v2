#!/usr/bin/env python3
"""Grid search over (C, gamma) for the production-matching 3-class SVC.

Reuses data loading and fold construction from
``scripts/eval_ranker_variants.py`` to keep methodology consistent. For
each (C, gamma) combination, fits the SVM on each of 5 folds and
reports raw NDCG@40. All 30 combos run in a single process.

Output: a JSON report with mean/std NDCG@40 for each (C, gamma) combo,
ranked by mean raw NDCG@40.

Wall-clock: ~5 min for the full 30-combo x 5-fold grid on CPU.

The "current" production hyperparams are C=0.2, gamma=0.03; this sweep
is to check whether the post-4-binary-source optimum has shifted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database  # noqa: E402
from pipeline import Config  # noqa: E402
from scripts.eval_ranker_variants import (  # noqa: E402
    FoldData,
    _load_production_candidates,
    _make_fold,
    _scores_margin_3class,
    _metrics,
)

GRID_C: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
GRID_GAMMA: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", default="svm_hparam_sweep.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--window-days", type=int, default=30)
    args = parser.parse_args()

    config = Config.load(args.config)
    db = Database(config.db_path)
    user = db.get_user_by_token("default")
    if user is None:
        raise RuntimeError("Missing default user token")

    window_days = args.window_days
    eval_config = replace(config, days=window_days)
    candidates, cand_emb = _load_production_candidates(db, eval_config, user.id)
    cand_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
    fb_stories, fb_labels, fb_vote_times = db.get_feedback_for_training(user_id=user.id)
    all_y = np.array(fb_labels, dtype=int)
    fb_vote_times = np.array(fb_vote_times, dtype=np.float64)
    fb_to_cand = np.array([cand_id_to_idx.get(s.id, -1) for s in fb_stories])
    valid_mask = fb_to_cand >= 0
    valid_positions = np.where(valid_mask)[0]
    y = all_y[valid_mask]

    if len(set(y)) < 3:
        raise RuntimeError("Need all three labels for stratified evaluation")

    # Field-level arrays are placeholders; needs_field=False so they are
    # never read.
    cand_field_emb = cand_emb
    cand_field_parts = np.empty((len(candidates), 4, 384), dtype=np.float32)
    fb_field_emb = cand_emb
    fb_field_parts = np.empty((len(fb_stories), 4, 384), dtype=np.float32)

    splits = list(
        StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0).split(
            np.zeros((len(y), 1)), y
        )
    )

    print(f"candidates={len(candidates)} valid_feedback={len(y)} folds={args.folds}")
    print(f"C grid: {GRID_C}")
    print(f"gamma grid: {GRID_GAMMA}")

    folds: list[FoldData] = []
    for fold_idx, (train_pos, test_pos) in enumerate(splits, start=1):
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
            y,
            valid_positions,
            train_pos,
            test_pos,
            config,
            needs_field=False,
        )
        folds.append(fold)
        print(f"fold {fold_idx}/{args.folds} prepared")

    results: dict[tuple[float, float], list[dict]] = {}
    total_combos = len(GRID_C) * len(GRID_GAMMA)
    combo_idx = 0
    for c in GRID_C:
        for gamma in GRID_GAMMA:
            combo_idx += 1
            t0 = time.time()
            cfg = replace(config.model, svm_c=c, svm_gamma=gamma)
            cfg_full = replace(config, model=cfg)
            fold_metrics: list[dict] = []
            for fold in folds:
                scores, probs = _scores_margin_3class(
                    fold, cfg_full, textsplit=False, mode="up"
                )
                m = _metrics(scores, fold, cfg_full, probs)
                fold_metrics.append(m)
            results[(c, gamma)] = fold_metrics
            elapsed = time.time() - t0
            mean_ndcg40 = float(np.mean([r["raw"]["ndcg_at_40"] for r in fold_metrics]))
            std_ndcg40 = float(np.std([r["raw"]["ndcg_at_40"] for r in fold_metrics]))
            mean_mmr_ndcg40 = float(
                np.mean([r["mmr"]["ndcg_at_40"] for r in fold_metrics])
            )
            mean_map = float(np.mean([r["raw"]["map"] for r in fold_metrics]))
            med_rank = float(np.mean([r["raw"]["median_rank"] for r in fold_metrics]))
            print(
                f"  [{combo_idx}/{total_combos}] "
                f"C={c:.2f} gamma={gamma:.3f} "
                f"raw_ndcg@40={mean_ndcg40:.3f} ± {std_ndcg40:.3f} "
                f"mmr_ndcg@40={mean_mmr_ndcg40:.3f} "
                f"raw_map={mean_map:.3f} med_rank={med_rank:.0f}  "
                f"({elapsed:.1f}s)"
            )

    sweep_rows: list[dict] = []
    for c in GRID_C:
        for gamma in GRID_GAMMA:
            rows = results[(c, gamma)]
            raw = [r["raw"] for r in rows]
            mmr = [r["mmr"] for r in rows]
            sweep_rows.append(
                {
                    "svm_c": c,
                    "svm_gamma": gamma,
                    "n_folds": len(rows),
                    "mean_raw_ndcg_at_40": float(
                        np.mean([r["ndcg_at_40"] for r in raw])
                    ),
                    "std_raw_ndcg_at_40": float(np.std([r["ndcg_at_40"] for r in raw])),
                    "mean_raw_ndcg_at_100": float(
                        np.mean([r["ndcg_at_100"] for r in raw])
                    ),
                    "mean_raw_map": float(np.mean([r["map"] for r in raw])),
                    "mean_raw_median_rank": float(
                        np.mean([r["median_rank"] for r in raw])
                    ),
                    "mean_mmr_ndcg_at_40": float(
                        np.mean([r["ndcg_at_40"] for r in mmr])
                    ),
                    "std_mmr_ndcg_at_40": float(np.std([r["ndcg_at_40"] for r in mmr])),
                }
            )
    sweep_rows.sort(key=lambda r: -r["mean_raw_ndcg_at_40"])

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    report = {
        "config": {
            "split": f"{args.folds}-fold-stratified",
            "window_days": window_days,
            "user_id": user.id,
            "n_candidates": len(candidates),
            "n_feedback_valid": len(y),
            "labels": label_counts,
            "kernel": config.model.svm_kernel,
            "knn_k": config.model.knn_k,
        },
        "grid_c": list(GRID_C),
        "grid_gamma": list(GRID_GAMMA),
        "current_production": {
            "svm_c": config.model.svm_c,
            "svm_gamma": config.model.svm_gamma,
        },
        "ranked": sweep_rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")
    print("\nTop 5 by raw ndcg@40:")
    for row in sweep_rows[:5]:
        print(
            f"  C={row['svm_c']:.2f} gamma={row['svm_gamma']:.3f}  "
            f"raw_ndcg@40={row['mean_raw_ndcg_at_40']:.3f} ± "
            f"{row['std_raw_ndcg_at_40']:.3f}  "
            f"raw_map={row['mean_raw_map']:.3f}  "
            f"med_rank={row['mean_raw_median_rank']:.0f}"
        )
    cur = next(
        (
            r
            for r in sweep_rows
            if r["svm_c"] == config.model.svm_c
            and r["svm_gamma"] == config.model.svm_gamma
        ),
        None,
    )
    if cur is not None:
        print(
            f"\nCurrent production (C={config.model.svm_c}, "
            f"gamma={config.model.svm_gamma}): "
            f"raw_ndcg@40={cur['mean_raw_ndcg_at_40']:.3f} ± "
            f"{cur['std_raw_ndcg_at_40']:.3f}"
        )


if __name__ == "__main__":
    main()
