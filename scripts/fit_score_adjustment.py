#!/usr/bin/env python3
"""Can hand orderings teach a small correction on top of the model score?

Fits final = production + w · features on the pairs from
``calibrate_rankings.py`` (pairwise logistic regression, strong L2) and
scores it leave-one-batch-out: each batch is predicted by weights fitted
on the other batches only. Compares pairwise agreement with the user
against production alone. Reads the snapshot read-only.

    uv run python scripts/fit_score_adjustment.py --db SNAPSHOT
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_rankings import DEFAULT_LOG  # noqa: E402

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "production": (),
    "+challenger": ("challenger",),
    "+points": ("log_points",),
    "+comments": ("log_comments",),
    "+age": ("log_age_days",),
    "+hn source": ("is_hn",),
    "+all": ("challenger", "log_points", "log_comments", "log_age_days", "is_hn"),
}


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def load_batches(log: Path, db_path: Path) -> list[list[dict[str, float]]]:
    """Per batch, best-first rows of features for each ordered story."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    now = time.time()
    batches = []
    for line in log.read_text().splitlines():
        record = json.loads(line)
        rows = []
        for i in record["order"]:
            points, comments, posted, source = db.execute(
                "SELECT score, comment_count, time, source FROM stories WHERE id = ?",
                (record["story_ids"][i],),
            ).fetchone()
            rows.append(
                {
                    "production": _logit(record["production"][i]),
                    "challenger": _logit(record["challenger"][i]),
                    "log_points": math.log1p(max(points or 0, 0)),
                    "log_comments": math.log1p(comments or 0),
                    "log_age_days": math.log1p(max(now - posted, 0) / 86400),
                    "is_hn": float(source in ("hn", "ch_seed")),
                }
            )
        batches.append(rows)
    return batches


def _pairs(
    batch: list[dict[str, float]], names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Feature differences (better − worse) and production's difference."""
    diffs, offsets = [], []
    for a, b in combinations(range(len(batch)), 2):  # a is ranked above b
        diffs.append([batch[a][n] - batch[b][n] for n in names])
        offsets.append(batch[a]["production"] - batch[b]["production"])
    return np.array(diffs, dtype=float).reshape(len(diffs), len(names)), np.array(
        offsets
    )


def fit(
    batches: list[list[dict[str, float]]], names: tuple[str, ...], c: float
) -> np.ndarray:
    """Weights w for production_scale · production + w · features."""
    x, y = [], []
    for batch in batches:
        diffs, offsets = _pairs(batch, names)
        for d, o in zip(diffs, offsets, strict=True):
            row = [o, *d]
            x += [row, [-v for v in row]]  # symmetric: both orientations
            y += [1, 0]
    model = LogisticRegression(C=c, fit_intercept=False)
    model.fit(np.array(x), np.array(y))
    return model.coef_[0]


def agreement(
    batch: list[dict[str, float]], names: tuple[str, ...], w: np.ndarray
) -> tuple[int, int]:
    diffs, offsets = _pairs(batch, names)
    margin = w[0] * offsets + diffs @ w[1:] if names else offsets
    return int((margin > 0).sum()), len(margin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--c", type=float, default=0.3, help="Inverse L2 strength")
    args = parser.parse_args()
    batches = load_batches(args.log, args.db)
    print(f"{len(batches)} batches, leave-one-batch-out pairwise agreement:")
    for label, names in FEATURE_SETS.items():
        agree = total = 0
        for held in range(len(batches)):
            train = batches[:held] + batches[held + 1 :]
            w = fit(train, names, args.c) if names else np.ones(1)
            a, t = agreement(batches[held], names, w)
            agree, total = agree + a, total + t
        weights = fit(batches, names, args.c) if names else np.ones(1)
        shown = ", ".join(
            f"{n} {v:+.2f}"
            for n, v in zip(("production", *names), weights, strict=True)
        )
        print(
            f"  {label:12} {agree}/{total} ({agree / total:.0%})   all-data w: {shown}"
        )


if __name__ == "__main__":
    main()
