"""Free offline controls for the preserved Decision-1/Strands evaluation.

Reads only the frozen folds and an immutable SQLite snapshot. No scoring, API,
model selection, deployment or writes to existing caches. New numeric evidence
and fresh content-bound control jobs are written under --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

UP = 2
WEIGHTS = (0.1, 0.2, 0.3, 0.5)


def ndcg40(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int((labels == UP).sum())
    if positives == 0:
        raise ValueError("NDCG needs at least one upvote")
    order = np.argsort(-scores, kind="stable")[:40]
    discounts = 1 / np.log2(np.arange(len(order)) + 2)
    ideal = (1 / np.log2(np.arange(min(40, positives)) + 2)).sum()
    return float(((labels[order] == UP) * discounts).sum() / ideal)


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    up = scores[labels == UP]
    other = scores[(labels == 0) | (labels == 1)]
    if not len(up) or not len(other):
        return None
    return float(
        (
            (up[:, None] > other[None, :]).sum()
            + 0.5 * (up[:, None] == other[None, :]).sum()
        )
        / (len(up) * len(other))
    )


def pct(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for x in np.unique(scores):
        mask = scores == x
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks / len(scores)


def request_hash(
    state: object, question: object, model_revision: str, dtype: str
) -> str:
    obj = {
        "state": state,
        "question": question,
        "model_revision": model_revision,
        "dtype": dtype,
    }
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def title_tokens(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.casefold()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_decision1 import checked_folds, prepare_strands_jobs, snapshot_popularity

    folds = checked_folds(a.work, None)
    paired_jobs_path = prepare_strands_jobs(a.work)
    jobs = [json.loads(x) for x in paired_jobs_path.read_text().splitlines()]
    stats = snapshot_popularity(a.snapshot, {c.id for f in folds for c in f.candidates})
    rng = np.random.default_rng(20261010)
    fold_rows: list[dict[str, object]] = []
    result: dict[str, object] = {
        "seed": 20261010,
        "null_draws": 500,
        "folds": fold_rows,
        "popularity_uses_snapshot_time_values_not_vote_time": True,
    }
    null = {str(w): [] for w in WEIGHTS}
    base_ndcg = []
    score_rows = []
    for f in folds:
        lab = np.asarray(f.labels)
        prod = np.asarray(f.production)
        points = np.asarray([stats[c.id][0] for c in f.candidates])
        comments = np.asarray([stats[c.id][1] for c in f.candidates])
        base = ndcg40(prod, lab)
        base_ndcg.append(base)
        profiles = [title_tokens(t) for t in f.profile_up + f.profile_down]
        near = []
        for c in f.candidates:
            t = title_tokens(c.title)
            near.append(any(len(t & p) / len(t | p) >= 0.5 for p in profiles if t | p))
        n = np.asarray(near)
        top = prod[np.argsort(-prod, kind="stable")[:40]]
        fold_rows.append(
            {
                "fold": f.fold,
                "candidates": len(lab),
                "production_auc": auc(prod, lab),
                "points_auc": auc(points, lab),
                "comments_auc": auc(comments, lab),
                "production_ndcg40": base,
                "points_ndcg40": ndcg40(points, lab),
                "near_title_count_jaccard_ge_half": int(n.sum()),
                "production_auc_excluding_near_titles": auc(prod[~n], lab[~n]),
                "top40_unique_production_scores": int(len(np.unique(top))),
            }
        )
        score_rows.append((pct(prod), lab))
    for _ in range(500):
        drawn = {str(w): [] for w in WEIGHTS}
        for (prod, lab), base in zip(score_rows, base_ndcg, strict=True):
            noise = rng.permutation(np.arange(1, len(prod) + 1)) / len(prod)
            for w in WEIGHTS:
                drawn[str(w)].append(ndcg40((1 - w) * prod + w * noise, lab) - base)
        for w in WEIGHTS:
            null[str(w)].append(float(np.mean(drawn[str(w)])))
    result["random_blend_ndcg_gain_null"] = {
        w: {
            "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "positive_fraction": float((np.asarray(v) > 0).mean()),
        }
        for w, v in null.items()
    }
    with (a.out / "empty-profile-content-bound-jobs.jsonl").open("w") as out:
        for j in jobs:
            if j["variant"] == "empty":
                out.write(json.dumps(j, sort_keys=True) + "\n")
    result["paired_jobs"] = len(jobs)
    result["stories_per_profile"] = len(jobs) // 2
    result["paired_jobs_path"] = str(paired_jobs_path)
    result["random_blend_controls_are_exploratory"] = True
    (a.out / "request-manifest.json").write_text(
        paired_jobs_path.with_name("manifest.json").read_text()
    )
    (a.out / "controls.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
