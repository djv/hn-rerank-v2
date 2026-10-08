#!/usr/bin/env python3
"""Paired story-level statistics for eval runs saved with ``--dump-scores``.

Each run is ``NAME=DUMP[,DUMP...]`` (e.g. the dev and fresh dumps of one
embedding setup); the first run is the baseline and every other run is a
challenger. Dumps are paired by position, and each pair must hold the same
folds with the same story IDs, so both runs rank exactly the same stories.

Per fold: AUC upvote vs rest (ties get half credit), average precision and
P@12 (upvotes among the first 12 cards, stable order as in the eval), over
judged stories. A run's metric is the mean over its folds, as in the eval.
For each challenger minus the baseline, per dump and pooled over all dumps:

- bootstrap: stratified within each fold (resample upvotes and non-upvotes
  with replacement, the same draw for both runs); 95% percentile interval of
  the mean difference and the share of draws above zero.
- block bootstrap: resample whole folds (between-block variation; coarse
  with few folds).
- sign-flip: exact permutation test of the per-fold differences.
- DeLong: variance of the difference of two correlated AUCs per fold,
  combined over folds as a mean (folds' test sets are disjoint within a
  dump; pooling dev and fresh may count a story twice). Holm-corrected over
  the challengers.

Reads only the JSON dumps.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

UP = 2
TOP_K = 12


@dataclass(frozen=True)
class Fold:
    """One fold's judged stories: upvote flags and each run's scores."""

    key: str
    positive: NDArray[np.bool_]
    scores: tuple[NDArray[np.float64], NDArray[np.float64]]


def _auc(scores: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    ranks = stats.rankdata(scores)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _average_precision(
    scores: NDArray[np.float64], positive: NDArray[np.bool_]
) -> float:
    hits = positive[np.argsort(-scores, kind="stable")]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def _precision_at_k(scores: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
    top = positive[np.argsort(-scores, kind="stable")][:TOP_K]
    return float(top.mean())


METRICS = {"AUC": _auc, "AP": _average_precision, "P@12": _precision_at_k}


def _delong(fold: Fold) -> tuple[float, float]:
    """Difference of two correlated AUCs (run 2 - run 1) and its variance."""
    pos, neg = fold.positive, ~fold.positive
    v10, v01 = [], []
    for scores in fold.scores:
        x, y = scores[pos], scores[neg]
        # psi(x, y) = 1 if x > y, 1/2 if tied: per-positive and per-negative means.
        cmp = (x[:, None] > y[None, :]) + 0.5 * (x[:, None] == y[None, :])
        v10.append(cmp.mean(axis=1))
        v01.append(cmp.mean(axis=0))
    s10 = np.cov(np.vstack(v10))
    s01 = np.cov(np.vstack(v01))
    contrast = np.array([-1.0, 1.0])
    variance = float(
        contrast @ s10 @ contrast / pos.sum() + contrast @ s01 @ contrast / neg.sum()
    )
    return float(v10[1].mean() - v10[0].mean()), variance


def _load(
    path: Path, variant: str
) -> list[tuple[int, list[int], list[int], list[float]]]:
    data = json.loads(path.read_text())
    folds = []
    for fold in data["folds"]:
        if variant not in fold["scores"]:
            raise SystemExit(f"{path}: no scores for variant {variant!r}")
        folds.append(
            (fold["fold"], fold["ids"], fold["labels"], fold["scores"][variant])
        )
    return folds


def _pair(base: Path, other: Path, variant: str, prefix: str) -> list[Fold]:
    folds = []
    for (fa, ids_a, labels, sa), (fb, ids_b, _labels_b, sb) in zip(
        _load(base, variant), _load(other, variant), strict=True
    ):
        if fa != fb or ids_a != ids_b:
            raise SystemExit(f"{base} and {other} rank different stories in fold {fa}")
        label = np.array(labels)
        judged = label >= 0
        positive = label[judged] == UP
        if positive.all() or not positive.any():
            continue
        folds.append(
            Fold(
                f"{prefix}{fa}",
                positive,
                (np.array(sa)[judged], np.array(sb)[judged]),
            )
        )
    return folds


def _per_fold(folds: list[Fold], metric: str) -> NDArray[np.float64]:
    fn = METRICS[metric]
    return np.array(
        [[fn(s, f.positive) for s in f.scores] for f in folds], dtype=np.float64
    )


def _bootstrap(
    folds: list[Fold], metric: str, draws: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    fn = METRICS[metric]
    out = np.empty(draws)
    groups = [(np.flatnonzero(f.positive), np.flatnonzero(~f.positive)) for f in folds]
    for d in range(draws):
        deltas = []
        for fold, (pos, neg) in zip(folds, groups, strict=True):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            positive = fold.positive[idx]
            a, b = (fn(s[idx], positive) for s in fold.scores)
            deltas.append(b - a)
        out[d] = np.mean(deltas)
    return out


def _sign_flip_p(deltas: NDArray[np.float64]) -> float:
    if len(deltas) > 16:
        raise SystemExit("sign-flip test enumerates 2^folds; use at most 16 folds")
    observed = abs(deltas.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(deltas))))
    return float((np.abs((signs * deltas).mean(axis=1)) >= observed - 1e-12).mean())


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def _report(
    folds: list[Fold], draws: int, block_draws: int, rng: np.random.Generator
) -> dict[str, Any]:
    row: dict[str, Any] = {"folds": len(folds)}
    for metric in METRICS:
        values = _per_fold(folds, metric)
        deltas = values[:, 1] - values[:, 0]
        boot = _bootstrap(folds, metric, draws, rng)
        blocks = rng.choice(len(deltas), size=(block_draws, len(deltas)))
        block_means = deltas[blocks].mean(axis=1)
        row[metric] = {
            "base": float(values[:, 0].mean()),
            "delta": float(deltas.mean()),
            "boot_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "boot_p_gt0": float((boot > 0).mean()),
            "block_ci": np.percentile(block_means, [2.5, 97.5]).tolist(),
            "sign_flip_p": _sign_flip_p(deltas),
            "folds_better": int((deltas > 0).sum()),
            "folds_worse": int((deltas < 0).sum()),
        }
    delong = [_delong(f) for f in folds]
    delta = float(np.mean([d for d, _v in delong]))
    se = float(np.sqrt(sum(v for _d, v in delong)) / len(delong))
    row["delong"] = {
        "delta": delta,
        "se": se,
        "p": float(2 * stats.norm.sf(abs(delta) / se)) if se > 0 else 1.0,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", help="NAME=DUMP[,DUMP...]; first is the baseline"
    )
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--block-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="Also write the results here")
    args = parser.parse_args()

    runs = []
    for spec in args.runs:
        name, sep, paths = spec.partition("=")
        if not sep:
            parser.error(f"run {spec!r} is not NAME=DUMP[,DUMP...]")
        runs.append((name, [Path(p) for p in paths.split(",")]))
    base_name, base_paths = runs[0]
    if any(len(paths) != len(base_paths) for _name, paths in runs):
        parser.error("every run needs as many dumps as the baseline")
    scopes = [p.stem for p in base_paths]
    if len(set(scopes)) != len(scopes) or "pooled" in scopes:
        parser.error("baseline dump file names must have unique, non-pooled stems")
    results: dict[str, Any] = {}
    for variant in args.variant or ["production"]:
        print(f"\n== {variant}: challengers minus {base_name}")
        rows: dict[str, dict[str, Any]] = {}
        for name, paths in runs[1:]:
            per_dump = [
                _pair(b, o, variant, f"{scope}:")
                for b, o, scope in zip(base_paths, paths, scopes, strict=True)
            ]
            rng = np.random.default_rng(args.seed)
            rows[name] = {
                scope: _report(folds, args.boot, args.block_boot, rng)
                for scope, folds in zip(scopes, per_dump, strict=True)
            }
            if len(per_dump) > 1:
                rows[name]["pooled"] = _report(
                    [f for folds in per_dump for f in folds],
                    args.boot,
                    args.block_boot,
                    rng,
                )
        scope_names = list(next(iter(rows.values())))
        for scope in scope_names:
            holm = _holm([rows[name][scope]["delong"]["p"] for name in rows])
            for name, adjusted in zip(rows, holm, strict=True):
                rows[name][scope]["delong"]["p_holm"] = adjusted
        for name, by_scope in rows.items():
            for scope, row in by_scope.items():
                parts = []
                for metric in METRICS:
                    m = row[metric]
                    lo, hi = m["boot_ci"]
                    parts.append(
                        f"{metric} {m['base']:.3f}{m['delta']:+.4f} "
                        f"[{lo:+.4f},{hi:+.4f}] P>0 {m['boot_p_gt0']:.2f}"
                    )
                auc = row["AUC"]
                blo, bhi = auc["block_ci"]
                d = row["delong"]
                print(f"{name:>12} {scope:>6} n={row['folds']:2d}  " + "  ".join(parts))
                print(
                    f"{'':>12} {'':>6}        AUC folds +{auc['folds_better']}/-{auc['folds_worse']}"
                    f"  block CI [{blo:+.4f},{bhi:+.4f}]  sign-flip p {auc['sign_flip_p']:.3f}"
                    f"  DeLong p {d['p']:.3f} (Holm {d['p_holm']:.3f})"
                )
        results[variant] = rows
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
