#!/usr/bin/env python3
"""Print a compact table from eval_ranker_variants.py JSON reports.

One row per variant/baseline: fold-mean raw metrics, a composite (mean of
AUC vs rest, MAP, NDCG@12, NDCG@40 and 1 - downvote share of the top 40),
and how many folds beat production on the composite. With --paired-metric
the per-fold differences against production are shown too. Several
reports can be passed; rows are prefixed with the report's file stem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    "auc_up_vs_rest",
    "auc_up_vs_down",
    "map",
    "ndcg_at_12",
    "ndcg_at_40",
    "up_recall_at_100",
    "known_downvote_fraction_at_40",
)
SHORT = {
    "auc_up_vs_rest": "auc_rest",
    "auc_up_vs_down": "auc_down",
    "map": "map",
    "ndcg_at_12": "ndcg12",
    "ndcg_at_40": "ndcg40",
    "up_recall_at_100": "rec100",
    "known_downvote_fraction_at_40": "down@40",
}


COMPOSITE = ("auc_up_vs_rest", "map", "ndcg_at_12", "ndcg_at_40")


def composite(metrics: dict[str, Any]) -> float | None:
    """Mean of the hill-climbing metrics; None if any is undefined."""
    values = [metrics.get(key) for key in COMPOSITE]
    down = metrics.get("known_downvote_fraction_at_40")
    if down is None or any(v is None for v in values):
        return None
    return (sum(values) + 1.0 - down) / (len(values) + 1)


def _fmt(value: float | None) -> str:
    return f"{value:9.3f}" if isinstance(value, (int, float)) else f"{'-':>9}"


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {**report.get("variants", {}), **report.get("baselines", {})}


def summarize(path: Path, paired_metric: str) -> list[str]:
    report = json.loads(path.read_text())
    rows = _rows(report)
    production = rows.get("production")
    lines = []
    base_composite = (
        [composite(f["raw"]) for f in production["per_fold"]] if production else []
    )
    for name, result in rows.items():
        mean = result["mean"]["raw"]
        folds = [fold["raw"] for fold in result["per_fold"]]
        comps = [composite(f) for f in folds]
        defined = [c for c in comps if c is not None]
        comp_mean = sum(defined) / len(defined) if defined else None
        wins = ""
        paired = ""
        if production is not None and name != "production":
            pairs = [
                (a, b)
                for a, b in zip(comps, base_composite, strict=True)
                if a is not None and b is not None
            ]
            wins = f"{sum(a > b for a, b in pairs)}/{len(pairs)}"
            if paired_metric:
                base = [f["raw"].get(paired_metric) for f in production["per_fold"]]
                paired = " ".join(
                    f"{a - b:+.3f}"
                    for a, b in zip((f.get(paired_metric) for f in folds), base)
                    if a is not None and b is not None
                )
        lines.append(
            f"{path.stem[:18]:18} {name:44}"
            + "".join(_fmt(mean.get(key)) for key in METRICS)
            + _fmt(comp_mean)
            + f"{wins:>6}   {paired}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--paired-metric",
        default="",
        help="Also show per-fold differences against production for this metric",
    )
    args = parser.parse_args()
    print(
        f"{'report':18} {'scorer':44}"
        + "".join(f"{SHORT[key]:>9}" for key in METRICS)
        + f"{'compos':>9}{'wins':>6}"
        + (f"   per-fold Δ{args.paired_metric}" if args.paired_metric else "")
    )
    for path in args.reports:
        for line in summarize(path, args.paired_metric):
            print(line)


if __name__ == "__main__":
    main()
