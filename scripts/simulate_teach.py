#!/usr/bin/env python3
"""Paired, retrospective Teach selection experiment on frozen feedback."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database
from pipeline import Config
from pipeline.linear_blend import percentile_scores
from scripts.eval_ranker_variants import (
    FoldData,
    _embedding_text_hashes,
    _make_fold,
    _production_scores,
    _required_labels_for_variants,
    _temporal_splits,
    _validate_splits,
    _validated_embeddings,
    frozen_database,
)

Rule = Literal["random", "entropy", "near_cutoff", "up_margin", "novel", "mix"]
RULES: tuple[Rule, ...] = (
    "random",
    "entropy",
    "near_cutoff",
    "up_margin",
    "novel",
    "mix",
)
Scorer = Callable[[FoldData], tuple[np.ndarray, np.ndarray | None]]
METRICS = ("auc", "precision_at_20", "up_rate_top_decile")


@dataclass(frozen=True)
class Metrics:
    auc: float | None
    precision_at_20: float
    up_rate_top_decile: float


@dataclass(frozen=True)
class Checkpoint:
    budget: int
    labels_added: int
    attained: bool
    story_ids: list[int]
    labels: list[int]
    scores: list[float]
    metrics: Metrics


@dataclass(frozen=True)
class Run:
    fold: int
    rule: Rule
    seed: int
    picked_ids: list[int]
    checkpoints: list[Checkpoint]


@dataclass(frozen=True)
class FoldAudit:
    fold: int
    train_size: int
    heldout_before_isolation: int
    heldout_after_isolation: int
    pool_ids: list[int]
    evaluation_ids: list[int]
    pool_size: int
    evaluation_size: int
    training_cutoff: float
    test_start: float
    test_end: float


@dataclass(frozen=True)
class Interval:
    difference: float | None
    low: float | None
    high: float | None
    defined_folds: int


@dataclass(frozen=True)
class Summary:
    rule: Rule
    budget: int
    mean: dict[str, float | None]
    against_random: dict[str, Interval]
    attained_folds: int


@dataclass(frozen=True)
class Pair:
    labels: np.ndarray
    scores: list[np.ndarray]
    random_scores: list[np.ndarray]


def metrics(labels: np.ndarray, scores: np.ndarray) -> Metrics:
    """Up versus rest; short sets use all available cards for precision."""
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Metrics require nonempty aligned label/score vectors")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    up = labels == 2
    order = np.argsort(-scores, kind="stable")
    return Metrics(
        float(roc_auc_score(up, scores)) if len(np.unique(up)) == 2 else None,
        float(up[order[:20]].mean()),
        float(up[order[: max(1, math.ceil(len(labels) / 10))]].mean()),
    )


def select(
    rule: Rule,
    scores: np.ndarray,
    probabilities: np.ndarray | None,
    embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
    mix_base: Rule = "near_cutoff",
) -> np.ndarray:
    """Return unique positional indices; a round's scores stay fixed per draw."""
    n = len(scores)
    if count < 0:
        raise ValueError("Selection count must be nonnegative")
    count = min(count, n)
    if not count:
        return np.empty(0, dtype=int)
    base = mix_base if rule == "mix" else rule
    if base == "mix":
        raise ValueError("mix cannot use itself as base")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite selection scores")
    if base in ("entropy", "up_margin"):
        if (
            probabilities is None
            or probabilities.shape != (n, 3)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        ):
            raise RuntimeError(f"{base} requires production three-class probabilities")
        if base == "entropy":
            # Same bits/epsilon convention as pipeline.ranking.get_entropy.
            positive = probabilities > 1e-9
            logs = np.zeros_like(probabilities)
            np.log2(probabilities, out=logs, where=positive)
            priority = (probabilities * logs).sum(axis=1)
        else:
            priority = np.abs(probabilities[:, 2] - 0.5)
    elif base == "near_cutoff":
        if not len(train_labels):
            raise ValueError("near_cutoff requires training labels")
        priority = np.abs(percentile_scores(scores) - (1 - (train_labels == 2).mean()))
    elif base == "novel":
        if not len(train_embeddings):
            raise ValueError("novel requires training embeddings")
        # Embeddings are validated L2-normalized by the evaluator loader.
        priority = np.max(embeddings @ train_embeddings.T, axis=1)
    elif base == "random":
        priority = np.zeros(n)
    else:
        raise ValueError(f"Unknown rule {base}")
    if rule == "random":
        return rng.choice(n, size=count, replace=False)
    order = np.argsort(priority, kind="stable")
    if rule != "mix":
        return order[:count]
    remaining = np.ones(n, dtype=bool)
    picks: list[int] = []
    for _ in range(count):
        if rng.random() < 0.75 and base != "random":
            chosen = int(order[remaining[order]][0])
        else:
            chosen = int(rng.choice(np.flatnonzero(remaining)))
        remaining[chosen] = False
        picks.append(chosen)
    return np.array(picks, dtype=int)


def subset(fold: FoldData, positions: np.ndarray) -> FoldData:
    """Keep the training snapshot but score only the requested candidate pool."""
    return replace(
        fold,
        candidates=[fold.candidates[i] for i in positions],
        cand_emb=fold.cand_emb[positions],
        similarities={},
        runtime_db=None,
    )


def simulate(
    fold: FoldData,
    fold_no: int,
    rule: Rule,
    seed: int,
    budgets: list[int],
    round_size: int,
    scorer: Scorer,
    mix_base: Rule = "near_cutoff",
) -> Run:
    """Split once, reveal labels cumulatively, and never train on E."""
    if round_size < 1 or budgets != sorted(set(budgets)) or any(b < 1 for b in budgets):
        raise ValueError("Positive round size and increasing positive budgets required")
    if fold.test_vote_times is None or fold.train_emb is None:
        raise ValueError("Fold requires captured vote times and training embeddings")
    test_vote_times = fold.test_vote_times
    order = sorted(
        range(len(fold.test_stories)),
        key=lambda i: (test_vote_times[i], fold.test_stories[i].id),
    )
    middle = len(order) // 2
    if not middle:
        raise ValueError(f"Fold {fold_no} needs at least two held-out stories")
    indices = {s.id: i for i, s in enumerate(fold.candidates)}
    pool = [indices[fold.test_stories[i].id] for i in order[:middle]]
    evaluation = np.array([indices[fold.test_stories[i].id] for i in order[middle:]])
    eval_labels = fold.test_actions[order[middle:]]
    labels = {
        s.id: int(a) for s, a in zip(fold.test_stories, fold.test_actions, strict=True)
    }
    times = dict(
        zip([s.id for s in fold.test_stories], fold.test_vote_times, strict=True)
    )
    rng = np.random.default_rng(seed)
    current = fold
    picked: list[int] = []
    checkpoints: list[Checkpoint] = []
    for budget in [0, *budgets]:
        while len(picked) < budget and pool:
            selection_fold = subset(current, np.array(pool))
            scores, probs = scorer(selection_fold)
            assert current.train_emb is not None
            selected = select(
                rule,
                scores,
                probs,
                selection_fold.cand_emb,
                current.train_emb,
                current.y_train,
                min(round_size, budget - len(picked)),
                rng,
                mix_base,
            )
            chosen = [pool[i] for i in selected]
            stories = [fold.candidates[i] for i in chosen]
            # Fresh in-memory fold DB on every scoring call: no stale fitted model.
            current = replace(
                current,
                train_stories=current.train_stories + stories,
                train_emb=np.vstack([current.train_emb, fold.cand_emb[chosen]]),
                y_train=np.concatenate(
                    [current.y_train, [labels[s.id] for s in stories]]
                ),
                train_vote_times=np.concatenate(
                    [current.train_vote_times, [times[s.id] for s in stories]]
                ),
                similarities={},
                runtime_db=None,
            )
            picked.extend(s.id for s in stories)
            chosen_set = set(chosen)
            pool = [i for i in pool if i not in chosen_set]
        scores, _ = scorer(subset(current, evaluation))
        checkpoints.append(
            Checkpoint(
                budget,
                len(picked),
                len(picked) == budget,
                [fold.candidates[i].id for i in evaluation],
                eval_labels.tolist(),
                [float(v) for v in scores],
                metrics(eval_labels, scores),
            )
        )
    return Run(fold_no, rule, seed, picked, checkpoints)


def paired_bootstrap(pairs: list[Pair], resamples: int = 2000) -> dict[str, Interval]:
    """Resample E within folds, sharing each draw across rules and matched seeds."""
    rng = np.random.default_rng(20261012)
    point: dict[str, list[float]] = {key: [] for key in METRICS}
    samples: dict[str, list[float]] = {key: [] for key in METRICS}

    def differences(pair: Pair, idx: np.ndarray) -> dict[str, float | None]:
        values: dict[str, list[float]] = {key: [] for key in METRICS}
        for scores, reference in zip(pair.scores, pair.random_scores, strict=True):
            a, b = (
                metrics(pair.labels[idx], scores[idx]),
                metrics(pair.labels[idx], reference[idx]),
            )
            for key in METRICS:
                av, bv = getattr(a, key), getattr(b, key)
                if av is not None and bv is not None:
                    values[key].append(av - bv)
        return {key: float(np.mean(v)) if v else None for key, v in values.items()}

    for pair in pairs:
        for key, value in differences(pair, np.arange(len(pair.labels))).items():
            if value is not None:
                point[key].append(value)
    # Identical paired predictions have exactly zero differences on every
    # possible resample. Avoid thousands of redundant sklearn calls.
    if all(
        np.array_equal(scores, reference)
        for pair in pairs
        for scores, reference in zip(pair.scores, pair.random_scores, strict=True)
    ):
        return {
            key: Interval(
                0.0 if point[key] else None,
                0.0 if point[key] else None,
                0.0 if point[key] else None,
                len(point[key]),
            )
            for key in METRICS
        }
    for _ in range(resamples):
        draws: dict[str, list[float]] = {key: [] for key in METRICS}
        for pair in pairs:
            idx = rng.integers(0, len(pair.labels), size=len(pair.labels))
            for key, value in differences(pair, idx).items():
                if value is not None:
                    draws[key].append(value)
        for key, values in draws.items():
            if values:
                samples[key].append(float(np.mean(values)))
    return {
        key: Interval(
            float(np.mean(point[key])) if point[key] else None,
            float(np.quantile(samples[key], 0.025)) if samples[key] else None,
            float(np.quantile(samples[key], 0.975)) if samples[key] else None,
            len(point[key]),
        )
        for key in METRICS
    }


def summarize(
    runs: list[Run], budgets: list[int], rules: list[Rule], seeds: int
) -> list[Summary]:
    lookup = {
        (r.fold, r.rule, r.seed, c.budget): c for r in runs for c in r.checkpoints
    }
    folds = sorted({r.fold for r in runs})
    summaries: list[Summary] = []
    for rule in rules:
        for budget in [0, *budgets]:
            pairs: list[Pair] = []
            means: dict[str, list[float]] = {key: [] for key in METRICS}
            for fold in folds:
                selected = [
                    lookup[fold, rule, seed if rule in ("random", "mix") else 0, budget]
                    for seed in range(seeds)
                ]
                reference = [
                    lookup[fold, "random", seed, budget] for seed in range(seeds)
                ]
                if not all(c.attained for c in selected + reference):
                    continue
                pairs.append(
                    Pair(
                        np.array(selected[0].labels),
                        [np.array(c.scores) for c in selected],
                        [np.array(c.scores) for c in reference],
                    )
                )
                for key in METRICS:
                    values = [getattr(c.metrics, key) for c in selected]
                    defined = [v for v in values if v is not None]
                    if defined:
                        means[key].append(float(np.mean(defined)))
            summaries.append(
                Summary(
                    rule,
                    budget,
                    {key: float(np.mean(v)) if v else None for key, v in means.items()},
                    paired_bootstrap(pairs),
                    len(pairs),
                )
            )
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Frozen read-only SQLite snapshot")
    parser.add_argument("--config")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--now", type=float)
    parser.add_argument("--budgets", default="50,100,200")
    parser.add_argument("--round-size", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--initial-train-frac",
        type=float,
        default=0.5,
        help="Share of timestamp groups before the first fold (evaluator default)",
    )
    parser.add_argument("--rules", default="all")
    parser.add_argument(
        "--mix-base", choices=[r for r in RULES if r != "mix"], default="near_cutoff"
    )
    parser.add_argument(
        "--exclude-updated-between", nargs=2, type=float, metavar=("LO", "HI")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        budgets = [int(b) for b in args.budgets.split(",")]
    except ValueError:
        parser.error("--budgets must be comma-separated integers")
    if not budgets or budgets != sorted(set(budgets)) or any(b < 1 for b in budgets):
        parser.error("--budgets must be increasing positive cumulative budgets")
    if min(args.folds, args.round_size, args.seeds) < 1:
        parser.error("folds, round-size and seeds must be positive")
    if not 0 < args.initial_train_frac < 1:
        parser.error("--initial-train-frac must be between 0 and 1")
    names = list(RULES) if args.rules == "all" else args.rules.split(",")
    if not names or any(r not in RULES for r in names) or len(set(names)) != len(names):
        parser.error("--rules must be all or distinct comma-separated rule names")
    rules = [cast(Rule, r) for r in names]
    if "random" not in rules:
        rules.insert(0, "random")
    now = args.now if args.now is not None else time.time()
    if not math.isfinite(now):
        parser.error("--now must be finite")
    window = args.exclude_updated_between
    if window and (not all(math.isfinite(v) for v in window) or window[0] > window[1]):
        parser.error("Exclusion window requires finite LO <= HI (inclusive)")
    if args.output.resolve() == Path(args.db).resolve() or args.output.suffix == ".db":
        parser.error("Output must not overwrite a database")
    config = Config.load(args.config or "config.toml")
    with patch("time.time", return_value=now), frozen_database(args.db) as (db, digest):
        execute(db, digest, config, args, rules, budgets, now)


def execute(
    db: Database,
    digest: str,
    config: Config,
    args: argparse.Namespace,
    rules: list[Rule],
    budgets: list[int],
    now: float,
) -> None:
    stories, raw_labels, raw_times = db.get_feedback_for_training(user_id=args.user_id)
    labels = np.array(raw_labels, dtype=int)
    times = np.array(raw_times, dtype=float)
    if not np.isfinite(times).all() or not set(labels.tolist()) <= {0, 1, 2}:
        raise ValueError("Invalid feedback labels or timestamps")
    window = args.exclude_updated_between
    dropped = [i for i, t in enumerate(times) if window and window[0] <= t <= window[1]]
    dropped_ids = [stories[i].id for i in dropped]
    keep = np.array(
        [i for i in range(len(stories)) if i not in set(dropped)], dtype=int
    )
    stories = [stories[i] for i in keep]
    labels, times = labels[keep], times[keep]
    groups = np.unique(times)
    if len(groups) < 3:
        raise ValueError(
            "Need at least three timestamp groups before confirmation reserve"
        )
    confirmation_start = groups[max(1, math.floor(len(groups) * 0.8))]
    valid = np.flatnonzero(times < confirmation_start)
    hashes = dict(
        zip([s.id for s in stories], _embedding_text_hashes(stories), strict=True)
    )
    embeddings = _validated_embeddings(
        stories,
        db.get_embeddings_batch(
            [s.id for s in stories], config.embedding_model_version, hashes
        ),
    )
    y, vote_times = labels[valid], times[valid]
    splits = _temporal_splits(
        y, vote_times, folds=args.folds, initial_train_frac=args.initial_train_frac
    )
    _validate_splits(
        splits,
        y,
        split_mode="temporal-expanding",
        required_train_labels=_required_labels_for_variants(["production"]),
    )
    runs: list[Run] = []
    audit: list[FoldAudit] = []
    for split in splits:
        fold = _make_fold(
            stories,
            embeddings,
            stories,
            np.arange(len(stories)),
            vote_times,
            y,
            valid,
            split.train_pos,
            split.test_pos,
            config,
            feedback_embeddings=embeddings,
            judged_only=True,
            needs_experimental=False,
        )
        assert fold.test_vote_times is not None
        middle = len(fold.test_stories) // 2
        # Same vote-time order as simulate()'s P/E split.
        test_vote_times = fold.test_vote_times
        by_time = sorted(
            fold.test_stories,
            key=lambda s, t=dict(zip([s.id for s in fold.test_stories], test_vote_times, strict=True)): (
                t[s.id],
                s.id,
            ),
        )
        audit.append(
            FoldAudit(
                split.fold_no,
                len(fold.train_stories),
                len(split.test_pos),
                len(fold.test_stories),
                [s.id for s in by_time[:middle]],
                [s.id for s in by_time[middle:]],
                middle,
                len(fold.test_stories) - middle,
                float(fold.train_vote_times.max()),
                float(vote_times[split.test_pos].min()),
                float(vote_times[split.test_pos].max()),
            )
        )
        for rule in rules:
            for seed in range(args.seeds if rule in ("random", "mix") else 1):
                try:
                    runs.append(
                        simulate(
                            fold,
                            split.fold_no,
                            rule,
                            seed,
                            budgets,
                            args.round_size,
                            lambda f: _production_scores(f, config, db),
                            cast(Rule, args.mix_base),
                        )
                    )
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Fold {split.fold_no}, rule {rule}, seed {seed}: {exc}"
                    ) from exc
        print(
            f"fold {split.fold_no}: train={len(fold.train_stories)} P={middle} E={len(fold.test_stories) - middle}",
            flush=True,
        )
    summary = summarize(runs, budgets, rules, args.seeds)
    go = any(
        s.rule != "random"
        and s.budget == 100
        and s.attained_folds == len(splits)
        and s.against_random["auc"].defined_folds == len(splits)
        and s.against_random["auc"].low is not None
        and s.against_random["auc"].low > 0
        for s in summary
    )
    report = {
        "schema_version": 1,
        "interpretation": "Retrospective, previously shown/voted stories only; current stored content, updated_at chronology. Latest 20% timestamp groups reserved as in evaluator. Not causal evidence; semantic duplicates may remain. Unattainable budgets excluded from summaries; undefined AUC is null.",
        "db_sha256": digest,
        "now": now,
        "user_id": args.user_id,
        "config": asdict(config),
        "budgets": budgets,
        "round_size": args.round_size,
        "seeds": args.seeds,
        "initial_train_frac": args.initial_train_frac,
        "rules": rules,
        "mix_base": args.mix_base,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261012,
        "exclude_updated_between": args.exclude_updated_between,
        "dropped_rows": len(dropped),
        "dropped_story_ids": dropped_ids,
        "confirmation_start": float(confirmation_start),
        "confirmation_rows": len(stories) - len(valid),
        "folds": [asdict(a) for a in audit],
        "runs": [asdict(r) for r in runs],
        "summary": [asdict(s) for s in summary],
        "decision": "go" if go else "no-go",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("rule         budget   AUC      P@20     top10%   AUC delta [95% CI]")
    for s in summary:

        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{value:.4f}"

        ci = s.against_random["auc"]
        print(
            f"{s.rule:12} {s.budget:6} {fmt(s.mean['auc']):>8} {fmt(s.mean['precision_at_20']):>8} {fmt(s.mean['up_rate_top_decile']):>8} {fmt(ci.difference)} [{fmt(ci.low)}, {fmt(ci.high)}] folds={s.attained_folds}/{len(splits)}"
        )
    print(
        f"{'go' if go else 'no-go'}: AUC improvement at budget 100 requires a positive paired CI in all folds"
    )


if __name__ == "__main__":
    main()
