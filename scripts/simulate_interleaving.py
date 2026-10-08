#!/usr/bin/env python3
"""Power check for live team-draft interleaving, replayed on judged blocks.

Each arm's frozen scores (``eval_ranker_variants.py --dump-scores``) rank a
held-out block's voted stories. A simulated day picks a random block and
serves decks: the arms' rankings of the stories not yet voted on are
interleaved (``pipeline.interleave.team_draft``), the first
``--votes-per-deck`` cards get the user's real vote and are credited to the
arm that added them, then the deck is re-interleaved. Models are not refit
within a block and every candidate is judged, so this is an optimistic
replay of the swipe deck, not a forecast of live effect sizes.

For each challenger against the first (control) arm and each duration it
reports how often a two-sided test rejects at ``--alpha`` (Bonferroni over
the challengers): a sign test over decks where the arms' upvote counts
differ, and a two-proportion test of credited upvote rates.
``--challenger-weight`` W ranks each challenger by W x its percentile rank
plus (1 - W) x the control's: 0 replays the control in every arm (the
false-positive rate), values below 1 shrink the replayed differences.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, norm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.interleave import team_draft
from pipeline.linear_blend import percentile_scores

UP, DOWN = 2, 0


@dataclass(frozen=True)
class Block:
    name: str
    ids: tuple[int, ...]
    labels: NDArray[np.int64]
    rankings: tuple[tuple[int, ...], ...]  # per arm, best first (row indices)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    variant: str
    files: tuple[Path, ...]


@dataclass(frozen=True)
class Credit:
    day: int
    deck: int
    arm: int
    label: int


def parse_arm(values: list[str]) -> ArmSpec:
    if len(values) < 3:
        raise argparse.ArgumentTypeError("--arm needs NAME VARIANT FILE [FILE ...]")
    return ArmSpec(values[0], values[1], tuple(Path(v) for v in values[2:]))


def load_blocks(arms: list[ArmSpec], *, challenger_weight: float) -> list[Block]:
    """Judged stories of every block, ranked by each arm. All arms must
    list the same blocks with identical story IDs and labels."""
    if len({len(a.files) for a in arms}) != 1:
        raise ValueError("every arm needs the same number of score files")
    blocks: list[Block] = []
    for file_index in range(len(arms[0].files)):
        dumps = [json.loads(a.files[file_index].read_text()) for a in arms]
        folds = [d["folds"] for d in dumps]
        if len({len(f) for f in folds}) != 1:
            raise ValueError("score files disagree on the number of folds")
        for fold_index, control in enumerate(folds[0]):
            labels = np.asarray(control["labels"], dtype=np.int64)
            judged = np.flatnonzero(labels >= 0)
            rankings = []
            control_rank = percentile_scores(
                np.asarray(control["scores"][arms[0].variant], dtype=np.float64)[judged]
            ).astype(np.float64)
            for index, (arm, fold) in enumerate(
                zip(arms, (f[fold_index] for f in folds), strict=True)
            ):
                if fold["ids"] != control["ids"] or fold["labels"] != control["labels"]:
                    raise ValueError(f"{arm.name}: block stories or labels differ")
                scores = percentile_scores(
                    np.asarray(fold["scores"][arm.variant], dtype=np.float64)[judged]
                ).astype(np.float64)
                if index:
                    scores = (
                        challenger_weight * scores
                        + (1 - challenger_weight) * control_rank
                    )
                # Ties keep candidate order, as the stable sort in serving does.
                order = np.argsort(-scores, kind="stable")
                rankings.append(tuple(int(i) for i in order))
            blocks.append(
                Block(
                    f"{file_index}-{control['fold']}",
                    tuple(int(control["ids"][i]) for i in judged),
                    labels[judged],
                    tuple(rankings),
                )
            )
    return blocks


def simulate_days(
    blocks: list[Block],
    *,
    days: int,
    votes_per_day: int,
    deck_size: int,
    votes_per_deck: int,
    rng: np.random.Generator,
) -> list[Credit]:
    credits: list[Credit] = []
    deck_no = 0
    for day in range(days):
        block = blocks[int(rng.integers(len(blocks)))]
        remaining = [list(r) for r in block.rankings]
        votes = 0
        while votes < votes_per_day and remaining[0]:
            picks = team_draft(remaining, deck_size, rng)[:votes_per_deck]
            for pick in picks:
                credits.append(
                    Credit(day, deck_no, pick.arm, int(block.labels[pick.item]))
                )
                for ranking in remaining:
                    ranking.remove(pick.item)
            votes += len(picks)
            deck_no += 1
    return credits


@dataclass(frozen=True)
class Comparison:
    deck_sign_p: float
    deck_wins: int
    deck_losses: int
    rate_p: float
    up_rate: float
    control_up_rate: float
    down_rate: float
    control_down_rate: float
    credited: int
    control_credited: int


def compare(credits: list[Credit], arm: int, control: int = 0) -> Comparison:
    per_deck: dict[int, list[int]] = {}
    ups = {arm: 0, control: 0}
    downs = {arm: 0, control: 0}
    totals = {arm: 0, control: 0}
    for c in credits:
        if c.arm not in totals:
            continue
        totals[c.arm] += 1
        ups[c.arm] += c.label == UP
        downs[c.arm] += c.label == DOWN
        tally = per_deck.setdefault(c.deck, [0, 0])
        tally[0 if c.arm == arm else 1] += c.label == UP
    wins = sum(a > b for a, b in per_deck.values())
    losses = sum(a < b for a, b in per_deck.values())
    sign_p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    n1, n0 = totals[arm], totals[control]
    p1 = ups[arm] / n1 if n1 else math.nan
    p0 = ups[control] / n0 if n0 else math.nan
    pooled = (ups[arm] + ups[control]) / (n1 + n0) if n1 + n0 else math.nan
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0)) if n1 and n0 else 0.0
    rate_p = float(2 * norm.sf(abs(p1 - p0) / se)) if se > 0 else 1.0
    return Comparison(
        float(sign_p),
        wins,
        losses,
        rate_p,
        p1,
        p0,
        downs[arm] / n1 if n1 else math.nan,
        downs[control] / n0 if n0 else math.nan,
        n1,
        n0,
    )


def summarize(
    runs: list[list[Comparison]], alpha: float, challengers: int
) -> dict[str, float]:
    """Rejection rates across replicates for one challenger and duration."""
    level = alpha / challengers
    flat = [r for replicate in runs for r in replicate]

    def rate(test: str, direction: int) -> float:
        hits = 0
        for c in flat:
            p = c.deck_sign_p if test == "deck_sign" else c.rate_p
            effect = (
                c.deck_wins - c.deck_losses
                if test == "deck_sign"
                else c.up_rate - c.control_up_rate
            )
            if p < level and (direction == 0 or np.sign(effect) == direction):
                hits += 1
        return hits / len(flat)

    return {
        "deck_sign_reject": rate("deck_sign", 0),
        "deck_sign_challenger_better": rate("deck_sign", 1),
        "deck_sign_challenger_worse": rate("deck_sign", -1),
        "rate_reject": rate("rate", 0),
        "rate_challenger_better": rate("rate", 1),
        "rate_challenger_worse": rate("rate", -1),
        "mean_up_rate": float(np.nanmean([c.up_rate for c in flat])),
        "mean_control_up_rate": float(np.nanmean([c.control_up_rate for c in flat])),
        "mean_down_rate": float(np.nanmean([c.down_rate for c in flat])),
        "mean_control_down_rate": float(
            np.nanmean([c.control_down_rate for c in flat])
        ),
        "mean_credited": float(np.mean([c.credited for c in flat])),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--arm",
        nargs="+",
        action="append",
        required=True,
        metavar="NAME VARIANT FILE",
        help="Arm name, score variant and one score file per period; first is control",
    )
    parser.add_argument("--days", type=int, nargs="+", default=[7, 14, 28, 56])
    parser.add_argument("--votes-per-day", type=int, default=50)
    parser.add_argument("--deck-size", type=int, default=12)
    parser.add_argument("--votes-per-deck", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--challenger-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    arms = [parse_arm(values) for values in args.arm]
    if len(arms) < 2:
        parser.error("need a control and at least one challenger arm")
    for name in ("votes_per_day", "deck_size", "votes_per_deck", "replicates"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.challenger_weight <= 1:
        parser.error("--challenger-weight must be in [0, 1]")
    if args.votes_per_deck > args.deck_size or min(args.days) < 1:
        parser.error("votes per deck must fit the deck and days must be positive")
    blocks = load_blocks(arms, challenger_weight=args.challenger_weight)
    rng = np.random.default_rng(args.seed)
    durations = sorted(set(args.days))
    results: dict[str, dict[int, list[list[Comparison]]]] = {
        a.name: {d: [] for d in durations} for a in arms[1:]
    }
    for _ in range(args.replicates):
        credits = simulate_days(
            blocks,
            days=durations[-1],
            votes_per_day=args.votes_per_day,
            deck_size=args.deck_size,
            votes_per_deck=args.votes_per_deck,
            rng=rng,
        )
        for days in durations:
            prefix = [c for c in credits if c.day < days]
            for index, arm in enumerate(arms[1:], 1):
                results[arm.name][days].append([compare(prefix, index)])
    report = {
        "settings": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k != "arm"
        },
        "arms": [{**asdict(a), "files": [str(f) for f in a.files]} for a in arms],
        "blocks": len(blocks),
        "judged_stories": sum(len(b.ids) for b in blocks),
        "power": {
            name: {
                str(days): summarize(runs, args.alpha, len(arms) - 1)
                for days, runs in by_days.items()
            }
            for name, by_days in results.items()
        },
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    for name, by_days in report["power"].items():
        for days, row in by_days.items():
            print(
                f"{name} {days}d: deck-sign reject {row['deck_sign_reject']:.2f} "
                f"(better {row['deck_sign_challenger_better']:.2f}), rate reject "
                f"{row['rate_reject']:.2f}; up {row['mean_up_rate']:.3f} vs "
                f"{row['mean_control_up_rate']:.3f}, down {row['mean_down_rate']:.3f} "
                f"vs {row['mean_control_down_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
