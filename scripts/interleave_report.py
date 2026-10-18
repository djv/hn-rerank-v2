#!/usr/bin/env python3
"""Credit live votes to interleaving arms and test each challenger.

A vote counts when the user's last impression of that story before the
vote (``interaction_events``, 1 s slack) was in a Recommended view whose
deck version is in ``interleave_decks`` with the story in that window; it
is credited to the arm that drafted the story. Undone votes are gone from
``feedback`` and changed votes count with their final action. Statistics
match ``scripts/simulate_interleaving.py``: per deck version, a sign test
of the challenger's against production's upvotes, and a two-proportion test
of credited upvote rates, Bonferroni over the challengers.

    uv run python scripts/interleave_report.py --db hn_rewrite.db \\
        --user-id 151 --since 1791500000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy.stats import binomtest, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, InterleaveDeck
from pipeline import FAR_ARM
from pipeline.interleave import PRODUCTION_ARM
from scripts.simulate_interleaving import Credit, compare

LABELS = {"down": 0, "neutral": 1, "up": 2}
IMPRESSION_SLACK_SECONDS = 1.0
# Keep the preregistered family size even if a challenger is safety-stopped.
PREREGISTERED_CHALLENGERS = 2
SAFETY_ALPHA = 0.01


@dataclass(frozen=True)
class ExcludedInterval:
    """Inclusive updated_at interval, for explicitly recorded calibration."""

    start: float
    end: float


def exclude_votes(
    votes: Sequence[Vote], intervals: Sequence[ExcludedInterval]
) -> list[Vote]:
    return [
        vote
        for vote in votes
        if not any(interval.start <= vote.at <= interval.end for interval in intervals)
    ]


@dataclass(frozen=True)
class Vote:
    story_id: int
    action: str
    at: float


@dataclass(frozen=True)
class Impression:
    story_id: int
    at: float
    version: int
    view: str
    window: str


@dataclass(frozen=True)
class CreditedVote:
    story_id: int
    version: int
    window: str
    arm: str
    action: str
    at: float


def credit_votes(
    decks: Sequence[InterleaveDeck],
    votes: Sequence[Vote],
    impressions: Sequence[Impression],
) -> tuple[list[CreditedVote], Counter[str]]:
    arm_of = {
        (d.version, window, sid): arm
        for d in decks
        for window, picks in d.windows.items()
        for sid, arm in picks
    }
    versions = {d.version for d in decks}
    seen: dict[int, list[Impression]] = {}
    for imp in sorted(impressions, key=lambda i: i.at):
        seen.setdefault(imp.story_id, []).append(imp)
    credited: list[CreditedVote] = []
    skipped: Counter[str] = Counter()
    for vote in votes:
        if vote.action not in LABELS:
            skipped["unknown_action"] += 1
            continue
        before = [
            i
            for i in seen.get(vote.story_id, [])
            if i.at <= vote.at + IMPRESSION_SLACK_SECONDS
        ]
        if not before:
            skipped["no_impression"] += 1
            continue
        last = before[-1]
        if last.view != "recommended":
            skipped[f"view_{last.view}"] += 1
        elif last.version not in versions:
            skipped["deck_not_interleaved"] += 1
        elif (key := (last.version, last.window, vote.story_id)) not in arm_of:
            skipped["story_not_in_window"] += 1
        elif arm_of[key] == FAR_ARM:
            skipped["far_card"] += 1
        else:
            credited.append(
                CreditedVote(
                    vote.story_id,
                    last.version,
                    last.window,
                    arm_of[key],
                    vote.action,
                    vote.at,
                )
            )
    return credited, skipped


def summarize(
    credited: Sequence[CreditedVote], arms: Sequence[str], since: float, alpha: float
) -> dict[str, object]:
    index = {arm: i for i, arm in enumerate(arms)}
    credits = [
        Credit(int((c.at - since) // 86400), c.version, index[c.arm], LABELS[c.action])
        for c in credited
        if c.arm in index
    ]
    if len(arms) != len(set(arms)) or not 2 <= len(arms) <= 3:
        raise ValueError("name one or two distinct preregistered challengers")
    if arms[0] != PRODUCTION_ARM:
        raise ValueError("production must be the first arm")
    level = alpha / PREREGISTERED_CHALLENGERS
    out: dict[str, object] = {
        "credited_by_arm": {
            arm: dict(Counter(c.action for c in credited if c.arm == arm))
            for arm in arms
        },
        "decks_with_credit": len({c.version for c in credited}),
        "per_test_alpha": level,
        "bonferroni_challengers": PREREGISTERED_CHALLENGERS,
        "up_rate_diff_ci_method": "newcombe_wilson",
        "up_rate_diff_ci_confidence": 0.95,
        "down_rate_test": "one_sided_pooled_z_challenger_greater",
        "safety_alpha": SAFETY_ALPHA,
    }
    for arm in arms[1:]:
        result = compare(credits, index[arm])
        n1, n0 = result.credited, result.control_credited
        arm_votes = [c for c in credited if c.arm == arm]
        control_votes = [c for c in credited if c.arm == PRODUCTION_ARM]
        up1 = sum(c.action == "up" for c in arm_votes)
        up0 = sum(c.action == "up" for c in control_votes)
        down1 = sum(c.action == "down" for c in arm_votes)
        down0 = sum(c.action == "down" for c in control_votes)
        down_p = greater_rate_p(down1, n1, down0, n0)
        up_diff = result.up_rate - result.control_up_rate if n1 and n0 else None
        down_diff = result.down_rate - result.control_down_rate if n1 and n0 else None
        out[arm] = {
            # Rates of an arm with no credited votes are NaN: report null.
            **{
                k: None if isinstance(v, float) and math.isnan(v) else v
                for k, v in asdict(result).items()
            },
            "deck_sign_rejects": result.deck_sign_p < level,
            "rate_rejects": result.rate_p < level,
            "up_rate_diff": up_diff,
            "up_rate_diff_ci": rate_difference_ci(up1, n1, up0, n0),
            "down_rate_diff": down_diff,
            "down_rate_p": down_p,
            "down_rate_rejects": down_p is not None and down_p < level,
            "down_safety_stop": (
                down_diff is not None
                and down_diff >= 0.10
                and down_p is not None
                and down_p < SAFETY_ALPHA
            ),
        }
    return out


def greater_rate_p(k1: int, n1: int, k0: int, n0: int) -> float | None:
    """One-sided pooled z-test: challenger rate greater than control."""
    if not n1 or not n0:
        return None
    pooled = (k1 + k0) / (n1 + n0)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    return float(norm.sf((k1 / n1 - k0 / n0) / se)) if se else 1.0


def rate_difference_ci(
    k1: int, n1: int, k0: int, n0: int
) -> tuple[float, float] | None:
    """95% Newcombe interval combining uncorrected Wilson score bounds.

    Newcombe (1998), independent proportions, method 10:
    https://pubmed.ncbi.nlm.nih.gov/9595617/
    This is descriptive and unadjusted; the primary z-test is unchanged.
    """
    if not n1 or not n0:
        return None
    p1, p0 = k1 / n1, k0 / n0
    ci1 = binomtest(k1, n1).proportion_ci(method="wilson")
    ci0 = binomtest(k0, n0).proportion_ci(method="wilson")
    diff = p1 - p0
    return (
        float(diff - math.hypot(p1 - ci1.low, ci0.high - p0)),
        float(diff + math.hypot(ci1.high - p1, p0 - ci0.low)),
    )


def load(
    db: Database, user_id: int, since: float, until: float | None
) -> tuple[list[InterleaveDeck], list[Vote], list[Impression]]:
    end = until if until is not None else float("inf")
    decks = db.get_interleave_decks(user_id, since=since)
    if not decks:
        raise ValueError(
            "no interleaved decks at or after --since; verify DB and profile"
        )
    votes = [
        Vote(int(sid), str(action), float(at))
        for sid, action, at in db.execute(
            "SELECT story_id, action, updated_at FROM feedback "
            "WHERE user_id = ? AND updated_at >= ? ORDER BY updated_at",
            (user_id, since),
        )
        if float(at) < end
    ]
    impressions = [
        Impression(int(sid), float(at), int(version), str(view), str(window))
        for sid, at, version, view, window in db.execute(
            "SELECT story_id, occurred_at, dashboard_version, sort_mode, age_filter "
            "FROM interaction_events WHERE user_id = ? AND event_type = 'impression' "
            "AND occurred_at >= ? ORDER BY occurred_at, received_at, event_id",
            (user_id, since - 86400.0),
        )
    ]
    return decks, votes, impressions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--since", required=True, type=float, help="Unix start time")
    parser.add_argument("--until", type=float, help="Exclusive Unix end time")
    parser.add_argument(
        "--exclude-updated-between",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
        help="Exclude an inclusive updated_at interval (repeatable); echo counts",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[PRODUCTION_ARM, "joined_all", "joined_no_metadata"],
        help="Arm names, production first",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        args.arms[0] != PRODUCTION_ARM
        or not 2 <= len(args.arms) <= 3
        or len(args.arms) != len(set(args.arms))
        or any(
            a not in {PRODUCTION_ARM, "joined_all", "joined_no_metadata"}
            for a in args.arms
        )
    ):
        parser.error(
            "--arms must start with production and name distinct trial challengers"
        )
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if not math.isfinite(args.since) or (
        args.until is not None
        and (not math.isfinite(args.until) or args.until <= args.since)
    ):
        parser.error("--since/--until must be finite and increasing")
    intervals = [
        ExcludedInterval(start, end) for start, end in args.exclude_updated_between
    ]
    if any(
        not math.isfinite(i.start) or not math.isfinite(i.end) or i.start > i.end
        for i in intervals
    ):
        parser.error("exclusion intervals must have finite START <= END")
    db = Database(str(args.db), read_only=True)
    try:
        decks, votes, impressions = load(db, args.user_id, args.since, args.until)
    finally:
        db.close()
    raw_votes = len(votes)
    votes = exclude_votes(votes, intervals)
    credited, skipped = credit_votes(decks, votes, impressions)
    report = {
        "user_id": args.user_id,
        "since": args.since,
        "until": args.until,
        "raw_votes": raw_votes,
        "excluded_votes": raw_votes - len(votes),
        "excluded_updated_intervals": [asdict(i) for i in intervals],
        "votes": len(votes),
        "credited": len(credited),
        "skipped": dict(skipped),
        "summary": summarize(credited, args.arms, args.since, args.alpha),
    }
    text = json.dumps(report, indent=2, allow_nan=False, default=str)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
