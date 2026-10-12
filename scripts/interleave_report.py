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
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database, InterleaveDeck
from pipeline.interleave import PRODUCTION_ARM
from scripts.simulate_interleaving import Credit, compare

LABELS = {"down": 0, "neutral": 1, "up": 2}
IMPRESSION_SLACK_SECONDS = 1.0


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
    level = alpha / max(len(arms) - 1, 1)
    out: dict[str, object] = {
        "credited_by_arm": {
            arm: dict(Counter(c.action for c in credited if c.arm == arm))
            for arm in arms
        },
        "decks_with_credit": len({c.version for c in credited}),
        "per_test_alpha": level,
    }
    for arm in arms[1:]:
        result = compare(credits, index[arm])
        out[arm] = {
            # Rates of an arm with no credited votes are NaN: report null.
            **{
                k: None if isinstance(v, float) and math.isnan(v) else v
                for k, v in asdict(result).items()
            },
            "deck_sign_rejects": result.deck_sign_p < level,
            "rate_rejects": result.rate_p < level,
        }
    return out


def load(
    db: Database, user_id: int, since: float, until: float | None
) -> tuple[list[InterleaveDeck], list[Vote], list[Impression]]:
    end = until if until is not None else float("inf")
    try:
        decks = db.get_interleave_decks(user_id, since=0.0)
    except sqlite3.OperationalError:
        decks = []
    votes = [
        Vote(int(sid), str(action), float(at))
        for sid, action, at in db.execute(
            "SELECT story_id, action, updated_at FROM feedback "
            "WHERE user_id = ? AND updated_at >= ? ORDER BY updated_at",
            (user_id, since),
        )
        if float(at) <= end
    ]
    impressions = [
        Impression(int(sid), float(at), int(version), str(view), str(window))
        for sid, at, version, view, window in db.execute(
            "SELECT story_id, occurred_at, dashboard_version, sort_mode, age_filter "
            "FROM interaction_events WHERE user_id = ? AND event_type = 'impression' "
            "AND occurred_at >= ?",
            (user_id, since - 86400.0),
        )
    ]
    return decks, votes, impressions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--since", required=True, type=float, help="Unix start time")
    parser.add_argument("--until", type=float)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[PRODUCTION_ARM, "joined_all", "joined_no_metadata"],
        help="Arm names, production first",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.arms[0] != PRODUCTION_ARM or len(args.arms) < 2:
        parser.error("--arms must start with production and name a challenger")
    db = Database(str(args.db), read_only=True)
    try:
        decks, votes, impressions = load(db, args.user_id, args.since, args.until)
    finally:
        db.close()
    credited, skipped = credit_votes(decks, votes, impressions)
    report = {
        "user_id": args.user_id,
        "since": args.since,
        "until": args.until,
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
