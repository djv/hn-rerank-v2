"""Interaction-ledger diagnostic: is dwell a usable label, does preference
drift exist, and where does the deck's effective engagement collapse?

Read-only. Motivated by ROADMAP.md B1/B2/B3 and
notes/improvement-brainstorm-2026-07-15.md — this is the "measure before you
build" pass those items call for. See `scripts/narrowing_report.py` for the
sibling *topical coverage* report; this script covers position bias, source
mix, dwell-vote agreement, and centroid drift, and ends with an explicit
next-step recommendation.

Caveats (also printed at runtime):
  - Impressions before 2026-07-15 are HN-survivor-biased (the ledger
    silently dropped every non-HN event before commit 3a5a77c). Use
    --since 2026-07-15 (the default) to exclude that window.
  - `feedback.updated_at` is mutation time, not vote-creation time, so the
    drift and dwell-vote joins below are approximate, not exact.
  - The candidate pool is currently HN-only (non-HN legs disabled in
    load_production_candidate_stories), so source-mix coverage measures
    HN-internal composition, not true cross-source reach.
  - Every logged event has ranker_arm='baseline' — every rate here is
    conditioned on what the baseline ranker chose to show. No counterfactual
    ("what if a different story had been shown") claim is valid from this
    data; that requires B2 (randomized exploration + logged propensities).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config  # noqa: E402
from scripts._ledger_common import (  # noqa: E402
    SINCE_LEDGER_FIX,
    cosine_distance,
    heaviest_user_id,
    iso_week,
    parse_since,
    pool_embeddings,
)

FeedbackLabel = Literal["up", "down", "neutral", "none"]

POSITION_BUCKET_WIDTH = 5
DWELL_CAP_MS_DEFAULT = 120_000
DRIFT_RECENT_DAYS_DEFAULT = 60
DRIFT_MATERIAL_THRESHOLD = 0.05
DWELL_AUC_USABLE_THRESHOLD = 0.65


@dataclass(frozen=True)
class LedgerWindow:
    user_id: int
    since: float | None
    impressions: int
    dwells: int
    article_opens: int
    comments_opens: int
    weeks: tuple[str, ...]


@dataclass(frozen=True)
class PositionStat:
    sort_mode: str
    bucket: int
    impressions: int
    vote_rate: float
    up_rate: float
    open_rate: float


@dataclass(frozen=True)
class DwellVoteStat:
    label: FeedbackLabel
    n: int
    p50_ms: float
    p90_ms: float


@dataclass(frozen=True)
class DriftStat:
    recent_n: int
    older_n: int
    centroid_cos_distance: float


# --- pure metric functions (unit-tested in tests/test_ledger_report.py) ---


def bucket_position(position: int, width: int = POSITION_BUCKET_WIDTH) -> int:
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    return (position // width) * width


def cap_dwell_ms(duration_ms: int, cap_ms: int = DWELL_CAP_MS_DEFAULT) -> int:
    return min(max(duration_ms, 0), cap_ms)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]. Empty input -> nan."""
    if not values:
        return float("nan")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))
    return ordered[idx]


def rank_auc(scores: list[float], labels: list[bool]) -> float:
    """AUC of `scores` as a predictor of `labels` (True = positive class).

    Standard Mann-Whitney U formulation: probability a random positive
    outranks a random negative, ties counting as 0.5. Returns nan if either
    class is empty.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    pos = [s for s, lbl in zip(scores, labels) if lbl]
    neg = [s for s, lbl in zip(scores, labels) if not lbl]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def centroid(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(vectors) == 0:
        raise ValueError("cannot take centroid of zero vectors")
    return vectors.mean(axis=0)


# --- report sections ---


def _load_window(
    db: Database, user_id: int, since_ts: float | None
) -> LedgerWindow:
    query = "SELECT event_type, occurred_at FROM interaction_events WHERE user_id = ?"
    params: tuple = (user_id,)
    if since_ts is not None:
        query += " AND occurred_at >= ?"
        params = (user_id, since_ts)
    rows = db.execute(query, params)
    counts: dict[str, int] = defaultdict(int)
    weeks: set[str] = set()
    for event_type, occurred_at in rows:
        counts[event_type] += 1
        weeks.add(iso_week(occurred_at))
    return LedgerWindow(
        user_id=user_id,
        since=since_ts,
        impressions=counts.get("impression", 0),
        dwells=counts.get("dwell", 0),
        article_opens=counts.get("article_open", 0),
        comments_opens=counts.get("comments_open", 0),
        weeks=tuple(sorted(weeks)),
    )


def _position_stats(
    db: Database, user_id: int, since_ts: float | None
) -> list[PositionStat]:
    """Vote/open rate per (sort_mode, position bucket).

    "Vote" and "open" are whether the SAME (story, dashboard_version)
    impression's story later has ANY feedback row / article-or-comments-open
    event for this user — an approximation (not the exact impression that
    triggered the action) but adequate for a bucket-level rate.
    """
    params: tuple = (user_id,)
    since_clause = ""
    if since_ts is not None:
        since_clause = " AND occurred_at >= ?"
        params = (user_id, since_ts)

    impressions = db.execute(
        "SELECT sort_mode, position, story_id FROM interaction_events "
        f"WHERE user_id = ? AND event_type = 'impression'{since_clause}",
        params,
    )
    if not impressions:
        return []

    voted_story_ids = {
        sid for (sid,) in db.execute(
            "SELECT DISTINCT story_id FROM feedback WHERE user_id = ? AND action != 'neutral'",
            (user_id,),
        )
    }
    up_story_ids = {
        sid for (sid,) in db.execute(
            "SELECT DISTINCT story_id FROM feedback WHERE user_id = ? AND action = 'up'",
            (user_id,),
        )
    }
    opened_story_ids = {
        sid for (sid,) in db.execute(
            "SELECT DISTINCT story_id FROM interaction_events "
            "WHERE user_id = ? AND event_type IN ('article_open', 'comments_open')",
            (user_id,),
        )
    }

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for sort_mode, position, story_id in impressions:
        groups[(sort_mode, bucket_position(position))].append(story_id)

    stats = []
    for (sort_mode, bucket), story_ids in sorted(groups.items()):
        n = len(story_ids)
        stats.append(
            PositionStat(
                sort_mode=sort_mode,
                bucket=bucket,
                impressions=n,
                vote_rate=sum(sid in voted_story_ids for sid in story_ids) / n,
                up_rate=sum(sid in up_story_ids for sid in story_ids) / n,
                open_rate=sum(sid in opened_story_ids for sid in story_ids) / n,
            )
        )
    return stats


def _source_mix(
    db: Database, user_id: int, since_ts: float | None, pool_ids: list[int]
) -> list[tuple[str, int, int]]:
    params: tuple = (user_id,)
    since_clause = ""
    if since_ts is not None:
        since_clause = " AND occurred_at >= ?"
        params = (user_id, since_ts)
    rows = db.execute(
        "SELECT DISTINCT story_id FROM interaction_events "
        f"WHERE user_id = ? AND event_type = 'impression'{since_clause}",
        params,
    )
    impressed_ids = [sid for (sid,) in rows]
    if not impressed_ids or not pool_ids:
        return []

    def source_counts(ids: list[int]) -> dict[str, int]:
        placeholders = ",".join("?" for _ in ids)
        result: dict[str, int] = defaultdict(int)
        for source, n in db.execute(
            f"SELECT source, COUNT(*) FROM stories WHERE id IN ({placeholders}) "
            "GROUP BY source",
            tuple(ids),
        ):
            result[source] = n
        return result

    impressed_counts = source_counts(impressed_ids)
    pool_counts = source_counts(pool_ids)
    sources = sorted(set(impressed_counts) | set(pool_counts))
    return [(s, impressed_counts.get(s, 0), pool_counts.get(s, 0)) for s in sources]


def _dwell_vote_stats(
    db: Database, user_id: int, since_ts: float | None, dwell_cap_ms: int
) -> tuple[list[DwellVoteStat], float]:
    params: tuple = (user_id,)
    since_clause = ""
    if since_ts is not None:
        since_clause = " AND occurred_at >= ?"
        params = (user_id, since_ts)
    rows = db.execute(
        "SELECT story_id, duration_ms FROM interaction_events "
        f"WHERE user_id = ? AND event_type = 'dwell' AND duration_ms IS NOT NULL{since_clause}",
        params,
    )
    per_story_ms: dict[int, int] = defaultdict(int)
    capped_count = 0
    for story_id, duration_ms in rows:
        capped = cap_dwell_ms(duration_ms, dwell_cap_ms)
        if capped != duration_ms:
            capped_count += 1
        per_story_ms[story_id] += capped

    if not per_story_ms:
        return [], float("nan")

    labels = dict(
        db.execute(
            "SELECT story_id, action FROM feedback WHERE user_id = ?", (user_id,)
        )
    )

    by_label: dict[FeedbackLabel, list[float]] = defaultdict(list)
    for story_id, ms in per_story_ms.items():
        label: FeedbackLabel = labels.get(story_id, "none")  # type: ignore[assignment]
        by_label[label].append(float(ms))

    stats = [
        DwellVoteStat(
            label=label,
            n=len(values),
            p50_ms=percentile(values, 50),
            p90_ms=percentile(values, 90),
        )
        for label, values in sorted(by_label.items())
    ]

    up_values = by_label.get("up", [])
    down_values = by_label.get("down", [])
    scores = up_values + down_values
    is_up = [True] * len(up_values) + [False] * len(down_values)
    auc = rank_auc(scores, is_up)

    if capped_count:
        print(f"note: {capped_count} of {len(rows)} dwell rows capped at {dwell_cap_ms}ms")

    return stats, auc


def _drift_stat(
    db: Database,
    config: Config,
    user_id: int,
    recent_days: int,
) -> DriftStat | None:
    import time

    rows = db.execute(
        "SELECT story_id, updated_at FROM feedback WHERE user_id = ? AND action = 'up'",
        (user_id,),
    )
    if not rows:
        return None
    cutoff = time.time() - recent_days * 86400
    recent_ids = [sid for sid, ts in rows if ts >= cutoff]
    older_ids = [sid for sid, ts in rows if ts < cutoff]
    if not recent_ids or not older_ids:
        return None

    all_ids = list({*recent_ids, *older_ids})
    hashed = _story_embeddings(db, config, all_ids)
    recent_vecs = np.array([hashed[i] for i in recent_ids if i in hashed], dtype=np.float32)
    older_vecs = np.array([hashed[i] for i in older_ids if i in hashed], dtype=np.float32)
    if len(recent_vecs) == 0 or len(older_vecs) == 0:
        return None

    return DriftStat(
        recent_n=len(recent_vecs),
        older_n=len(older_vecs),
        centroid_cos_distance=cosine_distance(centroid(recent_vecs), centroid(older_vecs)),
    )


def _story_embeddings(
    db: Database, config: Config, story_ids: list[int]
) -> dict[int, NDArray[np.float32]]:
    """Look up cached embeddings for specific stories by current text hash."""
    import hashlib

    from pipeline.ranking import story_embedding_text

    placeholders = ",".join("?" for _ in story_ids)
    rows = db.execute(
        f"SELECT id, title, url, score, time, text_content, source, "
        f"comment_count, discussion_url, comment_count_at_fetch, self_text, "
        f"top_comments, article_body FROM stories WHERE id IN ({placeholders})",
        tuple(story_ids),
    )
    if not rows:
        return {}
    from database import Story

    stories = [
        Story(
            id=r[0], title=r[1], url=r[2], score=r[3], time=r[4],
            text_content=r[5], source=r[6], comment_count=r[7],
            discussion_url=r[8], comment_count_at_fetch=r[9], self_text=r[10],
            top_comments=r[11], article_body=r[12],
        )
        for r in rows
    ]
    hashes = {
        s.id: hashlib.sha256(story_embedding_text(s).encode("utf-8")).hexdigest()
        for s in stories
    }
    return db.get_embeddings_batch(story_ids, config.embedding_model_version, hashes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Position bias, source mix, dwell-vote agreement, and preference "
            "drift over the interaction_events ledger; ends with an explicit "
            "B1/B2/B3 recommendation."
        )
    )
    parser.add_argument("--db", default="hn_rewrite.db")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--since", default=SINCE_LEDGER_FIX)
    parser.add_argument("--dwell-cap-ms", type=int, default=DWELL_CAP_MS_DEFAULT)
    parser.add_argument("--drift-recent-days", type=int, default=DRIFT_RECENT_DAYS_DEFAULT)
    args = parser.parse_args()

    config = Config.load(args.config)
    db = Database(args.db, read_only=True)
    try:
        print(
            "CAVEATS: pre-2026-07-15 impressions are HN-survivor-biased "
            "(fixed in 3a5a77c); feedback.updated_at is mutation time, not "
            "vote-creation time; candidate pool is HN-only (non-HN legs "
            "disabled); every event has ranker_arm='baseline' so no "
            "counterfactual claim is valid from this data alone."
        )
        since_ts = parse_since(args.since) if args.since else None
        if since_ts is not None:
            print(f"filtering to occurred_at >= {args.since}")
        else:
            print("warning: no --since filter; pre-fix biased events included")
        print()

        counts = db.execute(
            "SELECT user_id, COUNT(*) FROM interaction_events GROUP BY user_id"
        )
        excluded = [(u, n) for u, n in counts if n < args.min_events]
        if excluded:
            print(
                f"excluded {len(excluded)} user(s) below --min-events "
                f"{args.min_events}: {excluded}"
            )

        user_id = args.user_id if args.user_id is not None else heaviest_user_id(db)
        print(f"user_id={user_id}")
        print()

        window = _load_window(db, user_id, since_ts)
        print("== 1. denominators ==")
        print(
            f"impressions={window.impressions} dwells={window.dwells} "
            f"article_opens={window.article_opens} "
            f"comments_opens={window.comments_opens} weeks={len(window.weeks)}"
        )
        print()

        print("== 2. position bias ==")
        positions = _position_stats(db, user_id, since_ts)
        print(f"{'sort_mode':>12} {'bucket':>6} {'n':>6} {'vote_rate':>9} {'up_rate':>7} {'open_rate':>9}")
        for p in positions:
            print(
                f"{p.sort_mode:>12} {p.bucket:>6} {p.impressions:>6} "
                f"{p.vote_rate:>9.2f} {p.up_rate:>7.2f} {p.open_rate:>9.2f}"
            )
        print()

        print("== 3. source mix (impressed vs. current pool) ==")
        pool_ids, _pool_emb = pool_embeddings(db, config, user_id)
        source_mix = _source_mix(db, user_id, since_ts, pool_ids)
        print(f"{'source':>12} {'impressed':>9} {'pool':>6}")
        for source, impressed, pool_n in source_mix:
            print(f"{source:>12} {impressed:>9} {pool_n:>6}")
        print()

        print("== 4. dwell <-> vote agreement (the B3 gate) ==")
        dwell_stats, auc = _dwell_vote_stats(db, user_id, since_ts, args.dwell_cap_ms)
        print(f"{'label':>8} {'n':>6} {'p50_ms':>8} {'p90_ms':>8}")
        for d in dwell_stats:
            print(f"{d.label:>8} {d.n:>6} {d.p50_ms:>8.0f} {d.p90_ms:>8.0f}")
        print(f"dwell rank-AUC (up vs down): {auc:.3f}")
        print()

        print("== 5. preference drift (the B1 gate) ==")
        drift = _drift_stat(db, config, user_id, args.drift_recent_days)
        if drift is None:
            print("not enough split data (need upvotes both inside and outside the recent window)")
        else:
            print(
                f"recent_n={drift.recent_n} older_n={drift.older_n} "
                f"centroid_cos_distance={drift.centroid_cos_distance:.3f}"
            )
        print()

        print("== recommendation ==")
        if drift is not None and drift.centroid_cos_distance >= DRIFT_MATERIAL_THRESHOLD:
            print(
                f"drift={drift.centroid_cos_distance:.3f} >= {DRIFT_MATERIAL_THRESHOLD} "
                "-> do B1 (time-decay) next as a one-flag eval_ranker_variants.py ablation."
            )
        elif not np.isnan(auc) and auc >= DWELL_AUC_USABLE_THRESHOLD:
            print(
                f"dwell AUC={auc:.3f} >= {DWELL_AUC_USABLE_THRESHOLD} "
                "-> do the B3 consumer (dwell as a sample_weight modifier) next."
            )
        else:
            print(
                "both B1 and B3 gates are null/insufficient on this window -> "
                "B2 (interleaving) is the only path to new signal, but it needs "
                "ranker_arm to stop being constant, which is a REF-2-gated "
                "instrumentation task, not an analysis one."
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
