from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import (  # noqa: E402
    Config,
    Embedder,
    RankTrace,
    fast_rerank_for_user,
)


def _heaviest_user_id(db: Database) -> int:
    rows = db.execute(
        "SELECT user_id, COUNT(*) AS n FROM feedback "
        "GROUP BY user_id ORDER BY n DESC LIMIT 1"
    )
    if not rows:
        raise SystemExit("No feedback rows found; nothing to sweep.")
    return int(rows[0][0])


def _rank_top50_labels(
    db: Database, config: Config, embedder: Embedder, user_id: int
) -> tuple[dict[int, str], dict[str, int | float | str]]:
    """Run one rank, return {id: badge_label} for the top 50 plus the
    trace fields. badge_label is "primary" for items with no discovery
    badge, otherwise a comma-joined list of badge names."""
    trace = RankTrace()
    ranked = fast_rerank_for_user(db, config, embedder, user_id, trace=trace)
    out: dict[int, str] = {}
    for r in ranked[:50]:
        badges: list[str] = []
        if r.is_novel:
            badges.append("novel")
        if r.is_similar:
            badges.append("similar")
        if r.is_uncertain:
            badges.append("uncertain")
        if r.is_non_hn:
            badges.append("non_hn")
        if r.is_hot:
            badges.append("hot")
        if r.is_high_engagement:
            badges.append("hi_eng")
        if r.is_discussion_rich:
            badges.append("discuss")
        out[r.story.id] = ",".join(badges) if badges else "primary"
    return out, trace.to_log_fields()


def _classify_loss(
    lost_ids: set[int], uncapped_labels: dict[int, str]
) -> dict[str, int]:
    """Bucket lost stories by which discovery badge group they had in
    the uncapped run. Stories in the primary bucket have no badge."""
    counts = {
        "pri": 0,
        "sim": 0,
        "unc": 0,
        "novel": 0,
        "nonhn": 0,
        "other": 0,
    }
    for sid in lost_ids:
        label = uncapped_labels.get(sid, "primary")
        if label == "primary":
            counts["pri"] += 1
        else:
            if "similar" in label:
                counts["sim"] += 1
            if "uncertain" in label:
                counts["unc"] += 1
            if "novel" in label:
                counts["novel"] += 1
            if "non_hn" in label:
                counts["nonhn"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the recent_candidate_hn_limit across a range of "
            "values and report the top-50 overlap with the uncapped "
            "run, broken down by badge group."
        )
    )
    parser.add_argument("--db", default="hn_rewrite.db")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int)
    parser.add_argument(
        "--hn-caps",
        type=int,
        nargs="+",
        default=[1000, 1500, 2000, 2500, 3000, 5000, 10000],
        help="HN cap values to sweep (RSS fixed at --rss-limit).",
    )
    parser.add_argument("--rss-limit", type=int, default=500)
    parser.add_argument("--uncapped", type=int, default=100_000)
    args = parser.parse_args()

    base = Config.load(args.config)
    uncapped_config = replace(
        base,
        recent_candidate_hn_limit=args.uncapped,
        recent_candidate_rss_limit=args.uncapped,
    )

    db = Database(args.db, read_only=False)
    try:
        user_id = args.user_id if args.user_id is not None else _heaviest_user_id(db)
        embedder = Embedder(base.onnx_model_dir)

        # warmup
        print("warmup ...", end="", flush=True)
        _rank_top50_labels(db, uncapped_config, embedder, user_id)
        print(" done\n")

        # ground truth
        uncapped_labels, uf = _rank_top50_labels(db, uncapped_config, embedder, user_id)
        uncapped_ids = set(uncapped_labels)
        print(
            f"ground-truth (hn=rss={args.uncapped}): "
            f"top50={len(uncapped_ids)} "
            f"candidates={uf.get('candidates', '?')}\n"
        )

        header = (
            f"{'hn_cap':>8} {'shared':>7} {'lost':>5} "
            f"{'L_pri':>6} {'L_sim':>6} {'L_unc':>6} {'L_novel':>8} {'L_nonhn':>8} "
            f"{'cand':>6}"
        )
        print(header)

        for cap in args.hn_caps:
            capped_config = replace(
                base,
                recent_candidate_hn_limit=cap,
                recent_candidate_rss_limit=args.rss_limit,
            )
            capped_labels, cf = _rank_top50_labels(db, capped_config, embedder, user_id)
            capped_ids = set(capped_labels)
            shared = uncapped_ids & capped_ids
            lost_ids = uncapped_ids - capped_ids
            counts = _classify_loss(lost_ids, uncapped_labels)
            print(
                f"{cap:>8} {len(shared):>7} {len(lost_ids):>5} "
                f"{counts['pri']:>6} {counts['sim']:>6} {counts['unc']:>6} "
                f"{counts['novel']:>8} {counts['nonhn']:>8} "
                f"{cf.get('candidates', '?'):>6}"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
