from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

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
        raise SystemExit("No feedback rows found; nothing to compare.")
    return int(rows[0][0])


def _rank_top50(
    db: Database, config: Config, embedder: Embedder, user_id: int
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Run one rank and return {id: {title, source, score, badges}} for
    the top 50, plus the trace fields dict."""
    trace = RankTrace()
    ranked = fast_rerank_for_user(db, config, embedder, user_id, trace=trace)
    out: dict[int, dict[str, Any]] = {}
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
        out[r.story.id] = {
            "title": r.story.title,
            "source": r.story.source,
            "score": r.score,
            "badges": badges,
        }
    return out, trace.to_log_fields()


def _print_diff(
    uncapped: dict[int, dict[str, Any]],
    capped: dict[int, dict[str, Any]],
) -> None:
    shared = set(uncapped) & set(capped)
    lost = set(uncapped) - set(capped)
    gained = set(capped) - set(uncapped)
    print(f"shared: {len(shared)}")
    print(f"lost:   {len(lost)}")
    print(f"gained: {len(gained)}")

    if lost:
        print("\n--- lost from top 50 (uncapped -> capped) ---")
        for sid in sorted(lost):
            item = uncapped[sid]
            badge_str = ",".join(item["badges"]) if item["badges"] else "primary"
            print(
                f"  id={sid} source={item['source']} score={item['score']:.3f} "
                f"badges=[{badge_str}]  {item['title'][:80]}"
            )
    if gained:
        print("\n--- gained in top 50 (uncapped -> capped) ---")
        for sid in sorted(gained):
            item = capped[sid]
            badge_str = ",".join(item["badges"]) if item["badges"] else "primary"
            print(
                f"  id={sid} source={item['source']} score={item['score']:.3f} "
                f"badges=[{badge_str}]  {item['title'][:80]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the user's top 50 ranked stories with the candidate "
            "cap enabled vs uncapped, to see which stories the cap drops."
        )
    )
    parser.add_argument("--db", default="hn_rewrite.db")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--hn-limit", type=int, default=5000)
    parser.add_argument("--rss-limit", type=int, default=500)
    parser.add_argument("--uncapped", type=int, default=100_000)
    args = parser.parse_args()

    base = Config.load(args.config)
    uncapped_config = replace(
        base,
        recent_candidate_hn_limit=args.uncapped,
        recent_candidate_rss_limit=args.uncapped,
    )
    capped_config = replace(
        base,
        recent_candidate_hn_limit=args.hn_limit,
        recent_candidate_rss_limit=args.rss_limit,
    )

    db = Database(args.db, read_only=False)
    try:
        user_id = args.user_id if args.user_id is not None else _heaviest_user_id(db)
        embedder = Embedder(
            base.onnx_model_dir,
            batch_size=base.embedding_batch_size,
            ort_variant=base.embedding_ort_variant,
        )

        # warmup — populates the model cache and embeddings
        print("warmup ...", end="", flush=True)
        _rank_top50(db, uncapped_config, embedder, user_id)
        print(" done\n")

        uncapped, uf = _rank_top50(db, uncapped_config, embedder, user_id)
        print(
            f"uncapped (hn=rss={args.uncapped}): "
            f"top50={len(uncapped)} candidates={uf.get('candidates', '?')}"
        )

        capped, cf = _rank_top50(db, capped_config, embedder, user_id)
        print(
            f"capped   (hn={args.hn_limit}, rss={args.rss_limit}): "
            f"top50={len(capped)} candidates={cf.get('candidates', '?')}"
        )

        _print_diff(uncapped, capped)
        print(
            f"\ncandidates: {uf.get('candidates', '?')} (uncapped) -> "
            f"{cf.get('candidates', '?')} (capped)"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
