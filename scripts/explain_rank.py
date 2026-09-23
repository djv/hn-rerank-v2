"""Explain per-story ranking; disposable read-only DB snapshot only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config, Embedder, load_production_candidate_stories  # noqa: E402
from pipeline.ranking import (  # noqa: E402
    RankScoreContext,
    _score_and_rank,
    get_or_compute_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-db", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--model-dir", required=False, default=None)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--story-ids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    db = Database(args.snapshot_db)
    try:
        config = Config.load()
        embedder = Embedder(
            args.model_dir or config.onnx_model_dir,
            model_version=config.embedding_model_version,
            max_tokens=config.embedding_max_tokens,
            batch_size=config.embedding_batch_size,
        )
        stories = load_production_candidate_stories(
            db, config, user_id=args.user_id, exclude_feedback=True
        )
        embeddings = get_or_compute_embeddings(stories, embedder, db)
        ctx = RankScoreContext()
        ranked = _score_and_rank(
            stories,
            embeddings,
            db,
            config,
            embedder,
            user_id=args.user_id,
            score_context=ctx,
        )
        by_id = {
            s.id: (pos, r) for pos, r in enumerate(ranked, start=1) for s in [r.story]
        }
        targets = args.story_ids or [r.story.id for r in ranked[: args.top_n]]
        index_by_id = {s.id: i for i, s in enumerate(stories)}
        for sid in targets:
            if sid not in by_id:
                print(f"{sid}: not in ranked pool")
                continue
            pos, r = by_id[sid]
            i = index_by_id[sid]
            up_sim = (
                float(ctx.cand_closest_up[i])
                if ctx.cand_closest_up is not None
                else float("nan")
            )
            down_sim = (
                float(ctx.cand_closest_down[i])
                if ctx.cand_closest_down is not None
                else float("nan")
            )
            match = ""
            if ctx.cand_closest_up_idx is not None and ctx.fb_up_titles:
                fb_row = int(ctx.cand_closest_up_idx[i])
                if 0 <= fb_row < len(ctx.fb_up_titles):
                    match = ctx.fb_up_titles[fb_row][:70]
            print(f"#{pos} score={r.score:.4f} src={r.story.source}")
            print(f"   {r.story.title[:80]}")
            print(
                f"   len={len(r.story.text_content)} closest_up={up_sim:.3f} closest_down={down_sim:.3f}"
            )
            print(f"   because-you-upvoted: {match}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
