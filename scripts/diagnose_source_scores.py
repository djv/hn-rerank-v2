"""Offline candidate-feature counterfactuals; use a disposable DB snapshot only."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config, Embedder, load_production_candidate_stories  # noqa: E402
from pipeline.ranking import _score_and_rank, get_or_compute_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-db", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    db = Database(args.snapshot_db)
    try:
        config = Config.load()
        embedder = Embedder(args.model_dir)
        stories = load_production_candidate_stories(
            db, config, user_id=args.user_id, exclude_feedback=True
        )
        embeddings = get_or_compute_embeddings(stories, embedder, db)
        lw_ids = {s.id for s in stories if s.source == "rss_lesswrong_com"}
        hn_lengths = [len(s.text_content) for s in stories if s.source == "hn"]
        target_length = int(statistics.median(hn_lengths)) if hn_lengths else 1000
        scores_by_variant: dict[str, dict[int, float]] = {}
        for variant in ("baseline", "hn_source", "hn_length", "hn_source_and_length"):
            changed = [
                replace(
                    s,
                    source="hn" if "source" in variant else s.source,
                    text_content="x" * target_length
                    if "length" in variant
                    else s.text_content,
                )
                if s.id in lw_ids and variant != "baseline"
                else s
                for s in stories
            ]
            # Training rows and normalized embeddings are identical across variants.
            ranked = _score_and_rank(
                changed, embeddings, db, config, embedder, user_id=args.user_id
            )
            scores = {r.story.id: float(r.score) for r in ranked}
            scores_by_variant[variant] = scores
            values = [scores[sid] for sid in lw_ids]
            baseline = scores_by_variant["baseline"]
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "lw_candidates": len(lw_ids),
                        "lw_median_score": statistics.median(values)
                        if values
                        else None,
                        "lw_median_delta": statistics.median(
                            [scores[sid] - baseline[sid] for sid in lw_ids]
                        )
                        if values
                        else None,
                        "hn_median_text_length": target_length,
                        "lw_median_text_length": statistics.median(
                            [len(s.text_content) for s in stories if s.id in lw_ids]
                        )
                        if lw_ids
                        else None,
                    }
                )
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
