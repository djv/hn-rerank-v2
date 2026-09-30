"""Print a profile's served Explore picks per window, with each Interest
pick's interest; disposable DB snapshot only (embeddings get written)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config, Embedder, load_production_candidate_stories  # noqa: E402
from pipeline.ranking import (  # noqa: E402
    get_or_compute_embeddings,
    interest_centers,
    rerank_candidates,
    serve_window,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-db", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--windows", nargs="*", default=["1d", "1w"])
    args = parser.parse_args()
    db = Database(args.snapshot_db)
    config = Config.load()
    embedder = Embedder(
        config.onnx_model_dir,
        model_version=config.embedding_model_version,
        max_tokens=config.embedding_max_tokens,
        batch_size=config.embedding_batch_size,
    )
    stories = load_production_candidate_stories(
        db, config, user_id=args.user_id, exclude_feedback=True
    )
    embeddings = get_or_compute_embeddings(stories, embedder, db)
    deck = rerank_candidates(
        db, config, embedder, stories, embeddings, user_id=args.user_id
    )

    feedback, labels, _ = db.get_feedback_for_training(user_id=args.user_id)
    ups = [story for story, label in zip(feedback, labels) if label == 2]
    up_embs = get_or_compute_embeddings(ups, embedder, db)
    centers = interest_centers(args.user_id, up_embs, config.model.interest_cluster_k)
    if not len(centers):
        print("No interests (no upvotes).")
        return
    up_sims = up_embs @ centers.T
    up_interest = np.argmax(up_sims, axis=1)
    names = []
    for c in range(len(centers)):
        members = np.flatnonzero(up_interest == c)
        nearest = members[np.argsort(-up_sims[members, c])][:2]
        names.append(
            f"[{len(members)} ups] " + " | ".join(ups[i].title[:30] for i in nearest)
        )
    row_of = {story.id: i for i, story in enumerate(stories)}

    now = time.time()
    for window in args.windows:
        served = serve_window(deck.window(window), window, now)
        print(f"=== {window}")
        for r in served.explore:
            badge = (
                "unsure"
                if r.is_uncertain
                else "novel"
                if r.is_novel
                else "interest"
                if r.is_interest
                else "?"
            )
            line = f"{badge:8} {r.score:+.3f} {r.story.title[:60]}"
            if r.is_interest:
                c = int(np.argmax(embeddings[row_of[r.story.id]] @ centers.T))
                line += f"\n         <- {names[c]}"
            print(line)


if __name__ == "__main__":
    main()
