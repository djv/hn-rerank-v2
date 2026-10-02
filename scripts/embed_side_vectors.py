#!/usr/bin/env python3
"""Encode side-model vectors (embeddinggemma) for ranking candidates and voted
stories, for ``ModelConfig.side_embedding_enabled`` (pipeline/side_embeddings.py).

Voted stories go first, then candidates newest first, so new stories are
covered soon after each regen. Writes only the additive ``side_embeddings``
table; a story whose text changed is encoded again. Meant to run niced from
a timer on the VPS (0.22 s/story there at 128 tokens on 2 threads).

Usage:
    uv run python scripts/embed_side_vectors.py --dry-run
    nice -n 19 uv run python scripts/embed_side_vectors.py --limit 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database, Story  # noqa: E402
from pipeline import Config, load_production_candidate_stories  # noqa: E402
from pipeline.side_embeddings import (  # noqa: E402
    SideEmbedder,
    side_input,
    side_text_hash,
)


def stories_to_encode(db: Database, config: Config) -> tuple[list[Story], int]:
    """Stories without a current side vector, voted ones first, and the
    number considered."""
    voted, _, _ = db.get_feedback_for_training(user_id=None)
    candidates = load_production_candidate_stories(
        db, config, user_id=None, exclude_feedback=False
    )
    ordered: dict[int, Story] = {s.id: s for s in voted if s.title}
    for story in sorted(candidates, key=lambda s: s.time, reverse=True):
        ordered.setdefault(story.id, story)
    hashes = {sid: side_text_hash(s) for sid, s in ordered.items()}
    have = db.get_side_embeddings_batch(
        list(ordered),
        config.side_embedding_model_version,
        hashes,
        expected_dim=config.side_embedding_dim,
    )
    return [s for sid, s in ordered.items() if sid not in have], len(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode side-model vectors for candidates and voted stories."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--limit", type=int, default=2000, help="stories per run")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="count only")
    args = parser.parse_args()
    if args.limit <= 0 or args.threads <= 0 or args.batch_size <= 0:
        parser.error("--limit, --threads and --batch-size must be positive")

    config = Config.load(args.config)
    db = Database(config.db_path, read_only=args.dry_run)
    todo, considered = stories_to_encode(db, config)
    print(f"{len(todo)} of {considered} stories need a side vector", flush=True)
    if args.dry_run or not todo:
        return

    embedder = SideEmbedder(
        config.side_embedding_model_dir,
        max_tokens=config.side_embedding_max_tokens,
        threads=args.threads,
        batch_size=args.batch_size,
    )
    started = time.perf_counter()
    batch = todo[: args.limit]
    for start in range(0, len(batch), 32):
        chunk = batch[start : start + 32]
        vectors = embedder.encode([side_input(s) for s in chunk])
        if vectors.shape[1] != config.side_embedding_dim:
            raise ValueError(
                f"side model returned {vectors.shape[1]}-d vectors, "
                f"config expects {config.side_embedding_dim}"
            )
        db.upsert_side_embeddings(
            config.side_embedding_model_version,
            [(s.id, side_text_hash(s), v) for s, v in zip(chunk, vectors, strict=True)],
        )
        done = start + len(chunk)
        if done % 256 < 32 or done == len(batch):
            rate = (time.perf_counter() - started) / done
            print(f"encoded {done}/{len(batch)}: {rate:.3f} s/story", flush=True)


if __name__ == "__main__":
    main()
