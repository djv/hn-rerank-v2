#!/usr/bin/env python3
"""Prepare a model-specific embedding cache in a non-live database snapshot."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database
from pipeline import Config, Embedder, get_or_compute_embeddings, story_embedding_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="SQLite snapshot to update")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    config = Config.load(args.config)
    db = Database(args.db)
    try:
        with np.load(args.snapshot, allow_pickle=False) as data:
            required = {"story_ids", "text_hashes", "embeddings"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    "Embedding snapshot missing arrays: " + ", ".join(sorted(missing))
                )
            story_ids = np.asarray(data["story_ids"], dtype=np.int64)
            text_hashes = np.asarray(data["text_hashes"], dtype=str)
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        if embeddings.shape != (len(story_ids), 384):
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        if len(text_hashes) != len(story_ids):
            raise ValueError("Snapshot story IDs and text hashes have different lengths")
        if len(set(story_ids.tolist())) != len(story_ids):
            raise ValueError("Snapshot contains duplicate story IDs")
        if not np.isfinite(embeddings).all():
            raise ValueError("Snapshot embeddings contain non-finite values")

        imported = 0
        stale = 0
        absent = 0
        for start in range(0, len(story_ids), 500):
            batch_ids = [int(value) for value in story_ids[start : start + 500]]
            stories = {story.id: story for story in db.get_stories(batch_ids)}
            for offset, story_id in enumerate(batch_ids, start=start):
                story = stories.get(story_id)
                if story is None:
                    absent += 1
                    continue
                current_hash = hashlib.sha256(
                    story_embedding_text(story).encode("utf-8")
                ).hexdigest()
                if current_hash != text_hashes[offset]:
                    stale += 1
                    continue
                db.upsert_embedding(
                    story_id,
                    config.embedding_model_version,
                    current_hash,
                    embeddings[offset],
                )
                imported += 1

        user = db.get_user_by_token("default")
        if user is None:
            raise RuntimeError("Missing default user token")
        feedback_stories, _labels, _vote_times = db.get_feedback_for_training(user.id)
        embedder = Embedder(
            config.onnx_model_dir,
            model_version=config.embedding_model_version,
            max_tokens=config.embedding_max_tokens,
            batch_size=config.embedding_batch_size,
            ort_variant=config.embedding_ort_variant,
        )
        get_or_compute_embeddings(feedback_stories, embedder, db)
        print(
            f"prepared model={config.embedding_model_version} imported={imported} "
            f"stale={stale} absent={absent} feedback={len(feedback_stories)}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
