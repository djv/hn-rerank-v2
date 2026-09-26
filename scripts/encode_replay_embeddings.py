#!/usr/bin/env python3
"""Encode one user's feedback stories with a candidate embedding model.

Writes an .npz (story_ids, text_hashes, embeddings, model) for
``eval_ranker_variants.py --candidate-pool heldout-feedback
--replay-embeddings``. Reads a read-only SQLite snapshot; never writes to
the dashboard database. Texts are production's ``story_embedding_text``,
truncated at --max-tokens, optionally prefixed (e.g. nomic's task prefix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Story  # noqa: E402
from pipeline import Config, story_embedding_text  # noqa: E402
from scripts.bakeoff_embedding_models import (  # noqa: E402
    BakeoffModel,
    _encode,
    _model_paths,
)
from scripts.eval_ranker_variants import (  # noqa: E402
    _embedding_text_hashes,
    frozen_database,
)


def _save(path: Path, stories: list[Story], vectors: np.ndarray, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        story_ids=np.array([s.id for s in stories], dtype=np.int64),
        text_hashes=_embedding_text_hashes(stories),
        embeddings=vectors.astype(np.float32),
        model=np.array(model),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--repo", help="Hugging Face repo id")
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Export the stored production embeddings instead of encoding",
    )
    parser.add_argument("--onnx-file", default="onnx/model.onnx")
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prefix", default="", help="Prepended to every text")
    parser.add_argument(
        "--title-only",
        action="store_true",
        help="Embed only the title (hashes still track the full text)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Encode only the first N (timing)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.from_db and not args.repo:
        parser.error("--repo is required unless --from-db")
    config = Config.load(args.config)
    with frozen_database(config.db_path) as (db, _snapshot_hash):
        stories, _labels, _times = db.get_feedback_for_training(user_id=args.user_id)
        if args.limit is not None:
            stories = stories[: args.limit]
        hashes = _embedding_text_hashes(stories)
        stored = (
            db.get_embeddings_batch(
                [s.id for s in stories],
                config.embedding_model_version,
                dict(zip([s.id for s in stories], hashes, strict=True)),
            )
            if args.from_db
            else {}
        )
    if args.from_db:
        missing = [s.id for s in stories if s.id not in stored]
        if missing:
            raise SystemExit(f"{len(missing)} stories have no stored embedding")
        _save(
            args.output,
            stories,
            np.stack([stored[s.id] for s in stories]),
            f"stored:{config.embedding_model_version}",
        )
        print(f"wrote {args.output}: {len(stories)} stored embeddings")
        return
    spec = BakeoffModel(
        name=args.repo.rsplit("/", 1)[-1],
        repo_id=args.repo,
        onnx_filename=args.onnx_file,
        pooling=args.pooling,
    )
    tokenizer_dir, model_path = _model_paths(spec)
    texts = [
        args.prefix + (s.title if args.title_only else story_embedding_text(s))
        for s in stories
    ]
    vectors, seconds = _encode(
        texts,
        tokenizer_dir=tokenizer_dir,
        model_path=model_path,
        pooling=args.pooling,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    _save(
        args.output,
        stories,
        vectors,
        f"{args.repo}|{args.pooling}|{args.max_tokens}|prefix={args.prefix!r}"
        f"|title_only={args.title_only}",
    )
    print(
        f"wrote {args.output}: {len(stories)} x {vectors.shape[1]} "
        f"in {seconds:.0f}s ({seconds / max(len(stories), 1):.3f}s/story)"
    )


if __name__ == "__main__":
    main()
