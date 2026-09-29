#!/usr/bin/env python3
"""Encode one user's feedback stories with a candidate embedding model.

Writes an .npz (story_ids, text_hashes, embeddings, model) for
``eval_ranker_variants.py --candidate-pool heldout-feedback
--replay-embeddings``. Reads a read-only SQLite snapshot; never writes to
the dashboard database. Texts are production's ``story_embedding_text``,
truncated at --max-tokens, optionally prefixed (e.g. nomic's task prefix).

Encoding runs in pieces of --checkpoint-every stories. After each piece the
finished vectors go to ``<output stem>.partial.npz`` and a progress line
(count, s/story, ETA) is printed, so a stopped run resumes where it left
off. Vectors from --reuse files made with the same settings are taken
as-is when the story text is unchanged (same text hash).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from typing import Any

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Story  # noqa: E402
from pipeline import Config, story_embedding_text  # noqa: E402
from pipeline.embedding_sections import (  # noqa: E402
    STORY_SECTIONS,
    pool_sections,
    story_section_text,
)
from pipeline.ranking import clean_text  # noqa: E402
from scripts.bakeoff_embedding_models import (  # noqa: E402
    DEVICES,
    POOLINGS,
    BakeoffModel,
    _encode,
    _model_paths,
)
from scripts.eval_ranker_variants import (  # noqa: E402
    _embedding_text_hashes,
    frozen_database,
    sample_feedback_positions,
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


def load_reusable(paths: list[Path], model: str) -> dict[tuple[int, str], np.ndarray]:
    """(story id, text hash) -> vector from earlier files with the same
    settings string; files made with other settings are skipped."""
    found: dict[tuple[int, str], np.ndarray] = {}
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            if str(data["model"]) != model:
                print(f"not reusing {path}: made with other settings", flush=True)
                continue
            for story_id, text_hash, vector in zip(
                data["story_ids"], data["text_hashes"], data["embeddings"], strict=True
            ):
                found[(int(story_id), str(text_hash))] = vector
    return found


def split_budget_text(story: Story, tokenizer: Any, body_tokens: int) -> str:
    """Title + self text + article cut at ``body_tokens``, then the comments.

    Keeps room for the discussion in one fixed-length text, where the
    production order lets a long article crowd the comments out.
    """
    body = story_section_text(story, "body")
    ids = tokenizer(body, add_special_tokens=False)["input_ids"][:body_tokens]
    comments = clean_text(story.top_comments)
    head = tokenizer.decode(ids)
    return f"{head} {comments}".strip() if comments else head


def token_chunks(text: str, tokenizer: Any, size: int, max_chunks: int) -> list[str]:
    """Up to ``max_chunks`` consecutive pieces of ``size`` tokens (minus room
    for the tokenizer's special tokens); at least one piece."""
    step = max(size - 2, 1)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    pieces = [
        tokenizer.decode(ids[start : start + step])
        for start in range(0, len(ids), step)
    ][:max_chunks]
    return pieces or [text]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument(
        "--repo",
        help="Hugging Face repo id, or a local export_embedding_onnx.py directory",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Export the stored production embeddings instead of encoding",
    )
    parser.add_argument("--onnx-file", default="onnx/model.onnx")
    parser.add_argument("--pooling", choices=POOLINGS, default="mean")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prefix", default="", help="Prepended to every text")
    parser.add_argument(
        "--title-only",
        action="store_true",
        help="Embed only the title (hashes still track the full text)",
    )
    parser.add_argument(
        "--section",
        choices=STORY_SECTIONS,
        help="Embed one part of each story (pipeline/embedding_sections.py)",
    )
    parser.add_argument(
        "--split-body-tokens",
        type=int,
        help="One text: title + body cut at N tokens, then the comments "
        "(the whole text still stops at --max-tokens)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        help="Split each text into chunks of N tokens, embed each and average "
        "them (covers the whole text instead of its first --max-tokens)",
    )
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Encode only the first N (timing)")
    parser.add_argument(
        "--max-feedback-per-class",
        type=int,
        help="Encode only the votes eval_ranker_variants.py keeps with the same flag",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="cpu",
        help="gpu: OpenVINO on the Intel iGPU (needs the embedding-experiment group)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save finished vectors and print progress every N stories",
    )
    parser.add_argument(
        "--reuse",
        type=Path,
        action="append",
        default=[],
        help="Earlier output with the same settings; its unchanged stories "
        "are not encoded again (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.from_db and not args.repo:
        parser.error("--repo is required unless --from-db")
    config = Config.load(args.config)
    with frozen_database(config.db_path) as (db, _snapshot_hash):
        stories, labels, _times = db.get_feedback_for_training(user_id=args.user_id)
        if args.max_feedback_per_class is not None:
            keep = sample_feedback_positions(
                np.array(labels, dtype=int), args.max_feedback_per_class
            )
            stories = [stories[i] for i in keep]
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
        repo_id=None if Path(args.repo).is_dir() else args.repo,
        local_dir=args.repo if Path(args.repo).is_dir() else None,
        onnx_filename=args.onnx_file,
        pooling=args.pooling,
    )
    tokenizer_dir, model_path = _model_paths(spec)
    # "--section full" is the default text, so it combines with the others.
    section = None if args.section == "full" else args.section
    if sum(bool(x) for x in (section, args.title_only, args.split_body_tokens)) > 1:
        parser.error("--section, --title-only and --split-body-tokens are exclusive")
    model = (
        f"{args.repo}|{args.pooling}|{args.max_tokens}|prefix={args.prefix!r}"
        f"|title_only={args.title_only}|section={args.section}"
        f"|split_body_tokens={args.split_body_tokens}|chunk_tokens={args.chunk_tokens}"
        f"|max_chunks={args.max_chunks}|device={args.device}"
    )
    partial = args.output.with_name(f"{args.output.stem}.partial.npz")
    reusable = load_reusable([*args.reuse, partial], model)
    vectors: list[np.ndarray | None] = [
        reusable.get((s.id, str(h))) for s, h in zip(stories, hashes, strict=True)
    ]
    todo = [i for i, vector in enumerate(vectors) if vector is None]
    print(
        f"{len(stories) - len(todo)} of {len(stories)} stories already encoded; "
        f"encoding {len(todo)}",
        flush=True,
    )
    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_dir)

    def text_of(story: Story) -> str:
        if section:
            body = story_section_text(story, section)
        elif args.title_only:
            body = story.title
        elif args.split_body_tokens:
            body = split_budget_text(story, tokenizer, args.split_body_tokens)
        else:
            body = story_embedding_text(story)
        return args.prefix + body

    def chunks_of(story: Story) -> list[str]:
        text = text_of(story)
        if args.chunk_tokens:
            return token_chunks(text, tokenizer, args.chunk_tokens, args.max_chunks)
        return [text]

    def save_finished(path: Path) -> None:
        done = [(i, v) for i, v in enumerate(vectors) if v is not None]
        if not done:
            return
        tmp = path.with_name(f"{path.stem}.tmp.npz")
        _save(tmp, [stories[i] for i, _ in done], np.stack([v for _, v in done]), model)
        os.replace(tmp, path)

    # Longest stories first, so a GPU that cannot fit them fails in the first
    # piece rather than after most of the run.
    todo.sort(key=lambda i: -len(text_of(stories[i])))
    seconds = 0.0
    step = max(args.checkpoint_every, 1)
    for start in range(0, len(todo), step):
        piece = todo[start : start + step]
        chunks = [chunks_of(stories[i]) for i in piece]
        offsets = np.cumsum([0] + [len(c) for c in chunks])
        flat, took = _encode(
            [chunk for story_chunks in chunks for chunk in story_chunks],
            tokenizer_dir=tokenizer_dir,
            model_path=model_path,
            pooling=args.pooling,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=args.device,
        )
        for j, i in enumerate(piece):
            vectors[i] = pool_sections(flat[offsets[j] : offsets[j + 1]])
        seconds += took
        save_finished(partial)
        done = start + len(piece)
        rate = seconds / done
        print(
            f"encoded {done}/{len(todo)}: {rate:.2f}s/story, "
            f"ETA {rate * (len(todo) - done) / 60:.0f} min",
            flush=True,
        )
    if not stories:
        raise SystemExit("no stories to encode")
    save_finished(args.output)
    partial.unlink(missing_ok=True)
    print(
        f"wrote {args.output}: {len(stories)} stories, {len(todo)} newly encoded "
        f"in {seconds:.0f}s ({seconds / max(len(todo), 1):.3f}s/story)"
    )


if __name__ == "__main__":
    main()
