#!/usr/bin/env python3
"""Time embedding models on the laptop's Intel iGPU (OpenVINO) with real texts.

Encodes a sample of one user's voted stories (production embedding text,
truncated to --max-tokens) on the GPU and reports seconds per story plus
the worst cosine against onnxruntime CPU fp32 on a few texts, so f16 error
shows up. Reads a read-only snapshot; never writes to the DB. Needs the
embedding-experiment group. Run it with ``batch``.

Tuning on bge-base (2026-09-27): batch 4/8/16, padding to the longest text
or to multiples of 64, and static [8, 512] shapes all ran 0.24-0.27 s/story
(86% of texts hit 512 tokens); the THROUGHPUT hint segfaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config, story_embedding_text  # noqa: E402
from scripts.bakeoff_embedding_models import (  # noqa: E402
    POOLINGS,
    BakeoffModel,
    _encode,
    _model_paths,
    _runner,
)
from scripts.eval_ranker_variants import frozen_database  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument(
        "--gpu-device",
        choices=("gpu", "gpu-f32"),
        default="gpu",
        help="gpu-f32: full precision, for models that overflow f16",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face repo id, or a local export_embedding_onnx.py directory",
    )
    parser.add_argument("--onnx-file", default="onnx/model.onnx")
    parser.add_argument(
        "--cpu-onnx-file",
        help="fp32 file for the CPU check when --onnx-file is an fp16 export",
    )
    parser.add_argument("--pooling", choices=POOLINGS, default="mean")
    parser.add_argument("--prefix", default="", help="Prepended to every text")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sample", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-check", type=int, default=4)
    args = parser.parse_args()

    config = Config.load(args.config)
    with frozen_database(config.db_path) as (db, _snapshot_hash):
        stories, _labels, _times = db.get_feedback_for_training(user_id=args.user_id)
    rng = np.random.default_rng(0)
    picked = rng.choice(
        len(stories), size=min(args.sample, len(stories)), replace=False
    )
    texts = [args.prefix + story_embedding_text(stories[int(i)]) for i in picked]
    spec = BakeoffModel(
        name=args.repo.rsplit("/", 1)[-1],
        repo_id=None if Path(args.repo).is_dir() else args.repo,
        local_dir=args.repo if Path(args.repo).is_dir() else None,
        onnx_filename=args.onnx_file,
        pooling=args.pooling,
    )
    tokenizer_dir, model_path = _model_paths(spec)

    cpu_model_path = model_path
    if args.cpu_onnx_file:
        _, cpu_model_path = _model_paths(
            BakeoffModel(
                name=spec.name,
                repo_id=None if Path(args.repo).is_dir() else args.repo,
                local_dir=args.repo if Path(args.repo).is_dir() else None,
                onnx_filename=args.cpu_onnx_file,
                pooling=args.pooling,
            )
        )

    def encode(batch: list[str], device: str) -> tuple[np.ndarray, float]:
        return _encode(
            batch,
            tokenizer_dir=tokenizer_dir,
            model_path=cpu_model_path if device == "cpu" else model_path,
            pooling=args.pooling,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            device=args.gpu_device if device == "gpu" else "cpu",
        )

    # The first GPU pass includes kernel compilation per shape; time the second.
    encode(texts, "gpu")
    gpu, seconds = encode(texts, "gpu")
    # Free the GPU model before loading the CPU one: holding both ran the
    # laptop out of memory.
    _runner.cache_clear()
    line = f"{spec.name}: {seconds / len(texts):.3f}s/story on GPU"
    if args.cpu_check:
        check = texts[: args.cpu_check]
        cpu, cpu_seconds = encode(check, "cpu")
        cosines = (gpu[: len(check)] * cpu).sum(axis=1)
        line += (
            f", CPU {cpu_seconds / len(check):.3f}s/story, "
            f"min cos GPU vs CPU {cosines.min():.4f}"
        )
    print(line, flush=True)


if __name__ == "__main__":
    main()
