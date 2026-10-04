#!/usr/bin/env python3
"""Combine encode_replay_embeddings.py files into one, for section strategies.

``mean``: weighted average of the per-story vectors, renormalized (one
vector the size of each part). ``concat``: parts side by side, part i scaled
by sqrt(weight_i) and the whole renormalized, so weight is each part's share
of the squared norm (equal weights match repeating --replay-embeddings in
eval_ranker_variants.py). Every file must hold the same stories with the
same text hashes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def combine(
    parts: list[NDArray[np.float32]], weights: list[float], mode: str
) -> NDArray[np.float32]:
    if len(parts) != len(weights) or not parts:
        raise ValueError("Need one weight per part")
    if any(w <= 0 for w in weights):
        raise ValueError("Weights must be positive")
    if mode == "mean":
        combined = sum(w * p for w, p in zip(weights, parts, strict=True))
    else:
        combined = np.concatenate(
            [np.sqrt(w) * p for w, p in zip(weights, parts, strict=True)], axis=1
        )
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return np.asarray(combined / np.clip(norms, 1e-12, None), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--weights", help="Comma-separated; default equal")
    parser.add_argument("--mode", choices=("mean", "concat"), default="mean")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights = (
        [float(w) for w in args.weights.split(",")]
        if args.weights
        else [1.0] * len(args.inputs)
    )
    loaded = [np.load(path, allow_pickle=False) for path in args.inputs]
    ids, hashes = loaded[0]["story_ids"], loaded[0]["text_hashes"]
    for path, data in zip(args.inputs, loaded, strict=True):
        if not (
            np.array_equal(data["story_ids"], ids)
            and np.array_equal(data["text_hashes"], hashes)
        ):
            raise SystemExit(f"{path} holds different stories or texts")
    parts = [np.asarray(data["embeddings"], dtype=np.float32) for data in loaded]
    vectors = combine(parts, weights, args.mode)
    np.savez(
        args.output,
        story_ids=ids,
        text_hashes=hashes,
        embeddings=vectors,
        model=np.array(
            f"{args.mode}({','.join(str(p) for p in args.inputs)};"
            f"weights={','.join(map(str, weights))})"
        ),
    )
    print(f"wrote {args.output}: {vectors.shape[0]} x {vectors.shape[1]}")


if __name__ == "__main__":
    main()
