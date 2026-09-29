#!/usr/bin/env python3
"""Find why an ONNX embedding model gives NaN at f16 on the laptop iGPU.

1. On the CPU at f32 (OpenVINO), records the largest |value| of every Add and
   MatMul output for a few real texts; anything past 65504 (the f16 maximum)
   overflows on the GPU.
2. On the GPU at f16, encodes the same texts with each
   ``ACTIVATIONS_SCALE_FACTOR`` in ``--scales`` (0 = unset) and prints the
   cosine against the CPU f32 vectors, plus how many came out NaN.

Reads a read-only snapshot; never writes to the DB. Needs the
embedding-experiment group. Run it with ``batch``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config, story_embedding_text  # noqa: E402
from scripts.bakeoff_embedding_models import (  # noqa: E402
    POOLINGS,
    BakeoffModel,
    _empty_cache,
    _model_inputs,
    _model_paths,
    _pool,
)
from scripts.eval_ranker_variants import frozen_database  # noqa: E402

F16_MAX = 65504.0


def _cache_spec(model: Any) -> list[tuple[str, int, int, Any]]:
    return [
        (
            port.get_any_name(),
            port.get_partial_shape()[1].get_length(),
            port.get_partial_shape()[3].get_length(),
            port.get_element_type().to_dtype(),
        )
        for port in model.inputs
        if port.get_any_name().startswith("past_key_values")
    ]


def _encode(
    core: Any,
    model_path: Path,
    device: str,
    config: dict[str, Any],
    feeds: list[tuple[dict[str, NDArray[Any]], NDArray[Any]]],
    pooling: Any,
) -> NDArray[np.float32]:
    model = core.read_model(str(model_path))
    cache = _cache_spec(model)
    compiled = core.compile_model(model, device, config)
    request = compiled.create_infer_request()
    output = next(
        port
        for port in compiled.outputs
        if not port.get_any_name().startswith("present")
    )
    vectors = []
    for feed, mask in feeds:
        results = request.infer(feed | _empty_cache(cache, feed))
        name = output.get_any_name()
        vectors.append(_pool({name: np.array(results[output])}, mask, pooling)[0])
    return np.stack(vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--repo", required=True, help="Local export directory")
    parser.add_argument("--onnx-file", default="onnx/model.onnx")
    parser.add_argument("--pooling", choices=POOLINGS, default="last")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sample", type=int, default=6)
    parser.add_argument("--scales", default="0,8,32,128")
    parser.add_argument("--skip-trace", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    with frozen_database(config.db_path) as (db, _snapshot_hash):
        stories, _labels, _times = db.get_feedback_for_training(user_id=args.user_id)
    rng = np.random.default_rng(0)
    picked = rng.choice(len(stories), size=args.sample, replace=False)
    texts = [args.prefix + story_embedding_text(stories[int(i)]) for i in picked]
    tokenizer_dir, model_path = _model_paths(
        BakeoffModel(
            name="debug",
            repo_id=None,
            local_dir=args.repo,
            onnx_filename=args.onnx_file,
            pooling=args.pooling,
        )
    )
    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_dir)
    openvino: Any = importlib.import_module("openvino")
    core = openvino.Core()
    input_names = {
        port.get_any_name() for port in core.read_model(str(model_path)).inputs
    }
    # One text per request: no padding, so every value is a real activation.
    feeds = []
    for text in texts:
        encoded = tokenizer(
            [text], truncation=True, max_length=args.max_tokens, return_tensors="np"
        )
        feeds.append((_model_inputs(encoded, input_names), encoded["attention_mask"]))
    lengths = [int(mask.sum()) for _feed, mask in feeds]
    print(f"texts: {len(texts)}, tokens {min(lengths)}-{max(lengths)}", flush=True)

    if not args.skip_trace:
        model = core.read_model(str(model_path))
        traced = [
            node.output(0)
            for node in model.get_ordered_ops()
            if node.get_type_name() in ("Add", "MatMul")
            and node.output(0).get_element_type().is_real()
            and node.output(0).get_partial_shape().rank.get_length() == 3
        ]
        names = [output.get_node().get_friendly_name() for output in traced]
        model.add_outputs(traced)
        cache = _cache_spec(model)
        compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        request = compiled.create_infer_request()
        peaks: dict[str, float] = {}
        for feed, _mask in feeds:
            results = request.infer(feed | _empty_cache(cache, feed))
            for name, port in zip(names, compiled.outputs[-len(traced) :]):
                value = float(np.abs(np.array(results[port])).max())
                peaks[name] = max(peaks.get(name, 0.0), value)
        del request, compiled, model
        over = {name: peak for name, peak in peaks.items() if peak > F16_MAX}
        print(
            f"traced {len(peaks)} Add/MatMul outputs; peak |value| "
            f"{max(peaks.values()):.0f}; {len(over)} exceed f16 max",
            flush=True,
        )
        for name, peak in sorted(peaks.items(), key=lambda item: -item[1])[:12]:
            print(f"  {peak:12.0f}  {name}", flush=True)

    reference = _encode(
        core,
        model_path,
        "CPU",
        {"INFERENCE_PRECISION_HINT": "f32"},
        feeds,
        args.pooling,
    )
    for scale in (float(value) for value in args.scales.split(",")):
        gpu_config: dict[str, Any] = {"INFERENCE_PRECISION_HINT": "f16"}
        if scale:
            gpu_config["ACTIVATIONS_SCALE_FACTOR"] = scale
        try:
            vectors = _encode(core, model_path, "GPU", gpu_config, feeds, args.pooling)
        except RuntimeError as error:
            print(f"scale {scale:g}: failed: {str(error).splitlines()[0]}", flush=True)
            continue
        nan = int(np.isnan(vectors).any(axis=1).sum())
        cosines = (np.nan_to_num(vectors) * reference).sum(axis=1)
        print(
            f"scale {scale:g}: NaN rows {nan}/{len(feeds)}, cos vs CPU f32 "
            f"min {cosines.min():.4f} mean {cosines.mean():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
