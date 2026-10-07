#!/usr/bin/env python3
"""Find where an ONNX embedding model's OpenVINO output departs from onnxruntime.

1. Encodes a few real texts with onnxruntime on the CPU (the reference),
   OpenVINO CPU f32, and OpenVINO GPU at f32 and f16, and prints each one's
   cosine to the reference. If OpenVINO's CPU is off too, the GPU is not to
   blame: OpenVINO's ONNX import or one of its ops differs.
2. --trace: runs one short text through onnxruntime and OpenVINO CPU f32 with
   every float tensor exposed, matches the tensors by their ONNX names and
   prints the first nodes in graph order whose relative error passes --tol,
   then the worst error per op type. Needs the ``onnx`` package, which is not
   a project dependency: ``uv run --group embedding-experiment --with onnx``.
   Writes ``<model>.debug-taps.onnx`` (graph only; it reuses the weights
   file) next to the model.

Reads a read-only snapshot; never writes to the DB. Needs the
embedding-experiment group. Run it with ``batch``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config, story_embedding_text  # noqa: E402
from scripts.bakeoff_embedding_models import (  # noqa: E402
    MEDIA_INPUTS,
    POOLINGS,
    _empty_media,
    _model_inputs,
    _normalize,
    _pool,
    _runner,
)
from scripts.eval_ranker_variants import frozen_database  # noqa: E402

Feed = dict[str, NDArray[Any]]
SKIPPED_OPS = {"Constant", "Parameter", "Result", "ShapeOf"}


def _compile(core: Any, model: Any, device: str, precision: str) -> Any:
    config: dict[str, Any] = {"INFERENCE_PRECISION_HINT": precision}
    if device == "CPU":
        config["INFERENCE_NUM_THREADS"] = 2
    return core.compile_model(model, device, config)


def _media(model: Any) -> Feed:
    return _empty_media(
        (port.get_any_name(), port.get_partial_shape()[-1].get_length())
        for port in model.inputs
        if port.get_any_name() in MEDIA_INPUTS
    )


def _vectors(
    core: Any,
    model_path: Path,
    device: str,
    precision: str,
    feeds: list[tuple[Feed, NDArray[Any]]],
    pooling: Any,
) -> NDArray[np.float32]:
    model = core.read_model(str(model_path))
    media = _media(model)
    request = _compile(core, model, device, precision).create_infer_request()
    rows = []
    for feed, mask in feeds:
        results = request.infer(feed | media)
        outputs = {
            port.get_any_name(): np.array(value) for port, value in results.items()
        }
        rows.append(_pool(outputs, mask, pooling)[0])
    return np.stack(rows)


def _openvino_tensors(
    core: Any, model_path: Path, feed: Feed
) -> dict[str, NDArray[Any]]:
    """Every f32 tensor OpenVINO CPU computes, keyed by each of its ONNX names."""
    model = core.read_model(str(model_path))
    media = _media(model)
    taps = [
        output
        for node in model.get_ordered_ops()
        if node.get_type_name() not in SKIPPED_OPS
        for output in node.outputs()
        if output.get_element_type().get_type_name() == "f32"
    ]
    model.add_outputs(taps)
    compiled = _compile(core, model, "CPU", "f32")
    results = compiled.create_infer_request().infer(feed | media)
    values: dict[str, NDArray[Any]] = {}
    for port in compiled.outputs:
        value = np.array(results[port])
        for name in port.get_names():
            values[name] = value
    return values


def _onnxruntime_tensors(
    model_path: Path, wanted: set[str], feed: Feed
) -> tuple[dict[str, NDArray[Any]], dict[str, str]]:
    """The same tensors from onnxruntime, plus each tensor's producing op type."""
    onnx: Any = importlib.import_module("onnx")
    proto = onnx.load(str(model_path), load_external_data=False)
    existing = {output.name for output in proto.graph.output}
    producers: dict[str, str] = {}
    for node in proto.graph.node:
        for name in node.output:
            if name and name in wanted:
                producers[name] = f"{node.op_type} {node.name}"
    proto.graph.output.extend(
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        for name in producers
        if name not in existing
    )
    # Same directory, so the relative weights path in the graph still resolves.
    debug_path = model_path.with_name(model_path.stem + ".debug-taps.onnx")
    onnx.save(proto, str(debug_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(debug_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = {meta.name for meta in session.get_inputs()}
    media = {
        meta.name: np.zeros((0, int(meta.shape[-1])), dtype=np.float32)
        for meta in session.get_inputs()
        if meta.name in MEDIA_INPUTS
    }
    names = [meta.name for meta in session.get_outputs()]
    values = session.run(names, {k: v for k, v in feed.items() if k in inputs} | media)
    return dict(zip(names, (np.asarray(v) for v in values), strict=True)), producers


def _trace(core: Any, model_path: Path, feed: Feed, tol: float) -> None:
    ov_values = _openvino_tensors(core, model_path, feed)
    ort_values, producers = _onnxruntime_tensors(model_path, set(ov_values), feed)
    for name in ("sentence_embedding",):
        if name in ov_values and name in ort_values:
            a = _normalize(ort_values[name].astype(np.float32))
            b = _normalize(ov_values[name].astype(np.float32))
            print(f"traced run: {name} cos {float((a * b).sum()):.4f}", flush=True)
    first: list[str] = []
    by_kind: dict[str, list[float]] = defaultdict(list)
    compared = 0
    for name, producer in producers.items():
        a, b = ort_values[name], ov_values[name]
        kind = producer.split(" ", 1)[0]
        if a.shape != b.shape:
            if a.size == b.size:
                b = b.reshape(a.shape)
            else:
                first.append(f"  shape {a.shape} vs {b.shape}  {producer}")
                continue
        if a.size == 0:
            continue
        compared += 1
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        rel = float(np.linalg.norm(b64 - a64) / (np.linalg.norm(a64) + 1e-12))
        if not np.isfinite(b64).all() or not np.isfinite(a64).all():
            rel = float("inf")
        by_kind[kind].append(rel)
        if rel > tol:
            first.append(
                f"  rel {rel:.3g}  {producer}  shape {a.shape}  "
                f"|ort| max {np.abs(a64).max():.3g}  |ov| max {np.abs(b64).max():.3g}"
            )
    print(
        f"compared {compared} tensors; {len(first)} past rel {tol:g}; "
        "first 25 in graph order:",
        flush=True,
    )
    for line in first[:25]:
        print(line, flush=True)
    print("worst rel per op type:", flush=True)
    for kind, rels in sorted(by_kind.items(), key=lambda item: -max(item[1])):
        print(
            f"  {kind:16s} n={len(rels):4d}  max {max(rels):.3g}  "
            f"median {np.median(rels):.3g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--repo", required=True, help="Local export directory")
    parser.add_argument(
        "--tokenizer", help="Tokenizer repo or directory (default: --repo)"
    )
    parser.add_argument("--onnx-file", default="onnx/model.onnx")
    parser.add_argument("--pooling", choices=POOLINGS, default="sentence")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--sample", type=int, default=6)
    parser.add_argument("--skip-devices", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-tokens", type=int, default=48)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    config = Config.load(args.config)
    with frozen_database(config.db_path) as (db, _snapshot_hash):
        stories, _labels, _times = db.get_feedback_for_training(user_id=args.user_id)
    rng = np.random.default_rng(0)
    picked = rng.choice(len(stories), size=args.sample, replace=False)
    texts = [args.prefix + story_embedding_text(stories[int(i)]) for i in picked]
    model_path = Path(args.repo) / args.onnx_file
    tokenizer: Any = AutoTokenizer.from_pretrained(args.tokenizer or args.repo)
    openvino: Any = importlib.import_module("openvino")
    core = openvino.Core()
    print(
        f"openvino {openvino.__version__}, onnxruntime {ort.__version__}, "
        f"GPU {core.get_property('GPU', 'FULL_DEVICE_NAME')}"
    )
    input_names = {
        port.get_any_name() for port in core.read_model(str(model_path)).inputs
    }

    def feeds_for(max_tokens: int) -> list[tuple[Feed, NDArray[Any]]]:
        # One text per request: no padding, so every value is a real activation.
        out = []
        for text in texts:
            encoded = tokenizer(
                [text], truncation=True, max_length=max_tokens, return_tensors="np"
            )
            out.append((_model_inputs(encoded, input_names), encoded["attention_mask"]))
        return out

    if not args.skip_devices:
        feeds = feeds_for(args.max_tokens)
        lengths = [int(mask.sum()) for _feed, mask in feeds]
        print(f"texts: {len(texts)}, tokens {min(lengths)}-{max(lengths)}", flush=True)
        run_cpu, _names = _runner(model_path, "cpu")
        reference = np.stack(
            [_pool(run_cpu(feed), mask, args.pooling)[0] for feed, mask in feeds]
        )
        for device, precision in (("CPU", "f32"), ("GPU", "f32"), ("GPU", "f16")):
            try:
                vectors = _vectors(
                    core, model_path, device, precision, feeds, args.pooling
                )
            except RuntimeError as error:
                print(f"{device} {precision}: failed: {str(error).splitlines()[0]}")
                continue
            cosines = (np.nan_to_num(vectors) * reference).sum(axis=1)
            print(
                f"OpenVINO {device} {precision} vs onnxruntime CPU: cos min "
                f"{cosines.min():.4f} mean {cosines.mean():.4f}",
                flush=True,
            )
    if args.trace:
        feed, mask = feeds_for(args.trace_tokens)[0]
        print(f"trace: 1 text, {int(mask.sum())} tokens", flush=True)
        _trace(core, model_path, feed, args.tol)


if __name__ == "__main__":
    main()
