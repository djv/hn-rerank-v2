#!/usr/bin/env python3
"""CPU speed benchmark: production encoder (mxbai-embed-xsmall-v1) vs Qwen3-Embedding-0.6B.

Read-only. Pulls real production candidate texts from the configured DB
(SELECT only, no writes), embeds them with both encoders under identical
onnxruntime CPUExecutionProvider settings, and reports throughput/latency
per text-length bucket. This is a speed-only first cut — see
scripts/bakeoff_embedding_models.py + scripts/eval_ranker_variants.py for
the (separate, out-of-scope-here) quality/NDCG comparison.

Usage:
    uv run python scripts/bench_qwen_embed_speed.py --n 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database, Story  # noqa: E402
from pipeline import Config, load_production_candidate_stories  # noqa: E402
from pipeline.ranking import _process_rss_kb, story_embedding_text  # noqa: E402

Pooling = Literal["mean", "last_token"]

QWEN3_REPO_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX"
# Single self-contained file (no external .onnx_data blob) — the realistic
# quantized weight a CPU deployment would actually use, not the 2.4GB fp32 export.
QWEN3_ONNX_FILENAME = "onnx/model_quantized.onnx"


@dataclass(frozen=True)
class EncoderSpec:
    label: str
    tokenizer_dir: str
    model_path: Path
    pooling: Pooling
    max_tokens: int
    dim: int


def _session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, a_min=1e-12, a_max=None)


def _pool(
    token_vectors: NDArray[np.float32],
    attention_mask: NDArray[np.int64],
    pooling: Pooling,
) -> NDArray[np.float32]:
    if pooling == "mean":
        mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        summed = (token_vectors * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts
    # last_token: index of the final non-pad position per row. Robust to
    # either left- or right-padding (unlike sum(mask)-1, which assumes the
    # 1s are contiguous from index 0).
    reversed_mask = attention_mask[:, ::-1]
    last_from_end = np.argmax(reversed_mask, axis=1)
    last_indices = attention_mask.shape[1] - 1 - last_from_end
    rows = np.arange(token_vectors.shape[0])
    return token_vectors[rows, last_indices, :]


def _decoder_extra_inputs(
    session: ort.InferenceSession, batch_size: int, attention_mask: NDArray[np.int64]
) -> dict[str, NDArray[Any]]:
    """Build position_ids + empty KV-cache inputs for a decoder-with-past ONNX export.

    The Qwen3 embedding ONNX export (transformers.js style) is a causal LM
    graph with a KV cache, even though we only ever run a single forward pass
    (no generation). It requires position_ids and one zero-length
    past_key_values.{i}.{key,value} pair per layer.
    """
    extra: dict[str, NDArray[Any]] = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        if name == "position_ids":
            # cumsum(mask)-1 gives correct positions under left- or right-padding.
            positions = np.cumsum(attention_mask, axis=1) - 1
            extra[name] = np.clip(positions, a_min=0, a_max=None).astype(np.int64)
        elif name.startswith("past_key_values."):
            shape = [
                batch_size if isinstance(dim, str) and dim == "batch_size"
                else 0 if isinstance(dim, str)
                else dim
                for dim in input_meta.shape
            ]
            extra[name] = np.zeros(shape, dtype=np.float32)
    return extra


def _load_texts(config_path: str, n: int) -> list[str]:
    config = Config.load(config_path)
    db = Database(config.db_path, read_only=True)
    stories: list[Story] = load_production_candidate_stories(
        db, config, user_id=None, exclude_feedback=False
    )
    texts = [story_embedding_text(story) for story in stories]
    texts = [text for text in texts if text.strip()]
    if len(texts) > n:
        # Evenly sample across the pool rather than just taking the head
        # (which is score/gravity-sorted) so length distribution is realistic.
        step = len(texts) / n
        texts = [texts[int(i * step)] for i in range(n)]
    return texts


def _length_bucket(n_tokens: int) -> str:
    if n_tokens < 256:
        return "short (<256 tok)"
    if n_tokens < 1024:
        return "medium (256-1023 tok)"
    return "long (>=1024 tok)"


def _encode_timed(
    spec: EncoderSpec, texts: list[str], batch_size: int
) -> tuple[NDArray[np.float32], list[float], list[str], float, int | None]:
    """Returns (embeddings, per_text_seconds, length_bucket_per_text, model_load_s, rss_delta_kb)."""
    rss_before = _process_rss_kb()
    load_started = time.perf_counter()
    tokenizer: Any = AutoTokenizer.from_pretrained(spec.tokenizer_dir)
    session = ort.InferenceSession(
        str(spec.model_path),
        sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )
    input_names = {meta.name for meta in session.get_inputs()}
    all_output_names = [meta.name for meta in session.get_outputs()]
    hidden_state_name = (
        "last_hidden_state" if "last_hidden_state" in all_output_names else all_output_names[0]
    )
    # Only request the pooled-from output, not the KV-cache "present.*" outputs
    # — we never reuse them (single forward pass per batch), and requesting
    # them would force extra copy-out work that real single-pass embedding
    # usage wouldn't pay.
    requested_outputs = [hidden_state_name]
    is_decoder_with_past = any(name.startswith("past_key_values.") for name in input_names)
    model_load_s = time.perf_counter() - load_started

    def build_onnx_inputs(tok_inputs: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        onnx_inputs = {name: tok_inputs[name] for name in input_names if name in tok_inputs}
        if is_decoder_with_past:
            batch_size_here = tok_inputs["input_ids"].shape[0]
            onnx_inputs.update(
                _decoder_extra_inputs(session, batch_size_here, tok_inputs["attention_mask"])
            )
        return onnx_inputs

    # Warm-up: pay allocator/first-call cost outside the timed region.
    warm_inputs = tokenizer(
        ["warmup"], padding=True, truncation=True, max_length=spec.max_tokens,
        return_tensors="np",
    )
    session.run(requested_outputs, build_onnx_inputs(warm_inputs))

    embeddings: list[NDArray[np.float32]] = []
    per_text_seconds: list[float] = []
    buckets: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=spec.max_tokens,
            return_tensors="np",
        )
        n_tok = int(inputs["input_ids"].shape[1])
        buckets.extend(_length_bucket(n_tok) for _ in batch_texts)

        batch_started = time.perf_counter()
        outputs = session.run(requested_outputs, build_onnx_inputs(inputs))[0]
        elapsed = time.perf_counter() - batch_started
        per_text_seconds.extend([elapsed / len(batch_texts)] * len(batch_texts))

        token_vectors = np.asarray(outputs, dtype=np.float32)
        pooled = _pool(token_vectors, inputs["attention_mask"], spec.pooling)
        embeddings.append(_normalize(pooled).astype(np.float32))

    rss_after = _process_rss_kb()
    rss_delta = (
        rss_after - rss_before if rss_before is not None and rss_after is not None else None
    )
    return np.concatenate(embeddings, axis=0), per_text_seconds, buckets, model_load_s, rss_delta


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]


def _report(
    spec: EncoderSpec,
    per_text_seconds: list[float],
    buckets: list[str],
    model_load_s: float,
    rss_delta_kb: int | None,
) -> None:
    total_s = sum(per_text_seconds)
    n = len(per_text_seconds)
    print(f"\n=== {spec.label} (dim={spec.dim}) ===")
    print(f"  model load:      {model_load_s:6.2f}s")
    if rss_delta_kb is not None:
        print(f"  RSS delta:       {rss_delta_kb / 1024:8.1f} MB")
    print(f"  total embed:     {total_s:6.2f}s for {n} texts")
    print(f"  throughput:      {n / total_s if total_s else float('nan'):8.2f} texts/sec")
    print(f"  latency p50/p95: {_percentile(per_text_seconds, 50) * 1000:6.1f}ms / "
          f"{_percentile(per_text_seconds, 95) * 1000:6.1f}ms per text")

    by_bucket: dict[str, list[float]] = {}
    for bucket, seconds in zip(buckets, per_text_seconds):
        by_bucket.setdefault(bucket, []).append(seconds)
    for bucket in ("short (<256 tok)", "medium (256-1023 tok)", "long (>=1024 tok)"):
        values = by_bucket.get(bucket)
        if not values:
            continue
        avg_ms = statistics.mean(values) * 1000
        print(f"    {bucket:24s} n={len(values):4d}  avg={avg_ms:7.1f}ms/text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--n", type=int, default=200, help="Number of story texts to benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--qwen-max-tokens", type=int, default=4096,
        help="Cap Qwen3 context to match production's 4096-token budget",
    )
    args = parser.parse_args()

    print(f"Loading up to {args.n} real candidate texts from {args.config} (read-only)...")
    texts = _load_texts(args.config, args.n)
    print(f"Loaded {len(texts)} non-empty texts.")

    config = Config.load(args.config)

    # --- current production encoder (mxbai-embed-xsmall-v1) ---
    mxbai_spec = EncoderSpec(
        label="mxbai-embed-xsmall-v1 (production)",
        tokenizer_dir=config.onnx_model_dir,
        model_path=Path(config.onnx_model_dir) / "model.onnx",
        pooling="mean",
        max_tokens=config.embedding_max_tokens,
        dim=384,
    )
    embeddings, per_text_s, buckets, load_s, rss_delta = _encode_timed(
        mxbai_spec, texts, args.batch_size
    )
    assert embeddings.shape == (len(texts), 384)
    _report(mxbai_spec, per_text_s, buckets, load_s, rss_delta)

    # --- candidate: Qwen3-Embedding-0.6B (quantized ONNX export) ---
    print(f"\nDownloading {QWEN3_REPO_ID}:{QWEN3_ONNX_FILENAME} (~600MB, cached after first run)...")
    qwen_model_path = Path(hf_hub_download(repo_id=QWEN3_REPO_ID, filename=QWEN3_ONNX_FILENAME))
    qwen_spec = EncoderSpec(
        label="Qwen3-Embedding-0.6B int8 (candidate)",
        tokenizer_dir=QWEN3_REPO_ID,
        model_path=qwen_model_path,
        pooling="last_token",
        max_tokens=args.qwen_max_tokens,
        dim=1024,
    )
    embeddings, per_text_s, buckets, load_s, rss_delta = _encode_timed(
        qwen_spec, texts, args.batch_size
    )
    assert embeddings.shape == (len(texts), 1024)
    _report(qwen_spec, per_text_s, buckets, load_s, rss_delta)

    print(
        "\nNote: feature vector is [embedding | 10 metadata cols] "
        "(pipeline/ranking.py:_svm_personalization_features). "
        "384->1024 widens the SVM input ~2.7x and the embeddings BLOB cache ~2.7x per story."
    )


if __name__ == "__main__":
    main()
