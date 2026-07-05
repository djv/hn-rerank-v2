from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config, story_embedding_text  # noqa: E402


PRODUCTION_BATCH_SIZE = 32
EMBEDDING_DIM = 384
DEFAULT_VARIANTS = (
    "current",
    "spin_off",
    "spin_off_graph_all",
    "spin_off_auto_threads",
)
DEFAULT_BATCH_SIZES = (1, 8, 32, 64)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    disable_spinning: bool = False
    graph_all: bool = False
    intra_op_num_threads: int | None = 2
    inter_op_num_threads: int | None = 1


VARIANT_SPECS: dict[str, VariantSpec] = {
    "current": VariantSpec("current"),
    "spin_off": VariantSpec("spin_off", disable_spinning=True),
    "spin_off_graph_all": VariantSpec(
        "spin_off_graph_all",
        disable_spinning=True,
        graph_all=True,
    ),
    "spin_off_auto_threads": VariantSpec(
        "spin_off_auto_threads",
        disable_spinning=True,
        graph_all=True,
        intra_op_num_threads=0,
        inter_op_num_threads=1,
    ),
}


class BenchmarkEmbedder:
    def __init__(self, model_dir: str, variant: VariantSpec) -> None:
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_dir)
        session_options = _session_options_for_variant(variant)
        self.session: Any = ort.InferenceSession(
            str(Path(model_dir) / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [input_meta.name for input_meta in self.session.get_inputs()]
        self.max_tokens = 512

    def encode(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        embeddings: list[NDArray[np.float32]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="np",
            )
            onnx_inputs = {name: inputs[name] for name in self.input_names if name in inputs}
            outputs = self.session.run(None, onnx_inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"]

            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(
                np.float32
            )
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(
                np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
            )
            mean_embeddings = sum_embeddings / sum_mask

            norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            embeddings.append((mean_embeddings / norms).astype(np.float32))

        return np.concatenate(embeddings, axis=0)


def _session_options_for_variant(variant: VariantSpec) -> Any:
    session_options = ort.SessionOptions()
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    if variant.intra_op_num_threads is not None:
        session_options.intra_op_num_threads = variant.intra_op_num_threads
    if variant.inter_op_num_threads is not None:
        session_options.inter_op_num_threads = variant.inter_op_num_threads
    if variant.disable_spinning:
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if variant.graph_all:
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return session_options


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_batch_sizes(value: str) -> list[int]:
    batch_sizes = [int(part) for part in _parse_csv(value)]
    if any(size <= 0 for size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return batch_sizes


def _selected_variants(value: str) -> list[str]:
    variants = _parse_csv(value)
    unknown = sorted(set(variants) - set(VARIANT_SPECS))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown variant(s): " + ", ".join(unknown)
        )
    return variants


def load_sample_texts(db: Database, sample_size: int, seed: int) -> list[str]:
    rows = db.execute(
        """
        SELECT id, title, url, score, time, text_content, source, comment_count,
               discussion_url, comment_count_at_fetch, self_text, top_comments,
               article_body
        FROM stories
        """
    )
    texts = [
        story_embedding_text(Database._row_to_story(row)).strip()
        for row in rows
    ]
    texts = [text for text in texts if text]
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:sample_size]


def token_length_percentiles(tokenizer: Any, texts: list[str]) -> dict[str, int]:
    if not texts:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    encoded = tokenizer(texts, padding=False, truncation=True, max_length=512)
    lengths = sorted(len(input_ids) for input_ids in encoded["input_ids"])
    return {
        "p50": _percentile_int(lengths, 0.50),
        "p90": _percentile_int(lengths, 0.90),
        "p95": _percentile_int(lengths, 0.95),
        "p99": _percentile_int(lengths, 0.99),
        "max": lengths[-1],
    }


def _percentile_int(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, int(round((len(values) - 1) * percentile)))
    return int(values[idx])


def drift_against_baseline(
    baseline: NDArray[np.float32], candidate: NDArray[np.float32]
) -> dict[str, float | bool | list[int]]:
    finite = bool(np.isfinite(candidate).all())
    if baseline.shape != candidate.shape or baseline.size == 0:
        return {
            "shape": list(candidate.shape),
            "finite": finite,
            "norm_min": 0.0,
            "norm_mean": 0.0,
            "norm_max": 0.0,
            "min_cosine_to_baseline": 0.0,
            "max_abs_vector_drift": float("inf"),
            "acceptable": False,
        }

    baseline_norm = np.linalg.norm(baseline, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    denominator = np.clip(baseline_norm * candidate_norm, a_min=1e-12, a_max=None)
    cosine = np.sum(baseline * candidate, axis=1) / denominator
    min_cosine = float(np.min(cosine))
    max_abs_drift = float(np.max(np.abs(baseline - candidate)))
    norm_min = float(np.min(candidate_norm))
    norm_mean = float(np.mean(candidate_norm))
    norm_max = float(np.max(candidate_norm))
    acceptable = (
        candidate.shape == (baseline.shape[0], EMBEDDING_DIM)
        and finite
        and 0.999 <= norm_min <= 1.001
        and 0.999 <= norm_max <= 1.001
        and min_cosine >= 0.99999
    )
    return {
        "shape": list(candidate.shape),
        "finite": finite,
        "norm_min": norm_min,
        "norm_mean": norm_mean,
        "norm_max": norm_max,
        "min_cosine_to_baseline": min_cosine,
        "max_abs_vector_drift": max_abs_drift,
        "acceptable": bool(acceptable),
    }


def benchmark_variant(
    *,
    model_dir: str,
    variant_name: str,
    batch_size: int,
    texts: list[str],
    runs: int,
    baseline: NDArray[np.float32],
) -> dict[str, Any]:
    embedder = BenchmarkEmbedder(model_dir, VARIANT_SPECS[variant_name])
    warmup_texts = texts[: min(len(texts), max(1, min(batch_size, 8)))]
    if warmup_texts:
        embedder.encode(warmup_texts, batch_size=batch_size)

    run_ms: list[float] = []
    embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    for _ in range(max(runs, 0)):
        start = time.perf_counter()
        embeddings = embedder.encode(texts, batch_size=batch_size)
        run_ms.append((time.perf_counter() - start) * 1000.0)
    if not run_ms:
        embeddings = embedder.encode(texts, batch_size=batch_size)

    median_ms = statistics.median(run_ms) if run_ms else 0.0
    docs_per_sec = (len(texts) / (median_ms / 1000.0)) if median_ms > 0 else 0.0
    drift = drift_against_baseline(baseline, embeddings)
    return {
        "variant": variant_name,
        "batch_size": batch_size,
        "sample_size": len(texts),
        "run_ms": [round(value, 3) for value in run_ms],
        "median_run_ms": round(median_ms, 3),
        "docs_per_sec": round(docs_per_sec, 3),
        "ms_per_doc": round(median_ms / len(texts), 6) if texts else 0.0,
        "output_shape": drift["shape"],
        "finite": drift["finite"],
        "norm_min": drift["norm_min"],
        "norm_mean": drift["norm_mean"],
        "norm_max": drift["norm_max"],
        "min_cosine_to_baseline": drift["min_cosine_to_baseline"],
        "max_abs_vector_drift": drift["max_abs_vector_drift"],
        "acceptable": drift["acceptable"],
    }


def _baseline_embeddings(model_dir: str, texts: list[str]) -> NDArray[np.float32]:
    embedder = BenchmarkEmbedder(model_dir, VARIANT_SPECS["current"])
    warmup_texts = texts[: min(len(texts), 8)]
    if warmup_texts:
        embedder.encode(warmup_texts, batch_size=PRODUCTION_BATCH_SIZE)
    return embedder.encode(texts, batch_size=PRODUCTION_BATCH_SIZE)


def _print_table(results: list[dict[str, Any]]) -> None:
    print("variant                batch  sample  median_ms  docs/sec  ms/doc  min_cosine  max_abs")
    for row in results:
        print(
            f"{row['variant']:<22} {row['batch_size']:>5} {row['sample_size']:>7} "
            f"{row['median_run_ms']:>9.1f} {row['docs_per_sec']:>9.1f} "
            f"{row['ms_per_doc']:>7.3f} {row['min_cosine_to_baseline']:>10.7f} "
            f"{row['max_abs_vector_drift']:>7.1e}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark local ONNX embedding session and batch-size variants."
    )
    parser.add_argument("--db", default=None, help="SQLite DB path; defaults to config.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=list(DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--variants",
        type=_selected_variants,
        default=list(DEFAULT_VARIANTS),
        help="Comma-separated variants.",
    )
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--write-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if args.runs < 0:
        raise SystemExit("--runs must be non-negative")

    config = Config.load(args.config)
    db_path = args.db or config.db_path
    db = Database(db_path, read_only=True)
    try:
        texts = load_sample_texts(db, args.sample_size, args.seed)
    finally:
        db.close()
    if not texts:
        raise SystemExit("No non-empty story embedding texts found.")

    tokenizer = AutoTokenizer.from_pretrained(config.onnx_model_dir)
    token_stats = token_length_percentiles(tokenizer, texts)
    baseline = _baseline_embeddings(config.onnx_model_dir, texts)

    results: list[dict[str, Any]] = []
    for variant_name in args.variants:
        for batch_size in args.batch_sizes:
            results.append(
                benchmark_variant(
                    model_dir=config.onnx_model_dir,
                    variant_name=variant_name,
                    batch_size=batch_size,
                    texts=texts,
                    runs=args.runs,
                    baseline=baseline,
                )
            )

    payload: dict[str, Any] = {
        "baseline": {"variant": "current", "batch_size": PRODUCTION_BATCH_SIZE},
        "config": {
            "db": db_path,
            "model_dir": config.onnx_model_dir,
            "sample_size_requested": args.sample_size,
            "seed": args.seed,
            "runs": args.runs,
        },
        "token_length_percentiles": token_stats,
        "results": results,
    }
    if not args.json_only:
        print(
            f"sample_size={len(texts)} model_dir={config.onnx_model_dir} "
            f"baseline=current/batch{PRODUCTION_BATCH_SIZE}"
        )
        print(f"token_lengths={token_stats}")
        _print_table(results)
        print("\njson")

    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.write_output is not None:
        args.write_output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
