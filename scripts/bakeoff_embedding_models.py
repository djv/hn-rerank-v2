#!/usr/bin/env python3
"""Create validated, model-specific embedding snapshots for offline ranking eval.

The script never writes to the configured dashboard database.  Each snapshot
contains the exact production candidate IDs and text hashes, so
``eval_ranker_variants.py --embeddings-file`` rejects stale or mismatched data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
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
from pipeline import (  # noqa: E402
    Config,
    DEFAULT_ONNX_MODEL_DIR,
    load_production_candidate_stories,
    story_embedding_text,
)
from scripts.eval_ranker_variants import (  # noqa: E402
    _candidate_indices_with_feedback,
    _embedding_text_hashes,
    _snapshot_candidate_stories,
    _snapshot_feedback,
)


Pooling = Literal["mean", "cls"]


@dataclass(frozen=True)
class BakeoffModel:
    name: str
    repo_id: str | None
    onnx_filename: str
    pooling: Pooling
    local_dir: str | None = None


MODELS: dict[str, BakeoffModel] = {
    "minilm": BakeoffModel(
        name="minilm",
        repo_id=None,
        onnx_filename="model.onnx",
        pooling="mean",
        local_dir=DEFAULT_ONNX_MODEL_DIR,
    ),
    "mxbai_xsmall": BakeoffModel(
        name="mxbai_xsmall",
        repo_id="mixedbread-ai/mxbai-embed-xsmall-v1",
        onnx_filename="onnx/model.onnx",
        pooling="mean",
    ),
    "arctic_xs": BakeoffModel(
        name="arctic_xs",
        repo_id="Snowflake/snowflake-arctic-embed-xs",
        onnx_filename="onnx/model.onnx",
        pooling="cls",
    ),
    "bge_small": BakeoffModel(
        name="bge_small",
        repo_id="BAAI/bge-small-en-v1.5",
        onnx_filename="onnx/model.onnx",
        pooling="cls",
    ),
}


def _parse_models(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(MODELS))
    if unknown:
        raise argparse.ArgumentTypeError("Unknown model(s): " + ", ".join(unknown))
    return names


def _session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.intra_op_num_threads = 0
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def _model_paths(spec: BakeoffModel) -> tuple[str, Path]:
    if spec.local_dir is not None:
        model_dir = Path(spec.local_dir)
        return str(model_dir), model_dir / spec.onnx_filename
    if spec.repo_id is None:
        raise ValueError(f"Model {spec.name} has no local directory or repository")
    model_path = Path(
        hf_hub_download(repo_id=spec.repo_id, filename=spec.onnx_filename)
    )
    return spec.repo_id, model_path


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, a_min=1e-12, a_max=None)


def _encode(
    texts: list[str],
    *,
    tokenizer_dir: str,
    model_path: Path,
    pooling: Pooling,
    max_tokens: int,
    batch_size: int,
) -> tuple[NDArray[np.float32], float]:
    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_dir)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )
    input_names = {meta.name for meta in session.get_inputs()}
    chunks: list[NDArray[np.float32]] = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="np",
        )
        outputs = session.run(
            None, {name: inputs[name] for name in input_names if name in inputs}
        )[0]
        token_vectors = np.asarray(outputs, dtype=np.float32)
        if pooling == "cls":
            pooled = token_vectors[:, 0, :]
        else:
            mask = np.expand_dims(inputs["attention_mask"], axis=-1).astype(np.float32)
            pooled = (token_vectors * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1), a_min=1e-9, a_max=None
            )
        chunks.append(_normalize(pooled).astype(np.float32))
        completed = min(start + batch_size, len(texts))
        if completed == len(texts) or completed % max(batch_size, 100) == 0:
            print(f"{completed}/{len(texts)} encoded", flush=True)
    return np.concatenate(chunks, axis=0), time.perf_counter() - started


def _candidate_stories(
    db: Database, config: Config, user_id: int, max_candidates: int | None
) -> tuple[list[Story], list[Story], list[int], list[float]]:
    stories = load_production_candidate_stories(
        db, config, user_id=user_id, exclude_feedback=False
    )
    feedback_stories, _labels, _vote_times = db.get_feedback_for_training(user_id)
    indices = _candidate_indices_with_feedback(
        stories,
        max_candidates=max_candidates,
        required_story_ids={story.id for story in feedback_stories},
    )
    return (
        [stories[index] for index in indices],
        feedback_stories,
        _labels,
        _vote_times,
    )


def _snapshot_path(output_dir: Path, name: str, max_tokens: int) -> Path:
    return output_dir / f"{name}-tokens{max_tokens}.npz"


def _snapshot_matches(path: Path, stories: list[Story]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "story_ids",
                "text_hashes",
                "embeddings",
                "stories_json",
                "feedback_stories_json",
                "feedback_labels",
                "feedback_vote_times",
            }
            if not required <= set(data.files):
                return False
            return (
                np.array_equal(
                    np.asarray(data["story_ids"], dtype=np.int64),
                    np.array([story.id for story in stories], dtype=np.int64),
                )
                and np.array_equal(
                    np.asarray(data["text_hashes"], dtype="<U64"),
                    _embedding_text_hashes(stories),
                )
                and np.asarray(data["embeddings"]).shape == (len(stories), 384)
            )
    except (KeyError, OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 384-dimensional ONNX embedding snapshots for model bakeoff."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int)
    parser.add_argument(
        "--reference-snapshot",
        type=Path,
        help=(
            "Use the frozen candidates and feedback in this .npz snapshot instead "
            "of reading the live database. Required for a fair sequential bakeoff."
        ),
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=["minilm", "mxbai_xsmall", "arctic_xs", "bge_small"],
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        choices=(256, 512, 1024, 2048, 4096),
        default=256,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Deterministic candidate cap matching eval_ranker_variants.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/hn-rewrite-embedding-bakeoff"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.reference_snapshot is not None and args.max_candidates is not None:
        raise SystemExit("--reference-snapshot cannot be combined with --max-candidates")

    config = Config.load(args.config)
    user_id: int | None = args.user_id
    if args.reference_snapshot is not None:
        stories = _snapshot_candidate_stories(args.reference_snapshot)
        feedback_stories, feedback_labels_array, feedback_vote_times_array = (
            _snapshot_feedback(args.reference_snapshot)
        )
        feedback_labels = [int(label) for label in feedback_labels_array]
        feedback_vote_times = [float(vote_time) for vote_time in feedback_vote_times_array]
    else:
        db = Database(config.db_path, read_only=True)
        try:
            if user_id is None:
                user = db.get_user_by_token("default")
                if user is None:
                    raise SystemExit("Missing default user token; pass --user-id")
                user_id = user.id
            stories, feedback_stories, feedback_labels, feedback_vote_times = (
                _candidate_stories(db, config, user_id, args.max_candidates)
            )
        finally:
            db.close()
    if not stories:
        raise SystemExit("No production candidate stories found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts = [story_embedding_text(story) for story in stories]
    story_ids = np.array([story.id for story in stories], dtype=np.int64)
    text_hashes = _embedding_text_hashes(stories)
    feedback_story_ids = np.array(
        [story.id for story in feedback_stories], dtype=np.int64
    )
    model_results: list[dict[str, object]] = []
    report: dict[str, object] = {
        "user_id": user_id,
        "candidate_count": len(stories),
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "max_candidates": args.max_candidates,
        "reference_snapshot": (
            str(args.reference_snapshot) if args.reference_snapshot is not None else None
        ),
        "models": model_results,
    }

    for name in args.models:
        spec = MODELS[name]
        path = _snapshot_path(args.output_dir, name, args.max_tokens)
        if not args.force and _snapshot_matches(path, stories):
            print(f"{name}: reusing {path}")
            model_results.append({"name": name, "snapshot": str(path), "reused": True})
            continue
        tokenizer_dir, model_path = _model_paths(spec)
        print(f"{name}: loading {model_path}")
        embeddings, elapsed_s = _encode(
            texts,
            tokenizer_dir=tokenizer_dir,
            model_path=model_path,
            pooling=spec.pooling,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        if embeddings.shape != (len(stories), 384):
            raise RuntimeError(
                f"{name} emitted {embeddings.shape}; this bakeoff only supports 384 dimensions"
            )
        np.savez_compressed(
            path,
            story_ids=story_ids,
            text_hashes=text_hashes,
            embeddings=embeddings,
            stories_json=np.array(
                json.dumps([asdict(story) for story in stories], separators=(",", ":"))
            ),
            feedback_story_ids=feedback_story_ids,
            feedback_stories_json=np.array(
                json.dumps(
                    [asdict(story) for story in feedback_stories], separators=(",", ":")
                )
            ),
            feedback_labels=np.asarray(feedback_labels, dtype=np.int8),
            feedback_vote_times=np.asarray(feedback_vote_times, dtype=np.float64),
        )
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        result = {
            "name": name,
            "spec": asdict(spec),
            "snapshot": str(path),
            "model_sha256": model_sha256,
            "elapsed_seconds": round(elapsed_s, 3),
            "stories_per_second": round(len(stories) / elapsed_s, 3),
            "reused": False,
        }
        model_results.append(result)
        print(json.dumps(result, sort_keys=True))

    report_path = args.output_dir / f"bakeoff-tokens{args.max_tokens}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
