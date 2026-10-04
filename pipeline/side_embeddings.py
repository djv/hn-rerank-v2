"""Optional second embedding model, ranked side by side with the stored one.

With ``ModelConfig.side_embedding_enabled`` the SVM and the linear models see
each story's stored vector and a second model's vector joined, each part
scaled 1/sqrt(2) (the evaluation replay's layout). Offline (2026-10-02,
FINDINGS.md "Fresh-vote, impression-pool and live-yield evals") embeddinggemma
-300m at 128 tokens added AUC +0.011 and top-12 upvotes +0.056 over 12 time
blocks. Attribution, Explore and dedup keep the stored vectors, so their
similarity thresholds keep their meaning.

Side vectors are encoded out of process by ``scripts/embed_side_vectors.py``
into the ``side_embeddings`` table; ranking only reads them. Stories without
one get a zero side part; below ``side_embedding_min_coverage`` the rerank
uses the stored vectors alone.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from database import Database, Story

from .config import Config
from .ranking import story_embedding_text

# embeddinggemma's task prefix for classification-style inputs.
SIDE_PREFIX = "task: classification | query: "


def side_text_hash(story: Story) -> str:
    """The text a side vector was made from; same hash as ``embeddings``."""
    return hashlib.sha256(story_embedding_text(story).encode("utf-8")).hexdigest()


def side_input(story: Story) -> str:
    return SIDE_PREFIX + story_embedding_text(story)


def side_by_side(
    stored: NDArray[np.float32], side: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Rows of both models joined, each part scaled 1/sqrt(2)."""
    if stored.shape[0] != side.shape[0]:
        raise ValueError("stored and side vectors must have the same rows")
    return (np.hstack([stored, side]) / np.sqrt(2.0)).astype(np.float32)


class SideEmbedder:
    """embeddinggemma-style ONNX model (``sentence_embedding`` output) in a
    Hugging Face snapshot layout: tokenizer files plus ``onnx/model.onnx``."""

    def __init__(
        self, model_dir: str, *, max_tokens: int, threads: int = 2, batch_size: int = 4
    ) -> None:
        from transformers import AutoTokenizer

        if max_tokens <= 0 or threads <= 0 or batch_size <= 0:
            raise ValueError("max_tokens, threads and batch_size must be positive")
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(Path(model_dir) / "onnx" / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """L2-normalized vectors in input order; longest texts batch together."""
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        rows: dict[int, NDArray[np.float32]] = {}
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            encoded = self.tokenizer(
                [texts[i] for i in batch],
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="np",
            )
            feed = {k: encoded[k] for k in self.input_names if k in encoded}
            names = [o.name for o in self.session.get_outputs()]
            outputs = dict(zip(names, self.session.run(None, feed), strict=True))
            vectors = np.asarray(outputs["sentence_embedding"], dtype=np.float32)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            rows.update(zip(batch, vectors, strict=True))
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([rows[i] for i in range(len(texts))])


class SideVectorCache:
    """Side vectors read from the DB once per story text; a rerank asks
    the DB only for stories it has not seen (or whose text changed)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # story_id -> (model_version, text_hash, vector)
        self._vectors: dict[int, tuple[str, str, NDArray[np.float32]]] = {}

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()

    def get(
        self, db: Database, config: Config, stories: list[Story]
    ) -> dict[int, NDArray[np.float32]]:
        version = config.side_embedding_model_version
        hashes = {s.id: side_text_hash(s) for s in stories}
        with self._lock:
            found = {
                sid: vec
                for sid, (model, text_hash, vec) in self._vectors.items()
                if model == version and hashes.get(sid) == text_hash
            }
        missing = [sid for sid in hashes if sid not in found]
        if missing:
            loaded = db.get_side_embeddings_batch(
                missing,
                version,
                {sid: hashes[sid] for sid in missing},
                expected_dim=config.side_embedding_dim,
            )
            with self._lock:
                for sid, vec in loaded.items():
                    self._vectors[sid] = (version, hashes[sid], vec)
            found.update(loaded)
        return found


_CACHE = SideVectorCache()


@dataclass(frozen=True)
class SideSpace:
    candidates: NDArray[np.float32]
    feedback: NDArray[np.float32]
    coverage: float


def model_space(
    candidates: list[Story],
    candidate_embeddings: NDArray[np.float32],
    feedback: list[Story],
    feedback_embeddings: NDArray[np.float32],
    db: Database,
    config: Config,
    cache: SideVectorCache = _CACHE,
) -> SideSpace | None:
    """Stored and side vectors joined for the model, or None (stored only)
    when side vectors cover too few of the stories."""
    stories = list({s.id: s for s in candidates + feedback}.values())
    if not stories:
        return None
    vectors = cache.get(db, config, stories)
    coverage = sum(s.id in vectors for s in stories) / len(stories)
    if coverage < config.model.side_embedding_min_coverage:
        logging.warning(
            "side_embeddings_low_coverage coverage=%.4f stories=%d; ranking on "
            "stored vectors only",
            coverage,
            len(stories),
        )
        return None
    zero = np.zeros(config.side_embedding_dim, dtype=np.float32)

    def side(items: list[Story]) -> NDArray[np.float32]:
        if not items:
            return np.empty((0, config.side_embedding_dim), dtype=np.float32)
        return np.stack([vectors.get(s.id, zero) for s in items])

    return SideSpace(
        side_by_side(candidate_embeddings, side(candidates)),
        side_by_side(feedback_embeddings, side(feedback)),
        coverage,
    )
