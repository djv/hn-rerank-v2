from __future__ import annotations

import hashlib
import html
import logging
import re
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from bs4 import BeautifulSoup
from cachetools import LRUCache
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from database import Database, Story
from .config import (
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_SOURCE,
    Config,
    is_hn_source,
)


# SVM model cache: keyed on (user_id, feedback_signature, schema_version)
# to skip SVC.fit(). Bump _MODEL_SCHEMA_VERSION whenever the feature schema
# (number / semantics of meta columns appended to the embedding) changes;
# the cache key then changes for every user, forcing a clean re-fit.
_MODEL_CACHE_STORAGE_MAXSIZE = 10_000
_MODEL_CACHE: LRUCache[tuple[int, str, int], tuple[SVC, StandardScaler]] = LRUCache(
    maxsize=_MODEL_CACHE_STORAGE_MAXSIZE
)
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_SCHEMA_VERSION = 2  # +1 whenever meta-column schema changes (see ARCHITECTURE)


@dataclass
class RankTrace:
    """Low-overhead timing and counters for one personalized rank run."""

    timings_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_timing(name, (time.perf_counter() - start) * 1000)

    def add_timing(self, name: str, elapsed_ms: float) -> None:
        self.timings_ms[name] = self.timings_ms.get(name, 0.0) + elapsed_ms

    def set_count(self, name: str, value: int) -> None:
        self.counts[name] = value

    def set_label(self, name: str, value: str) -> None:
        self.labels[name] = value

    def to_log_fields(self) -> dict[str, int | float | str]:
        fields: dict[str, int | float | str] = {}
        fields.update(self.counts)
        fields.update(self.labels)
        for name, value in self.timings_ms.items():
            fields[f"{name}_ms"] = round(value, 1)
        return fields

    def format_log_fields(self) -> str:
        fields = self.to_log_fields()
        return " ".join(f"{key}={fields[key]}" for key in sorted(fields))


class _NullTrace:
    """No-op RankTrace sentinel. Used as the default trace argument so callers
    can write ``with trace.stage(\"x\"):`` unconditionally and skip
    ``if trace is not None else nullcontext()`` boilerplate at every site."""

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        yield

    def set_count(self, name: str, value: int) -> None:
        pass

    def set_label(self, name: str, value: str) -> None:
        pass


NULL_TRACE: _NullTrace = _NullTrace()


@dataclass
class RankScoreContext:
    """Reusable arrays from one rank pass for downstream discovery badges."""

    feedback_labels: list[int] = field(default_factory=list)
    feedback_embeddings: NDArray[np.float32] | None = None
    cand_closest_up: NDArray[np.float32] | None = None
    cand_closest_down: NDArray[np.float32] | None = None
    cand_closest_neutral: NDArray[np.float32] | None = None


def _feedback_signature(db: Database, user_id: int) -> str:
    feedback = db.get_all_feedback(user_id=user_id)
    hasher = hashlib.sha256()
    for f in sorted(feedback, key=lambda x: x.story_id):
        hasher.update(f"{f.story_id}:{f.action}:{f.updated_at}".encode())
    return hasher.hexdigest()


def _get_cached_model(
    user_id: int | None, signature: str
) -> tuple[SVC, StandardScaler] | None:
    if user_id is None:
        return None
    with _MODEL_CACHE_LOCK:
        key = (user_id, signature, _MODEL_SCHEMA_VERSION)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
    return None


def _set_cached_model(
    user_id: int | None,
    signature: str,
    svm: SVC,
    scaler: StandardScaler,
    maxsize: int = 20,
) -> None:
    if user_id is None:
        return
    with _MODEL_CACHE_LOCK:
        key = (user_id, signature, _MODEL_SCHEMA_VERSION)
        _MODEL_CACHE[key] = (svm, scaler)
        while len(_MODEL_CACHE) > maxsize:
            _MODEL_CACHE.popitem()


# Comment selection and depth tuning
COMMENT_DEPTH_PENALTY = 25  # Points a reply must overcome per nesting level
TOP_COMMENT_LIMIT = 40
TOP_COMMENT_CORE_THREADS = 4
TOP_COMMENT_REPLIES_PER_CORE_THREAD = 5
TOP_COMMENT_MAX_PER_THREAD = 6
GOOD_TOPLEVEL_MIN_LEN = 200
GOOD_TOPLEVEL_MIN_REPLIES = 3
TOP_COMMENT_TOP_LEVEL_BUDGET = TOP_COMMENT_LIMIT // 3
HOT_MIN_SCORE = 20
DASHBOARD_QUEUE_SIZE = 12
PRIMARY_PER_COMBO = 12
DISCOVERY_PER_BADGE = 2
SOURCE_CATEGORIES: tuple[str, ...] = ("hn_live", "archive", "reddit", "rss")


def source_category_onehot(source: str) -> NDArray[np.float32]:
    """Return a length-4 binary vector classifying ``source`` into one of
    ``SOURCE_CATEGORIES`` (live HN, archive seed, reddit, generic RSS).

    Sources that do not match any category return an all-zero vector rather
    than a 5th negative class — the model is expected to learn the implicit
    "other" prior from the absence of all four bits.
    """
    if source == "hn":
        idx = 0
    elif source in {BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE}:
        idx = 1
    elif source.startswith("rss_reddit_"):
        idx = 2
    elif source.startswith("rss_") or source.startswith("reddit_"):
        idx = 3
    else:
        return np.zeros(len(SOURCE_CATEGORIES), dtype=np.float32)
    out = np.zeros(len(SOURCE_CATEGORIES), dtype=np.float32)
    out[idx] = 1.0
    return out


def source_category_stack(sources: list[str]) -> NDArray[np.float32]:
    """Vectorized helper: stack ``source_category_onehot`` for a list of
    sources. Returns an (n, 4) float32 array."""
    if not sources:
        return np.zeros((0, len(SOURCE_CATEGORIES)), dtype=np.float32)
    return np.stack([source_category_onehot(s) for s in sources], axis=0)


@dataclass(frozen=True)
class RankedStory:
    story: Story
    score: float
    best_match_title: str
    prob_down: float | None = None
    prob_neutral: float | None = None
    prob_up: float | None = None
    is_uncertain: bool = False
    is_novel: bool = False
    is_discussion_rich: bool = False
    is_high_engagement: bool = False
    is_hot: bool = False
    is_similar: bool = False
    is_non_hn: bool = False
    is_recent: bool = False
    combo_keys: str = ""


def clean_text(raw_text: str, min_len: int = 0) -> str:
    if not raw_text:
        return ""
    if "<" not in raw_text and ">" not in raw_text and "&" not in raw_text:
        txt = html.unescape(raw_text)
    else:
        try:
            txt = BeautifulSoup(raw_text, "html.parser").get_text(" ", strip=True)
        except Exception:
            txt = re.sub(r"<[^>]*>", " ", raw_text)
        txt = html.unescape(txt)

    txt = re.sub(r"[\u2800-\u28FF\u2500-\u27BF]+", "", txt)
    txt = re.sub(r"[#*^\\/|\\-_+]{3,}", "", txt)
    txt = re.sub(r"\s+([.,;:!?])", r"\1", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    if len(txt) <= min_len:
        return ""
    alnum = sum(c.isalnum() for c in txt)
    if len(txt) > 0 and (alnum / len(txt)) < 0.5:
        return ""
    return txt


def _extract_comments_recursive(
    children: list,
    depth: int = 0,
    parent_points: int = 0,
    top_thread_index: int | None = None,
    order_path: tuple[int, ...] = (),
) -> list[dict]:
    MIN_COMMENT_LENGTH = 60
    results = []
    for sibling_index, child in enumerate(children):
        if not isinstance(child, dict) or child.get("type") != "comment":
            continue
        points = child.get("points") or 0
        if depth > 0 and points == 0:
            points = parent_points
        score = -points + depth * COMMENT_DEPTH_PENALTY
        child_top_thread_index = sibling_index if depth == 0 else top_thread_index
        child_order_path = (*order_path, sibling_index)
        child_comments = child.get("children") or []
        child_results = _extract_comments_recursive(
            child_comments,
            depth + 1,
            parent_points=points,
            top_thread_index=child_top_thread_index,
            order_path=child_order_path,
        )
        descendant_count = len(child_results)
        text = child.get("text", "")
        if text:
            clean = clean_text(text, min_len=MIN_COMMENT_LENGTH)
            if clean:
                results.append(
                    {
                        "id": child.get("id"),
                        "text": clean,
                        "score": score,
                        "depth": depth,
                        "top_thread_index": child_top_thread_index,
                        "sibling_index": sibling_index,
                        "order_path": child_order_path,
                        "reply_count": len(child_comments),
                        "descendant_count": descendant_count,
                        "text_len": len(clean),
                    }
                )
        results.extend(child_results)
    return results


def _comment_rank_key(comment: dict) -> tuple:
    return (
        -comment["descendant_count"],
        -min(comment["text_len"], 3000),
        comment["order_path"],
    )


def _select_top_comments(
    comments: list[dict],
    limit: int = TOP_COMMENT_LIMIT,
) -> list[dict]:
    """Select comment text for embeddings/TLDRs.

    Prefer large discussion cores (top engaged threads) and breadth of
    substantive top-level comments.  Replies compete on equal footing with
    top-level (no depth penalty); the previous ``score``-based rank key
    effectively preferred top-level regardless of substance, since Algolia
    returns ``points: null`` for HN comments.
    """
    if not comments:
        return []

    selected = []
    selected_indexes = set()
    per_thread: dict[int, int] = {}

    def add(comment: dict) -> None:
        if len(selected) >= limit:
            return
        index = id(comment)
        thread_index = comment["top_thread_index"]
        if index in selected_indexes:
            return
        if per_thread.get(thread_index, 0) >= TOP_COMMENT_MAX_PER_THREAD:
            return
        selected.append(comment)
        selected_indexes.add(index)
        per_thread[thread_index] = per_thread.get(thread_index, 0) + 1

    top_level = [c for c in comments if c["depth"] == 0]
    good_top_level = [
        c
        for c in top_level
        if c["text_len"] >= GOOD_TOPLEVEL_MIN_LEN
        or c["descendant_count"] >= GOOD_TOPLEVEL_MIN_REPLIES
    ]

    n_cores = min(TOP_COMMENT_CORE_THREADS, len(good_top_level))
    core_roots = sorted(
        good_top_level,
        key=lambda c: (-c["descendant_count"], c["top_thread_index"]),
    )[:n_cores]
    core_threads = {c["top_thread_index"] for c in core_roots}

    for root in sorted(core_roots, key=_comment_rank_key):
        add(root)

    for thread_index in sorted(core_threads):
        replies = [
            c
            for c in comments
            if c["top_thread_index"] == thread_index and c["depth"] > 0
        ]
        for reply in sorted(replies, key=_comment_rank_key)[
            :TOP_COMMENT_REPLIES_PER_CORE_THREAD
        ]:
            add(reply)

    top_level_added = sum(1 for c in selected if c["depth"] == 0)
    for comment in sorted(good_top_level, key=_comment_rank_key):
        if top_level_added >= TOP_COMMENT_TOP_LEVEL_BUDGET:
            break
        add(comment)
        top_level_added += 1

    for comment in sorted(comments, key=_comment_rank_key):
        add(comment)
        if len(selected) >= limit:
            break

    return selected


def compose_story_text(
    title: str,
    self_text: str = "",
    comments: str = "",
    article_body: str = "",
) -> str:
    clean_title = clean_text(title)
    clean_self = clean_text(self_text)[:6000]
    clean_comments = clean_text(comments)[:6000]
    clean_article = clean_text(article_body)[:4000]

    parts = []
    if clean_title:
        parts.append(f"{clean_title}.")
    if clean_self:
        parts.append(clean_self)
    if clean_article:
        parts.append(clean_article)
    if clean_comments:
        parts.append(clean_comments)

    return " ".join(parts).strip()


def story_embedding_text(story: Story) -> str:
    """Return the exact text used for the current production embedding version."""
    if story.text_content:
        return story.text_content
    return compose_story_text(
        story.title,
        story.self_text,
        story.top_comments,
        story.article_body,
    )


# Algolia Fetching
class Embedder:
    def __init__(self, model_dir: str = "onnx_model"):
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_dir)
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(Path(model_dir) / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.max_tokens = 512

    def encode(self, texts: list[str], batch_size: int = 32) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="np",
            )

            onnx_inputs = {}
            for input_meta in self.session.get_inputs():
                name = input_meta.name
                if name in inputs:
                    onnx_inputs[name] = inputs[name]

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
            normalized_embeddings = mean_embeddings / norms

            embeddings.append(normalized_embeddings)

        return np.concatenate(embeddings, axis=0)


def get_or_compute_embeddings(
    stories: list[Story],
    embedder: Embedder,
    db: Database,
) -> NDArray[np.float32]:
    if not stories:
        return np.empty((0, 384), dtype=np.float32)

    import hashlib

    embedding_texts = {s.id: story_embedding_text(s) for s in stories}
    story_hashes = {
        s.id: hashlib.sha256(embedding_texts[s.id].encode("utf-8")).hexdigest()
        for s in stories
    }

    ids = [s.id for s in stories]
    model_version = "all-MiniLM-L6-v2|mean|norm|256"

    cached = db.get_embeddings_batch(ids, model_version, story_hashes)
    missing_stories = [s for s in stories if s.id not in cached]

    if missing_stories:
        texts = [embedding_texts[s.id] for s in missing_stories]
        computed = embedder.encode(texts)
        for s, vec in zip(missing_stories, computed):
            db.upsert_embedding(s.id, model_version, story_hashes[s.id], vec)
            cached[s.id] = vec

    return np.array([cached[story_id] for story_id in ids], dtype=np.float32)


# Ranking

# Normalization constant for text-length metadata feature
_LOG_TEXTLEN_SCALE = 12.0  # log1p(~100000) ≈ 11.5


_SIM_CHUNK_SIZE = 1024


def _knn_similarity(
    query_emb: NDArray[np.float32],
    ref_emb: NDArray[np.float32],
    k: int,
    chunk_size: int = _SIM_CHUNK_SIZE,
) -> NDArray[np.float32]:
    """Mean of top-k cosine similarities, chunked over queries for low memory."""
    if ref_emb.shape[0] == 0:
        return np.zeros(query_emb.shape[0], dtype=np.float32)
    k_actual = min(k, ref_emb.shape[0])
    if k_actual <= 0:
        return np.zeros(query_emb.shape[0], dtype=np.float32)
    n_ref = ref_emb.shape[0]
    n = query_emb.shape[0]
    result = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_chunk = query_emb[start:end] @ ref_emb.T
        if k_actual == n_ref:
            topk = sim_chunk
        else:
            topk = np.partition(sim_chunk, n_ref - k_actual, axis=1)[:, -k_actual:]
        result[start:end] = topk.mean(axis=1)
    return result.astype(np.float32)


def _chunked_max_dot(
    query: NDArray[np.float32],
    ref: NDArray[np.float32],
    chunk_size: int = _SIM_CHUNK_SIZE,
) -> NDArray[np.float32]:
    """Chunked max-similarity: equivalent to np.max(query @ ref.T, axis=1)."""
    if ref.shape[0] == 0:
        return np.zeros(query.shape[0], dtype=np.float32)
    n = query.shape[0]
    result = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        result[start:end] = np.max(query[start:end] @ ref.T, axis=1)
    return result.astype(np.float32)


def _topk_mean(values: NDArray[np.float32], k: int) -> float:
    if k <= 0 or len(values) == 0:
        return 0.0
    k_actual = min(k, len(values))
    if k_actual == len(values):
        return float(values.mean())
    return float(np.partition(values, len(values) - k_actual)[-k_actual:].mean())


def _svm_personalization_features(
    embeddings: NDArray[np.float32],
    text_lengths: np.ndarray,
    sim_to_upvoted: np.ndarray,
    sim_to_downvoted: np.ndarray,
    closest_upvoted: np.ndarray,
    closest_downvoted: np.ndarray,
    positive_cluster_similarity: np.ndarray | None = None,
    is_hn_live: np.ndarray | None = None,
    is_archive: np.ndarray | None = None,
    is_reddit: np.ndarray | None = None,
    is_rss: np.ndarray | None = None,
) -> NDArray[np.float32]:
    """Production SVM features: embeddings, text length, feedback similarity,
    and 4-binary source category.

    Meta column layout (after the 384-d embedding):
      0  text_length (log1p, [0, 1])
      1  sim_to_upvoted        ([-1, 1] → [0, 1])
      2  sim_to_downvoted      ([-1, 1] → [0, 1])
      3  closest_upvoted       ([-1, 1] → [0, 1])
      4  closest_downvoted     ([-1, 1] → [0, 1])
      5  positive_cluster_similarity ([-1, 1] → [0, 1])
      6  is_hn_live            (0/1)
      7  is_archive            (0/1)
      8  is_reddit             (0/1)
      9  is_rss                (0/1)
    """
    meta = np.zeros((len(embeddings), 10), dtype=np.float32)
    meta[:, 0] = (
        np.clip(np.log1p(np.maximum(text_lengths, 0)), 0, _LOG_TEXTLEN_SCALE)
        / _LOG_TEXTLEN_SCALE
    )
    meta[:, 1] = (np.clip(sim_to_upvoted, -1, 1) + 1) / 2
    meta[:, 2] = (np.clip(sim_to_downvoted, -1, 1) + 1) / 2
    meta[:, 3] = (np.clip(closest_upvoted, -1, 1) + 1) / 2
    meta[:, 4] = (np.clip(closest_downvoted, -1, 1) + 1) / 2
    if positive_cluster_similarity is not None:
        meta[:, 5] = (np.clip(positive_cluster_similarity, -1, 1) + 1) / 2
    if is_hn_live is not None:
        meta[:, 6] = is_hn_live
    if is_archive is not None:
        meta[:, 7] = is_archive
    if is_reddit is not None:
        meta[:, 8] = is_reddit
    if is_rss is not None:
        meta[:, 9] = is_rss
    return np.concatenate([embeddings, meta], axis=1)


def _positive_cluster_centers(
    positive_embeddings: NDArray[np.float32],
    n_clusters: int,
) -> NDArray[np.float32]:
    if len(positive_embeddings) == 0 or n_clusters <= 0:
        return np.zeros((0, positive_embeddings.shape[1]), dtype=np.float32)
    unique_positive = np.unique(positive_embeddings, axis=0)
    if len(unique_positive) <= n_clusters:
        centers = unique_positive
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        kmeans.fit(unique_positive)
        centers = kmeans.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.clip(norms, a_min=1e-12, a_max=None)
    return centers.astype(np.float32)


def _positive_cluster_similarity(
    query_embeddings: NDArray[np.float32],
    positive_embeddings: NDArray[np.float32],
    n_clusters: int,
) -> NDArray[np.float32]:
    centers = _positive_cluster_centers(positive_embeddings, n_clusters)
    return _similarity_to_positive_cluster_centers(query_embeddings, centers)


def _similarity_to_positive_cluster_centers(
    query_embeddings: NDArray[np.float32],
    centers: NDArray[np.float32],
) -> NDArray[np.float32]:
    if len(query_embeddings) == 0 or len(centers) == 0:
        return np.zeros(len(query_embeddings), dtype=np.float32)
    return np.max(query_embeddings @ centers.T, axis=1).astype(np.float32)


def _minmax01(values: np.ndarray) -> NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float32)
    span = float(values.max() - values.min()) if len(values) else 0.0
    if span <= 1e-8:
        return np.full(len(values), 0.5, dtype=np.float32)
    return ((values - values.min()) / span).astype(np.float32)


def _rank_percentiles(values: np.ndarray) -> NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def _softmax_rows(values: np.ndarray) -> NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _loocv_knn_features(
    fb_embeddings: np.ndarray,
    class_embs: np.ndarray,
    class_indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(class_embs)
    sim_mat = fb_embeddings @ class_embs.T
    if n > 1:
        for idx, tp in enumerate(class_indices):
            sim_mat[tp, idx] = -2.0
    sim_to = np.zeros(len(fb_embeddings), dtype=np.float32)
    k_eff = min(k, n)
    for i in range(len(fb_embeddings)):
        sims = sim_mat[i]
        exclude = 1 if i in class_indices else 0
        n_available = max(1, n - exclude)
        k_use = min(k_eff, n_available)
        sim_to[i] = _topk_mean(sims, k_use)
    sim_mat_clean = fb_embeddings @ class_embs.T
    if n > 1:
        for idx, tp in enumerate(class_indices):
            sim_mat_clean[tp, idx] = -1.0
    closest = np.max(sim_mat_clean, axis=1)
    return sim_to, closest


def _score_and_rank(
    candidates: list[Story],
    candidate_embeddings: NDArray[np.float32],
    db: Database,
    config: Config,
    embedder: Embedder,
    user_id: int | None = None,
    trace: RankTrace | _NullTrace = NULL_TRACE,
    score_context: RankScoreContext | None = None,
) -> list[RankedStory]:
    if not candidates:
        return []

    now = time.time()
    scores = None
    probs = None
    feedback_stories, feedback_labels, _vote_times = db.get_feedback_for_training(
        user_id=user_id
    )

    n_feedback = len(feedback_labels)
    if trace is not None:
        trace.set_count("feedback_total", n_feedback)
    if score_context is not None:
        score_context.feedback_labels = list(feedback_labels)

    # Multiclass SVM: 0=down, 1=neutral, 2=up
    unique_classes = set(feedback_labels)
    fb_labels_arr = np.array(feedback_labels)
    n_up = int((fb_labels_arr == 2).sum())
    n_down = int((fb_labels_arr == 0).sum())
    n_neutral = int((fb_labels_arr == 1).sum())
    if trace is not None:
        trace.set_count("feedback_up", n_up)
        trace.set_count("feedback_down", n_down)
        trace.set_count("feedback_neutral", n_neutral)

    if (
        n_up >= config.model.min_up_for_svm
        and n_down >= config.model.min_down_for_svm
        and len(unique_classes) >= 2
    ):
        try:
            # Model cache lookup is intentionally before training-feature
            # construction. Cache hits still need candidate-side features,
            # but they do not need LOOCV training matrices.
            fb_sig = _feedback_signature(db, user_id) if user_id is not None else ""
            cached_model: tuple[SVC, StandardScaler] | None = None
            if fb_sig:
                cached_model = _get_cached_model(user_id, fb_sig)

            if trace is not None:
                with trace.stage("feedback_embedding"):
                    fb_embeddings = get_or_compute_embeddings(
                        feedback_stories, embedder, db
                    )
            else:
                fb_embeddings = get_or_compute_embeddings(
                    feedback_stories, embedder, db
                )
            if score_context is not None:
                score_context.feedback_embeddings = fb_embeddings

            # Personalization: mean/closest per class from ALL real feedback
            fb_labels_arr = np.array(feedback_labels)
            up_mask = fb_labels_arr == 2
            down_mask = fb_labels_arr == 0
            neutral_mask = fb_labels_arr == 1
            fb_up_embs = fb_embeddings[up_mask]
            fb_down_embs = fb_embeddings[down_mask]
            fb_neutral_embs = fb_embeddings[neutral_mask]

            n_up = int(up_mask.sum())
            n_down = int(down_mask.sum())
            k = config.model.knn_k
            emb_dim = candidate_embeddings.shape[1]

            with trace.stage("svm_candidate_feature_prep"):
                cand_sim_to_up = _knn_similarity(candidate_embeddings, fb_up_embs, k)
                cand_sim_to_down = _knn_similarity(
                    candidate_embeddings, fb_down_embs, k
                )
                cand_closest_up = _chunked_max_dot(candidate_embeddings, fb_up_embs)
                cand_closest_down = _chunked_max_dot(candidate_embeddings, fb_down_embs)
                cand_closest_neutral = _chunked_max_dot(
                    candidate_embeddings, fb_neutral_embs
                )
                positive_cluster_centers = _positive_cluster_centers(
                    fb_up_embs, config.model.positive_cluster_k
                )
                cand_positive_cluster_sim = _similarity_to_positive_cluster_centers(
                    candidate_embeddings, positive_cluster_centers
                )
                cand_text_lengths = np.array([len(s.text_content) for s in candidates])
                cand_source_onehot = source_category_stack(
                    [s.source for s in candidates]
                )
                cand_is_hn_live = cand_source_onehot[:, 0]
                cand_is_archive = cand_source_onehot[:, 1]
                cand_is_reddit = cand_source_onehot[:, 2]
                cand_is_rss = cand_source_onehot[:, 3]

                cand_features = _svm_personalization_features(
                    candidate_embeddings,
                    text_lengths=cand_text_lengths,
                    sim_to_upvoted=cand_sim_to_up,
                    sim_to_downvoted=cand_sim_to_down,
                    closest_upvoted=cand_closest_up,
                    closest_downvoted=cand_closest_down,
                    positive_cluster_similarity=cand_positive_cluster_sim,
                    is_hn_live=cand_is_hn_live,
                    is_archive=cand_is_archive,
                    is_reddit=cand_is_reddit,
                    is_rss=cand_is_rss,
                )
            if score_context is not None:
                score_context.cand_closest_up = cand_closest_up.astype(np.float32)
                score_context.cand_closest_down = cand_closest_down.astype(np.float32)
                score_context.cand_closest_neutral = cand_closest_neutral.astype(
                    np.float32
                )

            if cached_model is not None:
                if trace is not None:
                    trace.set_label("model_cache", "hit")
                svm, scaler = cached_model
            else:
                if trace is not None:
                    trace.set_label("model_cache", "miss")
                with trace.stage("svm_training_feature_prep"):
                    # LOOCV k-NN for training: exclude self from reference set
                    fb_sim_to_up = np.zeros(len(fb_embeddings), dtype=np.float32)
                    fb_sim_to_down = np.zeros(len(fb_embeddings), dtype=np.float32)
                    if n_up > 0:
                        up_indices = np.where(up_mask)[0]
                        fb_sim_to_up, fb_closest_up = _loocv_knn_features(
                            fb_embeddings, fb_up_embs, up_indices, k
                        )
                    else:
                        fb_closest_up = np.zeros(len(fb_embeddings), dtype=np.float32)

                    if n_down > 0:
                        down_indices = np.where(down_mask)[0]
                        fb_sim_to_down, fb_closest_down = _loocv_knn_features(
                            fb_embeddings, fb_down_embs, down_indices, k
                        )
                    else:
                        fb_closest_down = np.zeros(len(fb_embeddings), dtype=np.float32)

                    fb_positive_cluster_sim = _similarity_to_positive_cluster_centers(
                        fb_embeddings, positive_cluster_centers
                    )

                    fb_text_lengths = np.array(
                        [len(s.text_content) for s in feedback_stories]
                    )

                    # 4-binary source category one-hot per feedback story.
                    fb_source_onehot = source_category_stack(
                        [s.source for s in feedback_stories]
                    )
                    fb_is_hn_live = fb_source_onehot[:, 0]
                    fb_is_archive = fb_source_onehot[:, 1]
                    fb_is_reddit = fb_source_onehot[:, 2]
                    fb_is_rss = fb_source_onehot[:, 3]

                    fb_features = _svm_personalization_features(
                        fb_embeddings,
                        text_lengths=fb_text_lengths,
                        sim_to_upvoted=fb_sim_to_up,
                        sim_to_downvoted=fb_sim_to_down,
                        closest_upvoted=fb_closest_up,
                        closest_downvoted=fb_closest_down,
                        positive_cluster_similarity=fb_positive_cluster_sim,
                        is_hn_live=fb_is_hn_live,
                        is_archive=fb_is_archive,
                        is_reddit=fb_is_reddit,
                        is_rss=fb_is_rss,
                    )

                # Ensure all three classes (0, 1, 2) are present
                missing = {0, 1, 2} - set(feedback_labels)
                if missing:
                    fb_features = np.concatenate(
                        [
                            fb_features,
                            np.zeros(
                                (len(missing), fb_features.shape[1]), dtype=np.float32
                            ),
                        ],
                        axis=0,
                    )
                    labels = list(feedback_labels) + list(missing)
                else:
                    labels = list(feedback_labels)

                # Compute balanced weights for real feedback; 1e-6 for dummies
                counts = Counter(feedback_labels)
                n_classes = len(counts)
                n_real = len(feedback_labels)
                weights = [
                    n_real / (n_classes * counts[lbl]) for lbl in feedback_labels
                ]
                weights.extend([1e-6] * len(missing))
                sample_weights = np.array(weights, dtype=np.float64)

                scaler = StandardScaler()
                fb_features_meta_scaled = np.clip(
                    scaler.fit_transform(fb_features[:, emb_dim:]), -2.5, 2.5
                )
                fb_features_scaled = np.hstack(
                    [fb_features[:, :emb_dim], fb_features_meta_scaled]
                )
                svm = SVC(
                    C=config.model.svm_c,
                    kernel=config.model.svm_kernel,
                    gamma=config.model.svm_gamma,
                    cache_size=16,
                    random_state=0,
                    decision_function_shape="ovr",
                )
                if trace is not None:
                    with trace.stage("svm_fit"):
                        svm.fit(
                            fb_features_scaled, labels, sample_weight=sample_weights
                        )
                else:
                    svm.fit(fb_features_scaled, labels, sample_weight=sample_weights)
                if fb_sig:
                    _set_cached_model(
                        user_id, fb_sig, svm, scaler, config.max_cached_models
                    )

            with trace.stage("svm_candidate_scale"):
                cand_features_meta_scaled = np.clip(
                    scaler.transform(cand_features[:, emb_dim:]), -2.5, 2.5
                )
                cand_features_scaled = np.hstack(
                    [cand_features[:, :emb_dim], cand_features_meta_scaled]
                )

            class_order = list(svm.classes_)
            idx_up = class_order.index(2)
            if trace is not None:
                with trace.stage("decision"):
                    decision = svm.decision_function(cand_features_scaled)
            else:
                decision = svm.decision_function(cand_features_scaled)
            if decision.ndim == 1:
                raw_scores = decision if class_order[-1] == 2 else -decision
                probs = np.column_stack(
                    [1 - _minmax01(raw_scores), _minmax01(raw_scores)]
                )
            else:
                raw_scores = decision[:, idx_up]
                probs = _softmax_rows(decision)
            scores = _minmax01(raw_scores)
        except Exception as e:
            logging.error("Failed to fit feedback SVM: %r", e)
    elif trace is not None:
        trace.set_label("model_cache", "skipped")

    svm_scores = scores
    svm_probs = probs

    # Tier 2: centroid-based scores (always compute when feedback exists)
    tier2_scores: NDArray[np.float32] | None = None
    if n_feedback > 0:
        with trace.stage("tier2"):
            fb_embs = get_or_compute_embeddings(feedback_stories, embedder, db)
            fb_labels_arr = np.array(feedback_labels)
            up_mask = fb_labels_arr == 2
            down_mask = fb_labels_arr == 0

            if up_mask.any() or down_mask.any():
                up_emb = (
                    fb_embs[up_mask].mean(axis=0)
                    if up_mask.any()
                    else np.zeros(384, dtype=np.float32)
                )
                down_emb = (
                    fb_embs[down_mask].mean(axis=0)
                    if down_mask.any()
                    else np.zeros(384, dtype=np.float32)
                )

                sim_up = candidate_embeddings @ up_emb
                sim_down = candidate_embeddings @ down_emb
                tier2_scores = sim_up - sim_down
                tier2_scores = (tier2_scores - tier2_scores.min()) / (
                    tier2_scores.max() - tier2_scores.min() + 1e-8
                )
            else:
                tier2_scores = np.full(len(candidates), 0.5, dtype=np.float32)

    # Tier 1: HN gravity (frontpage-like) — always computed for cold-start blend.
    # Per-source priors are now learned by the SVM via the 4-binary source
    # category features; the previous `* 2` non-HN boost was removed since the
    # SVM already has the prior signal directly and the boost double-counted.
    tier1_scores = np.array(
        [
            s.score / max(((now - s.time) / 3600.0 + 2.0) ** 1.8, 0.1)
            for s in candidates
        ],
        dtype=np.float32,
    )
    if tier1_scores.max() > 0:
        tier1_scores = tier1_scores / tier1_scores.max()

    # Three-way blend between tier 1 (gravity), tier 2 (centroid), tier 3 (SVM)
    # α_2 ramps from 0→1 as n_feedback grows; tier 1 fades out smoothly.
    alpha_2 = float(np.clip(n_feedback / config.model.tier2_blend_window, 0.0, 1.0))

    if svm_scores is not None and tier2_scores is not None:
        n_min = min(n_up, n_down)
        blend_start = min(config.model.min_up_for_svm, config.model.min_down_for_svm)
        alpha_3 = float(
            np.clip(
                (n_min - blend_start) / config.model.tier3_blend_window,
                0.0,
                1.0,
            )
        )
        t1_weight = 1.0 - alpha_2
        t2_weight = alpha_2 * (1.0 - alpha_3)
        t3_weight = alpha_2 * alpha_3
        scores = np.asarray(
            t1_weight * tier1_scores
            + t2_weight * tier2_scores
            + t3_weight * svm_scores,
            dtype=np.float32,
        )
    elif tier2_scores is not None:
        t1_weight = 1.0 - alpha_2
        scores = np.asarray(
            t1_weight * tier1_scores + alpha_2 * tier2_scores,
            dtype=np.float32,
        )
    else:
        scores = tier1_scores

    assert scores is not None

    ranked: list[RankedStory] = []
    if svm_probs is not None:
        try:
            idx_down = class_order.index(0)
            idx_neutral = class_order.index(1)
            idx_up = class_order.index(2)
            for idx, (s, score) in enumerate(zip(candidates, scores)):
                ranked.append(
                    RankedStory(
                        story=s,
                        score=float(score),
                        best_match_title="",
                        prob_down=float(svm_probs[idx, idx_down]),
                        prob_neutral=float(svm_probs[idx, idx_neutral]),
                        prob_up=float(svm_probs[idx, idx_up]),
                    )
                )
        except (ValueError, IndexError, NameError) as e:
            logging.error("Error mapping probability class indices: %r", e)
            ranked = []

    if not ranked:
        for s, score in zip(candidates, scores):
            ranked.append(
                RankedStory(
                    story=s,
                    score=float(score),
                    best_match_title="",
                )
            )

    if n_feedback == 0:
        return sorted(ranked, key=lambda x: x.story.score, reverse=True)
    else:
        return sorted(ranked, key=lambda x: x.score, reverse=True)


# MMR
def mmr_filter(
    ranked: list[RankedStory],
    embeddings_map: dict[int, NDArray[np.float32]],
    threshold: float = 0.85,
    limit: int = 40,
) -> list[RankedStory]:
    selected = []
    discarded = set()

    for idx, item in enumerate(ranked):
        if item.story.id in discarded:
            continue

        emb = embeddings_map.get(item.story.id)
        selected.append(item)

        if emb is not None:
            for other in ranked[idx + 1 :]:
                if other.story.id in discarded:
                    continue
                other_emb = embeddings_map.get(other.story.id)
                if other_emb is not None:
                    sim = float(np.dot(emb, other_emb))
                    if sim > threshold:
                        discarded.add(other.story.id)

        if len(selected) >= limit:
            break

    # Sort selected items back to their original relative order in ranked
    selected.sort(key=lambda x: ranked.index(x))
    return selected


def rerank_candidates(
    db: Database,
    config: Config,
    embedder: Embedder,
    candidates: list[Story],
    cand_embeddings: NDArray[np.float32] | None = None,
    user_id: int | None = None,
    trace: RankTrace | _NullTrace = NULL_TRACE,
) -> list[RankedStory]:
    """Rank candidates and attach discovery badges.

    This wraps :func:`_score_and_rank` (which only does tier blend + sort)
    and adds badge attribution + 7 discovery passes (uncertainty, novelty,
    similarity, discussion-rich, high-engagement, hot, non-HN).

    Use this in production; the private ``_score_and_rank`` is intended for
    tier-blend tests that need to assert on ranking without badge side effects.
    """
    if not candidates:
        return []
    if trace is not None:
        trace.set_count("candidates", len(candidates))

    if cand_embeddings is None:
        if trace is not None:
            with trace.stage("candidate_embedding"):
                cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)
        else:
            cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    score_context = RankScoreContext()
    ranked = _score_and_rank(
        candidates,
        cand_embeddings,
        db,
        config,
        embedder,
        user_id=user_id,
        trace=trace,
        score_context=score_context,
    )

    # Build MMR embeddings map once (used per combo)
    embeddings_map: dict[int, NDArray[np.float32]] = {}
    if config.model.enable_mmr:
        embeddings_map = {s.id: vec for s, vec in zip(candidates, cand_embeddings)}

    with trace.stage("badge_similarity"):
        # Reuse vectors from _score_and_rank when SVM was trained;
        # otherwise compute them (chunked) on demand.
        if score_context.cand_closest_up is not None:
            cand_closest_up = score_context.cand_closest_up
            cand_closest_down = score_context.cand_closest_down
            cand_closest_neutral = score_context.cand_closest_neutral
        else:
            feedback_stories, feedback_labels, _ = db.get_feedback_for_training(
                user_id=user_id
            )
            fb_labels_arr = np.array(feedback_labels)
            up_mask = fb_labels_arr == 2
            down_mask = fb_labels_arr == 0
            neutral_mask = fb_labels_arr == 1
            fb_embs = get_or_compute_embeddings(feedback_stories, embedder, db)
            cand_closest_up = (
                _chunked_max_dot(cand_embeddings, fb_embs[up_mask])
                if up_mask.any()
                else np.zeros(len(candidates), dtype=np.float32)
            )
            cand_closest_down = (
                _chunked_max_dot(cand_embeddings, fb_embs[down_mask])
                if down_mask.any()
                else np.zeros(len(candidates), dtype=np.float32)
            )
            cand_closest_neutral = (
                _chunked_max_dot(cand_embeddings, fb_embs[neutral_mask])
                if neutral_mask.any()
                else np.zeros(len(candidates), dtype=np.float32)
            )

        assert cand_closest_down is not None and cand_closest_neutral is not None
        cand_max_sim = np.maximum.reduce(
            [cand_closest_up, cand_closest_down, cand_closest_neutral]
        )

    # Hoist now_ts and compute per-age-bucket metadata early so all
    # threshold computations below can produce per-candidate arrays.
    # Per-bucket thresholds are required because archive candidates have
    # structurally higher absolute scores/comment counts (months/years of
    # accumulation) than recent candidates; a single global threshold
    # would be archive-dominated and make it nearly impossible for
    # recent stories to earn Top/Talk-worthy badges. Similar logic
    # applies to Novel/Similar/Unsure so badges remain meaningful
    # within each age cohort.
    cand_scores = np.array([s.score for s in candidates])
    now_ts = time.time()
    recent_cutoff = int(now_ts) - 30 * 86400
    cand_velocities = np.array(
        [s.score / max((now_ts - s.time) / 3600.0, 0.1) for s in candidates]
    )

    story_id_to_idx = {s.id: idx for idx, s in enumerate(candidates)}
    idx_for = story_id_to_idx.__getitem__

    def get_entropy(r: RankedStory) -> float:
        ent = 0.0
        for p in (r.prob_down, r.prob_neutral, r.prob_up):
            if p is not None and p > 1e-9:
                ent -= p * np.log2(p)
        return ent

    def _novel_sort_key(r: RankedStory) -> float:
        return float(1.0 - cand_max_sim[idx_for(r.story.id)])

    def _similar_sort_key(r: RankedStory) -> float:
        return float(cand_closest_up[idx_for(r.story.id)])

    def _discussion_sort_key(r: RankedStory) -> float:
        return float(r.story.comment_count or 0)

    def _engagement_sort_key(r: RankedStory) -> float:
        return float(cand_scores[idx_for(r.story.id)])

    def _hot_sort_key(r: RankedStory) -> float:
        return float(cand_velocities[idx_for(r.story.id)])

    def _entropy_sort_key(r: RankedStory) -> float:
        return float(get_entropy(r))

    # Per-combo deck construction. Three combos: recent_hn, recent_nonhn,
    # archive_hn. Each combo gets PRIMARY_PER_COMBO primary cards (MMR if
    # enabled, otherwise top-score), plus DISCOVERY_PER_BADGE cards for each
    # of Hot/Top/Talk (Popular, HN only) and Unsure/Novel/Similar (Explore).
    #
    # Cards carry space-separated combo_keys so the client can filter by
    # age+source without computing offsets (e.g. "recent_hn recent_mixed").
    COMBO_DEFS: list[tuple[str, str, int]] = [
        ("recent", "hn", PRIMARY_PER_COMBO),
        ("recent", "nonhn", PRIMARY_PER_COMBO),
        ("archive", "hn", PRIMARY_PER_COMBO),
        ("archive", "nonhn", PRIMARY_PER_COMBO),
    ]

    final: list[RankedStory] = []

    for age, source, primary_limit in COMBO_DEFS:
        # Filter candidate pool to this combo
        if age == "recent":
            age_pool = [r for r in ranked if r.story.time >= recent_cutoff]
        else:
            age_pool = [r for r in ranked if r.story.time < recent_cutoff]

        if source == "hn":
            combo_pool = [r for r in age_pool if is_hn_source(r.story.source)]
        else:
            combo_pool = [r for r in age_pool if not is_hn_source(r.story.source)]

        if not combo_pool:
            continue

        source_key = age + ("_hn" if source == "hn" else "_non-hn")
        mixed_key = age + "_mixed"

        # --- Primary selection ---
        if primary_limit > 0:
            if config.model.enable_mmr and embeddings_map:
                primary = mmr_filter(
                    combo_pool,
                    embeddings_map,
                    threshold=config.model.diversity_threshold,
                    limit=primary_limit,
                )
            else:
                combo_sort = sorted(combo_pool, key=lambda r: r.score, reverse=True)
                primary = combo_sort[:primary_limit]

            final.extend(
                replace(r, combo_keys=f"{source_key} {mixed_key}") for r in primary
            )
        else:
            primary = []

        primary_ids = {r.story.id for r in primary}

        # --- Popular (HN only): Hot + Top + Talk ---
        # Each pass sees the full combo_pool and can OR badges onto stories
        # already in primary, matching the old global Hot/Top/Talk behavior.
        # Cascade: Hot → Top → Talk (each excludes prior picks).
        if source == "hn":
            # Hot: full combo_pool, gated by velocity percentile + score floor
            hot_threshold = (
                np.percentile(cand_velocities, config.model.hot_badge_percentile)
                if len(cand_velocities)
                else 0
            )
            hot_pool = [
                r
                for r in combo_pool
                if r.story.score >= HOT_MIN_SCORE
                and cand_velocities[idx_for(r.story.id)] >= hot_threshold
            ]
            if hot_pool:
                hot_pool.sort(key=_hot_sort_key, reverse=True)
                for r in hot_pool[:DISCOVERY_PER_BADGE]:
                    existing = next(
                        (i for i, f in enumerate(final) if f.story.id == r.story.id),
                        None,
                    )
                    new_r = replace(
                        r, is_hot=True, combo_keys=f"{source_key} {mixed_key}"
                    )
                    if existing is not None:
                        final[existing] = replace(
                            final[existing],
                            is_hot=True,
                            combo_keys=f"{source_key} {mixed_key}",
                        )
                    else:
                        final.append(new_r)

            # Top: full combo_pool minus Hot picks, sorted by engagement score
            hot_and_primary = {r.story.id for r in final if r.is_hot} | primary_ids
            top_pool = sorted(
                [r for r in combo_pool if r.story.id not in hot_and_primary],
                key=_engagement_sort_key,
                reverse=True,
            )[:DISCOVERY_PER_BADGE]
            for r in top_pool:
                final.append(
                    replace(
                        r,
                        is_high_engagement=True,
                        combo_keys=f"{source_key} {mixed_key}",
                    )
                )

            # Talk: full combo_pool minus Hot+Top picks, sorted by discussion
            hot_top_and_primary = {r.story.id for r in final} | primary_ids
            talk_pool = sorted(
                [
                    r
                    for r in combo_pool
                    if r.story.id not in hot_top_and_primary
                    and (r.story.comment_count or 0) > 0
                ],
                key=_discussion_sort_key,
                reverse=True,
            )[:DISCOVERY_PER_BADGE]
            for r in talk_pool:
                final.append(
                    replace(
                        r,
                        is_discussion_rich=True,
                        combo_keys=f"{source_key} {mixed_key}",
                    )
                )

        # --- Explore: Unsure + Novel + Similar ---
        # Explore passes see the full combo pool (minus primary) and can
        # stack badges on stories already picked by Popular, matching the
        # old parallel-pass design. Within Explore, Unsure/Novel/Similar
        # are serial (mutually exclusive) to keep the badge mix varied.
        explore_pool = [r for r in combo_pool if r.story.id not in primary_ids]
        explore_picked = set[int]()  # only track Explore picks, not Popular

        unsure_items = sorted(
            [r for r in explore_pool if r.prob_down is not None],
            key=_entropy_sort_key,
            reverse=True,
        )[:DISCOVERY_PER_BADGE]
        for r in unsure_items:
            existing = next(
                (i for i, f in enumerate(final) if f.story.id == r.story.id), None
            )
            new_r = replace(
                r, is_uncertain=True, combo_keys=f"{source_key} {mixed_key}"
            )
            if existing is not None:
                final[existing] = replace(
                    final[existing],
                    is_uncertain=True,
                    combo_keys=f"{source_key} {mixed_key}",
                )
            else:
                final.append(new_r)
            explore_picked.add(r.story.id)

        novel_items = sorted(
            [r for r in explore_pool if r.story.id not in explore_picked],
            key=_novel_sort_key,
            reverse=True,
        )[:DISCOVERY_PER_BADGE]
        for r in novel_items:
            final.append(
                replace(r, is_novel=True, combo_keys=f"{source_key} {mixed_key}")
            )
            explore_picked.add(r.story.id)

        similar_items = sorted(
            [r for r in explore_pool if r.story.id not in explore_picked],
            key=_similar_sort_key,
            reverse=True,
        )[:DISCOVERY_PER_BADGE]
        for r in similar_items:
            final.append(
                replace(r, is_similar=True, combo_keys=f"{source_key} {mixed_key}")
            )
            explore_picked.add(r.story.id)

    # Set is_recent and is_non_hn on every story in `final` (these flags are
    # source/time based, not rank-based, so they always reflect the current
    # candidate's metadata regardless of how it was selected).
    final = [
        replace(
            r,
            is_non_hn=(not is_hn_source(r.story.source)),
            is_recent=(r.story.time >= recent_cutoff),
        )
        for r in final
    ]

    final.sort(key=lambda r: r.score, reverse=True)
    return final
