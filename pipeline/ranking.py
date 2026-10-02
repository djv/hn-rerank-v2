from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import resource
import threading
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, TypeAlias, TypedDict

import numpy as np
import onnxruntime as ort
from bs4 import BeautifulSoup
from cachetools import LRUCache
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from clients.tui.src.hn_rerank.models import VIEWS, WINDOWS, View, Window
from database import Database, Story
from .config import (
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_SOURCE,
    Config,
    DEFAULT_EMBEDDING_MAX_TOKENS,
    DEFAULT_EMBEDDING_MODEL_VERSION,
    DEFAULT_ONNX_MODEL_DIR,
    is_hn_source,
)
from . import linear_blend
from .model_manifest import ModelManifest, verify_model_dir

if TYPE_CHECKING:
    from ch_client import ChItem


# Lazy seam: `transformers` costs ~0.9s at import and is only needed to
# construct a real Embedder. Production imports it on first use;
# tests patch this name directly (see
# test_embedder_uses_configured_batch_and_ort_variant).
AutoTokenizer: Any = None


def _tokenizer_cls() -> Any:
    if AutoTokenizer is not None:
        return AutoTokenizer
    from transformers import AutoTokenizer as _Cls

    return _Cls


EmbeddingOrtVariant: TypeAlias = Literal[
    "current",
    "spin_off",
    "spin_off_graph_all",
    "spin_off_auto_threads",
]


def _process_rss_kb() -> int | None:
    """Return current process RSS on Linux, or None when unavailable."""
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


# SVM model cache: keyed on (user_id, feedback_signature, schema_version)
# to skip SVC.fit(). Bump _MODEL_SCHEMA_VERSION whenever the feature schema
# (number / semantics of meta columns appended to the embedding) changes;
# the cache key then changes for every user, forcing a clean re-fit.
_MODEL_CACHE_STORAGE_MAXSIZE = 10_000


# Cached value: (svm, scaler, positive_cluster_centers). The centers depend
# only on the up-voted feedback embeddings — same invalidation as the SVM —
# so they are cached alongside it to skip the per-regen KMeans on cache hits.
# ``centers`` may be None for entries written before this field existed (or by
# tests); callers must fall back to recomputing when it is None.
class PrecomputedRbfSVC:
    """Exact RBF SVC with bounded candidate-kernel inference memory."""

    def __init__(self, *, c: float, gamma: float, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.gamma = gamma
        self.chunk_size = chunk_size
        self._svc = SVC(
            C=c,
            kernel="precomputed",
            cache_size=16,
            random_state=0,
            decision_function_shape="ovr",
        )
        self._training_features: NDArray[np.float64] | None = None

    @property
    def classes_(self) -> NDArray[np.int64]:
        return self._svc.classes_

    def fit(
        self,
        features: NDArray[np.float64],
        labels: list[int],
        *,
        sample_weight: NDArray[np.float64],
    ) -> PrecomputedRbfSVC:
        self._training_features = np.asarray(features)
        kernel = rbf_kernel(features, features, gamma=self.gamma)
        self._svc.fit(kernel, labels, sample_weight=sample_weight)
        return self

    def decision_function(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._training_features is None:
            raise RuntimeError("fit must run before decision_function")
        chunks = []
        for start in range(0, len(features), self.chunk_size):
            kernel = rbf_kernel(
                features[start : start + self.chunk_size],
                self._training_features,
                gamma=self.gamma,
            )
            chunks.append(self._svc.decision_function(kernel))
        if not chunks:
            width = len(self.classes_) if len(self.classes_) > 2 else 1
            return np.empty((0, width), dtype=np.float64)
        return np.concatenate(chunks, axis=0)


_CachedClassifier: TypeAlias = SVC | PrecomputedRbfSVC
_CachedModel = tuple[_CachedClassifier, StandardScaler, "NDArray[np.float32] | None"]
_MODEL_CACHE: LRUCache[tuple[int, str, int], _CachedModel] = LRUCache(
    maxsize=_MODEL_CACHE_STORAGE_MAXSIZE
)
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_SCHEMA_VERSION = 6  # +1 whenever model/feature schema changes (see ARCHITECTURE)


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
    # F2 attribution: argmax row (into fb_up_titles) of the closest upvoted
    # feedback story per candidate, plus the aligned up-story titles.
    cand_closest_up_idx: NDArray[np.int64] | None = None
    fb_up_titles: list[str] = field(default_factory=list)
    # The upvoted stories' embeddings, for Explore's Interest clusters.
    fb_up_embeddings: NDArray[np.float32] | None = None


def _feedback_signature(db: Database, user_id: int) -> str:
    feedback = db.get_all_feedback(user_id=user_id)
    hasher = hashlib.sha256()
    for f in sorted(feedback, key=lambda x: x.story_id):
        hasher.update(f"{f.story_id}:{f.action}:{f.updated_at}".encode())
    return hasher.hexdigest()


def _get_cached_model(user_id: int | None, signature: str) -> _CachedModel | None:
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
    svm: _CachedClassifier,
    scaler: StandardScaler,
    maxsize: int = 20,
    *,
    centers: NDArray[np.float32] | None = None,
) -> None:
    if user_id is None:
        return
    with _MODEL_CACHE_LOCK:
        key = (user_id, signature, _MODEL_SCHEMA_VERSION)
        _MODEL_CACHE[key] = (svm, scaler, centers)
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
# Comment join for storage: explicit markdown boundary so the TLDR prompt
# sees segments, not soup. Matches server.py's COMMENT_PROMPT_CHAR_LIMIT —
# retain everything the prompt assembler will actually use.
HN_COMMENTS_SEPARATOR = "\n\n---\n\n"
HN_COMMENTS_CACHE_CHAR_LIMIT = 24_000
HOT_MIN_SCORE = 20
# A served view holds this many stories: the 12 the clients show (VIEW_LIMIT
# in templates/index.html and the TUI) plus 4 that slide in as cards ahead
# of them are voted.
VIEW_SIZE = 16
# Explore serves this many each of Unsure, Novel and Interest.
EXPLORE_PER_BADGE = 5
# Views are picked at this multiple of their served size, so that dedup,
# votes and stories ageing out of a window between warms leave enough.
SELECT_MARGIN = 2
# Windows are nested by age; "archive" is everything older than 30 days.
# Boundaries are inclusive: a story exactly 30 days old is in "1m".
WINDOW_SECONDS: dict[Window, int] = {
    "12h": 12 * 3600,
    "1d": 86400,
    "1w": 7 * 86400,
    "1m": 30 * 86400,
}
# Popular's gravity clock per window: age counts in units of this many hours,
# a third of the window (archive: of a year). With HN's own clock (1 hour)
# every window's Popular was the same few stories under a day old; on the
# 2026-09-28 snapshot this clock gives 1w a median age of a day and 1m of a
# week. Clients mirror this table for undo.
GRAVITY_TIME_SCALE: dict[Window, float] = {
    "12h": 4.0,
    "1d": 8.0,
    "1w": 56.0,
    "1m": 240.0,
    "archive": 2920.0,
}
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
    # Similarity to that upvoted story (0 when there is none).
    best_match_sim: float = 0.0
    prob_down: float | None = None
    prob_neutral: float | None = None
    prob_up: float | None = None
    is_uncertain: bool = False
    is_novel: bool = False
    is_discussion_rich: bool = False
    is_high_engagement: bool = False
    is_hot: bool = False
    is_interest: bool = False


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
        # Unescaping can recreate tag-looking fragments (for example
        # ``&lt;0>`` becomes ``<0>``) after BeautifulSoup has parsed the input.
        # Strip those residual tags before applying the text-only invariants.
        txt = re.sub(r"<[^>]*>", " ", txt)

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


class RankedComment(TypedDict):
    """Normalized comment from ``_extract_comments_recursive``.

    ``id`` stays ``Any`` (opaque JSON passthrough, never arithmeticked);
    everything the ranker reads is explicitly typed.
    """

    id: Any
    text: str
    score: int
    depth: int
    top_thread_index: int | None
    sibling_index: int
    order_path: tuple[int, ...]
    reply_count: int
    descendant_count: int
    text_len: int


def _extract_comments_recursive(
    children: Sequence[ChItem] | Sequence[dict[str, Any]],
    depth: int = 0,
    parent_points: int = 0,
    top_thread_index: int | None = None,
    order_path: tuple[int, ...] = (),
) -> list[RankedComment]:
    MIN_COMMENT_LENGTH = 60
    results: list[RankedComment] = []
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


def _comment_rank_key(comment: RankedComment) -> tuple[int, int, tuple[int, ...]]:
    return (
        -comment["descendant_count"],
        -min(comment["text_len"], 3000),
        comment["order_path"],
    )


def _select_top_comments(
    comments: list[RankedComment],
    limit: int = TOP_COMMENT_LIMIT,
) -> list[RankedComment]:
    """Select comment text for embeddings/TLDRs.

    Prefer large discussion cores (top engaged threads) and breadth of
    substantive top-level comments.  Replies compete on equal footing with
    top-level (no depth penalty); the previous ``score``-based rank key
    effectively preferred top-level regardless of substance, since Algolia
    returns ``points: null`` for HN comments.
    """
    if not comments:
        return []

    selected: list[RankedComment] = []
    selected_indexes: set[int] = set()
    per_thread: dict[int, int] = {}

    def add(comment: RankedComment) -> None:
        if len(selected) >= limit:
            return
        index = id(comment)
        thread_index = comment["top_thread_index"]
        if thread_index is None:
            return
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
        key=lambda c: (
            -c["descendant_count"],
            c["top_thread_index"] if c["top_thread_index"] is not None else -1,
        ),
    )[:n_cores]
    core_threads = {c["top_thread_index"] for c in core_roots}

    for root in sorted(core_roots, key=_comment_rank_key):
        add(root)

    for thread_index in sorted(t for t in core_threads if t is not None):
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


def join_top_comments(
    texts: list[str], limit: int = HN_COMMENTS_CACHE_CHAR_LIMIT
) -> str:
    """Join selected comment texts on a markdown boundary, within budget.

    Boundary-aware: each comment is appended whole or not at all — never
    sliced mid-comment. Oversized comments are skipped, not truncating the
    pack. Blank entries are skipped.
    """
    parts: list[str] = []
    total = 0
    for text in texts:
        if not text or not text.strip():
            continue
        sep_len = 0 if not parts else len(HN_COMMENTS_SEPARATOR)
        if total + sep_len + len(text) > limit:
            continue
        if parts:
            parts.append(HN_COMMENTS_SEPARATOR)
        parts.append(text)
        total += sep_len + len(text)
    return "".join(parts)


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


def _embedding_session_options(ort_variant: EmbeddingOrtVariant) -> ort.SessionOptions:
    session_options = ort.SessionOptions()
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    if ort_variant == "current":
        return session_options
    if ort_variant == "spin_off":
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return session_options
    if ort_variant == "spin_off_graph_all":
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        return session_options
    if ort_variant == "spin_off_auto_threads":
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = 0
        session_options.inter_op_num_threads = 1
        return session_options
    raise ValueError(f"Unknown embedding ORT variant: {ort_variant}")


# Single encode() calls slower than this get a loud WARN (stall monsters:
# one giant batch hogging CPU next to a warm, see WORKLOG 2026-09-08).
# Seam for tests: patched to 0 to force the WARN without a real 10s stall.
_EMBEDDING_SLOW_WARN_SECONDS = 10.0


class Embedder:
    model_version = DEFAULT_EMBEDDING_MODEL_VERSION
    max_tokens = DEFAULT_EMBEDDING_MAX_TOKENS
    # Provenance for new embedding rows (class attrs so test doubles that
    # skip __init__ still expose them; real inits overwrite below).
    model_onnx_sha: str = ""
    embedding_dim: int = 384
    model_manifest: ModelManifest | None = None

    def __init__(
        self,
        model_dir: str = DEFAULT_ONNX_MODEL_DIR,
        *,
        model_version: str = DEFAULT_EMBEDDING_MODEL_VERSION,
        max_tokens: int = DEFAULT_EMBEDDING_MAX_TOKENS,
        batch_size: int = 32,
        ort_variant: EmbeddingOrtVariant = "current",
    ) -> None:
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_version = model_version
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.tokenizer: Any = _tokenizer_cls().from_pretrained(model_dir)
        session_options = _embedding_session_options(ort_variant)
        self.session = ort.InferenceSession(
            str(Path(model_dir) / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.model_manifest, mismatches = self._check_manifest(model_dir)
        self.model_onnx_sha = (self.model_manifest.files or {}).get("model.onnx", "")
        self.embedding_dim = self._resolve_embedding_dim(model_dir)
        if mismatches:
            # Warn-and-serve: the bytes changed under the recorded baseline,
            # but stored vectors keep matching (version string untouched).
            # Loud log, no refusal — uptime beats strictness here.
            logging.error(
                "embedding_model_changed dir=%s serving_recorded_vectors "
                "mismatches=%s (re-baseline deliberately via setup_model.py)",
                model_dir,
                "; ".join(mismatches),
            )

    @staticmethod
    def _manifest_repo() -> str:
        return "mixedbread-ai/mxbai-embed-xsmall-v1"

    def _check_manifest(self, model_dir: str) -> tuple[ModelManifest, list[str]]:
        # Manifest IO must never break init (read-only mounts, fake dirs in
        # tests) and must never write: baselines belong to setup_model.py,
        # so request paths and test doubles can't alter prod state.
        # Verification is best-effort; serving is not gated on it.
        try:
            stored, mismatches = verify_model_dir(
                model_dir, self._manifest_repo(), self._manifest_revision()
            )
        except OSError as exc:
            logging.warning(
                "embedding_manifest_check_skipped dir=%s error=%r", model_dir, exc
            )
            return (
                ModelManifest(repo="", revision="", files={}, created_at=0.0),
                [],
            )
        if stored is None:
            logging.warning(
                "embedding_manifest_missing dir=%s; run setup_model.py to baseline",
                model_dir,
            )
            return ModelManifest(repo="", revision="", files={}, created_at=0.0), []
        return stored, mismatches

    @staticmethod
    def _manifest_revision() -> str:
        try:
            from setup_model import MODEL_REPO, MODEL_REVISION

            if MODEL_REPO == Embedder._manifest_repo():
                return MODEL_REVISION
        except ImportError:
            pass
        return "main"

    @staticmethod
    def _resolve_embedding_dim(model_dir: str) -> int:
        try:
            hidden = json.loads((Path(model_dir) / "config.json").read_text())[
                "hidden_size"
            ]
            dim = int(hidden)
            if dim > 0:
                return dim
        except (OSError, ValueError, KeyError, TypeError):
            pass
        logging.warning(
            "embedding_dim unreadable from %s/config.json; falling back to 384",
            model_dir,
        )
        return 384

    def encode(
        self, texts: list[str], batch_size: int | None = None
    ) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be positive")

        started = time.perf_counter()
        rss_before_kb = _process_rss_kb()
        batch_count = 0
        longest_tokens = 0
        embeddings = []
        for i in range(0, len(texts), effective_batch_size):
            batch_texts = texts[i : i + effective_batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="np",
            )
            batch_count += 1
            longest_tokens = max(longest_tokens, int(inputs["input_ids"].shape[1]))

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

        result = np.concatenate(embeddings, axis=0)
        rss_after_kb = _process_rss_kb()
        rss_delta_kb = (
            rss_after_kb - rss_before_kb
            if rss_before_kb is not None and rss_after_kb is not None
            else None
        )
        duration_s = time.perf_counter() - started
        logging.info(
            "embedding_perf texts=%d batches=%d batch_size=%d max_tokens=%d "
            "longest_tokens=%d duration_ms=%.1f rss_before_kb=%s "
            "rss_after_kb=%s rss_delta_kb=%s peak_rss_kb=%d",
            len(texts),
            batch_count,
            effective_batch_size,
            self.max_tokens,
            longest_tokens,
            duration_s * 1000,
            rss_before_kb,
            rss_after_kb,
            rss_delta_kb,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        if duration_s >= _EMBEDDING_SLOW_WARN_SECONDS:
            logging.warning(
                "embedding_slow texts=%d batches=%d longest_tokens=%d "
                "duration_s=%.1f (threshold_s=%.1f)",
                len(texts),
                batch_count,
                longest_tokens,
                duration_s,
                _EMBEDDING_SLOW_WARN_SECONDS,
            )
        return result


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
    model_version = embedder.model_version

    cached = db.get_embeddings_batch(
        ids, model_version, story_hashes, expected_dim=embedder.embedding_dim
    )
    missing_stories = [s for s in stories if s.id not in cached]

    if missing_stories:
        texts = [embedding_texts[s.id] for s in missing_stories]
        computed = embedder.encode(texts)
        for s, vec in zip(missing_stories, computed):
            db.upsert_embedding(
                s.id,
                model_version,
                story_hashes[s.id],
                vec,
                model_sha=embedder.model_onnx_sha,
                dim=embedder.embedding_dim,
            )
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


def _knn_mean_and_max(
    query_emb: NDArray[np.float32],
    ref_emb: NDArray[np.float32],
    k: int,
    chunk_size: int = _SIM_CHUNK_SIZE,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """Fused (top-k mean, max, argmax) cosine similarity in a single dot pass.

    Exactly equivalent to the pair
    ``(_knn_similarity(query, ref, k), _chunked_max_dot(query, ref))`` but
    computes the ``query @ ref.T`` matrix once instead of twice — the max is
    just the top-1 of the same similarity rows the k-NN mean already reduces.
    The argmax row index (into ref) is one extra reduction over the same
    chunk; callers needing attribution (F2) get it with no new matmul.
    """
    n = query_emb.shape[0]
    if ref_emb.shape[0] == 0:
        zeros = np.zeros(n, dtype=np.float32)
        return zeros, zeros.copy(), np.full(n, -1, dtype=np.int64)
    n_ref = ref_emb.shape[0]
    k_actual = min(k, n_ref)
    mean_out = np.zeros(n, dtype=np.float32)
    max_out = np.zeros(n, dtype=np.float32)
    argmax_out = np.full(n, -1, dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_chunk = query_emb[start:end] @ ref_emb.T
        max_out[start:end] = np.max(sim_chunk, axis=1)
        argmax_out[start:end] = np.argmax(sim_chunk, axis=1)
        if k_actual <= 0:
            continue
        if k_actual == n_ref:
            topk = sim_chunk
        else:
            topk = np.partition(sim_chunk, n_ref - k_actual, axis=1)[:, -k_actual:]
        mean_out[start:end] = topk.mean(axis=1)
    return (
        mean_out.astype(np.float32),
        max_out.astype(np.float32),
        argmax_out,
    )


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


def _engagement_features(stories: Sequence[Story]) -> NDArray[np.float32]:
    """log1p HN points and comment count (meta columns; StandardScaled later)."""
    return np.array(
        [
            [np.log1p(max(s.score, 0)), np.log1p(max(s.comment_count or 0, 0))]
            for s in stories
        ],
        dtype=np.float32,
    ).reshape(len(stories), 2)


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


# Explore's interests: KMeans over each user's upvotes, kept per user and
# warm-started from the previous centers when the upvotes change (0.02 s vs
# ~1 s from scratch for 735 upvotes on the VPS, 2026-09-30), so a vote moves
# the interests a little instead of reshuffling them.
_INTEREST_CACHE: LRUCache[int, tuple[str, NDArray[np.float32]]] = LRUCache(maxsize=64)
_INTEREST_CACHE_LOCK = threading.Lock()


def interest_centers(
    user_id: int | None, up_embeddings: NDArray[np.float32], k: int
) -> NDArray[np.float32]:
    """Up to *k* L2-normalized centers of the user's upvote embeddings, one
    per interest; the upvotes themselves when there are no more than *k*."""
    dim = up_embeddings.shape[1] if up_embeddings.ndim == 2 else 0
    if len(up_embeddings) == 0 or k <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    unique = np.unique(up_embeddings.astype(np.float32), axis=0)
    if len(unique) <= k:
        return unique
    signature = hashlib.sha256(unique.tobytes()).hexdigest()
    previous = None
    if user_id is not None:
        with _INTEREST_CACHE_LOCK:
            previous = _INTEREST_CACHE.get(user_id)
    if previous is not None and previous[0] == signature:
        return previous[1]
    if previous is not None and previous[1].shape == (k, dim):
        kmeans = KMeans(n_clusters=k, init=previous[1], n_init=1)
    else:
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=0)
    kmeans.fit(unique)
    centers = kmeans.cluster_centers_.astype(np.float32)
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    if user_id is not None:
        with _INTEREST_CACHE_LOCK:
            _INTEREST_CACHE[user_id] = (signature, centers)
    return centers


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
    means = np.zeros(len(fb_embeddings), dtype=np.float32)
    closest = np.zeros(len(fb_embeddings), dtype=np.float32)
    similarities = fb_embeddings @ class_embs.T
    own_columns = {int(row): column for column, row in enumerate(class_indices)}
    for row, values in enumerate(similarities):
        if row in own_columns:
            values = np.delete(values, own_columns[row])
        if len(values):
            means[row] = _topk_mean(values, min(k, len(values)))
            closest[row] = values.max()
    return means, closest


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
    linear_models: linear_blend.LinearBlendModels | None = None
    linear_dense_candidates: NDArray[np.float32] | None = None
    feedback_stories, feedback_labels, _vote_times = db.get_feedback_for_training(
        user_id=user_id
    )

    if config.model.deduplicate_training_feedback:
        from .feedback import deduplicate_feedback

        feedback_stories, feedback_labels, _vote_times = deduplicate_feedback(
            feedback_stories, feedback_labels, _vote_times
        )

    publication_train: NDArray[np.float32] | None = None
    publication_candidates: NDArray[np.float32] | None = None
    if config.model.publication_affinity_enabled:
        from .publication import publication_features

        publication_train, publication_candidates = publication_features(
            feedback_stories,
            feedback_labels,
            _vote_times,
            candidates,
            strength=config.model.publication_prior_strength,
        )

    n_feedback = len(feedback_labels)
    trace.set_count("feedback_total", n_feedback)
    if score_context is not None:
        score_context.feedback_labels = list(feedback_labels)

    # Multiclass SVM: 0=down, 1=neutral, 2=up
    unique_classes = set(feedback_labels)
    fb_labels_arr = np.array(feedback_labels)
    n_up = int((fb_labels_arr == 2).sum())
    n_down = int((fb_labels_arr == 0).sum())
    n_neutral = int((fb_labels_arr == 1).sum())
    trace.set_count("feedback_up", n_up)
    trace.set_count("feedback_down", n_down)
    trace.set_count("feedback_neutral", n_neutral)

    if (
        n_up >= config.model.min_up_for_svm
        and n_down >= config.model.min_down_for_svm
        and len(unique_classes) >= 2
    ):
        try:
            # Lookup precedes training-feature construction, but follows
            # embedding refresh: enrichment can change training inputs
            # without changing any vote or its timestamp.
            fb_sig = _feedback_signature(db, user_id) if user_id is not None else ""
            cached_model: _CachedModel | None = None

            with trace.stage("feedback_embedding"):
                fb_embeddings = get_or_compute_embeddings(
                    feedback_stories, embedder, db
                )
            if score_context is not None:
                score_context.feedback_embeddings = fb_embeddings

            # The vectors the models train and score on: the stored ones, or
            # stored + side model side by side. The score context (badges,
            # attribution) stays on the stored vectors either way.
            model_cand_emb, model_fb_emb = candidate_embeddings, fb_embeddings
            if config.model.side_embedding_enabled:
                from .side_embeddings import model_space

                with trace.stage("side_embeddings"):
                    space = model_space(
                        candidates,
                        candidate_embeddings,
                        feedback_stories,
                        fb_embeddings,
                        db,
                        config,
                    )
                trace.set_label("side_embeddings", "on" if space else "off")
                if space is not None:
                    model_cand_emb, model_fb_emb = space.candidates, space.feedback

            if fb_sig:
                signature = hashlib.sha256(fb_sig.encode())
                signature.update(model_fb_emb.tobytes())
                signature.update(
                    json.dumps(
                        [
                            embedder.model_version,
                            asdict(config.model),
                            [
                                (
                                    s.id,
                                    label,
                                    len(s.text_content),
                                    s.source,
                                    s.url
                                    if config.model.publication_affinity_enabled
                                    else None,
                                )
                                for s, label in zip(feedback_stories, feedback_labels)
                            ],
                        ],
                        sort_keys=True,
                    ).encode()
                )
                fb_sig = signature.hexdigest()
                cached_model = _get_cached_model(user_id, fb_sig)
                if config.model.linear_blend_enabled and user_id is not None:
                    linear_models = linear_blend.get_cached(
                        (user_id, fb_sig, _MODEL_SCHEMA_VERSION)
                    )
                    if linear_models is None:
                        # Refit the SVM too: the linear models train on its
                        # scaled feature rows, which a cache hit skips.
                        cached_model = None

            # Personalization: mean/closest per class from ALL real feedback
            fb_labels_arr = np.array(feedback_labels)
            up_mask = fb_labels_arr == 2
            down_mask = fb_labels_arr == 0
            neutral_mask = fb_labels_arr == 1
            fb_up_embs = model_fb_emb[up_mask]
            fb_down_embs = model_fb_emb[down_mask]

            n_up = int(up_mask.sum())
            n_down = int(down_mask.sum())
            k = config.model.knn_k
            emb_dim = model_cand_emb.shape[1]

            with trace.stage("svm_candidate_feature_prep"):
                # Fused (top-k mean, max) per class — one dot pass each instead
                # of the two _knn_similarity/_chunked_max_dot recomputed the
                # same candidate @ feedback matrix twice.
                cand_sim_to_up, cand_closest_up, cand_closest_up_idx = (
                    _knn_mean_and_max(model_cand_emb, fb_up_embs, k)
                )
                cand_sim_to_down, cand_closest_down, _ = _knn_mean_and_max(
                    model_cand_emb, fb_down_embs, k
                )
                cand_closest_neutral = _chunked_max_dot(
                    candidate_embeddings, fb_embeddings[neutral_mask]
                )
                # Cluster centers depend only on up-voted feedback, so reuse the
                # cached ones on a model-cache hit instead of rerunning KMeans.
                if cached_model is not None and cached_model[2] is not None:
                    positive_cluster_centers = cached_model[2]
                else:
                    positive_cluster_centers = _positive_cluster_centers(
                        fb_up_embs, config.model.positive_cluster_k
                    )
                cand_positive_cluster_sim = _similarity_to_positive_cluster_centers(
                    model_cand_emb, positive_cluster_centers
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
                    model_cand_emb,
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
            if publication_candidates is not None:
                cand_features = np.concatenate(
                    [cand_features, publication_candidates], axis=1
                )
            if config.model.engagement_features_enabled:
                cand_features = np.concatenate(
                    [cand_features, _engagement_features(candidates)], axis=1
                )
            if score_context is not None:
                ctx_up_embs = fb_up_embs
                ctx_closest_up, ctx_closest_up_idx = (
                    cand_closest_up,
                    cand_closest_up_idx,
                )
                ctx_closest_down = cand_closest_down
                if model_cand_emb is not candidate_embeddings:
                    # Side by side: badges and attribution thresholds are
                    # tuned on the stored vectors, so measure them there.
                    ctx_up_embs = fb_embeddings[up_mask]
                    _, ctx_closest_up, ctx_closest_up_idx = _knn_mean_and_max(
                        candidate_embeddings, ctx_up_embs, k
                    )
                    _, ctx_closest_down, _ = _knn_mean_and_max(
                        candidate_embeddings, fb_embeddings[down_mask], k
                    )
                score_context.cand_closest_up = ctx_closest_up.astype(np.float32)
                score_context.cand_closest_up_idx = ctx_closest_up_idx
                score_context.fb_up_titles = [
                    feedback_stories[i].title for i in np.flatnonzero(up_mask)
                ]
                score_context.fb_up_embeddings = ctx_up_embs
                score_context.cand_closest_down = ctx_closest_down.astype(np.float32)
                score_context.cand_closest_neutral = cand_closest_neutral.astype(
                    np.float32
                )

            if cached_model is not None:
                trace.set_label("model_cache", "hit")
                svm, scaler, _ = cached_model
            else:
                trace.set_label("model_cache", "miss")
                with trace.stage("svm_training_feature_prep"):
                    # LOOCV k-NN for training: exclude self from reference set
                    fb_sim_to_up = np.zeros(len(model_fb_emb), dtype=np.float32)
                    fb_sim_to_down = np.zeros(len(model_fb_emb), dtype=np.float32)
                    if n_up > 0:
                        up_indices = np.where(up_mask)[0]
                        fb_sim_to_up, fb_closest_up = _loocv_knn_features(
                            model_fb_emb, fb_up_embs, up_indices, k
                        )
                    else:
                        fb_closest_up = np.zeros(len(model_fb_emb), dtype=np.float32)

                    if n_down > 0:
                        down_indices = np.where(down_mask)[0]
                        fb_sim_to_down, fb_closest_down = _loocv_knn_features(
                            model_fb_emb, fb_down_embs, down_indices, k
                        )
                    else:
                        fb_closest_down = np.zeros(len(model_fb_emb), dtype=np.float32)

                    fb_positive_cluster_sim = _similarity_to_positive_cluster_centers(
                        model_fb_emb, positive_cluster_centers
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
                        model_fb_emb,
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

                if publication_train is not None:
                    fb_features = np.concatenate(
                        [fb_features, publication_train], axis=1
                    )
                if config.model.engagement_features_enabled:
                    fb_features = np.concatenate(
                        [fb_features, _engagement_features(feedback_stories)],
                        axis=1,
                    )

                # Ensure all three classes (0, 1, 2) are present
                missing = {0, 1, 2} - set(feedback_labels)
                if missing:
                    fb_features = np.concatenate(
                        [
                            fb_features,
                            np.zeros(
                                (len(missing), fb_features.shape[1]),
                                dtype=np.float32,
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
                if config.model.svm_precomputed_enabled:
                    if config.model.svm_kernel != "rbf" or not isinstance(
                        config.model.svm_gamma, float
                    ):
                        raise ValueError("precomputed SVM requires a numeric RBF gamma")
                    svm: _CachedClassifier = PrecomputedRbfSVC(
                        c=config.model.svm_c,
                        gamma=config.model.svm_gamma,
                        chunk_size=config.model.svm_precomputed_chunk_size,
                    )
                else:
                    svm = SVC(
                        C=config.model.svm_c,
                        kernel=config.model.svm_kernel,
                        gamma=config.model.svm_gamma,
                        cache_size=16,
                        random_state=0,
                        decision_function_shape="ovr",
                    )
                with trace.stage("svm_fit"):
                    svm.fit(fb_features_scaled, labels, sample_weight=sample_weights)
                if config.model.linear_blend_enabled:
                    try:
                        with trace.stage("linear_blend_fit"):
                            linear_models = linear_blend.fit_linear_blend(
                                fb_features_scaled,
                                labels,
                                sample_weights,
                                feedback_stories,
                                feedback_labels,
                                dense_c=config.model.linear_blend_dense_c,
                                tfidf_c=config.model.linear_blend_tfidf_c,
                                warm=linear_blend.latest(user_id)
                                if user_id is not None
                                else None,
                            )
                        if fb_sig and user_id is not None:
                            linear_blend.set_cached(
                                (user_id, fb_sig, _MODEL_SCHEMA_VERSION),
                                linear_models,
                                config.max_cached_models,
                            )
                    except Exception as e:
                        trace.set_label("linear_blend_fit", "error")
                        logging.error("Failed to fit linear blend: %r", e)
                if fb_sig:
                    _set_cached_model(
                        user_id,
                        fb_sig,
                        svm,
                        scaler,
                        config.max_cached_models,
                        centers=positive_cluster_centers,
                    )

            with trace.stage("svm_candidate_scale"):
                cand_features_meta_scaled = np.clip(
                    scaler.transform(cand_features[:, emb_dim:]), -2.5, 2.5
                )
                cand_features_scaled = np.hstack(
                    [cand_features[:, :emb_dim], cand_features_meta_scaled]
                )

            linear_dense_candidates = cand_features_scaled
            class_order = list(svm.classes_)
            idx_up = class_order.index(2)
            with trace.stage("decision"):
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
            trace.set_label("svm_fit", "error")
            logging.error("Failed to fit feedback SVM: %r", e)
    else:
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
            # Clamp age at zero: a story newer than `now` (clock skew) would
            # otherwise raise a negative base to a fractional power (complex).
            s.score / max((max((now - s.time) / 3600.0, 0.0) + 2.0) ** 1.8, 0.1)
            for s in candidates
        ],
        dtype=np.float32,
    )
    if tier1_scores.max() > 0:
        tier1_scores = tier1_scores / tier1_scores.max()

    # Three-way blend between tier 1 (gravity), tier 2 (centroid), tier 3 (SVM)
    # α_2 ramps from 0→1 as n_feedback grows; tier 1 fades out smoothly.
    alpha_2 = float(np.clip(n_feedback / config.model.tier2_blend_window, 0.0, 1.0))
    # The SVM tier's share; the linear blend (same votes, same kind of model)
    # ramps in with it.
    t3_weight = 0.0

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

    blend_share = t3_weight if config.model.linear_blend_ramp else 1.0
    if (
        linear_models is not None
        and linear_dense_candidates is not None
        and svm_scores is not None
        and blend_share > 0.0
    ):
        try:
            with trace.stage("linear_blend_score"):
                scores = linear_blend.blend_scores(
                    linear_models,
                    scores,
                    linear_dense_candidates,
                    candidates,
                    dense_weight=config.model.linear_blend_dense_weight * blend_share,
                    tfidf_weight=config.model.linear_blend_tfidf_weight * blend_share,
                )
        except Exception as e:
            trace.set_label("linear_blend_score", "error")
            logging.error("Failed to score linear blend: %r", e)

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
            trace.set_label("svm_probs", "error")
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


def get_entropy(r: RankedStory) -> float:
    """Shannon entropy (bits) of a story's prob_down/neutral/up distribution.

    ``None`` probabilities (no trained model / cold path) contribute nothing,
    so a story with no probabilities at all has entropy 0.
    """
    ent = 0.0
    for p in (r.prob_down, r.prob_neutral, r.prob_up):
        if p is not None and p > 1e-9:
            ent -= p * np.log2(p)
    return ent


def hn_gravity(points: int, posted: int, now: float, time_scale: float = 1.0) -> float:
    """HN front-page gravity, points / (age_hours / time_scale + 2) ** 1.8;
    ``time_scale`` 1 is HN's own and the tier-1 blend of ``_score_and_rank``,
    Popular uses ``GRAVITY_TIME_SCALE[window]`` (age clamped at zero)."""
    age_hours = max((now - posted) / 3600.0, 0.0)
    return points / (age_hours / time_scale + 2.0) ** 1.8


def in_window(posted: int, window: Window, now: float) -> bool:
    """Whether a story posted at *posted* belongs to *window* at *now*
    (age clamped at zero)."""
    age = max(now - posted, 0.0)
    if window == "archive":
        return age > WINDOW_SECONDS["1m"]
    return age <= WINDOW_SECONDS[window]


@dataclass(frozen=True)
class WindowViews:
    """The three views of one time window. Recommended is in model-score
    order and Popular in HN-gravity order. A cached deck's Explore is in
    pick order (Unsure, then Novel, then Interest picks, each best first) so
    ``serve_window`` can cap it per badge; a served Explore is in model-score
    order."""

    recommended: tuple[RankedStory, ...] = ()
    popular: tuple[RankedStory, ...] = ()
    explore: tuple[RankedStory, ...] = ()

    def view(self, name: View) -> tuple[RankedStory, ...]:
        if name == "recommended":
            return self.recommended
        return self.popular if name == "popular" else self.explore

    def stories(self) -> list[RankedStory]:
        """One card per story, in view order, with its badges from every
        view of this window it is in (a Recommended story that is also Hot
        shows 🔥)."""
        merged: dict[int, RankedStory] = {}
        for name in VIEWS:
            for r in self.view(name):
                prev = merged.get(r.story.id)
                merged[r.story.id] = (
                    r
                    if prev is None
                    else replace(
                        prev,
                        is_uncertain=prev.is_uncertain or r.is_uncertain,
                        is_novel=prev.is_novel or r.is_novel,
                        is_discussion_rich=prev.is_discussion_rich
                        or r.is_discussion_rich,
                        is_high_engagement=prev.is_high_engagement
                        or r.is_high_engagement,
                        is_hot=prev.is_hot or r.is_hot,
                        is_interest=prev.is_interest or r.is_interest,
                    )
                )
        return list(merged.values())


ViewMapper: TypeAlias = Callable[
    [Window, View, tuple[RankedStory, ...]], Iterable[RankedStory]
]


@dataclass(frozen=True)
class WindowDeck:
    """A deck: the views of every time window (see ``assemble_window_deck``)."""

    windows: dict[Window, WindowViews] = field(default_factory=dict)

    def window(self, name: Window) -> WindowViews:
        return self.windows.get(name, WindowViews())

    def stories(self) -> list[RankedStory]:
        """Every story once (as in the first window that has it), best model
        score first."""
        seen: dict[int, RankedStory] = {}
        for views in self.windows.values():
            for r in views.stories():
                seen.setdefault(r.story.id, r)
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)

    def is_empty(self) -> bool:
        return not any(
            views.view(name) for views in self.windows.values() for name in VIEWS
        )

    def map_views(self, fn: ViewMapper) -> WindowDeck:
        """A deck with every view replaced by ``fn(window, view, stories)``."""
        return WindowDeck(
            {
                w: WindowViews(
                    tuple(fn(w, "recommended", views.recommended)),
                    tuple(fn(w, "popular", views.popular)),
                    tuple(fn(w, "explore", views.explore)),
                )
                for w, views in self.windows.items()
            }
        )

    def without(self, story_ids: Collection[int]) -> WindowDeck:
        return self.map_views(
            lambda _w, _v, items: (r for r in items if r.story.id not in story_ids)
        )


@dataclass(frozen=True)
class ExploreContext:
    """Per-candidate arrays for the personalized Explore picks (Unsure/Novel/
    Interest), rows looked up by story id through ``row_of``. Absent
    (``None``) on the cold-deck path, which has no feedback to personalize
    against.

    ``cand_interest`` is each candidate's nearest interest (a cluster of the
    user's upvotes, ``interest_centers``); ``interest_sizes`` counts the
    upvotes in each. No interests (empty ``interest_sizes``), no Interest
    picks."""

    cand_max_sim: NDArray[np.float32]
    cand_interest: NDArray[np.int64]
    interest_sizes: NDArray[np.int64]
    row_of: Mapping[int, int]


def assemble_window_deck(
    ranked: Sequence[RankedStory],
    *,
    config: Config,
    now: float,
    explore: ExploreContext | None = None,
    is_feedback_match: Callable[[Story], bool] | None = None,
    trace: RankTrace | _NullTrace = NULL_TRACE,
) -> WindowDeck:
    """Pick every window's three views from a fully scored candidate pool,
    all at one *now*, each ``SELECT_MARGIN`` times its served size
    (``serve_window`` caps them at request time).

    Per window (see ``in_window``):

    - Recommended: the top stories by model score, no source quota.
    - Popular: the top HN stories by ``hn_gravity`` on the window's clock
      (``GRAVITY_TIME_SCALE``). Each card
      gets one badge from its own numbers: 🔥 Hot when its velocity
      (points/hour) is in the pool's top ``hot_badge_percentile`` and it has
      ``HOT_MIN_SCORE`` points, else 💬 Talk when it has at least as many
      comments as points, else 🏆 Top.
    - Explore, only with *explore*: Unsure (highest entropy), Novel
      (farthest from every vote) and Interest (the best story of each of
      the user's interests, least covered by the served Recommended first;
      ``interest_picks``), picked in that order, each excluding earlier
      picks and the window's Recommended stories.
      *is_feedback_match* marks stories that duplicate a voted one;
      Explore skips them and backfills from the rest (``canonicalize_hn_
      dupes`` would drop them downstream anyway; see WORKLOG 2026-07-10).
    """
    by_score = sorted(ranked, key=lambda r: r.score, reverse=True)
    velocities = [
        r.story.score / max((now - r.story.time) / 3600.0, 0.1) for r in by_score
    ]
    hot_threshold = (
        float(np.percentile(velocities, config.model.hot_badge_percentile))
        if velocities
        else 0.0
    )
    velocity_of = {r.story.id: v for r, v in zip(by_score, velocities)}

    def popular_card(r: RankedStory) -> RankedStory:
        story = r.story
        if story.score >= HOT_MIN_SCORE and velocity_of[story.id] >= hot_threshold:
            return replace(r, is_hot=True)
        if (story.comment_count or 0) >= story.score:
            return replace(r, is_discussion_rich=True)
        return replace(r, is_high_engagement=True)

    def take_unmatched(items: list[RankedStory], n: int) -> list[RankedStory]:
        """The first *n* of *items* that don't duplicate a voted story. Walks
        only as far as needed: the title-similarity check is expensive."""
        if is_feedback_match is None:
            return items[:n]
        out: list[RankedStory] = []
        for r in items:
            if len(out) >= n:
                break
            if not is_feedback_match(r.story):
                out.append(r)
        return out

    # Explore passes: sort key, badge, and whether the pass needs model
    # probabilities (Unsure does: no trained SVM, no Unsure).
    explore_passes: list[
        tuple[
            Callable[[RankedStory], float],
            Callable[[RankedStory], RankedStory],
            bool,
        ]
    ] = []
    if explore is not None:
        ctx = explore
        entropy = {r.story.id: get_entropy(r) for r in by_score}
        explore_passes = [
            (
                lambda r: entropy[r.story.id],
                lambda r: replace(r, is_uncertain=True),
                True,
            ),
            (
                lambda r: float(1.0 - ctx.cand_max_sim[ctx.row_of[r.story.id]]),
                lambda r: replace(r, is_novel=True),
                False,
            ),
        ]

    def interest_picks(
        pool: list[RankedStory], shown: list[RankedStory], picked: set[int]
    ) -> list[RankedStory]:
        """Round-robin over the user's interests: each round takes every
        interest's best-scoring story not yet picked (and not a voted
        story's duplicate). Interests the *shown* Recommended stories cover
        least go first, then bigger ones, so the served picks show the
        interests Recommended misses."""
        if explore is None or not len(explore.interest_sizes):
            return []
        ctx = explore
        sizes = ctx.interest_sizes

        def interest_of(r: RankedStory) -> int:
            return int(ctx.cand_interest[ctx.row_of[r.story.id]])

        covered = np.bincount([interest_of(r) for r in shown], minlength=len(sizes))
        order = sorted(range(len(sizes)), key=lambda c: (covered[c], -sizes[c]))
        queues: dict[int, list[RankedStory]] = {c: [] for c in order}
        for r in pool:  # model-score order
            if r.story.id not in picked:
                queues[interest_of(r)].append(r)
        heads = dict.fromkeys(order, 0)
        out: list[RankedStory] = []
        limit = EXPLORE_PER_BADGE * SELECT_MARGIN
        while len(out) < limit:
            took = False
            for c in order:
                queue = queues[c]
                while heads[c] < len(queue):
                    r = queue[heads[c]]
                    heads[c] += 1
                    if is_feedback_match is None or not is_feedback_match(r.story):
                        out.append(replace(r, is_interest=True))
                        took = True
                        break
                if len(out) >= limit:
                    break
            if not took:
                break
        return out

    windows: dict[Window, WindowViews] = {}
    for window in WINDOWS:
        pool = [r for r in by_score if in_window(r.story.time, window, now)]
        trace.set_count(f"window_pool_{window}", len(pool))
        recommended = pool[: VIEW_SIZE * SELECT_MARGIN]
        popular = sorted(
            (r for r in pool if is_hn_source(r.story.source)),
            key=lambda r: hn_gravity(
                r.story.score, r.story.time, now, GRAVITY_TIME_SCALE[window]
            ),
            reverse=True,
        )[: VIEW_SIZE * SELECT_MARGIN]
        picked = {r.story.id for r in recommended}
        explore_view: list[RankedStory] = []
        for key, mark, needs_probs in explore_passes:
            eligible = [
                r
                for r in pool
                if r.story.id not in picked
                and (not needs_probs or r.prob_down is not None)
            ]
            eligible.sort(key=key, reverse=True)
            for r in take_unmatched(eligible, EXPLORE_PER_BADGE * SELECT_MARGIN):
                picked.add(r.story.id)
                explore_view.append(mark(r))
        explore_view.extend(interest_picks(pool, recommended[:VIEW_SIZE], picked))
        windows[window] = WindowViews(
            tuple(recommended),
            tuple(popular_card(r) for r in popular),
            tuple(explore_view),
        )
    return WindowDeck(windows)


def serve_window(views: WindowViews, window: Window, now: float) -> WindowViews:
    """A cached window as served at *now*: stories that aged out of the
    window since the warm are dropped, then each view is capped
    (``VIEW_SIZE``; Explore ``EXPLORE_PER_BADGE`` per badge, in model-score
    order). Short windows stay short; they are never widened."""

    def current(items: tuple[RankedStory, ...]) -> list[RankedStory]:
        return [r for r in items if in_window(r.story.time, window, now)]

    explore: list[RankedStory] = []
    for badge in ("is_uncertain", "is_novel", "is_interest"):
        explore.extend(
            [r for r in current(views.explore) if getattr(r, badge)][:EXPLORE_PER_BADGE]
        )
    explore.sort(key=lambda r: r.score, reverse=True)
    return WindowViews(
        tuple(current(views.recommended)[:VIEW_SIZE]),
        tuple(current(views.popular)[:VIEW_SIZE]),
        tuple(explore),
    )


def rerank_candidates(
    db: Database,
    config: Config,
    embedder: Embedder,
    candidates: list[Story],
    cand_embeddings: NDArray[np.float32] | None = None,
    user_id: int | None = None,
    trace: RankTrace | _NullTrace = NULL_TRACE,
    is_feedback_match: Callable[[Story], bool] | None = None,
) -> WindowDeck:
    """Score candidates and pick every window's views.

    This wraps :func:`_score_and_rank` (tier blend + sort) and
    :func:`assemble_ranked_deck` (window views, badges, attribution).
    *is_feedback_match* is passed on to the Explore picks (see
    :func:`assemble_window_deck`).

    Use this in production; the private ``_score_and_rank`` is intended for
    tier-blend tests that need to assert on ranking without badge side effects.
    """
    if not candidates:
        return WindowDeck()
    trace.set_count("candidates", len(candidates))

    if cand_embeddings is None:
        with trace.stage("candidate_embedding"):
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

    return assemble_ranked_deck(
        ranked,
        candidates,
        cand_embeddings,
        db,
        config,
        embedder,
        user_id=user_id,
        score_context=score_context,
        trace=trace,
        is_feedback_match=is_feedback_match,
    )


# Minimum closest-up similarity for showing an attribution. A bogus
# "because you upvoted X" on a weak match is worse than none.
ATTRIBUTION_MIN_SIM = 0.35
# Badged cards (Hot/Top/Talk/Unsure/Novel) are in the deck for popularity
# or exploration, not for resembling an upvote, and full-text similarity to
# the nearest upvote is 0.6-0.8 for most loosely related stories (median
# 0.77 on profile 151's 1w deck, 2026-09-30). They name one only on a close
# match: 0.85 was picked by eye on that deck, just above the clear misses
# (0.64-0.81); good and bad matches overlap, so it trades some good lines
# for no bad ones. Interest (🎯) comes from the user's upvotes, so it keeps
# the floor.
BADGED_ATTRIBUTION_MIN_SIM = 0.85


def card_attribution(r: RankedStory) -> str:
    """The attribution a card shows, given its badges from every view
    (``WindowViews.stories``)."""
    badged = (
        r.is_hot
        or r.is_high_engagement
        or r.is_discussion_rich
        or r.is_uncertain
        or r.is_novel
    )
    if badged and r.best_match_sim < BADGED_ATTRIBUTION_MIN_SIM:
        return ""
    return r.best_match_title


def _fill_best_match_titles(
    deck: WindowDeck,
    candidates: list[Story],
    score_context: RankScoreContext | None,
) -> WindowDeck:
    """F2 attribution: name the closest upvoted story per deck card.

    Uses the argmax indices already computed for features (no new matmul).
    Empty when cold (no feedback), when the argmax is invalid, when the
    feedback title is gone, or below ATTRIBUTION_MIN_SIM.
    """
    if (
        score_context is None
        or score_context.cand_closest_up_idx is None
        or score_context.cand_closest_up is None
        or not score_context.fb_up_titles
    ):
        return deck
    row_of = {s.id: i for i, s in enumerate(candidates)}
    titles = score_context.fb_up_titles
    closest_up_idx = score_context.cand_closest_up_idx
    closest_up = score_context.cand_closest_up

    def match_for(story_id: int) -> tuple[str, float]:
        row = row_of.get(story_id)
        if row is None:
            return "", 0.0
        fb_row = int(closest_up_idx[row])
        sim = float(closest_up[row])
        if 0 <= fb_row < len(titles) and sim >= ATTRIBUTION_MIN_SIM and titles[fb_row]:
            return titles[fb_row], sim
        return "", 0.0

    def fill(r: RankedStory) -> RankedStory:
        title, sim = match_for(r.story.id)
        return replace(r, best_match_title=title, best_match_sim=sim) if title else r

    return deck.map_views(lambda _w, _v, items: (fill(r) for r in items))


def assemble_ranked_deck(
    ranked: list[RankedStory],
    candidates: list[Story],
    cand_embeddings: NDArray[np.float32],
    db: Database,
    config: Config,
    embedder: Embedder,
    *,
    user_id: int | None = None,
    score_context: RankScoreContext | None = None,
    trace: RankTrace | _NullTrace = NULL_TRACE,
    is_feedback_match: Callable[[Story], bool] | None = None,
) -> WindowDeck:
    """Pick the window views (with Explore) from an existing ranking."""
    if not candidates:
        return WindowDeck()
    if score_context is None:
        score_context = RankScoreContext()

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
            score_context.fb_up_embeddings = fb_embs[up_mask]
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

    with trace.stage("interest_clusters"):
        up_embs = score_context.fb_up_embeddings
        if up_embs is None:
            up_embs = np.zeros((0, cand_embeddings.shape[1]), dtype=np.float32)
        centers = interest_centers(user_id, up_embs, config.model.interest_cluster_k)
        if len(centers):
            cand_interest = np.argmax(cand_embeddings @ centers.T, axis=1)
            interest_sizes = np.bincount(
                np.argmax(up_embs @ centers.T, axis=1), minlength=len(centers)
            )
        else:
            cand_interest = np.zeros(len(candidates), dtype=np.int64)
            interest_sizes = np.zeros(0, dtype=np.int64)

    with trace.stage("window_assembly"):
        deck = assemble_window_deck(
            ranked,
            config=config,
            now=time.time(),
            explore=ExploreContext(
                cand_max_sim=cand_max_sim,
                cand_interest=cand_interest.astype(np.int64),
                interest_sizes=interest_sizes.astype(np.int64),
                row_of={s.id: idx for idx, s in enumerate(candidates)},
            ),
            is_feedback_match=is_feedback_match,
            trace=trace,
        )
    return _fill_best_match_titles(deck, candidates, score_context)
