from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal


@dataclass(frozen=True)
class ModelConfig:
    svm_c: float = 0.2
    svm_gamma: float | str = 0.03
    svm_kernel: str = "rbf"
    svm_precomputed_enabled: bool = False
    svm_precomputed_chunk_size: int = 512
    neutral_weight: float = 0.0
    enable_mmr: bool = False
    diversity_threshold: float = 0.75
    knn_k: int = 10
    positive_cluster_k: int = 4
    tier2_blend_window: int = 50
    tier3_blend_window: int = 60
    min_up_for_svm: int = 20
    min_down_for_svm: int = 20
    hot_badge_percentile: float = 99.5
    dedup_render_enabled: bool = True
    dedup_embedding_cosine_enabled: bool = True
    dedup_embedding_cosine_threshold: float = 0.87
    dedup_exclude_actions: tuple[str, ...] = ("up", "neutral")


# Shared across all worktrees of this repo (sibling to the `main` checkout)
# so the 87MB model and secrets never need copying/symlinking per worktree.
DEFAULT_ONNX_MODEL_DIR = "/home/dev/hn-rewrite/shared/onnx_model"
DEFAULT_EMBEDDING_MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|512"
DEFAULT_EMBEDDING_MAX_TOKENS = 512
DEFAULT_ENV_PATH = "/home/dev/hn-rewrite/shared/.env"

BQ_ARCHIVE_SOURCE = "bq_seed"
CH_ARCHIVE_SOURCE = "ch_seed"


def is_hn_source(source: str) -> bool:
    return source in {"hn", BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE}


BQ_ARCHIVE_CANDIDATE_LIMIT = 2000
CH_ARCHIVE_CANDIDATE_LIMIT = 2000
LIVE_WINDOW_LIMIT = 5000


@dataclass(frozen=True)
class RssConfig:
    enabled: bool = True
    per_feed_limit: int = 70
    feeds: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    db_path: str = "hn_rewrite.db"
    days: int = 30
    count: int = 40
    onnx_model_dir: str = DEFAULT_ONNX_MODEL_DIR
    embedding_model_version: str = DEFAULT_EMBEDDING_MODEL_VERSION
    embedding_max_tokens: int = DEFAULT_EMBEDDING_MAX_TOKENS
    embedding_batch_size: int = 32
    embedding_ort_variant: Literal[
        "current",
        "spin_off",
        "spin_off_graph_all",
        "spin_off_auto_threads",
    ] = "current"
    server_port: int = 8765
    regen_interval_seconds: int = 14400
    regen_initial_delay_seconds: int = 30
    regen_prewarm_top_n: int = 50
    prewarm_hn_full: bool = True
    prewarm_reddit_full: bool = True
    prewarm_lesswrong_full: bool = True
    # Number of top-hot stories per subreddit considered for Reddit
    # comment hydration each cycle. The hard per-cycle cap below keeps
    # the total runtime bounded even with 41 subreddit feeds.
    reddit_prewarm_top_per_sub: int = 10
    reddit_prewarm_max_per_cycle: int = 80
    # Minimum spacing (seconds) between any two Reddit fetches scheduled
    # via `reddit_fetch_queue.enqueue_all_reddit_fetches`. Used for the
    # prewarm phase; the topfeed phase uses a fixed 50s stride (one
    # subreddit per fetch at the limiter's natural 2s+jitter cadence,
    # but the queue spreads them out at 50s for 429 backoff headroom).
    reddit_min_fetch_spacing_seconds: float = 30.0
    article_fetch_max_per_run: int = 50
    article_fetch_concurrency: int = 10
    article_fetch_max_age_days: int = 30
    max_cached_models: int = 20
    # Two-leg candidate cap: the HN recent query uses tier-1 gravity
    # (score/age^1.8) so top-scoring stories are fetched first; the RSS
    # recent query uses pure recency because RSS sources have no
    # engagement score in the DB. Total fetched rows = hn_limit +
    # rss_limit + archive-leg-size, which is what flows into candidate
    # embedding, SVM feature prep, and decision_function. The
    # is_uncertain discovery pass is allowed to shift because that
    # signal is orthogonal to the SQL ordering.
    recent_candidate_hn_limit: int = 5000
    recent_candidate_rss_limit: int = 500
    tldr_prefetch_per_combo: int = 5
    # After the top-per-combo pass, regenerate up to this many additional
    # cold-deck stories whose cached TLDR's cache_key no longer matches
    # current story content (e.g. article_body was enriched after the TLDR
    # was generated). 0 disables. See server.py::_prefetch_tldrs_for_ranked.
    tldr_prefetch_stale_per_run: int = 3
    # On-demand HN comment refresh (tldr-detail): forces a real-time Algolia
    # re-fetch for recent, high-velocity threads even when top_comments is
    # already populated from prewarm, since CH prewarm has 1-24h latency on
    # brand-new comments. Any knob set to 0 disables the corresponding gate.
    tldr_refresh_recent_hours: float = 72.0
    tldr_refresh_min_comments: int = 30
    tldr_refresh_min_comments_per_hour: float = 8.0
    # Public demo abuse limits. Cached TLDR hits bypass the uncached TLDR
    # quota; these limits protect only new enrichment/LLM work and vote writes.
    tldr_uncached_per_user_limit: int = 12
    tldr_uncached_per_user_window_seconds: int = 3600
    tldr_uncached_global_limit: int = 60
    tldr_uncached_global_window_seconds: int = 3600
    feedback_per_user_limit: int = 120
    feedback_per_user_window_seconds: int = 600
    feedback_global_limit: int = 2000
    feedback_global_window_seconds: int = 3600
    dashboard_warm_vote_threshold: int = 10
    dashboard_warm_idle_seconds: float = 3.0
    session_create_per_ip_limit: int = 60
    session_create_per_ip_window_seconds: int = 3600
    profile_link_per_ip_limit: int = 120
    profile_link_per_ip_window_seconds: int = 3600
    model: ModelConfig = field(default_factory=ModelConfig)
    rss: RssConfig = field(default_factory=RssConfig)

    def __post_init__(self) -> None:
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if not self.embedding_model_version.strip():
            raise ValueError("embedding_model_version must not be empty")
        if self.embedding_max_tokens <= 0:
            raise ValueError("embedding_max_tokens must be positive")
        if self.dashboard_warm_vote_threshold <= 0:
            raise ValueError("dashboard_warm_vote_threshold must be positive")
        if self.dashboard_warm_idle_seconds <= 0:
            raise ValueError("dashboard_warm_idle_seconds must be positive")
        if self.model.svm_precomputed_chunk_size <= 0:
            raise ValueError("svm_precomputed_chunk_size must be positive")
        if self.embedding_ort_variant not in {
            "current",
            "spin_off",
            "spin_off_graph_all",
            "spin_off_auto_threads",
        }:
            raise ValueError(
                "embedding_ort_variant must be one of: current, spin_off, "
                "spin_off_graph_all, spin_off_auto_threads"
            )

    @classmethod
    def load(cls, path: str = "config.toml") -> Config:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            return cls()

        unknown_sections = set(data) - {"hn_rewrite"}
        if unknown_sections:
            raise ValueError(
                "Unknown config section(s): " + ", ".join(sorted(unknown_sections))
            )

        main_cfg = data.get("hn_rewrite", {})
        if not isinstance(main_cfg, dict):
            raise ValueError("[hn_rewrite] must be a table")

        defaults = cls()
        root_values = dict(main_cfg)
        model_cfg = root_values.pop("model", {})
        rss_cfg = root_values.pop("rss", {})

        root_field_names = {f.name for f in fields(cls)} - {"model", "rss"}
        unknown_root = set(root_values) - root_field_names
        if unknown_root:
            raise ValueError(
                "Unknown hn_rewrite config key(s): " + ", ".join(sorted(unknown_root))
            )

        model = _overlay_dataclass_config(
            defaults.model,
            model_cfg,
            section="hn_rewrite.model",
            tuple_fields={"dedup_exclude_actions"},
        )
        rss = _overlay_dataclass_config(
            defaults.rss,
            rss_cfg,
            section="hn_rewrite.rss",
            tuple_fields={"feeds"},
        )

        return replace(defaults, **root_values, model=model, rss=rss)


def _overlay_dataclass_config(
    defaults: Any,
    config: Any,
    *,
    section: str,
    tuple_fields: set[str] | None = None,
) -> Any:
    if not isinstance(config, dict):
        raise ValueError(f"[{section}] must be a table")

    field_names = {f.name for f in fields(defaults)}
    unknown = set(config) - field_names
    if unknown:
        raise ValueError(
            f"Unknown {section} config key(s): " + ", ".join(sorted(unknown))
        )

    values = dict(config)
    for name in tuple_fields or set():
        if name in values:
            values[name] = tuple(values[name])
    return replace(defaults, **values)
