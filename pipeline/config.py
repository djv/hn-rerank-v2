from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass(frozen=True)
class ModelConfig:
    svm_c: float = 0.2
    svm_gamma: float | str = 0.03
    svm_kernel: str = "rbf"
    svm_precomputed_enabled: bool = False
    svm_precomputed_chunk_size: int = 512
    deduplicate_training_feedback: bool = False
    publication_affinity_enabled: bool = False
    publication_prior_strength: float = 10.0
    # Opt-in (evaluation, 2026-09-25): log1p HN points and comment count as
    # two extra SVM meta columns.
    engagement_features_enabled: bool = False
    neutral_weight: float = 0.0
    enable_mmr: bool = False
    # Opt-in (evaluation, 2026-09-29): rank-blend the final score with a dense
    # logistic regression and a TF-IDF logistic regression (linear_blend.py).
    linear_blend_enabled: bool = False
    linear_blend_dense_weight: float = 0.2
    linear_blend_tfidf_weight: float = 0.3
    linear_blend_dense_c: float = 0.1
    linear_blend_tfidf_c: float = 4.0
    # Scale the blend weights by the SVM's tier-3 weight, so the linear models
    # ramp in with the SVM instead of taking half the ranking at the 20 up /
    # 20 down gate. Off only in the eval, to read the dense model alone.
    linear_blend_ramp: bool = True
    # Opt-in (evaluation, 2026-10-02): the SVM and linear models see each
    # story's stored vector and a second model's vector (Config.side_*)
    # side by side; attribution, Explore and dedup keep the stored vectors.
    # Off when fewer than side_embedding_min_coverage of the candidates and
    # feedback stories have a side vector; the rest get a zero side part.
    side_embedding_enabled: bool = False
    side_embedding_min_coverage: float = 0.98
    # "joined_logistic" (evaluation, 2026-10-07; pipeline/joined_classifier.py)
    # replaces the RBF SVM with one logistic regression over the same rows
    # plus TF-IDF words; it ignores svm_* and needs linear_blend_enabled off.
    # Live only as an interleaving challenger (Config.interleave_user_ids).
    classifier: Literal["svm", "joined_logistic"] = "svm"
    joined_features: Literal["all", "no_metadata"] = "all"
    joined_c: float = 4.0
    joined_embedding_weight: float = 16.0
    joined_numeric_scale: float = 0.15811388300841897  # sqrt(0.1 / 4)
    joined_word_scale: float = 1.0
    diversity_threshold: float = 0.75
    knn_k: int = 10
    positive_cluster_k: int = 4
    # Explore's Interest picks: clusters of the user's upvotes (k=10 on
    # profile 151 split AI into five and kept nomad life, health, investing,
    # gadgets and maps/transit apart, 2026-09-30).
    interest_cluster_k: int = 10
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
# so the ~90MB model and secrets never need copying/symlinking per worktree.
# Kept in sync with config.toml's [hn_rewrite] onnx_model_dir/
# embedding_model_version/embedding_max_tokens -- these are the fallback
# used only when config.toml doesn't override them (it currently does not).
DEFAULT_ONNX_MODEL_DIR = os.environ.get(
    "HN_ONNX_MODEL_DIR", "/home/dev/hn-rewrite/shared/mxbai-embed-xsmall-v1"
)
DEFAULT_EMBEDDING_MODEL_VERSION = "mxbai-embed-xsmall-v1|mean|norm|4096"
DEFAULT_EMBEDDING_MAX_TOKENS = 4096
DEFAULT_ENV_PATH = "/home/dev/hn-rewrite/shared/.env"
DEFAULT_SIDE_EMBEDDING_MODEL_DIR = os.environ.get(
    "HN_SIDE_MODEL_DIR", "/home/dev/hn-rewrite/shared/embeddinggemma-300m-onnx"
)
DEFAULT_SIDE_EMBEDDING_MODEL_VERSION = "embeddinggemma-300m|sentence|128|classification"

BQ_ARCHIVE_SOURCE = "bq_seed"
CH_ARCHIVE_SOURCE = "ch_seed"


def is_hn_source(source: str) -> bool:
    return source in {"hn", BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE}


BQ_ARCHIVE_CANDIDATE_LIMIT = 2000
CH_ARCHIVE_CANDIDATE_LIMIT = 2000
LIVE_WINDOW_LIMIT = 10_000


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
    # Second embedding model for ModelConfig.side_embedding_enabled, encoded
    # by scripts/embed_side_vectors.py into the side_embeddings table.
    side_embedding_model_dir: str = DEFAULT_SIDE_EMBEDDING_MODEL_DIR
    side_embedding_model_version: str = DEFAULT_SIDE_EMBEDDING_MODEL_VERSION
    side_embedding_max_tokens: int = 128
    side_embedding_dim: int = 768
    embedding_ort_variant: Literal[
        "current",
        "spin_off",
        "spin_off_graph_all",
        "spin_off_auto_threads",
    ] = "current"
    server_port: int = 8765
    regen_interval_seconds: int = 3600
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
    # Start-to-start spacing of full subreddit refreshes. Each regen asks
    # for one (votes trigger regens every ~20 min), but weekly-top feeds
    # barely change; a request inside the window waits for it to end.
    # 0 disables the throttle.
    reddit_refresh_min_interval_seconds: float = 7200.0
    article_fetch_max_per_run: int = 50
    # Regen-time article fetches for new RSS snippet stories regardless of
    # rank (the warm path only reaches stories already near the top). 0 off.
    rss_article_prewarm_max_per_run: int = 30
    # AINews issues split into one story per topic (pipeline/ainews.py);
    # the generic RSS path then skips the feed's whole-issue entries.
    ainews_enabled: bool = True
    ainews_feed_url: str = "https://www.latent.space/feed"
    ainews_max_tweets_per_run: int = 400
    article_fetch_concurrency: int = 10
    article_fetch_max_age_days: int = 30
    max_cached_models: int = 20
    # Team-draft interleaving (ROADMAP B2, pipeline/interleave.py): these
    # users' Recommended views mix production with each challenger arm,
    # and every served deck's arm per story is stored (interleave_decks).
    # Empty: off. Each arm adds a full model fit to the user's warm.
    interleave_user_ids: tuple[int, ...] = ()
    interleave_arms: tuple[Literal["joined_all", "joined_no_metadata"], ...] = (
        "joined_all",
        "joined_no_metadata",
    )
    # Two-leg candidate cap: the HN recent query uses tier-1 gravity
    # (score/age^1.8) so top-scoring stories are fetched first; the RSS
    # recent query uses pure recency because RSS sources have no
    # engagement score in the DB. Total fetched rows = hn_limit +
    # rss_limit + archive-leg-size, which is what flows into candidate
    # embedding, SVM feature prep, and decision_function. The
    # is_uncertain discovery pass is allowed to shift because that
    # signal is orthogonal to the SQL ordering.
    #
    # rss_limit matches hn_limit (was 500, ~4,200 in-window rows) so the
    # `ORDER BY time DESC` leg actually reaches the full `days` window
    # instead of truncating to the newest ~4 days of it (see WORKLOG
    # 2026-08-30: pool_rss_oldest_age_h measured ~93h against a 30-day
    # window). There's no non-HN engagement signal to sort by instead —
    # score is 0 for nearly all non-HN rows — so once the limit clears
    # the in-window row count, the ordering stops mattering and the
    # ranker's own scoring picks the winners from the full window.
    recent_candidate_hn_limit: int = 10_000
    recent_candidate_rss_limit: int = 5000
    non_hn_candidates_enabled: bool = True
    # Independent of source regeneration and per-user warm frequency.
    tldr_prefetch_interval_seconds: int = 14400
    # Summaries prefetched per view (Recommended/Popular/Explore) of the
    # default window, 1w, after each warm and regen.
    tldr_prefetch_per_view: int = 2
    # After the per-view pass, regenerate up to this many additional
    # deck stories whose cached TLDR's cache_key no longer matches
    # current story content (e.g. article_body was enriched after the TLDR
    # was generated). 0 disables. See server.py::_prefetch_tldrs_for_ranked.
    # Kept small: bulk prefetch trips Groq free-tier bans (875s retry-after
    # observed 2026-09-07), so the steady-state budget is ~10 stories/run.
    tldr_prefetch_stale_per_run: int = 1
    # Seconds between background TLDR prefetch LLM starts (capped at 15s
    # total offset). Gemini free allows ~10-15 RPM, so a Gemini deployment
    # wants ~5.0; Groq free tolerates 1.0.
    tldr_prefetch_stagger_seconds: float = 1.0
    # On-demand HN comment refresh (tldr-detail): forces a real-time Algolia
    # re-fetch for recent, high-velocity threads even when top_comments is
    # already populated from prewarm, since CH prewarm has 1-24h latency on
    # brand-new comments. Any knob set to 0 disables the corresponding gate.
    tldr_refresh_recent_hours: float = 72.0
    tldr_refresh_min_comments: int = 30
    tldr_refresh_min_comments_per_hour: float = 8.0
    # Regen live-count probe (Firebase descendants): for young HN threads
    # with a cached TLDR whose DB-visible growth is sub-threshold, confirm
    # the live count before spending a full comment hydration. Hydration
    # (Algolia, real-time) fires only on confirmed growth >= the
    # _needs_hn_prewarm threshold, so LLM regen happens strictly on known
    # new content. 0 disables probing.
    tldr_probe_max_threads_per_regen: int = 20
    tldr_probe_timeout_seconds: float = 10.0
    # Tap-time live-count probe (Firebase descendants): on TLDR open of a
    # young HN thread that would otherwise serve cached, confirm the live
    # count before trusting the cache. Ungated by velocity by design — the
    # user is already looking at this story. Miss/failure serves cached.
    # Kept small so cached taps stay snappy; 0 disables tap probing.
    tldr_tap_probe_timeout_seconds: float = 3.0
    # Between hourly regens, refresh live points and comment counts
    # (Firebase, no hydration or LLM) of the busiest young HN threads that
    # pass the tldr_refresh_* gate. 0 disables either knob.
    hot_refresh_interval_seconds: int = 600
    hot_refresh_max_stories: int = 30
    # Public demo abuse limits. Cached TLDR hits bypass the uncached TLDR
    # quota; these limits protect only new enrichment/LLM work and vote writes.
    tldr_uncached_per_user_limit: int = 24
    tldr_uncached_per_user_window_seconds: int = 3600
    tldr_uncached_global_limit: int = 120
    tldr_uncached_global_window_seconds: int = 3600
    # Uncached TLDR generations running at once, all users; extra requests
    # get a stale cached TLDR or a short 429 instead of queueing a thread.
    tldr_max_concurrent_generations: int = 8
    feedback_per_user_limit: int = 120
    feedback_per_user_window_seconds: int = 600
    feedback_global_limit: int = 2000
    feedback_global_window_seconds: int = 3600
    dashboard_warm_vote_threshold: int = 10
    dashboard_warm_idle_seconds: float = 3.0
    # Personalized re-ranks run on a bounded pool (warm_scheduler.py): votes,
    # stale page loads and post-regen refreshes all share it, so a regen that
    # marks every cached user stale can't start one rank thread per user.
    warm_pool_size: int = 2
    session_create_per_ip_limit: int = 60
    session_create_per_ip_window_seconds: int = 3600
    profile_link_per_ip_limit: int = 120
    profile_link_per_ip_window_seconds: int = 3600
    model: ModelConfig = field(default_factory=ModelConfig)
    rss: RssConfig = field(default_factory=RssConfig)

    def __post_init__(self) -> None:
        if self.tldr_prefetch_interval_seconds <= 0:
            raise ValueError("tldr_prefetch_interval_seconds must be positive")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if not self.embedding_model_version.strip():
            raise ValueError("embedding_model_version must not be empty")
        if self.embedding_max_tokens <= 0:
            raise ValueError("embedding_max_tokens must be positive")
        if not self.side_embedding_model_version.strip():
            raise ValueError("side_embedding_model_version must not be empty")
        if self.side_embedding_max_tokens <= 0:
            raise ValueError("side_embedding_max_tokens must be positive")
        if self.side_embedding_dim <= 0:
            raise ValueError("side_embedding_dim must be positive")
        if self.dashboard_warm_vote_threshold <= 0:
            raise ValueError("dashboard_warm_vote_threshold must be positive")
        if self.dashboard_warm_idle_seconds <= 0:
            raise ValueError("dashboard_warm_idle_seconds must be positive")
        if self.warm_pool_size < 1:
            raise ValueError("warm_pool_size must be >= 1")
        if self.model.svm_precomputed_chunk_size <= 0:
            raise ValueError("svm_precomputed_chunk_size must be positive")
        if self.model.classifier == "joined_logistic" and (
            self.model.linear_blend_enabled
        ):
            raise ValueError("joined_logistic replaces the linear blend; disable it")
        if not all(
            v > 0
            for v in (
                self.model.joined_c,
                self.model.joined_embedding_weight,
                self.model.joined_numeric_scale,
                self.model.joined_word_scale,
            )
        ):
            raise ValueError("joined_* settings must be positive")
        if len(set(self.interleave_arms)) != len(self.interleave_arms):
            raise ValueError("interleave_arms must not repeat an arm")
        if self.interleave_user_ids and not self.interleave_arms:
            raise ValueError("interleave_user_ids needs at least one arm")
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
        root_values = _typed_values(cls, root_values, section="hn_rewrite")

        model = _overlay_dataclass_config(
            defaults.model,
            model_cfg,
            section="hn_rewrite.model",
        )
        rss = _overlay_dataclass_config(
            defaults.rss,
            rss_cfg,
            section="hn_rewrite.rss",
        )

        return replace(defaults, **root_values, model=model, rss=rss)


_DC = TypeVar("_DC", bound="DataclassInstance")


def _coerce_toml_value(value: object, tp: object, where: str) -> object:
    """Check a TOML value against a dataclass field's declared type.

    TOML already distinguishes ints, floats, bools and strings, so this only
    widens int -> float and array -> tuple; anything else that doesn't match
    is a config error reported with its key, not a crash later at use.
    """
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    elif get_origin(tp) is tuple:
        (item_tp, _) = get_args(tp)
        if isinstance(value, (list, tuple)):
            return tuple(_coerce_toml_value(v, item_tp, where) for v in value)
    elif get_origin(tp) is Literal:
        if value in get_args(tp):
            return value
    elif isinstance(tp, UnionType):
        for option in get_args(tp):
            try:
                return _coerce_toml_value(value, option, where)
            except ValueError:
                continue
    raise ValueError(f"{where} must be {_type_name(tp)}, got {value!r}")


def _type_name(tp: object) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _typed_values(
    cls: type[DataclassInstance], values: dict[str, object], *, section: str
) -> dict[str, object]:
    hints = get_type_hints(cls)
    return {
        name: _coerce_toml_value(value, hints[name], f"{section}.{name}")
        for name, value in values.items()
    }


def _overlay_dataclass_config(
    defaults: _DC,
    config: object,
    *,
    section: str,
) -> _DC:
    if not isinstance(config, dict):
        raise ValueError(f"[{section}] must be a table")

    field_names = {f.name for f in fields(defaults)}
    unknown = set(config) - field_names
    if unknown:
        raise ValueError(
            f"Unknown {section} config key(s): " + ", ".join(sorted(unknown))
        )

    values = {str(key): value for key, value in config.items()}
    return replace(defaults, **_typed_values(type(defaults), values, section=section))
