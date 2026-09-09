from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray

from database import Database, Story, coerce_int

# ruff: noqa: F401 — re-exports for the public pipeline namespace.
from .config import (
    BQ_ARCHIVE_CANDIDATE_LIMIT,
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_CANDIDATE_LIMIT,
    CH_ARCHIVE_SOURCE,
    Config,
    DEFAULT_ENV_PATH,
    DEFAULT_ONNX_MODEL_DIR,
    LIVE_WINDOW_LIMIT,
    ModelConfig,
    RssConfig,
    _overlay_dataclass_config,
    is_hn_source,
)

from .ranking import (
    COMMENT_DEPTH_PENALTY,
    DASHBOARD_QUEUE_SIZE,
    DISCOVERY_PER_BADGE,
    Embedder,
    HOT_MIN_SCORE,
    PRIMARY_PER_COMBO,
    RankScoreContext,
    RankTrace,
    RankedComment,
    RankedStory,
    SOURCE_CATEGORIES,
    TOP_COMMENT_CORE_THREADS,
    TOP_COMMENT_LIMIT,
    TOP_COMMENT_MAX_PER_THREAD,
    TOP_COMMENT_REPLIES_PER_CORE_THREAD,
    TOP_COMMENT_TOP_LEVEL_BUDGET,
    NULL_TRACE,
    GOOD_TOPLEVEL_MIN_LEN,
    GOOD_TOPLEVEL_MIN_REPLIES,
    _LOG_TEXTLEN_SCALE,
    _MODEL_CACHE,
    _MODEL_CACHE_LOCK,
    _MODEL_CACHE_STORAGE_MAXSIZE,
    _MODEL_SCHEMA_VERSION,
    _NullTrace,
    _SIM_CHUNK_SIZE,
    _assemble_combo_deck,
    _chunked_max_dot,
    _comment_rank_key,
    _extract_comments_recursive,
    _feedback_signature,
    _get_cached_model,
    _knn_similarity,
    _loocv_knn_features,
    _minmax01,
    _positive_cluster_centers,
    _positive_cluster_similarity,
    _score_and_rank,
    _select_top_comments,
    _set_cached_model,
    _similarity_to_positive_cluster_centers,
    _softmax_rows,
    _svm_personalization_features,
    _topk_mean,
    clean_text,
    compose_story_text,
    get_or_compute_embeddings,
    mmr_filter,
    rerank_candidates,
    source_category_onehot,
    source_category_stack,
    story_embedding_text,
)

from .enrichment import (
    REDDIT_RSS_USER_AGENT,
    _article_failure_retry_time,
    _ch_story_item_to_story,
    _fetch_and_parse_feed,
    _is_fetchable_article_url,
    _merge_source_context,
    _reddit_subreddit_from_feed_url,
    _rss_source_name,
    _urllib_fetch,
    build_reddit_prewarm_factories,
    build_reddit_topfeed_factories,
    fetch_and_cache_article_bodies,
    fetch_rss_feeds,
    fetch_story,
    prewarm_lesswrong_stories,
    prewarm_reddit_top_stories,
    prewarm_top_stories,
    select_article_fetch_candidates,
)
from .hn_dupes import _load_feedback_context, _matches_feedback, canonicalize_hn_dupes
from .candidate_cache import (
    CandidatePool,
    get_candidate_pool,
    invalidate_candidate_pool,
)

from .render import (
    generate_dashboard_bytes,
    source_label_filter,
)
from reddit_feed_cache import cache as reddit_feed_cache
from reddit_limiter import limiter as reddit_limiter


COLD_DECK_LIMIT = 100


@dataclass(frozen=True)
class RedditRefreshResult:
    feeds: int
    changed_stories: int
    hydrated_stories: int
    prewarm_candidates: int

    @property
    def changed(self) -> bool:
        return self.changed_stories > 0 or self.hydrated_stories > 0


def _combo_keys_for_story(story: Story, recent_cutoff: int) -> str:
    age = "recent" if story.time >= recent_cutoff else "archive"
    source = "hn" if is_hn_source(story.source) else "non-hn"
    return f"{age}_{source} {age}_mixed"


def cold_ranked_candidates(candidates: list[Story], now_ts: int) -> list[RankedStory]:
    """Production zero-feedback scores and browser eligibility before selection."""
    recent_cutoff = now_ts - 30 * 86400
    return [
        RankedStory(
            story=story,
            score=story.score / ((now_ts - story.time) / 3600.0 + 2.0) ** 1.8,
            best_match_title="",
            is_non_hn=(not is_hn_source(story.source)),
            is_recent=(story.time >= recent_cutoff),
            combo_keys=_combo_keys_for_story(story, recent_cutoff),
        )
        for story in candidates
    ]


def build_cold_deck(
    db: Database,
    config: Config,
    user_id: int | None = None,
    embedder: Embedder | None = None,
    trace: RankTrace | _NullTrace = NULL_TRACE,
    *,
    candidates: list[Story] | None = None,
) -> list[RankedStory]:
    """Build a gravity-sorted, badge-annotated fallback deck — no embeddings,
    no personalization.

    Uses the same tier-1 gravity formula as ``_score_and_rank`` so a
    zero-vote user sees the same ranking as the cold deck.  See
    ``fast_rerank_for_user`` for the 0-vote short-circuit.

    Reuses the same production candidate legs as the
    personalized dashboard via ``load_production_candidate_stories``, and
    the same non-personalized Popular badge assembly (Hot/Top/Talk) as
    ``rerank_candidates`` via ``_assemble_combo_deck``, so cold-start decks
    have Popular and Archive combos populated instead of being flat
    recent-only cards. Explore (Unsure/Novel/Similar) is intentionally
    skipped — it's personalized and requires feedback to compute against.

    When *user_id* is provided, already-voted stories are excluded.

    When *embedder* is provided, the candidate pool is served from the
    process-wide cache (``pipeline.candidate_cache``) instead of a fresh
    SQL load — this is the production path (server.py passes the runtime
    embedder), and avoids re-paying the ~2-8s candidate SQL/embedding cost
    on every cold-deck build. Without an embedder (e.g. offline tests),
    falls back to a direct, uncached load; cold-deck ranking never uses
    embeddings so behavior is identical either way.
    """
    now_ts = int(time.time())
    if candidates is None:
        if embedder is not None:
            pool = get_candidate_pool(db, config, embedder, trace=trace)
            if user_id is not None:
                voted_ids = frozenset(_voted_story_ids(db, user_id))
                candidates, _ = pool.without_feedback(voted_ids)
            else:
                candidates = list(pool.stories)
        else:
            candidates = load_production_candidate_stories(
                db,
                config,
                user_id=user_id,
                exclude_feedback=user_id is not None,
                now_ts=now_ts,
                trace=trace,
            )
    if not candidates:
        return []

    recent_cutoff = now_ts - (30 * 86400)
    ranked = cold_ranked_candidates(candidates, now_ts)

    cand_scores = np.array([story.score for story in candidates])
    cand_velocities = np.array(
        [story.score / max((now_ts - story.time) / 3600.0, 0.1) for story in candidates]
    )
    story_id_to_idx = {story.id: idx for idx, story in enumerate(candidates)}

    cold = _assemble_combo_deck(
        ranked,
        config=config,
        recent_cutoff=recent_cutoff,
        cand_scores=cand_scores,
        cand_velocities=cand_velocities,
        idx_for=story_id_to_idx.__getitem__,
        embeddings_map=None,
        explore=None,
        trace=trace,
    )
    return cold[:COLD_DECK_LIMIT]


def load_production_candidate_stories(
    db: Database,
    config: Config,
    *,
    user_id: int | None,
    exclude_feedback: bool,
    now_ts: int | None = None,
    trace: RankTrace | _NullTrace = NULL_TRACE,
) -> list[Story]:
    """Load the same candidate legs used by the personalized dashboard.

    ``exclude_feedback=False`` is for offline evaluation: the initial pool
    needs feedback stories present so held-out folds can be measured.

    Runs three legs: recent HN by gravity, archive HN seeds (``bq_seed``/
    ``ch_seed``) by score, and — when ``config.non_hn_candidates_enabled``
    and ``config.rss.enabled`` — recent rows from the currently configured
    RSS/Reddit/LessWrong feeds. Sets ``pool_hn``, ``pool_archive``,
    ``pool_rss`` (post-``is_summarizable`` counts) and, when the RSS leg
    runs, ``pool_rss_oldest_age_h`` (age in hours of the oldest RSS row
    returned, exposing how much of the configured ``days`` window that
    leg's ``LIMIT`` actually reaches) on *trace* for diagnosing candidate
    pool composition.
    """
    if exclude_feedback and user_id is None:
        raise ValueError("user_id is required when exclude_feedback=True")

    now = now_ts if now_ts is not None else int(time.time())
    cutoff_ts = now - (config.days * 86400)
    feedback_filter = (
        "  AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?) "
        if exclude_feedback
        else ""
    )
    feedback_params: tuple[int, ...] = (
        (user_id,) if exclude_feedback and user_id else ()
    )

    # Production legs: recent HN by gravity, archive HN seeds by score,
    # and (when enabled) recent rows from currently configured feeds.
    hn_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories "
        "WHERE time >= ? AND source = 'hn' "
        f"{feedback_filter}"
        "ORDER BY CAST(score AS REAL) / POW((? - time) / 3600.0 + 2.0, 1.8) DESC, "
        "         time DESC "
        "LIMIT ?",
        (
            cutoff_ts,
            *feedback_params,
            now,
            config.recent_candidate_hn_limit,
        ),
    )
    archive_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories WHERE source IN (?, ?) AND text_content != '' "
        f"{feedback_filter}"
        "ORDER BY score DESC, time DESC LIMIT ?",
        (
            BQ_ARCHIVE_SOURCE,
            CH_ARCHIVE_SOURCE,
            *feedback_params,
            BQ_ARCHIVE_CANDIDATE_LIMIT + CH_ARCHIVE_CANDIDATE_LIMIT,
        ),
    )
    rss_rows: list[tuple[Any, ...]] = []
    if config.non_hn_candidates_enabled and config.rss.enabled:
        configured_sources = tuple(
            dict.fromkeys(_rss_source_name(feed) for feed in config.rss.feeds)
        )
        if configured_sources:
            placeholders = ",".join("?" for _ in configured_sources)
            rss_rows = db.execute(
                "SELECT id, title, url, score, time, text_content, source, comment_count, "
                "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
                "FROM stories "
                f"WHERE time >= ? AND source IN ({placeholders}) "
                f"{feedback_filter}"
                "ORDER BY time DESC LIMIT ?",
                (
                    cutoff_ts,
                    *configured_sources,
                    *feedback_params,
                    config.recent_candidate_rss_limit,
                ),
            )
    hn_stories = [
        s
        for s in (Database._row_to_story(row) for row in hn_rows)
        if is_summarizable(s)
    ]
    archive_stories = [
        s
        for s in (Database._row_to_story(row) for row in archive_rows)
        if is_summarizable(s)
    ]
    rss_stories = [
        s
        for s in (Database._row_to_story(row) for row in rss_rows)
        if is_summarizable(s)
    ]
    trace.set_count("pool_hn", len(hn_stories))
    trace.set_count("pool_archive", len(archive_stories))
    trace.set_count("pool_rss", len(rss_stories))
    if rss_stories:
        oldest_rss_time = min(s.time for s in rss_stories)
        trace.set_count("pool_rss_oldest_age_h", int((now - oldest_rss_time) / 3600))
    return hn_stories + rss_stories + archive_stories


def _voted_story_ids(db: Database, user_id: int) -> set[int]:
    """All story IDs a user has left feedback on, any action.

    Mirrors the ``id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)``
    exclusion in ``load_production_candidate_stories`` — used to apply the
    same exclusion as an in-memory mask over the cached candidate pool
    instead of a SQL filter.
    """
    rows = db.execute("SELECT story_id FROM feedback WHERE user_id = ?", (user_id,))
    return {row[0] for row in rows}


def _growth_threshold(fetched: int) -> int:
    """Known-growth bar: max(fetched // 3, 5) — ~33% with a 5-comment floor."""
    return max(fetched // 3, 5)


def _needs_hn_prewarm(s: Story) -> bool:
    """Whether the regen prewarm should refresh this HN story's ``top_comments``.

    Triggers when (a) ``top_comments`` is empty, (b) we have no fetch
    history (``comment_count_at_fetch <= 0``), or (c) the live comment
    count has grown meaningfully since the last prewarm (see
    ``_growth_threshold``).

    The threshold catches the 1->284 "stale single-comment stub" case
    (WORKLOG 2026-06-29) and keeps small stories (10-50 fetched comments)
    from sitting on stale ``top_comments`` while the ceiling-only
    ``max(50, ...)`` previously dominated ``fetched // 2`` until
    ``fetched >= 100``. ``prewarm_top_stories`` rewrites
    ``comment_count_at_fetch`` on every run, so this helper self-clears
    after a single regen cycle.
    """
    if not is_hn_source(s.source):
        return False
    if (s.comment_count or 0) <= 0:
        return False
    if not s.top_comments:
        return True
    fetched = s.comment_count_at_fetch or 0
    if fetched <= 0:
        return True
    growth = (s.comment_count or 0) - fetched
    return growth >= _growth_threshold(fetched)


# Firebase item endpoint: live descendants only (~500 bytes, no auth). The
# cheap "do we know there's new content" probe — full comment trees still
# come from Algolia, and only after confirmed growth.
_FIREBASE_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{sid}.json"
_PROBE_CONCURRENCY = 5
_HYDRATE_CONCURRENCY = 4


def _probe_eligible_threads(
    candidates: Sequence[Story],
    cached_ids: set[int],
    recent_hours: float,
    now: float,
) -> list[Story]:
    """Young cached-TLDR threads whose DB-visible growth is sub-threshold.

    Invisible to ``_needs_hn_prewarm`` (growth < bar) but possibly still
    growing — the DB count lags CH by 1-24h. Velocity-ordered so the regen
    cap spends probes on the hottest threads first. Stories with known
    growth or no fetch history are excluded: the normal prewarm path
    already owns them.
    """
    eligible: list[Story] = []
    for s in candidates:
        if not is_hn_source(s.source):
            continue
        if not s.top_comments or s.id not in cached_ids:
            continue
        if s.time <= 0:
            continue
        age_hours = (now - s.time) / 3600.0
        if age_hours <= 0 or age_hours > recent_hours:
            continue
        fetched = s.comment_count_at_fetch or 0
        if fetched <= 0:
            continue
        growth = (s.comment_count or 0) - fetched
        if growth >= _growth_threshold(fetched):
            continue
        eligible.append(s)
    eligible.sort(
        key=lambda s: (s.comment_count or 0) / max((now - s.time) / 3600.0, 1e-6),
        reverse=True,
    )
    return eligible


async def _probe_live_counts(
    stories: Sequence[Story], timeout_s: float
) -> dict[int, int]:
    """Live Firebase descendants per story id.

    Failures, unparseable bodies, and counts that move backwards are
    silently omitted — the caller treats absence as "no known growth".
    """
    counts: dict[int, int] = {}
    if not stories:
        return counts
    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)

    async with httpx.AsyncClient(timeout=timeout_s) as client:

        async def _one(s: Story) -> tuple[int, int] | None:
            async with sem:
                try:
                    resp = await client.get(_FIREBASE_ITEM_URL.format(sid=s.id))
                except Exception as exc:
                    logging.warning("tldr_probe story_id=%s failed: %r", s.id, exc)
                    return None
                if resp.status_code != 200:
                    logging.warning(
                        "tldr_probe story_id=%s status=%s", s.id, resp.status_code
                    )
                    return None
                try:
                    body = resp.json()
                    live = int((body or {}).get("descendants") or 0)
                except Exception:
                    logging.warning("tldr_probe story_id=%s unparseable body", s.id)
                    return None
                if live <= (s.comment_count or 0):
                    return None
                return (s.id, live)

        for result in await asyncio.gather(*(_one(s) for s in stories)):
            if result is not None:
                counts[result[0]] = result[1]
    return counts


# Regen-local probe memory: sid -> (probed_at, count_seen). Stops flat
# threads (live count stuck between DB count and at_fetch) from burning a
# probe slot every regen. Re-probe when the DB count moves (new CH info)
# or after _PROBE_MEMORY_TTL_S. Injectable via refresh_grown_threads's
# `memory` param so tests stay order-independent.
_PROBE_MEMORY: dict[int, tuple[float, int]] = {}
_PROBE_MEMORY_TTL_S = 24 * 3600
_PROBE_MEMORY_MAX = 2000


async def refresh_grown_threads(
    config: Config,
    db: Database,
    candidates: Sequence[Story],
    *,
    now: float | None = None,
    memory: dict[int, tuple[float, int]] | None = None,
) -> tuple[int, set[int]]:
    """Probe live counts for young cached threads; hydrate confirmed growth.

    Only when we know there's new content: hydration (real-time Algolia
    force-refresh — CH would hand back the same lagged data the probe just
    beat) fires strictly for threads whose probed live count exceeds the
    last fetched count. Counts are healed upward first, so even a failed
    hydration still fixes the velocity gate the tap path reads. The
    prefetch stale lane then regenerates TLDRs whose inputs actually
    changed — no new LLM path, zero spend when the top selection is stable.

    Returns (hydrated, touched): re-hydrated count plus every id whose DB
    row was upserted (count heals included) so the caller can reload
    exactly those candidates. Per-story failures are log lines, never
    fatal to regen.
    """
    at = time.time() if now is None else now
    cap = config.tldr_probe_max_threads_per_regen
    if cap <= 0:
        return (0, set())
    mem = _PROBE_MEMORY if memory is None else memory
    checked = confirmed = hydrated = skipped = 0
    touched: set[int] = set()
    try:
        cached_ids = set(db.get_tldr_cache_keys([s.id for s in candidates if s.id > 0]))
        eligible = _probe_eligible_threads(
            candidates, cached_ids, config.tldr_refresh_recent_hours, at
        )
        fresh = [
            s
            for s in eligible
            if (m := mem.get(s.id)) is None
            or (s.comment_count or 0) != m[1]
            or at - m[0] >= _PROBE_MEMORY_TTL_S
        ]
        skipped = len(eligible) - len(fresh)
        eligible = fresh[:cap]
        checked = len(eligible)
        if not eligible:
            return (0, set())
        by_id = {s.id: s for s in eligible}
        live_counts = await _probe_live_counts(
            eligible, config.tldr_probe_timeout_seconds
        )
        for s in eligible:
            mem[s.id] = (at, s.comment_count or 0)
            if len(mem) > _PROBE_MEMORY_MAX:
                oldest = min(mem, key=lambda k: mem[k][0])
                del mem[oldest]
        for sid, live in live_counts.items():
            s = by_id[sid]
            if live > (s.comment_count or 0):
                db.upsert_story(replace(s, comment_count=live))
                touched.add(sid)
        grown_ids = [
            sid
            for sid, live in live_counts.items()
            if live > (by_id[sid].comment_count_at_fetch or 0)
        ]
        confirmed = len(grown_ids)
        if grown_ids:
            hsem = asyncio.Semaphore(_HYDRATE_CONCURRENCY)

            async with httpx.AsyncClient(timeout=15.0) as client:

                async def _hydrate(sid: int) -> bool:
                    async with hsem:
                        try:
                            updated = await fetch_story(client, sid, db, force=True)
                        except Exception as exc:
                            logging.warning(
                                "tldr_probe hydrate story_id=%s failed: %r", sid, exc
                            )
                            return False
                        if updated is not None:
                            touched.add(sid)
                        return updated is not None

                hydrated = sum(
                    1
                    for ok in await asyncio.gather(
                        *(_hydrate(sid) for sid in grown_ids)
                    )
                    if ok
                )
        return (hydrated, touched)
    finally:
        logging.info(
            "tldr_probe checked=%s confirmed=%s hydrated=%s skipped=%s",
            checked,
            confirmed,
            hydrated,
            skipped,
        )


def is_summarizable(story: Story) -> bool:
    """True if the story has enough text to generate a TLDR (or can fetch it).

    A story is summarizable if it already has any text content, or if it's
    an HN or LessWrong story with comments that can be fetched on-demand
    or prewarmed.
    """
    if story.self_text or story.top_comments or story.article_body:
        return True
    if is_hn_source(story.source):
        if (story.comment_count or 0) > 0:
            return True
        if (story.comment_count_at_fetch or 0) > 0:
            return True
    if story.source == "rss_lesswrong_com":
        if (story.comment_count or 0) > 0:
            return True
        if (story.comment_count_at_fetch or 0) > 0:
            return True
    return False


async def fetch_candidates(
    config: Config,
    exclude_ids: set[int],
    exclude_urls: set[str],
    db: Database,
    embedder: Embedder | None = None,
) -> tuple[list[Story], int]:
    """Fetch candidate stories for the dashboard.

    One CH call does the work the old code did with ~125 Algolia calls:

    `ch_client.query_live_window(days=30, min_score=5, limit=5000)` returns
    every live HN story from the past 30 days with all fields populated
    (title, url, score, descendants, time, text). No per-story items
    call needed.

    Archive seeds (`ch_seed`, `bq_seed`) are read from the DB only.

    The 1-24h CH lag is acceptable: a 3h regen cycle means brand-new
    stories surface within 4h. Algolia's single-story items API is
    preserved as a fallback for `ch_seed`/`bq_seed` lazy fetches.

    Comment text for the top-20 ranked cards is fetched by
    `prewarm_top_stories` on every dashboard render (not here).
    """
    from ch_client import query_live_window

    # 1. Live window from CH (replaces ~125 Algolia search + items calls)
    try:
        live_window = query_live_window(
            days=30,
            min_score=5,
            limit=LIVE_WINDOW_LIMIT,
        )
    except Exception as exc:
        logging.error(
            "fetch_candidates: CH live_window failed (%r); live source empty", exc
        )
        live_window = []

    # 2. Build candidates from live window: insert new, update existing scores
    candidates: list[Story] = []
    fresh_metadata: dict[int, dict] = {}
    existing_stories = {
        s.id: s for s in db.get_stories([item["id"] for item in live_window])
    }
    for item in live_window:
        sid = coerce_int(item.get("id"))
        if sid <= 0 or sid in exclude_ids:
            continue
        fresh_metadata[sid] = {
            "score": coerce_int(item.get("points")),
            "comment_count": coerce_int(item.get("num_comments")),
        }
        existing = existing_stories.get(sid)
        if existing is not None:
            new_score = coerce_int(item.get("points"), existing.score)
            new_comments = coerce_int(
                item.get("num_comments"), existing.comment_count or 0
            )
            has_changes = new_score != existing.score or new_comments != (
                existing.comment_count or 0
            )
            if has_changes:
                updated = replace(existing, score=new_score, comment_count=new_comments)
                db.upsert_story(updated)
                candidates.append(updated)
            else:
                candidates.append(existing)
        else:
            new_story = _ch_story_item_to_story(item)
            if new_story is not None:
                db.upsert_story(new_story)
                candidates.append(new_story)

    # 3. Archive seeds from DB (read-only)
    rows = db.execute(
        """
        SELECT id FROM stories
        WHERE source IN (?, ?) AND text_content != ''
        ORDER BY score DESC, time DESC
        LIMIT ?
        """,
        (
            BQ_ARCHIVE_SOURCE,
            CH_ARCHIVE_SOURCE,
            BQ_ARCHIVE_CANDIDATE_LIMIT + CH_ARCHIVE_CANDIDATE_LIMIT,
        ),
    )
    archive_ids = [row[0] for row in rows if row[0] not in exclude_ids]
    if archive_ids:
        archive_stories = db.get_stories(archive_ids)
        candidates.extend(archive_stories)

    # 4. Skip fetch-time dedup — duplicate resolution is centralized at
    #    render time in `_apply_dedup_to_ranked` (see dedup.py). We still
    #    pass `exclude_urls` (built from the user's feedback set) into the
    #    RSS fetch so we don't re-pull URLs the user has already voted on.

    # 5. RSS feeds
    if config.rss.enabled:
        rss_stories = await fetch_rss_feeds(
            feeds=list(config.rss.feeds),
            per_feed=config.rss.per_feed_limit,
            days=config.days,
            exclude_urls=exclude_urls,
            db=db,
        )
        deduped_candidates: list[Story] = list(candidates) + rss_stories
    else:
        deduped_candidates = list(candidates)

    # 6. Filter to summarizable stories (no content = no TLDR possible)
    summarizable = [s for s in deduped_candidates if is_summarizable(s)]
    n_filtered = len(deduped_candidates) - len(summarizable)
    if n_filtered:
        logging.info(
            "fetch_candidates: filtered %d unsummarizable stories",
            n_filtered,
        )
    return summarizable, len(summarizable)


def fast_rerank_for_user(
    db: Database,
    config: Config,
    embedder: Embedder,
    user_id: int,
    trace: RankTrace | _NullTrace = NULL_TRACE,
) -> list[RankedStory]:
    """Fast rerank for a specific user. Called on each dashboard request."""
    trace.set_count("user_id", user_id)

    n_feedback = sum(db.count_feedback_by_action(user_id).values())
    trace.set_count("feedback_total", n_feedback)
    if n_feedback == 0:
        trace.set_label("model_cache", "skipped_cold_deck")
        # Zero-feedback cold deck is a pure gravity/time ranking — no
        # embeddings needed, so don't force the (embedder-requiring)
        # candidate pool cache here; keep the direct, uncached load.
        cold_deck = build_cold_deck(db, config, trace=trace)
        cold_final = canonicalize_hn_dupes(
            cold_deck,
            db,
            selected_limit=config.count,
            user_id=user_id,
            feedback_actions=tuple(config.model.dedup_exclude_actions),
            trace=trace,
        )
        trace.set_count(
            "deck_nonhn_final",
            sum(1 for r in cold_final if not is_hn_source(r.story.source)),
        )
        return cold_final

    with trace.stage("candidate_sql"):
        pool = get_candidate_pool(db, config, embedder, trace=trace)
        voted_ids = frozenset(_voted_story_ids(db, user_id))
    if not pool.stories:
        return []

    with trace.stage("candidate_embedding"):
        candidates, cand_embeddings = pool.without_feedback(voted_ids)
    if trace is not None:
        trace.set_count("candidates", len(candidates))
    if not candidates:
        return []

    # Built once per request and threaded into badge assembly so the
    # Explore passes (Unsure/Novel/Similar) can skip-and-backfill past
    # candidates that duplicate a story the user already voted on, instead
    # of silently losing badge slots to the downstream `canonicalize_hn_dupes`
    # feedback-match drop (see WORKLOG 2026-07-10).
    feedback_context = _load_feedback_context(
        db, user_id=user_id, actions=tuple(config.model.dedup_exclude_actions)
    )
    ranked = rerank_candidates(
        db=db,
        config=config,
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=cand_embeddings,
        user_id=user_id,
        trace=trace,
        is_feedback_match=lambda s: _matches_feedback(s, feedback_context),
    )
    trace.set_count(
        "deck_nonhn_pre_dedup",
        sum(1 for r in ranked if not is_hn_source(r.story.source)),
    )

    return finalize_ranked_deck(
        ranked, candidates, cand_embeddings, db, config, embedder, user_id, trace=trace
    )


def finalize_ranked_deck(
    ranked: list[RankedStory],
    candidates: list[Story],
    cand_embeddings: NDArray[np.float32],
    db: Database,
    config: Config,
    embedder: Embedder,
    user_id: int,
    *,
    trace: RankTrace | _NullTrace = NULL_TRACE,
) -> list[RankedStory]:
    """Shared serving/evaluation deduplication and canonicalization boundary."""
    with trace.stage("dedup"):
        id_to_emb: dict[int, NDArray[np.float32]] = {
            s.id: vec for s, vec in zip(candidates, cand_embeddings)
        }
        deduped = _apply_dedup_to_ranked(
            ranked,
            db,
            config,
            user_id,
            embeddings=id_to_emb,
            embedder=embedder,
        )
    trace.set_count(
        "deck_nonhn_post_dedup",
        sum(1 for r in deduped if not is_hn_source(r.story.source)),
    )
    with trace.stage("hn_dupes"):
        final = canonicalize_hn_dupes(
            deduped,
            db,
            candidate_stories=candidates,
            selected_limit=config.count,
            user_id=user_id,
            feedback_actions=tuple(config.model.dedup_exclude_actions),
            trace=trace,
        )
    trace.set_count(
        "deck_nonhn_final",
        sum(1 for r in final if not is_hn_source(r.story.source)),
    )
    return final


def _apply_dedup_to_ranked(
    ranked: list[RankedStory],
    db: Database,
    config: Config,
    user_id: int,
    embeddings: dict[int, NDArray[np.float32]] | None = None,
    embedder: Embedder | None = None,
) -> list[RankedStory]:
    """Filter *ranked* through :func:`dedup.dedup_ranked`.

    Pulls this user's feedback (so the URL exclusion logic has
    data) and applies a :class:`dedup.DedupConfig` built from the
    :class:`ModelConfig`. Preserves the caller's rank order.
    """
    from dedup import DedupConfig, dedup_ranked

    model_cfg = config.model
    dedup_cfg = DedupConfig(
        render_enabled=model_cfg.dedup_render_enabled,
        embedding_cosine_enabled=model_cfg.dedup_embedding_cosine_enabled,
        embedding_cosine_threshold=model_cfg.dedup_embedding_cosine_threshold,
        exclude_actions=tuple(model_cfg.dedup_exclude_actions),
    )
    feedback = db.get_all_feedback(user_id=user_id)

    # Merge feedback story embeddings so cross-source duplicates (e.g.
    # Slashdot rewriting an HN story the user has already voted on) can
    # be suppressed by the embedding dedup step. Feedback stories get
    # score=-1 so they always lose same-source tiebreaks against real
    # candidates.
    all_stories = [r.story for r in ranked]
    all_embeddings = dict(embeddings) if embeddings else {}
    if embedder is not None:
        fb_stories = db.get_feedback_stories(
            user_id,
            actions=tuple(model_cfg.dedup_exclude_actions),
        )
        if fb_stories:
            fb_embs = get_or_compute_embeddings(fb_stories, embedder, db)
            for s, vec in zip(fb_stories, fb_embs):
                all_embeddings[s.id] = vec
            all_stories.extend(fb_stories)

    survivor_stories = dedup_ranked(
        all_stories,
        feedback,
        dedup_cfg,
        user_id=user_id,
        embeddings=all_embeddings if all_embeddings else None,
    )
    survivors_by_id = {s.id: s for s in survivor_stories}
    return [r for r in ranked if r.story.id in survivors_by_id]


async def fetch_candidates_only(
    config: Config,
    db: Database,
    embedder: Embedder | None = None,
    prewarm_top_n: int | None = None,
    on_hn_candidates: Callable[[Sequence[Story]], None] | None = None,
) -> None:
    """Fetch and publish core candidates without waiting for Reddit."""

    feedback_records = db.get_all_feedback()
    feedback_ids = {f.story_id for f in feedback_records}
    feedback_urls = {f.url for f in feedback_records if f.url}

    candidates, n_fetched = await fetch_candidates(
        config, feedback_ids, feedback_urls, db, None
    )
    logging.info("Regen: fetched %d candidates", n_fetched)
    if on_hn_candidates is not None:
        try:
            on_hn_candidates([story for story in candidates if story.source == "hn"])
        except Exception:
            logging.exception("fetch_candidates_only: HN dupe callback failed")

    # Growth-gated live refresh: probe young cached threads for real
    # comment growth (CH counts lag 1-24h), hydrate confirmed ones before
    # prewarm/prefetch so the stale TLDR lane sees fresh inputs. Reload
    # every touched row: counts heal upward even without hydration, and the
    # prewarm gate below must read the healed counts.
    try:
        _, touched = await refresh_grown_threads(config, db, candidates)
    except Exception:
        logging.exception("fetch_candidates_only: growth refresh failed")
        touched = set()
    if touched:
        refreshed = {s.id: s for s in db.get_stories(sorted(touched))}
        candidates = [refreshed.get(s.id, s) for s in candidates]

    # HN prewarm
    if config.prewarm_hn_full and embedder is not None:
        needs_prewarm = [s.id for s in candidates if _needs_hn_prewarm(s)]
        if needs_prewarm:
            prewarmed = prewarm_top_stories(needs_prewarm, db, embedder)
            logging.info(
                "Regen: prewarmed %d/%d HN candidates (full mode)",
                prewarmed,
                len(needs_prewarm),
            )
    else:
        if prewarm_top_n is None:
            prewarm_top_n = config.regen_prewarm_top_n
        if prewarm_top_n > 0 and embedder is not None:
            hn_ids = [s.id for s in candidates if s.id > 0]
            hn_ids.sort(
                key=lambda sid: next(s.score for s in candidates if s.id == sid),
                reverse=True,
            )
            top_ids = hn_ids[:prewarm_top_n]
            if top_ids:
                prewarmed = prewarm_top_stories(top_ids, db, embedder)
                logging.info(
                    "Regen: prewarmed %d/%d top-by-score stories",
                    prewarmed,
                    len(top_ids),
                )

    # LessWrong prewarm
    if config.prewarm_lesswrong_full:
        needs_prewarm_lw = [
            s.id
            for s in candidates
            if s.source == "rss_lesswrong_com" and not s.top_comments
        ]
        if needs_prewarm_lw:
            prewarmed = await prewarm_lesswrong_stories(needs_prewarm_lw, db, embedder)
            logging.info(
                "Regen: prewarmed %d/%d LessWrong candidates (full mode)",
                prewarmed,
                len(needs_prewarm_lw),
            )


def refresh_reddit_candidates(
    config: Config,
    db: Database,
    embedder: Embedder | None,
) -> RedditRefreshResult:
    """Run the slow Reddit refresh phases outside core regeneration."""
    from reddit_fetch_queue import queue as reddit_fetch_queue

    now_ts = time.time()
    feedback_urls = {f.url for f in db.get_all_feedback() if f.url}
    eligible_feeds = [
        feed
        for feed in config.rss.feeds
        if (state := db.get_reddit_feed_state(feed)) is None
        or state.next_retry_at <= now_ts
    ]
    factories, feed_urls = build_reddit_topfeed_factories(
        eligible_feeds,
        config.rss.per_feed_limit,
        config.days,
        feedback_urls,
    )
    if factories:
        reddit_fetch_queue.enqueue_all_reddit_fetches(
            factories, [], min_stride_seconds=50.0
        )
        if not reddit_fetch_queue.wait_until_empty(timeout=5400.0):
            logging.warning("reddit_refresh: topfeed queue timed out")

    changed_ids: set[int] = set()
    for feed_url in feed_urls:
        cached = reddit_feed_cache.get(feed_url)
        if cached is None:
            db.record_reddit_feed_failure(
                feed_url, "fetch returned no snapshot", now_ts
            )
            continue
        for story in cached:
            existing = db.get_story(story.id)
            if existing is None or (
                existing.title,
                existing.url,
                existing.time,
                existing.self_text,
            ) != (story.title, story.url, story.time, story.self_text):
                changed_ids.add(story.id)
            db.upsert_story(story)
        db.record_reddit_feed_success(feed_url, [story.id for story in cached], now_ts)

    prewarm_ids: list[int] = []
    if config.prewarm_reddit_full:
        for feed_url in feed_urls:
            if len(prewarm_ids) >= config.reddit_prewarm_max_per_cycle:
                break
            for story in (reddit_feed_cache.get(feed_url) or [])[
                : config.reddit_prewarm_top_per_sub
            ]:
                if len(prewarm_ids) >= config.reddit_prewarm_max_per_cycle:
                    break
                existing = db.get_story(story.id)
                if existing is None or not existing.top_comments:
                    prewarm_ids.append(story.id)

    prewarm_factories, updated_ids = build_reddit_prewarm_factories(prewarm_ids, db)
    if prewarm_factories:
        reddit_fetch_queue.enqueue_all_reddit_fetches(
            [],
            prewarm_factories,
            min_stride_seconds=config.reddit_min_fetch_spacing_seconds,
        )
        if not reddit_fetch_queue.wait_until_empty(timeout=5400.0):
            logging.warning("reddit_refresh: prewarm queue timed out")

    if updated_ids and embedder is not None:
        updated = [story for sid in updated_ids if (story := db.get_story(sid))]
        if updated:
            get_or_compute_embeddings(updated, embedder, db)

    result = RedditRefreshResult(
        feeds=len(feed_urls),
        changed_stories=len(changed_ids),
        hydrated_stories=len(set(updated_ids)),
        prewarm_candidates=len(prewarm_ids),
    )
    logging.info(
        "reddit_refresh_complete feeds=%d changed=%d hydrated=%d candidates=%d",
        result.feeds,
        result.changed_stories,
        result.hydrated_stories,
        result.prewarm_candidates,
    )
    return result
