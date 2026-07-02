from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import feedparser
import httpx
import numpy as np
from bs4 import BeautifulSoup
from numpy.typing import NDArray

from database import Database, Story
from reddit_fetch_queue import CoroFactory

# ruff: noqa: F401 — all imports below are intentional re-exports for the
# public pipeline namespace. Consumers import from `pipeline` directly.
from .config import (
    BQ_ARCHIVE_CANDIDATE_LIMIT,
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_CANDIDATE_LIMIT,
    CH_ARCHIVE_SOURCE,
    Config,
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
    _rank_percentiles,
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
    RSS_USER_AGENT,
    REDDIT_RSS_USER_AGENT,
    _PAYWALL_DOMAINS,
    _article_failure_retry_time,
    _article_fetch_extra_priority,
    _article_fetch_failure_active,
    _ch_story_item_to_story,
    _coerce_int,
    _empty_story,
    _fetch_and_parse_feed,
    _is_fetchable_article_url,
    _parse_rate_limit_reset,
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

from .render import (
    BadgeView,
    DashboardCardView,
    TabGroupView,
    TabView,
    VoteCountsView,
    _build_badges,
    _build_dashboard_cards,
    _build_tab_groups,
    _get_pico_css,
    generate_dashboard_bytes,
    source_label_filter,
    time_ago_filter,
)
from reddit_feed_cache import cache as reddit_feed_cache
from reddit_limiter import limiter as reddit_limiter
from transformers import AutoTokenizer


COLD_DECK_LIMIT = 100
COLD_DECK_QUERY_LIMIT = 400


def _combo_keys_for_story(story: Story, recent_cutoff: int) -> str:
    age = "recent" if story.time >= recent_cutoff else "archive"
    source = "hn" if is_hn_source(story.source) else "non-hn"
    return f"{age}_{source} {age}_mixed"


def build_cold_deck(db: Database, user_id: int | None = None) -> list[RankedStory]:
    """Build a gravity-sorted fallback deck from existing SQLite rows.

    Uses the same tier-1 gravity formula as ``_score_and_rank`` so a
    zero-vote user sees the same ranking as the cold deck.  See
    ``fast_rerank_for_user`` for the 0-vote short-circuit.

    When *user_id* is provided, the SQL filters out stories the user
    has already voted on.
    """
    now_ts = int(time.time())
    if user_id is not None:
        rows = db.execute(
            "SELECT id, title, url, score, time, text_content, source, comment_count, "
            "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
            "FROM stories "
            "WHERE id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?) "
            "ORDER BY CAST(score AS REAL) / POW((? - time) / 3600.0 + 2.0, 1.8) DESC, "
            "         time DESC "
            "LIMIT ?",
            (user_id, now_ts, COLD_DECK_QUERY_LIMIT),
        )
    else:
        rows = db.execute(
            "SELECT id, title, url, score, time, text_content, source, comment_count, "
            "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
            "FROM stories "
            "ORDER BY CAST(score AS REAL) / POW((? - time) / 3600.0 + 2.0, 1.8) DESC, "
            "         time DESC "
            "LIMIT ?",
            (now_ts, COLD_DECK_QUERY_LIMIT),
        )
    stories = [Database._row_to_story(row) for row in rows]
    recent_cutoff = now_ts - (30 * 86400)
    cold: list[RankedStory] = []
    for story in stories:
        if not is_summarizable(story):
            continue
        cold.append(
            RankedStory(
                story=story,
                score=story.score / ((now_ts - story.time) / 3600.0 + 2.0) ** 1.8,
                best_match_title="",
                is_non_hn=(not is_hn_source(story.source)),
                is_recent=(story.time >= recent_cutoff),
                combo_keys=_combo_keys_for_story(story, recent_cutoff),
            )
        )
        if len(cold) >= COLD_DECK_LIMIT:
            break
    return cold


def load_production_candidate_stories(
    db: Database,
    config: Config,
    *,
    user_id: int | None,
    exclude_feedback: bool,
    now_ts: int | None = None,
) -> list[Story]:
    """Load the same four candidate legs used by the personalized dashboard.

    ``exclude_feedback=False`` is for offline evaluation: the initial pool
    needs feedback stories present so held-out folds can be measured.
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

    # Four production legs: recent HN by gravity, recent non-HN by recency,
    # archive HN seeds by score, and archive non-HN by recency.
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
    rss_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories "
        "WHERE time >= ? AND source != 'hn' AND source NOT IN (?, ?) "
        f"{feedback_filter}"
        "ORDER BY time DESC "
        "LIMIT ?",
        (
            cutoff_ts,
            BQ_ARCHIVE_SOURCE,
            CH_ARCHIVE_SOURCE,
            *feedback_params,
            config.recent_candidate_rss_limit,
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
    archive_rss_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories "
        "WHERE source != 'hn' AND source NOT IN (?, ?) AND time < ? "
        f"{feedback_filter}"
        "ORDER BY time DESC LIMIT ?",
        (
            BQ_ARCHIVE_SOURCE,
            CH_ARCHIVE_SOURCE,
            cutoff_ts,
            *feedback_params,
            config.recent_candidate_rss_limit,
        ),
    )
    rows = hn_rows + rss_rows + archive_rows + archive_rss_rows
    return [
        story
        for story in (Database._row_to_story(row) for row in rows)
        if is_summarizable(story)
    ]


def _needs_hn_prewarm(s: Story) -> bool:
    """Whether the regen prewarm should refresh this HN story's ``top_comments``.

    Triggers when (a) ``top_comments`` is empty, (b) we have no fetch
    history (``comment_count_at_fetch <= 0``), or (c) the live comment
    count has grown meaningfully since the last prewarm: at least
    ``max(fetched // 3, 5)`` new comments — roughly 33% growth with a
    5-comment floor.

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
    threshold = max(fetched // 3, 5)
    return growth >= threshold


# Source category one-hot. Order is significant: callers index into the
# returned vector and tests assert specific positions.


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


# Text processing helpers


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
        sid = _coerce_int(item.get("id"))
        if sid <= 0 or sid in exclude_ids:
            continue
        fresh_metadata[sid] = {
            "score": _coerce_int(item.get("points")),
            "comment_count": _coerce_int(item.get("num_comments")),
        }
        existing = existing_stories.get(sid)
        if existing is not None:
            new_score = _coerce_int(item.get("points"), existing.score)
            new_comments = _coerce_int(
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


# RSS Fetching
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
        return build_cold_deck(db)

    with trace.stage("candidate_sql"):
        candidates = load_production_candidate_stories(
            db,
            config,
            user_id=user_id,
            exclude_feedback=True,
        )
    if trace is not None:
        trace.set_count("candidates", len(candidates))
    if not candidates:
        return []

    with trace.stage("candidate_embedding"):
        cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    ranked = rerank_candidates(
        db=db,
        config=config,
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=cand_embeddings,
        user_id=user_id,
        trace=trace,
    )

    with trace.stage("dedup"):
        id_to_emb: dict[int, NDArray[np.float32]] = {
            s.id: vec for s, vec in zip(candidates, cand_embeddings)
        }
        return _apply_dedup_to_ranked(
            ranked,
            db,
            config,
            user_id,
            embeddings=id_to_emb,
            embedder=embedder,
        )


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
) -> None:
    """Fetch new candidates into shared DB; prewarm top-N hot per sub.

    The Reddit fetch pipeline runs in two phases via the shared
    :class:`reddit_fetch_queue.RedditFetchQueue`:

    1. **Topfeed phase.** All 41 subreddit topfeeds are enqueued at a
       fixed 50s stride. Reddit returns entries in hot/score-desc order
       and the factory stores them in ``reddit_feed_cache`` in that
       order. The phase blocks until the queue drains (with a generous
       90-min timeout for slow networks / rate-limit backoffs).
    2. **Prewarm phase.** After the topfeed phase completes, the first
       ``config.reddit_prewarm_top_per_sub`` (default 10) stories per
       subreddit are read from the cache. These are the hottest
       ``N * 41`` stories across all subs. Per-post RSS fetches for
       their comments are enqueued at ``config.reddit_min_fetch_spacing_seconds``
       (default 30s) stride. Multi-cycle completion is expected: the
       90-min drain timeout covers ~180 of 410 prewarm tasks; the rest
       finish in subsequent cycles.

    The two-phase flow is needed because the prewarm IDs come from the
    topfeed cache, which is populated by the topfeed phase. A single
    interleaved enqueue can't do that — the prewarm factories would have
    no IDs to enqueue.

    New Reddit stories discovered by this cycle's topfeed are ranked in
    this cycle (their stories are upserted post-drain) but their
    per-post comment fetch is best-effort. Stories already in the cache
    from a previous cycle's topfeed are prewarmed in this cycle.
    """
    from reddit_fetch_queue import queue as reddit_fetch_queue

    feedback_records = db.get_all_feedback()
    feedback_ids = {f.story_id for f in feedback_records}
    feedback_urls = {f.url for f in feedback_records if f.url}

    candidates, n_fetched = await fetch_candidates(
        config, feedback_ids, feedback_urls, db, None
    )
    logging.info("Regen: fetched %d candidates", n_fetched)

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

    # Reddit topfeed + prewarm via the shared queue. Both are enqueued
    # together via `enqueue_all_reddit_fetches` so they interleave on a
    # single `min_stride_seconds` window. The single drain at the end of
    # the regen flow handles the case where the queue never fully drains
    # (we log a warning but continue — prewarm counts reflect only what
    # ran).
    topfeed_factories, reddit_feed_urls = build_reddit_topfeed_factories(
        list(config.rss.feeds),
        config.rss.per_feed_limit,
        config.days,
        feedback_urls,
    )

    prewarm_ids: list[int] = []
    prewarm_factories: list[CoroFactory] = []
    prewarm_updated_ids: list[int] = []

    if topfeed_factories:
        # Phase 1: topfeed. Enqueue all 41 subs at a fixed 50s stride.
        # The factory writes parsed Stories to ``reddit_feed_cache`` in
        # the order Reddit returned them (hot/score-desc). We don't need
        # any prewarm IDs before this phase completes — the prewarm phase
        # reads the cache to determine the top-N per sub.
        reddit_fetch_queue.enqueue_all_reddit_fetches(
            topfeed_factories,
            [],
            min_stride_seconds=50.0,
        )
        # 41 subs × 50s = 2050s nominal; 90-min drain gives ample slack
        # for 429 backoffs and slow networks.
        topfeed_drained = reddit_fetch_queue.wait_until_empty(timeout=5400.0)
        if not topfeed_drained:
            logging.warning(
                "fetch_candidates_only: reddit topfeed queue did not drain "
                "in 5400s, continuing with partial cache"
            )

    # Phase 1.5: persist cached topfeed stories into SQLite. The prewarm
    # factories load each story row from the DB with `db.get_story`, so
    # the rows must exist before prewarm is enqueued. Without this step
    # brand-new topfeed discoveries would no-op through the prewarm
    # factory's missing-row short-circuit.
    if topfeed_factories and reddit_feed_urls:
        for feed_url in reddit_feed_urls:
            cached = reddit_feed_cache.get(feed_url)
            if cached:
                for story in cached:
                    if db.get_story(story.id) is not None:
                        continue
                    db.upsert_story(story)

    if config.prewarm_reddit_full and reddit_feed_urls:
        # Phase 2: prewarm. After the topfeed phase, read the cache and
        # take the first ``reddit_prewarm_top_per_sub`` stories per
        # subreddit. These are the hottest N per sub across all 41 subs.
        # Skip rows whose DB copy already has `top_comments` (already
        # hydrated by a previous cycle), and stop at
        # ``reddit_prewarm_max_per_cycle`` to keep the per-cycle work
        # bounded under current rate limits.
        n_per_sub = config.reddit_prewarm_top_per_sub
        max_per_cycle = config.reddit_prewarm_max_per_cycle
        for feed_url in reddit_feed_urls:
            if max_per_cycle <= 0 or len(prewarm_ids) >= max_per_cycle:
                break
            cached = reddit_feed_cache.get(feed_url)
            if not cached:
                continue
            for story in cached[:n_per_sub]:
                if len(prewarm_ids) >= max_per_cycle:
                    break
                existing = db.get_story(story.id)
                if existing and existing.top_comments:
                    continue
                prewarm_ids.append(story.id)

        if prewarm_ids:
            prewarm_factories, prewarm_updated_ids = build_reddit_prewarm_factories(
                prewarm_ids, db
            )

        if prewarm_factories:
            reddit_fetch_queue.enqueue_all_reddit_fetches(
                [],
                prewarm_factories,
                min_stride_seconds=config.reddit_min_fetch_spacing_seconds,
            )
            # 90-min drain covers ~180 of 410 at 30s stride. The rest
            # continue in the background and finish over the next 2-3
            # regen cycles. Multi-cycle completion is by design.
            prewarm_drained = reddit_fetch_queue.wait_until_empty(timeout=5400.0)
            if not prewarm_drained:
                logging.warning(
                    "fetch_candidates_only: reddit prewarm queue did not "
                    "drain in 5400s, continuing with partial comments"
                )

    if topfeed_factories or prewarm_factories:
        # Post-drain: collect Reddit topfeed stories from cache and
        # extend the candidates list. Persistence was handled by the
        # Phase 1.5 upsert loop above; re-upsert here is redundant.
        for feed_url in reddit_feed_urls:
            cached = reddit_feed_cache.get(feed_url)
            if cached:
                candidates.extend(cached)

        # Post-drain: recompute embeddings for stories whose
        # `top_comments`/`self_text` changed during prewarm.
        if prewarm_updated_ids and embedder is not None:
            updated_stories = [db.get_story(sid) for sid in prewarm_updated_ids]
            updated_stories = [s for s in updated_stories if s is not None]
            if updated_stories:
                get_or_compute_embeddings(updated_stories, embedder, db)

        logging.info(
            "Regen: reddit topfeed=%d prewarm=%d (top_per_sub=%d, prewarm_ids=%d)",
            len(reddit_feed_urls),
            len(prewarm_updated_ids),
            config.reddit_prewarm_top_per_sub,
            len(prewarm_ids),
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
