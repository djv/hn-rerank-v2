from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import threading
import time
import tomllib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from typing import Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import httpx
import numpy as np
import onnxruntime as ort
from bs4 import BeautifulSoup
from cachetools import LRUCache
from jinja2 import Environment, FileSystemLoader
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from database import Database, Story
from reddit_fetch_queue import CoroFactory

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
from reddit_feed_cache import cache as reddit_feed_cache
from reddit_limiter import limiter as reddit_limiter
from transformers import AutoTokenizer


RSS_USER_AGENT = "hn-rewrite/1.0 (+https://github.com/local/hn-rewrite)"
REDDIT_RSS_USER_AGENT = "hn-rewrite/1.0 personal RSS reader; contact: local dashboard"


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


@dataclass(frozen=True)
class BadgeView:
    kind: str
    icon: str
    label: str
    tooltip: str

    @property
    def css_class(self) -> str:
        return f"badge badge--{self.kind}"


@dataclass(frozen=True)
class TabView:
    value: str
    label_html: str
    active: bool = False


@dataclass(frozen=True)
class TabGroupView:
    key: str
    aria_label: str
    css_class: str
    data_attr: str
    segmented: bool
    tabs: tuple[TabView, ...]


@dataclass(frozen=True)
class VoteCountsView:
    up: int
    neutral: int
    down: int


@dataclass(frozen=True)
class DashboardCardView:
    story: Story
    score: float
    best_match_title: str
    badges: tuple[BadgeView, ...]
    combo_keys: str
    is_enriched: bool
    is_hn_attr: str
    sort_popular_attr: str
    sort_explore_attr: str
    is_recent_attr: str
    article_url: str
    comments_url: str
    source_label: str
    time_ago: str
    show_source_badge: bool
    show_score: bool


# Text processing helpers
def _coerce_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ch_story_item_to_story(item: dict) -> Story | None:
    """Convert a CH live-window item dict (Algolia shape) to a Story row.

    The CH live_window query returns the same fields as Algolia items
    (id, type, title, url, points, num_comments, created_at_i, text,
    children). For live `hn` source, we insert directly with the CH
    data; comment hydration is handled later by the bulk prewarm path.
    """
    sid = _coerce_int(item.get("id"))
    title = clean_text(str(item.get("title") or ""))
    if sid <= 0 or not title:
        return None
    self_text = clean_text(str(item.get("text") or ""))
    text_content = compose_story_text(title, self_text)
    if not text_content:
        return None
    return Story(
        id=sid,
        title=title,
        url=item.get("url") or None,
        score=_coerce_int(item.get("points")),
        time=_coerce_int(item.get("created_at_i")),
        text_content=text_content,
        source="hn",
        comment_count=_coerce_int(item.get("num_comments")),
        discussion_url=f"https://news.ycombinator.com/item?id={sid}",
        comment_count_at_fetch=_coerce_int(item.get("num_comments")),
        self_text=self_text,
        top_comments="",
        article_body="",
    )


def _empty_story(sid: int) -> Story:
    return Story(
        id=sid, title="", url=None, score=0, time=0, text_content="", source="hn"
    )


async def fetch_story(
    client: httpx.AsyncClient, sid: int, db: Database
) -> Story | None:
    story = db.get_story(sid)
    if story is not None:
        if story.text_content == "":
            if story.title == "":
                pass  # corrupted _empty_story, fall through to API re-fetch
            else:
                return None
        comments_stale = story.top_comments == "" or (story.comment_count or 0) > (
            story.comment_count_at_fetch or 0
        )
        if not comments_stale:
            return story
        if story.top_comments != "" and (story.comment_count_at_fetch or 0) > 50:
            return story

    url = f"https://hn.algolia.com/api/v1/items/{sid}"
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            if story is None:
                db.upsert_story(_empty_story(sid))
            return story if story else None

        item = resp.json()
        if not item or item.get("type") != "story":
            if story is None:
                db.upsert_story(_empty_story(sid))
            return story if story else None

        title = html.unescape(item.get("title", ""))
        story_url = item.get("url")
        score = item.get("points") or 0
        comment_count = item.get("num_comments")
        created_at = item.get("created_at_i") or 0
        story_text = clean_text(str(item.get("story_text") or item.get("text") or ""))

        children = item.get("children", [])
        all_comments = _extract_comments_recursive(children)
        selected = _select_top_comments(all_comments)
        top_comment_texts = " ".join(c["text"] for c in selected)[:10000]

        text_content = compose_story_text(
            title=title,
            self_text=story_text,
            comments=top_comment_texts,
            article_body="",
        )

        if not text_content:
            if story is None:
                db.upsert_story(
                    Story(
                        id=sid,
                        title="",
                        url=None,
                        score=0,
                        time=0,
                        text_content="",
                        source="hn",
                    )
                )
            return story if story else None

        source = story.source if story is not None else "hn"
        story = Story(
            id=sid,
            title=title,
            url=story_url or None,
            score=score,
            time=created_at,
            text_content=text_content,
            source=source,
            comment_count=comment_count
            if comment_count is not None
            else len(all_comments),
            discussion_url=f"https://news.ycombinator.com/item?id={sid}",
            comment_count_at_fetch=comment_count
            if comment_count is not None
            else len(all_comments),
            self_text=story_text,
            top_comments=top_comment_texts,
            article_body="",
        )

        db.upsert_story(story)
        return story
    except Exception as e:
        logging.error("Error fetching story %s: %r", sid, e)
        return story if story else None


def prewarm_top_stories(
    story_ids: list[int],
    db: Database,
    embedder: Embedder | None = None,
    max_levels: int = 5,
) -> int:
    """Bulk-prewarm comment text for the top-N stories.

    Fetches full comment trees for the given story IDs in a single ClickHouse
    query, selects top comments using the same logic as Algolia, recomposes
    text_content, and writes back to the stories table. Skips stories that
    already have a recent comment fetch (no need to redo the work).

    Always-on; called from `render_dashboard` after ranking. The CH bulk path
    handles up to 200 stories in <1s vs ~30s for parallel Algolia.

    Args:
        story_ids: ordered list of story IDs to prewarm (e.g., top-20 ranked).
        db: Database instance for read + write.
        embedder: Optional Embedder; if provided, recomputes the embedding
            for any story whose text_content changed.
        max_levels: comment tree depth (default 5; covers ~95% of trees).

    Returns:
        Number of stories whose top_comments was updated.
    """
    from ch_client import query_stories_with_comments

    if not story_ids:
        return 0

    target_ids = [int(s) for s in story_ids if int(s) > 0]
    if not target_ids:
        return 0

    try:
        ch_items = query_stories_with_comments(target_ids, max_levels=max_levels)
    except Exception as exc:
        logging.warning("prewarm_top_stories: CH bulk query failed (%r)", exc)
        return 0

    if not ch_items:
        return 0

    existing_map = {s.id: s for s in db.get_stories(target_ids)}
    updated: list[Story] = []
    for sid, item in ch_items.items():
        existing = existing_map.get(sid)
        if existing is None:
            continue
        children = item.get("children") or []
        all_comments = _extract_comments_recursive(children)
        selected = _select_top_comments(all_comments)
        top_comments = " ".join(c["text"] for c in selected)[:10000]
        if not top_comments:
            continue
        comment_count = _coerce_int(
            item.get("num_comments"), existing.comment_count or 0
        )
        new_text_content = compose_story_text(
            title=existing.title,
            self_text=existing.self_text,
            comments=top_comments,
            article_body=existing.article_body,
        )
        if not new_text_content:
            continue
        updated_story = replace(
            existing,
            top_comments=top_comments,
            text_content=new_text_content,
            comment_count=comment_count,
            comment_count_at_fetch=comment_count,
        )
        db.upsert_story(updated_story)
        updated.append(updated_story)

    if updated and embedder is not None:
        get_or_compute_embeddings(updated, embedder, db)

    return len(updated)


async def prewarm_reddit_top_stories(
    story_ids: list[int],
    db: Database,
    embedder: Embedder | None = None,
    *,
    spread_window_seconds: float | None = None,
) -> int:
    """Bulk-prewarm Reddit RSS comment text for top-N stories.

    Convenience wrapper: builds prewarm factories via
    :func:`build_reddit_prewarm_factories`, enqueues them on the shared
    queue, waits for the queue to drain, then recomputes embeddings for
    the stories whose ``top_comments`` or ``self_text`` changed.

    The regen path uses :func:`fetch_candidates_only` directly, which
    interleaves topfeed and prewarm factories via
    :func:`reddit_fetch_queue.enqueue_all_reddit_fetches` and drains
    once. This standalone wrapper exists for tests and ad-hoc callers
    that want a simple blocking prewarm.

    Returns number of stories whose top_comments or self_text changed.
    """
    if not story_ids:
        return 0
    from reddit_fetch_queue import queue as reddit_fetch_queue

    factories, updated_ids = build_reddit_prewarm_factories(story_ids, db)
    window = (
        spread_window_seconds
        if spread_window_seconds is not None
        else (len(factories) * reddit_fetch_queue.MIN_FETCH_SPACING)
    )
    reddit_fetch_queue.enqueue_spread(
        len(factories),
        time.monotonic(),
        "prewarm",
        factories,
        window_seconds=window,
    )
    drained = reddit_fetch_queue.wait_until_empty(timeout=1500.0)
    if not drained:
        logging.warning(
            "prewarm_reddit: queue did not drain in 1500s, returning partial count"
        )

    if updated_ids and embedder is not None:
        updated_stories = [db.get_story(sid) for sid in updated_ids]
        updated_stories = [s for s in updated_stories if s is not None]
        if updated_stories:
            get_or_compute_embeddings(updated_stories, embedder, db)

    return len(updated_ids)


def build_reddit_prewarm_factories(
    story_ids: list[int], db: Database
) -> tuple[list[CoroFactory], list[int]]:
    """Build Reddit prewarm coroutine factories for the given story IDs.

    Each factory fetches the story's per-post RSS feed, extracts
    self_text and top_comments, recomposes text_content, and writes
    back to the stories table. Factories are NO-OPs for stories that
    don't need prewarming (already-hydrated, non-Reddit, or no URL).

    Returns:
        (factories, updated_ids) where ``updated_ids`` is a shared list
        that each factory appends to as it successfully writes back to
        the DB. Read ``len(updated_ids)`` after the queue drains to get
        the prewarm count.

    The regen pipeline uses
    :func:`reddit_fetch_queue.enqueue_all_reddit_fetches` to interleave
    prewarm factories with topfeed factories on a single shared window,
    so the returned list is enqueued rather than awaited in-place.
    """
    if not story_ids:
        return [], []
    from server import _fetch_reddit_rss_context  # late import to avoid circular

    counter_lock = threading.Lock()
    updated_ids: list[int] = []

    def make_prewarm_factory(sid: int) -> CoroFactory:
        async def factory() -> None:
            # Cheap circuit-open short-circuit to avoid db.get_story and
            # the inner fetch path when the limiter has opened. The
            # inner `_fetch_reddit_rss_context → acquire` is the actual
            # rate-limit gate; calling `acquire()` here would reserve
            # a second slot per HTTP (acquire + _fetch_reddit_rss_context
            # both call it), wasting rate-limit budget under the
            # concurrent-reservation contract. See WORKLOG 2026-06-28
            # "Limiter concurrency race fix".
            if reddit_limiter.circuit_open:
                return
            story = db.get_story(sid)
            if not story or not story.url:
                return
            if not story.source.startswith("rss_reddit_"):
                return
            try:
                ctx = await _fetch_reddit_rss_context(story.url)
            except Exception as exc:
                logging.warning(
                    "prewarm_reddit: fetch failed for story_id=%s: %r",
                    sid,
                    exc,
                )
                return
            if ctx is None or not ctx.top_comments:
                return
            if story.top_comments and len(ctx.top_comments) <= len(story.top_comments):
                return
            new_self_text = (
                ctx.self_text
                if len(ctx.self_text) > len(story.self_text)
                else story.self_text
            )
            new_text_content = compose_story_text(
                story.title,
                new_self_text,
                ctx.top_comments,
                story.article_body or "",
            )
            if not new_text_content:
                return
            updated = replace(
                story,
                self_text=new_self_text,
                top_comments=ctx.top_comments,
                text_content=new_text_content,
                comment_count=story.comment_count or ctx.comment_count or None,
                comment_count_at_fetch=max(
                    story.comment_count_at_fetch, ctx.comment_count
                ),
                discussion_url=story.discussion_url or story.url,
            )
            db.upsert_story(updated)
            with counter_lock:
                updated_ids.append(updated.id)

        return factory

    return [make_prewarm_factory(sid) for sid in story_ids], updated_ids


async def prewarm_lesswrong_stories(
    story_ids: list[int],
    db: Database,
    embedder: Embedder | None = None,
) -> int:
    """Prewarm top_comments and self_text for LessWrong RSS stories.

    Fetches each post's body and top-voted comments via LessWrong's GraphQL
    endpoint (one request per story). Serialized to avoid rate limits.

    Returns number of stories whose top_comments or self_text changed.
    """
    if not story_ids:
        return 0
    from server import _extract_lesswrong_post_id, _fetch_lesswrong_context

    prewarmed: list[Story] = []
    for sid in story_ids:
        story = db.get_story(sid)
        if not story or not story.url:
            continue
        if story.source != "rss_lesswrong_com":
            continue

        post_id = _extract_lesswrong_post_id(story.url)
        if not post_id:
            continue

        try:
            ctx = await _fetch_lesswrong_context(post_id)
        except Exception as exc:
            logging.warning(
                "prewarm_lesswrong: fetch failed for story_id=%s: %r",
                sid,
                exc,
            )
            continue

        if ctx is None or not (ctx.self_text or ctx.top_comments):
            continue

        # Idempotent: skip only if BOTH top_comments and self_text are
        # already populated with equal or richer data. Otherwise, the new
        # data may have richer top_comments even when self_text is shorter
        # (e.g. RSS snippet was truncated to SELF_TEXT_PROMPT_CHAR_LIMIT
        # but the GraphQL body fits in fewer chars).
        top_comments_fresh = not story.top_comments or len(ctx.top_comments) > len(
            story.top_comments
        )
        self_text_fresh = not story.self_text or len(ctx.self_text) > len(
            story.self_text
        )
        if not (top_comments_fresh or self_text_fresh):
            continue

        new_self_text = (
            ctx.self_text
            if len(ctx.self_text) > len(story.self_text)
            else story.self_text
        )
        new_text_content = compose_story_text(
            story.title,
            new_self_text,
            ctx.top_comments,
            story.article_body or "",
        )
        if not new_text_content:
            continue

        updated = replace(
            story,
            self_text=new_self_text,
            top_comments=ctx.top_comments,
            text_content=new_text_content,
            comment_count=story.comment_count or ctx.comment_count or None,
            comment_count_at_fetch=max(story.comment_count_at_fetch, ctx.comment_count),
            score=max(story.score, ctx.score),
            discussion_url=story.discussion_url or story.url,
        )
        db.upsert_story(updated)
        prewarmed.append(updated)

    if prewarmed and embedder is not None:
        get_or_compute_embeddings(prewarmed, embedder, db)

    return len(prewarmed)


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
def _reddit_subreddit_from_feed_url(feed_url: str) -> str | None:
    parsed = urlparse(feed_url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if domain not in {"reddit.com", "old.reddit.com"}:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0].lower() == "r":
        return re.sub(r"[^a-z0-9_]+", "_", parts[1].lower()).strip("_")
    return None


def _rss_source_name(feed_url: str) -> str:
    subreddit = _reddit_subreddit_from_feed_url(feed_url)
    if subreddit:
        return f"rss_reddit_{subreddit}"

    domain = urlparse(feed_url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    for prefix in ("rss.", "feeds.", "feed."):
        if domain.startswith(prefix):
            domain = domain[len(prefix) :]
    return f"rss_{domain.replace('.', '_')}"


def _urllib_fetch(url: str, user_agent: str) -> tuple[int, str]:
    """Sync fetch via urllib.
    Re-exported from http_fetch for backward compatibility with
    callers that import it from pipeline (e.g. server.py). Prefer
    `http_fetch.urllib_fetch` for new code."""
    from http_fetch import urllib_fetch as _impl

    return _impl(url, user_agent)


def _parse_rate_limit_reset(headers: dict[str, str]) -> float | None:
    """Extract Reddit's x-ratelimit-reset value (seconds) from response headers.

    Returns None if the header is missing or unparseable.
    """
    raw = headers.get("x-ratelimit-reset")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


async def _fetch_and_parse_feed(
    feed_url: str,
    per_feed: int,
    cutoff: float,
    now: float,
    exclude_urls: set[str],
) -> list[Story]:
    """Fetch a single RSS feed (Reddit or non-Reddit) and parse to Stories.

    Returns an empty list on any failure. Used by both ``fetch_rss_feeds``
    (non-Reddit sync) and the Reddit topfeed factories. Reddit responses
    notify the shared ``reddit_limiter`` on 429 (with
    ``x-ratelimit-reset``) and on 200.
    """
    is_reddit = bool(_reddit_subreddit_from_feed_url(feed_url))
    try:
        from http_fetch import fetch_with_urllib_fallback

        source_name = _rss_source_name(feed_url)
        headers = {"User-Agent": RSS_USER_AGENT}
        if is_reddit:
            headers["User-Agent"] = REDDIT_RSS_USER_AGENT

        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            status, content, resp_headers = await fetch_with_urllib_fallback(
                client, feed_url, headers
            )
            if is_reddit:
                if status == 429:
                    rl_reset = _parse_rate_limit_reset(resp_headers)
                    reddit_limiter.on_429(rate_limit_reset=rl_reset)
                elif status == 200:
                    reddit_limiter.on_success()
            if status != 200:
                return []

        parsed = feedparser.parse(content)
        stories: list[Story] = []

        for entry in parsed.entries[:per_feed]:
            link = entry.get("link")
            if not link:
                continue
            if link in exclude_urls:
                continue

            published_parsed = entry.get("published_parsed")
            if published_parsed:
                pub_time = time.mktime(published_parsed)
            else:
                updated_parsed = entry.get("updated_parsed")
                if updated_parsed:
                    pub_time = time.mktime(updated_parsed)
                else:
                    pub_time = now

            if pub_time < cutoff:
                continue

            title = entry.get("title", "Untitled")

            summary = ""
            if "content" in entry and entry.content:
                summary = entry.content[0].value
            elif "summary" in entry:
                summary = entry.summary

            clean_summary = clean_text(summary)
            snippet = clean_summary[:1000]
            self_text = snippet
            text_content = compose_story_text(title, self_text)

            # RSS <comments> element — Tildes and Lobsters provide a
            # separate discussion URL. Reddit, LessWrong, and personal
            # blogs don't have one (entry.comments is None).
            comments_url = entry.get("comments")

            h = hashlib.md5(link.encode("utf-8")).digest()
            val = int.from_bytes(h[:4], "big")
            synthetic_id = -(val % (2**31))

            # Reddit's topfeed RSS does not include <score> or
            # <num_comments> elements in the entry body (confirmed
            # 2026-06-29 by saving a real r/MachineLearning/top/.rss body
            # and grepping for those names; zero matches across the
            # entire XML). The Reddit JSON API carries these but is
            # blocked for unauthenticated access (403 with HTML block
            # page on every sub we tested). So score and num_comments
            # are hardcoded to 0 here; the on-demand per-post RSS
            # path in server.py:247 populates them when a user opens
            # a card and we fetch the comments thread.
            score = 0
            num_comments = 0

            story = Story(
                id=synthetic_id,
                title=title,
                url=link,
                score=score,
                time=int(pub_time),
                text_content=text_content,
                self_text=self_text,
                source=source_name,
                comment_count=num_comments,
                comment_count_at_fetch=num_comments,
                discussion_url=comments_url,
            )
            stories.append(story)

        return stories
    except Exception as e:
        logging.error("Failed to fetch RSS feed %s: %r", feed_url, e)
        return []


def build_reddit_topfeed_factories(
    feeds: list[str],
    per_feed: int,
    days: int,
    exclude_urls: set[str],
) -> tuple[list[CoroFactory], list[str]]:
    """Build Reddit topfeed coroutine factories + their feed URLs.

    Each factory checks ``reddit_feed_cache`` first (skip if hit),
    acquires the shared ``reddit_limiter``, fetches via
    :func:`_fetch_and_parse_feed`, and writes to ``reddit_feed_cache``.

    Returns ``(factories, reddit_feed_urls)``. Factories are NOT enqueued
    here; the regen pipeline uses
    :func:`reddit_fetch_queue.enqueue_all_reddit_fetches` to interleave
    them with prewarm factories on a single shared window. Test callers
    can enqueue via the regular ``enqueue_spread`` path.

    The ``days``, ``per_feed``, and ``exclude_urls`` arguments are
    captured by the closures so each factory has the same filtering
    behavior as the legacy in-line ``fetch_rss_feeds`` path.
    """
    reddit_feeds = [f for f in feeds if _reddit_subreddit_from_feed_url(f)]
    if not reddit_feeds:
        return [], []
    now = time.time()
    cutoff = now - (days * 86400)

    def make_topfeed_factory(feed_url: str) -> CoroFactory:
        async def factory() -> None:
            cached = reddit_feed_cache.get(feed_url)
            if cached is not None:
                return
            if not await reddit_limiter.acquire():
                return
            stories = await _fetch_and_parse_feed(
                feed_url, per_feed, cutoff, now, exclude_urls
            )
            if stories:
                reddit_feed_cache.set(feed_url, stories)

        return factory

    return [make_topfeed_factory(f) for f in reddit_feeds], reddit_feeds


async def fetch_rss_feeds(
    feeds: list[str],
    per_feed: int,
    days: int,
    exclude_urls: set[str],
    db: Database,
) -> list[Story]:
    """Fetch non-Reddit RSS feeds synchronously and upsert to DB.

    Reddit RSS feeds are NOT fetched here — they are rate-limited and
    must be enqueued via :func:`build_reddit_topfeed_factories` plus
    :func:`reddit_fetch_queue.enqueue_all_reddit_fetches` by the regen
    pipeline. Calling this function with a feed list that includes
    Reddit URLs is safe: they are filtered out.

    Returns the list of Stories upserted (non-Reddit only).
    """
    now = time.time()
    cutoff = now - (days * 86400)

    other_feeds = [f for f in feeds if not _reddit_subreddit_from_feed_url(f)]
    tasks = [
        _fetch_and_parse_feed(f, per_feed, cutoff, now, exclude_urls)
        for f in other_feeds
    ]
    feed_results = list(await asyncio.gather(*tasks)) if tasks else []

    all_stories: list[Story] = []
    for res in feed_results:
        for s in res:
            db.upsert_story(s)
            all_stories.append(s)

    return all_stories


# Embedder


# HTML Render
def time_ago_filter(seconds: int) -> str:
    diff = int(time.time()) - seconds
    if diff < 0:
        return "now"
    if diff < 60:
        return f"{diff}s ago"
    minutes = diff // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def source_label_filter(source: str) -> str:
    if not source:
        return ""
    if source == "hn":
        return "HN"
    if source == BQ_ARCHIVE_SOURCE:
        return "BQ Seed"
    if source == CH_ARCHIVE_SOURCE:
        return "CH Seed"

    label = source
    if label.startswith("rss_"):
        label = label[4:]
    # Historical rows from feeds hosted at rss.* were stored as rss_rss_*.
    if label.startswith("rss_"):
        label = label[4:]
    if label.startswith("reddit_"):
        subreddit = label[len("reddit_") :]
        return f"r/{subreddit}"

    known = {
        "slashdot_org": "Slashdot",
        "mshibanami_github_io": "GitHub Trending",
        "tildes_net": "Tildes",
        "lesswrong_com": "LessWrong",
        "lobste_rs": "Lobsters",
        "discourse_haskell_org": "Haskell Discourse",
        "latent_space": "Latent Space",
        "scottaaronson_blog": "Scott Aaronson",
        "simonwillison_net": "Simon Willison",
        "lwn_net": "LWN",
        "openai_com": "OpenAI",
        "huggingface_co": "Hugging Face",
        "blog_cloudflare_com": "Cloudflare",
        "blog_janestreet_com": "Jane Street",
        "well-typed_com": "Well-Typed",
        "tweag_io": "Tweag",
        "ocaml_org": "OCaml",
        "quantamagazine_org": "Quanta",
        "www_worksinprogress_news": "Works in Progress",
        "erictopol_substack_com": "Ground Truths",
        "theskepticalcardiologist_substack_com": "Skeptical Cardiologist",
        "sciencebasedmedicine_org": "Science-Based Medicine",
    }
    if label in known:
        return known[label]

    return label.replace("_", ".")


def _article_fetch_failure_active(
    db: Database,
    story_id: int,
    now_ts: float,
) -> bool:
    failure = db.get_article_fetch_failure(story_id)
    if not failure:
        return False
    return bool(failure["permanent"]) or float(failure["next_retry_at"]) > now_ts


def _article_fetch_extra_priority(
    item: RankedStory, position: int, now_ts: float
) -> float:
    story = item.story
    age_hours = max((now_ts - story.time) / 3600.0, 0.1)
    score_velocity = story.score / age_hours
    comment_velocity = (story.comment_count or 0) / age_hours
    return (
        (1_000_000.0 / (position + 1))
        + float(item.score) * 1_000.0
        + story.score * 2.0
        + (story.comment_count or 0) * 1.5
        + score_velocity * 50.0
        + comment_velocity * 50.0
    )


_PAYWALL_DOMAINS: frozenset[str] = frozenset(
    {
        "bloomberg.com",
        "economist.com",
        "ft.com",
        "nytimes.com",
        "wsj.com",
        "reuters.com",
        "axios.com",
    }
)


def _is_fetchable_article_url(url: str) -> bool:
    """Return False for URLs that will never yield extractable article text."""
    if not url:
        return False
    if url.startswith("https://news.ycombinator.com"):
        return False
    host = urlparse(url).hostname or ""
    if any(host == d or host.endswith("." + d) for d in _PAYWALL_DOMAINS):
        return False
    if "youtube.com/watch" in url or "youtu.be/" in url:
        return False
    return True


def select_article_fetch_candidates(
    *,
    ranked: list[RankedStory],
    dashboard_selected: list[RankedStory],
    db: Database,
    max_per_run: int,
    max_age_days: int = 30,
    now_ts: float | None = None,
) -> list[Story]:
    """Choose bounded article-body fetch targets.

    Dashboard-visible stories are ordered first. Remaining budget is filled
    from ranked extras by rank, score, comments, and engagement velocity.
    """
    if max_per_run <= 0:
        return []
    if now_ts is None:
        now_ts = time.time()
    min_time = now_ts - (max_age_days * 86400)

    def eligible(story: Story) -> bool:
        if not story.url or story.article_body:
            return False
        if not _is_fetchable_article_url(story.url):
            return False
        if (
            story.source.startswith("rss_reddit_")
            or story.source == "rss_lesswrong_com"
        ):
            return False
        if story.time < min_time:
            return False
        if _article_fetch_failure_active(db, story.id, now_ts):
            return False
        return True

    selected: list[Story] = []
    selected_ids: set[int] = set()
    for item in dashboard_selected:
        story = item.story
        if story.id in selected_ids or not eligible(story):
            continue
        selected.append(story)
        selected_ids.add(story.id)
        if len(selected) >= max_per_run:
            return selected

    extras: list[tuple[float, Story]] = []
    for position, item in enumerate(ranked):
        story = item.story
        if story.id in selected_ids or not eligible(story):
            continue
        extras.append((_article_fetch_extra_priority(item, position, now_ts), story))

    extras.sort(key=lambda pair: pair[0], reverse=True)
    for _priority, story in extras:
        selected.append(story)
        selected_ids.add(story.id)
        if len(selected) >= max_per_run:
            break
    return selected


def _article_failure_retry_time(failure_count: int, now_ts: float) -> float:
    delay_seconds = min(86400, 3600 * (2 ** max(0, failure_count - 1)))
    return now_ts + delay_seconds


async def fetch_and_cache_article_bodies(
    *,
    db: Database,
    embedder: Embedder,
    stories: list[Story],
    concurrency: int = 10,
) -> dict[int, Story]:
    if not stories:
        return {}

    logging.info(
        "article_fetch: fetching %d candidate bodies (concurrency=%d)",
        len(stories),
        concurrency,
    )

    from server import ARTICLE_BODY_CHAR_LIMIT, _fetch_article_body_with_result

    sem = asyncio.Semaphore(max(1, concurrency))
    model_version = "all-MiniLM-L6-v2|mean|norm|256"

    success = [0]
    error_counts: dict[str, int] = {}

    async def fetch_one(story: Story) -> tuple[int, Story | None]:
        async with sem:
            result = await _fetch_article_body_with_result(story.url or "")
            if result.body:
                body = result.body[:ARTICLE_BODY_CHAR_LIMIT]
                new_text = compose_story_text(
                    story.title,
                    story.self_text,
                    story.top_comments,
                    body,
                )
                updated = replace(story, article_body=body, text_content=new_text)
                db.upsert_story(updated)
                new_vec = embedder.encode([new_text])[0]
                new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
                db.upsert_embedding(story.id, model_version, new_hash, new_vec)
                db.clear_article_fetch_failure(story.id)
                success[0] += 1
                return story.id, updated

            now_ts = time.time()
            previous = db.get_article_fetch_failure(story.id)
            previous_count = int(previous["failure_count"]) if previous else 0
            failure_count = previous_count + 1
            permanent = (
                result.permanent
                or (result.error == "empty_extraction" and failure_count >= 3)
                or (
                    result.error is not None
                    and result.error in ("http_401", "http_403")
                    and failure_count >= 3
                )
            )
            next_retry_at = (
                now_ts + 3650 * 86400
                if permanent
                else _article_failure_retry_time(failure_count, now_ts)
            )
            db.record_article_fetch_failure(
                story.id,
                story.url or "",
                status=result.status,
                error=result.error,
                permanent=permanent,
                next_retry_at=next_retry_at,
            )
            error_key = result.error or (
                f"http_{result.status}" if result.status else "unknown"
            )
            error_counts[error_key] = error_counts.get(error_key, 0) + 1
            return story.id, None

    results = await asyncio.gather(*(fetch_one(story) for story in stories))
    error_summary = (
        " ".join(f"{k}={v}" for k, v in sorted(error_counts.items()))
        if error_counts
        else "(none)"
    )
    logging.info(
        "article_fetch: ok=%d failed=%d errors=[%s]",
        success[0],
        len(stories) - success[0],
        error_summary,
    )
    return {sid: updated for sid, updated in results if updated is not None}


_pico_css_cache: str | None = None


def _get_pico_css() -> str:
    global _pico_css_cache
    if _pico_css_cache is None:
        path = Path("templates/pico.min.css")
        _pico_css_cache = path.read_text(encoding="utf-8") if path.exists() else ""
    return _pico_css_cache


def _build_badges(
    item: RankedStory, *, hot_badge_percentile: int
) -> tuple[BadgeView, ...]:
    badges: list[BadgeView] = []
    if item.is_uncertain:
        badges.append(
            BadgeView(
                kind="uncertain",
                icon="🤔",
                label="Unsure",
                tooltip="Model is highly uncertain about this story (high entropy distribution)",
            )
        )
    if item.is_novel:
        badges.append(
            BadgeView(
                kind="novel",
                icon="✨",
                label="Novel",
                tooltip="Semantically distant from anything you've voted on",
            )
        )
    if item.is_discussion_rich:
        badges.append(
            BadgeView(
                kind="talk",
                icon="💬",
                label="Talk-worthy",
                tooltip="High HN comment count for its age cohort",
            )
        )
    if item.is_high_engagement:
        badges.append(
            BadgeView(
                kind="top",
                icon="🏆",
                label="Top",
                tooltip="High HN score for its age cohort",
            )
        )
    if item.is_hot:
        badges.append(
            BadgeView(
                kind="hot",
                icon="🔥",
                label="Hot",
                tooltip=(
                    f"Top {hot_badge_percentile}% by engagement velocity "
                    "(points/hour) and score ≥ 20"
                ),
            )
        )
    if item.is_similar:
        badges.append(
            BadgeView(
                kind="similar",
                icon="🎯",
                label="Similar",
                tooltip="Most similar to your upvoted stories for its age cohort",
            )
        )
    return tuple(badges)


def _build_dashboard_cards(
    ranked: list[RankedStory], *, hot_badge_percentile: int
) -> list[DashboardCardView]:
    cards: list[DashboardCardView] = []
    for item in ranked:
        story = item.story
        cards.append(
            DashboardCardView(
                story=story,
                score=item.score,
                best_match_title=item.best_match_title,
                badges=_build_badges(item, hot_badge_percentile=hot_badge_percentile),
                combo_keys=item.combo_keys,
                is_enriched=len(story.text_content) >= 1000,
                is_hn_attr="0" if item.is_non_hn else "1",
                sort_popular_attr=(
                    "1"
                    if item.is_hot or item.is_high_engagement or item.is_discussion_rich
                    else "0"
                ),
                sort_explore_attr=(
                    "1"
                    if item.is_uncertain or item.is_similar or item.is_novel
                    else "0"
                ),
                is_recent_attr="1" if item.is_recent else "0",
                article_url=story.url or "",
                comments_url=story.discussion_url or "",
                source_label=source_label_filter(story.source),
                time_ago=time_ago_filter(story.time),
                show_source_badge=story.source != "hn",
                show_score=story.score > 0,
            )
        )
    return cards


def _build_tab_groups() -> tuple[TabGroupView, ...]:
    return (
        TabGroupView(
            key="sort",
            aria_label="Sort order",
            css_class="tab-bar tab-bar--sort",
            data_attr="sort",
            segmented=False,
            tabs=(
                TabView("recommended", "<u>R</u>ecommended", True),
                TabView("popular", "<u>P</u>opular"),
                TabView("explore", "E<u>x</u>plore"),
                TabView("date", "<u>D</u>ate"),
            ),
        ),
        TabGroupView(
            key="age",
            aria_label="Age filter",
            css_class="tab-bar tab-bar--segmented",
            data_attr="age",
            segmented=True,
            tabs=(
                TabView("recent", "R<u>e</u>cent", True),
                TabView("archive", "<u>A</u>rchive"),
            ),
        ),
        TabGroupView(
            key="source",
            aria_label="Source filter",
            css_class="tab-bar tab-bar--segmented",
            data_attr="source",
            segmented=True,
            tabs=(
                TabView("mixed", "<u>M</u>ixed", True),
                TabView("hn", "<u>H</u>N"),
                TabView("non-hn", "<u>N</u>on-HN"),
            ),
        ),
    )


def generate_dashboard_bytes(
    ranked: list[RankedStory],
    config: Config,
    db: Database,
    user_id: int | None = None,
    user_token: str | None = None,
    dashboard_version: int | None = None,
    dashboard_latest_version: int | None = None,
) -> bytes:
    """Render dashboard to bytes without writing to disk."""
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["time_ago"] = time_ago_filter
    env.filters["source_label"] = source_label_filter

    pico_css = _get_pico_css()

    raw_vote_counts = (
        db.count_feedback_by_action(user_id)
        if user_id
        else {"up": 0, "neutral": 0, "down": 0}
    )
    vote_counts = VoteCountsView(
        up=raw_vote_counts["up"],
        neutral=raw_vote_counts["neutral"],
        down=raw_vote_counts["down"],
    )
    hot_badge_percentile = int(round(config.model.hot_badge_percentile))

    template = env.get_template("index.html")
    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        cards=_build_dashboard_cards(ranked, hot_badge_percentile=hot_badge_percentile),
        tab_groups=_build_tab_groups(),
        server_port=config.server_port,
        pico_css=pico_css,
        user_id=user_id,
        user_token=user_token,
        vote_counts=vote_counts,
        dashboard_version=dashboard_version or 0,
        dashboard_latest_version=dashboard_latest_version or 0,
    )
    return html_content.encode("utf-8")


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
