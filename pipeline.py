from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import threading
import time
import tomllib
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import feedparser
import httpx
import numpy as np
import onnxruntime as ort
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from database import Database, Story
from transformers import AutoTokenizer


RSS_USER_AGENT = "hn-rewrite/1.0 (+https://github.com/local/hn-rewrite)"
REDDIT_RSS_USER_AGENT = "hn-rewrite/1.0 personal RSS reader; contact: local dashboard"


@dataclass(frozen=True)
class ModelConfig:
    svm_c: float = 0.2
    svm_gamma: float | str = 0.03
    svm_kernel: str = "rbf"
    neutral_weight: float = 0.0
    enable_mmr: bool = False
    diversity_threshold: float = 0.75
    knn_k: int = 10
    positive_cluster_k: int = 4
    tier3_threshold: int = 20
    tier3_blend_window: int = 60
    min_up_for_svm: int = 20
    min_down_for_svm: int = 20


# SVM model cache: keyed on (user_id, feedback_signature) to skip SVC.fit().
_MODEL_CACHE: OrderedDict[tuple[int, str], tuple[SVC, StandardScaler]] = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()


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
        key = (user_id, signature)
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
            return _MODEL_CACHE[key]
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
        key = (user_id, signature)
        _MODEL_CACHE[key] = (svm, scaler)
        while len(_MODEL_CACHE) > maxsize:
            _MODEL_CACHE.popitem(last=False)


# Comment selection and depth tuning
COMMENT_DEPTH_PENALTY = 25  # Points a reply must overcome per nesting level
TOP_COMMENT_LIMIT = 40
TOP_COMMENT_CORE_THREADS = 4
TOP_COMMENT_REPLIES_PER_CORE_THREAD = 5
TOP_COMMENT_MAX_PER_THREAD = 6
GOOD_TOPLEVEL_MIN_LEN = 200
GOOD_TOPLEVEL_MIN_REPLIES = 3
TOP_COMMENT_TOP_LEVEL_BUDGET = TOP_COMMENT_LIMIT // 3
UNCERTAIN_DISCOVERY_SLOT_LIMIT = 5
DISCOVERY_SLOT_LIMIT = 5
POPULARITY_DISCOVERY_SLOT_LIMIT = 8
NON_HN_DISCOVERY_SLOT_LIMIT = 8
HOT_MIN_SCORE = 20
DASHBOARD_QUEUE_SIZE = 12
BQ_ARCHIVE_SOURCE = "bq_seed"
BQ_ARCHIVE_CANDIDATE_LIMIT = 2000
CH_ARCHIVE_SOURCE = "ch_seed"
CH_ARCHIVE_CANDIDATE_LIMIT = 2000
LIVE_WINDOW_LIMIT = 2000


def is_hn_source(source: str) -> bool:
    return source in {"hn", BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE}


def is_summarizable(story: Story) -> bool:
    """True if the story has enough text to generate a TLDR (or can fetch it).

    A story is summarizable if it already has any text content, or if it's
    an HN story with comments that can be fetched on-demand or prewarmed.
    """
    if story.self_text or story.top_comments or story.article_body:
        return True
    if is_hn_source(story.source):
        if (story.comment_count or 0) > 0:
            return True
        if (story.comment_count_at_fetch or 0) > 0:
            return True
    return False


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
    onnx_model_dir: str = "onnx_model"
    server_port: int = 8765
    regen_interval_seconds: int = 10800
    regen_prewarm_top_n: int = 50
    reddit_prewarm_top_n: int = 20
    prewarm_hn_full: bool = True
    prewarm_reddit_full: bool = True
    article_fetch_max_per_run: int = 100
    article_fetch_concurrency: int = 10
    article_fetch_max_age_days: int = 30
    max_cached_models: int = 20
    model: ModelConfig = field(default_factory=ModelConfig)
    rss: RssConfig = field(default_factory=RssConfig)

    @classmethod
    def load(cls, path: str = "config.toml") -> Config:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            return cls()

        main_cfg = data.get("hn_rewrite", {})
        model_cfg = main_cfg.get("model", {})
        rss_cfg = main_cfg.get("rss", {})

        return cls(
            db_path=main_cfg.get("db_path", "hn_rewrite.db"),
            days=main_cfg.get("days", 30),
            count=main_cfg.get("count", 40),
            onnx_model_dir=main_cfg.get("onnx_model_dir", "onnx_model"),
            server_port=main_cfg.get("server_port", 8765),
            regen_interval_seconds=main_cfg.get("regen_interval_seconds", 10800),
            regen_prewarm_top_n=main_cfg.get("regen_prewarm_top_n", 50),
            reddit_prewarm_top_n=main_cfg.get("reddit_prewarm_top_n", 20),
            prewarm_hn_full=main_cfg.get("prewarm_hn_full", True),
            prewarm_reddit_full=main_cfg.get("prewarm_reddit_full", True),
            article_fetch_max_per_run=main_cfg.get("article_fetch_max_per_run", 100),
            article_fetch_concurrency=main_cfg.get("article_fetch_concurrency", 10),
            article_fetch_max_age_days=main_cfg.get("article_fetch_max_age_days", 30),
            max_cached_models=main_cfg.get("max_cached_models", 20),
            model=ModelConfig(
                svm_c=model_cfg.get("svm_c", 0.2),
                svm_gamma=model_cfg.get("svm_gamma", 0.03),
                svm_kernel=model_cfg.get("svm_kernel", "rbf"),
                neutral_weight=model_cfg.get("neutral_weight", 0.0),
                enable_mmr=model_cfg.get("enable_mmr", False),
                diversity_threshold=model_cfg.get("diversity_threshold", 0.75),
                knn_k=model_cfg.get("knn_k", 10),
                positive_cluster_k=model_cfg.get("positive_cluster_k", 4),
                tier3_threshold=model_cfg.get("tier3_threshold", 20),
                tier3_blend_window=model_cfg.get("tier3_blend_window", 60),
                min_up_for_svm=model_cfg.get("min_up_for_svm", 20),
                min_down_for_svm=model_cfg.get("min_down_for_svm", 20),
            ),
            rss=RssConfig(
                enabled=rss_cfg.get("enabled", True),
                per_feed_limit=rss_cfg.get("per_feed_limit", 70),
                feeds=tuple(rss_cfg.get("feeds", [])),
            ),
        )


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


def _dashboard_primary_limit(config_count: int) -> tuple[int, int]:
    num_uncertain = UNCERTAIN_DISCOVERY_SLOT_LIMIT if config_count >= 10 else 0
    primary_limit = min(max(1, config_count), DASHBOARD_QUEUE_SIZE)
    return primary_limit, num_uncertain


# Text processing helpers
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
        logging.error(f"Error fetching story {sid}: {e}")
        return story if story else None


async def fetch_stories_by_id(
    ids: list[int], db: Database, client: httpx.AsyncClient | None = None
) -> list[Story]:
    if not ids:
        return []

    stories = db.get_stories(ids)
    valid_stories: list[Story] = []
    found_ids: set[int] = set()
    for s in stories:
        if s.text_content == "":
            continue
        valid_stories.append(s)
        comments_fresh = s.top_comments != "" and (s.comment_count or 0) <= (
            s.comment_count_at_fetch or 0
        )
        if comments_fresh:
            found_ids.add(s.id)
    missing_ids = [sid for sid in ids if sid not in found_ids]

    # Corrupted stories first, then cap stale-comment refetches at 100
    if missing_ids:
        db_ids = {s.id for s in stories}
        story_map = {s.id: s for s in stories}
        corrupted_refetch = [
            sid
            for sid in missing_ids
            if sid in db_ids
            and story_map[sid].title == ""
            and story_map[sid].text_content != ""
        ]
        stale_refetch = sorted(
            [
                sid
                for sid in missing_ids
                if sid in db_ids and sid not in corrupted_refetch
            ],
            reverse=True,
        )[:100]
        new_ids = [sid for sid in missing_ids if sid not in db_ids]
        missing_ids = corrupted_refetch + stale_refetch + new_ids

    if not missing_ids:
        return valid_stories

    created_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)
        created_client = True

    try:
        sem = asyncio.Semaphore(10)

        async def _fetch_and_cache(sid: int) -> Story | None:
            async with sem:
                return await fetch_story(client, sid, db)

        tasks = [_fetch_and_cache(sid) for sid in missing_ids]
        fetched = await asyncio.gather(*tasks)

        for s in fetched:
            if s:
                valid_stories.append(s)
    finally:
        if created_client:
            await client.aclose()

    return valid_stories


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

    def _coerce_int_safe(value, default):
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

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
        comment_count = _coerce_int_safe(
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
) -> int:
    """Bulk-prewarm Reddit RSS comment text for top-N stories.

    Fetches each post's RSS feed, extracts self_text and top_comments,
    recomposes text_content, and writes back to the stories table.
    Serialized (one at a time) to avoid Reddit 429 rate limits.

    Returns number of stories whose top_comments or self_text changed.
    """
    if not story_ids:
        return 0
    from server import _fetch_reddit_rss_context  # late import to avoid circular

    prewarmed: list[Story] = []
    for sid in story_ids:
        story = db.get_story(sid)
        if not story or not story.url:
            continue
        if not story.source.startswith("rss_reddit_"):
            continue

        try:
            ctx = await _fetch_reddit_rss_context(story.url)
        except Exception as exc:
            logging.warning(
                "prewarm_reddit: fetch failed for story_id=%s: %r",
                sid,
                exc,
            )
            continue

        if ctx is None or not ctx.top_comments:
            continue

        # Idempotent: skip if already populated with equal or richer data
        if story.top_comments and len(ctx.top_comments) <= len(story.top_comments):
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

    `ch_client.query_live_window(days=7, min_score=5, limit=2000)` returns
    every live HN story from the past 7 days with all fields populated
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
            days=7,
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

    # 4. Dedup by URL
    deduped_candidates = []
    seen_urls = set()
    for s in candidates:
        if s.url:
            if s.url in exclude_urls or s.url in seen_urls:
                continue
            seen_urls.add(s.url)
        deduped_candidates.append(s)

    # 5. RSS feeds
    if config.rss.enabled:
        rss_stories = await fetch_rss_feeds(
            feeds=list(config.rss.feeds),
            per_feed=config.rss.per_feed_limit,
            days=config.days,
            exclude_urls=exclude_urls | seen_urls,
            db=db,
        )
        deduped_candidates.extend(rss_stories)

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
    Used as fallback when httpx is blocked by Cloudflare (different TLS
    fingerprint — system OpenSSL handshake often gets through)."""
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=15) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


async def fetch_rss_feeds(
    feeds: list[str],
    per_feed: int,
    days: int,
    exclude_urls: set[str],
    db: Database,
) -> list[Story]:
    now = time.time()
    cutoff = now - (days * 86400)

    async def fetch_and_parse(feed_url: str) -> list[Story]:
        try:
            source_name = _rss_source_name(feed_url)
            headers = {"User-Agent": RSS_USER_AGENT}
            if _reddit_subreddit_from_feed_url(feed_url):
                headers["User-Agent"] = REDDIT_RSS_USER_AGENT

            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                resp = await client.get(feed_url, headers=headers)
                content: str | None = None
                if resp.status_code == 200:
                    content = resp.text
                elif resp.status_code in (403, 503):
                    logging.info(
                        "RSS %s: httpx %d, retrying with urllib",
                        feed_url,
                        resp.status_code,
                    )
                    status, body = await asyncio.to_thread(
                        _urllib_fetch, feed_url, headers["User-Agent"]
                    )
                    if status == 200:
                        content = body
                    else:
                        logging.warning(
                            "RSS %s: urllib fallback returned %d",
                            feed_url,
                            status,
                        )
                if content is None:
                    return []

            parsed = feedparser.parse(content)
            stories = []

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

                h = hashlib.md5(link.encode("utf-8")).digest()
                val = int.from_bytes(h[:4], "big")
                synthetic_id = -(val % (2**31))

                story = Story(
                    id=synthetic_id,
                    title=title,
                    url=link,
                    score=0,
                    time=int(pub_time),
                    text_content=text_content,
                    self_text=self_text,
                    source=source_name,
                    comment_count=None,
                    discussion_url=None,
                )
                stories.append(story)

            return stories
        except Exception as e:
            logging.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []

    reddit_feeds = [f for f in feeds if _reddit_subreddit_from_feed_url(f)]
    other_feeds = [f for f in feeds if not _reddit_subreddit_from_feed_url(f)]

    tasks = [fetch_and_parse(f) for f in other_feeds]
    feed_results = list(await asyncio.gather(*tasks)) if tasks else []

    # Reddit RSS frequently returns 429 when several subreddit feeds are fetched in
    # parallel from the same host, so keep those requests serialized.
    for feed in reddit_feeds:
        feed_results.append(await fetch_and_parse(feed))

    all_stories = []
    for res in feed_results:
        for s in res:
            db.upsert_story(s)
            all_stories.append(s)

    return all_stories


# Embedder


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

# Normalization constants for metadata features
_LOG_POINTS_SCALE = 8.0  # log1p(~3000) ≈ 8
_AGE_DAYS_SCALE = 30.0  # cap at 30 days
_LOG_COMMENTS_SCALE = 7.0  # log1p(~1000) ≈ 6.9
_LOG_TEXTLEN_SCALE = 12.0  # log1p(~100000) ≈ 11.5
_LOG_QUALITY_SCALE = 8.0  # log1p(score/(age_hours+1)) rarely exceeds 8
_LOG_VELOCITY_SCALE = 8.0  # log1p(score/max(age_h, 0.1)) rarely exceeds 8


def _knn_similarity(
    query_emb: NDArray[np.float32], ref_emb: NDArray[np.float32], k: int
) -> NDArray[np.float32]:
    """Mean of top-k cosine similarities between query and reference embeddings."""
    if ref_emb.shape[0] == 0:
        return np.zeros(query_emb.shape[0], dtype=np.float32)
    sim_mat = query_emb @ ref_emb.T
    k_actual = min(k, ref_emb.shape[0])
    if k_actual <= 0:
        return np.zeros(query_emb.shape[0], dtype=np.float32)
    if k_actual == sim_mat.shape[1]:
        topk = sim_mat
    else:
        topk = np.partition(sim_mat, sim_mat.shape[1] - k_actual, axis=1)[:, -k_actual:]
    return topk.mean(axis=1).astype(np.float32)


def _topk_mean(values: NDArray[np.float32], k: int) -> float:
    if k <= 0 or len(values) == 0:
        return 0.0
    k_actual = min(k, len(values))
    if k_actual == len(values):
        return float(values.mean())
    return float(np.partition(values, len(values) - k_actual)[-k_actual:].mean())


def _augment_features(
    embeddings: NDArray[np.float32],
    scores: list[int] | np.ndarray,
    age_seconds: list[float] | np.ndarray,
    comment_counts: np.ndarray | None = None,
    text_lengths: np.ndarray | None = None,
    hn_quality: np.ndarray | None = None,
    score_velocity: np.ndarray | None = None,
    comment_velocity: np.ndarray | None = None,
    sim_to_upvoted: np.ndarray | None = None,
    sim_to_downvoted: np.ndarray | None = None,
    closest_upvoted: np.ndarray | None = None,
    closest_downvoted: np.ndarray | None = None,
    comment_score_ratio: np.ndarray | None = None,
    is_hn: np.ndarray | None = None,
    engagement_ratio: np.ndarray | None = None,
) -> NDArray[np.float32]:
    n = len(scores)
    n_meta = 1
    for f in (
        comment_counts,
        text_lengths,
        hn_quality,
        comment_score_ratio,
        is_hn,
        engagement_ratio,
    ):
        if f is not None:
            n_meta += 1
    if score_velocity is not None and comment_velocity is not None:
        n_meta += 2
    if sim_to_upvoted is not None:
        n_meta += 4

    meta = np.zeros((n, n_meta), dtype=np.float32)
    col = 0

    # log points
    meta[:, col] = (
        np.clip(np.log1p(np.maximum(scores, 0)), 0, _LOG_POINTS_SCALE)
        / _LOG_POINTS_SCALE
    )
    col += 1

    if comment_counts is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(comment_counts, 0)), 0, _LOG_COMMENTS_SCALE)
            / _LOG_COMMENTS_SCALE
        )
        col += 1

    if text_lengths is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(text_lengths, 0)), 0, _LOG_TEXTLEN_SCALE)
            / _LOG_TEXTLEN_SCALE
        )
        col += 1

    if hn_quality is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(hn_quality, 0)), 0, _LOG_QUALITY_SCALE)
            / _LOG_QUALITY_SCALE
        )
        col += 1

    if comment_score_ratio is not None:
        meta[:, col] = comment_score_ratio
        col += 1

    if score_velocity is not None and comment_velocity is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(score_velocity, 0)), 0, _LOG_VELOCITY_SCALE)
            / _LOG_VELOCITY_SCALE
        )
        col += 1
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(comment_velocity, 0)), 0, _LOG_VELOCITY_SCALE)
            / _LOG_VELOCITY_SCALE
        )
        col += 1

    if sim_to_upvoted is not None:
        assert (
            sim_to_downvoted is not None
            and closest_upvoted is not None
            and closest_downvoted is not None
        )
        meta[:, col] = (np.clip(sim_to_upvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(sim_to_downvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(closest_upvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(closest_downvoted, -1, 1) + 1) / 2
        col += 1

    if is_hn is not None:
        meta[:, col] = is_hn
        col += 1

    if engagement_ratio is not None:
        meta[:, col] = engagement_ratio
        col += 1

    return np.concatenate([embeddings, meta], axis=1)


def _svm_personalization_features(
    embeddings: NDArray[np.float32],
    text_lengths: np.ndarray,
    sim_to_upvoted: np.ndarray,
    sim_to_downvoted: np.ndarray,
    closest_upvoted: np.ndarray,
    closest_downvoted: np.ndarray,
    positive_cluster_similarity: np.ndarray | None = None,
) -> NDArray[np.float32]:
    """Production SVM features: embeddings, text length, and feedback similarity."""
    meta = np.zeros((len(embeddings), 6), dtype=np.float32)
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


def rank_stories(
    candidates: list[Story],
    candidate_embeddings: NDArray[np.float32],
    db: Database,
    config: Config,
    embedder: Embedder,
    user_id: int | None = None,
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

    # Multiclass SVM: 0=down, 1=neutral, 2=up
    unique_classes = set(feedback_labels)
    fb_labels_arr = np.array(feedback_labels)
    n_up = int((fb_labels_arr == 2).sum())
    n_down = int((fb_labels_arr == 0).sum())

    if (
        n_up >= config.model.min_up_for_svm
        and n_down >= config.model.min_down_for_svm
        and len(unique_classes) >= 2
    ):
        try:
            fb_embeddings = get_or_compute_embeddings(feedback_stories, embedder, db)

            # Personalization: mean/closest per class from ALL real feedback
            fb_labels_arr = np.array(feedback_labels)
            up_mask = fb_labels_arr == 2
            down_mask = fb_labels_arr == 0
            fb_up_embs = fb_embeddings[up_mask]
            fb_down_embs = fb_embeddings[down_mask]

            n_up = int(up_mask.sum())
            n_down = int(down_mask.sum())
            k = config.model.knn_k

            # k-NN similarity (mean of top-k similarities to class)
            cand_sim_to_up = _knn_similarity(candidate_embeddings, fb_up_embs, k)
            cand_sim_to_down = _knn_similarity(candidate_embeddings, fb_down_embs, k)

            # LOOCV k-NN for training: exclude self from reference set
            fb_sim_to_up = np.zeros(len(fb_embeddings), dtype=np.float32)
            fb_sim_to_down = np.zeros(len(fb_embeddings), dtype=np.float32)
            if n_up > 0:
                up_indices = np.where(up_mask)[0]
                sim_up_mat = fb_embeddings @ fb_up_embs.T
                if n_up > 1:
                    for idx, tp in enumerate(up_indices):
                        sim_up_mat[tp, idx] = -2.0  # exclude self
                k_eff_up = min(k, n_up)
                for i in range(len(fb_embeddings)):
                    sims = sim_up_mat[i]
                    exclude = 1 if i in up_indices else 0
                    n_available = max(1, n_up - exclude)
                    k_use = min(k_eff_up, n_available)
                    fb_sim_to_up[i] = _topk_mean(sims, k_use)
                sim_up_mat_clean = fb_embeddings @ fb_up_embs.T
                if n_up > 1:
                    for idx, tp in enumerate(up_indices):
                        sim_up_mat_clean[tp, idx] = -1.0
                fb_closest_up = np.max(sim_up_mat_clean, axis=1)
            else:
                fb_closest_up = np.zeros(len(fb_embeddings), dtype=np.float32)

            if n_down > 0:
                down_indices = np.where(down_mask)[0]
                sim_down_mat = fb_embeddings @ fb_down_embs.T
                if n_down > 1:
                    for idx, tp in enumerate(down_indices):
                        sim_down_mat[tp, idx] = -2.0
                k_eff_down = min(k, n_down)
                for i in range(len(fb_embeddings)):
                    sims = sim_down_mat[i]
                    exclude = 1 if i in down_indices else 0
                    n_available = max(1, n_down - exclude)
                    k_use = min(k_eff_down, n_available)
                    fb_sim_to_down[i] = _topk_mean(sims, k_use)
                sim_down_mat_clean = fb_embeddings @ fb_down_embs.T
                if n_down > 1:
                    for idx, tp in enumerate(down_indices):
                        sim_down_mat_clean[tp, idx] = -1.0
                fb_closest_down = np.max(sim_down_mat_clean, axis=1)
            else:
                fb_closest_down = np.zeros(len(fb_embeddings), dtype=np.float32)

            cand_closest_up = (
                np.max(candidate_embeddings @ fb_up_embs.T, axis=1)
                if up_mask.any()
                else np.zeros(len(candidates))
            )
            cand_closest_down = (
                np.max(candidate_embeddings @ fb_down_embs.T, axis=1)
                if down_mask.any()
                else np.zeros(len(candidates))
            )
            positive_cluster_centers = _positive_cluster_centers(
                fb_up_embs, config.model.positive_cluster_k
            )
            fb_positive_cluster_sim = _similarity_to_positive_cluster_centers(
                fb_embeddings, positive_cluster_centers
            )
            cand_positive_cluster_sim = _similarity_to_positive_cluster_centers(
                candidate_embeddings, positive_cluster_centers
            )

            fb_text_lengths = np.array([len(s.text_content) for s in feedback_stories])

            fb_features = _svm_personalization_features(
                fb_embeddings,
                text_lengths=fb_text_lengths,
                sim_to_upvoted=fb_sim_to_up,
                sim_to_downvoted=fb_sim_to_down,
                closest_upvoted=fb_closest_up,
                closest_downvoted=fb_closest_down,
                positive_cluster_similarity=fb_positive_cluster_sim,
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
            weights = [n_real / (n_classes * counts[lbl]) for lbl in feedback_labels]
            weights.extend([1e-6] * len(missing))
            sample_weights = np.array(weights, dtype=np.float64)

            emb_dim = candidate_embeddings.shape[1]

            # Model cache: skip SVC.fit() when feedback is unchanged
            fb_sig = _feedback_signature(db, user_id) if user_id is not None else ""
            cached_model: tuple[SVC, StandardScaler] | None = None
            if fb_sig:
                cached_model = _get_cached_model(user_id, fb_sig)
            if cached_model is not None:
                svm, scaler = cached_model
            else:
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
                    random_state=0,
                    decision_function_shape="ovr",
                )
                svm.fit(fb_features_scaled, labels, sample_weight=sample_weights)
                if fb_sig:
                    _set_cached_model(
                        user_id, fb_sig, svm, scaler, config.max_cached_models
                    )

            cand_text_lengths = np.array([len(s.text_content) for s in candidates])
            cand_features = _svm_personalization_features(
                candidate_embeddings,
                text_lengths=cand_text_lengths,
                sim_to_upvoted=cand_sim_to_up,
                sim_to_downvoted=cand_sim_to_down,
                closest_upvoted=cand_closest_up,
                closest_downvoted=cand_closest_down,
                positive_cluster_similarity=cand_positive_cluster_sim,
            )
            cand_features_meta_scaled = np.clip(
                scaler.transform(cand_features[:, emb_dim:]), -2.5, 2.5
            )
            cand_features_scaled = np.hstack(
                [cand_features[:, :emb_dim], cand_features_meta_scaled]
            )

            class_order = list(svm.classes_)
            idx_up = class_order.index(2)
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
            logging.error(f"Failed to fit feedback SVM: {e}")

    svm_scores = scores
    svm_probs = probs

    # Tier 2: centroid-based scores (always compute when feedback exists)
    tier2_scores: NDArray[np.float32] | None = None
    if n_feedback > 0:
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

    # Soft blend between tier 2 and SVM (tier 3)
    if svm_scores is not None and tier2_scores is not None:
        n_min = min(n_up, n_down)
        blend_start = min(config.model.min_up_for_svm, config.model.min_down_for_svm)
        alpha = float(
            np.clip(
                (n_min - blend_start) / config.model.tier3_blend_window,
                0,
                1,
            )
        )
        scores = np.asarray(
            (1 - alpha) * tier2_scores + alpha * svm_scores,
            dtype=np.float32,
        )
    elif tier2_scores is not None:
        scores = tier2_scores
    else:
        # Tier 1: No votes → HN gravity (frontpage-like)
        scores = np.array(
            [
                (s.score if is_hn_source(s.source) else s.score * 2)
                / max(((now - s.time) / 3600.0 + 2.0) ** 1.8, 0.1)
                for s in candidates
            ],
            dtype=np.float32,
        )
        if scores.max() > 0:
            scores = scores / scores.max()

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
            logging.error(f"Error mapping probability class indices: {e}")
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


def rerank_candidates(
    db: Database,
    config: Config,
    embedder: Embedder,
    candidates: list[Story],
    cand_embeddings: NDArray[np.float32] | None = None,
    user_id: int | None = None,
) -> list[RankedStory]:
    if not candidates:
        return []

    if cand_embeddings is None:
        cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    ranked = rank_stories(
        candidates,
        cand_embeddings,
        db,
        config,
        embedder,
        user_id=user_id,
    )

    limit, num_uncertain = _dashboard_primary_limit(config.count)
    if config.model.enable_mmr:
        embeddings_map = {s.id: vec for s, vec in zip(candidates, cand_embeddings)}
        final = mmr_filter(
            ranked,
            embeddings_map,
            threshold=config.model.diversity_threshold,
            limit=limit,
        )
    else:
        final = ranked[:limit]

    selected_ids = {item.story.id for item in final}
    remaining = [r for r in ranked if r.story.id not in selected_ids]

    # Calculate parameters for remaining discovery passes
    feedback_stories, feedback_labels, _ = db.get_feedback_for_training(user_id=user_id)
    fb_labels_arr = np.array(feedback_labels)
    up_mask = fb_labels_arr == 2
    down_mask = fb_labels_arr == 0
    neutral_mask = fb_labels_arr == 1
    fb_embeddings = get_or_compute_embeddings(feedback_stories, embedder, db)
    fb_up_embs = fb_embeddings[up_mask]
    fb_down_embs = fb_embeddings[down_mask]
    fb_neutral_embs = fb_embeddings[neutral_mask]

    cand_closest_up = (
        np.max(cand_embeddings @ fb_up_embs.T, axis=1)
        if up_mask.any()
        else np.zeros(len(candidates))
    )
    cand_closest_down = (
        np.max(cand_embeddings @ fb_down_embs.T, axis=1)
        if down_mask.any()
        else np.zeros(len(candidates))
    )
    cand_closest_neutral = (
        np.max(cand_embeddings @ fb_neutral_embs.T, axis=1)
        if neutral_mask.any()
        else np.zeros(len(candidates))
    )
    cand_max_sim = np.maximum.reduce(
        [cand_closest_up, cand_closest_down, cand_closest_neutral]
    )
    sim_threshold = np.percentile(cand_max_sim, 10) if len(cand_max_sim) else 0.0
    # Similar threshold is applied to extra-slot stories only (see
    # `is_similar` exclusion in the primary-attribution block below).
    similar_threshold = (
        np.percentile(cand_closest_up, 97) if len(cand_closest_up) else 0.0
    )

    cand_comment_counts = np.array([s.comment_count or 0 for s in candidates])
    cand_scores = np.array([s.score for s in candidates])
    now_ts = time.time()
    cand_velocities = np.array(
        [s.score / max((now_ts - s.time) / 3600.0, 0.1) for s in candidates]
    )
    hot_threshold = np.percentile(cand_velocities, 99.5) if len(cand_velocities) else 0

    story_id_to_idx = {s.id: idx for idx, s in enumerate(candidates)}

    # Determine uncertainty candidates among remaining candidates
    def get_entropy(r: RankedStory) -> float:
        ent = 0.0
        for p in (r.prob_down, r.prob_neutral, r.prob_up):
            if p is not None and p > 1e-9:
                ent -= p * np.log2(p)
        return ent

    if remaining and num_uncertain > 0:
        uncertain_candidates = [r for r in remaining if r.prob_down is not None]
        uncertain_candidates.sort(key=get_entropy, reverse=True)
        uncertain_candidates = uncertain_candidates[:num_uncertain]
    else:
        uncertain_candidates = []

    # Compute discovery thresholds. Badges are applied to any story that passes
    # these criteria (primary or extra-slot); the extra-slot passes below
    # additionally source from remaining_decorated to surface qualifying stories
    # outside the primary ranked set, respecting per-pass slot caps.
    # discussion_threshold uses 98th-pct of non-zero comments so that stories
    # with zero comments are filtered out cleanly by the >0 guard.
    nonzero_comments = cand_comment_counts[cand_comment_counts > 0]
    discussion_threshold = (
        np.percentile(nonzero_comments, 98) if len(nonzero_comments) else 0
    )
    engagement_threshold = np.percentile(cand_scores, 98) if len(cand_scores) else 0

    ranked_decorated = [
        replace(r, is_non_hn=(not is_hn_source(r.story.source))) for r in ranked
    ]

    # Reconstruct final from the undecorated ranked candidates
    final_ids = {item.story.id for item in final}
    final = [r for r in ranked_decorated if r.story.id in final_ids]

    # Apply the same badge criteria to primary-ranked stories. Extra-slot
    # passes below still source from remaining_decorated and respect their
    # per-pass slot caps; this attribution only adds badges to stories
    # already in the primary ranked set, so it does not take slots away
    # from the extra-slot pulls. `is_similar` is intentionally NOT applied
    # here: the Similar badge is reserved for extra-slot stories (see
    # similar_threshold usage in the Similar discovery pass below), so it
    # always signals "surfaced from outside primary because of high
    # semantic match" rather than a near-tautology on top-ranked stories.
    uncertain_entropy_threshold = (
        get_entropy(uncertain_candidates[-1]) if uncertain_candidates else float("inf")
    )
    final = [
        replace(
            r,
            is_uncertain=(
                r.prob_down is not None
                and get_entropy(r) >= uncertain_entropy_threshold
            ),
            is_novel=(cand_max_sim[story_id_to_idx[r.story.id]] <= sim_threshold),
            is_discussion_rich=(
                cand_comment_counts[story_id_to_idx[r.story.id]] >= discussion_threshold
                and cand_comment_counts[story_id_to_idx[r.story.id]] > 0
            ),
            is_high_engagement=(
                cand_scores[story_id_to_idx[r.story.id]] >= engagement_threshold
            ),
            is_hot=(
                cand_velocities[story_id_to_idx[r.story.id]] >= hot_threshold
                and cand_velocities[story_id_to_idx[r.story.id]] > 0
                and r.story.score >= HOT_MIN_SCORE
            ),
            is_non_hn=(not is_hn_source(r.story.source)),
        )
        for r in final
    ]

    # And remaining_decorated contains candidates not in the primary path.
    remaining_decorated = [r for r in ranked_decorated if r.story.id not in final_ids]

    # 1. Surface uncertainty items
    uncertain_ids = {r.story.id for r in uncertain_candidates}
    uncertain_items = [
        replace(r, is_uncertain=True)
        for r in remaining_decorated
        if r.story.id in uncertain_ids
    ]
    final.extend(uncertain_items)
    selected_ids = final_ids | {item.story.id for item in uncertain_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 2. Surface up to 5 novel stories
    novel_pool = [
        r
        for r in remaining_decorated
        if cand_max_sim[story_id_to_idx[r.story.id]] <= sim_threshold
    ]
    if novel_pool:
        novel_scores = np.array([r.score for r in novel_pool], dtype=np.float32)
        novel_distances = np.array(
            [1.0 - cand_max_sim[story_id_to_idx[r.story.id]] for r in novel_pool],
            dtype=np.float32,
        )
        score_pct = _rank_percentiles(novel_scores)
        novelty_pct = _rank_percentiles(novel_distances)
        novelty_rank = 0.7 * score_pct + 0.3 * novelty_pct
        novel_order = np.argsort(-novelty_rank, kind="mergesort")
        novel_pool = [novel_pool[int(i)] for i in novel_order]
    novel_items = [replace(r, is_novel=True) for r in novel_pool[:DISCOVERY_SLOT_LIMIT]]
    final.extend(novel_items)
    selected_ids |= {item.story.id for item in novel_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 3. Surface up to 5 most similar stories
    similar_pool = [
        r
        for r in remaining_decorated
        if cand_closest_up[story_id_to_idx[r.story.id]] >= similar_threshold
    ]
    similar_pool.sort(
        key=lambda r: cand_closest_up[story_id_to_idx[r.story.id]], reverse=True
    )
    similar_items = [
        replace(r, is_similar=True) for r in similar_pool[:DISCOVERY_SLOT_LIMIT]
    ]
    final.extend(similar_items)
    selected_ids |= {item.story.id for item in similar_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 4. Surface up to 5 discussion-rich stories
    discussion_pool = [
        r
        for r in remaining_decorated
        if cand_comment_counts[story_id_to_idx[r.story.id]] >= discussion_threshold
        and cand_comment_counts[story_id_to_idx[r.story.id]] > 0
    ]
    discussion_pool.sort(key=lambda r: r.story.comment_count or 0, reverse=True)
    discussion_items = [
        replace(r, is_discussion_rich=True)
        for r in discussion_pool[:DISCOVERY_SLOT_LIMIT]
    ]
    final.extend(discussion_items)
    selected_ids |= {item.story.id for item in discussion_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 5. Surface up to 8 high-engagement stories
    engagement_pool = [
        r
        for r in remaining_decorated
        if cand_scores[story_id_to_idx[r.story.id]] >= engagement_threshold
    ]
    engagement_pool.sort(key=lambda r: r.score, reverse=True)
    engagement_items = [
        replace(r, is_high_engagement=True)
        for r in engagement_pool[:POPULARITY_DISCOVERY_SLOT_LIMIT]
    ]
    final.extend(engagement_items)
    selected_ids |= {item.story.id for item in engagement_items}
    remaining_decorated = [
        r for r in remaining_decorated if r.story.id not in selected_ids
    ]

    # 6. Surface up to 8 hot stories
    hot_pool = [
        r
        for r in remaining_decorated
        if cand_velocities[story_id_to_idx[r.story.id]] >= hot_threshold
        and cand_velocities[story_id_to_idx[r.story.id]] > 0
        and r.story.score >= HOT_MIN_SCORE
    ]
    hot_pool.sort(
        key=lambda r: cand_velocities[story_id_to_idx[r.story.id]], reverse=True
    )
    hot_items = [
        replace(r, is_hot=True) for r in hot_pool[:POPULARITY_DISCOVERY_SLOT_LIMIT]
    ]
    final.extend(hot_items)
    selected_ids |= {item.story.id for item in hot_items}

    # 7. Surface up to 8 non-HN stories to balance the dashboard
    non_hn_pool = [r for r in remaining_decorated if not is_hn_source(r.story.source)]
    non_hn_pool.sort(key=lambda r: r.score, reverse=True)
    non_hn_items = [
        replace(r, is_non_hn=True) for r in non_hn_pool[:NON_HN_DISCOVERY_SLOT_LIMIT]
    ]
    final.extend(non_hn_items)
    selected_ids |= {item.story.id for item in non_hn_items}

    final.sort(key=lambda r: r.score, reverse=True)
    return final


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


def select_article_fetch_candidates(
    *,
    ranked: list[RankedStory],
    dashboard_selected: list[RankedStory],
    db: Database,
    max_per_run: int = 100,
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
        if story.source.startswith("rss_reddit_"):
            return False
        if story.url.startswith("https://news.ycombinator.com"):
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

    from server import ARTICLE_BODY_CHAR_LIMIT, _fetch_article_body_with_result

    sem = asyncio.Semaphore(max(1, concurrency))
    model_version = "all-MiniLM-L6-v2|mean|norm|256"

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
                return story.id, updated

            now_ts = time.time()
            previous = db.get_article_fetch_failure(story.id)
            previous_count = int(previous["failure_count"]) if previous else 0
            failure_count = previous_count + 1
            permanent = result.permanent or (
                result.error == "empty_extraction" and failure_count >= 3
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
            return story.id, None

    results = await asyncio.gather(*(fetch_one(story) for story in stories))
    return {sid: updated for sid, updated in results if updated is not None}


def generate_dashboard_bytes(
    ranked: list[RankedStory],
    config: Config,
    db: Database,
    user_id: int | None = None,
    user_token: str | None = None,
) -> bytes:
    """Render dashboard to bytes without writing to disk."""
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["time_ago"] = time_ago_filter
    env.filters["source_label"] = source_label_filter

    pico_css_path = Path("templates/pico.min.css")
    pico_css = (
        pico_css_path.read_text(encoding="utf-8") if pico_css_path.exists() else ""
    )

    all_fb = db.get_all_feedback(user_id=user_id)
    fb_map = {f.story_id: f.action for f in all_fb}

    template = env.get_template("index.html")
    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        stories=ranked,
        server_port=config.server_port,
        pico_css=pico_css,
        fb_map=fb_map,
        user_token=user_token,
    )
    return html_content.encode("utf-8")


def fast_rerank_for_user(
    db: Database,
    config: Config,
    embedder: Embedder,
    user_id: int,
) -> list[RankedStory]:
    """Fast rerank for a specific user. Called on each dashboard request."""
    now_ts = int(time.time())
    cutoff_ts = now_ts - (config.days * 86400)

    recent_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories WHERE time >= ? AND source NOT IN (?, ?) "
        "AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)",
        (cutoff_ts, BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE, user_id),
    )
    archive_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, self_text, top_comments, article_body "
        "FROM stories WHERE source IN (?, ?) AND text_content != '' "
        "AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?) "
        "ORDER BY score DESC, time DESC LIMIT ?",
        (
            BQ_ARCHIVE_SOURCE,
            CH_ARCHIVE_SOURCE,
            user_id,
            BQ_ARCHIVE_CANDIDATE_LIMIT + CH_ARCHIVE_CANDIDATE_LIMIT,
        ),
    )
    rows = recent_rows + archive_rows
    candidates = [Database._row_to_story(row) for row in rows]
    if not candidates:
        return []

    cand_embeddings = get_or_compute_embeddings(candidates, embedder, db)

    return rerank_candidates(
        db=db,
        config=config,
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=cand_embeddings,
        user_id=user_id,
    )


async def fetch_candidates_only(
    config: Config,
    db: Database,
    embedder: Embedder | None = None,
    prewarm_top_n: int | None = None,
) -> None:
    """Fetch new candidates into shared DB; prewarm top-N by score."""
    feedback_records = db.get_all_feedback()
    feedback_ids = {f.story_id for f in feedback_records}
    feedback_urls = {f.url for f in feedback_records if f.url}

    candidates, n_fetched = await fetch_candidates(
        config, feedback_ids, feedback_urls, db, None
    )
    logging.info(f"Regen: fetched {n_fetched} candidates")

    # HN prewarm
    if config.prewarm_hn_full and embedder is not None:
        needs_prewarm = [
            s.id
            for s in candidates
            if is_hn_source(s.source)
            and not s.top_comments
            and (s.comment_count or 0) > 0
        ]
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

    # Reddit prewarm
    if config.prewarm_reddit_full:
        needs_prewarm_reddit = [
            s.id
            for s in candidates
            if s.source.startswith("rss_reddit_") and not s.top_comments
        ]
        if needs_prewarm_reddit:
            prewarmed = await prewarm_reddit_top_stories(
                needs_prewarm_reddit, db, embedder
            )
            logging.info(
                "Regen: prewarmed %d/%d Reddit candidates (full mode)",
                prewarmed,
                len(needs_prewarm_reddit),
            )
    else:
        reddit_prewarm_top_n = config.reddit_prewarm_top_n
        if reddit_prewarm_top_n > 0:
            reddit_ids = [
                s.id for s in candidates if s.source.startswith("rss_reddit_")
            ]
            top_reddit = reddit_ids[:reddit_prewarm_top_n]
            if top_reddit:
                prewarmed = await prewarm_reddit_top_stories(top_reddit, db, embedder)
                logging.info(
                    "Regen: prewarmed %d/%d Reddit RSS stories",
                    prewarmed,
                    len(top_reddit),
                )
