from __future__ import annotations

import asyncio
import calendar
import hashlib
import html
import logging
import re
import threading
import time
from dataclasses import replace
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import feedparser
import httpx

import arctic_shift
from arctic_shift import ArcticPost, ArcticShiftError
from ch_client import DEFAULT_MAX_LEVELS
from database import Database, Story, StoryIdentityConflict, coerce_int

if TYPE_CHECKING:
    from ch_client import ChItem
from reddit_fetch_queue import CoroFactory
from reddit_feed_cache import cache as reddit_feed_cache
from reddit_limiter import limiter as reddit_limiter
from . import rank_gate
from .config import RedditSource
from .ainews import AINEWS_SOURCE, tweet_id_from_url
from .ranking import (
    Embedder,
    RankedStory,
    get_or_compute_embeddings,
    clean_text,
    compose_story_text,
    _extract_comments_recursive,
    _select_top_comments,
    join_top_comments,
    story_embedding_text,
)


RSS_USER_AGENT = "hn-rewrite/1.0 (+https://github.com/local/hn-rewrite)"
REDDIT_RSS_USER_AGENT = "hn-rewrite/1.0 personal RSS reader; contact: local dashboard"

# Matches server.py's SELF_TEXT_PROMPT_CHAR_LIMIT: no point retaining RSS
# content beyond what the TLDR prompt assembler will actually use.
RSS_SELF_TEXT_CHAR_LIMIT = 8_000
# LessWrong GraphQL during regen prewarm (tap-time fetches keep 15s).
LESSWRONG_PREWARM_TIMEOUT_S = 45.0


def _ch_story_item_to_story(item: ChItem) -> Story | None:
    """Convert a CH live-window item dict (Algolia shape) to a Story row.

    The CH live_window query returns the same fields as Algolia items
    (id, type, title, url, points, num_comments, created_at_i, text,
    children). For live `hn` source, we insert directly with the CH
    data; comment hydration is handled later by the bulk prewarm path.
    """
    sid = coerce_int(item.get("id"))
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
        score=coerce_int(item.get("points")),
        time=coerce_int(item.get("created_at_i")),
        text_content=text_content,
        source="hn",
        comment_count=coerce_int(item.get("num_comments")),
        discussion_url=f"https://news.ycombinator.com/item?id={sid}",
        comment_count_at_fetch=0,
        self_text=self_text,
        top_comments="",
        article_body="",
    )


def _empty_story(sid: int) -> Story:
    return Story(
        id=sid, title="", url=None, score=0, time=0, text_content="", source="hn"
    )


async def fetch_thread_comments(client: httpx.AsyncClient, sid: int) -> str:
    """Selected top comments of HN story *sid* from Algolia, without touching
    the DB; "" when the item is missing or the fetch fails."""
    try:
        resp = await client.get(f"https://hn.algolia.com/api/v1/items/{sid}")
        item = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError) as exc:
        logging.warning("thread comments fetch failed for %s: %r", sid, exc)
        return ""
    if not isinstance(item, dict) or item.get("type") != "story":
        return ""
    comments = _extract_comments_recursive(item.get("children", []))
    return join_top_comments([c["text"] for c in _select_top_comments(comments)])


async def fetch_story(
    client: httpx.AsyncClient, sid: int, db: Database, *, force: bool = False
) -> Story | None:
    """Fetch a single story from Algolia, refreshing the DB row.

    ``force=True`` bypasses the cached-row short-circuits below and always
    queries Algolia for the current comment tree — used by the tldr-detail
    on-demand refresh for recent, high-velocity HN threads where prewarm's
    CH data may already be 1-24h stale. ``force=False`` (default) keeps the
    original cheap staleness check so routine callers (e.g. the empty-
    top_comments lazy fetch) don't cause extra Algolia traffic.
    """
    story = db.get_story(sid)
    if story is not None:
        if story.text_content == "":
            if story.title == "":
                pass  # corrupted _empty_story, fall through to API re-fetch
            else:
                return None
        if not force:
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
        top_comment_texts = join_top_comments([c["text"] for c in selected])

        # Algolia carries no article body: preserve the stored one, or every
        # force-refresh (tap active-refresh today, regen growth-refresh next)
        # would wipe it and silently degrade the TLDR to discussion-only.
        existing_body = story.article_body if story is not None else ""
        text_content = compose_story_text(
            title=title,
            self_text=story_text,
            comments=top_comment_texts,
            article_body=existing_body,
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
            article_body=existing_body,
        )

        db.upsert_story(story, comments_authoritative=bool(top_comment_texts))
        return db.get_story(sid)
    except Exception as e:
        logging.error("Error fetching story %s: %r", sid, e)
        return story if story else None


# Stories per CH fetch chunk in prewarm_top_stories. Keeps a single failed
# chunk (network blip, CH overload) from zeroing out an entire prewarm run.
_PREWARM_CHUNK_SIZE = 100


_PREWARM_EMBED_BATCH = 32


def prewarm_top_stories(
    story_ids: list[int],
    db: Database,
    embedder: Embedder | None = None,
    max_levels: int = DEFAULT_MAX_LEVELS,
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
        max_levels: comment tree depth (``ch_client.DEFAULT_MAX_LEVELS``).

    Returns:
        Number of stories whose top_comments was updated.
    """
    from ch_client import is_transient_error, query_stories_with_comments

    if not story_ids:
        return 0

    target_ids = [int(s) for s in story_ids if int(s) > 0]
    if not target_ids:
        return 0

    # Chunk the CH fetch: one chunk failing (network blip, CH overload)
    # should cost that chunk, not the entire prewarm run.
    ch_items: dict[int, ChItem] = {}
    for i in range(0, len(target_ids), _PREWARM_CHUNK_SIZE):
        chunk = target_ids[i : i + _PREWARM_CHUNK_SIZE]
        try:
            ch_items.update(query_stories_with_comments(chunk, max_levels=max_levels))
        except Exception as exc:
            logging.warning(
                "prewarm_top_stories: CH bulk query failed for chunk of %d (%r)",
                len(chunk),
                exc,
            )
            if not is_transient_error(exc):
                break

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
        top_comments = join_top_comments([c["text"] for c in selected])
        comment_count = coerce_int(
            item.get("num_comments"), existing.comment_count or 0
        )
        if not top_comments:
            # Nothing usable (e.g. only one-word replies): mark the tree as
            # fetched so the regen prewarm waits for growth instead of
            # re-querying it every hour.
            if not existing.top_comments:
                db.upsert_story(
                    replace(
                        existing,
                        comment_count=comment_count,
                        comment_count_at_fetch=comment_count,
                    ),
                    comments_authoritative=True,
                )
            continue
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
        db.upsert_story(updated_story, comments_authoritative=True)
        persisted = db.get_story(sid)
        if persisted is not None:
            updated.append(persisted)

    if updated and embedder is not None:
        # Small batches so a rerank waiting on a vote is not stuck behind
        # a 1,000-story embedding run (regen thread only, never a rerank).
        for i in range(0, len(updated), _PREWARM_EMBED_BATCH):
            rank_gate.wait_idle()
            get_or_compute_embeddings(
                updated[i : i + _PREWARM_EMBED_BATCH], embedder, db
            )

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


class _SourceContext(Protocol):
    """Structural shape shared by RedditRssContext and LessWrongContext."""

    self_text: str
    top_comments: str
    comment_count: int


def _merge_source_context(
    story: Story,
    ctx: _SourceContext,
    article_body: str | None,
    *,
    prefer_longer_comments: bool,
) -> Story:
    """Merge a fetched source context (Reddit/LessWrong) into ``story``.

    ``article_body`` must already be the caller's best-known body string
    (on-demand callers pre-merge ``story.article_body or article_body``
    themselves; prewarm callers pass ``story.article_body`` directly) —
    this helper does not re-derive it.

    ``prefer_longer_comments`` — on-demand paths keep whichever of
    story/ctx top_comments is longer (True); prewarm callers have already
    established ctx is richer and take ctx.top_comments outright (False).

    ``score`` is merged via ``max`` only when ``ctx`` carries one
    (LessWrongContext); RedditRssContext has no ``score`` field, so it
    falls back to ``story.score`` and the max is a no-op.
    """
    self_text = (
        ctx.self_text if len(ctx.self_text) > len(story.self_text) else story.self_text
    )
    if prefer_longer_comments:
        top_comments = (
            ctx.top_comments
            if len(ctx.top_comments) > len(story.top_comments)
            else story.top_comments
        )
    else:
        top_comments = ctx.top_comments
    new_text = compose_story_text(
        story.title, self_text, top_comments, article_body or ""
    )
    return replace(
        story,
        self_text=self_text,
        top_comments=top_comments,
        text_content=new_text,
        discussion_url=story.discussion_url or story.url,
        comment_count=max(story.comment_count or 0, ctx.comment_count) or None,
        comment_count_at_fetch=max(story.comment_count_at_fetch, ctx.comment_count),
        score=max(story.score, getattr(ctx, "score", story.score)),
    )


def build_reddit_prewarm_factories(
    story_ids: list[int], db: Database, reddit_source: RedditSource = "rss"
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

    With ``reddit_source="arctic_shift"`` the thread comes from Arctic
    Shift instead of the per-post RSS feed.
    """
    if not story_ids:
        return [], []
    # late import to avoid circular
    from server import _fetch_reddit_arctic_context, _fetch_reddit_rss_context

    use_arctic = reddit_source == "arctic_shift"

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
            if not use_arctic and reddit_limiter.circuit_open:
                return
            story = db.get_story(sid)
            if not story or not story.url:
                return
            if not story.source.startswith("rss_reddit_"):
                return
            try:
                ctx = await (
                    _fetch_reddit_arctic_context(story.url)
                    if use_arctic
                    else _fetch_reddit_rss_context(story.url)
                )
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
            updated = _merge_source_context(
                story, ctx, story.article_body, prefer_longer_comments=False
            )
            if not updated.text_content:
                return
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
            # Background prewarm: a slow connect costs no user latency.
            ctx = await _fetch_lesswrong_context(
                post_id, timeout_s=LESSWRONG_PREWARM_TIMEOUT_S
            )
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
        metadata_fresh = (
            ctx.comment_count > (story.comment_count or 0)
            or ctx.comment_count > story.comment_count_at_fetch
            or ctx.score > story.score
        )
        if not (top_comments_fresh or self_text_fresh or metadata_fresh):
            continue

        updated = _merge_source_context(
            story, ctx, story.article_body, prefer_longer_comments=True
        )
        if not updated.text_content:
            continue

        db.upsert_story(updated)
        prewarmed.append(updated)

    if prewarmed and embedder is not None:
        get_or_compute_embeddings(prewarmed, embedder, db)

    return len(prewarmed)


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


def rss_story_id(link: str) -> int:
    """Synthetic (negative) story id of an RSS entry, keyed by its link.

    Arctic Shift stories use the same key (the post permalink), so a post
    keeps its id, votes and caches when ``reddit_source`` changes.
    """
    h = hashlib.md5(link.encode("utf-8")).digest()
    return -(int.from_bytes(h[:4], "big") % (2**31))


_REDDIT_TOP_WINDOW_SECONDS: dict[str, float] = {
    "hour": 3600.0,
    "day": 86400.0,
    "week": 7 * 86400.0,
    "month": 30 * 86400.0,
    "year": 365 * 86400.0,
}
# Reddit's listing defaults (no ``t`` / ``limit`` in the feed URL).
_REDDIT_DEFAULT_TOP_WINDOW = "day"
_REDDIT_DEFAULT_LIMIT = 25
_REDDIT_MEDIA_HOSTS = frozenset(
    {"reddit.com", "redd.it", "v.redd.it", "i.redd.it", "preview.redd.it"}
)


def reddit_top_query(feed_url: str) -> tuple[str, float, int] | None:
    """``(subreddit, window_seconds, limit)`` of a ``/r/<sub>/top`` feed URL.

    Returns None for other listings (hot/new), which Arctic Shift cannot
    rank, and for ``t=all``: Reddit picks all-time leaders before the age
    cutoff, which a windowed archive search cannot reproduce.
    """
    if not _reddit_subreddit_from_feed_url(feed_url):
        return None
    parsed = urlparse(feed_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[2].lower() not in {"top", "top.rss"}:
        return None
    query = parse_qs(parsed.query)
    window_name = (query.get("t") or [_REDDIT_DEFAULT_TOP_WINDOW])[0].lower()
    if window_name not in _REDDIT_TOP_WINDOW_SECONDS:
        return None
    window = _REDDIT_TOP_WINDOW_SECONDS[window_name]
    try:
        limit = int((query.get("limit") or [_REDDIT_DEFAULT_LIMIT])[0])
    except ValueError:
        limit = _REDDIT_DEFAULT_LIMIT
    return parts[1], window, max(limit, 0)


def reddit_post_self_text(post: ArcticPost) -> str:
    """Post text as the RSS path stores it: external link, then the body."""
    pieces: list[str] = []
    host = urlparse(post.url).netloc.lower().removeprefix("www.")
    if not post.is_self and post.url and host not in _REDDIT_MEDIA_HOSTS:
        pieces.append(post.url)
    body = clean_text(post.selftext)
    if body:
        pieces.append(body)
    return " ".join(pieces)[:RSS_SELF_TEXT_CHAR_LIMIT]


def reddit_post_story(post: ArcticPost, source_name: str) -> Story:
    """Story row for an archived post, shaped like a Reddit RSS entry.

    Score and comment count stay 0 as on the RSS path (Reddit RSS carries
    neither), so ranking features do not change with ``reddit_source``.
    """
    link = f"https://www.reddit.com{post.permalink}"
    self_text = reddit_post_self_text(post)
    return Story(
        id=rss_story_id(link),
        title=post.title,
        url=link,
        score=0,
        time=post.created_utc,
        text_content=compose_story_text(post.title, self_text),
        self_text=self_text,
        source=source_name,
        comment_count=0,
        comment_count_at_fetch=0,
        discussion_url=None,
    )


async def _fetch_arctic_topfeed(
    feed_url: str,
    per_feed: int,
    cutoff: float,
    now: float,
    exclude_urls: set[str],
) -> list[Story]:
    """A subreddit top feed from Arctic Shift; empty list on any failure."""
    query = reddit_top_query(feed_url)
    if query is None:
        logging.warning("arctic_shift: unsupported Reddit feed %s", feed_url)
        return []
    subreddit, window, limit = query
    try:
        posts = await arctic_shift.top_posts(
            subreddit,
            window_seconds=window,
            limit=min(per_feed, limit),
            now=now,
        )
    except ArcticShiftError as exc:
        logging.warning("arctic_shift: top feed %s failed: %s", feed_url, exc)
        return []
    except Exception:
        logging.exception("Unexpected error fetching Arctic Shift feed %s", feed_url)
        return []
    source_name = _rss_source_name(feed_url)
    stories = [reddit_post_story(post, source_name) for post in posts]
    return [s for s in stories if s.time >= cutoff and s.url not in exclude_urls]


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
    skip_title_prefixes: tuple[str, ...] = (),
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
                pub_time = calendar.timegm(published_parsed)
            else:
                updated_parsed = entry.get("updated_parsed")
                if updated_parsed:
                    pub_time = calendar.timegm(updated_parsed)
                else:
                    pub_time = now

            if pub_time < cutoff:
                continue

            title = entry.get("title", "Untitled")
            if skip_title_prefixes and title.startswith(skip_title_prefixes):
                continue

            summary = ""
            if "content" in entry and entry.content:
                summary = entry.content[0].value
            elif "summary" in entry:
                summary = entry.summary

            clean_summary = clean_text(summary)
            snippet = clean_summary[:RSS_SELF_TEXT_CHAR_LIMIT]
            self_text = snippet
            text_content = compose_story_text(title, self_text)

            # RSS <comments> element — Tildes and Lobsters provide a
            # separate discussion URL. Reddit, LessWrong, and personal
            # blogs don't have one (entry.comments is None).
            comments_url = entry.get("comments")

            synthetic_id = rss_story_id(link)

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
    except (URLError, httpx.HTTPError, TimeoutError) as e:
        # Both httpx and the urllib fallback (fetch_with_urllib_fallback)
        # were exhausted -- a genuine network-down/DNS/timeout condition,
        # or a status code neither side treats as retryable. Expected and
        # transient; not worth a full traceback. urllib's read timeout
        # raises a bare TimeoutError, not URLError (2026-10-08).
        logging.warning("Failed to fetch RSS feed %s: %r", feed_url, e)
        return []
    except Exception:
        # Anything else (malformed feed content, a parsing bug, ...) is
        # unexpected -- keep the full traceback per the project's
        # no-silent-failures rule.
        logging.exception("Unexpected error fetching RSS feed %s", feed_url)
        return []


def build_reddit_topfeed_factories(
    feeds: list[str],
    per_feed: int,
    days: int,
    exclude_urls: set[str],
    reddit_source: RedditSource = "rss",
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

    With ``reddit_source="arctic_shift"`` the factories read Arctic Shift
    (:func:`_fetch_arctic_topfeed`) and bypass ``reddit_limiter``, which
    paces requests to reddit.com only.
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
            if reddit_source == "arctic_shift":
                stories = await _fetch_arctic_topfeed(
                    feed_url, per_feed, cutoff, now, exclude_urls
                )
                if stories:
                    reddit_feed_cache.set(feed_url, stories)
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
    skip_title_prefixes: tuple[str, ...] = (),
) -> list[Story]:
    """Fetch non-Reddit RSS feeds synchronously and upsert to DB.

    Entries whose title starts with one of ``skip_title_prefixes`` are
    dropped (AINews issues, which ``pipeline.ainews`` splits by topic).

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
        _fetch_and_parse_feed(
            f, per_feed, cutoff, now, exclude_urls, skip_title_prefixes
        )
        for f in other_feeds
    ]
    feed_results = list(await asyncio.gather(*tasks)) if tasks else []

    all_stories: list[Story] = []
    for res in feed_results:
        for s in res:
            try:
                db.upsert_story(s)
            except StoryIdentityConflict:
                logging.warning("rss_identity_conflict story_id=%s url=%s", s.id, s.url)
                continue
            all_stories.append(s)

    return all_stories


# Embedder


# HTML Render
def _article_fetch_failure_active(
    db: Database,
    story_id: int,
    now_ts: float,
) -> bool:
    failure = db.get_article_fetch_failure(story_id)
    if failure is None:
        return False
    if tweet_id_from_url(failure.url) and not failure.last_error.startswith("tweet_"):
        # Recorded by the HTML fetcher before tweets went through
        # fxtwitter (2026-09-29); the tweet path gets its own attempt.
        return False
    return failure.permanent or failure.next_retry_at > now_ts


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
        # Bot-blocked (403 on every fetch, 2026-09-27).
        "openai.com",
    }
)

_BINARY_SUFFIXES: tuple[str, ...] = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".ps",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".tgz",
    ".rar",
    ".7z",
    ".dmg",
    ".iso",
    ".deb",
    ".rpm",
    ".apk",
    ".exe",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
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
    if urlparse(url).path.lower().endswith(_BINARY_SUFFIXES):
        return False
    return True


# Sources whose text arrives complete: never fetch their URL as an article
# (an AINews card's URL is the whole issue page).
_NO_ARTICLE_SOURCES = frozenset({"rss_lesswrong_com", AINEWS_SOURCE})


def _article_fetch_eligible(
    story: Story, db: Database, min_time: float, now_ts: float
) -> bool:
    if not story.url or story.article_body:
        return False
    if not _is_fetchable_article_url(story.url):
        return False
    if story.source.startswith("rss_reddit_") or story.source in _NO_ARTICLE_SOURCES:
        return False
    if story.time < min_time:
        return False
    if _article_fetch_failure_active(db, story.id, now_ts):
        return False
    return True


def select_rss_article_prewarm(
    candidates: Sequence[Story],
    db: Database,
    *,
    max_per_run: int,
    max_age_days: int = 30,
    now_ts: float | None = None,
) -> list[Story]:
    """Newest RSS stories (not Reddit/LessWrong/AINews) still lacking article text.

    The warm-path fetcher only reaches stories that already rank near the
    top, so a feed that ships a short snippet never earns the text that
    would let it rank. Rows are re-read from the DB: feed parsing yields
    fresh objects without the stored body.
    """
    if max_per_run <= 0:
        return []
    now_ts = time.time() if now_ts is None else now_ts
    min_time = now_ts - (max_age_days * 86400)
    rss_ids = [
        s.id
        for s in candidates
        if s.source.startswith("rss_")
        and not s.source.startswith("rss_reddit_")
        and s.source not in _NO_ARTICLE_SOURCES
    ]
    stored = [
        s
        for s in db.get_stories(rss_ids)
        if _article_fetch_eligible(s, db, min_time, now_ts)
    ]
    stored.sort(key=lambda s: s.time, reverse=True)
    return stored[:max_per_run]


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
        return _article_fetch_eligible(story, db, min_time, now_ts)

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


# Errors that become permanent after this many consecutive failures: pages
# that extract to nothing, and sites that refuse us (auth walls, bot blocks).
_ARTICLE_REPEAT_PERMANENT_ERRORS = frozenset(
    {"empty_extraction", "http_401", "http_403"}
)
_ARTICLE_REPEAT_PERMANENT_AFTER = 3
_ARTICLE_PERMANENT_RETRY_S = 3650 * 86400


def record_article_fetch_failure_outcome(
    db: Database,
    story_id: int,
    url: str,
    *,
    status: int | None,
    error: str | None,
    permanent: bool,
    now_ts: float | None = None,
) -> None:
    """Record one failed article fetch under the shared retry policy.

    Exponential backoff (1 h doubling, capped at 1 day); permanent when the
    fetcher said so (404/410, non-HTML, blocked URL) or after repeated
    empty extractions / 401 / 403. Used by the proactive fetcher and by the
    TLDR tap path so both apply the same rules.
    """
    now_ts = time.time() if now_ts is None else now_ts
    previous = db.get_article_fetch_failure(story_id)
    failure_count = (previous.failure_count if previous else 0) + 1
    permanent = permanent or (
        error in _ARTICLE_REPEAT_PERMANENT_ERRORS
        and failure_count >= _ARTICLE_REPEAT_PERMANENT_AFTER
    )
    next_retry_at = (
        now_ts + _ARTICLE_PERMANENT_RETRY_S
        if permanent
        else _article_failure_retry_time(failure_count, now_ts)
    )
    db.record_article_fetch_failure(
        story_id,
        url,
        status=status,
        error=error,
        permanent=permanent,
        next_retry_at=next_retry_at,
    )


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
    model_version = embedder.model_version

    success = [0]
    error_counts: dict[str, int] = {}

    async def fetch_one(story: Story) -> tuple[int, Story | None]:
        async with sem:
            try:
                await rank_gate.wait_idle_async()
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
                    # Deck stories are ranking copies without comments or
                    # self text; the stored row merges them back in. Embed
                    # the stored text, or its hash never matches and the
                    # next pool rebuild re-embeds the story under the pool
                    # lock (a 45 s rerank stall after each regen).
                    updated = db.get_story(story.id) or updated
                    embed_text = story_embedding_text(updated)
                    await rank_gate.wait_idle_async()
                    new_vec = embedder.encode([embed_text])[0]
                    new_hash = hashlib.sha256(embed_text.encode("utf-8")).hexdigest()
                    db.upsert_embedding(
                        story.id,
                        model_version,
                        new_hash,
                        new_vec,
                        model_sha=embedder.model_onnx_sha,
                        dim=embedder.embedding_dim,
                    )
                    db.clear_article_fetch_failure(story.id)
                    success[0] += 1
                    return story.id, updated

                record_article_fetch_failure_outcome(
                    db,
                    story.id,
                    story.url or "",
                    status=result.status,
                    error=result.error,
                    permanent=result.permanent,
                )
                error_key = result.error or (
                    f"http_{result.status}" if result.status else "unknown"
                )
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
                return story.id, None
            except Exception:
                logging.exception(
                    "article_fetch: unexpected exception story_id=%d url=%s",
                    story.id,
                    story.url or "",
                )
                record_article_fetch_failure_outcome(
                    db,
                    story.id,
                    story.url or "",
                    status=None,
                    error="internal_exception",
                    permanent=False,
                )
                error_counts["internal_exception"] = (
                    error_counts.get("internal_exception", 0) + 1
                )
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
