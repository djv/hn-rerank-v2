"""Arctic Shift client for Reddit posts and comment trees.

Reddit retires RSS on 2026-11-13 and blocks anonymous ``.json``/HTML, so
``Config.reddit_source = "arctic_shift"`` reads subreddit top lists and
comment trees from the Arctic Shift archive instead
(https://github.com/ArthurHeitmann/arctic_shift). Free, no key; its
operator asks for "a couple requests per second" at most.

Arctic Shift stores each post at creation and re-fetches it after roughly
36 hours; until then ``score`` is 1 and ``num_comments`` 0. A post
therefore reaches the by-score top list about 1.5 days after posting.
Search sorts only by time, so :func:`top_posts` pages through every post
in the window, ranks by score locally, then loads full records for the
leaders.

The archive reads the official Reddit API, so it is expected to stop when
Reddit closes public API access (March 2027) or earlier on a takedown.
Evidence: FINDINGS.md "Reddit after the RSS shutdown".
"""

from __future__ import annotations

import asyncio
import html
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

ARCTIC_SHIFT_BASE_URL = "https://arctic-shift.photon-reddit.com"
ARCTIC_SHIFT_USER_AGENT = "hn-rewrite/1.0 personal reader (Reddit top posts)"
ARCTIC_SHIFT_TIMEOUT_SECONDS = 60.0
# Minimum start-to-start spacing between requests across all threads.
REQUEST_SPACING_SECONDS = 0.5
MAX_ATTEMPTS = 2
# Top lists are fetched in the background, so they can wait out the
# archive's overload answers (422 "Timeout. Maybe slow down a bit", which
# cleared within about a minute on 2026-10-08).
TOP_POSTS_ATTEMPTS = 4
MAX_RETRY_WAIT_SECONDS = 30.0
# Wait before the retry when no x-ratelimit-reset header says otherwise.
RETRY_DELAY_SECONDS = 2.0
MAX_SEARCH_PAGES = 30
# "auto" pages return 100-1000 rows; a page under 100 is the last one.
AUTO_PAGE_MIN_ROWS = 100
COMMENT_TREE_LIMIT = 200
# Leaders by search-time score whose full records are loaded; extra room
# for removed posts that the full record reveals.
DETAIL_CANDIDATE_FACTOR = 2

_REMOVED_TEXTS = frozenset({"[removed]", "[deleted]"})

_throttle_lock = threading.Lock()
_next_request_at = 0.0


class ArcticShiftError(Exception):
    """A request failed after retries or returned an unexpected shape."""


@dataclass(frozen=True)
class ArcticPost:
    id: str
    permalink: str
    title: str
    url: str
    selftext: str
    author: str
    score: int
    num_comments: int
    created_utc: int
    is_self: bool
    removed: bool


@dataclass(frozen=True)
class ArcticComment:
    author: str
    body: str
    score: int
    created_utc: int
    depth: int


@dataclass(frozen=True)
class _PostScore:
    id: str
    score: int
    created_utc: int


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=ARCTIC_SHIFT_BASE_URL,
        headers={"User-Agent": ARCTIC_SHIFT_USER_AGENT},
        timeout=ARCTIC_SHIFT_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


async def _wait_turn() -> None:
    global _next_request_at
    with _throttle_lock:
        now = time.monotonic()
        start = max(now, _next_request_at)
        _next_request_at = start + REQUEST_SPACING_SECONDS
    if start > now:
        await asyncio.sleep(start - now)


def _retry_wait(resp: httpx.Response) -> float:
    raw = resp.headers.get("x-ratelimit-reset")
    try:
        wait = float(raw) if raw is not None else RETRY_DELAY_SECONDS
    except ValueError:
        wait = RETRY_DELAY_SECONDS
    return min(max(wait, 0.0), MAX_RETRY_WAIT_SECONDS)


async def _get_data(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, str | int],
    attempts: int = MAX_ATTEMPTS,
) -> list[object]:
    """GET ``path`` and return its ``data`` list, retrying transient errors."""
    last_error = ""
    for attempt in range(attempts):
        await _wait_turn()
        try:
            resp = await client.get(path, params=params)
        except httpx.HTTPError as exc:
            last_error = repr(exc)
            if attempt + 1 < attempts:
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            continue
        # 422 has been seen transiently on valid queries (2026-10-08).
        if resp.status_code in (422, 429) or resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code}"
            if attempt + 1 < attempts:
                await asyncio.sleep(_retry_wait(resp))
            continue
        if resp.status_code != 200:
            raise ArcticShiftError(f"{path}: HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ArcticShiftError(f"{path}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ArcticShiftError(f"{path}: response is not an object")
        if payload.get("error"):
            raise ArcticShiftError(f"{path}: {payload['error']}")
        data = payload.get("data")
        if data is None:
            return []
        if not isinstance(data, list):
            raise ArcticShiftError(f"{path}: data is not a list")
        return data
    raise ArcticShiftError(f"{path}: {last_error}")


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _parse_score(raw: object) -> _PostScore | None:
    if not isinstance(raw, dict):
        return None
    post_id = raw.get("id")
    created = _int(raw.get("created_utc"))
    if not isinstance(post_id, str) or not post_id or created is None:
        return None
    return _PostScore(
        id=post_id, score=_int(raw.get("score")) or 0, created_utc=created
    )


def parse_post(raw: object) -> ArcticPost | None:
    """Normalize one archived post record; ``None`` when malformed."""
    if not isinstance(raw, dict):
        return None
    post_id = raw.get("id")
    permalink = raw.get("permalink")
    created = _int(raw.get("created_utc"))
    if (
        not isinstance(post_id, str)
        or not post_id
        or not isinstance(permalink, str)
        or not permalink.startswith("/r/")
        or created is None
    ):
        return None
    selftext = html.unescape(_str(raw.get("selftext")))
    # Not ``removed_by_category``: it is a snapshot from archiving time, and
    # e.g. r/ClaudeAI's AutoModerator holds nearly every new post
    # ("automod_filtered") that moderators then approve. Such posts filled
    # Reddit's own weekly top list (2026-10-08).
    removed = selftext.strip() in _REMOVED_TEXTS
    return ArcticPost(
        id=post_id,
        permalink=permalink,
        title=html.unescape(_str(raw.get("title"))),
        url=_str(raw.get("url")),
        selftext=selftext,
        author=_str(raw.get("author")),
        score=_int(raw.get("score")) or 0,
        num_comments=_int(raw.get("num_comments")) or 0,
        created_utc=created,
        is_self=raw.get("is_self") is True,
        removed=removed,
    )


async def _search_scores(
    client: httpx.AsyncClient,
    subreddit: str,
    after: int,
    before: int,
    attempts: int = MAX_ATTEMPTS,
) -> list[_PostScore]:
    seen: dict[str, _PostScore] = {}
    cursor = after
    for _page in range(MAX_SEARCH_PAGES):
        rows = await _get_data(
            client,
            "/api/posts/search",
            {
                "subreddit": subreddit,
                "after": cursor,
                "before": before,
                "sort": "asc",
                "limit": "auto",
                "fields": "id,score,created_utc",
            },
            attempts,
        )
        parsed = [p for raw in rows if (p := _parse_score(raw)) is not None]
        new = [p for p in parsed if p.id not in seen]
        for post in new:
            seen[post.id] = post
        if len(rows) < AUTO_PAGE_MIN_ROWS or not new:
            break
        last = max(p.created_utc for p in parsed)
        if last <= cursor:
            break
        cursor = last
    else:
        logging.warning(
            "arctic_shift: %s search stopped after %d pages",
            subreddit,
            MAX_SEARCH_PAGES,
        )
    return list(seen.values())


async def _posts_by_ids(
    client: httpx.AsyncClient, ids: list[str], attempts: int = MAX_ATTEMPTS
) -> list[ArcticPost]:
    posts: list[ArcticPost] = []
    for start in range(0, len(ids), 100):
        rows = await _get_data(
            client,
            "/api/posts/ids",
            {"ids": ",".join(ids[start : start + 100])},
            attempts,
        )
        posts.extend(p for raw in rows if (p := parse_post(raw)) is not None)
    return posts


async def posts(ids: list[str]) -> list[ArcticPost]:
    """Full archived records for ``ids`` (missing ids are skipped).

    Raises :class:`ArcticShiftError` when a request fails.
    """
    if not ids:
        return []
    async with _client() as client:
        return await _posts_by_ids(client, ids)


async def top_posts(
    subreddit: str,
    *,
    window_seconds: float,
    limit: int,
    now: float | None = None,
) -> list[ArcticPost]:
    """Top ``limit`` posts of ``subreddit`` created in the last window.

    Ranked by archived score (highest first, then newest); removed posts
    are dropped. Raises :class:`ArcticShiftError` when a request fails.
    """
    if limit <= 0:
        return []
    end = int(now if now is not None else time.time())
    start = int(end - window_seconds)
    async with _client() as client:
        scores = await _search_scores(client, subreddit, start, end, TOP_POSTS_ATTEMPTS)
        leaders = sorted(scores, key=lambda p: (-p.score, -p.created_utc))[
            : limit * DETAIL_CANDIDATE_FACTOR
        ]
        posts = await _posts_by_ids(client, [p.id for p in leaders], TOP_POSTS_ATTEMPTS)
    kept = [p for p in posts if not p.removed and start <= p.created_utc <= end]
    kept.sort(key=lambda p: (-p.score, -p.created_utc))
    return kept[:limit]


def _parse_comment(raw: object, depth: int) -> ArcticComment | None:
    if not isinstance(raw, dict):
        return None
    created = _int(raw.get("created_utc"))
    if created is None:
        return None
    return ArcticComment(
        author=_str(raw.get("author")),
        body=html.unescape(_str(raw.get("body"))),
        score=_int(raw.get("score")) or 0,
        created_utc=created,
        depth=depth,
    )


def _walk_tree(nodes: Sequence[object], depth: int, out: list[ArcticComment]) -> None:
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "t1":
            continue
        data = node.get("data")
        comment = _parse_comment(data, depth)
        if comment is None or not isinstance(data, dict):
            continue
        out.append(comment)
        replies = data.get("replies")
        if isinstance(replies, dict):
            inner = replies.get("data")
            children = inner.get("children") if isinstance(inner, dict) else None
            if isinstance(children, list):
                _walk_tree(children, depth + 1, out)


async def comment_tree(post_id: str) -> list[ArcticComment]:
    """Archived comments of one post, flattened with their depth.

    Raises :class:`ArcticShiftError` when the request fails.
    """
    async with _client() as client:
        nodes = await _get_data(
            client,
            "/api/comments/tree",
            {"link_id": post_id, "limit": COMMENT_TREE_LIMIT},
        )
    comments: list[ArcticComment] = []
    _walk_tree(nodes, 0, comments)
    return comments


def reset_throttle() -> None:
    """Test helper: forget the last request time."""
    global _next_request_at
    with _throttle_lock:
        _next_request_at = 0.0
