"""ClickHouse client for HN bulk data queries.

Replaces per-story parallel Algolia calls with SQL queries for N stories
plus their comment trees. CH Playground is free, no auth, no rate limit
worth mentioning for our usage (~0.3s per 100-story bulk query, ~1s for
2,000 stories).

Comment trees are fetched by walking each story's `kids` array level by
level (`query_comments_bulk`), one `id IN (...)` query per level, rather
than joining against the full comments table. A single-query join
(`INNER JOIN (SELECT * FROM hackernews_history FINAL WHERE type =
'comment' ...)`, repeated once per level) reliably exceeds the shared
play.clickhouse.com instance's query memory limit regardless of batch
size — see incident notes in WORKLOG.md.

Public API:
- query_live_window(days, min_score, limit) -> list of story metadata dicts
- query_stories_bulk(story_ids) -> {id: story_dict} (no comments)
- query_comments_bulk(story_ids, max_levels) -> {id: nested top-level comments}
- query_stories_with_comments(story_ids, max_levels) -> {id: full item dict}
- query_single_story(story_id) -> item dict (lazy fallback; 15min cache)
- clear_cache() -> None (test helper)

Each result item matches the Algolia items shape:
  {
    "id": int,
    "type": "story" | "comment",
    "title": str,
    "url": str | None,
    "points": int,         # stories only; 0 for comments
    "num_comments": int,   # stories only; 0 for comments
    "created_at_i": int,   # unix timestamp (stories only)
    "story_text": str,     # self-post text (stories only)
    "text": str,           # body (stories: self_text; comments: comment text)
    "children": [comment, comment, ...],  # nested in HN kids order (max_levels)
  }

Caching:
- Bulk queries: 1h TTL, keyed by (tuple(sorted_ids), max_levels)
- Single story: 15min TTL, keyed by story_id
- Capped at 128 entries; LRU eviction on overflow
- Process-local; lost on restart (regen rebuilds)
"""

from __future__ import annotations

import threading
import time
from typing import Any, TypedDict

from cachetools import TTLCache
import httpx

from database import coerce_int


CH_PLAYGROUND_URL = "https://play.clickhouse.com/?user=play&default_format=JSON"
CH_TIMEOUT_SECONDS = 30.0

_CACHE_TTL_BULK_SECONDS = 3600
_CACHE_TTL_SINGLE_SECONDS = 900
_CACHE_MAX_ENTRIES = 128


# Bulk entries are keyed by (sorted story ids, comment depth); single-story
# entries by (story id, comment depth). The timers look up time.monotonic at
# call time so tests can drive expiry.
_bulk_cache: TTLCache[tuple[tuple[int, ...], int], dict[int, ChItem]] = TTLCache(
    maxsize=_CACHE_MAX_ENTRIES,
    ttl=_CACHE_TTL_BULK_SECONDS,
    timer=lambda: time.monotonic(),
)
_single_cache: TTLCache[tuple[int, int], ChItem] = TTLCache(
    maxsize=_CACHE_MAX_ENTRIES,
    ttl=_CACHE_TTL_SINGLE_SECONDS,
    timer=lambda: time.monotonic(),
)
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Drop all cached entries. Test helper."""
    with _cache_lock:
        _bulk_cache.clear()
        _single_cache.clear()


def _post_ch(query: str) -> list[dict[str, Any]]:
    """Execute a CH query and return the data rows."""
    resp = httpx.post(
        CH_PLAYGROUND_URL,
        content=query,
        timeout=CH_TIMEOUT_SECONDS,
    )
    if resp.status_code >= 400:
        # raise_for_status() alone discards the response body, which is
        # where CH puts the actual error (e.g. "Code: 241 ...
        # MEMORY_LIMIT_EXCEEDED"). Without this, failures all look like a
        # bare "500 Internal Server Error" in the logs.
        try:
            body = resp.text[:500]
        except Exception:
            body = "<unreadable response body>"
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"{exc}: {body}", request=exc.request, response=exc.response
            ) from exc
    payload: Any = resp.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("ClickHouse returned an unexpected JSON payload shape")


class ChItem(TypedDict):
    """Algolia-items-shaped story/comment dict built from a CH row.

    The single typed model for the CH boundary: producers
    (``_build_story_dict``/``_build_comment_dict``) construct it, consumers
    (``pipeline/enrichment.py``) read it. Raw ``_post_ch`` rows stay
    ``dict[str, Any]`` — that's the JSON edge.
    """

    id: int
    type: str
    title: str
    url: str | None
    points: int
    num_comments: int
    created_at_i: int
    story_text: str
    text: str
    children: list[ChItem]


def _build_story_dict(row: dict[str, Any]) -> ChItem:
    """Map a CH story row to the Algolia items shape."""
    return {
        "id": coerce_int(row.get("id")),
        "type": "story",
        "title": row.get("title") or "",
        "url": row.get("url") or None,
        "points": coerce_int(row.get("score")),
        "num_comments": coerce_int(row.get("descendants")),
        "created_at_i": coerce_int(row.get("ts")),
        "story_text": row.get("text") or "",
        "text": row.get("text") or "",
        "children": [],  # populated by query_stories_with_comments
    }


def _build_comment_dict(row: dict[str, Any]) -> ChItem:
    """Map a CH comment row to the Algolia items shape (recursive children)."""
    return {
        "id": coerce_int(row.get("id")),
        "type": "comment",
        "title": "",
        "url": None,
        "points": 0,
        "num_comments": 0,
        "created_at_i": coerce_int(row.get("ts")),
        "story_text": "",
        "text": row.get("text") or "",
        "children": [],  # populated by query_stories_with_comments
    }


def _build_live_window_query(days: int, min_score: int, limit: int) -> str:
    return f"""
SELECT
    id,
    title,
    url,
    text,
    score,
    descendants,
    toUnixTimestamp(time) AS ts
FROM hackernews_history FINAL
WHERE type = 'story'
  AND deleted = 0 AND dead = 0
  AND score >= {int(min_score)}
  AND time >= now() - INTERVAL {int(days)} DAY
ORDER BY score DESC, ts DESC
LIMIT {int(limit)}
"""


def query_live_window(
    days: int = 30,
    min_score: int = 5,
    limit: int = 5000,
) -> list[ChItem]:
    """Return list of recent high-score story metadata dicts (no comments)."""
    if days <= 0:
        raise ValueError("days must be a positive integer")
    if min_score < 0:
        raise ValueError("min_score must be non-negative")
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    rows = _post_ch(_build_live_window_query(days, min_score, limit))
    return [_build_story_dict(row) for row in rows]


def _build_stories_bulk_query(story_ids: list[int]) -> str:
    ids_csv = ",".join(str(int(i)) for i in story_ids)
    return f"""
SELECT
    id,
    type,
    title,
    url,
    text,
    score,
    descendants,
    toUnixTimestamp(time) AS ts
FROM hackernews_history FINAL
WHERE id IN ({ids_csv})
"""


def query_stories_bulk(story_ids: list[int]) -> dict[int, ChItem]:
    """Return {story_id: full story dict} for the given IDs. No comments."""
    if not story_ids:
        return {}
    rows = _post_ch(_build_stories_bulk_query(story_ids))
    out: dict[int, ChItem] = {}
    for row in rows:
        if row.get("type") == "story":
            out[coerce_int(row.get("id"))] = _build_story_dict(row)
    return out


# Comment fetching walks the `kids` arrays level by level instead of joining
# against the full `hackernews_history` comments table. The previous
# chained-CTE approach (`INNER JOIN (SELECT * FROM hackernews_history FINAL
# WHERE type = 'comment' ...)`, repeated once per level) forced CH to
# materialise every HN comment ever posted, once per level; against the
# shared play.clickhouse.com instance this reliably blew the query memory
# limit (Code: 241, MEMORY_LIMIT_EXCEEDED) regardless of batch size. An
# `id IN (...)` lookup against the exact IDs named by `kids` touches only
# the rows we need and is effectively free.
_ID_CHUNK_SIZE = 5000
# Cap the number of comment IDs fetched at any one level so a single
# pathological megathread can't blow the query back up; this is well above
# any real HN thread's per-level fanout.
_MAX_LEVEL_FRONTIER = 20_000


def _chunked(ids: list[int], size: int = _ID_CHUNK_SIZE) -> list[list[int]]:
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def _build_story_kids_query(story_ids: list[int]) -> str:
    """Return each story's `kids` array (level-0 comment IDs).

    Uses argMax(kids, update_time) rather than FINAL to pick the latest
    version of each story row; CH Playground's FINAL can collapse to a
    version with an empty kids array in some cases, argMax is explicit.
    """
    ids_csv = ",".join(str(int(i)) for i in story_ids)
    return f"""
SELECT id, argMax(kids, update_time) AS kids
FROM hackernews_history
WHERE id IN ({ids_csv}) AND type = 'story' AND deleted = 0 AND dead = 0
GROUP BY id
"""


def _build_comment_level_query(comment_ids: list[int]) -> str:
    """Return comment rows (with their own `kids`) for an exact ID list.

    Deleted/dead state is taken from each comment's latest version, not
    filtered per row: a row-level `deleted = 0` kept comments deleted later
    (their older versions pass). Removed comments still come back so the
    walk reaches their replies; their text is blanked by the caller.
    """
    ids_csv = ",".join(str(int(i)) for i in comment_ids)
    return f"""
SELECT
    id,
    any(by) AS by,
    any(parent) AS parent,
    argMax(text, update_time) AS text,
    argMax(kids, update_time) AS kids,
    argMax(deleted, update_time) OR argMax(dead, update_time) AS removed
FROM hackernews_history
WHERE id IN ({ids_csv}) AND type = 'comment'
GROUP BY id
"""


# Deep enough for nearly every HN thread; each level is one small query and
# the walk stops as soon as a level comes back empty. At 5 levels a sampled
# thread lost 10-22% of its comments (2026-09-29).
DEFAULT_MAX_LEVELS = 30


def query_comments_bulk(
    story_ids: list[int],
    max_levels: int = DEFAULT_MAX_LEVELS,
) -> dict[int, list[ChItem]]:
    """Return {story_id: [top-level comment, ...]} for the given stories.

    Walks the `kids` arrays breadth-first, one cheap `id IN (...)` query per
    level, instead of joining against the full comments table, then nests
    each comment's replies under `children` in HN's own `kids` order (HN
    ranks top-level comments that way), so thread-aware comment selection
    sees real depth and reply counts. Deleted/dead comments stay in the tree
    with empty text so their replies keep their place.
    """
    if not story_ids:
        return {}
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")

    root_kids: dict[int, list[int]] = {sid: [] for sid in story_ids}
    nodes: dict[int, ChItem] = {}
    node_kids: dict[int, list[int]] = {}

    frontier: list[int] = []
    for chunk in _chunked(list(story_ids)):
        for row in _post_ch(_build_story_kids_query(chunk)):
            sid = coerce_int(row.get("id"))
            kids = [coerce_int(k) for k in row.get("kids") or []]
            if sid in root_kids:
                root_kids[sid] = kids
                frontier.extend(kids)

    level = 0
    while frontier and level < max_levels:
        ids = list(dict.fromkeys(frontier))[:_MAX_LEVEL_FRONTIER]
        next_frontier: list[int] = []
        for chunk in _chunked(ids):
            for row in _post_ch(_build_comment_level_query(chunk)):
                cid = coerce_int(row.get("id"))
                if cid in nodes:
                    continue
                node = _build_comment_dict(row)
                if row.get("removed"):
                    node["text"] = ""
                nodes[cid] = node
                kids = [coerce_int(k) for k in row.get("kids") or []]
                node_kids[cid] = kids
                next_frontier.extend(kids)
        frontier = next_frontier
        level += 1

    def attach(cid: int) -> ChItem:
        node = nodes[cid]
        node["children"] = [attach(k) for k in node_kids.get(cid, []) if k in nodes]
        return node

    return {
        sid: [attach(k) for k in kids if k in nodes] for sid, kids in root_kids.items()
    }


def _build_single_story_query(story_id: int) -> str:
    return f"""
SELECT
    id, type, by, parent, title, url, text, score, descendants,
    toUnixTimestamp(time) AS ts, kids
FROM hackernews_history FINAL
WHERE id = {int(story_id)}
"""


def query_stories_with_comments(
    story_ids: list[int],
    max_levels: int = DEFAULT_MAX_LEVELS,
) -> dict[int, ChItem]:
    """Return {story_id: item dict with children} for the given stories.

    Combines query_stories_bulk + query_comments_bulk. Single network
    roundtrip for comments, plus one for stories (could be combined but
    keeping them separate is simpler and the data volume is small).

    `children` holds the nested comment tree (see query_comments_bulk).
    """
    if not story_ids:
        return {}
    bulk_key = (tuple(sorted(story_ids)), max_levels)
    with _cache_lock:
        cached = _bulk_cache.get(bulk_key)
    if cached is not None:
        return cached
    stories = query_stories_bulk(story_ids)
    if not stories:
        return {}
    comments_by_story = query_comments_bulk(list(stories.keys()), max_levels)
    for sid, item in stories.items():
        item["children"] = comments_by_story.get(sid, [])
    with _cache_lock:
        _bulk_cache[bulk_key] = stories
    return stories


def query_single_story(
    story_id: int, max_levels: int = DEFAULT_MAX_LEVELS
) -> ChItem | None:
    """Return a single story's item dict, or None if not found.

    Cache TTL: 15 min (single-story fetches are rare; only used as lazy
    fallback for stories outside the prewarm window).
    """
    if story_id <= 0:
        raise ValueError("story_id must be a positive integer")
    key = (int(story_id), int(max_levels))
    with _cache_lock:
        cached = _single_cache.get(key)
    if cached is not None:
        return cached
    story_rows = _post_ch(_build_single_story_query(story_id))
    if not story_rows:
        return None
    story_dict = _build_story_dict(story_rows[0])
    comments_by_story = query_comments_bulk([story_id], max_levels)
    story_dict["children"] = comments_by_story.get(story_id, [])
    with _cache_lock:
        _single_cache[key] = story_dict
    return story_dict
