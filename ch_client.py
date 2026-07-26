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
- query_comments_bulk(story_ids, max_levels) -> {id: list of comment dicts}
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
    "children": [comment, comment, ...],  # recursive (best-effort, max_levels)
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
from typing import Any

from cachetools import TLRUCache
import httpx


CH_PLAYGROUND_URL = "https://play.clickhouse.com/?user=play&default_format=JSON"
CH_TIMEOUT_SECONDS = 30.0

_CACHE_TTL_BULK_SECONDS = 3600
_CACHE_TTL_SINGLE_SECONDS = 900
_CACHE_MAX_ENTRIES = 128


def _cache_ttu(key: tuple[Any, ...], _value: Any, now: float) -> float:
    return now + _ttl_for_key(key)


_cache: TLRUCache[tuple[Any, ...], Any] = TLRUCache(
    maxsize=_CACHE_MAX_ENTRIES,
    ttu=_cache_ttu,
    timer=lambda: time.monotonic(),
)
_cache_lock = threading.Lock()


def _cache_get(key: tuple) -> Any | None:
    with _cache_lock:
        return _cache.get(key)


def _cache_put(key: tuple, value: Any) -> None:
    with _cache_lock:
        _cache[key] = value


def _ttl_for_key(key: tuple[Any, ...]) -> int:
    return (
        _CACHE_TTL_SINGLE_SECONDS
        if key[0] == "single_story"
        else _CACHE_TTL_BULK_SECONDS
    )


def clear_cache() -> None:
    """Drop all cached entries. Test helper."""
    with _cache_lock:
        _cache.clear()


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


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_story_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Map a CH story row to the Algolia items shape."""
    return {
        "id": _to_int(row.get("id")),
        "type": "story",
        "title": row.get("title") or "",
        "url": row.get("url") or None,
        "points": _to_int(row.get("score")),
        "num_comments": _to_int(row.get("descendants")),
        "created_at_i": _to_int(row.get("ts")),
        "story_text": row.get("text") or "",
        "text": row.get("text") or "",
        "children": [],  # populated by query_stories_with_comments
    }


def _build_comment_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Map a CH comment row to the Algolia items shape (recursive children)."""
    return {
        "id": _to_int(row.get("id")),
        "type": "comment",
        "title": "",
        "url": None,
        "points": 0,
        "num_comments": 0,
        "created_at_i": _to_int(row.get("ts")),
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
) -> list[dict[str, Any]]:
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


def query_stories_bulk(story_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Return {story_id: full story dict} for the given IDs. No comments."""
    if not story_ids:
        return {}
    rows = _post_ch(_build_stories_bulk_query(story_ids))
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("type") == "story":
            out[_to_int(row.get("id"))] = _build_story_dict(row)
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
    """Return comment rows (with their own `kids`) for an exact ID list."""
    ids_csv = ",".join(str(int(i)) for i in comment_ids)
    return f"""
SELECT
    id,
    any(by) AS by,
    any(parent) AS parent,
    argMax(text, update_time) AS text,
    argMax(kids, update_time) AS kids
FROM hackernews_history
WHERE id IN ({ids_csv}) AND type = 'comment' AND deleted = 0 AND dead = 0
GROUP BY id
"""


def query_comments_bulk(
    story_ids: list[int],
    max_levels: int = 5,
) -> dict[int, list[dict[str, Any]]]:
    """Return {story_id: [comment_dict, ...]} for the given stories.

    Walks the `kids` arrays breadth-first, one cheap `id IN (...)` query per
    level, instead of joining against the full comments table. Each comment
    dict has the Algolia items shape (id, type, text, kids). Children are
    not recursively nested in the result; the caller is expected to use
    parent IDs to reconstruct the tree if needed.
    """
    if not story_ids:
        return {}
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")

    by_story: dict[int, list[dict[str, Any]]] = {sid: [] for sid in story_ids}

    # frontier maps comment_id -> owning root story_id, for the level about
    # to be fetched.
    frontier: dict[int, int] = {}
    for chunk in _chunked(list(story_ids)):
        for row in _post_ch(_build_story_kids_query(chunk)):
            sid = _to_int(row.get("id"))
            for kid in row.get("kids") or []:
                frontier[_to_int(kid)] = sid

    level = 0
    while frontier and level < max_levels:
        ids = list(frontier.keys())[:_MAX_LEVEL_FRONTIER]
        next_frontier: dict[int, int] = {}
        for chunk in _chunked(ids):
            for row in _post_ch(_build_comment_level_query(chunk)):
                cid = _to_int(row.get("id"))
                sid = frontier.get(cid)
                if sid is None:
                    continue
                if sid in by_story:
                    by_story[sid].append(_build_comment_dict(row))
                for kid in row.get("kids") or []:
                    next_frontier[_to_int(kid)] = sid
        frontier = next_frontier
        level += 1

    return by_story


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
    max_levels: int = 5,
) -> dict[int, dict[str, Any]]:
    """Return {story_id: item dict with children} for the given stories.

    Combines query_stories_bulk + query_comments_bulk. Single network
    roundtrip for comments, plus one for stories (could be combined but
    keeping them separate is simpler and the data volume is small).

    Children are organized as a flat list keyed by parent id within the item
    dict, so callers can reconstruct the tree by walking kids.
    """
    if not story_ids:
        return {}
    bulk_key = ("stories_with_comments", tuple(sorted(story_ids)), max_levels)
    cached = _cache_get(bulk_key)
    if cached is not None:
        return cached
    stories = query_stories_bulk(story_ids)
    if not stories:
        return {}
    comments_by_story = query_comments_bulk(list(stories.keys()), max_levels)
    for sid, item in stories.items():
        item["children"] = comments_by_story.get(sid, [])
    _cache_put(bulk_key, stories)
    return stories


def query_single_story(story_id: int, max_levels: int = 5) -> dict[str, Any] | None:
    """Return a single story's item dict, or None if not found.

    Cache TTL: 15 min (single-story fetches are rare; only used as lazy
    fallback for stories outside the prewarm window).
    """
    if story_id <= 0:
        raise ValueError("story_id must be a positive integer")
    key = ("single_story", int(story_id), int(max_levels))
    cached = _cache_get(key)
    if cached is not None:
        return cached
    story_rows = _post_ch(_build_single_story_query(story_id))
    if not story_rows:
        return None
    story_dict = _build_story_dict(story_rows[0])
    comments_by_story = query_comments_bulk([story_id], max_levels)
    story_dict["children"] = comments_by_story.get(story_id, [])
    _cache_put(key, story_dict)
    return story_dict
