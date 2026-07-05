"""ClickHouse client for HN bulk data queries.

Replaces per-story parallel Algolia calls with one SQL query for N stories
plus their comment trees. CH Playground is free, no auth, no rate limit
worth mentioning for our usage (~0.3s per 100-story bulk query, ~1s for
2,000 stories).

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
    resp.raise_for_status()
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


def _build_comments_bulk_query(story_ids: list[int], max_levels: int) -> str:
    """Build a chained-CTE query that returns comments for N stories.

    Returns rows with columns: id, type, by, parent, story_id, text, kids.
    For level0, story_id == parent (the comment's parent is the story).
    For deeper levels, story_id is inherited from the level0 comment that
    initiated the chain.

    Uses argMax(kids, update_time) on the story row to ensure we get the
    latest version of each story's kids array. CH Playground's FINAL
    keyword can collapse to a version with empty kids in some cases;
    argMax is explicit and reliable.
    """
    if not story_ids:
        return "SELECT 1 WHERE 0"
    ids_csv = ",".join(str(int(i)) for i in story_ids)
    ctes = [
        f"""
story_kids AS (
    SELECT id, argMax(kids, update_time) AS kids
    FROM hackernews_history
    WHERE id IN ({ids_csv}) AND type = 'story' AND deleted = 0 AND dead = 0
    GROUP BY id
)""",
        """
expanded_kids AS (
    SELECT arrayJoin(sk.kids) AS id, sk.id AS story_id
    FROM story_kids sk
    WHERE length(sk.kids) > 0
)""",
        """
level0 AS (
    SELECT h.id, h.type, h.by, h.parent, ek.story_id, h.text, h.kids
    FROM (SELECT * FROM hackernews_history FINAL WHERE type = 'comment' AND deleted = 0 AND dead = 0) AS h
    INNER JOIN expanded_kids ek ON h.id = ek.id
)""",
    ]
    prev = "level0"
    for lvl in range(1, max_levels):
        cur = f"level{lvl}"
        ctes.append(
            f"""
{cur} AS (
    SELECT h.id, h.type, h.by, h.parent, p.story_id, h.text, h.kids
    FROM (SELECT * FROM hackernews_history FINAL WHERE type = 'comment' AND deleted = 0 AND dead = 0) AS h
    INNER JOIN {prev} p ON h.parent = p.id
)"""
        )
        prev = cur
    union_parts = [
        f"SELECT id, type, by, parent, story_id, text, kids FROM {p}"
        for p in ["level0"] + [f"level{i}" for i in range(1, max_levels)]
    ]
    return "WITH " + ",\n".join(ctes) + "\n" + "\nUNION ALL\n".join(union_parts)


def query_comments_bulk(
    story_ids: list[int],
    max_levels: int = 5,
) -> dict[int, list[dict[str, Any]]]:
    """Return {story_id: [comment_dict, ...]} for the given stories.

    Each comment dict has the Algolia items shape (id, type, text, kids).
    Children are not recursively nested in the result; the caller is expected
    to use parent IDs to reconstruct the tree if needed.
    """
    if not story_ids:
        return {}
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")
    rows = _post_ch(_build_comments_bulk_query(story_ids, max_levels))
    by_story: dict[int, list[dict[str, Any]]] = {sid: [] for sid in story_ids}
    for row in rows:
        sid = _to_int(row.get("story_id"))
        if sid in by_story:
            by_story[sid].append(_build_comment_dict(row))
    return by_story


def _build_single_story_query(story_id: int) -> str:
    return f"""
SELECT
    id, type, by, parent, title, url, text, score, descendants,
    toUnixTimestamp(time) AS ts, kids
FROM hackernews_history FINAL
WHERE id = {int(story_id)}
"""


def _build_single_story_comments_query(story_id: int, max_levels: int) -> str:
    return _build_comments_bulk_query([story_id], max_levels)


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
    comment_rows = _post_ch(_build_single_story_comments_query(story_id, max_levels))
    story_dict["children"] = [_build_comment_dict(r) for r in comment_rows]
    _cache_put(key, story_dict)
    return story_dict
