# Archived: Algolia per-story parallel hydration

This directory contains the previous per-story parallel Algolia hydration
implementation, which was replaced by a single bulk ClickHouse query in
`ch_client.query_stories_with_comments` on 2026-06-26.

## What's here

- `hydrate_comments_algolia.py` — the old `hydrate_comments_from_algolia`
  function and its `_extract_comments_recursive` / `_select_top_comments`
  helpers, extracted from the previous `scripts/_seed_common.py`. The
  function is preserved verbatim: one Algolia `/api/v1/items/{id}` call per
  story, run with `asyncio.gather` and a 10-way concurrency semaphore.

## Why archived (not deleted)

This implementation is still useful as a fallback if ClickHouse becomes
unavailable or rate-limited. To restore the old behavior, import from this
directory instead of `ch_client`:

```python
# In scripts/_seed_common.py
from scripts._archive.algolia.hydrate_comments_algolia import (
    hydrate_comments_from_algolia,
)
```

The archived code is exercised by the tests in `tests/_archive/algolia/` to
guarantee it remains runnable.

## Why CH bulk is preferred

- **10-100× faster** for bulk operations (1 SQL query vs N parallel HTTP calls)
- **No rate limits** (CH Playground has no per-request rate limit)
- **No third-party dependency** (CH is a public dataset, not an API)
- **Owns the data** (no external service for HN comment data)

## Algolia calls that stay active (not archived)

These per-call Algolia endpoints are still used in production because they
are real-time (no 1-24h CH lag):

- `pipeline.fetch_story` — single-story items for lazy TLDR detail
- `pipeline.refetch_story_text` — single-story refetch for growth-triggered comment update
- `pipeline.fetch_candidates` — 7-day search (no CH equivalent)

See `/home/dev/hn-rewrite/pipeline.py` and the
`plans/algolia-to-clickhouse.md` document for the full architecture.
