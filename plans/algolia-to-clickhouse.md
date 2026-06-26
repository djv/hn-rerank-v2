# Algolia → ClickHouse Migration (Future Work)

> Recorded 2026-06-26. Not implemented; deferred until value/cost is reassessed.

## Why

- Algolia's HN API (`hn.algolia.com`) is currently a hard dependency for four call sites. We want to know what fraction of those calls ClickHouse can replace, and at what cost.
- The CH `hackernews_history` table already covers all story metadata plus the recursive comment tree (`text`, `by`, `time`, `parent`, `kids`, `deleted`, `dead`).

## Algolia call inventory (current state)

| # | Caller | Endpoint | Source target | Replacable? |
|---|--------|----------|---------------|-------------|
| 1 | `pipeline.fetch_story` | `GET /api/v1/items/{id}` | live + archive | ✅ |
| 2 | `pipeline.refetch_story_text` | `GET /api/v1/items/{id}` | live + archive | ✅ |
| 3 | `pipeline.fetch_candidates` | `GET /api/v1/search` | live `hn` only | ❌ |
| 4 | `_seed_common.hydrate_comments_from_algolia` | `GET /api/v1/items/{id}` | archive `bq_seed`/`ch_seed` | ✅ |

Calls 1, 2, 4 are item-fetches (full story + comment tree in one request). Call 3 is the live 7-day search index and has no CH equivalent.

## CH coverage of the data we need

- ✅ `title`, `url`, `score`, `descendants`, `time`, `text` (self-post), `by`
- ✅ `text` on comments is non-empty (HTML-encoded, matches HN API format)
- ✅ `kids` array works for tree-walking; `parent` also works for BFS
- ✅ `descendants` matches BFS walk exactly when filtering `deleted=0 AND dead=0`
- ❌ No search index — cannot replace call 3
- ⚠️ Comment `score` is 0 in CH (HN doesn't expose comment points either)
- ⚠️ ~1–24h latency for new content — affects live `hn` source stories posted in the last day

## Performance (measured on a 950-comment story)

| Approach | Time | Notes |
|---|---|---|
| Algolia single request | ~0.3s | Full tree, one HTTP call |
| Python BFS, 10 sequential CH calls | 2.7s | All 1018 rows, all levels |
| Chained CTE for 5 levels | 0.24s | ~913 rows (~95% of trees) |
| `WITH RECURSIVE` | broken | CH 26.7 doesn't support it |

Algolia wins on raw latency. CH wins on ownership, no rate limit, no third-party dependency, free.

## Migration options (3 levels of ambition)

### Option A — Do nothing
- Status quo. Algolia stays for all 4 uses. ~35 calls/regen.
- Risk: none.

### Option B — Replace archive seed hydration only (call #4)
- New `hydrate_comments_from_ch` in `scripts/_seed_common.py`.
- Use BFS or chained CTE.
- Live `fetch_story` (call #1) and `refetch_story_text` (call #2) keep Algolia — necessary because live `hn` stories may not be in CH yet (1–24h lag).
- Tradeoff: 1000-story seed would take ~10–20 min instead of 1–2 min (Algolia parallel). Worth it for self-contained archive data.

### Option C — Replace all 3 item-fetches (calls 1, 2, 4)
- `fetch_story`, `refetch_story_text`, `hydrate_comments_from_algolia` all use CH.
- `fetch_candidates` (call 3) stays on Algolia.
- Tradeoff: TLDR detail for a 950-comment thread goes from ~0.3s → ~2.7s (or ~0.24s with chained CTE for 5 levels, ~95% coverage).
- Benefit: zero Algolia item-call dependency, no 429s, no third-party for comment data.

## Recommendation

**Option C with chained CTE (0.24s for 5 levels)** is the sweet spot:
- Self-contained comment data (no third-party for items endpoint)
- Acceptable latency (~0.5s with one round-trip)
- ~95% comment coverage (the other 5% are 6+ level deep replies)
- Live search stays on Algolia (no replacement exists)

## When to revisit

- When Algolia rate limits become a real problem
- When the BFS CTE can be expressed in a single query (CH roadmap)
- When Algolia HN API deprecates (no public timeline, but it has been the unofficial HN API for 8+ years)
- When we want to drop a third-party dependency for privacy/reliability reasons

## Test plan when implemented

- Mock CH responses in `tests/test_pipeline.py` to match Algolia item shape; verify `fetch_story` accepts both formats
- New `tests/test_hydrate_comments_from_ch.py` with BFS, CTE, and edge-case (deleted/empty/poll) scenarios
- Live smoke test: pick 3 stories from the recent CH seed (167 rows), compare BFS comments vs Algolia items API comments
