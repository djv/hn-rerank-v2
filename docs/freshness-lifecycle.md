# Fetch, enrichment, and cache lifecycle

Inspected against the live source tree and deployed `8155162` behavior. Local
freshness edits are **not deployed**: one-hour core interval, rechecking hydrated
LessWrong candidates, and retaining higher engagement counts independently of
text length. These are provisional pending this lifecycle review.

## Not one FSM

There are several implicit state machines, implemented with locks, timers,
versions, content-presence tests and queues. There is no unified per-story
`last_checked / next_due / fetching / retry_at` lifecycle across sources.

### Core regeneration (`server.py:regen_loop`)

Startup delay -> event or interval wait -> fetch candidates -> HN growth probes
and comment prewarm -> LessWrong prewarm -> invalidate shared candidate pool /
rebuild cold deck -> bump dashboard versions -> schedule cached-user warms ->
submit independent Reddit refresh and article/TLDR background work -> wait.

The deployed default is four hours **after completion**, not a fixed wall-clock
cadence. Feedback can also trigger regeneration after an idle period. The core
loop is serial; adding more frequent triggers does not create concurrent core
runs. A failed cycle is logged and returns to the wait loop.

### Source acquisition / hydration (`pipeline/__init__.py`, `enrichment.py`)

- HN: ClickHouse live window refreshes scores/counts. Comment hydration is gated
  by missing text, missing fetch history, or meaningful count growth. Limited
  Firebase count probes and Algolia hydration can compensate for CH lag for
  selected young threads. Upstream CH itself can lag 1–24 hours.
- Archive: existing DB rows, not a full upstream re-fetch each cycle.
- RSS: feed entries are fetched; that is not equivalent to revisiting every
  existing article/thread in the database.
- LessWrong (deployed): GraphQL prewarm is selected only for candidates without
  comment text. Inside prewarm, unchanged/shorter text skips the write, and the
  merge retains the first nonzero displayed count. On-demand source hydration
  also requires missing body or comments. These gates can make hydrated posts
  permanently stale; more regen cycles alone cannot fix them.
- Reddit: independent coalescing worker (one running, at most one pending rerun).
  Feed cache TTL is four hours. Fetch queue uses ~50-second spacing, limiter,
  circuit breaker and persisted retry state. Prewarm selects unhydrated stories,
  top 10 per subreddit, capped at 80 per cycle. This is not entire-pool refresh.

Core acquisition also excludes globally recorded feedback IDs/URLs in relevant
fetch paths. Its acquisition list is not identical to the production ranking
pool loaded from SQLite.

### Shared candidate snapshot (`pipeline/candidate_cache.py`)

Missing/invalidated -> load DB candidates and embeddings under one lock -> ready.
Ready snapshots have no TTL. Explicit invalidation on cold-deck rebuild advances
the generation. A SQLite write alone does not update this snapshot. Per-user
ranking masks feedback out of the shared pool.

### Per-user deck (`server.py:Handler`)

- Matching version: serve cached document immediately; no age-based expiration.
- Old version: serve stale document and request background warm.
- No document: serve cold deck or skeleton; schedule personalization as needed.
- Warm: debounce, coalesce requested version, one running warm per user, rank
  from local candidate snapshot, atomically publish HTML+feed document, then
  schedule another attempt if a newer version was requested.
- Core and changed Reddit completion explicitly rebuild/invalidate and publish
  new versions. Background article/summary work does not itself perform that
  complete publication sequence.

A warm is **reranking/rendering**, not general upstream refetching. Feed `ready`
means the document has reached its requested ranking version, not that every
story has recently been checked upstream.

### Article / summary caches

Article work deduplicates in-flight story IDs and uses bounded selection and
concurrency. It then prefetches summaries. TLDR caching uses an input-content
hash; unchanged stored content can keep the same cached summary indefinitely.
Active/missing-comment HN paths can probe/hydrate before returning a cached
summary. Provider cooldown/quota failures can serve stale summary fallback.

### TUI

`refresh_feed` downloads the feed and polls while its ranking version is pending.
Once ready, the refresh loop stops. `r` clears local summary caches and fetches
the feed; it is not an upstream-refetch command. The deployed client has no
periodic idle refresh. Local changes add a 60-second lightweight version probe:
fetch only on version change, preserve selection, clear local summaries, and
defer while reading/help/setup/voting or another refresh is active. Probe errors
leave the usable deck alone. Changed versions after a server restart also refresh.

## Consequence for the next design

Do not equate a shorter global regen interval with freshness. A complete design
needs separate metadata/content due times, bounded source-specific rechecks,
retry/backoff and coalescing, explicit snapshot/deck publication after changes,
and client observation of new published versions. Preserve Reddit rate limits
and avoid regenerating embeddings/TLDRs for count-only changes unnecessarily.

## Ranking observation (separate investigation)

The inspected profile's live Recent feed contained 51 stories, including 10
LessWrong stories; those 10 held Recommended positions 1–10 with rank scores
approximately 0.976–0.984. Ordering is server-provided and score-sorted, not a TUI
source-grouping artifact. Source features are learned in the ranker; identifying
which features caused this concentration still needs model-level attribution.
No ranking algorithm or diversity change has been made.
