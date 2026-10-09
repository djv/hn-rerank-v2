# Server and TUI review — 2026-10-09

Requested three-model read-only assessment, executed sequentially: OpenCode
`opencode-go/muse-spark-1.3-contributor`, Codex, Claude Code Haiku. Actual
responses: [Muse](MUSE.md), [Codex](CODEX.md), [Claude](CLAUDE.md).
No broader runtime fixes were made. The independent narrow extraction reviews
are in [the deployment folder](../extraction-deploy-review-20261009/).

## Priorities after source verification

1. **Confirmed, medium: retryable HTTP 500/503 hides a story for the session.**
   `api.py:291` makes server errors generic APIError; `app.py:1259` hides a
   first-time summary on that path. Retain stories for retryable server failures
   and show retry feedback. A 503 can come from unfinished server-side generation.
2. **Confirmed cancellation path, medium: quitting cancels pending votes.**
   `app.py:2130` cancels all workers, including serialized votes at `1806`.
   A queued unsent vote can be lost; an already sent request has uncertain
   server completion. Claude's claim that the server never records it is too
   strong. Consider a bounded drain and explicit pending-vote quit behavior.
3. **Confirmed scheduling, lower priority: pending reranks wait for the
   minute poll.** Vote completion marks the feed unready (`app.py:1837`),
   while the normal poll is every 60 seconds (`817`). Consider more frequent
   polling only while awaiting a deck. Zoomed reading intentionally disables
   polling (`1460`); inspect real reading behavior before changing that policy.
4. **Confirmed edge case, lower priority: first counts poll only establishes
   a baseline (`app.py:1497`).** A counts update between feed load and that
   poll can be missed until another change or manual refresh. Synchronize the
   initial baseline with the delivered feed/counts if this matters in practice.

## Measured latency and freshness limits

[LIVE.json](LIVE.json) is the pre-deploy 17:51 UTC snapshot: profile151 last
24-hour rerank n=75, median 15.15 s, p95 25.35 s, max 51.27 s. Three individual
local HTTP probes were 6–16 ms. Those timings do not measure public HTTPS,
terminal rendering or the time from a keypress to a usable summary. Stored
html timings were all zero; do not infer that rendering has zero cost.
The null `max_concurrent_reranks` field is an unavailable attribute queried by
the probe, not a configured null concurrency limit. Actual `warm_pool_size=2`
was verified from source/config; TLDR generation concurrency is eight.

Muse's 304–306-hour source rows are **not a demonstrated active-feed outage**.
The [live configured source list](ACTIVE-SOURCES.json) was checked against
`_rss_source_name`: every source in that
range is no longer configured. Two active sources with stored-fetch age above
24 hours are Dwarkesh (~184 h) and r/ocaml (~135 h); a stored story timestamp
does not measure feed polling or prove missing new content. No upstream/feed
health probe was run. The historical 1582 failure rows have no window or
denominator and are not an error rate.

The authorized extraction-only restart produced 1.3–1.5 s dashboard/feed
responses during normal startup warming; see the separate deployment evidence.
It is a distinct workload from the pre-deploy quiet local probes.

## Other suggestions requiring verification

Unconfirmed votes already warn and are deliberately never blindly retried;
reconciliation may improve clarity without duplicate votes. Counts changes
invalidate window caches; the actual switching cost is unmeasured. Shared TLDR
join and client timeouts are both 150 s: investigate a response margin if
observed, but Claude's suggested server timeout *above* the client timeout
would not give the client time to receive the server's error.

No reviewer observed the physical terminal. Narrow/wide layout, focus, help,
vote/undo controls, summary cancellation and server lock coverage were partial.
Passing prior tests do not close those gaps. No confirmed general UI verdict,
ranking improvement, live ingestion outage or trial outcome is claimed.
No finding blocks the two-line blank extraction guard; all broader proposals
remain recommendations awaiting a requested implementation scope.
