# Review: hn-rerank next priorities (2026-10-14)

"Verified" means checked in the supplied source. "Unverified" means the supplied files don't show it.

## Ordering

**[High]** Put the snapshot first. It is the only item with a fixed deadline and an irreversible failure mode: `feedback` is mutable final state, and later edits or undos can overwrite trial votes that `--until` cannot recover (`docs/evaluations/model-ablation-20261007/INTERLEAVING.md:211-217`). Reddit comes second as a bounded diagnostic with no established outage. Queue comes third as passive observation.

## Snapshot

- Verified: T0 + 28 d = 1793851642.75 = 2026-11-05 04:07:22.75 UTC = 2026-11-04 23:07:22.75 EST. DST ends Nov 1, so UTC−5 applies (`INTERLEAVING.md:150-151`).
- **[Medium]** "Due" is the wrong word. Capture at or just after the endpoint (≥04:07:23 UTC) and before any further voting or calibration (`:213-214`). An early capture loses votes in [capture, endpoint). A late capture must disclose its capture time and any changes in between (`:215-217`). 23:07 ET is evening voting time, so plan for the user not to vote between the endpoint and the capture.
- **[Medium]** "Arrange" needs a boundary. The addendum creates no timer (`:215`), and this review authorizes no scheduling, so the user chooses the mechanism. Cite the full checklist: `.backup`, chmod 600, record capture time and SHA-256, keep the file unchanged, run the fixed report command (`:222-238`). The separate `rank_perf` check still applies (`:242`).
- Keeping preservation distinct from extending the trial is correct (`:247-249`).

## Reddit

- **[Medium] Internally inconsistent.** `failed_feeds` counts feeds whose in-process cache is `None` after the 120 s retry pass (`pipeline/__init__.py:1365-1388,1485`). A feed that "retained a successful cache" would not count as failed, unless "retained" means rows already in the DB. Unverified. Restate as: "2 feeds had no snapshot after one retry."
- **[Low]** `requests`/`overloaded` aggregate the whole refresh, prewarm included (`:1481`), so they cannot be attributed to individual feeds. Failed feeds are recorded (`:1394`) and filtered later by `next_retry_at` (`:1343-1344`), so `failed_feeds` may undercount feeds still in backoff (unverified: `db` not supplied).
- **Bound:** read-only logs plus `get_reddit_feed_state` over the next 2–3 refreshes (2 h apart, `config.py:171`). Escalate only if the same feed fails in ≥2 consecutive refreshes or `transport_errors>0` repeats.
- **[Low]** The default `reddit_source` is `"rss"`, and that source is noted as retired 2026-11-13 (`config.py:173-175`). Confirm read-only that the deployed config overrides it. Do not change it here.

## Queue

- **[Medium]** "Correct stored Recommended ranks" is only partly true. `served_rank` is enumerated after restored IDs are filtered out (`clients/tui/src/hn_rerank/queue.py:39-40,49`), so ranks can shift relative to the original served order. Qualify the claim or verify it against `feed.orders`.
- Mixed progression is guaranteed by construction (`queue.py:73-76`). With Popular capped at 5 (`:66`), long sessions drift toward Recommended and Explore. One session cannot show variety or reading value, and satisfaction is not measured at all.
- `default_sort` alternates by UTC day (`queue.py:20-23`), so the sample should cover both parities.
- **[Medium, unverified]** Trial credit uses the last Recommended impression (`INTERLEAVING.md:21-24`). Whether Queue logs Recommended-origin cards as Recommended impressions isn't shown. Either way: no Queue balance changes before the endpoint, and keep Queue observations out of trial interpretation.

## Recommended next action

1. Freeze at ≥04:07:23 UTC Nov 5, before further voting, using the documented checklist. The user picks the mechanism.
2. Run the bounded read-only Reddit check.
3. Observe Queue passively with no changes until after the endpoint.

Today: ask the user how the capture will happen.
