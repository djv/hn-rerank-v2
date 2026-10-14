# Next priorities review assessment — 2026-10-14

Authorized by the user's `rv`. Independent Claude cloud session:
https://claude.ai/code/session_011UmhXs1QTPLfQiLEkzgLuX

## Confirmed corrections

- Snapshot preservation should be first: mutable feedback cannot recover overwritten endpoint votes. Capture at or after 2026-11-05 04:07:22.75 UTC, before subsequent voting; record actual capture time/hash and disclose any delay. Capture remains unscheduled and this review creates no timer or trial change. INTERLEAVING.md:211–238 documents the procedure.
- Distinguish persistent stored prior feed snapshots from the current refresh's in-process cache. `failed_feeds` counts missing current snapshots only for feeds eligible for this refresh; backoff can exclude other feeds (pipeline/__init__.py:1340–1344,1483–1488).
- Latest live read-only journal check supersedes the earlier two-failure observation: October14 15:38:12 UTC completed with failed_feeds=0, requests=159, overloaded=33, transport_errors=0. Ingestion monitoring is now lower priority; these aggregates do not attribute overload to individual feeds or prove every configured feed was attempted.
- A single Queue session establishes progression and sampled rank mapping, not subjective quality or a useful balance. Continue observations without tuning from that sample.

## Findings not established as bugs

- Queue's filtering of restored IDs is intentional: those rows are client insertions missing from the server order. app.py:2080–2091 removes the restoration marker when a row returns in an API feed. Existing tests test_restored_rows_do_not_count_in_served_ranks and test_refresh_returned_undo_card_keeps_server_ranks cover both states. No new rank defect established by the review's queue.py-only reading.
- Queue origin attribution is present: app.py:1438–1448 emits original sort_mode, served_rank/version and surface=queue separately. The review lacked app.py; its attribution concern is resolved by current source inspection.
- No blanket prohibition on future client balance changes follows from the fixed Recommended trial. The existing user direction and this small sample support observation now; any later change needs its own scope and attribution assessment.

## Result and boundary

Next action is to choose a concrete endpoint capture arrangement; Reddit recovered and can be monitored through ordinary refresh logs; Queue remains normal-use observation. No code, database, runtime, snapshot scheduling or trial changes. Source inspection only; no tests rerun for this documentation-only review.
