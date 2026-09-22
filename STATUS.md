# HN Rerank status

## Objective

Simplify freshness using existing workers and publication/version paths. Fix
stale LessWrong metadata and let idle TUI clients observe new decks. Investigate
LessWrong concentration without changing ranking. Preserve unrelated web JSON
and TUI badge WIP.

## Verified result

- Local core refresh default reduced from four hours to one hour after each
  completed cycle. Hydrated LessWrong candidates are rechecked; growing counts
  and scores persist even if content is unchanged/shorter. Richer stored text
  is retained. No new queue or ranking algorithm changes.
- Local TUI checks ranking versions every 60 seconds while idle. Changed or
  reset versions reuse normal refresh, clear local summary caches and retain
  selection. Reading/help/setup/voting/active refresh defer polling; probe
  failures leave the existing deck usable.
- Generated properties cover rising counts, delayed metadata, idempotence,
  and persistence through prewarm using in-memory SQLite. Client integration
  tests cover version transitions, selection, guards and passive failures.
- Cache-only summary endpoint and TUI lookahead of ten stories implemented
  locally. Speculation never generates summaries; misses back off one minute.
  VPS inspection showed low load (~0.12) and ~4.2 GB available memory.
- Latest checks: backend 811 passed; TUI 79 passed / 1 skipped. Ruff, touched
  Python formatting, type checks and diff whitespace checks pass.
- Live profile inspection found all 10 LessWrong entries at positions 1–10 of
  51 Recent recommendations (scores ~0.976–0.984). This is server score ordering,
  not a TUI sorting bug. Feature-level causality remains unverified.

## Workspace / blocker

All current freshness, web JSON refill and TUI polish edits are uncommitted and
undeployed. Do not deploy the entire dirty tree as a freshness-only patch.
Runtime restart/live smoke verification is outstanding. No production data was
changed by freshness tests. Reddit retains its four-hour feed TTL and existing
rate limits; archive rows are not fully refetched each cycle.

## Handoff / next step

User explicitly authorized: (1) separate source refresh from summary generation,
(2) deploy isolated freshness/client changes, (3) investigate LessWrong ranking.
Do NOT ask again for that deployment authorization. No ranking changes authorized.

Latest local additions (not committed/deployed):
- `background_cadence.py`: thread-safe single-flight start-to-start gate.
- `pipeline/config.py`: source regen default 3600 seconds; new
  `tldr_prefetch_interval_seconds=14400`, validated positive.
- `server.py`: shared `Handler._tldr_prefetch_gate` gates automatic TLDR work
  in `_warm_background_tasks` across ALL user warms and regen triggers. Article
  fetching happens before the gate. Gate released in finally, even on failure.
  Foreground requested summaries stay unchanged. Gate is process-local and
  resets on service restart; this is a cadence bound, NOT a daily dollar cap.
- `tests/test_background_cadence.py`: generated operation-sequence and concurrent
  claim tests. One existing server test resets the gate via monkeypatch.
- Latest full backend run: **813 passed**, Ruff and ty clean. TUI last verified
  **79 passed / 1 skipped** before cadence-only backend additions.

Ranking investigation completed to model-sensitivity level:
- `scripts/diagnose_source_scores.py` ran on private VPS snapshot, NEVER live DB.
- Strong nonlinear source/length interaction identified; quantitative results
  in `FINDINGS.md`. Do not claim an additive LW boost or alter ranking.
- Snapshot retained `/tmp/hn-freshness-attribution-snapshot.db` on VPS, chmod 600.
  It contains private data; do not copy into repo or publish.

Deployment is the remaining task. Preserve all unrelated WIP. Suggested steps:
1. Inspect fresh status/diff. Stage only freshness/TUI files, not the web JSON
   refill migration. The latter lives in `templates/index.html`,
   `pipeline/render.py`, `clients/tui/src/hn_rerank/models.py`,
   `tests/test_feed_api.py`, `clients/tui/tests/test_boundaries.py`; leave those
   unstaged. `tests/test_server.py` mixes web assertions and the new gate-test
   reset: stage ONLY the gate-test reset hunk. Docs also mix scopes; stage
   deliberately. Existing TUI badge/legend edits in app/editorial tests are
   intentional and can travel with the client freshness changes.
2. Commit focused changes. Verify the committed snapshot in a separate worktree
   (not just dirty-tree tests), preserving original WIP. Include new files:
   background gate/tests, source properties, cache-only API tests, TUI freshness
   tests, and diagnostic script. Full suite needs
   `HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model`.
3. Push focused commit to origin/main; VPS checkout is
   `/home/dev/hn-rewrite/main`, clean at last inspection (8155162). Check again,
   then fast-forward and restart `hn_rewrite.service`. On VPS `uv` is NOT in
   noninteractive PATH: use `/home/dev/.local/bin/uv`. Model is
   `/home/dev/hn-rewrite/shared/onnx_model`. Preserve existing stash.
4. Verify dashboard/feed and cache-only hit/miss, bounded error journal, and
   completed regeneration. Service port 8766. Verify the specific LW URL's
   count no longer sticks at 4. Use readonly SQLite inspection, not manual
   production DB edits. TUI source uses local editable installation and needs
   restart; do not steal focus. Summarize what is deployed vs leftover WIP.
5. Update this status to verified deployment outcome; archive old status.

No deployment or commit has happened during this continuation. Original main
HEAD remains 419cd9d (prior badge deploy documentation).

Lifecycle evidence: `docs/freshness-lifecycle.md`; change notes: `WORKLOG.md`.
Prior status archived in `docs/status-archive/before-freshness-simplification.md`.
