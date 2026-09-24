# HN Rerank status

## Reader refresh deploy — 0404066 live on VPS

Cooldown fallback now marks blocked refresh retryable (with reason + delay)
instead of disguising it as a cache hit; TUI preserves readable text and
selection when forced regeneration fails. Verified in worktree at the exact
commit (backend 820, TUI subset 83+1 skip, Ruff/format/ty clean), pushed
bb39263..0404066, VPS fast-forwarded clean with no stash to preserve.
Exact VPS suite 820 passed; service restarted, active. Live: dashboard 200,
uncached regeneration 200 then cached hit 200 with no retryable flag on the
normal path, strict journal error scan clean. Coverage inspection (FINDINGS.md)
left scheduling untouched; summary spot-check showed no new prompt issue.
Leftover local WIP (TUI s-cycle + echo guard, web refill, ranking, source
review) stays uncommitted and was never on the VPS path. Details:
docs/reader-improvements.md.

## Current TLDR deployment — b77afc6

Discussion emphasis (detail-v14) deployed and live-tested: the reported Reddit
story regenerated successfully with bold key terms in all 4 Discussion bullets.
Exact VPS suite 820 passed; Ruff/format/ty clean. Service active after restart,
dashboard and cached summary 200, bounded journal clean. Subscriptions unchanged.


Fixed pane budgets from daa8112 remain deployed to VPS: article-only and discussion-only target
240 words / 6–8 bullets; combined targets 120 words / 3–4 bullets per section.
No viewport dimensions or resize-driven regeneration. Short sources may stay
short. Replaces the undeployed article-only doubling experiment.

Verified exact VPS commit: 820 backend tests passed, Ruff/format/ty clean.
Service restarted active; dashboard 200, Latent Space bio-security generation
succeeded, subsequent cached request 200; bounded journal showed no errors.
Live output was 7 bullets / 414 whitespace-delimited words (previously 4 bullets,
564 characters): meaningfully longer, but model overshot the 240-word prompt
target. Actual TUI screen fit remains unverified; no hard word truncation.

Unrelated local TUI, web rail and refill WIP remains uncommitted and was not
sent to the VPS. Next: inspect actual reading-pane fit before further tuning.


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

## Deployment verified — ab858ed live on VPS

Commit `ab858ed` (freshness/client only; web JSON refill WIP deliberately left
unstaged) was verified in worktree `../hn-rerank-verify` at the exact commit:
backend **812 passed** (813 dirty-tree minus the one unstaged web test),
TUI **75 passed / 1 skipped** (79 minus 4 unstaged web-validation tests),
Ruff/format/ty clean in both trees. Worktree venvs were re-pinned to system
Python 3.12 / SQLite 3.45.1 after a fresh sync pulled 3.14/3.50 and tripped
two unrelated migration integrity tests — environment artifact, not a code
regression. Worktree removed after verification.

Pushed `8155162..ab858ed` to `origin/main`. VPS `/home/dev/hn-rewrite/main`
was clean at `8155162` with the `pre-f03c34e-deploy-identical-detail-v12`
stash preserved; fast-forwarded to `ab858ed`, service restarted, active.
Live checks: dashboard 200, `/api/feed` 200 (68 stories, 25 badged),
`/api/tldr-cache/<id>` 200 on hit / 401 unauthenticated, error journal clean,
first regen completed with `prewarmed 9/9 LessWrong candidates`.

Reported LW story healed operationally via deployed app code (no manual SQL):
`comment_count` 4→28, `at_fetch` 23→28, `score` 0→138, matching live
LessWrong GraphQL. A second restart published it; post-restart regen completed
clean. Two transient `ConnectTimeout`s preceded the heal (LW tarpitting the VPS
IP after the regen burst + probes); cleared after ~4 min. No ranking changes.

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

Known freshness gap (not fixed by this deploy): regen prewarm only rechecks
stories still surfacing in current RSS feeds. Deck-visible stories aged out of
feeds (like the healed 13-day-old LW post) never get rechecked automatically;
healing those needs either on-demand refresh or extending prewarm selection to
snapshot/deck-visible stories — future work, needs a design pass.

Zero-score LW backfill: 47/54 healed via the app prewarm path (38 serial,
9 with 6-way parallel). 7 remain, all LessWrong-throttled timeouts, still
score=0. Deletion declined: 2 of the 7 carry user feedback that deletion
would orphan (project rule: never destructively modify the DB). Retry the
heal later; do not DELETE these rows.

Remaining WIP (still local, undeployed): web JSON refill migration in
`templates/index.html`, `pipeline/render.py`,
`clients/tui/src/hn_rerank/models.py`, `tests/test_feed_api.py`,
`clients/tui/tests/test_boundaries.py`, plus one test hunk in
`tests/test_server.py`. Suggested steps for that separate track:
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
