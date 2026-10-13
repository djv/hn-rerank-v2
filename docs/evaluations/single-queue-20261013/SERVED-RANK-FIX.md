# Queue served-rank correction — 2026-10-13

The user approved fixing the remaining served-rank telemetry edge after
the first normal Queue session. Scope: client restoration handling only.

## Reproduction and correction

The earlier assessment incorrectly described a ready-feed rebuild race.
`set_feed` only stores the feed; the ready path clears returned restoration
markers before `rebuild`. The ready regression passed before any correction.

The real failure uses an accepted still-ranking response (`ready=false`):
an undone story 2 is present in the API's Recommended order `[2,3]`, but the
client retains its local restoration marker. At the next round, Queue's
synthetic-row filter omits story 2 and gives story 3 served rank 0 instead
of 1. The new integration regression failed before correction with exactly
`[QueueEntry(3, recommended, 0, 7)]` instead of the two served entries.

`refresh_feed` now clears local markers for cards returned by the server
regardless of readiness. Missing undo cards remain local restorations and
are inserted while ranking is pending. The existing synthetic-row rank
test remains unchanged. No ranker, queue mix, endpoint, database schema,
trial configuration or historical telemetry was changed.

The regression runs both readiness states and checks composition, position,
queue slot and serving version through the real Reader refresh path.
Focused Queue/refresh tests: 29 passed. Full gates are recorded in STATUS.

## Verification boundaries

Isolated VPS backend suite: 1,445 passed / 1 skipped, with one existing
sklearn convergence warning; root Ruff and ty passed. Root/standalone
Ruff and touched-file formatting passed locally. Standalone full TUI:
274 passed /1 skipped in76.03s; standalone Ruff, ty and two-file formatting
also passed. The correction is ready for the user's next reader launch.

The corrected client API fetched all five live windows for existing
profile151; all were ready, each composed 26 Queue cards, and every card's
served-rank mapping matched the API order. This was read-only: no votes,
artificial interaction events or summary requests. The existing server
remained PID121960, started20:59:38 UTC; bounded warning journal was empty.
No server code changed or service restart was performed for this client fix.
A fresh user reader launch is needed to observe corrected telemetry in use.

## First normal session

Before this correction, the 21:07:12–21:21:48 UTC session delivered 77
Queue events:37 impressions,39 dwell segments and one article open.
The12h queue advanced through12 Recommended,9 Explore and5 Popular slots,
then continued at slots26–30 with Recommended from a newer serving version.
All21 Recommended Queue impressions across windows mapped to their stored
trial cards (including one `far`, which is intentionally not arm-credited).
Repeated impressions revisited existing slots; these are not duplicate queue
entries. Service warnings remained empty after the deployment restart.

This confirms live Queue delivery and progression into a second round,
not a ranking improvement or a subjective usability verdict. Queue begins
with the same Recommended cards by design; Explore and Popular follow.
