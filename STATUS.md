# HN Rerank status

## Objective
Align the server and the web dashboard with the terminal client's simpler
model: one shared summary request per story, one poller and one reload
path, optimistic ordered votes, two timeout classes.

## Verified result
- TUI simplified and pushed (2592706, 4274de9, ff85910, d1b4bea): timeouts
  10s/150s; zoom as a property; `summary_requests` shared by selection and
  prefetch under `Semaphore(4)`; 60s poller + `reload(manual=...)`; votes
  hide at once and are sent in order, a failure reverts with an error.
- TUI tests: 132 pass, ~25s at `-n 8`; ruff, format, ty clean. Read-only
  live run against the VPS: feed, prefetched summaries, sort, zoom, quit OK.
- Details: FINDINGS.md "TUI simplification — 2026-09-26".

## Blocker / limits
- The GitHub repo is public: every push is public. `DEFAULT_SERVER` in
  `clients/tui/src/hn_rerank/app.py` names the tailnet host.
- Not published to PyPI (user's call); `dist/` wheel predates this work.
- Another session is editing feeds/server files in parallel; coordinate
  before touching `server.py`.
- Ranking study (archived status) still holds: only votes after 2026-09-25
  are a clean holdout.

## Next step
Map the server (dashboard versions, `ranking-ready`, warm scheduler, cold
deck, TLDR prefetch) and the web client (`templates/index.html` polling,
voting, TLDR loading) against the TUI model; propose with diagrams, then
refactor in stages.
