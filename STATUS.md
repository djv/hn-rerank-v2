# HN Rerank status

Saved 2026-10-02 20:40 UTC. Previous handoff:
[Gemma evaluation fix](docs/status-archive/before-popular-badges-20261002.md).

## Objective
Show every independently qualifying Popular badge. User-selected floors:
Top >=100 points; Talk >=50 comments with comments >=points. Keep Hot's
predicate and the Popular HN-gravity order.

## Verified result
- Independent badges implemented and deployed as `7813480`; cards may
  have Hot/Top/Talk together or no Popular badge.
- Web tooltips and TUI legend explain the predicates and stacking; Hot's
  percentile tooltip retains 99.5 rather than rounding to 100.
- Two new expectations reproduced the old defects. Local affected suites:
  222 passed; lint, touched formatting and types pass.
- Isolated VPS full backend: 1,095 passed / 1 skipped; TUI: 162 passed /
  1 skipped. Both ran under batch with one CPU and 3G max RAM.
- Backend and terminal CI passed for `7813480`. Architecture and ranking
  spec updated; deployment evidence is in FINDINGS.md.
- VPS restart 20:34:33 UTC; all five live profile-151 feeds were ready.
  Story 49908394 (335 points / 637 comments) now has Hot+Top+Talk. Dashboard
  and cached/uncached summary POSTs returned 200; no new-process errors.

## Blocker / limits
- No blocker for this change. Normal background RSS/Reddit retries remain.
- Capped encoder replacement and archive yield classification findings
  (#2/#3) remain open; discussion selection and Hot saturation are unchanged.
- Unrelated WIP remains in the original checkout: TUI client test, reader
  mockups, TLDR inspection script and kernel log.

## Next step
- Badge stacking is complete. Resume review fixes #2/#3, discussion
  selection or Hot-threshold changes only on request.
- Earlier live-yield, Popular-order, reader mockup, LLM reset and parked
  TLDR work remain in
  `docs/status-archive/before-ranking-review-handoff-20261002.md`.
