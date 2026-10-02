# HN Rerank status

Updated 2026-10-02. Previous handoff:
[Gemma evaluation fix](docs/status-archive/before-popular-badges-20261002.md).

## Objective
Show every independently qualifying Popular badge. User-selected floors:
Top >=100 points; Talk >=50 comments with comments >=points. Keep Hot's
predicate and the Popular HN-gravity order.

## Verified result
- Independent badges implemented in isolated `feat/stack-popular-badges`
  checkout; cards may have Hot/Top/Talk together or no Popular badge.
- Web tooltips and TUI legend explain the predicates and stacking; Hot's
  percentile tooltip retains 99.5 rather than rounding to 100.
- Two new expectations reproduced the old defects. Local affected suites:
  222 passed; lint, touched formatting and types pass.
- Isolated VPS full backend: 1,095 passed / 1 skipped; TUI: 162 passed /
  1 skipped. Both ran under batch with one CPU and 3G max RAM.
- Architecture and ranking spec updated. Live deployment remains pending.

## Blocker / limits
- Live deployment smoke remains pending. Laptop memory is tight; use one
  batch job at a time.
- Capped encoder replacement and archive yield classification findings
  (#2/#3) remain open; discussion selection and Hot saturation are unchanged.
- Unrelated WIP remains in the original checkout: TUI client test, reader
  mockups, TLDR inspection script and kernel log.

## Next step
- Deploy, restart the service and verify live stacked badges with
  authenticated reads and bounded journal checks.
- Earlier live-yield, Popular-order, reader mockup, LLM reset and parked
  TLDR work remain in
  `docs/status-archive/before-ranking-review-handoff-20261002.md`.
