# HN Rerank status

Saved 2026-10-02 17:15 UTC. Previous review handoff:
[before this fix](docs/status-archive/before-gemma-eval-fix-20261002.md).

## Objective
Fix review finding #1: offline production evaluation must include the frozen
source's Gemma side vectors and reject unexpected model fallback.

## Verified result
- Fold DBs now forward side-vector reads to the source snapshot, including
  when reused for variants with a different side flag.
- Enabled side-coverage fallback aborts evaluation; cold/sparse profiles keep
  normal behavior. Replay-plus-side configurations and variant overrides
  fail before DB access.
- Regressions prove exact score/probability parity with direct source scoring
  across both SVM paths and reused folds, and differences from stored-only
  ranking. Missing/stale/wrong-dimension vectors fail explicitly.
- Affected suites: 106 passed. Full suite: 1,077 passed / 18 skipped. Ruff,
  touched Python formatting and whitespace checks pass; zero new type errors.
- Evidence: FINDINGS.md, "Gemma production-evaluation parity fixed".
  Evaluation guide and ARCHITECTURE.md updated. No production DB access or
  service restart; historical explicit Gemma replay results are unchanged.

## Blocker / limits
- Global `ty` still reports the existing diagnostic in untracked
  `scripts/inspect_tldr_failures.py:86`; preserved as unrelated WIP.
- Review findings #2 (capped replacement encoding) and #3 (archive-source
  yield classification) remain open and outside this authorized fix.
- Other WIP preserved: `clients/tui/tests/test_client.py`, `docs/mockups/`,
  `scripts/inspect_tldr_failures.py`, `kernel.errors.txt`.

## Next step
- Fix #1 is complete and saved with this handoff on `main`.
- Resume #2/#3 only when requested. Keep tests on temporary/in-memory DBs.
- Earlier live-yield, Popular, reader mockup, LLM reset and parked TLDR work
  remain in `docs/status-archive/before-ranking-review-handoff-20261002.md`;
  preserve their authorization boundaries.
