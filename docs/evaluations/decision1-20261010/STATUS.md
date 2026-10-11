# HN offline evaluation safeguards — 2026-10-11

## Objective
Apply only the verified review issue that can corrupt reported metrics: a snapshot
swap-and-restore during SQLite reading. Offline only; scoring remains paused.

## Verified result
The reader now requires unchanged device/inode/size/mtime_ns/ctime_ns across its
hash/read/hash interval as well as the expected frozen digest. Both in-place
mutation/restoration and replacement/restoration previously published reports;
the regression now rejects both before report publication. Paired-report tests:
57 passed. Full backend: 1,431 passed / 1 skipped / existing sklearn warning in
26.96 seconds. Ruff, touched format, scoped ty and diff whitespace pass. Global ty
matches the earlier 11 missing optional inference-import diagnostics byte for byte.

The earlier frozen-input/v3-plan/full-cohort/per-chunk-proof safeguards and
warm-start test correction remain saved. The independent cloud review assessed
these and the next-step proposal. Conditional padding-sampling concern is not
proof of an inference defect; lower-impact alternate-scope output replacement
and pre-existing backend default are outside this requested fix. No model calls,
production restart, deployment, or actual snapshot/cache changes were made.

## Blocker
No authorized faithful scoring route. Real numerical parity/loading and held-out
quality remain unverified. Parity checks are per-chunk samples; first prepare is
the trust anchor, not authentication of historical artifacts.

## Next step
Keep Strands stopped and Colab skipped. If a faithful route is authorized, begin
with a small pilot checking every row against plain scoring, then require complete
paired v3 coverage before quality metrics. Evidence: GUARD-FIXES-20261011.md and
WARM-START-CHECK-20261011.md. Previous status: STATUS-BEFORE-SNAPSHOT-RACE-20261011.md.
