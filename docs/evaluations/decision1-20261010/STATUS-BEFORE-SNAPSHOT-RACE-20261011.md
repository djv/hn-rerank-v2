# HN offline evaluation safeguards — 2026-10-11

## Objective
Finish the three saved evaluation safeguards, then correct the separately
authorized warm-start test failure. Offline only; keep scoring stopped.

## Verified result
Frozen input receipts and v3 immutable plans are integrated. NULL comments mark
only their control unavailable. Every cached chunk checks parity before
publication; audit/report require passed receipts and tolerance <= 1e-4. Source-only
Claude cloud review completed; its findings were corrected and root-tested.
The warm-start test now checks the training objective against a precise reference
instead of assuming a solver tolerance bounds unseen predictions. Serving code
is unchanged. Full backend 1,429 passed / 1 skipped; Ruff/format/scoped ty clean;
global typing has only the same 11 absent optional inference imports. All 104
retained scores and snapshot digest match; actual v3 audit/report stop at zero
validated paired scores. Strands MainPID 0; live HN PID 531386 unchanged.

## Blocker
No authorized faithful scoring route. Real numerical parity/loading and held-out
quality remain unverified. Parity checks are per-chunk samples; receipt authenticity
is outside the contract. First prepare is the historical trust anchor.

## Next step
Keep Strands stopped and Colab skipped. When a faithful route is authorized,
populate every paired v3 job and require full validated coverage before metrics.
Evidence: GUARD-FIXES-20261011.md and WARM-START-CHECK-20261011.md.
