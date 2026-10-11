# Offline paired evaluation safety gates

Verified 2026-10-11 UTC (2026-10-10 New York), branch `eval-decision1`.

## Result

`eval_decision1.py` now prepares the entire selected frozen scope under real
and empty reader profiles, independently of legacy cache presence. `audit`
requires both profiles for every selected story. `report` stops before metrics
if any request is missing, malformed, stale or unversioned; missing candidates
are never silently dropped. Legacy Decision-1 reports are disabled until they
have an equivalent paired provenance contract.

`score_cache.py` binds schema, full state/question, model ID/revision, dtype and
worker SHA256. Scores must be finite numeric probabilities in [0,1]. Duplicate
JSON keys, booleans and NaN are rejected. Atomic create-only writes preserve
existing records and reject invalid/stale entries rather than replacing them.

`strands_worker.py` checks all job identities and existing outputs before loading
an engine. Adapter and base resolutions are immutable, local-only snapshots;
the base loader receives the pinned local path. Recorded package versions are
checked before model loading. The worker source digest binds these version pins
and the base-model revision. No environment or weights were restored.

Completed-score reports use one fixed blend weight, 0.2, and the same candidate
pool for production, real/empty profiles and popularity controls. The primary
comparison is raw Strands real-minus-empty within-fold pooled judged-story AUC,
with a paired story bootstrap. Repeated judged stories across folds are refused
because the bootstrap would otherwise require clustering. Undefined AUC remains
unknown. Secondary comparisons are exploratory; no adaptive blend selection,
confirmation-set access or automatic winner/deployment is performed. The weight
was fixed after earlier exploratory evidence; it is not an untouched confirmation.

## Live retained-artifact checks

- Frozen scope: 2,066 distinct stories across folds 410/414/413/414/415.
- Prepared jobs: 4,132, exactly 2,066 real plus 2,066 empty, including the 11
  stories omitted by the earlier residual-jobs preparation.
- Valid versioned scores: 0; missing: 2,066 per profile. The 104 legacy scores
  remain byte-for-byte preserved but cannot prove the new request provenance.
- Actual `audit` and `report` exit nonzero on incomplete coverage. No
  `report_strands_v2.json` was created; no efficacy metric was computed.
- Frozen snapshot SHA256 remains
  `a485783c87c599ef7b3f49b446a5cb98bedac50ba9a64444aab28f8bf7b13894`.
- Every candidate exists in the snapshot and has points. Twenty-six have missing
  comment counts. The refreshed free-controls command refuses these values rather
  than substituting zero; no new numeric controls report was produced. Earlier
  control outputs remain historical evidence. Popularity values are snapshot-time
  diagnostics, not vote-time causal features.
- Strands user service: MainPID 0, failed after its earlier authorized stop,
  exit status 143; no worker process. Live HN service remains active with its
  existing PID/start time; no restart/configuration/runtime changes.
- Dating labels and Random100 were outside this change and were not edited.
  Colab remains skipped. No provider calls, installs/downloads, inference or
  deployment occurred.

Private job files and profiles stay under `/tmp/decision1-eval/work_heldout/` on
the VPS and are not committed. The immutable prepared job plan is under
`paired_plans/9e685425b71e14dc681c4300d3348405180091f105aa9b5be9e9fc581ff61dd4/`.
The machine-readable audit is `cache_audit_v2.json` in that work directory.

## Verification

- Full backend suite: **1,362 passed, 1 skipped, 1 existing sklearn convergence
  warning**, 22.74 seconds, VPS, four pytest workers and one library thread each.
- New cache/worker/paired-report synthetic coverage: 94 tests; worker tests
  substitute engine/library objects and do not exercise real inference.
- Repository Ruff clean; all eight relevant Python files format-clean.
- Strict typing clean for evaluation/cache/control modules and all four test
  files. Worker typing passes with absent optional imports excluded.
- Full strict typing reports only the same 11 absent torch/Strands imports from
  the intentionally removed inference environment; prior draft had 24 diagnostics.
  No new type diagnostics. Live loading, numerical parity or speed of this new
  worker version were not tested because the model environment remains absent.

## Next step

Preserve the stopped state. Before future efficacy reporting, obtain an explicitly
authorized scoring route, populate complete versioned paired scores, and resolve
the missing-comment control policy using the frozen data without fabricating
values. Keep the prior CPU timing/parity evidence separate from this gate change.
No claim about held-out HN quality or accuracy follows from these checks.
