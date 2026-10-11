# Completed offline HN safeguards — 2026-10-11

Scope: the three saved HN review defects, resumed from the phone project with `rr`.
Offline only. No inference environment, provider calls, model download, Colab,
deployment or service restart. The separate warm-start test correction is described
in WARM-START-CHECK-20261011.md; production classifier settings are unchanged.

## Current contract

- First `prepare --snapshot PATH` creates `frozen_inputs_v1.json`. This is the
  **first-prepare trust anchor**, not retroactive proof of fold generation or of
  historical authenticity. Existing receipts never rewrite; every subsequent
  prepare/audit/report validates folds, and a report also requires the snapshot.
  `cmd_folds` publishes its output create-only and refuses an existing fold scope.
- `paired_plans_v3/<manifest-sha256>/` contains create-only jobs and manifest bytes.
  The manifest binds the frozen input digests, complete job scope, selection options,
  fixed blend weight and worker identity. Audit/report require this exact plan.
  Equivalent relative/absolute work paths resolve to the same plan.
- Reports read the exact receipt-bound fold bytes. Snapshot reads compare their
  hash directly to the receipt and before/after reading; SQLite WAL/journal/shm
  files are refused. Inputs/plan are revalidated before result publication.
- NULL comments make the entire selected-pool comment control unavailable, with
  missing IDs/count explicitly recorded. No imputation or candidate removal.
  Primary/model/production/points metrics retain the complete candidate pool.
  Absent rows, missing/nonfinite points and nonfinite non-NULL comments still stop.
- Every prefix-cached chunk checks at least one row against the plain path before
  publishing any score in that chunk. Checks use the end of each chunk, including
  its longest row after length bucketing. All chunk probabilities are validated
  before any write. CLI tolerance must be finite, nonnegative and <= 1e-4.
  The wrapper explicitly passes `--check 1 --parity-tolerance 0.0001`.
- Every score records plain provenance or a per-chunk parity receipt binding
  request digests, checked differences and tolerance. Worker preflight, audit and
  report reject missing/failed/loose parity receipts. This remains **sampled
  parity**, not a guarantee that every unselected score equals plain inference.
  `--check N` checks N end rows per chunk; `--check 0` is intentionally unsupported
  for all generic-worker callers. More plain comparisons increase CPU cost.
- Reports/audits publish atomically with fsync. A failed invocation preserves any
  earlier result; consumers must check that result's embedded plan/input digests.
  v2 results and all legacy cache files remain historical, never a v3 fallback.
  Receipts do not authenticate against an actor replacing all files consistently.

## Independent review and corrections

Claude cloud session https://claude.ai/code/session_01VaP2FCDF66FE2hmxDSFsx1,
mailbox branch r/20261011-041544-hn-offline-gates-20261011. Source/tests/diff only;
no private inputs, labels, profiles, scores, database or secrets uploaded. Static
review; no reviewer-run tests or inference. Raw review is retained privately on
the laptop at evaluation/20261011-hn-resume/REVIEW.md.

Review confirmed NULL-comment handling and receipt binding after first prepare.
Its medium parity finding was verified against code: only the first group chunk
was checked and checks were not recorded. Corrected with checks on every chunk
and mandatory cache receipts. New regressions cover a passing first chunk and
diverging later chunk, plus loose/missing/failed receipts stopping before metrics.
Also corrected path spelling, snapshot substitution, create-only fold publication,
ctime-only hardlink cleanup checks, atomic derived outputs and CLI documentation.
The first-prepare anchor and sampled-parity limits are explicit above. Missing
points remain a deliberate stop required by the saved handoff. No second reviewer
was launched; the final corrections were verified by the root agent's tests.

## Verified retained state

- Final real plan: 4,132 jobs / 2,066 candidates per profile. v3 directory
  e37bdffbed7255febbada9b6e5b1a34479be700f6630b73a2cf8761e33ea5f6a.
- Actual prepare succeeds; audit and report reject incomplete paired coverage
  before efficacy metrics. Valid paired scores: 0. No HN accuracy conclusion.
- All 104 legacy scores match RETAINED-CACHE.sha256.
- Snapshot hash remains a485783c87c599ef7b3f49b446a5cb98bedac50ba9a64444aab28f8bf7b13894.
- Strands `strands-hn-heldout.service` MainPID 0; live `hn_rewrite.service`
  MainPID 531386 / start 2026-10-11 03:26:55 UTC remains unchanged.
- Full backend: 1,429 passed / 1 skipped / existing sklearn convergence warning.
  Focused checks: 173 passed, plus 73 cache/worker tests after the final parity
  receipt assertions. Repository Ruff, touched-file formatting and whitespace
  clean. Scoped strict ty clean; global ty retains only the same 11 absent optional
  Torch/Strands imports. No numerical inference or live loading certification.

## Next step

Keep scoring paused until a faithful scoring route is authorized. Populate the
entire v3 paired scope before reporting. Keep the examined dating confirmation
cohort and all live/frozen model artifacts outside this work.


## Verified review follow-up: snapshot-read race

The second independent source review reproduced a narrower remaining race:
the snapshot could be replaced after the first expected digest and restored
before the second digest, so unchanged bytes concealed different values read
by SQLite. The agent independently reproduced it with temporary synthetic DBs.
The user selected only serious, verified fixes; this metric-correctness defect
was selected. Conditional padding concerns and lower-impact result retention /
pre-existing CLI-default findings were excluded from this change.

The reader now compares device, inode, size, mtime_ns and ctime_ns before and
after the complete hash/read/hash interval. Restoration cannot restore ctime;
replacement changes identity. Existing digest and sidecar checks remain.
One parameterized regression covers in-place modification/restoration and inode
replacement/restoration, verifies rejection before report publication, and
confirms original bytes are restored. Both cases failed before the fix.

Verification: 57 paired-report tests; full backend 1,431 passed / 1 skipped /
existing sklearn warning, 26.96 s. Ruff, touched format, scoped ty and diff check
pass; global ty is byte-identical to the earlier 11 optional-inference-import
diagnostics. No live DB, snapshot, retained scores, model loading or services
were changed. Cloud review: https://claude.ai/code/session_01LKV23uGhU6mKqUD6BcAVsE.
