# Representation-input harness: status

## 2026-10-09: bounded step (implementation + baseline gate)

- Implemented (this dir only): `PLAN.md` (frozen protocol), `harness.py`
  (`manifest` / `validate` / `compare`-refuses / `self-check`).
- Local: `ruff check`, `ruff format --check`, `ty check`, `self-check`
  (see below). No runtime code touched; no laptop heavy work.
- VPS: baseline probability reproduction only (no encoding, no fits
  beyond the 4 baseline folds, no full two-arm run).

### Baseline result (VPS)

- Goal: more upvotes through a repeatable Muse advantage that improves ML.
- Original saved-reference gate failed: max probability difference .4344247878.
  Root diagnosis: numerical-library thread initialization differs. Original
  script with explicit one-thread shell settings matches the new harness exactly.
- Frozen one-thread reference gate PASSED: 3,079 rows, max probability and
  production-score difference 0.0; no missing/drifted vectors; warm store off.
- Reports: validation_report_original_reference.json and
  validation_report_threads1.json; amendment and exact reference in PLAN.md.
- Headroom: load .49, ~5 GB available RAM before validation; no live writes.
- Checks: artifact self-check, Ruff/format/ty passed. VPS source regression
  suite: 1,213 passed, one skipped, 65.75 s, niced/one CPU.
- Root optimized primary bootstrap AUC/pair counting and corrected manifest
  effective-input hashes and changed-input count after baseline validation;
  validation scoring logic is unchanged. New helper checks passed.

### Not started (explicitly deferred)

New embedding inference, challenger vectors, full two-arm fits, heavy
test sweep.

Next: manifest counts and deterministic full-text encoder parity, then implement
the actual isolated encoding/comparison stage under the frozen upvote protocol.
No ranking improvement or prospective validation is established yet.

## 2026-10-09: parity + challenger encode (user-authorized next stage)

- Implemented (this dir only): `harness.py` now has `parity` (20-row
  full-text gate), `encode` (resumable challenger vectors), `compare --arm`
  (one-arm fits, real scorer), `report` (frozen gates). `compare` without
  challenger vectors still refuses. Root-required settings preserved:
  `ENCODE_BATCH=1`, `ENCODE_CHUNK=16` (shared-VPS memory safety),
  `_guard_runtime_headroom` (refuses when load > 4 or available RAM < 2 GB),
  `guardrail_up_not_down` pre-result gate.
- Local: `ruff check`, `ruff format --check`, `ty check` clean;
  `self-check` 13 groups passed. No runtime code touched; no laptop heavy work.
- VPS manifest re-verified with new harness: n=3848 matched=3797
  mismatched=51 clean_body_empty=509 changed_inputs_if_encoded=3237
  (matches frozen `manifest_threads1.json`).
- Parity20 (VPS, explicit `OMP/OPENBLAS/MKL_NUM_THREADS=1`, nice, ONNX
  intra_op=1): PASSED. 20/20 rows cosine >= .9999 vs stored vectors, min
  0.9999998212. 27.5 s encode (1.376 s/row), linear projection 4455 s
  (~74 min) for 3237 rows. Report: `parity_report.json` (this dir).
  Parity ran under harness sha16 `e1bc2da4029f3f11`; later changes are
  resource comments/helper/report guardrail only (current `c6dbeef4...`).
- Challenger encode job RUNNING (task-owned, no duplicate): PID 3989858,
  started 2026-10-09 15:36 UTC, `nice -n 10`, thread vars verified 1/1/1
  in `/proc/3989858/environ`, VmRSS ~686 MB. Log:
  `/tmp/opencode/representation-input-20261009/encode.log`. Out dir:
  `/tmp/opencode/representation-input-20261009/challenger/`
  (`challenger_vectors.npz` + `challenger_done.json` + `encode_report.json`
  on completion). Progress at 15:38 UTC: 272/3237 rows (~0.33 s/row
  observed, well under projection). Headroom at launch: load 0.83,
  ~4.3 GB available.
- Pending: encode completion, then fresh-process `compare --arm baseline`
  and `--arm challenger` (task-owned snapshot copies, warm store off,
  fixed `eval_now`), then `report` (frozen gates + conditional 2000-rep
  bootstrap). No ranking improvement established yet.

## 2026-10-09: follow-on runner waiting (15:43 UTC)

- Encode progress: 1408/3237 rows (~43%) at 15:42 UTC, PID 3989858 healthy,
  load ~1.1, ~4.4 GB available. No duplicate encode launched.
- Follow-on runner launched (task-owned, durable): PID 3991658, nohup +
  `nice -n 10`, explicit thread vars 1/1/1. Script:
  `/tmp/opencode/representation-input-20261009/run_after_encode.sh`
  (local copy in this dir). Log:
  `.../followon.log`. Status:
  `.../followon_status.json` (currently `runner_start`, waiting).
- Runner behavior: verified config sha16 `99a2ad40f7f9973f` and froze
  `config_frozen.toml` in the task dir for both arms; encoder hashes match
  frozen (`9ba32b8568b5bf0a` / `da0e79933b9ed517`). Waits for exact encode
  PID, verifies `encode_report.json` complete + 3237 vectors, then runs
  baseline and challenger compares (fresh processes, separate snapshot
  copies, `taskset -c 1` if supported) and `report`. Exits factually on
  encode failure / resource pause (load > 4 or avail < 2 GB) / phase
  failure, no restart loops; gate failure is recorded, not retried.
- Clarification (docs only, running source untouched): the harness comment
  suggesting wider batches would not improve wall time is unsupported and
  withdrawn. `ENCODE_BATCH=1` / `ENCODE_CHUNK=16` are required
  memory-safety choices for the shared VPS, nothing more.
