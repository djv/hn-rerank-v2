# Verified bounded ONNX diagnostic and result review — 2026-10-11

The user chose the existing VPS and explicitly approved the bounded older-model
check, then requested one independent `rv` after a daemon interruption.
Result review: https://claude.ai/code/session_01RqsUygvHUhFN716vErD5n4
Branch r/20261011-164101-hn-onnx-result; review commit ed13d7a.
No earlier review report, private context, raw request/token payload or credentials
was provided to this reviewer. REVIEW.md holds the independent assessment.

## Actual result

- Pinned candidate revision31a83e0244b26d939ae33f87189d6c8204003e53; q8 graph,
  external weights and tokenizer hashes verified; configs/public fixture hashes
  recorded in provenance.json. New isolated five-package environment uses
  onnxruntime1.31.0 and numpy2.3.3; original scorer environment unchanged.
- CPUExecutionProvider loaded the actual graph in1.819s. Candidate tokenizer,
  renderer, option positions and public reference token IDs matched exactly.
  Public fixtures passed the explicit0.03 diagnostic threshold with no answer
  changes, max probability drift0.02581483088153147. This is not1e-4 parity
  certification or HN efficacy evidence. The tokenizer regex warning was retained;
  no automatic tokenizer change was made against the pinned public references.
- Eight requests were prepared from four frozen stories: real lengths2248/2388/
  2563/2907; empty198/315/306/723. The first real batch4 did not finish. No private
  standalone check or empty batch ran; no per-score HN runtime exists.
- Approved unit started16:36:55UTC and exited16:39:31UTC, within20min. It used
  CPUQuota300%, Nice15, MemoryMax3758096384bytes(3.5GiB), SwapMax134217728bytes,
  no restart, OOMPolicy=stop and KillMode=control-group. ORT intra3/inter1,
  sequential execution and spinning disabled were checked in actual script.
- Watchdog stopped after five consecutive2s samples of increasing hard-limit
  events: last max652, OOM/OOM-kill0, peak3.5GiB, swap89493504bytes. Host available
  RAM was2836721664bytes at stop, above the1GiB guard. This is the prescribed
  resource stop, not an OOM or proof of host-wide exhaustion.
- Download page cache, weights and activations share that cgroup. Anonymous/file
  memory breakdown and PSI were not captured. Consequently the data do not prove
  the model cannot fit/run on the VPS, or establish an under/over5s HN score.
- Unit MainPID0 and empty ControlGroup; process check found no owned descendants.
  Full scorer inactive/MainPID0; live HN PID531386/start03:26:55UTC unchanged.
  Worker SHA remainsbc0f794696797a4f73ec028469a49c027722d3def3cebe5c20074f3baedd1c82.
  Zero scores published, no model adopted, service restart, paid route or phone action.

## Verified review findings

No serious scope/safety defect was identified or reproduced. Limits and cleanup
match the approval; reviewed script SHA256 matches execution provenance.

Two moderate evidence gaps hold:
1. run.py:102 installs the runtime by exact version but does not enforce or record
   the proposed wheel SHA256. packages.txt freezes actual resolved versions, but
   does not supply artifact hashes or a hash-enforced lock. Do not describe this
   runtime installation as fully artifact-verified.
2. run.py:57 overwrites the final memory sample, and logs lack phase timestamps.
   memory.stat/PSI and in-flight phase duration are absent; SIGTERM leaves result
   complete=false with the reason only in task.log. This limits diagnosis and
   should be addressed before a separately authorized retry is treated as decisive.
   No observability or stop-policy fix was applied in this read-only review.

The review's hypothetical duplicate pairing/first-question concerns were checked
against all4132 plan rows without inference: duplicate(fold,story_id,variant)keys0;
all4132 rows have exactly one question;2066real/2066empty. PLAN-SHAPE.json holds
counts only. These are not defects in the actual plan. Tokenizer warning impact
on other text remains unverified; exact public fixture tokenization holds.

## Disposition

Approved diagnostic is finished with a resource stop and incomplete HN timings.
No restart, cap increase, batching change, further model run or full scoring is
authorized by `rv`. Source syntax checks passed before launch; no production
source changed and no unrelated/full test suite was rerun. Review wait exited0.
Original scores/weights and this diagnostic's model/private prepared tokens are
retained; private prepared file is0600 under a0700 folder and was not uploaded.
The under-five-second goal remains unmet. Any retry needs an explicit revised
proposal and decision; this evidence supports neither impossibility nor readiness.
