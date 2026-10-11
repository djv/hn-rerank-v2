# Archived save-now HN handoff

Superseded 2026-10-11: the safeguards are implemented and tested. Current result,
review corrections and limits: [GUARD-FIXES-20261011.md](GUARD-FIXES-20261011.md).
Current objective/blocker: [STATUS.md](STATUS.md). The original save boundary below
is historical evidence, not the current task status.

# Save-now handoff: confirmed HN evaluation defects

The user requested `fix`, then an independent `rv`, then `ss now`. Work stopped
at the save boundary. The three requested fixes are not complete.

## Verified current state

- Worktree: `/home/dev/hn-rewrite/hn-rewrite-decision1`, `eval-decision1`.
- Strands remains stopped: user service MainPID 0, failed after status 143.
  No model environment was restored; Colab remains skipped.
- All 104 retained legacy cache files match the pre-fix checksum list. The
  1,190,858,752-byte snapshot SHA256 remains
  `a485783c87c599ef7b3f49b446a5cb98bedac50ba9a64444aab28f8bf7b13894`.
- Existing real/empty plan contains 4,132 jobs, but has no validated paired
  scores. There is no supported held-out Strands quality/accuracy conclusion.
- Live HN user-service PID 531386 remained active across this save. The agent
  did not restart, deploy, or score. Dating labels and Random100 were untouched.

## Confirmed defects, still pending integration

1. Twenty-six candidates have NULL comments. That currently prevents every
   complete report, although points and primary paired metrics remain possible.
   Permit NULL comments only as explicitly unavailable comment controls; keep
   all candidates in primary/model/points comparisons. Reject absent rows,
   missing/nonfinite points, and nonfinite non-NULL comments.
2. A synthetic worker returns cached .95 versus plain .05, logs +.9000 and
   still publishes accepted scores. Add finite nonnegative parity tolerance
   (proposed 1e-4), at least one check per profile group, and validate checked
   scores before publishing any chunk. Wrapper must explicitly enable checks.
3. Changing only labels previously changed primary AUC difference +1 to -1
   while cache coverage stayed complete; re-prepare reused the job path and
   rewrote its receipt. Bind folds and snapshot to create-only receipts and
   immutable plan manifests, and validate them before audit/report metrics.

These are reproduced vulnerabilities, not proof of corruption or wrong real
model scores. Conditional prompt-order/package/base-model questions from the
prior review remain unresolved; no inference environment was rebuilt to test them.

## Saved, tested component

`frozen_inputs.py` and `tests/test_decision1_frozen_inputs.py` implement streaming
SHA256 validation and strict create-only `work/frozen_inputs_v1.json` receipts.
API: `freeze_inputs(work, snapshot)`, `read_frozen_inputs(work)`, and
`validate_frozen_inputs(work, snapshot=None, require_snapshot=False)`.
Initial freezing requires folds and snapshot; reports must require the snapshot.
Receipts are never rewritten. Input hashing detects modification/replacement.
Receipt reading tolerates ctime changes from concurrent temporary hardlink
cleanup while still checking inode, size and mtime.

This component is not called by prepare/audit/report yet. No real-data receipt
has been created. Existing plan and manifests remain untouched.

Focused component tests: 32 passed. Full backend: 1,394 passed, 1 skipped,
one existing sklearn convergence warning. Repository Ruff and changed-file
format pass; new module/tests strict ty pass. Global ty retains the same 11
optional Torch/Strands import diagnostics; no new diagnostics.

## Resume

1. Integrate the saved receipt API into prepare/audit/report. Preserve old
   manifests; use a new `paired_plans_v3` namespace, and never rewrite receipts.
   Require `--snapshot` on first prepare and every report. New manifests/report
   must carry trusted folds and snapshot hashes, not self-recomputed claims.
2. Implement missing-comment policy and parity failure gate above. Add
   regressions for changed labels/snapshot, repeated prepare, NULL comments,
   mismatch/nonfinite scores and no cache publication on parity failure.
3. Run focused/full checks and the already authorized independent `rv` of
   completed changes. This result review has not run; save-now superseded
   further implementation/review launches. Do not call all fixes complete.
4. Retain stopped state until a faithful scoring route is authorized. Preserve
   existing scores, snapshot, labels and Random100. Offline only, no deployment.

Private original review/reproduction evidence is retained on the laptop in
`~/.local/share/phone-private/evaluation/20261010-hn-gates-rv/`; synthetic worker
proof also exists on VPS at `/tmp/hn-gates-rv-verification.txt` and
`/tmp/test_hn_gates_rv_verification.py`. Do not upload personal histories, profiles,
scores or database contents to a reviewer or Git.
