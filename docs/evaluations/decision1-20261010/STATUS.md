# Decision-1 / Strands offline evaluation — save-now checkpoint

Objective: save the current reviewed/fix state at the user's `ss now` boundary;
keep Strands stopped, skip Colab, preserve existing data and avoid deployment.

Verified result: the initial paired cache/report gates are committed. The new
create-only frozen-input helper and its tests are saved, but not yet integrated
into prepare/audit/report. Helper tests 32 passed; full backend 1,394 passed /
1 skipped; Ruff/format pass; scoped ty clean and zero new global diagnostics
(11 existing optional Torch/Strands imports). All 104 legacy scores and the
SHA-verified snapshot remain unchanged. Strands MainPID 0; no inference launched.
The live HN service stayed at PID 531386 during this save, without agent restart.

Blocker: three reproduced review findings remain unfixed in the evaluator/worker:
NULL comments block primary reporting, parity mismatches can publish scores,
and mutable labels/snapshot lack receipt enforcement. All 4,132 paired jobs
remain unscored under the strict contract, so held-out quality is unknown.
There is no authorized fast scoring route; model environment remains absent.

Next step: integrate frozen-input receipts and immutable plan manifests; make
NULL-comment controls explicitly unavailable while preserving primary metrics;
reject parity mismatches before cache publication. Then run the authorized
independent result review, which was not launched before save-now. Do not resume
scoring or deploy. See [REVIEW-FIX-HANDOFF.md](REVIEW-FIX-HANDOFF.md).
