# HN paired scoring — stopped 2026-10-11

## Objective
Evaluate the frozen held-out real/empty preference comparison. The user rejected
the CPU runtime estimate ("no", "too slow"), cancelling this scoring run.

## Verified result
The independent cloud plan review completed and all 16 batch4/check4 pilot
comparisons passed (maximum difference 4.6373844e-5, tolerance 1e-4).
The full run published another 40 scores with passing sampled parity before stop.
The log therefore records 56 total published scores including the pilot; this
stop did not perform a fresh cache audit. Full 4,132-score coverage is incomplete.
No held-out ranking quality result was generated.

The agent stopped only its own hn-strands-full-20261011.service. Verified user
unit inactive/dead, MainPID=0, and no strands_worker.py/eval_decision1.py process.
Validated caches, frozen input, model files, and logs were retained.
Live hn_rewrite.service remains active, PID531386, start03:26:55UTC unchanged.
Source unchanged; last source suite 1,431 passed / 1 skipped.

## Blocker
This free CPU route is too slow for the user. First full-run chunks took roughly
108–114 seconds per four scores, including one full-prompt parity check.
The earlier shorter estimate was optimistic. No complete efficacy evidence.

## Next step
Keep CPU scoring stopped; do not automatically resume. Select and benchmark a
faster route before a new full run. Paid GPU/API use, Colab, confirmation access
and deployment remain unauthorized. Execution facts: SCORING-20261011.md.
