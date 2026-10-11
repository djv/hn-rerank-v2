# Separate warm-start regression correction — 2026-10-11

The user explicitly included this separate fix after the full HN safeguard suite
exposed tests/test_joined_classifier.py's existing assertion. Seed 134217729 failed
both the modified tree and a clean 4ce5fcc baseline. Unseen candidate probabilities
differed by 0.00669789, exceeding the test's assumed 0.005 bound. No production
classifier code was changed.

The assertion conflated lbfgs's gradient stopping tolerance with a bound on unseen
predictions, and promised fewer warm iterations for every random dataset. Neither
follows from a convex objective. At the failing seed cold/warm used 12/2 iterations;
normalized regularized objectives were 0.1280928976 / 0.1281029286 against a
tol=1e-8 reference of 0.1280904357. Both default fits have small objective error,
despite the larger unseen probability difference.

The property now compares both default fits' mean multiclass log loss plus
L2/(2*C*n) with the more accurate reference, within 1e-4 absolute objective error.
It also requires finite normalized candidate probabilities. Explicit examples
retain the discovered seed and seed 0; the latter still checks the representative
warm-start speed improvement (12 cold / 7 warm iterations). No universal speed or
prediction bound is asserted, and no test tolerance on the old prediction check
was widened. This is a test correctness correction, not a serving/model change.

Verification: affected checks included in 173 focused tests; full backend
1,429 passed / 1 skipped; Ruff/format/scoped ty clean. Production and the active
interleaving trial were not restarted or redeployed. This test-only change is a
separate commit from the offline evaluation safeguards.
