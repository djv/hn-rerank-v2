# Focused verification of corrections

Verified CLI modelUsage: `claude-opus-5-5`; completed successfully.
Actual read-only reviewer response follows. Feature checks and final
gates were still running when this review was received.

**Verdict:** No implementation blocker found. Corrections 1, 2 and 4 are fixed. Correction 3 is enough for the four old-driver runs, with one small gap in the provenance. The item 5 checks are still running.

1. **Fixed.** `compare_eval_scores.py:226-228` now labels each period by its full file stem and rejects duplicate or `pooled` stems. The test asserts that the development, fresh and pooled results all survive.
2. **Fixed.**
   - The model gets the full tier-3 share: 12 (or 11 after dedup) minority votes, a start of 10 and a window of 1 give α3=1, and α2 is also 1.
   - The served score is asserted to equal `_minmax01(P(up)−P(down))`, the production control is unchanged inside the adapter, and both dedup settings are checked through `training_ids`.
   - The feature-subset arithmetic is correct. The placeholder test mirrors `ranking.py:1449-1472` and raises before `estimator.fit`.
   - The focused log shows 40 passed.
   - Minor gaps: the placeholder test doesn't run through `ranking.py` itself, and the projected (histogram/MLP) subset path is untested.
3. **Sufficient for these four runs.** Every other broad report used drivers `69f6546c`/`4412b56f`. The reruns used exactly those versions and assert exact equality of IDs, labels and every score (challenger and production). The summary is written only after those checks and the post-run hash check pass. Two gaps remain:
   - The hash inventory covers only Python, lock, replay and DB files. The tracked-diff hash changed partway through (`b78a…` → `aefb…`) without tripping it, probably from doc or test edits.
   - The selected-model dumps the new checks are compared against (`broad-check-adapter-*`, tracked diff `65e0…`) predate the manifest and weren't rerun. The launcher checks only production-score parity against them. Drift is unlikely, but a `features=all` rerun with an exact-parity check would close this cheaply.
4. **Fixed.** `SINGLE-MODEL-PLAN.md:151-157` now states this correctly, and `BROAD-ONE-CLASSIFIER.md:34` qualifies metadata norm versus learned importance.
5. **In progress.**
   - The plan file was written before the runs started. The launcher checks ID, label and production parity and that no test story repeats across the 12 blocks, with sign-flip as the main block test.
   - The first run, `feature-removal-numeric-development.log`, is at fold 3 of 8.
   - The four comparisons have no correction for multiple tests.
   - The type gate tolerates one error (`inspect_tldr_failures.py:86`); if it fires, report it rather than calling the check clean.
   - The format check covers only 7 files, not the other modified tracked files (`ranking.py`, `linear_blend.py`, `render.py`, `eval_ranker_variants.py`, tests).

**Unverified:**
- No scores or hashes were recomputed; the parity results rest on the scripts' assertions and the artifacts existing.
- The feature-removal results and the final tests, lint, format and type checks aren't finished.
- Still open: near-duplicate leakage, comment text written after the vote, calibration, equivalence, live quality and performance.
