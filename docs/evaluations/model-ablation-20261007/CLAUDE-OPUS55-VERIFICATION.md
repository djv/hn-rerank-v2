# Independent verification, October 7

Verified CLI modelUsage: `claude-opus-5-5`; completed successfully.
This is the actual reviewer response. Some documentation was updated
while the review ran; follow-up corrections are recorded separately.

# Review: one-classifier offline experiments (read-only)

**Verdict:** No leakage or scoring bug turned up in the paths that actually ran. The completed `broad-final-summary.json` agrees with the raw reports. Provenance is weak, though, and the narrative docs are stale, so the work can't be called verified yet.

## Confirmed defects

1. **Medium: provenance.** `eval_ranker_variants.py:3072-3074` hashes `git diff HEAD`, which ignores untracked files. Both experiment scripts and `compare_eval_scores.py` are untracked. `eval_one_classifier.py:228` hashes only itself, not the helpers it imports (`:31-38`). The raw reports show the code changed in the middle of the grid:
   - Logistic weight 1 and 4 ran under driver `f06d9358`; the selected weight 16 ran under `69f6546c`.
   - The SVM screen ran under `492ffa05`; the SVM checks under `4412b56f`.
   - The dirty-diff hash also moved (`0548…` → `45d2…` → `65e0…`).
   - Replay files are recorded by path only, with no content hash.
   - **Consequence:** the challengers in one selection grid didn't run the same code, and the runs can't be reproduced. Matching production controls doesn't show the challenger code was the same.
   - **Fix:** hash every experiment module, the orchestration scripts and the replay files. Rerun logistic weight 1/4 (and the SVM screen) under the final drivers, or show that the driver changes don't affect scores.
2. **Low–medium: `compare_eval_scores.py:226,237-240`.** Each run's label is the file name up to its first hyphen. For the `/tmp` files, the development and fresh dumps get the same label, and the fresh result silently overwrites the development one. `/tmp/hn-broad-final-results-20261007.py` calls `_report` directly, so the current numbers are unaffected. Running the script itself on these files would give wrong per-period results. **Fix:** label by full file name, or assert labels are unique.
3. **Low: test gaps.** In `tests/test_one_classifier.py:91-177`, 12 up/12 down gives a nonlinear-model share of (12−10)/60 ≈ .03. The served score is then mostly the older gravity/centroid tiers, and the test never checks the final ranking. Also missing:
   - a routing/parity test for `eval_one_classifier` (only the SVM script has one);
   - tests that the feature subsets pick the right columns;
   - a test that zero-filled placeholder rows for a missing class fail closed;
   - a test of the deduplication path (every run used `deduplicate_training_feedback=false`).

## Checked and sound

- **Word/numeric row alignment:** the same frozen rows and dedup function are used, and labels are checked (`:178`). Candidates use the same list in the same order. Placeholder rows raise an error, and `eval_ranker_variants.py:276` then aborts the fold. A small hardening: also compare the stored `training_ids`.
- **Fitted on training rows only:** word minimum document frequency and IDF (`eval_single_preference_model.py:147-155`), PCA/SVD (`eval_one_classifier.py:134-167`), and the metadata-only scaler (`ranking.py:1475-1481`).
- **Vector cache:** keys are content hashes of pure-function inputs; labels enter only through which rows are selected.
- **P(up)−P(down):** the up column is replaced (`:204-206`), probabilities are kept (`:273-280`), and min-max scaling is monotone (`ranking.py:1556-1558`, float32). In every fold the minority-class count is far above 80, so the served order is exactly the P(up)−P(down) order. The signs of the SVM's pairwise margins match scikit-learn.
- **Parity:** the final summary asserted identical IDs, labels and production scores against `check-linear-*`, `screen-embedding-stored-gemma1` and `check-embedding-confirmation`, and that test IDs don't repeat.

## How to read the results

- **Reused labels:** the 8 historical blocks test the same period as the screen that picked weight 16, scale .158 and C4, so they are in-sample for selection. The 4 recent blocks were already used in earlier checks. No block is untouched.
- **Joined logistic model vs current blend, 12 blocks:**
  - AUC +.0113; story interval [+.0044, +.0180]; block interval [+.0012, +.0228].
  - The exact sign-flip test gives p = .065 (9 of 12 blocks better). A percentile bootstrap over 12 blocks is too optimistic.
  - Top-12 upvotes 94 → 97, downvotes 2 → 6 (1 → 3 in each period).
  - There's no correction for the roughly 17 + 30 settings searched.
- **The feature ablations undercut the "replace both" story** (development folds, fixed settings, compared against the blend, not against the selected model):
  - Without metadata: AUC .8214, 40/48 upvotes, 1 downvote. That beats the selected model (.8191, 37/48, 2) on every metric.
  - Without words: .8200. Words alone: .7982.
  - So the word features add little inside the joined model. This is a hypothesis; it hasn't been checked on the 12 blocks.
- **Metadata influence is overstated.** The audit's mean squared norms (embeddings 16, metadata ~8.55, words 1) measure penalty/kernel scale, not influence. They are uncentered, and 5 of the 10 metadata columns are similarities computed from the embeddings.
- **The "merge exactly" claim is wrong** (`CLAUDE-OPUS55-REVIEW.md:16`). Percentile ranks of separately normalized scores are nonlinear and depend on the candidate pool. Even in logit space, one joint fit is not the sum of two separate fits. Production's two models also train on different rows and weights (`linear_blend.py:150` vs `:162`). The correction at `SINGLE-MODEL-PLAN.md:151` still overstates this: the joint model can represent the sum, but that is all.
- **Title audit:** it works as a sensitivity check and was run: 10 of 2,231 stories removed, AUC difference unchanged. It only catches exact normalized titles, though, so near-duplicates and URL-less reposts are untested.
- **Label shuffle:** .511/.530 only checks that labels are routed correctly. A global shuffle can't detect duplicate or current-content leakage. Comment text written after the vote is still an open question.
- **Matched cohort:** the selected embedding is the incumbent stored + Gemma 1, so there's no embedding-model win. That check also pools the 3 selection folds with the 4 reserved ones. Keep the 3,234-story cohort and the 3,848-profile metrics separate.
- **Unequal tuning:** only logistic got block-scale tuning, and weight 16 sits at the edge of the grid. The tree and neural models were limited to compressed (projected) inputs.
- **Stale text:**
  - `SINGLE-MODEL.md:4-5` still names the linear SVM as strongest, and `:144` says the sweep is in progress.
  - `STATUS.md:38-39` still calls the summaries pending.
  - The claimed full-suite result of 1,120 passed / 18 skipped isn't in the `/tmp` logs. Those show 1,123 passed / 1 skipped, plus an older log with one failure.

## Needed before calling it verified

1. Fix provenance (item 1) and rerun or reconcile the mixed-driver grid settings.
2. Declare in advance a paired 12-block check of the no-metadata and no-words variants against the selected model, labelled exploratory.
3. Add a near-duplicate audit (embedding cosine or fuzzy titles) and a training-only label permutation.
4. Report the reserved-only matched result, and use the sign-flip test as the main block-level test.
5. Add the missing tests and update the documents.

**Not verified:** tests weren't rerun; score-array parity relies on the script's assertions; replay file contents, driver differences, and comment text after the vote weren't checked. There's no evidence for calibrated probabilities, equivalence, live quality or performance.
