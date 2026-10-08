# Claude Opus 5.5 read-only opinion

Verified CLI modelUsage: `claude-opus-5-5`. Review received 2026-10-07.

No bug confirmed in the running word-kernel screen. Routing, training-only IDF/min-df, candidate order and reuse of the fold database all look correct. The biggest untested lever looks like **block weighting**: by construction, the embeddings are only a small part of every kernel tried so far.

## Implementation findings in the current code
1. **Weak row alignment check.** `scripts/eval_single_preference_model.py:147-153` checks only row count and the label sequence, not story IDs. Today the rows line up because `:256-262` re-reads the same fold DB with the same dedup as `pipeline/ranking.py:1174-1183`. But the feedback SELECT has no ORDER BY (`database.py:1307-1316`), so this rests on SQLite returning the same order twice. Fix: record the story IDs ranking actually used and compare them with `LexicalInputs.training_ids`.
2. **Routing catches more than the challenger.** `:243-246` sends every `_production_scores` call with the blend disabled to the experimental kernel. That includes internal callers (`eval_ranker_variants.py:1475`, `:2793`). So a run can't include an "RBF alone" control, and with `word_c>0` a `prod[linear_blend_dense_weight=0]` control raises at `:252`. Keep each run to production plus one challenger.
3. **Leakage risk specific to words.** Test/train isolation uses normalized URLs only (`eval_ranker_variants.py:1406-1407`), so posts without a URL each get their own group. TF-IDF text also includes current comments (`linear_blend.py:43-49`, `ranking.py:593-594`). Exact-word features exploit same-title reposts more than embeddings do. Add a sensitivity row that excludes test stories whose normalized title matches a training title.
4. **Hard-to-read weights.** The dense kernel is divided by the mean squared norm (`:141-143`); the word kernel is raw cosine. Report `linear_scale` per fold so "effective word C" can be compared across runs.

The early result (linear + word C4) is the best of 8 settings, so it's optimistic until the 12 blocks finish.

## Can one classifier absorb all three roles?
- **The two logistic regressions: yes, exactly.** Dense LR and TF-IDF LR are linear models on separate feature blocks, so their sum is one linear model on the joined features. Each model's own regularization can be reproduced by scaling its block (scale s gives effective C·s²).
- **What is lost:** averaging three independently trained models, and the robustness of the rank-percentile blend.
- **The RBF's nonlinear role** is mostly already carried by the kNN/cluster metadata columns (`ranking.py:1013-1023`). That fits the finding that linear C4 ≈ RBF.
- **Linear + word kernels in one SVM** are the same as one linear model, so serving becomes a dot product (latency not measured).

**Estimate to verify cheaply:** the stored+Gemma vector has unit norm (`side_embeddings.py:54`), while ~10 standardized metadata columns add roughly 7–10 to the squared norm. So embeddings are about 10% of the linear-kernel diagonal and about 5–8% of the RBF exponent. That probably muted the earlier encoder comparisons.

## Proposed bakeoff
**Held fixed:** snapshot `912bd080…`, evaluation time `1791401987.9873054`, fold boundaries, production metadata construction and clipping, StandardScaler on metadata only (fit on training), balanced weights, production dedup flag, URL isolation, judged-only pool, exact blend control in every run, random_state 0. No new dependencies.

**Embedding weight w_e:** one scalar on the whole unit-norm embedding block. It preserves cosine geometry and isn't StandardScaler, but the user should confirm it fits the scaling rule.

- **Stage 0 (no metrics):** ID-alignment check; per-fold block sizes (mean squared norm and pairwise distance for embeddings, metadata, words); count of title-matched test stories.
- **Stage A (4 dev folds, all 3,848 votes, stored+Gemma1; 5–6 runs):**
  - A1–A2: linear SVM C4 + best word C from the running screen (4 if tied), w_e ∈ {4, 16}. w_e=1 is already being run.
  - A3–A5: one multinomial `LogisticRegression` (lbfgs, max_iter 3000, balanced weights) on [s·√w_e·emb, s·meta, tfidf], C=4, s=√(0.1/4)≈.158. This reproduces both production LR penalties in one joint fit. w_e ∈ {1, 4, 16}.
    - Score P(up)−P(down), as in `linear_blend.py:98-103`; prob_* from predict_proba (only the entropy view at `ranking.py:1787` uses them).
  - A6, only if the running RBF+word result is within .003 dev AUC of linear+word: RBF C16/γ.01 + best word C at w_e=4.
  - **Excluded:** gradient-boosted trees (1,152 dims plus sparse words on 3.8k rows), MLPs, fine-tuned encoders/SetFit (retraining per fold on the UHD GPU), ordinal LR (needs a new dependency), new encoders this round.
- **Stage B:** best SVM-family and best LR-family run by dev AUC → 8 historical + 4 recent blocks, paired story/block intervals, label shuffle for the finalist.
- **Stage C (only if the best w_e > 1):** finalist on the 3,234 text-hash cohort with stored only / stored+Gemma1-128 / stored+Gemma1-256 GPU vectors; same 3 folds plus the reserved latest 20%.

That's about 12 runs at most. Fits are CPU-only (minutes each), on the VPS or the laptop via `batch` on AC.

## Comparing embeddings on a shared cohort
- **Cohort:** feedback stories whose current text hash matches every vector set. Compute splits once on that cohort.
- **Candidates:** `restrict_feedback` filters feedback only. In judged-only mode that also fixes the pool; in current-pool mode candidates need the same filter.
- **Incumbent:** the blend on stored+Gemma1-128 inside the same cohort. Report only paired per-fold differences.
- **Caveat:** changing the encoder moves both the embedding block and the kNN metadata, so don't credit a gain to one channel.

## Decision rule (to be fixed in advance; user to confirm)
Accept a finalist if, over the 12 blocks:
- pooled AUC ≥ blend −.005;
- top-12 upvotes ≥ blend −3/144;
- downvotes ≤ blend +2/144.

If two finalists both pass, prefer the LR (simplest to serve, calibrated probabilities). These are practical tolerances, not proof of equivalence.

## Validation on future votes
Freeze the code SHA, configs and snapshot now (time T). Later, take a new snapshot, train both models on votes up to T, and test on votes after T (judged plus impression pools, including unvoted cards). The live blend chose what was shown, which favours it, so a tie leans toward the challenger. Don't tune on these votes.

## Leakage pitfalls
- Fit IDF, min-df, the scaler, kNN features and cluster centres on training rows only (the current code does).
- w_e and C picked on dev folds that overlap the 12 blocks make Stage B exploratory; never tune on the recent blocks.
- Same-title posts without URLs are not isolated.
- Comments are today's, not those present at vote time.
- Cohort filtering must also cover candidates.
