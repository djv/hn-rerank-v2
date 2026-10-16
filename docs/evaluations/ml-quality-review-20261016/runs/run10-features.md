## 1. Top findings, ranked

**1. Fresh stories get a zero side-vector — penalizes exactly the newest cards.**
Category: defect. Evidence: `pipeline/side_embeddings.py` `model_space`/`side()` (`vectors.get(s.id, zero)`); STATUS 2026-10-16 confirms "side-vector norm shrinkage for uncovered stories." Below `side_embedding_min_coverage` (0.98, `pipeline/config.py`) the rerank falls back to stored-only, but above it the uncovered ~2% (disproportionately brand-new candidates the 30-min timer hasn't encoded) get a zero 768-d half, shrinking their norm by ~1/√2 and distorting every RBF distance involving them.
Effect: hurts stay-current and discovery (novelty tax on new stories); save-time roughly neutral otherwise; no sameness effect. Effort: S (mean-imputation, stored-only fallback per story, or encode-before-serve for uncovered candidates). Validation: `scripts/eval_ranker_variants.py` with/without zero-fill on a fresh-vote holdout (`--holdout-after`), metric top-12 up-rate + short-window (12h/1d) up-rate, plus coverage-vs-age audit. Confidence: high that the bias exists; medium on size.

**2. Positive-cluster feature has no LOOCV — the one similarity feature without self-exclusion.**
Category: defect (mild train leakage). Evidence: `pipeline/ranking.py` — k-NN training features use `_loocv_knn_features` (self excluded), but `fb_positive_cluster_sim` is computed against centers fit on all positives including the row itself. Candidates get no such boost, so the SVM overweight cluster membership and offline numbers flatter the cluster feature.
Effect: small save-time loss via overfit; neutral-to-positive for sameness (cluster features pull toward past up-vote blobs). Effort: S (exclude-self or out-of-fold centers). Validation: `eval_ranker_variants.py` temporal folds + newest-20% confirmation; expect AUC delta ≈ 0, judge on top-12 up-rate and discovery up-votes. Confidence: medium.

**3. `neutral_weight = 0.0` is a dead knob — neutrals train as a full third class.**
Category: defect (docs/code mismatch). Evidence: `pipeline/config.py:36` defines `neutral_weight`; nothing in `pipeline/` reads it — training weights come from balanced class counts in `pipeline/ranking.py` (`Counter(feedback_labels)`), so neutrals get full weight contrary to ARCHITECTURE §3.3's `neutral_weight=0.0` framing.
Effect: a 3-way boundary wastes capacity on the noisiest class; honoring 0 (or down-weighting) could sharpen up-vs-rest at the cost of the "unsure" signal — sameness roughly neutral. Prior note: neutrals-as-down gave no gain (STATUS 2026-10-11), but neutrals-as-zero-weight was not the tested variant. Effort: S. Validation: eval variants neutral-weight ∈ {0, 0.25, 1} on temporal folds + short-history curve; watch down-share of top 12 as guardrail. Confidence: medium.

**4. Double rank-normalization washes out margins.**
Category: untried improvement. Evidence: `pipeline/ranking.py` `_minmax01(raw_scores)` per pool, then `pipeline/linear_blend.py:blend_scores` re-converts production/dense/word scores via `percentile_scores` before the fixed 0.5/0.2/0.3 mix. Two rank transforms discard how strongly each model prefers a story; FINDINGS (2026-10-12 review, §"Short-window") already notes min-max breaks fixed score bars. Effect: save-time only; sameness neutral. Effort: S (blend z-scored raw margins instead; keep percentile as a variant). Validation: `eval_ranker_variants.py` head-to-head on 8 folds + newest-20% confirmation, primary top-12 up-rate, AUC secondary. Confidence: low-medium (hill-climb step 5 adopted percentile blend, but raw-margin blending was not the tested comparator).

**5. Unsure entropy uses uncalibrated SVM-softmax probabilities.**
Category: evaluation gap / untried improvement. Evidence: `pipeline/ranking.py` `_softmax_rows` over OvR margins; ARCHITECTURE §3.3 admits they are "not calibrated"; FINDINGS "Stacked ranker" notes the same. The joined challenger keeps fitted probabilities for Unsure, but production Explore does not use them.
Effect: mis-ranks the Unsure badge → worse Explore (discovery) and noisier reading-signal work. Effort: S (temperature-scale or reuse joined #2's P(up)−P(down) spread for Unsure only — no ranking change, trial-safe to evaluate offline). Validation: offline Unsure precision (up+neutral rate of Unsure picks) on temporal folds; later dwell/open correlation. Confidence: medium.

**6. Cold/new-profile feed is pure HN gravity — popular and same-y by construction.**
Category: untried improvement. Evidence: `pipeline/ranking.py` tier blend (`alpha_2 = n_feedback/tier2_blend_window`, SVM gate 20/20) leaves `tier1_scores` (HN gravity) dominant for new profiles; gravity has weight 0 for heavy voters, so this only affects the first-feed aim.
Effect: directly against discovery/calmer-habit for newcomers; save-time OK. Effort: M (source-prior cold deck, Discover-without-votes via cross-profile centers, or interest-round-robin from day one). Validation: cold simulation (truncate profile 151 to first N votes / held-out users), metric: novel up-rate + topic spread at N ∈ {0, 10, 20}. Confidence: low-medium (direction clear, magnitude unknown).

**7. ARCHITECTURE §3.3 still pins `C=0.1`; live is `svm_c=4.0` + blend.**
Category: evaluation gap (stale doc, not a code bug). Evidence: ARCHITECTURE "Current hyperparameters … `config.toml` pins `svm_c = 0.1`" vs `config.toml:37` `svm_c = 4.0` + `linear_blend_enabled = true` (deployed per FINDINGS "Linear blend live", `26b9474`).
Effect: future tuning reads the wrong baseline; no direct ranking effect. Effort: S (one-paragraph doc fix). Validation: none needed. Confidence: high.

## 2. Checked and fine, or already tried — dropped

- Raw-embedding scaling: correctly untouched by `StandardScaler` (meta columns only, clipped ±2.5) — conforms to the standing rule.
- LOOCV self-exclusion for k-NN mean/max: present (`_loocv_knn_features`).
- Stacked second-stage (GBM/ordinal/pairwise/LR over OOF scores): tried and rejected (FINDINGS "Stacked ranker"; pairwise also raised top-12 down-share) — not re-proposed.
- TF-IDF weight/C/text variants: swept, keep 1–2 grams + body + domain, 0.3 ≈ 0.4 (FINDINGS "TF-IDF tuning sweep") — not re-proposed.
- C/gamma hill-climb (C 0.1→2→4, γ variants) and stored+gemma 50/50 incl. 70/30 split: tried, current values kept (FINDINGS "Incremental ranker hill-climb" steps 1–10).
- Engagement metadata (points/comments as SVM columns): deliberately excluded after worse held-out ranking (ARCHITECTURE §3.2).
- Equal 1/√2 side-by-side weighting: the 70/30 alternative was tested and lost (hill-climb step 9).
- Title-only TF-IDF and body/comments-as-second-vector: tried, hurt P@12 (TF-IDF sweep; hill-climb step 6).
- Pairwise/listwise losses as a class: covered by the rejected `stack[pair]` arm; no new evidence to reopen before the trial ends.
- `neutral_weight`: only the as-down variant was tested; the as-zero variant (proposal 3) is the untried remainder.

## 3. Open questions

- What fraction of 12h/1d-window candidates currently rank with a zero side part, and what is their served-position deficit vs stored-only scoring?
- Does honoring `neutral_weight=0` shift Unsure/Discover composition (neutrals carry exploration signal) even if top-12 up-rate is flat?
- After 2026-11-05, can `deck_scores` (serve-time logging since 2026-10-12) support a raw-margin score bar, given per-pool min-max breaks fixed thresholds today?

