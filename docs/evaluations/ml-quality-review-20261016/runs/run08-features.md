## 1. Ranked findings

**1. Blend twice discards margin strength — recalibrate instead of re-percentiling.**
Category: defect. Evidence: `pipeline/ranking.py` `_minmax01` (SVM/tier scores stretched to [0,1] per pool) → tier blend → `pipeline/linear_blend.py:15-22,150-175` `percentile_scores` again in `blend_scores`. Each stage is monotonic within one rerank, so order per component survives, but cardinal gaps do not: a dominant SVM margin and a coin-flip count the same in the 0.5/0.2/0.3 average. Weights then don't mean what they seem.
Effect: weaker dud-cutting (save time) with no diversity benefit; sameness neutral. Effort: S. Validation: `scripts/eval_ranker_variants.py`, held-out-feedback folds, compare live blend vs z-scored / Platt-calibrated blend with learned weights; primary top-12 up-rate + down-share, guardrail novel-ups/week and non-HN share. Confidence: medium-high.

**2. Missing side vectors shrink the vector — penalizes exactly the newest/thinnest stories.**
Category: defect (confirms 2026-10-14 review). Evidence: `pipeline/side_embeddings.py:48-54,179-188` — uncovered story gets zero side part then `/sqrt(2)`, norm `1/sqrt(2)` vs 1 for covered stories. Coverage gate (`side_embedding_min_coverage` 0.98) still allows ~2% shrunk rows per rerank.
Effect: hurts stay-current and discovery (fresh RSS/Reddit rows most likely uncovered); sameness neutral-to-worse. Effort: S (renormalize covered part, or fall back to stored-only row). Validation: unit test + replay eval with simulated-missing side vectors, same metrics as above. Confidence: high (mechanism reproduced; live impact small since coverage is ~99.9%).

**3. `neutral_weight` knob is dead; neutrals get full balanced weight.**
Category: defect (docs/config). Evidence: `pipeline/config.py:36` `neutral_weight=0.0`, ARCHITECTURE §3.3 cites it, but `pipeline/ranking.py:1592-1600` builds `sample_weights` purely balanced over all three classes — no reference to the knob. Meanwhile up-vs-neutral is the weakest separation everywhere (probe AUC 0.63–0.67, FINDINGS 2026-09-28). Softmax-over-margins "probabilities" driving Unsure entropy inherit this.
Effect: dud-cutting blunted on the hardest boundary; uncertain. Sameness: unclear. Effort: S. Validation: `eval_ranker_variants.py` — neutral-downweighted / up-vs-rest-pairwise variant vs production on newest-20% confirmation + hand-order pairs (`scripts/calibrate_rankings.py`); watch top-12 down-share and Unsure hit-rate. Note: "neutrals as down" was tried once with no gain (FINDINGS 2026-10-11) — try graded weights, not binary folding. Confidence: high the knob is unused; low on which weight wins.

**4. Single flat text string dilutes title/domain signal.**
Category: untried improvement. Evidence: `pipeline/ranking.py:compose_story_text` (title + up to 6k/6k/4k chars, first-come truncation), `linear_blend.py:38-44` (`tfidf_text` = first 5,000 chars of already-composed `text_content`), side model 128 tokens (`pipeline/config.py`). Long comment threads bury the title; thin RSS snippets get little to go on (source-yield review 2026-09-26). Field-level embeddings are named in ARCHITECTURE §3.2 as untested-in-production, not rejected; two-vector body+comments gained AUC but lost P@12 and was dropped (hill-climb step 6).
Effect: helps discovery (thin/non-HN rows) and save-time; small sameness risk if lexical features overfit popular phrasing — mitigate with domain/source tokens already in TF-IDF. Effort: M (needs new embedding version + cache invalidation). Validation: `scripts/probe_embeddings.py` + `eval_ranker_variants.py --replay-embeddings` on title-upweighted / field-concat variants; require P@12 gain plus novel-ups non-decrease. Confidence: low-medium.

**5. Cluster-similarity training feature lacks the LOOCV self-exclusion the k-NN features have.**
Category: defect (small, uncertain). Evidence: `pipeline/ranking.py` `_loocv_knn_features` excludes self for `sim_to_up/down`, but `fb_positive_cluster_sim` is computed against `_positive_cluster_centers(fb_up_embs)` fit on the same rows including self — optimistic on training, unmatchable at serve time.
Effect: tiny train-serve skew; likely negligible at k=4 with thousands of votes. Effort: S. Validation: leave-one-out-center variant in eval harness; expect ~null, ship only if free. Confidence: low.

**6. Stale hyperparameter docs invite mistuned follow-ups.**
Category: defect (docs). Evidence: ARCHITECTURE §3.3 claims `C=0.1`, `config.toml` pinned; actual `config.toml:37` is `svm_c = 4.0` (live since `26b9474`, 2026-09-29). Effort: S. Validation: none needed. Confidence: high.

**7. Evaluation gap: judged-pool metrics can't see the thin-pool problem.**
Category: evaluation gap. Evidence: short-window check (FINDINGS 2026-10-12) — 12h/1d AUC 0.76–0.80 yet downs persist because the pool is thin; FEATURE-REMOVALS/BROADER already flag judged-only pools and min-max-bar non-transfer (`_minmax01` pool-relative). Next: after 11-05, replay rank-caps and `prob_down` bars on logged serve-time `deck_scores` (already planned in STATUS), scored by ups/session with downs/card guardrail plus novel-ups/week. Effort: M. Confidence: medium.

## 2. Checked — fine or already tried, not re-proposed

- StandardScaler touches only meta columns from `emb_dim:` onward with ±2.5 clip — raw-embedding scaling prohibition respected.
- Stacked second stage (GBM/ordinal/pairwise/logreg over OOF scores + metadata): rejected, raised top-12 down-share (FINDINGS "Stacked ranker").
- Engagement points/comments features: hurts (composite 0.635); publication affinity, recency weighting, training-dedup: no gain.
- TF-IDF 1–2 grams + domain/src tokens at weight 0.3 (0.4 ≈ tie): keep; title-only/char-grams/C/half-life swept, nothing better.
- C 0.1→4 + dense blend: shipped; gamma/knn_k/cluster-k sweeps flat.
- Side-by-side Gemma-300m: shipped; Gemma-2, Harrier, bge/arctic swaps: tie or worse.
- Joined-logistic #2/#4 and metadata-scale middle: in live trial until 2026-11-05; short-history curve favors challengers only at N≥200.
- MMR/caps/Far-1 diversity Phase 0 (no rule passed gate), Discover R1 rule, Teach active-learning sim (no-go), blind re-vote ceiling (+0.007, unsettled): respected as constraints, not re-argued.
- Pairwise/LLM feature mining (RANKING-REASSESSMENT-20261009): no supported signal; not re-proposed.

## 3. Open questions

- Which blend inputs does the trial favor — if #4 (no metadata) wins on top picks but loses wide AUC, is the right post-trial model a per-window or per-position blend rather than one global weight?
- Should Unsure entropy rest on uncalibrated SVM softmax at all, or on the joined classifier's fitted probabilities (trial will contain both)?
- For new profiles (first feed matters, research parked): does any proposal above transfer, or is the cold path (gravity + Discover) a separate problem needing its own eval split?

