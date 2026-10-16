## 1. Top findings, ranked

**1. Future text leaks into retrospective features — biggest offline bias.**
Category: defect. Evidence: `scripts/eval_ranker_variants.py:_load_production_candidates` (~564), `_validated_embeddings`, `story_embedding_text` (`pipeline/ranking.py:story_embedding_text`); candidates and feedback vectors use *current* `text_content`/`top_comments`/`article_body`. A story voted in June is embedded with comments fetched later, and its similarity/TF-IDF features see discussion that did not exist at vote time. TF-IDF vocab itself is clean (fit on train only, `_tfidf_features`, `min_df=2`) and LOOCV/scaler are clean (`_loocv_knn_features`, `_fit_scale` train-fit only) — the leak is the *input text*, not the fitter.
Effect: inflates all content models' AUC/NDCG; favors word-heavy challengers; says nothing about save-time live, discovery, or sameness. Effort: M (title-only / time-capped-text ablation first, then frozen-text snapshot). Validation: rerun `eval_ranker_variants.py --candidate-pool heldout-feedback` with `--variants` title-only TF-IDF (`tfidf_text=title`) vs full; expect word-model gain to shrink. Confidence: high.

**2. Judged-only replay mis-measures prevalence and discovery.**
Category: evaluation gap. Evidence: `eval_ranker_variants.py:_metrics` (~1883+; `judged` dict, `auc_up_vs_rest/down/all`, `ndcg_at_k`); `candidate-pool heldout-feedback` ranks only voted stories. Unjudged = unknown, positives are dense, so AUC/NDCG levels are not live up-rates. The `impressions` pool fixes this only partially: shown-but-unvoted is treated as not-upvote, but the pool is what *production* put on screen (`_load_skipped_pool`, `_impression_pools` ~1383, diversity `PLAN.md` "Limits"). A challenger reaching deeper is judged on production's exposure.
Effect: save-time proxy only; discovery/novelty underestimated; sameness unmeasured in main eval. Effort: S–M. Validation: always report `heldout-feedback` *and* `impressions` pools side by side; add exposure-aware check (score only within shown set; rank-weighted / IPW sensitivity). Confidence: high.

**3. Live trial statistics ignore within-deck clustering; power is low.**
Category: defect/gap. Evidence: `scripts/interleave_report.py:summarize/compare`, `greater_rate_p` (pooled z), `rate_difference_ci` (Newcombe/Wilson, explicitly "descriptive and unadjusted"); `scripts/simulate_interleaving.py:compare` same tests; `INTERLEAVING.md` reporting addendum ("Clustering and adaptation remain limitations"), power replay optimistic note; STATUS 2026-10-12: ~61 credits/4.5d (~13.5/d) vs replay's 50/d — ~125/arm at 28d ≈ replay's 7-day row.
Effect: per-vote tests overstate precision; sign test per deck is directionally right but underpowered; no equivalence claim licensed. Sameness/diversity are guardrails only. Effort: S (analysis-only). Validation: pre-freeze endpoint DB snapshot (already prescribed: `.backup` + `--exclude-updated-between 1791763112.57 1791763180.12`), report deck-clustered SEs (cluster bootstrap by version, sign test primary), Newcombe as descriptive. Confidence: high.

**4. Attribution has position/exposure bias the report does not model.**
Category: defect. Evidence: `pipeline/interleave.py:team_draft` (fresh random round order, good), `pipeline/__init__.py:draft_recommended`, far-card truncation to 11+far (`FAR_ARM`, `place_far_card`); `interleave_report.py:credit_votes` (last impression +1s slack, `view=="recommended"`, `arm_of` lookup, `far_card` skip). Random order balances arms in expectation, but served prefix cuts rounds (now ~3/3/2), head positions get more votes, and 1s slack + mutable `feedback` (undo/edit overwrites) make credits fragile without the frozen snapshot.
Effect: small bias toward arms winning early picks; misattributes re-votes. Effort: S. Validation: balance check (picks per arm per position, `deck_scores` vs impression positions per far-card caveat in `diversity-20261012/PLAN.md`); always run report on frozen endpoint copy. Confidence: medium.

**5. Main metric suite does not cover the four aims; diversity lives elsewhere.**
Category: evaluation gap. Evidence: `_metrics` has `auc_up_vs_*`, `ndcg_at_k`, `up_recall_at_k`, plus `discovery_upvotes_at_12` (median-split "novel", not the 0.60 Discover rule), `non_hn_upvotes_at_12`, `known_downvote_fraction_at_12`; pairwise diversity / topic entropy / novel-up counts exist only in `diversity-20261012/PLAN.md` Phase 0 gate (+0.05 diversity, up-rate ≤−2pts, novel-ups ≥ B) and `scripts/narrowing_report.py` (cluster coverage). No freshness/stay-current split, no stopping/calmer-habit cost, no calibration requirement (`brier_up` only when `calibration_available`).
Effect: AUC-chasing risks same-y feeds. Effort: S (wire existing metrics into every eval). Validation: every `eval_ranker_variants.py` run must also report: top-12 down-rate, novel-up rate at sim<0.60, non-HN rate, mean pairwise 1−cosine + topic entropy on top-k, Brier when probs exist. Confidence: high.

**6. Proposed minimal offline protocol (adopt as gate).**
Temporal expanding folds + URL-group isolation (keep `_temporal_splits`, `_make_fold` group-key logic; `stratified` stays retired) → reserved latest-20% `--confirmation` as the decision fold → `--candidate-pool heldout-feedback` + `impressions` dual report → text-leak sensitivity (title-only) → calibration rows excluded by default → `compare_eval_scores.py` triple uncertainty (stratified-story bootstrap + whole-block bootstrap + sign-flip/DeLong with Holm; predeclared equivalence margin; "nonsignificant ≠ equivalent" per `model-ablation-20261007/README.md`) → `--train-recent` short-history curve as first-feed proxy → `--leak-check` shuffle baseline. Rank by *up-rate − down-cost per diversity point*, never AUC alone. Effort: M (mostly wiring). Validation: re-score #2 vs #4 12-block study; require conclusions stable across both pools and both bootstraps. Confidence: medium.

## 2. Checked — fine or already tried

- Temporal splits train strictly before test blocks (`_temporal_splits:vote_times < block[0]`) — fine.
- URL-group isolation for test votes and candidate dedup (`_make_fold:group_key=_url_group`) — fine.
- TF-IDF `min_df=2`/train-only vocab; joined `keep>=2` training stories; scaler fit-train-only — fine.
- Raw-embedding scaling ban (metadata-only `StandardScaler`, clip ±2.5) — fine, do not relax.
- Stratified shuffle retired — correct, do not resurrect.
- Triple uncertainty + Holm in `compare_eval_scores.py`; "no equivalence without margin" in ablation README — already tried, keep enforcing.
- Optimistic power replay caveats, frozen-endpoint + calibration-exclusion addendum — already documented in `INTERLEAVING.md`; just follow it.
- Diversity Phase 0 gate, Discover/far-card logging in `deck_scores` — already built; extend, don't reinvent.
- LLM-feature reassessment (`RANKING-REASSESSMENT-20261009.md`: no reusable missing signal) — do not re-propose Muse-word features without new evidence.

## 3. Open questions

- Can a time-travel text snapshot (title + pre-vote comments only) be reconstructed, or is title-only the durable leak bound?
- Should `impressions`-pool unvoted be down-weighted by position/exposure (IPW) rather than treated as full negatives?
- What is the predeclared equivalence margin for AUC/P@12 and the per-deck clustered primary test for any post-trial continuation?

