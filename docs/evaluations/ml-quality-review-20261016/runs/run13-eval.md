# Evaluation-methodology review: hn-rerank ML ranker

## 1. Top findings / proposals, ranked

### P1. Make the exposure-aware pool the gate, not judged-only NDCG — evaluation gap
**Evidence:** `docs/RANKER_EVALUATION.md:34-48`; `scripts/eval_ranker_variants.py:1876` (`_metrics` docstring: "unjudged cards have unknown relevance"); report `interpretation` (~line 3030: "not historical exposure"); `INTERLEAVING.md` ("optimistic… every story judged… selected on these labels").
Heldout-feedback ranks only already-voted stories (high prevalence, NDCG>0.8 without live quality — docs admit this), while `current`-pool has sparse judged overlap. Neither measures what Recommended serves: top-8/12 from thousands of mostly-unjudged candidates under exposure bias. #2/#4 were selected largely on the optimistic pool.
**Aims effect:** current practice over-rewards familiarity (hurts discovery, sameness); exposure-aware top-k up/down-rate protects save-time + calmer habit without hiding novelty loss.
**Effort:** M. **Validation:** `eval_ranker_variants.py --candidate-pool current` + `--holdout-after` impression pools (`_impression_pools`, `_metrics` `auc_up_vs_all` where shown-unvoted = not-up) as primary; report `recommended_1w` deck metrics + `discovery_upvotes_at_12`/`non_hn_upvotes_at_12`; block-bootstrap CIs (as in `diversity-20261012/PLAN.md` Phase 0); require `--leak-check`. **Confidence:** high.

### P2. Close the future-content leak: freeze text/scores at vote time — defect
**Evidence:** `RANKER_EVALUATION.md` ("current stored text… can postdate votes"); `eval_ranker_variants.py:_stack_meta_features` ("Points are the stored (latest) score, not the score at vote time"); `_metrics` deck path uses snapshot embeddings incl. later `top_comments`; report admits "`updated_at` approximates chronology."
Training on earlier votes but scoring candidates with text/comments/scores fetched *after* the vote inflates metadata and comment-dependent models — exactly the high-metric skepticism in AGENTS.md. This likely flatters #2 (all-features) over #4.
**Aims:** fixing it mainly de-flatters stay-current/popularity signals; prevents shipping a model that wins offline and loses live. Neutral on sameness.
**Effort:** M (snapshot `text_content`/`top_comments`/`score` as-of vote time, or sensitivity run excluding comment-heavy rows). **Validation:** rerun #2 vs #4 vs production on time-frozen text; expect metadata lift to shrink; confirm with `--confirmation` split. **Confidence:** medium-high (leak path confirmed in code comments; magnitude unmeasured).

### P3. Treat the interleaving trial as underpowered directional, not decisive — evaluation gap
**Evidence:** `INTERLEAVING.md` power table (28d: 70%/54% under optimistic 486/arm) vs STATUS 2026-10-12 check (61 credits/4.5d → ~125/arm at 28d ≈ replay's 7-day row); `simulate_interleaving.py` docstring ("optimistic… every candidate is judged"); `interleave_report.py:credit_votes` (last-impression +1s slack attribution); stopping-rule review (per-vote tests ignore clustering/skipped exposure); mutable `feedback` needs endpoint snapshot (addendum now requires it).
Per-vote z/sign tests assume independent votes; decks cluster votes, skipped cards get no exposure credit, position bias is unmodeled. Far-card (`far` arm, live since 2026-10-12) contaminates Discover measurement.
**Aims:** over-reading a noisy up-rate risks shipping a same-y winner or discarding a discovering challenger. Report rate-diff with Newcombe CI + deck-clustered SE, down-safety, and novel-up-votes/week guardrails; pre-commit "no detectable difference" as an allowed outcome (already the decision rule — hold to it).
**Effort:** S (analysis-only). **Validation:** re-run `interleave_report.py` on frozen endpoint snapshot; sensitivity with cluster-bootstrap by deck; disclose capture time. **Confidence:** high.

### P4. Promote diversity/novelty/coverage to first-class offline metrics — evaluation gap
**Evidence:** `_metrics` already computes `discovery_upvotes_at_12`, `non_hn_upvotes_at_12`, `known_downvote_fraction_at_12` but model selection keys on AUC/NDCG; diversity gate lives only in `diversity-20261012/PLAN.md` (≥+0.05 pairwise diversity, up-rate within −2pts, novel-ups ≥ baseline) — never applied to ranker variants.
AUC is at ceiling (prod ~0.82; blind re-vote ceiling 0.785, FINDINGS 2026-10-11). AUC/NDCG cannot see filter bubbles.
**Suite:** per proposal: top-12/40 credited up-rate + down-rate (safety), `discovery_upvotes_at_12` (sim<0.60 to prior ups), non-HN up-votes, mean pairwise (1−cosine) + topic entropy on served 8, judged coverage/eligible positives. Gate model changes like Phase 0: no diversity/novelty regression for any dud-reduction.
**Aims:** directly measures discovery + sameness alongside save-time; stay-current via short-window split (§P5). **Effort:** S (metrics exist; wire into `summarize_eval_report.py` composite). **Validation:** any post-trial variant must pass diversity gate on `current`-pool + Phase-0b day-pools before interleaving. **Confidence:** medium.

### P5. Standardize the split protocol and uncertainty reporting — untried improvement
**Evidence:** `_temporal_splits`/`_holdout_splits`/`_recent_training` + URL-group isolation with `fold_audit` before/after counts (good); but `std` is "fold variation, not uncertainty" (report line ~3032), folds are ~350–700 cards so NDCG@k is noisy (docs recommend `auc_up_vs_rest`), and semantic near-dupes with different URLs remain (docs admit).
**Protocol:** temporal expanding-window (default, `initial_train_frac=0.5`) + frozen `--confirmation` tail as required confirmation; `--train-recent 200/400/800` cold-start curve for new-profile claims; mandatory `--leak-check --leak-seeds 5` with `random_expected_ndcg` comparison; paired story-level tests via `--dump-scores` + `compare_eval_scores.py`; block-bootstrap CIs, never fold-std as significance; always report coverage warnings (<10 positives/<20 judged → null, not zero).
**Aims:** protects all four aims by stopping noisy or leaked "wins." **Effort:** S (mostly discipline + a wrapper script). **Validation:** re-emit #2/#4 tables with paired CIs; #4's short-history intervals already include zero (INTERLEAVING.md) — keep that framing. **Confidence:** medium.

## 2. Checked — fine or already tried
- TF-IDF/IDF fitted on train only: `_tfidf_features` (train-vocab) and `joined_classifier.py:83-105` (fit IDF on training stories) — **fine, no test-vocab leak**.
- KNN self-exclusion mask + clip-to-[−2.5,2.5] per ARCHITECTURE §3.2 — **fine**.
- Raw L2 embeddings never standard-scaled (`_fit_scale` touches `emb_dim:` onward) — **fine**.
- URL-group isolation train/test/candidate + `_with_skipped` URL-group guard — **fine, residual semantic-dup risk only**.
- `auc_up_vs_rest` as steady metric; Brier null for uncalibrated softmax — **fine**.
- Re-vote calibration ceiling (AUC 0.785, ±0.12) and recency-weighting no-gain — **already tried, don't re-propose**.
- Stacked/GBM second-stage, TF-IDF sweeps, short-history curve, metadata ×0.5/×0.25 — **already tried** (FINDINGS/INTERLEAVING follow-ups); middle variants landed between endpoints.
- Reassessment verdict "build nothing from Muse rationales" — **agree; not re-proposed**.

## 3. Open questions
- Can time-frozen story text/scores be reconstructed for votes, or only bounded by sensitivity analyses?
- What is the deck-clustered effective sample size of the trial (votes per deck, intra-deck correlation)?
- Should shown-but-unvoted count as negative (`auc_up_vs_all`) be the primary offline objective, given dwell/open signals are still maturing?

