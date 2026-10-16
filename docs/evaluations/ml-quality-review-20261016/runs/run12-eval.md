# Evaluation-methodology review: hn-rerank

## 1. Ranked findings

**1. Offline content/engagement are read at snapshot time — metrics are upper bounds. Category: defect.**
Evidence: `scripts/eval_ranker_variants.py:_stack_meta_features` ("Points are the stored (latest) score, not the score at vote time"); `_load_production_candidates`/`_validated_embeddings` use current `story_embedding_text()` (title+`text_content`+comments); short-window recheck explicitly warns rescoring reads final points/comments and current `top_comments` (`FINDINGS.md` "Short-window Recommended"); Discover/Diversity Phase 0 plans admit "HN points/comments read at snapshot time (leak)".
Effect: inflates AUC/NDCG for engagement- and comment-heavy variants; risks shipping a "win" that is hindsight. No sameness effect directly, but leaky quality priors can narrow to rich threads.
Effort: M. Validation: rerun canonical `eval_ranker_variants.py --candidate-pool heldout-feedback --holdout-after <post-tuning-date> --holdout-blocks 4` with (a) engagement columns zeroed, (b) `top_comments` stripped from eval text, and (c) `--leak-check`; require the variant to beat production on NDCG@12/upvotes-per-48 with paired block bootstrap (`scripts/compare_eval_scores.py`). Confidence: high that leak exists; medium on size.

**2. Judged-only replay is optimistic; impressions pool confounds exposure with preference. Category: evaluation gap.**
Evidence: `eval_ranker_variants.py --candidate-pool` help ("not comparable metrics"); `heldout-feedback` ranks only voted stories (missed-upvotes oracle: 48 slots for 683 UPs, 95.8% miss structural — `docs/evaluations/missed-upvotes-20261009/REPORT.md` §1); `impressions`/`diversity-20261012/PLAN.md`/`discovery-20261012/PLAN.md` treat shown-but-unvoted as not-liked on an already production-filtered pool ("reaches deeper" untested).
Effect on aims: overstates save-time hits; understates discovery risk (a rule reaching deeper is judged on stories production surfaced). Sameness: cannot detect narrowing beyond shown pool.
Effort: S/M. Validation: keep both pools but separate claims: `heldout-feedback` for rank order (AUC up-vs-down, NDCG@12, upvotes/48); `impressions` for served-view outcomes (up/down-rate per pick, novel-ups caught) with sensitivity treating unvoted as (i) negative, (ii) ignored. Use temporal expanding splits (`_temporal_splits`) + URL-group isolation (`_make_fold: _url_group`) in both. Confidence: high.

**3. Canonical eval lacks enforced diversity/novelty/coverage gates. Category: evaluation gap.**
Evidence: `eval_ranker_variants.py:compute` already emits `discovery_upvotes_at_12`, `non_hn_upvotes_at_12`, `known_downvote_fraction_at_12`, `auc_up_vs_rest/down/all`, Brier — but ablation decisions cite AUC/NDCG only (`model-ablation-20261007/README.md`); diversity/novelty live in one-off `~/revote/` scripts (`FINDINGS.md` "Diversity/Discover Phase 0", "Sameness and novel up-votes": Recommended diversity −0.15–0.21 vs random, novel ups 6%, far cards 11% vs 22%).
Effect: AUC-near-ceiling wins (blind re-vote: re-votes AUC 0.785 vs production 0.778, CI −0.12–+0.12 — `FINDINGS.md` 2026-10-11) can pass while same-y-ness worsens. Fix directly serves discovery/calmer-habit guardrails.
Effort: S. Validation: promote existing metrics into every `eval_ranker_variants.py` report with Diversity Phase 0 gate (diversity gain ≥0.05, up-diff ≥−2pts/lower-bound ≥−5, novel-ups ≥B) plus down-rate and total-ups guardrails; split short (12h/1d) vs long (1w/1m). Confidence: medium-high.

**4. Live trial estimand is credited-votes, not exposure; power was overstated. Category: defect (known, mitigated by fixed endpoint).**
Evidence: `scripts/interleave_report.py:credit_votes/summarize/greater_rate_p/rate_difference_ci` — last-impression attribution (1s slack), pooled two-proportion + sign test, Bonferroni-2, Newcombe/Wilson descriptive; skips Popular/Explore/far (`FAR_ARM`); `docs/evaluations/stopping-rule-review-20261013/ASSESSMENT.md` verifies: pooled test ignores deck/session clustering and continuous monitoring, mutable `feedback` needs frozen snapshot, denominator ignores skipped-card exposure. `INTERLEAVING.md` power replay is labeled optimistic (frozen models, every candidate judged, arms selected on labels, 50 votes/day); actual ~13.5 credited/day at 2026-10-12 implies ~125/arm at 28d (≈7-day replay row, low power).
Effect: inconclusive endpoint likely; chasing significance by extending pooling repeats the error. No sameness measurement in trial (novel-ups/week is separate Phase 2).
Effort: S (report-only). Validation: freeze endpoint snapshot before further voting (`INTERLEAVING.md` addendum commands), run `interleave_report.py --since T0 --until T0+28d` with calibration exclusion; report deck-sign + rate + Newcombe CI + down-safety, plus descriptive exposure rates/position summaries; accept "no detectable large difference", no pooling of repeated peeks. Confidence: high.

**5. Small-n claims need paired uncertainty + leak-check + post-tuning confirmation. Category: evaluation gap.**
Evidence: 4–12 blocks, ~100 ups/rule (ablation README: "Four blocks insufficient"; Diversity/Discover "directional read only"); `compare_eval_scores.py` paired story/block bootstraps exist but decisions sometimes quote point estimates; `--leak-check` (shuffled labels), `--confirmation`/`--holdout-after`, `--train-recent` cold-start curve exist and are underused.
Effect: prevents false save-time wins; cold-start curve (`INTERLEAVING.md`: N=100–1600) is the only new-profile proxy and reuses heavy-user recent votes.
Effort: S. Validation: require for any post-trial change: (i) paired story+block bootstrap clear of zero, (ii) shuffled leak-check at prevalence, (iii) confirmation on votes newer than all tuning (`--holdout-after`), (iv) `--train-recent 100/200/400` curve for first-feed impact. Confidence: high.

## 2. Checked: fine or already tried

- Temporal expanding-window splits; `stratified` retired — correct, keep.
- URL-group train/test/candidate isolation in `_make_fold` — good; title-fuzzy cross-posts remain a minor gap.
- LOOCV kNN self-exclusion, `_fit_scale` on meta columns only (no raw-embedding scaling) — correct.
- Stacking via out-of-fold content scores; `_smoothed_label_rates` train-only — sound.
- Team-draft after dedup/HN-dupe canonicalization, fresh per-round permutation, `SharedFeatures` one-vote snapshot — sound; far-card exclusion tested.
- Bonferroni-2 over preregistered challengers, one-sided down-harm test + safety stop, Wilson/Newcombe CI as descriptive — appropriate.
- Stacked ranker, incremental hill-climb, untuned-embedding probe, TF-IDF sweep, one-classifier/SINGLE-MODEL/BROAD-ONE-CLASSIFIER, MMR/caps/Far-1, Discover R1–R4, re-vote calibration — already tried; do not re-propose without new variation.
- `deck_scores` serve-time logging for future rank-cap/`prob_down` replays — right next step; use it post-2026-11-05.

## 3. Open questions

- Is vote-time text/points recoverable (snapshots, `comment_count_at_fetch`, CH history) for a truly historical replay, or is snapshot-time bounding the best achievable?
- What is the exposure unit for guardrails (served card vs impressed card vs deck), given partial impressions per version (median 2–3)?
- What equivalence margin defines "no worse" for up-rate and novel-ups/week, so a simpler model can be adopted without claiming superiority?
- How to evaluate a genuinely new profile (cold deck, no votes) rather than truncating profile 151's history?

