## Training data and labels review (read-only)

### 1. Ranked findings

**1. Calibration re-votes train production — exclude them. defect.**
Evidence: `STATUS.md` "Next step" (exclude 29 rows `updated_at` 1791763112.57–1791763180.12 from fresh-vote evals); `FINDINGS.md` "Blind re-vote calibration 2026-10-11" (17 changed + 12 June adds applied to live profile 151); `pipeline/ranking.py:_score_and_rank` trains on all `get_feedback_for_training` rows with no exclusion. Those 29 labels were made blind against a stand-in server, a different context from feed votes, yet they now shape every production fit. Effect on aims: removes off-distribution taste distortion (save-time/discovery neutral-to-positive; no sameness cost — 29/4071 rows). Effort: S (one `updated_at`-window filter + test). Validate: `scripts/eval_ranker_variants.py --candidate-pool heldout-feedback --split temporal` with/without the window; metrics AUC, P@12 up-rate, novel-up rate. Confidence: high (hygiene), low on metric gain.

**2. Same-action re-votes never refresh `updated_at` — timestamps go stale. defect.**
Evidence: `database.py:record_feedback` (`ON CONFLICT … WHERE feedback.action != excluded.action`) — re-confirming a vote is a no-op, so `vote_times` from `get_feedback_for_training` (`database.py:1462`) and `_feedback_signature` (`pipeline/ranking.py`) never see it. Any future time-decay, temporal split ("`updated_at` approximates chronology", `eval_ranker_variants.py:3037`), or cache invalidation built on `updated_at` silently misorders re-confirmed taste. Effect: none on today's ranking (production ignores `vote_times`), but it biases every recency experiment toward "no drift" and delays cache refresh on re-confirmation. Sameness neutral. Effort: S (update timestamp always, action conditionally + regression test). Validate: unit test + re-run half-life variants after fix. Confidence: high it is a defect; medium it changes the recency null.

**3. Cross-post votes double-train; the dedup fix exists but is off. untried improvement.**
Evidence: `pipeline/config.py:30` (`deduplicate_training_feedback=False`), `pipeline/feedback.py:deduplicate_feedback` (latest vote per normalized URL), `pipeline/ranking.py:1274` (gated call), `scripts/eval_one_classifier.py:248` (eval support). An HN + RSS/Reddit copy of one article currently trains as two independent rows, overweighting cross-posted topics. Effect: enabling should slightly *reduce* sameness (good for discovery) at neutral save-time cost. Effort: S. Validate: temporal heldout-feedback eval dedup on/off; AUC + P@12 + diversity-20261012 metrics (pairwise diversity, novel ups). Confidence: medium.

**4. Implicit signals still unused in production — endorse the owner's planned dwell check, don't preempt it. untried improvement / evaluation gap.**
Evidence: `database.py:24,513` (`interaction_events`: impression/article_open/comments_open/dwell), `database.py:1531` (`get_capped_dwell_by_story`, labelled "sample_weight experiments"), `scripts/eval_ranker_variants.py:_scores_margin3_dwell`; `STATUS.md` "Next step" already schedules the dwell-vs-votes rerun ~2026-10-27 and a 2-week opens/dwell-vs-votes check; `FINDINGS.md` "First reading-signal review": dwell separates reads from skips/downs (AUC 0.82) but *not* up from neutral. Proposal *after* that read-out only: long-dwell neutrals as soft ups or `1 + gain·capped_dwell/cap` sample weights — never raw dwell as a positive label. Effect: save-time positive if scoped to neutral disambiguation; sameness risk is real (dwell favors longform) — cap per story, and monitor topic spread/novel ups. Effort: M. Validate: temporal + impressions-pool evals (`--candidate-pool impressions`), metrics P@12, AUC, down-rate, novel-up rate; ship only with flat-or-better diversity. Confidence: low-medium.

**5. Exposure bias is measured nowhere in the live loop. evaluation gap.**
Evidence: training sees only voted stories while candidates exclude voted ones (`pipeline/__init__.py:load_production_candidate_stories`, `fast_rerank_for_user`); all trial arms share one `training_feedback` snapshot (`pipeline/ranking.py:rerank_candidates`), so credited votes co-train every arm; serve-time `deck_scores` + `interaction_events` logging exists since 2026-10-12 but no propensity/exposure-normalized training or metric. Do not change training before 2026-11-05. Post-trial proposal: exposure-normalized offline metric (credited ups per exposed slot, per stopping-rule review) and, only then, a small IPS/positivity-weighted pilot. Effect: discovery-positive (values novel exposure), no feed change now. Effort: M/L. Validate: `scripts/simulate_interleaving.py` + impressions-pool replay; report position-by-arm. Confidence: medium the gap is real; low on estimator choice.

**6. `neutral_weight` is a dead knob — remove it. defect (hygiene).**
Evidence: `pipeline/config.py:36` (`neutral_weight=0.0`), cited in `ARCHITECTURE.md` 3.3, but no use in `ranking.py`/`linear_blend.py`/`joined_classifier.py` (balanced `n/(classes·count)` weights instead, `ranking.py:~1595`). Given the neutral-as-down null (next section), the right fix is deletion, not wiring. Effort: S. Confidence: high.

### 2. Checked — fine or already tried, do not re-propose

- Plain exponential vote half-life (30/60d): null, +0.004–0.006 with CI from/through 0 (`FINDINGS.md` 2026-10-11; earlier nulls 2026-09-29, 2026-10-02) — but note finding 2 stales its timestamps.
- Neutral-as-down: null on held-out (+0.001) and harmful on re-votes (−0.040); keep 3-class (`FINDINGS.md` 2026-10-11).
- Shown-unvoted as neutral/down (30%): ±1–3 cards, ≈+0.001 AUC — noise (`FINDINGS.md` "Fresh-vote, impression-pool" 2026-10-02).
- Ordinal/pairwise stack: rejected; pair raised top-12 down share 0.010→0.062 (`FINDINGS.md` ~3100).
- Label noise itself already quantified: blind re-votes AUC 0.785, 59% agreement, old votes flip 0.33 vs 0.14 (`FINDINGS.md` 2026-10-11).
- Correct as-is: balanced class weights; LOOCV self-exclusion for kNN training features; 1e-6 dummy rows for absent classes; meta-only StandardScaler + ±2.5 clip (raw L2 embeddings untouched); one-row-per-(user,story) PK; eval URL-group isolation + temporal splits.

### 3. Open questions

- Should `updated_at` mean "last seen/confirmed" or "last changed"? Finding 2's fix depends on that semantic choice (also affects `_MODEL_CACHE` invalidation rate).
- After finding 1, should the 29 calibration rows be reverted, excluded from training only, or kept as gold validation data (46 blind votes could anchor a flip-aware eval with ~77 more votes per `STATUS.md`)?
- For finding 4, which neutral subset gets the dwell treatment — all long-dwell neutrals, or only non-Popular ones (Popular dwell may mean hate-reading)?
- Post-trial, is the exposure unit the deck, the slot, or the (deck, window) — needed before any IPS pilot?

