# Short history, new profiles, freshness — review

## 1. Ranked findings / proposals

**1. Cold first feed is HN-gravity only; RSS/Reddit score ~0 and sink.**
Category: defect. Evidence: `pipeline/__init__.py:158-166` (`cold_ranked_candidates` scores every candidate by `hn_gravity(score,…)`); `pipeline/__init__.py:1018-1050` (0-vote path returns the cold deck, no Explore); RSS rows carry score 0 (ARCH 3.4.4, RSS leg ordered by time only). A brand-new profile's Recommended is therefore effectively HN-only, and non-HN discovery is deferred until votes arrive. Aims: hurts discovery and first-feed quality; sameness up (HN monoculture). Effort S: add a recency-ordered non-HN fill (e.g. top-N recent RSS by time, capped) to cold Recommended, or a small static non-HN prior. Validation: `scripts/eval_ranker_variants.py` cold-start replay + Import-AI gate (`scripts/check_import_ai_recommended.py`) for RSS presence; metric: novel-up rate, not AUC. Confidence: high that the defect exists; medium on the fix's size.

**2. Tier-1 gravity also buries non-HN for 1–49-vote users; source prior arrives only with the SVM.**
Category: defect. Evidence: `pipeline/ranking.py:1765-1780` (tier-1 is pure HN gravity, normalized by the pool max so score-0 RSS ≈ 0); source one-hot exists only in SVM features (`ranking.py:1490s`, `source_category_stack`). Below the dual gate (`config.py`: `min_up/min_down=20`) there is no per-source prior at all. FINDINGS 2026-10-02 found other-RSS yielded 23% up in Recommended vs 6% HN — the users who most need RSS see the least of it. Sameness: narrows topics early. Effort S: give tier-1/2 a tiny static source prior (or carry the one-hot into tier-2). Validation: `--train-recent 100/200` curve (eval script `--train-recent`), slice up-rate by source. Confidence: medium-high.

**3. Neutral-heavy newcomers lose gravity before gaining the SVM.**
Category: defect. Evidence: `pipeline/ranking.py:1783` (`alpha_2 = n_feedback/50` counts neutrals) vs `ranking.py:1787-1800` (SVM needs `min(n_up,n_down)`). A user with 40 ups + 40 neutrals + 5 downs has `alpha_2≈1` (tier-1 gone) but no tier-3 — pure centroid-diff. Effort S: base `alpha_2` on `n_up+n_down` (exclude neutrals) or on `min()` like tier-3. Validation: short-history replay stratified by neutral share; check top-12 down-rate doesn't rise. Sameness: neutral (keep discovery, fewer duds either way). Confidence: medium.

**4. Tier-2 centroid-diff is the weakest possible personalizer and ignores existing structure.**
Category: untried improvement. Evidence: `pipeline/ranking.py:~1690-1720` (unnormalized up/down means, raw dot, neutrals ignored, single vector per class) while the file already computes fused top-k means (`_knn_mean_and_max`) and `interest_centers` (k=10, `ranking.py`) for Explore. A 5-vote user with two interests gets their average. Effort S: score tier-2 by kNN-mean-up minus kNN-mean-down (already computed in the SVM branch) or nearest-interest max. Sameness: strictly widens vs one centroid. Validation: `--train-recent 50/100/200` on profile 151 + small profiles; metric top-12 up-rate. Confidence: medium (cheap, well-precedented).

**5. Act on the short-history curve: faster ramp / lower gate for the joined challenger.**
Category: untried improvement (curve already ran). Evidence: INTERLEAVING.md "Short-history curve" — from N=200 both challengers match/beat production; #2 AUC +0.011–0.013 with block intervals excluding zero at N≥400. Current ramp (`tier3_blend_window=60`, full weight at min=80) leaves this gain on the table for months of real voting. Effort S–M: per-classifier ramp (faster `alpha_3` for `joined_logistic`), or gate 20→10 for the challenger only. Validation: same `--train-recent` harness, plus down-rate guardrail (joined #2 had 6 downs/144 vs 2 production in the metadata table). Sameness: joined models carry word features — watch topic narrowing via novel-up counts. Confidence: medium.

**6. Population prior for new users (151's centroid as tier-2 seed, fast decay).**
Category: untried improvement. Evidence: no cross-user prior exists; `warm_store.py` only warm-starts solvers. A new profile could start tier-2 from the main profile's up-centroid (vectors only, no titles exposed) decaying to zero over ~50 own votes. Small-data-safe, no high-capacity fit. Sameness risk: inherits 151's taste — mitigate with the decay and the Discover/far-card path (live since 2026-10-12). Effort M. Validation: simulate new users via `--train-recent` with seeded centroid vs gravity; metric: up-rate at N≤100. Confidence: low-medium (untried; showcase-privacy note: centroid leaks coarse taste, not votes).

**7. No age/velocity/importance feature; "stay current" rests on windows, not the ranker.**
Category: evaluation + feature gap. Evidence: meta cols are length/similarity/source only (`ranking.py:1490s`); `engagement_features_enabled=false`, `publication_affinity_enabled=false` (`config.py`, `config.toml` enables neither); Popular velocity/badges (`ranking.py:2184-2199`, `GRAVITY_TIME_SCALE:354`) never enter Recommended scores. A stale high-sim story can outrank a major today-story in 1w/1m. Already-tried nearby: raw points/comments rejected as archive-inflating (ARCH 3.2); recency half-life 30/60d gave +0.004–0.006, CI from 0 (FINDINGS 2026-10-11). Differing version: use *velocity* (points/hour) and *log-age*, not raw points — structurally near-zero for archive, so the original contamination objection doesn't apply. Effort M. Validation: temporal `eval_ranker_variants` with archive leg present + fresh-upvote capture metric (time-to-surface), not AUC. Sameness: velocity favors big stories — cap its weight, keep Discover/far-card untouched. Confidence: low-medium.

## 2. Checked / already tried (one line each)

- Dual gate 20/20 + 20/80 ramp verified (`ranking.py:1783-1800`, `config.py:73-77`); short-history curve already ran (INTERLEAVING.md) — proposal 5 acts on it, doesn't repeat it.
- Raw engagement (points/comments) correctly rejected/off; velocity variant above differs.
- Recency-decay and neutral-as-down both null (FINDINGS 2026-10-11) — not re-proposed.
- Blind re-vote headroom +0.007 unsettled, ±0.07 needs ~77 more votes — no metric claim made here.
- Discover (0.60 sim floor) + far card live; diversity Phase 0 gate unpassed, Far 1 near-miss — untouched by these proposals except as guardrails.
- Missed-upvote HN-tail pattern is descriptive, not causal (docs `missed-upvotes-20261009`) — no blanket HN boost proposed.
- Import-AI freshness gate exists (`check_import_ai_recommended.py`) — proposed as validator, not new work.
- L2 embeddings never scaled; meta clip [-2.5,2.5] intact — proposals add meta columns only under the same scaler.
- No DB deletion/migration proposed; all validation is read-only replay plus existing scripts.

## 3. Open questions

- What is a new profile's actual first-feed source mix today (HN vs RSS share in cold Recommended)? A one-query check would size finding 1.
- Should `alpha_2` exclude neutrals, or do neutrals carry cold-start signal worth keeping?
- Is a 151-seeded prior acceptable for the public showcase, or should the prior be global-popularity only?
- Which stay-current metric (fresh-up capture? Import-AI pass rate?) should gate a velocity feature, since AUC won't?

