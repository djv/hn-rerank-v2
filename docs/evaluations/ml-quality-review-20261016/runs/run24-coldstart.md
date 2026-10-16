## 1. Top findings, ranked

**P1 — Cold first feed buries all score-0 sources. Category: defect.**
Evidence: `pipeline/__init__.py:cold_ranked_candidates` scores `hn_gravity(points,…)`; `pipeline/ranking.py:hn_gravity` → 0 when `points=0`. RSS/Reddit rows have ~0 score, so a 0-vote user gets an HN-gravity ordering over `load_production_candidate_stories` legs (`pipeline/__init__.py:load_production_candidate_stories`). First feed ≈ Popular, no discovery. Affects: saves time OK, discovery/current fail for new profiles, sameness high (HN-only).
Fix (after 2026-11-05): cold score = gravity for HN + recency/source prior for non-HN (e.g. time-decayed per-feed prior, or tier-1 blend with small constant). Effort S. Validate offline: `scripts/eval_ranker_variants.py` cold/skip-pool folds + `feed-metrics.py` by-source up-rate and `sameness-metrics.py` novel up-votes/week; require no topic-entropy loss. Confidence high that defect is real, medium on best prior.

**P2 — Gravity vanishes exactly when the SVM is still off. Category: defect.**
Evidence: `pipeline/ranking.py:_score_and_rank` — `alpha_2=clip(n/50)`, `t1=1-alpha_2`; SVM needs `n_up,n_down≥20` (`pipeline/config.py:min_up/down_for_svm`), full weight only at `min≥80` (`tier3_blend_window=60`). A 50-up/5-down user has `alpha_2=1`, no SVM → pure centroid-diff, zero popularity/recency signal. Hurts stay-current most in thin 12h/1d windows (FINDINGS 2026-10-12: 12h median score 0.31, Popular 62% down).
Fix: keep a small gravity floor (e.g. 0.1–0.2) until `alpha_3>0`, or make `alpha_2` use `min` not total. Effort S. Validate: short-history curve replay (INTERLEAVING.md followups) stratified by window + 12h/1d up-vs-down AUC and downs-per-card, with sameness guard. Confidence medium-high.

**P3 — Tier-2 ignores neutrals and uses raw means. Category: untried improvement.**
Evidence: tier-2 block in `_score_and_rank` means up vs down only; neutral-only mass → flat 0.5; no shrinkage, no TF-IDF, no source prior. Below the 20/20 gate this *is* the ranker (tiers 1–2).
Fix: shrinkage centroid (add global-mean shrinkage ∝1/n), fold neutrals as 0.5-weight or weak-down, add hashed TF-IDF cosine fallback. Cheap, low-capacity, appropriate for <100 votes. Effect: more hits without narrowing topics (centroid stays broad). Effort S–M. Validate: short-history replays at N=10/25/50/100/200, AUC + top-8 up-rate + diversity delta. Confidence medium.

**P4 — No freshness/importance feature in the model; badges don't rank. Category: evaluation gap + untried improvement.**
Evidence: `_svm_personalization_features` meta cols 0–9 are length/similarity/source-onehot (`pipeline/ranking.py`); age/points/velocity absent. `engagement_features_enabled=false` (`pipeline/config.py`) after archive-inflated offline gains (ARCHITECTURE 3.2). Freshness lives only in SQL order (`recent_candidate_*_limit`), window clocks (`GRAVITY_TIME_SCALE`), Hot/Top/Talk badges, `serve_window` age-out, `place_far_card` horizon, and hot-refresh/Firebase probes — none steer Recommended order.
Fix (post-trial): small calibrated additive prior — log-points/hour velocity + log-age decay, standardized, clipped like other meta cols, gated by its own ramp and zeroed for archive — not raw points. Effect: stay-current up, duds down in 12h/1d; sameness risk if popularity dominates, so cap weight and monitor novel up-votes. Effort M. Validate: temporal folds in `eval_ranker_variants.py` + short-window score-bar replay (`FINDINGS 2026-10-12`, now with `deck_scores`) + Import-AI gate as live stay-current check. Confidence medium-low on weight; high that gap is real.

**P5 — No borrowed prior for new/small profiles. Category: untried improvement.**
Evidence: `fast_rerank_for_user` 0-vote path serves cold deck; `interest_centers`/Explore need ups; joined challengers fall back to tiers 1–2 without both classes (`pipeline/joined_classifier.py:fit`, `pipeline/ranking.py:rerank_candidates`). Population prior (profile-151 centroid/cluster rates, per-source base rates) never used.
Fix: for tiers 1–2 only, blend candidate score with a frozen population centroid similarity + per-source base rate, weight decaying to 0 by ~100 votes. Helps first feed and small profiles; main-user risk ~nil if gated by vote count. Sameness: population prior could import main-user bubble — validate topic spread explicitly. Effort M. Validate: leave-one-profile-out + short-history replays from N=0; metrics: AUC, up-rate, normalized topic entropy, novel-ups/week. Confidence low-medium; needs privacy note (profiles are the user's own).

**P6 — Short-history validation doesn't measure what matters. Category: evaluation gap.**
Evidence: STATUS trial follow-up cites short-history AUC curve (#2 beats production N≥400). FINDINGS 2026-10-12 shows AUC near ceiling (0.78 vs 0.785 re-vote ceiling) while Recommended 12h is 14% up/39% down and novel ups are 6%. Curve replays score voted pools only.
Fix: standardize post-trial replay on impression/skipped pools (`_load_skipped_pool`, `_impression_pools` in `eval_ranker_variants.py`), plus per-window freshness splits and discovery guardrails (novel-ups/week, pairwise diversity). Effort S (protocol, not model). Confidence high.

## 2. Checked / already tried

- Engagement (points/comments) features rejected: archive-inflated, worse 30-day held-out — ARCHITECTURE 3.2; do not retry raw.
- Recency half-life weighting +0.004–0.006, CI covers 0; neutral-as-down no gain — FINDINGS "Blind re-vote calibration 2026-10-11"; dropped.
- Stacked/GBM/ordinal/pair second stages, TF-IDF sweeps, Untuned-embedding probe, incremental hill-climb — no adoptable win per FINDINGS ranker sections; not re-proposed.
- Joined-logistic #2/#4 already in live team-draft trial until 2026-11-05 — STATUS/INTERLEAVING.md; no production change before then.
- Discover R1 (best-score far, τ=0.60) and Far-1 last-slot validated — FINDINGS "Discover/Diversity Phase 0 2026-10-12"; leave in place.
- Cold-deck margin 32×, warm-start persistence, regen warm-wait, candidate legs (HN gravity 10k / RSS time-DESC 5k / archive 4k) look correct for throughput; RSS leg reaches full 30-day window.
- Standard-scaling only meta cols with ±2.5 clip; raw L2 embeddings untouched — correct, do not change.
- Hot-refresh, Firebase growth probes, `in_window` clamp, `serve_window` age-out, Arctic Shift 36-h score lag handling — freshness plumbing looks sound; gap is ranking weight, not data.

## 3. Open questions

- What should a new profile's first 8 cards optimize: main-profile prior, global popularity, or pure recency? Owner parked first-feed work 2026-10-12.
- Is a small gravity/popularity floor acceptable given Popular's 68% down-rate, or does Popular mislabel "current-important"?
- Which stay-current metric decides P4: Import-AI gate pass, 12h up-rate, or Hot/Top coverage?

