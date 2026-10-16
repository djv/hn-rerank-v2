## 1. Top findings / proposals (ranked)

**P1 — Cold/first feed Recommended is HN-gravity only; RSS/Reddit score 0 so never surface. — defect**
Evidence: `pipeline/__init__.py:cold_ranked_candidates` scores `hn_gravity(score…)`; RSS rows carry ~0 score (`pipeline/config.py` two-leg comment); cold deck has `explore=None` (`pipeline/__init__.py:build_cold_deck`). A 0-vote user (`fast_rerank_for_user`, 0-vote short-circuit) gets an HN-frontpage clone in every window, no Explore.
Effect: saves time OK, discovery + calmer habit bad (same-y HN-only first impression), stay-current OK for HN only. New-profile first feed — the parked goal — cannot succeed as-is.
Effort: S (cold-deck mixing rule or per-source quota / recency mix for 0-vote Recommended only).
Validate: `scripts/eval_ranker_variants.py` cold variant (`build_cold_deck`, `--folds`, held-out up-rate + novel-up rate + topic entropy); do not use AUC alone. Confidence: high.

**P2 — Tier-2 centroid-diff is crude and unregularized; best near-term gain below the gate. — untried improvement**
Evidence: `pipeline/ranking.py:1735-1780`: raw mean up minus mean down, then per-pool min-max. No shrinkage, no kNN, neutrals ignored except in `n_feedback` ramp (`:1783` `alpha_2=n_feedback/50`, neutrals count toward ramp but not direction). Single-class case uses zero-vector for missing side (sensible but unvalidated).
Effect: helps save-time + discovery for tiers 1–2 without narrowing topics (centroid is broad); faster useful personalization.
Effort: S–M. Try: kNN-mean tier-2 (`_knn_similarity` already exists), shrinkage toward global centroid, neutral-as-weak-down weight <1.
Validate: `--train-recent 50/100/200` replays (INTERLEAVING.md short-history protocol), metrics NDCG@12 + upvotes/144 + novel-ups + diversity, temporal folds. Confidence: medium.

**P3 — Dual gate + 20→80 ramp strands down-sparse users in tier-2 for a long time. — defect/evaluation gap**
Evidence: gate `n_up≥20 & n_down≥20` (`ARCHITECTURE.md:3.3`, `pipeline/config.py:min_up/down`), `alpha_3=(n_min-20)/60` (`ranking.py:1793`); user with 50 ups/5 downs sees pure tier-2. Short-history curve: at N=100 (~22 ups) all arms coincide; #2 beats production only at N≥400 (block CIs excl. zero), #4 never (INTERLEAVING.md).
Effect: slow ramp hurts save-time for light users; no sameness harm (tier-2 is broad).
Effort: M. Options: asymmetric gate (e.g. require 20 ups + fewer downs with stronger regularization), or population-prior warm start (main-profile-151 centroid/weights as prior, decay with votes) — untested and directly serves small profiles.
Validate: same `--train-recent` curve at N=50/100/200 + down-sparse subsamples; require novel-up guardrail not to drop. Confidence: medium-low (small-N variance; leakage risk — see P6).

**P4 — No importance / stay-current signal independent of taste. — untried improvement**
Evidence: Tier-1 is pure HN gravity on snapshot points; velocity badges (Hot p99.5 +20pts, Top 100, Talk 50) are display-only (`ranking.py:assemble_window_deck`); archive Popular never Hot. Short-window read: 12h/1d pools are thin (median score 0.31/0.42 vs 0.64/0.84 1w/1m), score bar rejected (FINDINGS "Short-window score bar": rescoring leaks final points/comments + `top_comments`; `_minmax01` per-pool bar not portable; costs ⅓ of ups).
Effect: stay-current is the gap; a calibrated importance prior (velocity percentile, Top/Talk, publisher freshness like Import AI gate) would lift it with small sameness cost if capped to 1–2 slots (cf. Far-card precedent).
Effort: M. Validate: after 2026-11-05 replay rank caps / `prob_down` thresholds on `deck_scores` (STATUS Next step), judged by ups/session with downs/card guardrail + novel-ups/week. Confidence: medium.

**P5 — Future-dated / clock-skew stories get age-0 maximum gravity. — defect (small)**
Evidence: `in_window` and `hn_gravity` and tier-1 all clamp age at 0 (`ranking.py:hn_gravity`, `in_window`, tier-1 list-comp). Future RSS dates (Import AI gate explicitly fails future newest issues) rank as freshest.
Effect: stay-current corruption; rare but cheap to fix (quarantine/clamp future to window edge + log).
Effort: S. Validate: unit test + `check_import_ai_recommended.py` future-date case. Confidence: high (mechanism), low (frequency).

**P6 — Short-history evidence itself is weak; fix eval before changing ramp. — evaluation gap**
Evidence: curve reuses profile-151 recent votes as a new user's first votes (INTERLEAVING.md limits); rescoring uses final metadata/embeddings (FINDINGS short-window limits); trial accrual ~13.5/day vs 50/day replay assumption (INTERLEAVING.md addendum).
Effect: prevents false ramp "wins" that narrow topics.
Effort: S (protocol, not model): frozen snapshots, `--train-recent` with time-cutoff features only, report NDCG + up-rate + diversity + novel-ups together.
Validate: re-run existing followup artifacts with cutoff-correct features. Confidence: high.

## 2. Checked — fine or already tried

- Engagement (points/comments) features: tried, rejected — inflated archive offline, worse 30-day held-out (`ARCHITECTURE.md:3.2`). Do not re-propose.
- Publication affinity / source priors: source one-hots kept for pool prior; affinity variant tried (FINDINGS ~3668); RSS-time-DESC leg + 5k limit already reaches 30d window. Fine.
- Recency weighting (+0.004–0.006, CI from 0) and neutral-as-down (no gain): tried in blind re-vote calibration. Drop.
- Joined #2/#4 as faster-ramp fix: already live challengers; below 20/20 they fall back to gravity+centroid (fit rejects missing class, `rerank_candidates` serves production alone). Lowering their gate is high-capacity (~229k TF-IDF cols) on tiny N — do not do without P6.
- Metadata ×0.5/×0.25: between #2 and #4 endpoints, neither beats both (INTERLEAVING.md). Drop.
- MMR/caps/Far-1/Discover-R1/score-bar/Teach: Phase-0 diversity/discovery reads done (MMR-0.8 free +0.040; Far-1 +0.049 just missed gate; Discover best-score R1 passes; score bar costs ⅓ ups; Teach ≈ random). Far card live as `far` arm; combo MMR+Far untested — allowed as P4-adjacent only.
- Warm-start (`warm_store.py`, `SharedFeatures`, 30s→~10s first rerank): latency only, ranking-identical by design. Fine.
- Side-vector zero-part shrinkage (`side_embeddings.py:48-54,179-188`): real but above-gate model-quality topic, out of this angle; coverage gate 0.98 handles it.

## 3. Open questions

- What should a first feed optimize — HN-best, source-mixed, or Explore-heavy — and is RSS score-0 permanent or backfillable with velocity/domain priors?
- Is down-sparsity (few downs) or total-vote sparsity the binding cold-start constraint for small profiles?
- Can `deck_scores` (live since 2026-10-12) support cutoff-correct short-history replays, or is feature-time travel still leaked?
- Should importance (velocity/Top/Talk/fresh publisher issue) be a rank term, a Popular-only rule, or 1 reserved Recommended slot?

