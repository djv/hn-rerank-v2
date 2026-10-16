## 1. Top findings / proposals (ranked)

**P1 — Ship Combo (MMR-0.8 + Far-1 tail) as an interleaving arm, not as production order.**
Category: untried improvement. Evidence: `pipeline/ranking.py:mmr_filter` exists but ARCHITECTURE 3.4 states serving `enable_mmr` branch was dropped — Recommended is pure score order; `config.py:ModelConfig.enable_mmr=False`; `assemble_window_deck` Recommended = `pool[:VIEW_SIZE*margin]`. FINDINGS "Diversity Phase 0": MMR-0.8 cost −0.3pts up-rate for +0.040 diversity; Far-1 cost −1.1pts for +0.049 and doubled novel ups (8 vs 4); neither passed the ≥0.05 gate alone; Combo suggested but untested (`diversity-20261012/PLAN.md` Phase 0b).
Effect: saves time ~neutral (MMR-0.8 held up-rate), discovery up (Far slot forces one far story; MMR spreads within-topic neighbours — the measured gap is pairwise diversity −0.15–0.21 with topic entropy flat, i.e. close neighbours within topics, FINDINGS "Sameness"), stay-current neutral (head unchanged), calmer habit up (less same-y). Effort: S (both rules coded; `place_far_card` + `mmr_filter` composition only). Validation: run Phase 0b script (`~/revote/diversity-phase0.py` logic) on post-2026-10-13 days after 2026-11-05 with `deck_scores`-logged serve-time scores; gate unchanged (diversity ≥+0.05, up-rate ≥−2pts, novel ups ≥B); then 3-week team-draft arm via `pipeline/interleave.py:team_draft`. Confidence: medium.

**P2 — Defect: KNN/cluster features directly reward near-duplicates of past ups.**
Category: defect. Evidence: `pipeline/ranking.py:_svm_personalization_features` meta cols 1/3/5 (`sim_to_upvoted`, `closest_upvoted`, `positive_cluster_similarity` from `_knn_mean_and_max` / `_positive_cluster_similarity`, k=10, 4 clusters); attribution median raw sim 0.77 (`centered_pair_similarity` comment). Far vs near up-rate 11% vs 22% (FINDINGS "Sameness") confirms the model pays for closeness.
Effect: fixing reduces sameness at the source rather than patching order afterward; risk is cutting hits if penalised crudely — must be a *feature*, not a filter. Proposal: add one novelty meta column (e.g. `1−closest_up`, or distance to nearest interest center) and let the SVM/learned blend price it, or cap `closest_up` contribution above ~0.85. Keeps hits (learned weight) while widening topics. Effort: M (feature + `_MODEL_SCHEMA_VERSION` bump + retrain). Validation: `scripts/eval_ranker_variants.py` temporal folds; report NDCG/AUC *plus* pairwise diversity, topic entropy, novel-up recall on held-out ups (`sameness-metrics.py` definitions, tau 0.60). Confidence: medium; uncertain magnitude, direction sound.

**P3 — Unsure entropy is uncalibrated exploration; replace with Thompson/ensemble uncertainty.**
Category: untried improvement. Evidence: `pipeline/ranking.py:get_entropy` over `_softmax_rows(decision)`; ARCH 3.3: "softmax values are not calibrated probabilities". `discovery-20261012/PLAN.md` R4 (topic-bandit Thompson) matched R1 offline (10.4% vs 11.8%) but never shipped; Teach parked after entropy showed +0.002 vs random (FINDINGS "Teach Phase 0"). Unsure also vanishes pre-gate (`prob_down is None`).
Effect: discovery up with fewer dud explores (samples where model is genuinely uncertain, not just mid-score); time/calmer neutral-to-up if down-rate falls. Effort: M (Beta-per-topic or dropout-ensemble variance; reuse `interest_centers` clusters as bandit arms). Validation: `scripts/simulate_teach.py` protocol + Discover Phase-0 replay (up-rate of top-3 far/unsure picks vs random-far with paired bootstrap). Confidence: low-medium (R4 evidence thin, n≈12 ups/rule).

**P4 — Extend Interest round-robin coverage into Recommended tail (slots 9–11), keep Far-1 at 12.**
Category: untried improvement. Evidence: `assemble_window_deck:interest_picks` already round-robins least-covered interests first for Explore; Recommended has no coverage term. `place_far_card` already pulls from `views.explore is_discover` and deletes it from Explore — cannibalising Discover measurement (PLAN.md "Far card" readout caveat).
Effect: widens topics inside the high-attention view without spending an extra slot; far-card stays the novelty extreme, coverage slots fix within-topic bunching. Small hit risk in slots 9–11 (slide-ins already 12% vs 22% up, FINDINGS "Live feed metrics"). Effort: S-M. Validation: same Phase-0 replay with k=8 + coverage-tail variant; metrics per rule incl. short/long split; guardrail novel ups/week. Confidence: medium.

**P5 — Evaluation gap: offline pools contain only shown stories, so deeper-pool diversification is unmeasurable today.**
Category: evaluation gap. Evidence: both Phase-0 plans' "Limits: pool holds only stories already shown"; `serve_window` caps + `SELECT_MARGIN=2`/`COLD_DECK_MARGIN=32` mean ranks below ~24 were never screened. `deck_scores` logging live since 2026-10-12 (STATUS) now stores serve-time scores — the fix the "Short-window score bar" review explicitly deferred to post-2026-11-05.
Effect: all above proposals risk under/over-estimating from censored pools; closing it protects all four aims. Effort: S (replay rank-cap / prob_down-threshold on `deck_scores` + `simulate_interleaving.py` power replay). Validation: re-run diversity/discovery replays on full candidate pools with day-old models; report ups/session with downs/card guardrail. Confidence: high that the gap exists; medium on size.

## 2. Checked — fine or already tried

- MMR 0.6/0.7 clears diversity bar but loses −4pts up-rate; topic caps worst trade (Cap-1 −8.1pts) — tried, drop (FINDINGS "Diversity Phase 0").
- Novel-as-farthest ≈ random (3.9% vs 4.5%); Discover R1 best-scored-far 11.8% passes gate — tried, shipped as `DISCOVER_MAX_SIM_UP=0.60` — keep.
- Dedup (URL + 0.87 cosine, `config.py:dedup_*`) correctly removes identicals; 11 near-dupes/822 story-days confirms it is not the sameness cause — fine.
- Engagement features excluded (archive inflation); source 4-one-hot kept (~0.38 NDCG lift) — tried, keep (`ARCHITECTURE 3.2`).
- Fixed score bar (0.346) rejected: min-max per-pool scores don't transfer; costs 1/3 of short-window ups — tried, drop.
- Side-embedding zero-fill norm shrinkage (`side_embeddings.py:48-54`) is real but coverage-gated off path; linear-blend/TF-IDF already capture lexical novelty — no action pre-trial.
- Raw embeddings never scaled (only meta cols `emb_dim:` onward, clipped ±2.5) — contract honoured; no proposal touches it. No DB deletion proposed.

## 3. Open questions

- What is the target diversity level? Random is an upper bound; how much pairwise-diversity gain (≈+0.04–0.05) is worth −1pt up-rate to the owner?
- Should Unsure optimise learning (Teach) or reading pleasure (Discover)? The two objectives pick different cards.
- Does `interest_cluster_k=10` still split current ups sensibly, or has drift made coverage slots redundant?
