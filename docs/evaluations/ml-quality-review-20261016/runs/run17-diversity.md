# Diversity / discovery / exploration review

## 1. Ranked findings & proposals

**1. Ship the untested Combo: MMR(0.8) + Far-1 last slot in Recommended.**
Category: untried improvement. Evidence: `docs/evaluations/diversity-20261012/PLAN.md` Phase 0b (already specifies this exact rule, data maturing 2026-11-05); `FINDINGS.md` "Diversity Phase 0" — Far 1: −1.1 pts up-rate, +0.049 diversity, 8 novel ups vs baseline 4 (just missed the 0.05 bar); MMR 0.8: −0.3 pts, +0.040, fewer downs. Neither passed alone; the combination was suggested but never run. Effect: keeps hits (both near-free on up-rate) while widening topics; sameness falls since only slot 12 is forced-far and MMR-0.8 overlaps baseline 88%. Effort: S (re-rank rule on production order, then interleaving arm per PLAN Phase 1). Validation: rerun Phase 0b script (`~/revote/diversity-phase0.py` pattern) on post-10-13 days; gate per PLAN (diversity ≥ +0.05, up-rate ≥ −2 pts, novel ups ≥ B), paired bootstrap, split short/long windows. Confidence: medium.

**2. Defect: Recommended is pure score order; the only anti-sameness slot cannibalizes Explore.**
Category: defect. Evidence: `pipeline/ranking.py:2142` (`assemble_window_deck` — "Recommended: the top stories by model score, no source quota", no MMR in serving; `enable_mmr` branch dropped per ARCHITECTURE 3.4); `pipeline/ranking.py:2373` (`place_far_card` removes the far story from Explore: `explore = tuple(r ... if r.story.id != far.story.id)`). So Discover's measured up-rate from now on is picks 2–4, and the far card is usually Discover's top pick (PLAN "Far card" readout caveats admit this). Effect on aims: saves time (keeps 11 best) but caps discovery at 1 card/window and muddies the Discover readout. Effort: S — keep far card in both slots with distinct badges, or exclude slot-12 from Discover baselines in analysis. Validation: `scripts/preview_explore.py` + `deck_scores` join on (dashboard_version, story_id) per PLAN; compare far-card vs Discover-2–4 up-rate and combined novel ups. Confidence: high.

**3. The model itself rewards sameness: 3 of 10 meta-features are up-similarity.**
Category: defect (by design, worth countering at re-rank level). Evidence: `pipeline/ranking.py:_svm_personalization_features` (cols 1/3/5 = k-NN-mean-to-ups, max-to-up, cluster-to-up, all mapped to [0,1]) fed by `_knn_mean_and_max` (single dot pass per class); plus tier-2 `sim_up − sim_down` centroid term (`ranking.py:~1780`) blended with weight α₂ from 50 votes. A near-duplicate of a past up scores on all four simultaneously. `FINDINGS.md` "Sameness" confirms the symptom: pairwise diversity 0.15–0.21 below random within covered topics (entropy same — neighbours, not missing topics). Effect: any dud-cutting intensifies the bubble unless paired with a spread constraint. Effort: M to retrain with a novelty feature; S to penalize at selection (proposals 1, 4). Validation: `scripts/eval_ranker_variants.py` — add mean-pairwise-diversity + novel-ups (sim<0.60) to `_metrics` alongside AUC/NDCG; ablation zeroing cols 1/3/5 at inference (like the existing `strip_hn` pattern). Confidence: high for the mechanism, medium for size.

**4. Replace coarse 40-topic caps with interest-coverage penalty; never use score bars.**
Category: untried improvement (caps already tried and rejected — do not retry as-is). Evidence: `FINDINGS.md` "Diversity Phase 0": Cap-1 lost 8.1 pts up-rate for +0.044 diversity ("topics too coarse"); score-bar check: cut 12h/1d ups were the *more novel* ones (median sim 0.73 vs 0.78). Instead reuse `interest_picks` machinery (`ranking.py:2250`, round-robin over `interest_centers`, k=10, least-covered first) as a soft penalty inside Recommended: `score − λ·(already-picked share of candidate's interest)`. Effect: widens within-topic neighbours (the actual diagnosed gap) with ~zero hit cost; topic spread rises without the Cap-1 collapse. Effort: M. Validation: same Phase-0 harness, k=8, report up-rate/diversity/novel-ups; then team-draft arm vs production. Confidence: medium.

**5. Defect: Unsure entropy is computed from uncalibrated pseudo-probabilities.**
Category: defect. Evidence: `pipeline/ranking.py:_softmax_rows` applied to raw SVM `decision_function` margins for non-joined fits (`probs = _softmax_rows(decision)`), then `get_entropy` ranks Unsure; ARCHITECTURE 3.3 admits "not calibrated probabilities". Softmax over margins exaggerates certainty gaps, so entropy order need not track true uncertainty. The joined-logistic challengers (`joined_all`, already fitted live in every warm) *have* fitted probabilities. Effect: exploration budget spent on wrong cards; calmer habit unaffected, discovery weakened. Effort: S–M — rank Unsure by challenger entropy (or temperature-scale/Platt-scale the SVM margins) behind a flag. Validation: existing `scripts/eval_unsure_votes.py` + Teach simulation (`scripts/simulate_teach.py` showed entropy barely beats random — re-run with calibrated probs). Confidence: medium (mechanism certain, magnitude unmeasured).

**6. Calibrated exploration next step: Thompson sampling over interests (R4), not more entropy tuning.**
Category: untried improvement (partially tried: R4 offline hit 10.4% up vs 4.5% random-far, `docs/evaluations/discovery-20261012/PLAN.md` + FINDINGS "Discover Phase 0"). Promote R4 from a Discover pick rule to one Explore slot with a Beta(ups, shown) per interest cluster. Effect: systematic novel-interest trials; short-window pools stay fresh (stays current), bounded to 1–3 slots (calmer habit). Effort: M. Validation: offline replay on shown-pool days (same harness as Discover Phase 0); live as interleaving arm with novel-ups/week guardrail. Confidence: low–medium (only ~12 ups/rule offline).

**7. Evaluation gap: three definitions of "novel" disagree; diversity is not a standard eval metric.**
Category: evaluation gap. Evidence: live/Phase-0 novel = max-sim-to-earlier-ups < 0.60 (`DISCOVER_MAX_SIM_UP`, `ranking.py:DISCOVER_MAX_SIM_UP`); `eval_ranker_variants.py:_novel_candidate_ids` splits at the fold *median* instead; sameness topics (40 k-means) differ again. `eval_ranker_variants` reports AUC/NDCG but no spread metric, so ranker changes ship blind to sameness. Fix: unify on 0.60 + report mean-pairwise-diversity, topic entropy, novel-ups@12 for every variant; after 11-05 replay caps/bars on `deck_scores` (STATUS "Next step" already schedules this). Effort: S. Confidence: high.

## 2. Checked — fine or already tried

- Discover R1 (best-scored far story, τ=0.60) passed its gate and is live in Explore (`ranking.py:assemble_window_deck` Explore passes; 3/badge served) — don't replace, measure ~11-02.
- MMR 0.6/0.7 clears diversity but costs −4 pts up-rate; topic caps lose more — correctly rejected (FINDINGS Diversity Phase 0).
- Old Novel (farthest-from-votes) ≈ random (3.9% vs 4.5%) — correctly replaced.
- Team-draft after dedup/canonicalization + `far`-arm exclusion keeps trial unbiased (`pipeline/interleave.py:team_draft`; `ranking.py:2345,2373`) — sound.
- Dedup at cosine 0.87 (`config.py:ModelConfig`) removes dupes, not neighbours — fine, not a diversity lever.
- Side-by-side embeddings keep badges/Explore on stored vectors (`ranking.py` ctx branch) — thresholds keep meaning; fine.
- `serve_window` per-badge caps then score-sort; Interest least-covered-first — reasonable; badge-mixing at serve is cosmetic.
- Blind re-vote AUC headroom +0.007 ± 0.12, recency/neutral-as-down no-gain — not re-proposed.
- No restart/production change before 2026-11-05 — all above are offline-then-arm proposals.

## 3. Open questions

- Should τ be per-window (0.65 scored 15.3% up long-window vs 8.0% at 0.55)? Phase-0b data can answer.
- This checkout's `config.toml` shows `svm_c = 4.0` while ARCHITECTURE cites 0.1 — which served profile 151, and does C affect neighbour-picking sharpness?
- Can `deck_scores` (live since 10-12) give served-position novel-ups/week before 11-02, or only after?
- Does calibrated (joined) entropy actually beat raw-margin entropy on `eval_unsure_votes`?
