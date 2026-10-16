# Diversity / discovery / exploration review

## 1. Ranked findings and proposals

**1. Unsure gets first pick of the pool and starves Discover — reorder or constrain it.**
Category: defect. Evidence: `pipeline/ranking.py:2228-2250` — Explore passes run Unsure (highest entropy, `needs_probs=True`) before Discover (best score among `closest_up < 0.60`), each excluding earlier picks (`picked`), so an Unsure pick removes a far high-scored story Discover would have taken; Discover has no backfill beyond its eligible list. `get_entropy` uses the ARCH 3.3 softmax-over-margins, explicitly "not calibrated probabilities." Effect: exploration budget goes to boundary cases (often near-duplicate up/down neighbours, dud-prone) instead of quality-among-far; saves little time, weakens discovery, sameness roughly unchanged or worse. Effort: S. Validation: rerun `~/revote/discover-phase0.py` pool with pass orders swapped (Unsure-first vs Discover-first) + `sameness-metrics.py`; metrics: Explore up-rate, novel ups caught, pairwise diversity, down-rate guardrail, paired bootstrap CIs. Confidence: medium-high.

**2. Ship the already-planned Combo (MMR-0.8 × 11 + Far-1 last slot) — the only near-free diversity lever measured.**
Category: untried improvement (planned, not yet tested). Evidence: FINDINGS "Diversity Phase 0": MMR-0.8 −0.3 pts up, +0.040 diversity, 1.3 pts fewer downs; Far-1 −1.1 pts, +0.049, novel ups 4→8; neither passed the ≥0.05 gate alone; Combo "suggested by this data and not tested," fixed for Phase 0b in `docs/evaluations/diversity-20261012/PLAN.md`. Serving Recommended today is pure score order (`ranking.py:~2293` `pool[:VIEW_SIZE*margin]`; serving MMR branch dropped per ARCH 3.4; `mmr_filter:1919` eval-only). Effect: small time cost, discovery up via the far slot, staying-current unchanged, calmer habit neutral; directly targets within-topic neighbour-picking (the measured failure: diversity −0.15–0.21 vs random at equal topic entropy). Effort: S (replay script exists). Validation: Phase 0b protocol exactly (day-old models, k=8, up/down/diversity/novel-ups, short-vs-long split). Confidence: medium.

**3. The model itself rewards sameness: every semantic feature is monotone in closeness to past ups.**
Category: defect (by design, worth countering at re-rank level). Evidence: ARCH 3.2 + `pipeline/ranking.py::_svm_personalization_features` (cols: kNN-mean/max to ups, max to ups, positive-cluster max) fed by `_knn_mean_and_max`; TF-IDF logreg (`pipeline/linear_blend.py`) and joined word block (`pipeline/joined_classifier.py:_inputs/_words`) likewise reward past-up words. Down-side features only partly offset this. Effect: top-12 concentrates within covered interests; re-ranking (not refitting) is the correct remedy — keeps hits, widens topics. Effort: M. Validation: `scripts/eval_ranker_variants.py` + diversity replay; require novel-ups ≥ B and up-loss ≤2 pts before any post-trial arm. Confidence: high that the mechanism exists; medium on magnitude per deck.

**4. Replace the 40-k-means topic cap (failed: −8.1 pts up) with a cap on the existing per-user interest assignment.**
Category: untried improvement. Evidence: FINDINGS "Diversity Phase 0" — Cap-1 "loses 8 points… 40 k-means topics are too coarse"; `ranking.py:interest_picks` + `assemble_ranked_deck:badge_similarity/interest_clusters` already compute `cand_interest` (nearest of `interest_cluster_k=10` up-clusters) for every candidate but use it only for Explore. Proposal: Recommended at most 2 per interest-cluster (or round-robin least-covered-interest tie-break), reusing that array — no new clustering. Effect: topic spread with far smaller hit cost than global caps; discovery up, time roughly neutral. Effort: M. Validation: add rule to `diversity-phase0.py`; same gate + topic-entropy split. Confidence: low-medium (unmeasured; interests derive from ups only, so caps still reflect old tastes).

**5. Far/Discover eligibility uses raw full-text cosine with a known length confound — use centered similarity.**
Category: defect. Evidence: `ranking.py:328` `DISCOVER_MAX_SIM_UP=0.60` applied to raw `cand_closest_up`; ARCH 3.4/attribution: raw 0.6–0.8 for "loosely related pairs," median 0.77 on profile 151's 1w deck; `centered_pair_similarity` (mean + top-3 PCs removed) exists but only gates the "Because you upvoted" line, never Discover/far/Interest. So some "far" cards are just long-text artefacts and some truly new topics score ≥0.60. Effect: fixes both false-novel duds (time, calm) and missed novel topics (discovery, sameness). Effort: M (centering needs pool PCs per rank; cache it). Validation: rerun Discover Phase 0 at τ=0.55/0.60/0.65 with raw vs centered sim; report R1 up-rate vs random-far + novel-ups. Confidence: medium.

**6. Far-card serving is fragile and its readout is confounded — fix fallbacks before reading 11-02.**
Category: defect + evaluation gap. Evidence: `FAR-CARD-REVIEW.md` M1 (aged-out far → plain 12, other eligible far ignored; `ranking.py:2373-2439`), L4 (undrafted union stories never candidates), L2 (report `credited`/`decks_with_credit` include `far`), L7/PLAN "Far card" (far card usually Discover's #1, so live Discover rate now measures picks 2–4; served vs `deck_scores` positions differ). Effect: understates far-card availability, flatters far-vs-Discover; no aim directly affected except measurement. Effort: S. Validation: unit tests named in the review §"Missing cases" + readout joining served impression positions (post-`583d0bd`) with `deck_scores` arms, far+Discover combined per PLAN. Confidence: high.

**7. Close the offline-eval gaps before claiming any diversity win.**
Category: evaluation gap. Evidence: both Phase 0s pool only shown stories, shown-unvoted=0, snapshot embeddings include future comments, ~110 up-votes/rule and single-digit novel ups (`REVIEW-claude-cloud.md:97`); FINDINGS "Short-window score bar": a 0.346 bar cuts the *more novel* cards first. Proposal: whole-pool replay (not shown-only), topic-entropy + pairwise-diversity + novel-ups/week with total-up/down guardrails, short-vs-long split, as already specified in PLAN Phase 0/0b gates. Effect: prevents dud-cutting that narrows topics. Effort: S. Validation: `sameness-metrics.py`, `short-window-scores.py`, `eval_ranker_variants.py` on frozen snapshot DB. Confidence: high.

## 2. Checked: fine or already tried

- MMR λ sweep, topic caps, Far-1: tried Phase 0, none passed gate; Combo pending Phase 0b — endorsed, not re-proposed.
- Discover R1 (best-score-among-far, τ=0.60) live in Explore; R2/R3/R4-bandit tested offline (R4 10.4% ≈ R1 11.8%) — R4-as-live-rule remains the only unshipped variant, folded into #4.
- Novel-as-farthest (R0) ≈ random (3.9%) with ~half downs — correctly replaced, do not revive.
- Teach active-learning Phase 0: tiny gain over random — correctly parked.
- Team-draft after dedup (`pipeline/__init__.py:1086-1102`, `ranking.py:2345`) unbiased in expectation; far tagged `FAR_ARM`, uncredited — sound for trial.
- URL + embedding dedup (`pipeline/__init__.py:1116-1170`) removes near-identicals only — does not solve within-topic neighbours; consistent with sameness finding.
- Attribution centering, raw-embedding no-scale rule, hn-dupe canonicalization: sound; centering reuse proposed in #5.
- No production ranking change before 2026-11-05: all above are offline replays + post-trial arms.

## 3. Open questions

- With novel ups ~6/week and ~12/rule offline, what minimum detectable effect justifies shipping Combo vs interest-caps?
- Is τ=0.60 still right under centered similarity, and does it differ short vs long windows (Discover Phase 0: τ=0.65 → 15.3% up)?
- Should Unsure survive at all once Discover + far card cover exploration, or move to calibrated joined probs / Thompson sampling (R4) after the trial?
- New-profile first feed has no Explore (no ups) — does far-card logic (`closest_up=None` skips) leave cold decks same-y, and is that in scope post-trial?

