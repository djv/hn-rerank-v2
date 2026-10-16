# Diversity / discovery / exploration review

## 1. Top findings, ranked

**1. The scorer itself rewards sameness; all diversity lives in post-passes — add a novelty term or fix the features.**
Category: defect. Evidence: `pipeline/ranking.py:_svm_personalization_features` (5 of 10 meta cols are `sim_to_up/closest_up/cluster_sim`), `_knn_mean_and_max` (one dot-pass per class, reused for attribution/Discover), `_positive_cluster_centers(k=4)`; `pipeline/linear_blend.py:tfidf_text` (domain+title+5k chars, words in ≥2 stories) and `pipeline/joined_classifier.py:_inputs` (embedding×√16 + TF-IDF words). Every learned component upweights near-duplicates of past ups; nothing penalises "5th AI-agent story this week". ARCH 3.2 confirms engagement-meta exclusion but keeps all similarity cols.
Effect: saves time (+) but directly causes the same-y failure (FINDINGS "Sameness": −0.15–0.21 pairwise diversity vs random, same topic entropy; only 6% novel ups). Any dud-cutting without touching this narrows topics further.
Effort: M (one feature + re-tune `svm_c`/`joined_c`). Validation: `scripts/eval_ranker_variants.py` temporal blocks + `~/revote/sameness-metrics.py` (pairwise diversity, topic entropy, novel-up recall alongside NDCG/up-rate; day-old-model protocol from diversity Phase 0). Confidence: high that it drives sameness; medium on the fix size.

**2. Ship the untested Combo (MMR-0.8 head + Far-1 tail) — do not ship MMR or Far alone.**
Category: untried improvement. Evidence: `docs/evaluations/diversity-20261012/PLAN.md` Phase 0b; FINDINGS "Diversity Phase 0": MMR-0.8 −0.3 pts up / +0.040 diversity, Far-1 −1.1 pts / +0.049 / 8 vs 4 novel ups; neither passed the ≥0.05 gate; Combo proposed but never run. Serving has no MMR (`ARCHITECTURE.md` 3.4: serving branch dropped, `mmr_filter` eval-only, `pipeline/ranking.py:mmr_filter`).
Effect: keeps hits (head stays production order) while guaranteeing one far hit + within-topic spread; widens topics without a bar's novelty cost (cf. score-bar check cutting the more-novel 12h ups). Effort: S (compose two tested functions; `place_far_card` + `mmr_filter` already exist). Validation: Phase 0b script on 2026-10-13→11-04 data, same gate (diversity ≥0.05, up diff ≥−2, novel ups ≥ B), short/long split. Confidence: medium.

**3. Unsure explores on uncalibrated pseudo-probabilities — replace with margin / fitted probabilities.**
Category: defect. Evidence: `pipeline/ranking.py:_softmax_rows` + `_score_and_rank` (`probs=_softmax_rows(decision)`); ARCH 3.3: "not calibrated probabilities… only for uncertainty entropy"; `get_entropy` on them picks Unsure. `JoinedLogistic.predict_proba` keeps fitted probs but serves only as trial challenger (`pipeline/joined_classifier.py`, `pipeline/config.py:classifier`). Live: Unsure ~5% up, Interest 9%, Novel 2% (`discovery-20261012/PLAN.md`).
Effect: misdirected exploration wastes the 3 Explore slots (calmer-habit cost, no discovery gain). Effort: S–M (score Unsure by `|margin|`/distance-to-boundary or challenger probs; no retrain). Validation: `scripts/eval_unsure_votes.py` + Discover Phase-0 replay (up-rate vs random-far, novel ups). Confidence: medium.

**4. Discover's far set has no within-far spread — cap per interest/topic inside τ<0.60.**
Category: untried improvement. Evidence: `pipeline/ranking.py:assemble_window_deck` Discover pass (best `score` among `closest_up<DISCOVER_MAX_SIM_UP=0.60`, no backfill) and `place_far_card` (best-scored far); `interest_picks` round-robins only for Interest. Discover Phase 0: R1 (best score) 11.8% beats random but R4 topic-bandit 10.4% nearly ties (`discovery-20261012/PLAN.md`, FINDINGS "Discover Phase 0") — topic spread was never combined with best-score.
Effect: prevents 3 Discover + 1 far card all from the same far cluster; discovery (+) with ~no hit cost. Effort: S (reuse `interest_centers`/40-topic k-means; at most 1 per interest among far candidates). Validation: same Discover replay + `sameness-metrics.py` novel-up/topic-entropy split. Confidence: medium.

**5. Dedup threshold (0.87) does not touch the sameness problem — keep URL dedup, add a soft within-topic penalty instead of lowering it.**
Category: defect/evaluation gap. Evidence: `pipeline/config.py:dedup_embedding_cosine_threshold=0.87`; `pipeline/__init__.py:finalize_ranked_deck` (dedup→canonicalize before `draft_recommended`/`place_far_card`); FINDINGS "Live feed metrics": 11 near-dup pairs/822 story-days — dedup works, yet pairwise gap persists because neighbours at 0.6–0.8 pass.
Effect: lowering to ~0.75 would drop valid distinct stories (sameness ≠ dupe); a soft MMR-style penalty keeps hits while widening. Effort: M. Validation: `cap_sweep.py`/`narrowing_report.py` + diversity Phase-0 harness; guardrail: novel ups/week. Confidence: low-medium on value; high that threshold-tweaking alone is wrong (topic caps already worst trade: −8.1 pts for +0.044).

**6. Offline evals cannot measure discovery — add diversity metrics to the ranker eval.**
Category: evaluation gap. Evidence: `scripts/eval_ranker_variants.py`, `eval_one_classifier.py`, `simulate_interleaving.py` report AUC/NDCG/up-rate only; diversity/discovery rely on laptop `~/revote/*.py` over shown-only pools (selection bias, snapshot-time comments/points — stated limits in both Phase-0 FINDINGS). `deck_scores` (live since 2026-10-12, STATUS.md) now logs serve-time scores/arms/positions but no replay uses it yet (STATUS "After 2026-11-05: replay rank caps and score bars on deck_scores").
Effect: without this every future "−1 pt up-rate, +0.05 diversity" trade is judged on biased pools; stay-current/sameness guardrails unenforceable. Effort: M (add pairwise 1−cosine, topic entropy, novel-up recall to existing replay; full-pool replay post-trial). Validation: itself — backfill on Oct data, compare shown-pool vs full-pool rows. Confidence: high.

## 2. Checked — fine or already tried, not re-proposed

- MMR λ∈{0.6–0.9} and per-topic caps (40 k-means): tried, gate failed; caps worst trade — FINDINGS "Diversity Phase 0".
- Novel = farthest-from-all-votes: tried, ≈random (3.9% vs 4.5%) — "Discover Phase 0"; correctly replaced by Discover R1.
- Topic-bandit R4 alone, source-quality R2, R3 blend: tried; R1 simplest winner — same section.
- Teach active-learning simulation: tiny gain over random, parked — FINDINGS "Teach Phase 0".
- Fixed score bar (0.346/0.505): cuts a third of short-window ups and the more-novel ones; min-max scores don't transfer — "Short-window score bar".
- Far-card trial safety (arm=`far`, no credit, 11-card draft unbiased in expectation): verified — `FAR-CARD-REVIEW.md`, `place_far_card`/`serve_window`/`draft_recommended`.
- Side embeddings keeping attribution/Explore/dedup on stored vectors: correct — `pipeline/side_embeddings.py:model_space`, ARCH 3.4.
- Explore excluding Recommended + feedback-dupe backfill; Interest round-robin over least-covered interests: sound, keep.
- Cold-deck gravity for zero-feedback users: acceptable; first-feed work parked per user.

## 3. Open questions

- Should `closest_up=None` (no up-votes) also suppress Discover, matching the far-card `has_upvotes` fix?
- Does `serve_window` need a fallback far list (FAR-CARD-REVIEW M1) or is horizon-preference enough — check Nov-02 readout miss rate?
- Which topic definition (Interest k=10 vs 40 k-means vs source) should be the standard spread metric for post-trial replays?
- Uncertain: exact τ for within-far caps; R4 bandit needs live vote feedback per topic not in offline pools.

