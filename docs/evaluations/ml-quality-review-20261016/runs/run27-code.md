# Ranking code-path review: correctness bugs

## 1. Top findings, ranked

**1. Neutrals inflate the tier-2 ramp and contribute no signal (defect).**
`n_feedback = len(feedback_labels)` counts neutrals, and `alpha_2 = n_feedback / tier2_blend_window` (ranking.py:1783) ramps gravity→centroid on that total — but tier-2 centroids use only up/down means (ranking.py:~1745–1763), and neutrals-only yields a flat 0.5 vector. Trigger: a new profile that taps "meh" often (e.g. 30 neutrals + 5 ups) reaches `alpha_2≈0.7`, so a 5-upvote centroid displaces gravity. Effect: hurts save-time (ranking on a tiny centroid) and *increases sameness* (centroid overfits few ups); discovery/currentness roughly neutral. Effort S (ramp on `n_up + n_down`). Validate offline with `scripts/eval_ranker_variants.py` on a neutral-heavy slice (user 151 votes relabeled, or a small profile): metric top-12 upvote rate plus `scripts/narrowing_report.py` topic spread; confidence medium.

**2. Side-vector zero-fill shrinks fresh stories' similarities (defect; already verified, not new).**
STATUS.md (2026-10-16) records this from the 2026-10-14 review: stories lacking a side vector get a zero side part, so `side_by_side` scales their stored part to 1/√2 norm (side_embeddings.py:48–54, 179–188). Trigger: any story ingested after the 30-min side-embed timer's last run — exactly the newest stories — gets ~halved k-NN similarities while `coverage ≥ 0.98` keeps side mode on. Effect: directly hurts stay-current and discovery (fresh/diverse items down-ranked), mild same-y tilt toward older embedded stock; save-time mixed. Effort S–M (skip-zero-part similarity or hold out uncovered stories from side mode). Validate with `eval_ranker_variants.py` split by side-coverage flag + `source_yield_report.py` freshness; confidence high.

**3. "Probabilities" are softmaxed OvR margins; the binary branch is dead (defect / eval gap).**
Dummy zero rows guarantee 3 classes, so `decision.ndim == 1` (ranking.py:1715) is unreachable and candidate `prob_*` always come from `_softmax_rows(decision)` (ranking.py:1722) — uncalibrated margins, not probabilities. These feed `get_entropy` → Explore Unsure picks. Trigger: any personalized deck; Unsure ordering inherits arbitrary margin scaling. Effect: wastes Explore slots (hurts discovery/save-time slightly), no sameness effect. Effort S (Platt/calibration check or entropy on calibrated scores; at minimum a test pinning 3-class shape). Validate via `scripts/eval_unsure_votes.py` (do high-entropy picks actually convert?) and `preview_explore.py`; confidence medium (dead branch high, impact medium).

**4. Dummy zero rows are included in the scaler fit (defect, tiny).**
Zero placeholder rows for absent classes are concatenated *before* `scaler.fit_transform(fb_features[:, emb_dim:])` (ranking.py:~1595–1604). Trigger: any profile missing a class (typical near the 20/20 gate: 1 dummy in ~41 rows shifts meta mean/var slightly). Effect: negligible ranking shift, but it also slightly moves the ±2.5 clip boundary; no aims impact worth naming. Effort S (fit on real rows only, transform dummies after). Validate: refit determinism check in `eval_one_classifier.py`; confidence high on existence, high on tininess.

**5. Tie/single-row normalizations disagree; zero-vote path sorts by raw points (defect, eval hygiene).**
`_minmax01` returns 0.5 on ties while tier-2's inline `(x−min)/(max−min+1e-8)` (ranking.py:1759) returns 0; and `n_feedback == 0` returns `sorted(... story.score ...)` raw points (ranking.py:1886) whereas the production cold path (`build_cold_deck`) uses gravity. Trigger: single-candidate windows, identical-sim pools, or any offline caller of `_score_and_rank` with zero votes. Effect: cold/eval rankings diverge from what production serves (new-profile first feed is the stated parked goal); no sameness effect. Effort S. Validate: unit test + `eval_ranker_variants.py` cold-start split; confidence high.

**6. Candidate-pool embeddings go stale within a regen cycle (defect).**
`get_candidate_pool` (candidate_cache.py) materializes embeddings once; enrichment landing afterwards (article-body fetch, prewarm hydration changing `text_content`) isn't re-embedded until `invalidate_candidate_pool()`, and the model-cache key only hashes feedback text lengths, never candidate hashes. Trigger: a story enriched after pool build, re-ranked by a vote-triggered warm in the same hour. Effect: small save-time/discovery loss on exactly the stories being invested in (article fetches); bounded by one cycle. Effort M (re-hash touched rows on warm, or invalidate on enrichment). Validate: replay comparing pool vs fresh vectors on touched ids; confidence low-medium (ordering of regen phases unconfirmed from code alone).

## 2. Checked and fine (or already tried)

- Source one-hot covers live sources: `rss_reddit_*` (both RSS and Arctic Shift via `_rss_source_name`, enrichment.py:817/763–813), `rss_ainews`, `rss_community_openai_com` all hit intended buckets; legacy `reddit_*`→RSS bucket is legacy only (`relabel_legacy_sources.py` exists).
- Embeddings never standard-scaled: scaler fits `features[:, emb_dim:]` only (ranking.py:1604), meta from `emb_dim:` onward per AGENTS.md.
- Train/inference k-NN asymmetry (LOOCV `_loocv_knn_features` vs full-set candidates) is correct by design; offline callers must pass `exclude_feedback=True` (the flag exists).
- No shared-mutable-state bug across interleave arms: `SharedFeatures` is filled once, read afterwards; `cand_word_counts` cached read-only; `team_draft` RNG is per-warm.
- No off-by-one in margins: `VIEW_SIZE×margin` (24) consistent across pool slice, `draft_recommended` limit, and arm `tops`; far card correctly takes served slot 12.
- `neutral_weight` (config.py:36) is a dead knob — never read; harmless, remove opportunistically.
- `_ranking_copy` stripping TLDR text is safe: ranking reads only `text_content`/title/url/counts, all retained.
- Zero-variance meta columns can't NaN (sklearn zeroes the scale + ±2.5 clip); empty-string embeddings still pool over special tokens.

## 3. Open questions

- Does the regen invalidate the candidate pool strictly after all enrichment (prewarm/article fetch), or can pool build precede hydration? Determines finding 6's severity.
- Should neutrals enter tier-2 at all (e.g. as a repulsive centroid), or stay out while excluded from the ramp denominator?
- For the live trial: do challenger arms' `score_context=None` Explore-less paths bias arm-credit comparisons if Explore picks drive votes?
