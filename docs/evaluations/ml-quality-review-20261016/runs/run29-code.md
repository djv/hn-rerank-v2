# Ranking code-path review: correctness bugs and proposals

## 1. Ranked findings

**1. `neutral_weight` is a dead knob; neutrals train as a full class.**
Category: defect. Evidence: `pipeline/config.py:36` defines `neutral_weight = 0.0`, but no code reads it (`grep` finds only the definition and ARCHITECTURE text). `pipeline/ranking.py:1592-1599` weights every real vote `n_real/(n_classes*counts[lbl])`, neutrals included, and the SVM/dense/TF-IDF all fit 3-class targets. Either the config lies (neutrals were meant to be ignored) or the docs lie. Effect on aims: a full neutral class pulls OvR boundaries toward bland middle-ground stories and flattens entropy (Unsure picks); zero-weighting could cut duds but risks narrowing to polarized topics — either direction must be checked against novel up-votes. Effort: S. Validation: `scripts/eval_ranker_variants.py`, neutral sample-weight {0 vs balanced}, metrics top-12 up-rate + top-12 down-share + novel up-votes, temporal folds. Confidence: high that the mismatch exists; medium on which direction is correct.

**2. Cluster-similarity feature leaks at train time; LOOCV under-reports small classes at train time.**
Category: defect. Evidence: `pipeline/ranking.py:1206-1223` (`_loocv_knn_features`) correctly excludes self for kNN means, but the positive-cluster feature for training rows (`~ranking.py:1530s`, `_similarity_to_positive_cluster_centers` on centers fitted with the row included) has no LOOCV — training ups score systematically higher than identical candidates. Conversely, a singleton class's training rows get similarity 0 (empty reference after self-removal) while an identical candidate gets ~1.0. Trigger: profiles near the 20/20 gate, few neutrals, or small `positive_cluster_k` relative to ups; effect washes out at 3,000 votes but matters for small/other profiles and the new-profile first feed. Effect: miscalibrated scores for young profiles, slightly worse cold-to-warm transition; no sameness effect expected. Effort: M (k-fold centers or leave-one-center-out). Validation: short-history curve (`docs/evaluations/model-ablation-20261007/INTERLEAVING.md` follow-ups) at N=40–200, AUC + top-12. Confidence: medium.

**3. Scaler is fitted on dummy zero rows for absent classes.**
Category: defect (small). Evidence: `pipeline/ranking.py:1592ff` — zero placeholder rows for missing classes (e.g. no neutrals, the common case) are concatenated *before* `StandardScaler.fit_transform` on the meta block. One all-zero row in 40+ shifts mean/variance slightly; relatively larger exactly for the small profiles the gate just admitted. Trigger: any vote set missing a class. Fix: fit on real rows, transform all. Effort: S. Validation: unit test (mean/var unchanged by dummies) + short-history curve at N≈40–80. Confidence: high it happens, low-medium on measurable gain.

**4. Uncovered side vectors shrink the joined vector (already verified, still the sharpest live defect).**
Category: defect (confirmed 2026-10-14, STATUS "Codebase review"). Evidence: `pipeline/side_embeddings.py:48-54,179-188` — uncovered stories get a zero side part, norm 1/√2 vs 1.0 for covered. Trigger: any fresh story arriving between 30-min side-embed timer runs while coverage stays ≥0.98, so the rerank stays in side-by-side mode with a few shrunk rows. Effect: newest stories (stay-current aim) systematically misranked in exactly the windows where currency matters; sameness neutral. Effort: S/M — renormalize joined rows to unit norm (changes the eval-replay layout, so it is a re-validation, not a free fix). Validation: rerun the 2026-10-02 side-by-side replay (`FINDINGS.md` "Fresh-vote…evals") with renormalization + a synthetic holdout of uncovered rows. Confidence: high on defect, medium on fix choice.

**5. Zero-vote cold path disagrees with the served cold deck.**
Category: defect (small, mostly eval-hygiene). Evidence: `pipeline/ranking.py:1886` returns `sorted(... story.score ...)` (raw points) while `build_cold_deck` (`pipeline/__init__.py`, `cold_ranked_candidates`) uses `hn_gravity`. `fast_rerank_for_user` short-circuits 0-vote users to the cold deck, so production is consistent — but any eval calling `_score_and_rank` with 0 votes measures a different cold start than served. Trigger: offline cold-start/new-profile experiments. Fix: one line (use `hn_gravity`). Effort: S. Validation: existing `tests/test_pipeline.py` cold-deck tests + a cold-start assertion. Confidence: high.

**6. Gravity goes to exactly zero past 50 votes — consider a small residual for stay-current.**
Category: untried improvement (differs from tried work). Evidence: `pipeline/ranking.py:1783`, `alpha_2 = clip(n_feedback/50,…)` → `t1_weight = 0` for heavy voters; currency then comes only from window boundaries. Already-tried neighbor: learned engagement columns in the SVM scored 0.635 and were rejected (FINDINGS "Ranking-quality study — 2026-09-25") — this proposal differs: a tiny *fixed* residual (e.g. 5–10% gravity/velocity), not learned weights. Effect: surfaces fast-moving important stories; risk is popularity bias narrowing discovery — guard with topic entropy + novel up-votes, keep far card/Discover as backstops. Effort: S. Validation: `eval_ranker_variants.py` short-window splits + missed-upvotes lens (`docs/evaluations/missed-upvotes-20261009`). Confidence: low-medium.

**7. Embeddings freeze at first sight; comment growth never re-enters similarity.**
Category: evaluation gap. Evidence: `get_or_compute_embeddings` keys on the text hash, and enrichment writes `top_comments` while `text_content` (the embedding input when non-empty, `story_embedding_text`) stays fixed — later discussion never re-embeds. Worth testing, not asserting: re-embed `refresh_grown_threads` touched stories and measure. Risk: discussion-rich = popular threads, could narrow topics; check novel up-votes. Effort: M. Validation: offline replay on touched-story cohorts, top-12 + diversity metrics, before any pipeline change. Confidence: low (may be pure noise).

## 2. Checked: fine or already tried

- Interleaving arm sharing is sound: separate `_MODEL_CACHE` entries (signature hashes `asdict(config.model)`), one vote snapshot for all arms (fix `623321b`), `team_draft` after dedup/canonicalization; counts 24→12 (11+far) consistent across `assemble_window_deck`/`draft_recommended`/`serve_window`/`place_far_card`.
- Source one-hot covers live sources (`rss_reddit_` checked before `rss_`; Codex/AINews are `rss_*`-prefixed → rss bit); all-zero "other" only for non-`rss_` strays.
- `_minmax01` ties→0.5 (`ranking.py:1191`), single-row percentiles→1.0: graceful degradation to tier blend, no crash path.
- Scaler zero-variance handled by sklearn + `clip(-2.5, 2.5)`; no NaN source found in meta columns.
- Candidate-pool invalidation on regen (`server.py:2326-2334`) correct; per-vote warms reuse within one cycle (see §1.7 for the only freshness gap).
- Already tried and not re-proposed: stacked ranker, engagement metadata, bge/Gemma swaps, TF-IDF sweep, MMR/topic caps (FINDINGS "Stacked ranker", "Ranking-quality study", "TF-IDF tuning sweep", "EmbeddingGemma 2"; `diversity-20261012/PLAN.md` Phase 0 gate).

## 3. Open questions

- Is `neutral_weight=0.0` live intent (wire it into sample weights) or a stale doc (delete the knob)?
- Does blending stored-space tier-2 centroid scores with side-by-side-space SVM scores distort heavy-voter order, or is per-component [0,1] normalization sufficient?
- Which enrichment writes actually rewrite `text_content` post-embedding, and how many pool embeddings per regen are stale because of it?

