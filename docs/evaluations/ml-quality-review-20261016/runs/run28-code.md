## 1. Top findings, ranked

**1. Side-vector zero-fill shrinks uncovered stories' norm — silent demotion.**
Category: defect (already flagged in STATUS/FINDINGS "side-vector norm shrinkage", still live in code).
Evidence: `pipeline/side_embeddings.py:48-54` (`side_by_side` joins then `/sqrt(2)`); `:179-188` (missing side → zero vector, joined anyway when coverage ≥ threshold).
Trigger: any rerank with coverage in `[0.98, 1)` — up to ~2% of the pool (~120–200 stories) get norm ≈0.707 vs 1.0 for covered stories, so the RBF kernel systematically down-weights them regardless of content. Newest stories and newly-added sources are the ones most often uncovered.
Effect: hurts discovery (novel topics punished) and stay-current (fresh stories punished); reduces sameness only by accident, at the cost of duds-for-new-items. Save-time roughly neutral, calmer habit neutral.
Effort: S (score uncovered rows on the stored-only path, or renormalize/mask; don't ship zero-fill).
Validation: `scripts/eval_ranker_variants.py` side-mode on time-block splits; metrics NDCG@12 + novel-upvote rate + topic spread; compare covered-vs-uncovered rank shift per story.
Confidence: high.

**2. Zero-feedback path sorts by raw score, not gravity — diverges from the served cold deck.**
Category: defect.
Evidence: `pipeline/ranking.py:1886-1887` (`n_feedback == 0` sorts by `x.story.score`) vs `pipeline/__init__.py:cold_ranked_candidates` (gravity) and `_score_and_rank`'s own `tier1_scores` (gravity, normalized).
Trigger: any direct `_score_and_rank` call with empty feedback — offline eval folds with zero training votes, scripts, and any future new-profile first-feed reuse. Live `fast_rerank_for_user` short-circuits to the cold deck, so production is unaffected today.
Effect: evals misread cold quality (old archive behemoths with score ≥200 outrank fresh HN); if reused for new profiles it harms save-time, stay-current, and discovery breadth.
Effort: S (sort the zero-feedback path by the computed gravity score).
Validation: `scripts/eval_ranker_variants.py` cold folds + `scripts/preview_explore.py`-style cold-deck dump; metrics NDCG@12 and served-age distribution.
Confidence: high on the code mismatch, medium on user impact (live path bypasses it).

**3. Cluster-similarity feature has no self-exclusion; KNN features do.**
Category: defect.
Evidence: `pipeline/ranking.py:1512` (LOOCV `_loocv_knn_features` excludes self) vs `:1178,1427-1430` (centers fit on all up-embeddings including the row being featurized, train and inference share them).
Trigger: small up sets (20–40 ups — exactly the dual-gate region and all small profiles). Training cluster-sim is inflated by self-contribution; the scaler is fit on the inflated column while candidates at serve time see lower values.
Effect: miscalibrated cluster weight for new/small profiles; likely small same-y push toward the dominant cluster. Hurts discovery slightly; other aims neutral.
Effort: S (per-row leave-one-out or center-recompute; or measure-and-document if negligible).
Validation: short-history curve from `docs/evaluations/model-ablation-20261007/INTERLEAVING.md` followups at N=20–200; metrics AUC + top-12 upvotes + topic spread.
Confidence: medium.

**4. Dummy zero rows for absent classes enter the scaler fit.**
Category: defect (small).
Evidence: `pipeline/ranking.py:1576-1604` (missing classes appended as all-zero rows *before* `scaler.fit_transform(fb_features[:, emb_dim:])`).
Trigger: voter with no neutral votes (common) — one zero meta-row among ~41 rows at the gate (~2.4% pull on mean/var), negligible at 3k votes. So this bites precisely the small/new profiles.
Effect: slight systematic skew of all scaled meta columns; direction unpredictable; no sameness effect expected.
Effort: S (fit scaler on real rows only, then transform dummies).
Validation: unit test + short-history curve at small N; metric AUC delta.
Confidence: high on mechanism, low on magnitude.

**5. Failed linear-blend refit can leave a stale word model in place.**
Category: defect (rare path).
Evidence: `pipeline/ranking.py:1658-1679` — on `fit_linear_blend` exception the handler logs but keeps the pre-miss cached `linear_models`; `pipeline/linear_blend.py:175-189` then scores with that stale IDF/`keep` mask under the new scaler. Reachable only on SVM-miss + linear-hit + refit exception (OOM/tokenizer edge).
Trigger: as above; serves a stale TF-IDF weighting silently instead of falling back to production score.
Effect: transient ranking corruption; no systematic sameness effect.
Effort: S (set `linear_models = None` in the `except` arm).
Validation: `tests/test_linear_blend.py`-style fault-injection test; offline replay with forced failure asserting fallback scores equal production.
Confidence: medium.

**6. Candidate pool freezes embeddings for a generation; enrichment after the build is invisible to warms.**
Category: defect / cache staleness (needs live order confirmation).
Evidence: `pipeline/candidate_cache.py` (`_ranking_copy`, `get_candidate_pool` computes embeddings once, no per-rank hash recheck) + `server.py:2328-2333` (invalidate at cold-deck rebuild).
Trigger: a story whose `text_content` hash changes (article-body fetch, comment prewarm, growth-hydration heal) *after* the pool build but before personal warms in the same generation — warms rank the stale vector. Uncertain: exact invalidate-vs-enrichment ordering was not fully traced from this copy.
Effect: fresh context ignored → hurts stay-current and discovery; save-time mildly negative.
Effort: M (per-rank hash revalidation of pool rows, or invalidate after each enrichment phase).
Validation: `rank_perf` timing + enrichment-aware replay on recent time blocks; metrics top-12 churn and NDCG@12; confirm live ordering via regen logs before building.
Confidence: medium (mechanism verified in code; live trigger order uncertain).

**7. No guard pins the scaler split at the embedding boundary.**
Category: evaluation gap.
Evidence: `pipeline/ranking.py:1604,1691` slice at runtime `emb_dim` (384 stored-only vs 384+768 side-joined). Correct today, but a side-dim/model swap silently moves the split; `_MODEL_SCHEMA_VERSION` and `feature_settings` (`ranking.py:236-250`) don't name the dim explicitly.
Trigger: future side-model change (cf. Gemma-1 vs Gemma-2 study in FINDINGS).
Effect: preventive; a mistake here would break cosine structure (the exact failure AGENTS.md warns about) and could collapse ranking while looking healthy.
Effort: S (assert `emb_dim` against configured dims; bump schema on dim change; add a side-dim-shuffle eval variant).
Validation: type/test gate + `eval_ranker_variants.py` side-mode run.
Confidence: high the guard is missing, medium on risk.

## 2. Checked — fine or already tried

- KNN train/inference self-exclusion parity; raw L2 embeddings never standard-scaled (meta-only slice) — correct.
- `_minmax01` ties → 0.5, `percentile_scores` single-row → 1.0, tier-2 constant pool → ties not NaN — no NaN path found.
- `SharedFeatures` reuse across interleave arms is safe: separate scaler/SVM fits per arm, separate model-cache keys via `asdict(config.model)`, one shared vote snapshot — the 2026-10-08 one-row-short failure mode stays fixed.
- Arctic Shift keeps `rss_reddit_*` sources via `enrichment.py:817-820` — no one-hot flip on `reddit_source` change; Codex Forum (`rss_community_openai_com`) groups with generic RSS by design.
- Far card tagged `FAR_ARM`, excluded from team-draft credit; draft-then-place ordering preserves arm balance — matches INTERLEAVING.md.
- Warm-store fits never score (solver seeds only); restored IDF-less models correctly can't predict — as documented.
- Already-tried, not re-proposed: stacked ranker, incremental hill-climb, TF-IDF sweep settings, EmbeddingGemma-2 side model, MMR/topic caps, metadata-scale middle variants (FINDINGS.md ranker sections; `model-ablation-20261007/` FEATURE-REMOVALS/SINGLE-MODEL/INTERLEAVING/BROADER; RANKING-REASSESSMENT-20261009; diversity/discovery/missed-upvotes 20261009–12 docs).

## 3. Open questions

- Live ordering: does `invalidate_candidate_pool` run after *all* enrichment (article/comment heals), or can warms slip in on pre-enrichment vectors?
- How often do live voters lack neutral votes (dummy-row trigger rate), and how large is the scaler shift at N≈40?
- Magnitude of the cluster self-bias at the gate — worth a 1-day offline measurement before fixing?
- Side-coverage distribution per window: is the sub-1.0 tail concentrated in fresh/RSS stories (highest discovery cost)?
