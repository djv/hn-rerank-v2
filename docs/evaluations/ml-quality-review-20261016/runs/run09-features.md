## 1. Top findings, ranked

**1. Uncovered stories get a zero side-vector half — representation defect.**
Category: defect. Evidence: `pipeline/side_embeddings.py:48-54` (`side_by_side` divides by √2) + `model_space` zero-fill fallback; STATUS.md 2026-10-16 already verified "norm shrinkage for uncovered stories". A story missing a side vector enters the SVM/linear models with half its embedding energy gone, so RBF distances to voted stories shrink systematically for exactly the newest/most-novel candidates. Effect: hurts stay-current and discovery, no sameness benefit — it penalizes far-from-upvote stories for an indexing reason. Effort: S (mean-impute per-part renormalisation, or exclude missing rows from the √2 scaling; keep the 0.98 coverage gate). Validate: `scripts/eval_ranker_variants.py` production vs patched scorer on user 151 temporal folds, metrics P@12 / NDCG@12 / Down@40 plus novel-upvote count; confirm on `--confirmation` newest-20%. Confidence: high.

**2. Training cluster feature lacks the LOOCV that k-NN features have.**
Category: defect. Evidence: `pipeline/ranking.py` — `_loocv_knn_features` excludes self, but `fb_positive_cluster_sim` is scored against centers fit on the same `fb_up_embs` with no exclusion. Training ups are biased high on meta col 5 (`_svm_personalization_features` layout), inflating its RBF importance; candidates at inference score systematically lower — a train/serve shift. Prior ablation already hints the column is weak: ARCH §3.13 `margin3_plus_cluster` ≈ −0.014 NDCG@40. Effect: small save-time loss; fixing (LOOCV cluster sim or dropping col 5) should not narrow topics. Effort: S. Validate: same eval harness, `margin3_up` vs `margin3_plus_cluster` vs LOOCV-cluster variant; watch AUC + P@12 + topic entropy. Confidence: medium.

**3. Double normalization discards margin magnitude before blending.**
Category: defect/untried improvement. Evidence: `pipeline/ranking.py:_minmax01` squashes the SVM margin per pool, then `pipeline/linear_blend.py:187-196` (`blend_scores`) percentile-ranks production, dense, and word scores before the 0.5/0.2/0.3 mix. A strong TF-IDF hit on a novel story gets the same percentile step as a tiny margin gap elsewhere; cross-window and cross-vote magnitudes are destroyed (also pollutes `deck_scores` comparability). Prior "Stacked ranker" rejection (FINDINGS.md 2026-09-29) does not cover this — that was a GBM/pairwise second stage, not a normalization fix. Effect: likely suppresses discovery/current outliers; magnitude-preserving blend (z-scored raw `up_minus_down`) should widen spread without adding duds. Effort: S/M. Validate: `prodlr[svm_c=4;lr_weight=0.2;tfidf_weight=0.3]` vs raw-score-blend variant; report P@12, NDCG@12, Down@40, novel/non-HN upvotes. Confidence: medium.

**4. Stale hyperparameter story + dead knob invite mistuning.**
Category: defect (docs/config). Evidence: `config.toml:37` live `svm_c = 4.0`, but ARCH §3.3 still says "`config.toml` pins `svm_c = 0.1`" and `ModelConfig.svm_c` default is 0.2 (`pipeline/config.py`); `neutral_weight = 0.0` is declared (`pipeline/config.py:36`) but never read — grep shows sample weights built inline in `ranking.py` with neutrals fully weighted. Gamma 0.03 was kept when the space grew 384→1152-d side-by-side. Effect: no direct ranking harm today, but the next tuner will misread the baseline; gamma deserves one cheap re-sweep under the live space. Effort: S. Validate: `--sweep-svm` grid around C 2–8 × γ 0.015–0.06 with side vectors on; gate on confirmation folds, not dev. Confidence: high (docs/knob), medium (gamma gain).

**5. "Other" sources share one implicit prior — discovery risk for rare feeds.**
Category: untried improvement. Evidence: `pipeline/ranking.py:source_category_onehot` returns all-zeros for non-matching sources (ARCH §3.2 calls this the "implicit other prior"). Any new/rare source inherits a prior learned mostly from majority RSS behavior. Differs from tried work: FEATURE-REMOVALS tested dropping *all* metadata (#4 arm); this is per-source handling (explicit other bit, pooled-rare intercept, or inference-time prior backoff for cold sources). Effect: neutral on save-time, potential discovery gain for novel feeds; no narrowing. Effort: S. Validate: held-out eval split by source rarity + novel-source upvote recall; interleaving arms already log `joined_all` vs `joined_no_metadata`. Confidence: low-medium, marked uncertain.

**6. Uncertainty/threshold path is uncalibrated — evaluation gap.**
Category: evaluation gap. Evidence: ARCH §3.13 admits neither the `_softmax_rows` proxy (`pipeline/ranking.py`) nor joined-logistic probabilities establish calibration, yet Unsure entropy, `DISCOVER_MAX_SIM_UP = 0.60`, and far-card rules depend on them; side-by-side mode re-ranks on joined vectors while badges/attribution stay on stored vectors (`side_embeddings.py` docstring; `ranking.py` ctx recompute). The 0.60 threshold was tuned pre-side-vectors. Effect: mis-set Unsure/Discover boundaries cost both duds-avoidance and discovery. Effort: S/M. Validate: reliability diagram of prob_up vs up-rate on `deck_scores`+impressions; re-tune 0.60 under side mode offline first. Confidence: medium-high.

## 2. Checked — fine or already tried

- Raw L2 embeddings unscaled, scaler+clip only on meta cols — correct; do not touch.
- k-NN LOOCV self-exclusion with `-1.0`/mask — correct as implemented.
- TF-IDF 1–2g over title+body+domain, C/weight sweep — tried, keep (FINDINGS.md "TF-IDF tuning sweep" 2026-09-29).
- Stacked GBM/pairwise second stage — tried and rejected (FINDINGS.md "Stacked ranker").
- EmbeddingGemma-2 side model, bge/arctic swaps, short-context probes — tried; Gemma-1 side-by-side stands (FINDINGS.md hill-climb steps 6–10; "EmbeddingGemma 2").
- Engagement (points/comments) features — tried, hurt (Ranking-quality study 2026-09-25); keep excluded.
- Joined-logistic #2/#4 as production replacement — already the live interleaving challengers; await 2026-11-05, do not pre-judge.
- MMR/topic-cap diversity rules — Phase 0 gated, none passed; Far 1 live as tagged card only (diversity-20261012/PLAN.md).
- Muse pairwise feature mining — reassessed, build-nothing verdict stands (RANKING-REASSESSMENT-20261009.md).
- Single-positive-KMeans k=4 and interest k=10 — noted above; broader cluster sweeps were flat (hill-climb; FINDINGS.md).

## 3. Open questions

- What fraction of live misses are coverage-zero side vectors vs genuine preference misses?
- Does the 0.60 Discover/far threshold transfer to side-by-side geometry, or is it systematically off?
- Is col-5 cluster signal redundant given k-NN means + Interest clusters, i.e. safe to delete?
- Which normalization (raw, z, percentile) preserves novel-story margins without inflating duds?

