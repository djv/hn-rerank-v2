# Model-and-features review (read-only)

## 1. Ranked findings

**1. Live `svm_c` disagrees with ARCHITECTURE — doc drift, retune risk. Defect.**
Evidence: `config.toml [hn_rewrite.model] svm_c = 4.0`; `ARCHITECTURE.md §3.3` says "`config.toml` pins `svm_c = 0.1`"; `pipeline/config.py:25` default is 0.2. Production therefore runs the hill-climb winner (C=4, FINDINGS "Incremental hill-climb" step 10), not the documented 0.1. Effect on aims: no direct feed effect today, but any post-trial retune from the doc value regresses quality (duds back, no sameness benefit). Effort S (one-line doc fix + assert effective config in evals). Validate: `Config.load("config.toml")` dump in `scripts/eval_ranker_variants.py` header; re-run C sweep only to confirm. Confidence high.

**2. Side-embedding zero-fill shrinks uncovered stories. Defect (already verified, still live).**
Evidence: `pipeline/side_embeddings.py:48-54` (`side_by_side` divides by √2) with `179-188` (missing → zero vector): uncovered norm 1/√2 vs 1 covered; gate `side_embedding_min_coverage=0.98` still lets ~2% through shrunk. STATUS/FINDINGS 2026-10-14 confirms. Effect: penalizes exactly the rare/new-source stories discovery needs; cuts novel up-votes while barely moving AUC. Effort S (renormalize covered part, or exclude-then-fallback per story, not pool-wide). Validate offline: `eval_ranker_variants.py`, stratified covered-vs-uncovered slice, metrics P@12 + novel-up + non-HN share, not AUC. Confidence high.

**3. `neutral_weight` is a dead knob; neutrals train as a full third class. Defect / untried.**
Evidence: `pipeline/config.py:36` defines `neutral_weight=0.0`, but `pipeline/ranking.py:1592-1600` builds balanced weights over all three classes and never reads it. Meanwhile blind re-votes show ~half of neutrals flip to down, and dwell shows neutral≈up in reading time (STATUS 2026-10-11/13). "Neutral-as-down" was tried with no gain — different from *down-weighting* neutrals (e.g. 0.25/0.5) or neutral-excluded scoring. Effect: neutrals are the swing mass between duds and discovery; correct weighting should cut downs without narrowing topics. Effort S. Validate: `eval_ranker_variants.py`/`eval_one_classifier.py` weight grid on recent + historical blocks; require NDCG@12 + down@12 + novel-up. Confidence medium.

**4. Metadata scale was fixed, never tuned — and one-hots have outsized leverage. Untried.**
Evidence: `pipeline/config.py:63-65` (`joined_embedding_weight=16`, `joined_numeric_scale=sqrt(0.1/4)`); `pipeline/joined_classifier.py::_inputs`; `pipeline/ranking.py:1602-1604,1690` (StandardScaler on meta only + clip ±2.5). FEATURE-REMOVALS shows removing metadata trades AUC (−0.009) for top-pick NDCG (+0.015): scale matters at the top. One-hot source bits after StandardScaler are high-leverage and directly set the per-source prior. Effect: rescaling shifts HN vs RSS/Reddit mix — the sameness lever; monitor non-HN up-votes per change. Effort S–M (grid over 2 params offline). Validate: `eval_single_preference_model.py` / `eval_one_classifier.py` on 12-block suite; gate on NDCG@12 + novel-up, AUC secondary. Confidence medium.

**5. Scores are pool-relative and uncalibrated — bars/thresholds don't transfer; Unsure entropy rests on softmaxed margins. Evaluation gap / defect.**
Evidence: `pipeline/ranking.py:1191,1723` (`_minmax01` per pool), `pipeline/linear_blend.py:percentile_scores/blend_scores` (rank percentiles), `pipeline/ranking.py:1718` (`_softmax_rows` over OVR margins → `prob_*` → Unsure). The 2026-10-12 score-bar check already found a fixed bar doesn't carry across pools. Effect: no serving change now, but future dud-cutting (bars, `prob_down` caps) and Unsure picks need calibrated scores or they will silently cut novel ups (the bar-cut ups were more novel). Effort M (log raw margins in `deck_scores`; fit Platt/isotonic offline; reliability diagrams). Validate: held-out calibration + replay rank-cap after 2026-11-05 per FINDINGS "Short-window score bar". Confidence medium-high.

**6. k-NN/cluster features are redundant and partially un-tuned; cluster sim lacks LOOCV. Untried/defect.**
Evidence: `pipeline/ranking.py::_knn_mean_and_max` — mean-k, max, argmax all from one dot matrix; plus `_positive_cluster_similarity` (k=4) from the same embeddings; `knn_k=10`, `positive_cluster_k=4` flat on old features (FINDINGS 2026-09-25) but never re-gridded on 394-d + side vectors (now wider `emb_dim`). Training k-NN excludes self (`_loocv_knn_features`) but cluster centers fit on all ups including self. Effect: redundant similarity columns waste capacity small data can't afford; k too small narrows to near-dupes (sameness), too large blurs. Effort S. Validate: grid k∈{5,10,25} × clusters {2,4,8} + drop-max/drop-cluster variants in `eval_ranker_variants.py`; watch topic entropy + novel-up. Confidence: redundancy high; gain low-medium. Uncertain whether LOOCV-on-clusters moves metrics.

**7. Text-length/embedding input mismatch for empty `text_content`. Possible defect (uncertain).**
Evidence: `pipeline/ranking.py::story_embedding_text` falls back to `compose_story_text` when `text_content` is empty, but candidate/training lengths use `len(s.text_content)` (feature-prep blocks ~1430/1560). Such stories embed long text yet report length 0. Effect: tiny pool-wide, but concentrated on thin RSS rows discovery needs. Effort S. Validate: count affected rows in a snapshot; ablation using composed length. Confidence low-medium (needs code-path confirmation against live rows).

**8. Word-model domain/source tokens may dominate TF-IDF; never ablated. Untried.**
Evidence: `pipeline/linear_blend.py:45-51` (`tfidf_text` = `dom_* src_*` + title + 5k chars; 2¹⁸ hash, keep≥2 stories). Fresh-vote evals show TF-IDF 0≈0.3>0.5 (plateau), and SINGLE-MODEL word-C screens move AUC without consistent top-pick wins — consistent with domain tokens acting as a second source prior that fights the 4-d one-hot. Effect: dropping/weighting `dom_/src_` tokens could shift source mix (sameness lever) either way. Effort S. Validate: token-ablation variants in eval; metrics NDCG@12 + per-source up-rate + novel-up. Confidence low-medium.

## 2. Checked — fine or already tried (do not re-propose)

- Raw embeddings unscaled, meta-only StandardScaler + clip: correct; keep.
- RBF SVM + dense + TF-IDF blend standing: removals tested 2026-10-07 (no removal preserves ordering; word-removal gain unproven).
- Single joined logistic (#2/#4) and linear/hybrid screens: tried; live trial in progress — await 2026-11-05.
- Stacked GBM/pairwise ranker: rejected as implemented (worse P@12, more downs).
- Engagement (points/comments) features: rejected (0.635).
- Field-level/body+comments vectors: raised AUC but cut P@12 and discovery 1.25→0.12 — rejected.
- Gemma side-by-side at 128 tokens: the one shipped gain; Gemma2/harrier tie — no re-encode.
- MMR/topic caps/far-card: diversity Phase 0 — no rule passed gate; Far 1 live as last-slot only.
- Recency weighting, neutral-as-down, unvoted-as-negatives, kNN-in-blend, up-minus-down scoring: no gain.
- Tier blend/dual gate (20/20→80), LOOCV self-exclusion, dedup/feedback-URL exclusion: sound.
- Missed-upvote 95.8% miss rate is a judged-pool artifact, not a ranker blind spot per se.

## 3. Open questions

- Which `svm_c` history applies to side-vector features — was C=4 re-confirmed after Gemma shipped, or only on stored vectors?
- Should `deck_scores` log raw margins + coverage flags so post-trial bar/cap replays are pool-comparable?
- What is the neutral-weight + dwell-weight interaction — are long-dwell neutrals soft ups (weight up) or satisficed skips (weight down)?

