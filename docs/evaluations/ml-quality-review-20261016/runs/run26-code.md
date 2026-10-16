## 1. Top findings, ranked

**1. Side-vector zero-fill demotes uncovered stories — defect**
Evidence: `pipeline/side_embeddings.py:179-188` (`vectors.get(s.id, zero)`); coverage gate `pipeline/ranking.py:~1345` + `side_embeddings.py:163-176`; already confirmed in `STATUS.md` Result 2026-10-14.
Trigger: coverage ≥ `side_embedding_min_coverage` (0.98) but <100%: the 0–2% without a side row get a zero side half, so `side_by_side` (`/sqrt(2)`) shrinks their joined norm while covered rows keep full norm. Newest stories are the likeliest uncovered.
Aims: hurts discovery/stay-current directly (fresh items demoted for a pipeline lag, not taste); sameness worsens (pool tilts to well-embedded old stock). Save-time roughly neutral.
Effort: S (score uncovered rows stored-only, or mean-fill + missing indicator; never zero-fill into the same RBF space).
Validate offline first: `scripts/eval_ranker_variants.py` temporal holdout, primary top-12 upvote rate + discovery upvotes (below-median sim to train ups) and non-HN upvotes; compare zero-fill vs fallback at simulated 97–99% coverage. Confidence: high.

**2. Scaler fitted on dummy zero rows for absent classes — defect**
Evidence: `pipeline/ranking.py:1584-1605` (concat `np.zeros` for `missing` classes, then `scaler.fit_transform(fb_features[:, emb_dim:])` unweighted; dummy `sample_weight` 1e-6 never reaches `StandardScaler`).
Trigger: any voter missing a class — almost always neutral for small/new profiles. Profile 151 likely has all three, so the live trial is least affected; the "first feed" aim is most affected.
Aims: systematic cold/small-user degradation; no sameness benefit, just noise in meta scaling (length/sims/source bits).
Effort: S (fit scaler on real rows only, then transform dummies; add regression test).
Validate: unit test on scaler mean/var with/without dummy + `eval_ranker_variants.py` on short-history slices (N=20–200, cf. INTERLEAVING.md short-history curve); watch top-12 up-rate, not AUC. Confidence: high on bug, medium on size.

**3. Joined challengers fail closed whenever any class is absent — defect**
Evidence: `pipeline/ranking.py:1584-1645` + `pipeline/joined_classifier.py:88-126` (`len(stories)!=len(labels)` raises); `rerank_candidates` treats `svm_fit` label as "did not fit" and serves production alone (`pipeline/ranking.py:2534-2546`). Documented as "missing a class falls back" (`ARCHITECTURE.md` 3.3).
Trigger: voter with zero neutrals (plausible for small profiles) → both `joined_all`/`joined_no_metadata` arms always error, `interleave=off`, trial loses power and the served sample is conditioned on neutral presence.
Aims: hurts discovery/current measurement (challenger never tested where it might differ most); sameness unknown.
Effort: S (train joined on real rows only with its own dummy-free path, or require only up+down for fit).
Validate: 2-class vote-set unit test + `scripts/simulate_interleaving.py` power replay. Confidence: high.

**4. Positive-cluster similarity has no LOOCV (train/inference mismatch) — defect**
Evidence: training `fb_positive_cluster_sim` uses centers fit on the same rows (`pipeline/ranking.py:~1560-1575`); KNN means/max correctly use `_loocv_knn_features`, cluster max does not.
Trigger: small `n_up` (20–80, the ramp region): self contributes ~1/20 of a center, inflating the training column the SVM then trusts.
Aims: overfits the dominant up-cluster → same-y feed risk; likely small save-time loss on fresh votes.
Effort: S/M (leave-one-out centers for train rows, or ablate column 5).
Validate: `eval_ranker_variants.py` column ablation on temporal + fresh-vote splits; metrics top-12 up-rate, discovery upvotes, down share. Do not re-tune C/gamma jointly (hill-climb already did: FINDINGS "Incremental ranker hill-climb"). Confidence: medium.

**5. `neutral_weight=0.0` is a dead knob — defect/evaluation gap**
Evidence: `pipeline/config.py:36` declares it; no read in `pipeline/` (only docs hits); actual weights are balanced `n/(classes*count)` (`pipeline/ranking.py:~1596-1600`), so neutrals are fully weighted despite the 0.0 name.
Trigger: always. Either docs or code is wrong.
Aims: if neutrals should be ignored, current code blurs boundaries (more duds/neutrals up top); if 0.0 is stale, remove it to avoid a future "tune" that does nothing. Sameness effect unknown.
Effort: S (wire it or delete it; document choice).
Validate: `eval_ranker_variants.py` neutral-weight sweep on temporal folds with top-12 + down-share gates. Confidence: high on dead code, low on which direction helps.

**6. No NaN/degenerate-fit guard after scoring — evaluation gap**
Evidence: tier-2 `(x-min)/(max-min+1e-8)` and `_minmax01` (`span<=1e-8→0.5`) let a single NaN embedding poison all scores (`pipeline/ranking.py:~1690-1720`); uniform decisions give uniform `_softmax_rows` → arbitrary Unsure picks. No finite-check before sort.
Trigger: corrupt/short-text embedding row or degenerate fit (cf. ARCHITECTURE 3.3: `C=0.5` all-negative).
Aims: rare but catastrophic dud ordering; no sameness effect when fixed (fail closed to gravity/centroid + log).
Effort: S (`nan_to_num`/finite mask + `svm_fit=degenerate` label + test).
Validate: fault-injection test + existing suite `uv run pytest tests/ -n 4`. Confidence: low (trigger unobserved).

## 2. Checked, looks fine or already tried

- Raw embeddings never standard-scaled (only `[:, emb_dim:]` scaled, `ranking.py:1602-1610`); matches AGENTS.md rule.
- KNN train/inference LOOCV with self-exclusion correct; fused `_knn_mean_and_max` equivalent.
- One vote snapshot shared by all arms (`rerank_candidates:2501-2530`); fixes 2026-10-08 inter-arm race.
- Draft-after-dedup/canonicalization + per-round random order (`draft_recommended`, `team_draft`); 3/2/7 split cause addressed.
- `SharedFeatures` reuse across arms is read-only after build; key covers feature settings; challenger word-count sharing is safe.
- Model-cache signature includes vote hash, embedding bytes, labels/lengths/sources, full `config.model` (`ranking.py:~1335-1360`); schema v6.
- Missing-embedding dim mismatch treated as miss (`database.py:_decode_embedding_blob`); no crash path seen.
- Single-row/tied `_minmax01→0.5` vs `percentile_scores→1.0` inconsistency is harmless (rank-invariant within one rank).
- SELECT_MARGIN/VIEW_SIZE/EXPLORE_PER_BADGE arithmetic consistent (24/12/6→3); far card tagged `far`, excluded from credit.
- Rejected directions not re-proposed: stacked ranker (FINDINGS 2026-09-29, inner-leak + top-12 loss), C/gamma hill-climb (kept C=4+blend), TF-IDF sweep (FINDINGS 2026-09-29), EmbeddingGemma-2 side model no-gain (FINDINGS 2026-10-07), reassessment "no missing feature" (RANKING-REASSESSMENT-20261009), diversity Far-1/MMR/topic-caps gate (diversity-20261012).
- `config.toml` live values noted: `svm_c=4.0`, blend + precomputed + side on, `interleave_user_ids=[151]`; no production change proposed before 2026-11-05.

## 3. Open questions

- Is `neutral_weight=0.0` intent "ignore neutrals" or a stale default? Answer determines fix for #5.
- Does any live source use the `reddit_` (no `rss_`) prefix? If so `source_category_onehot` maps true Reddit to generic RSS (idx3 not idx2).
- `side_embedding_dim=768` (config) vs actual EmbeddingGemma row dim — is the DB `expected_dim` filter silently causing the <100% coverage in #1?
- `interest_centers` cache key omits `k`; harmless across restarts but unsafe for in-process k-sweeps — worth a key fix with #4?

