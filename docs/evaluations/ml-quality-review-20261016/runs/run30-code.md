## Ranking code-path review: correctness

### 1. Top findings, ranked

**1. `neutral_weight = 0.0` is a dead knob — neutrals train at full weight.**
Category: defect. Evidence: `pipeline/config.py:36` defines `neutral_weight = 0.0`; nothing in `pipeline/` reads it (only definitions in `config.py`, `ARCHITECTURE.md:157`, `WORKLOG.md:10819`). `pipeline/ranking.py:~1590-1600` builds balanced weights as `n_real / (n_classes * counts[lbl])`, giving neutrals a full ~1/3 of mass. Trigger: any profile with neutrals — profile 151 has ~1,400 (see followup JSON label counts `{"0": 1597, "2": 835, "1": 1416}`). A neutral-heavy voter gets a boundary pulled toward "meh," which can suppress near-miss novel topics (discovery/sameness risk) while cutting some duds. Effort: S (wire the knob in or delete it). Validate: `scripts/eval_ranker_variants.py` temporal split, neutral-weight sweep; report NDCG **plus** novel up-vote rate and topic spread, not AUC alone. Confidence: medium (mechanism certain; whether 0.0 or removal is right needs the eval — note FINDINGS already reports "neutral as down" gave no gain, which is a different setting).

**2. Dummy zero-rows for absent classes contaminate the scaler fit.**
Category: defect. Evidence: `pipeline/ranking.py:~1575-1610` — when a class is missing, zero rows are appended to `fb_features` *before* `scaler.fit_transform(fb_features[:, emb_dim:])`. Trigger: any fit missing a class, e.g. a 20-up/20-down/0-neutral profile fits the scaler on 40 real rows + 1 zero row (~2.5% contamination, shifting meta means and shrinking variances). Negligible for profile 151 (all classes present) but it skews exactly the small-profile first-SVM fits the project cares about. Effort: S (fit scaler on real rows only, transform dummies after). Validate: unit test + short-history-curve replay (the `INTERLEAVING.md` followup harness) at N=40–200 with a held-out neutrals-absent slice. Confidence: high on mechanism, low on magnitude.

**3. Zero side-part shrinks uncovered stories' vector norm (known, still live).**
Category: defect (already verified 2026-10-16; STATUS "Codebase review"). Evidence: `pipeline/side_embeddings.py:48-54` (`side_by_side` divides by √2) + `:179-188` (missing stories get a zero side part when coverage ≥ `side_embedding_min_coverage = 0.98`). Trigger: coverage in [0.98, 1.0) lets up to ~2% of stories (typically the *newest* candidates the 30-min timer hasn't encoded) rank with norm ~0.707 vs 1.0 for covered ones, distorting RBF distances. Hurts stay-current most (newest stories penalized/shifted arbitrarily) and discovery if novel topics arrive via new sources. Effort: S/M (per-row fallback to stored-only, renormalize present part, or gate at 1.0). Validate: side-mask replay — mask the newest N candidates' side vectors, measure top-12 up-vote yield + freshness distribution, not just AUC. Confidence: medium-high.

**4. Offline zero-feedback path disagrees with the served cold deck.**
Category: evaluation gap. Evidence: `pipeline/ranking.py` tail (`if n_feedback == 0: return sorted(... story.score ...)`) orders by raw HN points, while the served path short-circuits before `_score_and_rank` — `pipeline/__init__.py:1037-1049` → `build_cold_deck` → `cold_ranked_candidates` (`:158-166`) uses `hn_gravity`. Trigger: any eval calling `_score_and_rank` directly with 0 votes (new-profile first-feed studies) measures raw-score order — old 200+-point archive seeds on top — instead of the gravity order new users actually see. That overstates archive bias and misjudges first-feed sameness. Effort: S (point new-profile evals at `build_cold_deck` or document the split). Validate: rerun the short-history/new-profile curve both ways; compare first-feed age distribution and source mix. Confidence: high.

**5. Interest picks complement the pre-draft top-12, not the served (drafted) set.**
Category: defect, trial-only. Evidence: `pipeline/ranking.py:assemble_window_deck` calls `interest_picks(pool, recommended[:VIEW_SIZE], picked)`, but `draft_recommended` reorders Recommended afterward from `arm_rankings`. Trigger: every interleaved deck for profile 151 — Interest is optimized to cover what production's top-12 misses, while the served set is a production/#2/#4 draft plus far card. Mild effect: Explore↔Recommended interest overlap or duplicated coverage. No fix before 2026-11-05 (frozen trial). Effort: S (recompute or pass drafted IDs post-trial). Validate: offline replay measuring interest-coverage against drafted vs pre-draft top-12 + Discover overlap rate. Confidence: medium.

**6. Tier-2 all-equal pool normalizes to 0; SVM ties normalize to 0.5.**
Category: defect, trivial. Evidence: tier-2 `(t - min)/(max - min + 1e-8)` vs `_minmax01` (span ≤ 1e-8 → 0.5). Trigger: degenerate pool with identical centroid sims (e.g. no up/down embeddings). One branch votes "worst," the other "middle." Effort: S (use `_minmax01` for tier-2). Validate: unit test. Confidence: high mechanism, low impact.

### 2. Checked — looks fine or already tried

- No standard-scaling of raw embeddings: only `[:, emb_dim:]` meta columns pass through `StandardScaler` (`ranking.py` fit + `svm_candidate_scale`); embedding block is concatenated unscaled — per AGENTS.md rule. One line.
- Train/inference feature parity: `emb_dim = model_cand_emb.shape[1]` is shared by both scalings, so side-by-side doubling applies to both — no slice mismatch. One line.
- `SharedFeatures` reuse across arms is intended: `feature_settings` covers votes/vectors/knn/clusters/side/publication/engagement/dedup; challengers differ only in classifier/blend knobs, so sharing `cand_features`/`fb_features`/`model_space` is correct; `cand_word_counts` is filled once, read-only after. One line.
- LOOCV self-exclusion present (`_loocv_knn_features` with `np.delete`); singleton-class rows yield 0/0, gated out in practice by the 20/20 dual gate. One line.
- Source one-hot covers the live taxonomy: Reddit/Arctic both flow through `_rss_source_name` (`enrichment.py:817-830`) to `rss_reddit_*` → idx 2; Codex Forum (`rss_community_openai_com`) and AINews (`rss_ainews`) land in the generic `rss_` bucket as designed; unmatched sources get all-zero "other" per docstring. One line.
- `team_draft` (`interleave.py`) uses a fresh `rng.permutation` per round, limit-checked, arms that exhaust stop — no shared mutable state; sequential `SharedFeatures` reuse is read-only after fill. One line.
- Challenger miss/fallback detection (`rerank_candidates`: `model_cache in {"hit","miss"}`, `svm_fit` error label → production-only) and one-vote-snapshot training are correct. One line.
- `JoinedLogistic.fit` rejects vote sets missing a class (`len(stories) != len(labels)`), so the SVM dummy-row path never leaks into the joined fit — correct fallback. One line.
- Doc drift noted, not a bug: `ARCHITECTURE.md` says `config.toml` pins `svm_c = 0.1`, but this tree's `config.toml` has `svm_c = 4.0` with blend + precomputed + side embeddings on — confirm which is the trial-intended value before citing ARCH numbers. One line.

### 3. Open questions

- Does enriching `top_comments`/`article_body` after the first embed actually change `text_content` (hence the embedding hash in `story_embedding_text`/`side_text_hash`)? If hydration leaves `text_content` frozen, candidates rank on stale text while the TF-IDF blend may see newer text — needs a DB-backed check, impossible in this read-only copy.
- What was the intended semantics of `neutral_weight = 0.0`: exclude neutrals from the fit, zero their sample weight, or something else? The eval scripts' `lr_target` variants may already answer this — check before wiring.
- Is the `svm_c = 4.0` in this tree's `config.toml` the live trial value (ARCH says 0.1)? Determines which C any offline validation should use as baseline.

