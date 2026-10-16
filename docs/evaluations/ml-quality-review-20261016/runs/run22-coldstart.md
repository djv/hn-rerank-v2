# Short history, new profiles, freshness review

## 1. Top findings / proposals (ranked by value per effort)

**P1. Heavy voters lose the entire popularity prior — add a small freshness/importance residual. (defect)**
Evidence: `pipeline/ranking.py:1783` (`alpha_2 = n/50`) → for n≥50 `t1_weight = 0`, so profile 151's Recommended is pure taste (`t3_weight=1`, `ranking.py:1788-1806`); the SVM feature set has no age/score/velocity column (`ranking.py:_svm_personalization_features` meta = textlen + sims + cluster + source one-hot; `engagement_features_enabled=false`, `publication_affinity_enabled=false` defaults in `pipeline/config.py`, unset in `config.toml`). Stay-current then depends only on candidate windows + the Popular view. A major event the user would want regardless of taste can only surface via taste similarity.
Effect: stay-current ↑, save-time ↑ (fewer stale-taste misses); sameness neutral-to-↓ if implemented as 1 reserved "important" slot rather than reweighting everything. Effort M. Validation: time-split `scripts/eval_ranker_variants.py` (production vs +gravity-residual variants, `gravity_weight`/`gravity_scale` options already exist) on NDCG@12 + top-12 novel-up rate + Import-AI gate (`scripts/check_import_ai_recommended.py`) as the freshness probe. Confidence: medium-high on the defect, medium on the fix shape. Note: raw engagement *features* were tried and rejected (ARCHITECTURE.md §3.2 — inflated archive metrics, worse 30-day held-out), so this proposes an inference-time residual, not retraining with score/age columns.

**P2. A new profile's first feed is a pure HN-frontpage clone with no Explore. (defect)**
Evidence: `pipeline/__init__.py:157-163` (`cold_ranked_candidates` = `hn_gravity` only; RSS rows have score≈0 so they sort bottom), `pipeline/__init__.py:186-187` (cold deck has no Explore), `pipeline/__init__.py:1035-1043` (0-vote short-circuit). First impression = same-y HN gravity, exactly the failure the Goals rank above duds; discovery and calmer-habit both suffer (nothing but the front page the user tries to avoid).
Effect: discovery ↑↑ for new profiles, sameness ↓; zero effect on profile 151. Effort S (e.g. guarantee K non-HN / recent-RSS slots per cold-deck window, or score-0 fallback to recency within the RSS leg). Validation: cold-deck deck eval (`build_cold_deck` path in `eval_ranker_variants.py`) reporting source mix + age distribution + simulated first-N-vote yield; needs a new small-profile/first-feed metric (see P5). Confidence: high on defect, medium on slot count. Differs from removed `PRIMARY_*` quotas (ARCHITECTURE.md §3.4): cold-deck-only, small, temporary until votes arrive.

**P3. Tier-2 (the only personalization for votes 1–19) is a single unnormalized centroid-diff. (defect / untried improvement)**
Evidence: `pipeline/ranking.py:1738-1763` — means of L2 vectors are never renormalized (magnitude shrinks with vote variance), one global centroid collapses multi-interest users (even though `interest_cluster_k=10` exists for Interest picks, `pipeline/config.py:70-72`), neutrals ignored, no count shrinkage. Below the dual gate (`min_up/down=20`, `pipeline/config.py:75-76`) this is *all* the personalization there is, and the 20/80 ramp keeps it dominant for a long time.
Effect: save-time ↑ for thin profiles; sameness ↓ if replaced by per-interest-max or top-k-mean-diff (multi-interest preserved) vs single centroid (averages tastes into mush). Effort S (candidate `cand_sim_to_up` top-k means are already computed in the SVM branch). Validation: short-history replays (`--train-recent`, `scripts/eval_ranker_variants.py:2470-2474`) at N=10–100 with tier-2 variants, metrics AUC + top-12 + topic-spread/novel-up proxy. Confidence: medium. Uncertain: whether shrinkage or kNN form wins — eval decides.

**P4. The dual gate + slow ramps strand lopsided voters in tier-2. (untried improvement)**
Evidence: gate needs 20 up AND 20 down; `alpha_3=(n_min−20)/60` (`ranking.py:1791-1800`) → full SVM only at 80/80; short-history curve (`docs/evaluations/model-ablation-20261007/INTERLEAVING.md:91+`) shows all arms coincide at N=100 (~22 ups) and challengers separate only at N≥200–400. A 50-up/5-down user is pure centroid forever (ARCHITECTURE.md §3.3). Given thousands of main-profile votes but tiny other profiles, this is the binding constraint for everyone except profile 151.
Effect: faster personalization (save-time ↑); risk: earlier SVM = narrower feed — must ship paired with Discover/far-card guardrails and watch novel-up rate. Effort S (config-gated: asymmetric gate, e.g. allow fit with ≥10 minority-class votes, and/or narrower windows). Validation: same short-history replays + per-arm `model_cache hit/miss` trace labels; require no topic-narrowing (P5 metric). Confidence: medium-low on net win; cheap to test, easy to reject.

**P5. No true new-user cold-start eval exists. (evaluation gap — do first)**
Evidence: the short-history curve trains on the N *most recent* of profile 151's votes as a stand-in for a new user's first votes (`INTERLEAVING.md:91+` caveat, `followups/launcher-source.txt:91-92`); `--train-recent` has no first-N mode and no multi-profile or topic-spread reporting. P2–P4 cannot be judged on AUC alone per the Goals.
Effect: enables all above; no direct feed change. Effort S: add `--train-first-N` + per-small-profile replays + a topic-spread/novel-up column alongside AUC/top-12. Confidence: high that the gap exists.

## 2. Checked — fine or already tried

- Vote-recency weighting tried, null: half-life 30/60d +0.004–0.006, CI from 0 (FINDINGS.md "Blind re-vote calibration 2026-10-11"); live timestamps still unused by design (ROADMAP.md:305-311). Park; revisit only via P5 replays.
- Publication-affinity prior tried, dead (NDCG→0; FINDINGS.md 2026-09-22); engagement columns rejected per §3.2 — hence P1's residual form, not features.
- TF-IDF/char-gram/C/half-life sweeps null beyond current blend (FINDINGS.md "TF-IDF tuning"); joined #2/#4 already in live trial to 2026-11-05 — no production change before then.
- Short-history curve result stands: challengers match/beat production from N≥200 (#2 intervals exclude zero at 400/800/1600); nothing below N≈100 separates — consistent with P3/P4 targeting tiers 1–2, not the SVM.
- Warm-start/shared-features/one-snapshot correctness, L2 no-scale rule (scaler touches meta from `emb_dim:` on), cold-deck 32× reserve, Arctic ~36h score lag and CH lag documented — all as designed.
- `neutral_weight=0.0` with neutrals still a modeled class: read but not fully traced; see open Q2.

## 3. Open questions

1. What is the target first-feed metric — which mix of HN/RSS, age spread, and up-rate within the first session counts as "good" for a stranger?
2. Does `neutral_weight=0.0` effectively drop neutrals from the fit while tier-2 ignores them entirely — is the neutral signal underused in short history?
3. Would an importance residual duplicate Popular, or does Recommended placement earn distinct reads (check `deck_scores` + dwell before building)?
4. For P4, which topic-spread guardrail (Discover share, far-card novel-ups, cluster coverage) is the ship/no-ship criterion?

