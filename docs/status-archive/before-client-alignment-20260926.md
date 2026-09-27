# HN Rerank status

## Objective
Improve ranking quality for user 1: establish baselines, hill-climb
offline, and check the best candidate against the user's own judgement.

## Verified result
- Offline (8 dev folds, composite): production 0.669; SVM C=4/γ=0.05 +
  logreg rank blend 0.714; + bge-base+mxbai 512-token embeddings 0.754.
  Newest-vote confirmation block too noisy at the top to separate them.
- Hand orderings, 8 batches (`scripts/calibrate_rankings.py`): production
  agrees on 43/80 pairs, challenger 37/80 — a tie.
- Decision: production ranking, config and embeddings unchanged. Study
  tooling committed; `engagement_features_enabled` is opt-in, default off.
- Details: FINDINGS.md "Ranking-quality study — 2026-09-25".

## Blocker / limits
- The newest 20% of votes has been looked at three times; only votes after
  2026-09-25 are a clean holdout.
- Without the ONNX model (laptop), 18 real-model `test_pipeline` tests skip
  with a setup hint; `uv run python setup_model.py` enables them.

## Next step
None required. Optional: after a few hundred new votes, rerun
`eval_ranker_variants.py` on votes after 2026-09-25 to recheck the
challenger.
- Done 2026-09-26: server RSS ~1.1 GB -> ~0.68 GB (WORKLOG). If it grows
  again, the probe method is there: phase RSS before/after `malloc_trim`.
