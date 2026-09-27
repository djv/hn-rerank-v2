# HN Rerank status

## Objective
Improve what the dashboard shows user 1: ranking quality (2026-09-25 study)
and, since 2026-09-26, which sources feed it.

## Verified result
- Ranking: production ranking, config and embeddings unchanged after the
  study (challenger tied on hand orderings). FINDINGS.md "Ranking-quality
  study — 2026-09-25".
- Sources (2026-09-26/27, live on the VPS): 12 low-yield feeds dropped,
  8 added (AI newsletters, The Register, r/expats, r/eupersonalfinance);
  Reddit topfeeds refresh at most every 2h; regen fetches article text for
  up to 30 RSS snippet rows per run, backlog backfilled; 272 legacy source
  labels relabelled (eval composite 0.669 -> 0.675). FINDINGS.md "Source
  yield review — 2026-09-26".
- Tools: `scripts/source_yield_report.py` (per-source yield),
  `scripts/backfill_rss_articles.py`, `scripts/relabel_legacy_sources.py`.

## Blocker / limits
- New feeds and the Reddit throttle have under a day of data.
- Only votes after 2026-09-25 are a clean ranking holdout.
- Without the ONNX model (laptop), 18 real-model `test_pipeline` tests skip
  with a setup hint; `uv run python setup_model.py` enables them.

## Next step
- 2026-09-28 20:00 local: scheduled task `hn-feed-yield-check` reports
  Reddit 429s, new-feed yield and article-text backlog; act on its
  recommendation.
- After a few hundred new votes (about a week): rerun
  `eval_ranker_variants.py` on votes after 2026-09-25 to recheck the
  challenger.
