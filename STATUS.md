# HN Rerank status

## Objective
Improve what the dashboard shows user 1: ranking quality (2026-09-25 study)
and, since 2026-09-26, which sources feed it.

## Verified result
- Ranking: production ranking, config and embeddings unchanged after the
  study (challenger tied on hand orderings). FINDINGS.md "Ranking-quality
  study — 2026-09-25".
- Sources (2026-09-26/27, live on the VPS): 20 feeds dropped (low yield
  on user 1's votes, or AI provider blogs: user preference, none to be
  added), 13 added (newsletters, independent blogs, science, r/expats,
  r/eupersonalfinance); Reddit topfeeds refresh at most every 2h; regen
  fetches article text for up to 30 RSS snippet rows per run, backlog
  backfilled; 272 legacy source labels relabelled (eval composite 0.669 ->
  0.675). FINDINGS.md "Source yield review — 2026-09-26".
- Tools: `scripts/source_yield_report.py` (per-source yield),
  `scripts/backfill_rss_articles.py`, `scripts/relabel_legacy_sources.py`.

- Server + web aligned with the terminal client (2026-09-26/27,
  `docs/server-web-alignment-plan.md`, all stages done): one deck decision,
  boot-epoch versions, no-op repeat votes, one summary generation per story,
  no vote-triggered regen, web client built from the embedded feed with the
  TUI's poller/votes/summaries/keys, slimmer ranking-ready/feedback.
  All live on the VPS since 2026-09-27 02:02 UTC (`f7cfbe7`); smoked in
  headless Chrome over the tailnet (read-only) with a clean journal.
- Page runs in headless Chrome in CI (`tests/test_browser.py`); unused CSS
  removed. VPS at `c1c676b` since 2026-09-27 11:22 UTC, smoked clean.
- Ranker hill-climb (2026-09-28, offline, uncommitted tooling): best is
  SVM C=2 + logreg rank blend 0.3 on stored embeddings: top-12 upvotes
  7.50 -> 8.25 of 12 (n.s.), AUC 0.726 -> 0.769 (p=0.034), more discovery
  and non-HN upvotes. Not deployed. FINDINGS.md "Incremental ranker
  hill-climb — 2026-09-28". Laptop iGPU encoding (`--device gpu`) works.
- Time-window selector (12h/1d/1w/1m/Archive replacing Date and Age,
  per-window feed, client prefetch; Popular by HN gravity, no server
  Explore shuffle) built in worktree `../hn-rerank-window` (branch
  `time-window`, uncommitted); suite/TUI/browser/ruff/ty green on 3.12.

## Blocker / limits
- New feeds and the Reddit throttle have under a day of data.
- Only votes after 2026-09-25 are a clean ranking holdout.
- Without the ONNX model (laptop), 18 real-model `test_pipeline` tests skip
  with a setup hint; `uv run python setup_model.py` enables them.

## Next step
- Check the hill-climb best on votes after 2026-09-25 (fresh read-only VPS
  snapshot) before any ranker change ships.
- Review, commit and deploy the time-window branch (feed schema v2: web
  and TUI must update together; live smoke after deploy).
- 2026-09-28 20:00 local: scheduled task `hn-feed-yield-check` reports
  Reddit 429s, new-feed yield, article-text backlog and whether to keep
  r/transit / r/MachineLearning; act on its recommendation.
- After a few hundred new votes (about a week): rerun
  `eval_ranker_variants.py` on votes after 2026-09-25 to recheck the
  challenger.
