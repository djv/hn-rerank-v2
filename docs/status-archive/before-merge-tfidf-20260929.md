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
- AINews per-topic source (`rss_ainews`, 2026-09-29): each `[AINews]` issue in
  `latent.space/feed` becomes one card per Twitter-recap topic, with the linked
  tweets' text (fxtwitter) as discussion; whole-issue cards dropped. Added
  r/singularity, r/ClaudeAI, r/LocalLLM, r/ClaudeCode (25 items each). Reader
  `o` opens the topic's first tweet, `c` the topic in the issue. Live on the
  VPS at `d8643b5` since 04:25 UTC; regen stored 51 cards, TLDR smoked.
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
- Ranker hill-climb (2026-09-28, offline): best on stored embeddings is
  SVM C=2 + logreg rank blend 0.3 (top-12 upvotes 7.50 -> 8.25 of 12, AUC
  0.726 -> 0.769 vs production). Best overall: stored + embeddinggemma-300m
  side by side, C=4 + blend 0.3: top-12 0.688 -> 0.708, AUC 0.769 -> 0.791
  (8/8 folds, p=0.002); fresh votes within noise. Not deployed (needs gemma
  on the VPS). FINDINGS.md "Incremental ranker hill-climb — 2026-09-28".
- Time-window selector (12h/1d/1w/1m/Archive replacing Date and Age,
  `d` cycles it; per-window feed schema v2, client prefetch; Popular by
  HN gravity, no server Explore shuffle): live on the VPS at `fc6461f`
  since 2026-09-28 17:39 UTC. Smoked: every window serves 16/16/15, bad
  window 400, the TUI parses the live feed, TLDR loads, journal clean.

- Embedding probe (2026-09-28, offline): untuned per-embedding probes
  (`scripts/probe_embeddings.py`) put harrier-270m level with gemma and
  stored; metadata alone gets AUC 0.66; up vs neutral is the weak spot.
  harrier-270m f16 NaN = Gemma 3 overflow, fixed by OpenVINO
  `ACTIVATIONS_SCALE_FACTOR` 8. FINDINGS.md "Untuned embedding probe".

## Blocker / limits
- New feeds and the Reddit throttle have under a day of data.
- Only votes after 2026-09-25 are a clean ranking holdout.
- Without the ONNX model (laptop), 18 real-model `test_pipeline` tests skip
  with a setup hint; `uv run python setup_model.py` enables them.

## Next step
- After the 2026-09-28 reboot: `/tmp` is wiped at boot; the eval dir was
  copied to `~/.local/state/hn-rerank-eval/hn-eval-local` (copy it back to
  `/tmp/hn-eval-local`, whose scripts use that path). Then, one GPU job at
  a time: finish harrier-0.6b (`s300-harrier06.partial.npz` resumes;
  batch 2, GPU only, no CPU runs) and its evals, then jina-v5-nano,
  mdbr-leaf-mt, Qwen3-0.6B f16, KaLM-mini-v2.5 (export with
  `--trust-remote-code`, transformers 4.45.2) — commands in
  `queue3.sh`/`try2.sh` there. Give every model two probe rows (alone,
  stored+X) in `probe_embeddings.py`; judge against stored and gemma
  alone, not only the tuned combo.
- Reformulation (user approved to run after the reboot): stacked model =
  out-of-fold content scores (SVM, logreg) + metadata (per-source upvote
  prior, points/comments at fetch time, length, domain, age, Show/Ask HN)
  into sklearn gradient boosting; then an ordinal/pairwise ranking
  objective on the same features; ablations content/meta/both. Same folds
  and metrics (top-12 upvotes, AUC, down/neutral share).
- Check the hill-climb best on votes after 2026-09-25 (fresh read-only VPS
  snapshot) before any ranker change ships.
- 2026-09-28 20:00 local: scheduled task `hn-feed-yield-check` reports
  Reddit 429s, new-feed yield, article-text backlog and whether to keep
  r/transit / r/MachineLearning; act on its recommendation.
- After a few hundred new votes (about a week): rerun
  `eval_ranker_variants.py` on votes after 2026-09-25 to recheck the
  challenger.
