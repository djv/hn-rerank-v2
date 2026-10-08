# HN Rerank status

Updated 2026-10-08. Previous handoff (one-classifier research, Why-this-story
preview, pending Reddit/Gemma items):
[before interleaving](docs/status-archive/before-interleaving-20261007.md).

## Objective
Compare shortlist #2 (one joined logistic classifier, all features) and #4
(same, no metadata) live against production through team-draft
interleaving on profile 151's Recommended views for 28 days, plus four
offline follow-ups the user chose (power replay, fit timing,
metadata-scale middle variant, short-history curve).

## Result
- Implemented (off by default): `model.classifier = "joined_logistic"`,
  `interleave_user_ids`, `interleave_decks` table,
  `scripts/interleave_report.py`; challengers warm-start and share
  production's feature build. Design, power, timing and the fixed
  28-day decision rule:
  [INTERLEAVING.md](docs/evaluations/model-ablation-20261007/INTERLEAVING.md).
- Power replay (optimistic): 28 days gives ~70% (#2) / ~54% (#4) power for
  the upvote-rate test; null false-positive rate 1-2%.
- VPS timing (copy of live DB): post-vote rerank 21 s with interleaving vs
  ~10 s production alone.
- Gates: VPS copy 1,168 passed (3 git-dependent eval tests fail there only;
  they pass locally), Ruff clean, ty only the existing inspection-script
  diagnostic.
- Deployed main 8d85747 to the VPS (includes the Why-this-story
  expansion; rollback tag deploy-pre-interleave-20261007 = 3475235).
  Live since T0 = 1791432442.75 (2026-10-08 04:07:22 UTC): every window's
  first 12 Recommended cards split 4/4/4; post-vote reranks 11-20 s;
  smoke tests 200 (dashboard, feed, ranking-ready, cached and uncached
  tldr-detail); no journal errors. The first deploy (7de14af, 03:55 UTC)
  drafted before deduplication and left 3/2/7 splits; its decks are
  excluded. Two diagnostic profiles (213, 214) came from smoke tests.
  First check, 04:15 UTC: 11 votes since 03:58, all with impressions;
  9 predate T0 and are excluded, 2 credited (production up, #2 neutral).
- Offline follow-ups finished (labels reused, exploratory). Metadata
  x0.5/x0.25 fall between #2 and #4; neither beats both, so the arms stay
  #2/#4. In the short-history curve, from 200 votes up both challengers
  match or beat production (#2's AUC gain has block intervals excluding
  zero at N >= 400; #4's do not). Tables:
  [INTERLEAVING.md](docs/evaluations/model-ablation-20261007/INTERLEAVING.md);
  artifacts in followups/.

- 2026-10-08 11:27 UTC check: VPS healthy (8d85747, no journal errors,
  reranks 10-20 s); still 2 credited votes. Why preview service stopped.
- Reddit (RSS ends November 13): only RSS still reaches Reddit; `.json`
  and HTML return 403. Arctic Shift (free, no key) is the only drop-in
  route found: same top posts by score, but scores fill in only after
  ~36 h, and it likely ends with Reddit's public API by March 2027.
  Devvit and the Data API are not usable here. Evidence in FINDINGS.md.
- Subreddits (user choice): dropped transit, expats, eupersonalfinance;
  added mlscaling, accelerate, OpenAI, codex, agi, Aging (21 feeds).
- Restart fallback: the shared cold deck now keeps 512 stories per view
  (was 32), so profile 151 no longer drops to ~5 Popular stories for the
  ~90 s after a restart (WORKLOG 2026-10-08). Deployed 6ae84d2 at 12:33 UTC:
  during the 90 s cold window, 1w and 1m served 16 Popular / 16 Recommended;
  no journal errors; dashboard and tldr-cache 200.
- Arctic Shift adapter merged and deployed off by default
  (`reddit_source = "rss"`; ARCHITECTURE.md 3.4.4). All 21 feeds compared:
  395/470 RSS stories match; 66 of 75 misses are posts under 36 h old.

## Blocker
None.

## Next step
- Final analysis at T0 + 28 days, 2026-11-05 04:07 UTC (Nov 4 23:07 ET):
  `uv run python scripts/interleave_report.py --db hn_rewrite.db
  --user-id 151 --since 1791432442.75 --until 1793851642.75` on the VPS.
  Weekly descriptive reports and safety checks per INTERLEAVING.md.
- Reddit switch on 2026-11-11 (user choice; RSS ends November 13): rerun
  `uv run python scripts/compare_reddit_sources.py` (laptop or VPS), set
  `reddit_source = "arctic_shift"` under `[hn_rewrite]` in config.toml,
  deploy, restart, then check `reddit_refresh_complete` and a Reddit
  card's tldr-detail. Rollback: remove the line and restart.
- Carried over: 1m Popular age-mix
  check, real-terminal tint/footer check, October 21 Gemma 2 future-vote
  recheck (/home/d/TASKS.md).
