# HN Rerank status

Updated 2026-10-08. Previous handoff (one-classifier research, Why-this-story
preview, pending Reddit/Gemma items):
[before interleaving](docs/status-archive/before-interleaving-20261007.md).

## Objective
Compare shortlist #2 (one joined logistic classifier, all features) and #4
(same, no metadata) live against production through team-draft
interleaving on profile 151's Recommended views for 28 days, plus four
offline follow-ups the user chose (power replay, fit timing,
metadata-scale middle variant, short-history curve). Keep Reddit stories
flowing after Reddit RSS ends on November 13 (Arctic Shift).

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
  All six now have stories (agi and mlscaling via Arctic Shift).
- Restarts: the cold deck keeps 32 x the served view (now 384 per view),
  so profile 151 gets full views in the cold window (6ae84d2). A restart's
  first rerank took 77-101 s (~60 s cold classifier fits): fits now persist
  in `~/.cache/hn-rewrite/warm_start` and the first regen waits for the
  startup warms (c4147e7, deployed 15:33 UTC; regen waited 51 s; fits
  saved). The warm-start gain is measured at the next restart (pending).
- Interleaving: one arm-fit failure (13:15:56 UTC, a vote between arms;
  1 of 24 warms) is fixed: all arms train on one vote snapshot (623321b).
  Views now show 8 stories (served 12, Explore 3 per badge; 37e0333,
  user's choice), so arms draft from their top 24 (INTERLEAVING.md).
- Arctic Shift (ARCHITECTURE.md 3.4.4) is the Reddit source since 13:32
  UTC (user moved the switch up from November 11). Comparison: 395/470 RSS
  stories match; 66 of 75 misses are posts under 36 h old. Codex
  (gpt-6.1-sol) review: 5 findings fixed (f50f4b7). The archive's global
  422 overload cost 1-2 feeds per refresh; a second pass for failed feeds
  (234124f) left 0 failed in the 15:00 refresh (142 requests, 35
  overloaded, 34 min). Prefetched threads carry archive comments.
- Profiles: 20 test/anonymous profiles deleted with the user's approval
  (users 214 -> 194; backup `hn_rewrite.db.pre_test_profiles_20261008T144729Z`
  on the VPS, kept). Only profile 151 votes.
- RSS read timeouts now log one warning line (5c092d0).
- TUI `o`/`c` open via `~/bin/hn-open` (system-setup): a new Chrome window
  when the everyday Chrome runs, else surf.

## Blocker
None.

## Next step
- Measurement restart (automatic, after the Reddit refresh started 15:35
  UTC): deploys 5c092d0 and logs the first rerank with saved warm starts
  (expect ~25-30 s vs 81 s). Record it here.
- User: restart the TUI from a fresh shell (8 per section, `hn-open`).
- Arctic Shift: check `reddit_refresh_arctic` lines over the next days and
  a Reddit card's tldr-detail (live tap) with archive comments. Rollback:
  delete `reddit_source` in config.toml and restart (RSS until Nov 13).
- Final analysis at T0 + 28 days, 2026-11-05 04:07 UTC (Nov 4 23:07 ET):
  `uv run python scripts/interleave_report.py --db hn_rewrite.db
  --user-id 151 --since 1791432442.75 --until 1793851642.75` on the VPS.
  Weekly descriptive reports and safety checks per INTERLEAVING.md.
- Carried over: 1m Popular age-mix check, real-terminal tint/footer check,
  October 21 Gemma 2 future-vote recheck (/home/d/TASKS.md).
