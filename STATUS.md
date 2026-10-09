# HN Rerank status

Updated 2026-10-09. Previous handoff (one-classifier research, Why-this-story
preview, pending Reddit/Gemma items):
[before interleaving](docs/status-archive/before-interleaving-20261007.md).

## Objective
Current maintenance task: implement user-selected background section refresh
on s/h/l, selected-summary-only r with a live stats check, and a story-specific
regeneration choice when comments exceed a known summary snapshot. Codex plan
review completed; verified corrections are incorporated in
[PLAN.md](docs/evaluations/section-refresh-20261009/PLAN.md). Implementation and
isolated validation are complete and saved under the user’s ss request.
Independent result review was offered and remains unapproved; none was launched.
Feature deployment, restart and live smoke remain separately authorized.

Compare shortlist #2 (one joined logistic classifier, all features) and #4
(same, no metadata) live against production through team-draft
interleaving on profile 151's Recommended views for 28 days, plus four
offline follow-ups the user chose (power replay, fit timing,
metadata-scale middle variant, short-history curve). Keep Reddit stories
flowing after Reddit RSS ends on November 13 (Arctic Shift).

## Result
- Section refresh/r/stats feature implemented and saved. Final matching
  VPS source copy passed backend 1,238 / 1 skipped and TUI 195 / 1 skipped;
  Ruff, ty, 17 touched-file format checks and diff whitespace check clean.
  Generation snapshots are stored with cached text; legacy rows stay unknown.
  Forced callers share one fresh follow-up, including callers waking after it
  completed. No provider action, live DB migration or feature deployment.
  [Implementation checks](docs/evaluations/section-refresh-20261009/IMPLEMENTATION-CHECKS.txt)
  include source hashes, skipped checks and the physical/live verification gap.
- Section-refresh comparison completed with cheap Muse and agent verification;
  eight VPS Pilot checks passed. Fetch-on-return used one current-window GET
  per tested round trip; background departure refresh used two. Both applied
  fresh counts/orders and kept the old deck usable during held responses.
  Live TUI buffer checks confirmed help/zoom/back and wide/narrow layouts.
  At that comparison stage the user chose report-first; production changes came
  later under the approved implementation choices. Evidence/limitations:
  [comparison assessment](docs/evaluations/section-refresh-20261009/ASSESSMENT.md).
- 2026-10-09 general server/TUI review completed sequentially with Muse,
  Codex and Claude Code. Confirmed TUI concerns: retryable 500/503 hides
  stories, quit cancels pending votes, minute polling delays pending decks,
  and initial counts baseline can miss an update. No broad fixes made;
  [review synthesis](docs/evaluations/general-review-20261009/SYNTHESIS.md)
  distinguishes confirmed behavior from hypotheses and physical UI gaps.
- Blank-body guard deployed at 17:58 UTC to VPS `bf6e307` only. Service
  active, config hash unchanged, dashboard/feed/readiness/cache smoke and
  simulated deployed-source guard passed; bounded journal scan quiet.
  Uncached live generation and natural empty fetch not exercised.
  [Deployment evidence](docs/evaluations/extraction-deploy-review-20261009/DEPLOYMENT.md).
- Save/cleanup verification, 2026-10-09: isolated VPS checkout of laptop
  HEAD plus current runtime changes passed backend 1,218 / 1 skipped and
  TUI 174 / 1 skipped. Repository Ruff/ty and touched-Python format clean;
  mockup JavaScript syntax passed. No deployment or live trial change.
  WIP includes uncapped wide TUI pane proportions, blank-response guard,
  read-only TLDR diagnostic (provider replay opt-in), and illustrative
  reader mockup. Host-local kernel log retained and ignored.
- 2026-10-09 17:25 UTC live check: service active; 77 profile-151 votes
  since T0, 19 credited across 7 decks (51 Popular / 7 Explore excluded).
  Production 4 UP / 3 neutral / 0 DOWN; #2 3/3/0; #4 3/2/1.
  222 impressions; no other profile's votes or interactions since T0.
  Median rank time 15.1 s over 139 rows (max 104.0 s). Descriptive only:
  too few credited votes for an efficacy conclusion. Trial unchanged;
  final analysis remains November 5 04:07:22 UTC. Evidence:
  [live check](docs/evaluations/live-interleaving-check-20261009.json).
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
  startup warms (c4147e7). Measured at the 16:04 UTC restart (5c092d0):
  first rerank 30.3 s (was 81-101 s); challenger fits 0.7 s each (were
  ~20-28 s), linear blend fit 3.5 s (11-12.5 s). Linear blend scoring
  still takes 7.1 s cold (empty word-count cache).
- Interleaving: one arm-fit failure (13:15:56 UTC, a vote between arms;
  1 of 24 warms) is fixed: all arms train on one vote snapshot (623321b).
  Views now show 8 stories (served 12, Explore 3 per badge; 37e0333,
  user's choice), so arms draft from their top 24 (INTERLEAVING.md).
- Arctic Shift (ARCHITECTURE.md 3.4.4) is the Reddit source since 13:32
  UTC (user moved the switch up from November 11). Comparison: 395/470 RSS
  stories match; 66 of 75 misses are posts under 36 h old. Codex
  (gpt-6.1-sol) review: 5 findings fixed (f50f4b7). The archive's global
  422 overload cost 1-2 feeds per refresh; a second pass for failed feeds
  (234124f) left 0 failed in the 15:00 and 15:36 refreshes (142/127
  requests, 35/28 overloaded, 34/28 min). Prefetched threads carry archive
  comments.
- Profiles: 20 test/anonymous profiles deleted with the user's approval
  (users 214 -> 194; backup `hn_rewrite.db.pre_test_profiles_20261008T144729Z`
  on the VPS, kept). Only profile 151 votes.
- RSS read timeouts now log one warning line (5c092d0).
- TUI `o`/`c` open via `~/bin/hn-open` (system-setup): a new Firefox window
  for every link (user 2026-10-08; Firefox has Bypass Paywalls Clean).
- Wide TUI reading pane now keeps the intended 1:2 list/reader ratio
  without a 100-column cap. Automated region checks pass at widths 100
  and 240; physical terminal rendering remains unverified in this save.
- Empty article responses: `_extract_article_body` now skips blank and
  whitespace-only input, preserving retryable `empty_extraction` without
  trafilatura's error log. Isolated VPS gates: 1,217 passed, 1 skipped;
  Ruff, touched-file format, and ty clean. Deployed 2026-10-09; scope above.

## Blocker
No code/check blocker. Independent result review and feature deployment await
authorization; live feature behavior remains unverified.

## Next step
- AI ranking research: goal is a repeatable Muse advantage transferable into
  the ML ranker. Research paused at the user's request (2026-10-09).
  User stopped the contrastive protocol/infrastructure branch.
  [Existing-evidence reassessment](docs/evaluations/RANKING-REASSESSMENT-20261009.md)
  completed with cheap Muse and root verification: no concrete missing feature
  justified; prior-UP explanations occur in fixes and damage, while related
  similarity/lexical features already exist. No new annotations or fits.
  No further research work or annotation spending while paused. Live ML
  interleaving continues unchanged; no winner established from 19 credits.
  This does not rule out every LLM approach. Resume only when requested.
- Optional: the remaining cold cost after a restart is linear blend
  scoring (7.1 s, candidate word counts recomputed); a persisted or
  prewarmed count cache would cut it.
- User: restart the TUI from a fresh shell (8 per section, `hn-open`).
- If requested, address the prioritized TUI review findings, starting with
  retryable summary errors and pending-vote quit behavior; physically check
  narrow/wide layout and controls. No broader implementation authorized yet.
- If approved, run the offered independent result review of the completed
  section/r/stats change, focusing on cache/API and concurrency behavior.
  Then separately authorize deployment/restart and live feature smoke.
  Source-checked plan corrections and validation evidence are saved in
  docs/evaluations/section-refresh-20261009/.
- Blank-body guard deployed; observe natural empty-response behavior.
  Substack's previously checked httpx 403 reproduced; urllib fallback returned
  full HTML. The historical empty-response cause remains unconfirmed.
- Arctic Shift: check `reddit_refresh_arctic` lines over the next days and
  a Reddit card's tldr-detail (live tap) with archive comments. Rollback:
  delete `reddit_source` in config.toml and restart (RSS until Nov 13).
- Final analysis at T0 + 28 days, 2026-11-05 04:07 UTC (Nov 4 23:07 ET):
  `uv run python scripts/interleave_report.py --db hn_rewrite.db
  --user-id 151 --since 1791432442.75 --until 1793851642.75` on the VPS.
  Weekly descriptive reports and safety checks per INTERLEAVING.md.
- Carried over: 1m Popular age-mix check, real-terminal tint/footer check,
  October 21 Gemma 2 future-vote recheck (/home/d/TASKS.md).
