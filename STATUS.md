# HN Rerank status

Updated 2026-10-13. Previous handoff (one-classifier research, Why-this-story
preview, pending Reddit/Gemma items):
[before interleaving](docs/status-archive/before-interleaving-20261007.md).

## Goals (user 2026-10-12)
What "working well" means, from the user's answers. Weigh every ranking or
view change against all four aims, not AUC alone:
- Save time: most cards worth reading.
- Discovery: good things the front pages would not surface.
- Stay current: the important developments get shown.
- Calmer habit: a bounded alternative to scrolling HN and Reddit.

The user reads in a mix of quick checks and catch-ups, so every window
matters. The worst failure is a same-y feed (a filter bubble). It ranks above
duds, missed great stories and voting effort. Any change that cuts duds
must not narrow topics or reduce novel up-votes. Audience: the user first,
then other people's profiles; also a public showcase and an ML playground.
A new profile's first feed therefore matters too.

Chosen next goals (user 2026-10-12): discovery (fewer same-y
feeds: Explore's Discover, live 2026-10-12, and Recommended's far card, live
2026-10-12 after the user lifted the Recommended freeze), a reading signal (TUI opens and dwell) and a good
first feed for new profiles (parked by the user 2026-10-12, in `~/TASKS.md`). Serve-time score logging (`deck_scores`
table, live since 2026-10-12) supports all three.

## Objective
Make the feed less same-y without losing hits (Goals above). Discover is
live in Explore. The far card (Recommended's last card) went live
2026-10-12; read both about 2026-11-02.

Compare shortlist #2 (one joined logistic classifier, all features) and #4
(same, no metadata) live against production through team-draft
interleaving on profile 151's Recommended views for 28 days, plus four
offline follow-ups the user chose (power replay, fit timing,
metadata-scale middle variant, short-history curve). Keep Reddit stories
flowing after Reddit RSS ends on November 13 (Arctic Shift).

## Result
- Stopping-rule review completed (user `rv`, 2026-10-13): one independent
  Claude cloud review and agent source verification, saved/pushed as
  `10be4fa`. Recommendation: preserve the original November5 snapshot/report,
  accept an inconclusive result, and define any continuation prospectively.
  Current per-vote test ignores clustering and skipped-card exposure; mutable
  feedback still requires snapshots. Missing far-card-test claim rejected:
  storage and report-exclusion tests already exist. Proposed sample targets
  and sequential methods are unvalidated, not approved for implementation.
  No stopping rule, trial date/arm, runtime or capture job changed.
  Evidence: `docs/evaluations/stopping-rule-review-20261013/ASSESSMENT.md`.
- Single-queue plan 2026-10-13 (user's original idea: one stream, no
  sections): `docs/evaluations/single-queue-20261013/PLAN.md`. Independent
  Claude cloud review said "revise"; the agent verified its code claims. The
  user chose client-first: Step 1 after 2026-11-05 is a client-only Queue
  sort (Recommended -> Explore -> top Popular -> done), with day-alternating
  live comparison; Step 2 is a serve-time interleaved queue only if Step 1
  sticks. Nothing built.
- Reading-signal review 2026-10-13 (FINDINGS "First reading-signal
  review"): Recommended 41% up / 9% down vs Popular 62% down in the user's
  last three TUI sessions. TUI `position` fixed to served rank (583d0bd);
  dwell idle cap 300 s -> 90 s (0e8686e); TUI gate 252 passed / 1 skipped.
  Live after the user restarts `hn`. Dwell separates reads from skips/downs
  (AUC 0.82) but not up from neutral; too small to use yet.
- Second resume check 2026-10-13 ~15:15 UTC: service still b1897f4, no
  warnings in 12 h. Profile151: 235 dwell, 5,333 impression, 2 article_open,
  still zero comments_open. Import AI RSS newest is still issue476 (voted),
  so the gate stays NOT_APPLICABLE until the next issue (~2026-10-19).
  User chose to wait for the November5 endpoint before any continuation
  design (no design work, snapshot or runtime change now).
- Resume check 2026-10-13: service remains active at b1897f4, started
  October12 16:09:54 UTC; recent warning journal empty. Corrected Import AI
  gate (source hash matched current laptop checkout) ran from the preserved
  isolated VPS check tree against the live DB: issue476 remains newest and
  already voted, exit3 NOT_APPLICABLE, not PASS. Profile151 now has 91 dwell,
  5,223 impression and one article_open events; zero comments_open. Latest
  Arctic Shift refresh at10:14 UTC has zero failed feeds. No runtime,
  feedback, trial or research change; next checks remain normal comments
  opening and the next fresh unseen/unvoted publisher issue.
- Ranking-trial report corrections approved after Claude cloud review and
  committed/pushed as `6ed5651` in `scripts/interleave_report.py`: fixed two-challenger alpha,
  downvote harm/safety tests, Newcombe/Wilson intervals, explicit calibration
  exclusions, half-open vote window and pre-T0 deck rejection. Missing decks
  fail explicitly. A read-only rerun of the saved October12 window now has
  60 credits / 15 decks (one pre-T0 production upvote removed), no significant
  arm difference or down-safety alarm. Full evidence and review:
  `docs/evaluations/ranking-trial-review-20261012/`. The final analysis must
  use a frozen endpoint DB; the reporting addendum documents capture and
  exclusion commands. Isolated VPS suite 1,419 passed / 1 skipped, plus 50
  focused checks, lint/type/format clean. No capture timer, runtime or
  trial-arm/date change.
- Import AI Recommended integration gate implemented (user 2026-10-12):
  `scripts/check_import_ai_recommended.py` checks real publisher RSS,
  read-only feedback/seen history and existing-profile served feeds. Fresh
  unseen/unvoted issues must be in the first eight visible Recommended cards
  in age-eligible windows. PASS/FAIL/NOT_APPLICABLE are distinct exits;
  no ranking boost, history edits, timer or service change. Live issue476
  correctly NOT_APPLICABLE after the user's vote. Gate tests plus full
  backend suite passed. Approved Claude cloud review found a confirmed
  medium false-PASS case for HN-first mixed timestamps and a low future-date
  skip; correction plan saved in `docs/evaluations/import-ai-gate-review-20261012/`.
  The user authorized corrections: stored-time windows, explicit future-date
  failure, API profile verification and stronger real-API/CLI regressions.
  Verification results are recorded in FINDINGS.md.
- Real reading delivery verified for the short 2026-10-12 session
  (18:39:50–18:44:01 UTC): 51 dwell events totaling 230.421 s, 47
  impressions and one article_open. Dwell and article-open delivery are
  confirmed; comments_open remains unverified. No synthetic telemetry or
  browser operation by the agent. Owned read-only watch ended.
- Reading-signal prerequisite check 2026-10-12 18:10 UTC: no local reader
  process; `hn` points to the client venv and imports this checkout's
  app.py/dwell.py with telemetry hooks present. Profile 151 still has
  5,141 impressions and zero opens/dwell. Await normal user reading;
  no reader launch, synthetic event or provider request was issued.
- Terminal client CI lint failure reproduced and fixed: dwell clamping
  now uses equivalent min/max expressions. Standalone VPS gate: 251 passed /
  1 skipped, lint/types/format/build and wheel --version clean. Backend gate
  1,376 passed / 1 skipped, lint/types/format clean. Runtime unchanged;
  Pushed 16f4cb3; GitHub backend/browser and Terminal client Ubuntu workflows
  both green, including installed-wheel headless startup. macOS remains
  unverified under the new triggers (no scheduled run found).
- Read-only trial safety check 2026-10-12 16:43 UTC: 61 credited votes
  across 16 decks after excluding 29 calibration re-votes. Production
  7 up / 8 neutral / 8 down; #2 10/8/2; #4 8/3/7. Descriptive only;
  neither safety threshold triggered (last-24h median rank time 19.8 s).
  Latest Arctic Shift refresh has zero failed feeds. Reading events still
  only impressions. `docs/evaluations/live-interleaving-check-20261012/`.
- Warm-start test fix completed 2026-10-12: the test uses tol=1e-6 and
  explicitly replays seed 1155516390, retaining the 5e-3 probability bound.
  This reduces the recorded seed's gap below the assertion bound; production
  tolerance is unchanged. Isolated VPS gates: 1,376 passed / 1 skipped,
  ruff, touched-file formatting and ty clean. The independent review and
  1,000-seed verification are saved in
  `docs/evaluations/live-feed-review-20261012/`. No runtime rollout needed.
- Read-only live check 2026-10-12 16:32 UTC: all five profile-151 feed
  windows ready, 12 Recommended with Discover at slot 12 / trial arm `far`,
  and 3 cards per Explore badge. Dashboard/readiness 200; service running,
  no post-start warnings. Only impressions have arrived (5,141 total);
  opens/dwell and physical rendering remain unverified. FINDINGS "Read-only
  live-feed check". User selected this check over the test fix.
- Blind re-vote calibration (46 votes): the user's re-votes separate the
  original ups at AUC 0.785 vs production 0.778. That headroom (+0.007,
  95% −0.12 to +0.12) is not settled; agreement 59%, old votes flip more.
  Recency weighting gave no gain on the re-votes (+0.004–0.006, CI from 0).
  At the user's request the re-votes went live on profile 151: 17 changed,
  12 June stories added; user 1 and trial credit unaffected.
  Training neutral votes as down gave no gain either. Five weak feeds
  (FIRE trio, r/MachineLearning, Haskell Discourse) dropped and deployed
  2026-10-12 00:23 UTC, with the user's OK to restart mid-trial.
  Serve-time score logging (`deck_scores`, cf9594d) deployed 2026-10-12
  12:06 UTC with the user's OK: VPS gate 1,344 passed / 1 skipped, ruff and
  ty clean; the startup warm wrote profile 151's row (5 windows, no null
  scores); a cold smoke profile wrote none; journal clean.
  Discover (replaces Novel in Explore; FINDINGS "Discover Phase 0"):
  built by Muse (a2caf47), merged (7c362d1) and deployed 2026-10-12 12:31
  UTC with the user's OK, no independent review (user's choice). VPS gate
  1,349 passed / 1 skipped, TUI 219 passed; live feed for profile 151 has
  3 Discover, 3 Unsure and 3 Interest cards in every window; journal clean.
  TUI reading signal (user 2026-10-12): the reader now logs article and
  comments opens and dwell (1 s minimum, 5 min idle cap, ends on focus
  loss), as the web does. Built by Muse, fixed on review, merged 4c3d758;
  VPS gate TUI 242 / backend 1,349 passed; payloads pass the server's
  validator. Quit now always exits within 3 s (other keys blocked, second
  q exits at once, unsaved votes listed after close; user 2026-10-12).
  Independent Muse review findings fixed. Live once the user restarts
  `hn`; no server restart.
  Teach (active learning in place of Explore): Phase 0 simulation showed
  only a tiny gain over random picks; parked by the user.
  Recommended diversity Phase 0 (plan `docs/evaluations/diversity-20261012/`):
  no re-rank rule passed the pre-set gate. Far 1 (last slot to a story far
  from up-votes) just missed it: +0.049 diversity against the 0.05 bar,
  −1.1 points up-rate and twice the novel up-votes. MMR 0.8 was free on
  up-rate but gained only +0.040. Topic caps cost the most up-rate.
  FINDINGS "Diversity Phase 0".
  Far card (user 2026-10-12: "u can make changes to recommended now", chose
  Far 1): Recommended's last served card is now the best-scored story
  unlike the user's up-votes (closest-up similarity < 0.60), with the ✨
  Discover badge. It is tagged arm `far`, so the trial credits it to no arm.
  - Built by Muse (2a5fcfc). The Claude cloud review found no high issues
    and 2 medium ones: ageing out, and profiles with no up-votes. Those and
    all the low ones were fixed (bf80ec8).
  - Merged b1897f4; VPS gate: backend 1,375 passed plus 1 pre-existing
    flaky failure (see Blocker); TUI 251 passed; ruff and ty clean.
  - Restarted 16:09:54 UTC with the user's OK. The first deck has a far card
    at position 12 in all 5 windows for profile 151, the 11 drafted cards
    split 4/4/3, and the trial row records `far`. No other profile got
    interleave rows. Journal clean.
  - Not verified live: the served feed's rendering (badge, last position)
    in the web and TUI.
  FINDINGS "Far card deploy".
  [Evidence](FINDINGS.md#blind-re-vote-calibration-recency-check-re-votes-applied--2026-10-11).
- Muse-primary summaries with Mistral fallback: branch `muse-primary-fallback`
  (ddb424d, pushed, not merged or deployed). A failed primary call falls back once,
  and a circuit breaker keeps a rate-limited primary off. Branch checks:
  backend 1,359 passed / 1 skipped, ruff/format/ty clean; Claude cloud
  review findings verified and fixed. Not tested against the real Go gateway.
- Codex Forum scheduled side-vector follow-up verified at 17:06 UTC: the
  ordinary timer encoded all 83 pending vectors and exited successfully.
  Read-only post-job coverage exceeded 99.99% across all 17 feedback profiles;
  all 29 Codex topics have current primary and side vectors. Production remains
  active at 794caa4 with its original 16:51:50 restart. No extra job or restart.
  [Follow-up evidence](docs/evaluations/codex-forum-review-20261010/DEPLOYMENT.md#scheduled-side-vector-follow-up).
- Approved Codex Forum rollout complete: VPS pinned to 794caa4, restarted
  16:51:50 UTC, active/running. Private online backup passed quick_check/hash;
  source/config/lock and unrelated archive hashes preserved. Normal regen and
  deck advancement completed; dashboard/feed/readiness passed, two Codex cards
  present. Live legacy-cache miss and nondestructive footer normalization
  verified. Side model remained on, post-start coverage above 99.42%; bounded
  journal scan quiet. No forced generation, new profile/vote, extra side job or
  TUI restart. [Deployment evidence](docs/evaluations/codex-forum-review-20261010/DEPLOYMENT.md).
- Codex Forum rollout-plan review completed with Claude cloud. The initial
  plan needed operational corrections; the agent verified and amended it.
  At 16:38, target-code read-only side coverage was 99.7769–99.8374% across
  checked profiles; the later cold-pool sample was 99.5143%, above the 98%
  cutoff. Twenty-eight Codex vectors need new hashes; refresh gates at rollout.
  Backup deadlines, concrete smoke signals, version-advance checks and rollback
  effects are now explicit. No second reviewer, backup, restart or deployment.
  [Plan and assessment](docs/evaluations/codex-forum-review-20261010/ROLLOUT-ASSESSMENT.md).
- Codex Forum medium review finding fixed: flattened-footer matching starts
  at a separate count token and avoids repeated digit/whitespace scanning.
  The new regressions failed before correction (32K runs took about nine
  seconds each; authored `v12` was truncated). Final checks: Codex tests 37
  passed, backend 1,304 / 1 skipped, Ruff/ty/two-file format clean. No rollout
  or additional reviewer. [Regex-fix evidence](docs/evaluations/codex-forum-review-20261010/FOOTER-REGEX-FIX.md).
- CI noise reduced (user 2026-10-11): `eval-decision1` CI was red on `ty`
  (11 unresolved optional imports in the decision1 Strands worker); 8c8358c
  allows `torch`/`strands_decider` via `[tool.ty.analysis]`, run 38143556782
  green. The Terminal client workflow now tests Ubuntu on pushes and
  Ubuntu+macOS on PRs, a weekly Monday run and manual runs; Windows is no
  longer tested (user choice), superseded runs cancel. Post-push runs on
  47ed9ea green; the PR/weekly macOS leg is unverified under the new trigger.
- TUI medium fixes implemented: q/Ctrl+q drain queued votes in order for up to
  five seconds, keeping the reader open on timeout or unconfirmed saving.
  Vote/undo remain disabled through successful exit. HTTP 5xx errors use the
  existing manual-retry path; permanent 4xx content-hide behavior remains.
  The approved Claude cloud review found an exit-guard gap and lost error detail;
  both were reproduced and corrected. Final checks and review bounds:
  [TUI validation](docs/evaluations/tui-medium-fixes-20261011/CHECKS.md).
  Final VPS gates: TUI 219 passed / 1 skipped; backend 1,264 / 1 skipped;
  repository Ruff, ty and five touched-Python format checks passed.
  No personal reader was restarted; the final process check found no local
  reader running. A launch from this checkout loads these client changes.
- Codex Forum source-safety fixes are integrated into current main from Muse's
  preserved isolated worktree. Forum topics never enter article-fetch lanes;
  generated RSS footers are stripped; legacy inputs and keys normalize without
  changing raw stored fields. The two remaining corrections restore LessWrong
  tap behavior and resolve test typing. Final integrated-source gates: 51
  focused passes, backend 1,301 passed / 1 skipped, Ruff/ty/eight-file format
  and whitespace checks clean. [Integration checks](docs/evaluations/codex-forum-review-20261010/CHECKS.md).
  No production checkout update, service restart or live generation.
- The fix plan and Muse assignment are saved beside the review evidence:
  [fix plan](docs/evaluations/codex-forum-review-20261010/FIX-PLAN.md) and
  [handoff](docs/evaluations/codex-forum-review-20261010/MUSE-HANDOFF.md).
  Muse acknowledged the task and wrote source changes in the isolated VPS
  worktree. Its incomplete validation was superseded by the supervising
  integration checks above; the original uncommitted branch is preserved.
  [Dispatch record](docs/evaluations/codex-forum-review-20261010/HANDOFF-RUN.md).
- Codex Forum result review: conditional rejection of clean stale summaries
  is rated low and parked at the user's request; unchanged-input outages still
  serve cached summaries. [Decision note](docs/evaluations/codex-forum-review-20261010/PARKED-FALLBACK.md).
  The unrelated LessWrong tap change and two introduced test type errors were
  corrected after the user chose Fix and integrate following the approved
  remaining-fixes plan review. No production rollout occurred.
- Codex Forum review complete: the actual deployed extractor returned reply
  text as article_body (three non-opening posts matched). The source can enter
  generic article prewarm and short-post tap fetching, so discussion can be
  framed as article text and cached without reply freshness. RSS count footers
  also enter self_text; no inaccurate generated summary was exercised.
  Public JSON access and partial-topic responses verified. Independent report
  and evidence: [review validation](docs/evaluations/codex-forum-review-20261010/VALIDATION.md).
  No runtime change, production mutation or additional reviewer.
- Codex Forum added to config.toml with its readable label in pipeline/render.py.
  Both hosts fetched HTTP 200; the production parser returned 25 valid topics
  with opening-post text. After the 2026-10-11 03:26:55 UTC restart, the normal
  refresh stored all 25 in the VPS DB. Existing-profile dashboard/feed/readiness
  GETs returned 200. VPS checks: 26 focused; backend 1,264 passed / 1 skipped;
  Ruff/ty/touched-file formatting clean. Evidence: FINDINGS.md, 2026-10-10.
- Explicit r now automatically reloads observed discussion growth once per
  count; background growth updates counts silently. TUI 203 / 1 skipped and
  backend 1,264 / 1 skipped passed in that task. The local reader was restarted
  under its prior authorization; physical r/live generation remains unverified.
- Authorized rollout complete: VPS runtime ad94386, service active; tested
  source manifest and unchanged config hash verified. Online DB backup passed
  quick_check before the nullable snapshot-column migration. Real HN stats are
  live; unsupported-source provenance and auth checks pass. Cached legacy text
  correctly has no fabricated snapshot; bounded journal scan had no matching
  errors. [Deployment evidence](docs/evaluations/section-refresh-20261009/DEPLOYMENT.md).
- Seven review fixes complete: malformed strict hydration rejected before writes,
  repeated-r intent consumed, stale feed replies rejected after vote/undo ack,
  profile stats state reset, warm replies carry source snapshots, and both
  misleading notices corrected. Queued sort refresh drains after a stale reply.
  Final isolated VPS gates: backend 1,264 passed / 1 skipped; TUI 202 passed /
  1 skipped. All executable-source/test/config/template hashes match this
  checkout. Evidence: [seven-fix checks](docs/evaluations/section-refresh-20261009/SEVEN-FIXES-CHECKS.md).
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
  Codex and Claude Code. The current TUI already handled 503; this task extends
  that protection to all 5xx and drains pending votes before quitting. Minute
  polling delays pending decks and the initial counts baseline can miss an update;
  those lower-priority findings remain outside the authorized fixes.
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
- TUI `o`/`c` open via `~/bin/hn-open` (system-setup): Brave Origin Beta
  for articles and comments (user 2026-10-09), user-confirmed working.
- Wide TUI reading pane now keeps the intended 1:2 list/reader ratio
  without a 100-column cap. Automated region checks pass at widths 100
  and 240; physical terminal rendering remains unverified in this save.
- Empty article responses: `_extract_article_body` now skips blank and
  whitespace-only input, preserving retryable `empty_extraction` without
  trafilatura's error log. Isolated VPS gates: 1,217 passed, 1 skipped;
  Ruff, touched-file format, and ty clean. Deployed 2026-10-09; scope above.

## Blocker
Final trial reproducibility requires an endpoint snapshot before later
vote edits/undo; capture is documented but not scheduled. Reporting code
and verification are complete. No trial winner is established.
Evidence-based continuation remains a reviewed proposal, not an authorized
design: outcome/exposure unit, useful effect, dependence treatment,
stopping/cap and harm/diversity guardrails are unresolved.
None for the far card or warm-start test. The recorded seed's failure is
fixed with tighter test-only convergence; full backend gates pass.
The Muse-fallback deploy waits for the trial to end (no restart before
2026-11-05 without the user's OK). No Codex rollout blocker remains. The TUI medium result review is complete
and its introduced findings corrected.
Codex Forum source-safety rollout is complete at approved target 794caa4.
The next ordinary side-vector refresh passed the read-only follow-up. Fresh
Codex generation and physical TUI behavior remain unverified. No reply/count
adapter is authorized. The conditional fallback is parked, not a blocker by itself.
No test profiles/votes or explicit generation request were issued; ordinary
startup provider warming was authorized.
ranking research remains paused and production trial configuration unchanged.

## Next step
- Stopping design: user chose (2026-10-13) to wait for the November5
  result before deciding. After it, if the user chooses continuation, define a separate
  prospective stage and its useful-improvement/uncertainty target, exposure
  and attribution rules, dependence treatment, stopping/cap and safeguards.
  Preserve the original November5 snapshot/report and do not pool repeated
  ordinary p-value checks into a winner claim. No winner is required at the
  original endpoint. Review examples are not validated sample sizing.
  See `docs/evaluations/stopping-rule-review-20261013/ASSESSMENT.md`.
- Import AI gate: run against the next fresh unseen/unvoted issue. Issue476
  is already voted, so the current live check cannot establish an unseen PASS.
  Use the actual service DB; no history reset or new timer.
- Far card: read about 2026-11-02 with Discover (PLAN.md "Far card":
  far-card up- and down-rate vs Recommended cards 9–11, using served
  positions from `deck_scores` cards joined on (dashboard_version,
  story_id), since TUI impression positions before 583d0bd are list indexes; novel up-votes per week; Discover plus the far
  card combined). Phase 0b (Combo, Far 1) runs 2026-11-05
  (`fu-20261012-104919-066`).
- Dwell vs votes: rerun the FINDINGS 2026-10-13 check about 2026-10-27
  (long-dwell neutrals as soft ups; drop 90 s-cap segments).
- Reading signal: confirm `comments_open` arrives in normal profile151 use
  (dwell and article_open are verified); after about two
  weeks, check how opens and dwell relate to votes before using them.
- Discover: read the result about 2026-11-02 (PLAN.md Phase 2: Discover
  up-rate, novel up-votes per week, Explore down-rate), next to the trial
  report.
- After 2026-11-05 and Diversity Phase 0b: build single-queue Step 1
  (PLAN.md above), then start its day-alternating test.
- After 2026-11-05: replay rank caps and score bars on `deck_scores`
  (FINDINGS "Short-window Recommended: score bar check").
- After the trial ends (2026-11-05): merge `muse-primary-fallback`, set
  `LLM_PROVIDER=gospark` and `LLM_FALLBACK_PROVIDER=mistral` in the VPS
  shared `.env`, restart with the user's OK, smoke-test a tap and watch Go
  usage. The breaker honors Go's long Retry-After until a restart.
- Fresh-vote evals: exclude profile 151's 29 re-vote rows
  (`updated_at` 1791763112.57–1791763180.12).
- Optional: about 77 more calibration votes (laptop `~/revote/calibrate.py`)
  would narrow the headroom to ±0.07.
- CI: watch the first weekly/PR Terminal client run (macOS leg). Windows-only
  flake `test_filter_change_starts_at_first_story[60]` is left unfixed since
  Windows is no longer tested.
- Codex Forum deployed at 794caa4: observe normal reader use. The ordinary
  side-timer refresh passed; sampled side coverage is above 99.99%.
  Fresh Codex generation and physical TUI rendering remain unverified; do not
  force them or reopen the parked fallback without a separately scoped request.
  No cleanup or reply/count adapter.
- Observe Codex Forum cards in normal ranked views. Forum replies and live
  discussion counts would require a separately scoped Discourse adapter.
- AI ranking research: goal is a repeatable Muse advantage transferable into
  the ML ranker. Research paused at the user's request (2026-10-09).
  User stopped the contrastive protocol/infrastructure branch.
  [Existing-evidence reassessment](docs/evaluations/RANKING-REASSESSMENT-20261009.md)
  completed with cheap Muse and root verification: no concrete missing feature
  justified; prior-UP explanations occur in fixes and damage, while related
  similarity/lexical features already exist. No new annotations or fits.
  No further research work or annotation spending while paused. Live ML
  interleaving continues unchanged; no winner established from 61 credits
  after excluding the calibration re-votes (2026-10-12 descriptive check).
  This does not rule out every LLM approach. Resume only when requested.
- Optional: the remaining cold cost after a restart is linear blend
  scoring (7.1 s, candidate word counts recomputed); a persisted or
  prewarmed count cache would cut it.
- User: restart the TUI from a fresh shell (8 per section, `hn-open`).
- Load the TUI fixes on a fresh reader launch and physically check quit/save
  and manual summary retry. Automated tests cover these paths; no live vote or
  provider request was made. Lower-priority polling/counts work remains unassigned.
- Section/r/stats feature and seven fixes saved and deployed. Observe the
  running TUI's s/h/l and r/growth-popup controls; automated Pilot tests pass,
  but physical interaction and forced live generation were not exercised.
  Reviews, regression evidence and deployment bounds are in
  docs/evaluations/section-refresh-20261009/.
- Blank-body guard deployed; observe natural empty-response behavior.
  Substack's previously checked httpx 403 reproduced; urllib fallback returned
  full HTML. The historical empty-response cause remains unconfirmed.
- Arctic Shift: check `reddit_refresh_arctic` lines over the next days and
  a Reddit card's tldr-detail (live tap) with archive comments. Rollback:
  delete `reddit_source` in config.toml and restart (RSS until Nov 13).
- Final analysis at T0 + 28 days, 2026-11-05 04:07 UTC (Nov 4 23:07 ET):
  freeze a SQLite-consistent endpoint copy before further voting; analyze
  that copy with the recorded calibration exclusion. Exact commands and
  capture limits are in INTERLEAVING.md's reporting addendum. No snapshot
  job is scheduled by this change. Weekly descriptive reports and separate
  latency safety checks continue; the report now provides down-safety flags.
- Carried over: 1m Popular age-mix check, real-terminal tint/footer check,
  October 21 Gemma 2 future-vote recheck (/home/d/TASKS.md).
