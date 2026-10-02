# HN Rerank status

Review status saved 2026-10-01 03:17 UTC (2026-09-30 local).

## Objective
Improve what the dashboard shows the user (live profile 151; user 1 is the
old profile, stopped 2026-09-24): ranking quality and which sources feed it.
Prior task: items 1/2/4 complete: review fixes committed/pushed and deployed,
live count refresh checked, and corrected blend evaluation rerun. Evidence:
[FINDINGS.md](FINDINGS.md#review-follow-through--2026-09-30-items-1-2-and-4).
Current task: TLDR diagnosis saved; user chose "Save diagnosis only; stop here."
Live replay reproduced prose-format rejection and raw PDF input; archive
exclusion confirmed in source. Fixes are parked; no runtime change or deployment.
Reader design task: Bloomberg-inspired HTML/JS concepts are ready for user
review; implementation in the app awaits the user's selection.
Ranking eval loop (2026-10-02): fresh-vote/impression evals done offline; two
candidate improvements (personalized Popular order, gemma side by side) await
the user's decision. Nothing deployed.

## Verified result
- Ranking evals (2026-10-02, snapshot of the live DB, profile 151, read-only):
  the live blend beats the legacy ranker on 394 unseen votes (AUC 0.795 ->
  0.865, P@12 0.375 -> 0.500). Knob tuning has plateaued (12 blocks: every
  C/LR/TF-IDF/kNN/half-life/source change within ±0.01 AUC). Gemma side by
  side: AUC +0.013 (11/12 blocks, p=0.001), P@12 +0.04, but 4.8 s/story on
  laptop CPU. Within Popular the model's order cuts top-12 downvotes from
  ~54% to 31% (70/30 model/gravity: 40%). Evaluator gained `--holdout-blocks`,
  `--candidate-pool impressions`, feed slices, gravity blend; new
  `scripts/badge_yield_report.py`. Tests 1049 passed / 1 timing flake (passes
  alone); Ruff/format clean; ty only the old untracked TLDR script.
  FINDINGS.md "Fresh-vote, impression-pool and live-yield evals".
- TLDR diagnosis: `49913192` returned completed prose twice (185/206 of 450
  tokens), so the identical retry lost Discussion; `46108780` contains raw
  `%PDF` in its stored article body. Archive HN sources are skipped by the
  dupe pipeline. Read-only replay; no database writes. FINDINGS.md
  "TLDR follow-up diagnosis".
- Review follow-through: four commits pushed (`04d830d`, `16b46ed`,
  `4071aec`, `4d1ff85`); local/VPS HEAD `4d1ff85`, VPS checkout clean,
  service active after 2026-10-01 03:01:48 UTC restart. Latest backend/Chrome
  CI passed. Live scheduled refresh at 03:11:56 UTC updated the same active
  card from 1054/702 to 1063 points/712 comments with an unchanged deck
  version; the original Article section stayed attached and collapsed.
  Backend 1040/18,
  Chrome 2, standalone TUI 162/1 on three OSes; lint/type/build gates pass.
  Corrected evaluator matches live production. P@12 development 0.625 -> 0.708,
  newest 20% 0.667 -> 0.667; served Recommended coverage insufficient.
  No ranking change from that replay. FINDINGS.md "Review follow-through".
- Reader mockup (2026-09-30): [interactive preview](http://127.0.0.1:8766/bloomberg-reader.html)
  in `docs/mockups/`: command palette, saved views, discussion-change briefing,
  linked Article/Discussion/Related tabs, Scan/Focus. Illustrative data only.
  Browser flows and desktop/mobile layouts checked; JS, Ruff and ty passed
  during mockup delivery. Preview service is active and HTTP 200 on this
  status save. Evidence: FINDINGS.md "Bloomberg-inspired reader mockup".
- Review fixes on `fe38d9f`: tightened SQLite/evaluator regressions first,
  then fixed concurrent preservation, web/TUI vote rollback, count/TLDR
  freshness, evaluator Interest parity, UTC dates and interaction bounds.
  Independent oracles strengthened; timing synchronization stabilized;
  resolved expected-failure marks removed. Backend 1040 passed / 18 skipped;
  current TUI snapshot 162 passed / 1 skipped; Chrome 2 passed; Ruff, format,
  ty clean. Six-file backend patch live on `fcbfc9a` after 00:49:21 UTC
  restart (2026-10-01): dashboard 200, cached TLDR 0.08s, uncached 3.14s,
  malformed batch rejects all four events without inserts. No application
  journal errors; slow-embedding warning remains. Uncommitted WIP/database
  history preserved. FINDINGS.md "Review fixes and verification".
- TUI `d304e58` (2026-09-30, local client, no server change): panes under
  30 rows (`COMPACT_HEIGHT`) keep the footer to one row (status beside
  "? help") and drop "Because you upvoted" from the heading. A 64x23 render
  shows a 2-line heading and 1-row footer (summary +4 rows). TUI tests pass
  (one unrelated prefetch timing test failed once under load, 3/3 alone).
- Live `fcbfc9a` (2026-09-30 21:55 UTC, restart): hot threads' points and
  comments stay live between regens (feeds read stored counts; a 10-min
  Firebase refresh of the 30 busiest young HN threads, no LLM;
  `counts_version` makes the TUI refetch; a generated TLDR reply carries
  counts, so `r` shows them). Interest badge icon is 🎯 again. Runs 22:05
  (`probed=20 changed=18 ms=729`) and 22:15 (`20/8, 813 ms`); /api/feed
  showed 49913571 at 667/412, equal to Firebase. Dashboard 200 (0.28 s),
  legend 🎯 Interest, tldr-cache 204 for the outgrown Argon summary,
  uncached TLDR 5.3 s, no journal errors. Tests 1023, TUI 158, browser,
  ruff, ty pass. FINDINGS.md "Hot-thread counts".
- Live `34051df` (2026-09-30 20:33 UTC, restart): 🧭 Interest replaces 🎯
  Similar; impressions log badge kinds in the new
  `interaction_events.badges` column (added on start, verified). After the
  restart: dashboard 200, deck ready in ~20 s, 1d and 1w each serve 5 🤔 /
  5 🧭 / 5 ✨ (1w 🧭: AI buildout, LeanFIRE in Bangkok, Opus 5.5 prompting
  guide, city shape, Live Avatar), cached TLDR 0.4 s, no journal errors.
  Not yet seen live: an impression with badges (no client events since
  16:42; open tabs and the TUI need a reload); uncached TLDR not exercised
  (every deck story cached). Tests 1018, TUI 156, browser, ruff, ty pass.
- Live `2190914` (2026-09-30 16:38 UTC, restart): "Because you upvoted" on
  a Hot/Top/Talk/Unsure/Novel card needs similarity >= 0.85 (was 0.35 for
  all; 49908757 Ubuntu <- RTX 5090 at 0.70). After the restart: dashboard
  200, 1d deck ready in ~20 s with 17/38 attributed, cached TLDR 0.4 s, no
  journal errors; uncached TLDR not exercised (every deck story cached).
  Also in this deploy: `D` (Shift+D) cycles the window back (web and TUI).
  TUI (`9c17ac0`, local reader): feed names instead of x.com, no `▲ 0`/`💬 0`
  for non-HN feeds, "Because you upvoted" in the heading, summary wait
  counter. Tests 1014, TUI 156, ruff, ty pass. FINDINGS.md "'Because you
  upvoted' coherence", "TUI review".
- Review fixes (2026-09-30, `fdda723`..`6cd3739`, FINDINGS.md "Code review —
  2026-09-30"): pointer-thread TLDR keying/backoff/rule, eval double blend
  and dense-model mismatch, blend ramp and cache bounds, AINews regen
  abort, three TUI bugs. Tests, ruff, ty and TUI tests pass. The tighter
  pointer rule still matches all 109 live follows. Live on the VPS since
  `28ddde5` (01:21 UTC 2026-09-30): dashboard 200, cached TLDR 0.4 s,
  uncached 14 s, no journal warnings. Since then hourly regens and 151's
  reranks (3-46 s) run clean; AINews fetched tweets again at 06:27.
- Linear blend live on the VPS since `26b9474` (2026-09-29 17:17 UTC):
  `svm_c = 4.0`, `linear_blend_enabled = true` (0.5 production + 0.2 dense
  LR + 0.3 TF-IDF LR). Rerank after a vote (live `371a2bd`, 21:43-21:48
  UTC, user voting, 11,360 candidates): 5.7-11 s, median ~7.4 s, one 17.9 s;
  earlier the same evening 10-24 s. Fixes: the blend refit warm-starts from
  the previous fit (0.3-1.3 s, was 5-18 s); background article/prewarm
  embedding pauses while a rerank runs; article fetch embeds the stored
  text (it re-embedded ~50 stories under the pool lock each regen, a 49 s
  stall). The first rerank after a restart is still ~54 s (cold caches).
  FINDINGS.md "Rerank latency — 2026-09-29".
  FINDINGS.md "Linear blend live: rank latency".
- Also live (from `f49ff0f`): time-window selector (12h/1d/1w/1m/Archive,
  `d`/`D` cycle; web and TUI reopen on the last window picked), AINews
  per-topic source (`rss_ainews`), 2026-09-26/27 source changes.
- TLDRs (live `ea1403b`, 2026-09-29 20:44 UTC): `LLM_PROVIDER=mistral`
  (`mistral-small-latest`, paid key with a $10 cap), 2-4 s per TLDR; the
  free `gofree` model took p50 45 s and left 60 of 132 failed or half-only.
  Same stories: Mistral is correct but more generic than longcat. Fixes:
  one retry when a reply has no bullets (lost halves), tweet URLs
  summarized via fxtwitter plus the page they link. Pointer threads (one
  short "Comments moved to / [dupe] / Discussion: item?id=N" comment, rule
  `e491c38`) follow the link on tap and prefetch, and skip summaries cached
  before the follow (38507672 served an invented discussion). Backfill
  (`scripts/follow_pointer_threads.py`, 22:10 UTC): 160 of 162 followed and
  embedded. A first, looser rule (live 20:44-22:06) matched 464, mostly real
  threads; it wrote no TLDRs or comments for them (checked). FINDINGS.md
  "TLDR providers and quality".
- ClickHouse source (live `7b1d70a`, 20:57 UTC): live-window query retried
  3 times; comments nested in HN order up to 30 levels (was a flat list,
  so thread-aware selection saw depth 0); live HN caps 5000 -> 10,000.
  First regen: 9,187 candidates, no errors, dashboard 0.23 s.
  FINDINGS.md "ClickHouse source review".
- TUI `a` (2026-10-01): back to Claude Code at the user's request; the
  uncommitted Codex WIP was reverted (committed `claude` default restored) and a
  status-line assertion kept. Focused tests 3 pass, Ruff clean; reader relaunched
  in pane `%58`. Physical `a` press not yet tried. FINDINGS.md "TUI Codex shortcut".
- TUI status line (committed `48e460f`, 2026-09-29): always one row; long messages
  end in `…` instead of wrapping to 3 rows (hints still stack below when
  both don't fit). TUI tests 152 pass, ruff clean.
- TUI headline dividers (committed 2026-09-30): a faint `─` row between
  stories (user: OK), live in the restarted reader; TUI tests 152 pass,
  ruff/ty clean. A zero-row meta-line underline looked link-like; dropped.
  FINDINGS.md "TUI headline dividers".
- Profile merge (live, 2026-09-29 14:01 UTC, user chose "July onward"):
  user 1's 2,332 votes since 2026-07-01 on stories 151 had not voted on
  copied to 151 (467 -> 2,799 votes). Backup
  `main/hn_rewrite.db.pre_merge_1_to_151_20260929T140130Z` on the VPS.
  Offline, merging raised AUC on 151's later votes ~0.84 -> 0.88-0.91 with
  the top 12 unchanged. FINDINGS.md "Merging user 1's votes...".
- Offline ranking (FINDINGS.md 2026-09-29 sections): best is stored +
  embeddinggemma side by side, `prodlr[svm_c=4.0;lr_weight=0.2;
  tfidf_weight=0.3]` (SVM C=4 rank-blended with logreg 0.2 and a word
  TF-IDF logreg 0.3). TF-IDF alone added AUC 0.791 -> 0.801 (7/8 folds) and
  was better or equal on all five unseen-vote checks (user 1's reserved
  newest 20%, 151's votes after 09-26/09-27, alone and merged).
- Linear blend offline case (`f01dbba`, `21f870c`, `pipeline/linear_blend.py`).
  Against the previous live ranker (`svm_c=0.1`, stored embeddings only) the
  full eval gives AUC 0.726 -> 0.790 (p=0.017), P@12 0.625 -> 0.719; the
  newest-20% run is flat (AUC 0.739 -> 0.751, P@12 0.667 both). TF-IDF alone
  on top of svm_c=4 + LR: +0.016 AUC. Codex review: no leakage, but the
  "unseen votes" checks overlap and informed tuning, so the size of the gain
  is unconfirmed. FINDINGS.md "Proposal vs the actual live ranker". The
  dense part of those numbers was an approximation (5 meta columns, not
  live's 10); fixed in `24dc55c`, not rerun.
- TF-IDF sweep (full eval, FINDINGS.md "TF-IDF tuning sweep"): nothing
  beats weight 0.3 beyond noise. Title-only input, source prior 0.1 and the
  joint model are worse at the top 12; char n-grams, TF-IDF C and half-life
  make no difference.
- Rejected on the full eval: MMR (costs upvotes at every threshold), one
  SVM per embedding (AUC up, top 12 down), skipped stories as weak
  negatives, binary up-vs-rest logreg, stacking as implemented (Codex found
  leaks; not a verdict on stacking). Embedding models: gemma best;
  harrier-0.6b ties it only without the instruct prefix.
- Eval fixes: tie-aware AUC and average-rank percentiles; URL-group
  isolation for synthetic training rows. Codex review of the evals:
  `/tmp/hn-eval-local/codex-ml-review.md` (how to ask: `~/AGENT-ACCESS.md`).
- Feed check (2026-09-29): Reddit 429s ~55-63/day (was ~150), every
  configured subreddit fresh except r/ocaml (7 failures); 531 RSS rows
  without article text; r/transit and r/MachineLearning kept (too few
  shown). FINDINGS.md "Feed yield check — 2026-09-29".

## Blocker / limits
- Reader mockup: waiting for design feedback. Its delivery-time backend run
  had 1024 passed / 18 skipped / 10 xfailed / 2 timing failures, including
  both failures on a focused rerun under `batch`. This is a prior validation
  snapshot; concurrent application/test edits have since changed the tree.
- Review fixes: committed/pushed and clean on the VPS. Separate shortcut and
  mockup WIP remains local. Open web tabs need reload for the final client.
  Live Chrome count changes and collapsed-section preservation checked;
  a day-long hot-refresh observation and user-observed TUI checks remain.
  Historical embedding slowness remains; real-user ranking gains unconfirmed.
  No blocker to the completed items 1/2/4; item 3 (TLDR investigation) was
  outside this round and remains outstanding.
- TLDR fixes parked at the user's request (diagnosis only; stop).
  The discussion call fails intermittently (`tldr: discussion call
  failed (status=None), salvaging article-only`: 54 times in 2 days, also
  before `fcbfc9a`); the partial summary is not cached. One live example
  reproduced a completed prose reply twice; other failures remain unclassified.
- Gemma side by side is not live and would need gemma on the VPS.
- Blend gain is unconfirmed (flat on the newest 20%).
- Go limit resets 2026-10-06 16:28 UTC: then set `LLM_PROVIDER=gospark`
  (now `mistral`) in the VPS `shared/.env` and restart. Mistral stops at its
  $10 cap (balance not checked). OpenRouter, Zen and Cerebras have no
  credit; Gemini free is 20 requests/day.
- The merge's live effect needs ~100 new votes from 151 to read.
- Offline gains over live are large on the full eval but flat on the newest 20%.
- `/tmp` is wiped at boot: the eval dir is mirrored to
  `~/.local/state/hn-rerank-eval/hn-eval-local` (last synced 2026-09-29 12:15);
  snapshots and merged copies live in `~/.local/state/hn-rerank-eval/`.

## Next step
- Ranking (user decision pending): (a) re-order Popular's gravity candidates
  by the model (70/30 or model-only), then live-check Popular's up/down rates
  with `scripts/badge_yield_report.py`; (b) gemma side by side is built
  behind `model.side_embedding_enabled` (off; uncommitted, tests pass).
  Deploy needs the user's OK: symlink the VPS HF snapshot to
  `shared/embeddinggemma-300m-onnx`, run `scripts/embed_side_vectors.py`
  niced (16,990 stories, ~1 h) plus a timer, flag on, restart, then check
  rerank time and the `side_embeddings` trace label. Rerun
  `~/.local/state/hn-rerank-eval/run_eval.sh` on a new snapshot after ~200
  more votes.
- Review the reader mockup and select concepts to refine or implement. Keep
  `hn-reader-mockup.service` available for review; stop it when review ends.
  Preserve concurrent review/fix work and keep any app integration scoped
  to the user's chosen concepts.
- Reload any open web tab; the local TUI was reloaded by the shortcut task.
  Live Chrome automatic refresh verified; over a day, check `hot_refresh`
  lines stay under ~1 s with no Firebase warnings. Confirm the TUI physically.
- Parked TLDR follow-up: strengthen bullet-format retry, guard raw PDF inputs,
  and include archive HN sources in bounded dupe checks, with regressions and
  verification if the user resumes. Keep current ranking settings:
  newest-vote P@12 is flat.
- Explore badges (user, 2026-09-30): Unsure stays until ~2026-10-14, then
  keep or drop it (and judge Novel, 🎯 Interest) from the per-badge upvote
  rates in `interaction_events.badges`. Offline, Unsure's votes taught the
  ranker no more than random ones (FINDINGS.md "Do Unsure votes teach...").
- Parked (user, 2026-09-30): title-based matching for "Because you
  upvoted" (full-text similarity is flat; Ubuntu's best upvote 0.70). Plan:
  time encoding ~11k titles on the VPS; title vectors in memory beside the
  candidate pool (or an additive `title_embeddings` table if slow; the
  `embeddings` PK is story_id, so no second row there); attribution = closest
  upvote title at >= ~0.55 for all cards, replacing 0.35/0.85; SVM features,
  Novel and Interest stay full-text. Before/after lines on 1d/1w for review,
  then deploy and check rerank time. Current 0.85 badged cut is eyeballed on
  one deck.
- **Embed-model hill-climb paused** (2026-09-30 15:28 UTC): Qwen3-0.6B small
  test killed by memory pressure (685MB encoder, 11.5GB/11.5GB used); did not
  reach checkpoint at 200 stories. harrier-270m small test: P@12 0.389 vs best
  0.556, AUC 0.708 vs best 0.720 (not promising alone). Both deferred pending
  memory recovery. Do not restart without explicit ask.
- Live blend, first read (2026-09-30, 186 new votes from 151): HN up rate on
  shown flat (10.0% -> 8.9%); the overall drop (13.8% -> 10.4%) is old
  RSS/Archive stories; not significant, no rollback. Re-read after ~200 more
  shown HN stories. Rollback: `linear_blend_enabled = false`, `svm_c = 0.1`
  in `config.toml`, deploy, restart. FINDINGS.md "Live blend, first read".
- Open review items (FINDINGS.md "Code review — 2026-09-30"): AINews
  shared-tweet cards, prose-reply retry for Responses-API providers, tweet
  `internal_exception` backoff. CH quota, pointer-follow race and
  live-window retry fixed 2026-09-30 (`bd2eaae`, live 07:03 UTC): the first
  regen prewarmed 7/45 HN stories (was 5-8/~330 hourly), no warnings. Not
  smoke-tested over HTTP (a cookie-less `GET /` creates a profile).
- Open TLDR gaps: raw PDF stored as article text (46108780 fails);
  archive dupe cards not swapped by dupe resolver (live `hn` only).
- Optional low value: embedding queue (jina-v5-nano, mdbr-leaf-mt, KaLM-mini)
  only if memory pressure resolves and small-test AUC > 0.01 above best.
