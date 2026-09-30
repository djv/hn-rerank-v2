# HN Rerank status

## Objective
Improve what the dashboard shows the user (live profile 151; user 1 is the
old profile, stopped 2026-09-24): ranking quality and which sources feed it.

## Verified result
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
- TUI `a` (committed `48e460f`, tests pass): Claude Code in a tmux pane split beside
  the reader with the article/comments links and a dig-deeper prompt.
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
- The TLDR discussion call fails intermittently (`tldr: discussion call
  failed (status=None), salvaging article-only`: 54 times in 2 days, also
  before `fcbfc9a`); the partial summary is not cached. Cause not checked.
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
- Restart the local TUI (`hn-rerank`; started before `fcbfc9a`) and confirm
  live: counts refresh without `r` after a `hot_refresh changed>0`, `r`
  shows new counts, and the compact layout in a short pane. Over a day,
  check `hot_refresh` lines stay under ~1 s with no Firebase warnings.
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
- Rerun the blend eval with fixed `prodlr` (live dense model, 10 SVM columns)
  against `prod[svm_c=0.1;linear_blend_enabled=false]`, full and newest-20%.
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
