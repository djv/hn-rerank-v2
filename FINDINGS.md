# HN Rerank findings

## Fresh-vote, impression-pool and live-yield evals — 2026-10-02

Snapshot `~/.local/state/hn-rerank-eval/snapshot-20261002.db` (read-only
`.backup` of the VPS DB, 04:18 UTC). Profile 151: 3,239 votes (755 up),
merged user-1 votes keep their original timestamps. Since the blend went
live (2026-09-29 17:17 UTC, `--holdout-after 1790702220`): 394 votes, 57 up,
233 down, 104 neutral; none informed any tuning. Runs, logs and
`compare.py` in `~/.local/state/hn-rerank-eval/{fresh,impressions}-20261002/`.

New evaluator options: `--holdout-blocks N` splits the held-out votes into
N time blocks, each trained on every earlier vote; `--candidate-pool
impressions` adds every story first shown to the user during a block (and
not voted elsewhere) to that block's votes, so shown-but-unvoted cards count
as not-upvoted. New metrics: `known_upvote_fraction_at_12` (P@12) and
`auc_up_vs_all` (upvotes vs every other card, judged or not).

Fresh votes, 4 blocks (15/18/11/13 ups, ~98 votes each), judged-only pool:

| ranker | P@12 | down@12 | AUC up vs rest | P@12 blocks w/t/l vs live |
|---|---|---|---|---|
| live blend (C=4, LR 0.2, TF-IDF 0.3) | 0.500 | 0.104 | 0.865 | - |
| legacy (C=0.1, no blend) | 0.375 | 0.188 | 0.795 | 1/0/3 |
| C=4, no blend | 0.542 | 0.083 | 0.861 | 2/2/0 |
| blend, TF-IDF 0 | 0.562 | 0.083 | 0.864 | 3/1/0 |
| blend, TF-IDF 0.5 | 0.500 | 0.104 | 0.856 | 1/2/1 |
| up-minus-down (`produd`, C=4) | 0.479 | 0.167 | 0.837 | 1/1/2 |
| + shown-unvoted as down (30%) | 0.500 | 0.125 | 0.863 | 1/2/1 |
| + shown-unvoted as neutral | 0.521 | 0.104 | 0.869 | 1/3/0 |
| + vote half-life 30 d | 0.521 | 0.104 | 0.866 | 1/3/0 |

Impression pool (same blocks + 25-50 shown-unvoted stories each): live
P@12 0.396, AUC-all 0.838; legacy 0.312 / 0.778; C=4 no blend and TF-IDF 0
both 0.458 / 0.835-0.838 (3/0/1); source prior 0.1 0.396 / 0.838, 0.2
0.333 / 0.833; shown-unvoted as neutral 0.396 / 0.844.

- The live blend beats the legacy ranker on unseen votes in both pools
  (AUC +0.06-0.07, P@12 +0.08-0.13). The development-fold gain holds up.
- Nothing tested beats the blend beyond noise: P@12 differences are 1-3
  cards out of 48. TF-IDF adds nothing on fresh votes (0 >= 0.3 > 0.5);
  user 1's development folds favoured 0.3 by about the same small margin.
- Up-minus-down scoring and a source prior make the top 12 worse.

Development folds (151's votes before the 2026-09-27 confirmation cut, 8
folds, ~160 judged votes and 25-58 ups each) pooled with the 4 fresh blocks,
paired deltas vs live (`combine.py`, 12 blocks):

| change | dP@12 (w/t/l) | dAUC (blocks better, Wilcoxon p) |
|---|---|---|
| dense LR 0.2 -> 0.4 | +0.014 (2/10/0) | +0.003 (9/12, p=0.010) |
| dense LR 0.2 -> 0 | -0.007 (1/9/2) | -0.003 (1/12, p=0.001) |
| TF-IDF 0.3 -> 0 | +0.028 (4/8/0) | -0.005 (3/12, p=0.18) |
| C=4 without blend | +0.021 (3/9/0) | -0.009 (2/12, p=0.06) |
| C 2 or 8 | 0 (0/12/0) | 0 |
| shown-unvoted as neutral | 0 (2/8/2) | +0.001 |
| kNN 0.2 (fresh only) | +0.042 (2/2/0) | +0.003 |

Knob tuning has plateaued: every change is within ±0.01 AUC and 1-3
top-12 cards. Only "more dense LR" is consistent, and too small to ship alone.

embeddinggemma-300m side by side with the stored vectors (same settings as
the 2026-09-28 study; 544 of 151's 3,239 voted stories newly encoded on the
iGPU at 0.49 s/story, the rest reused; `~/.local/state/hn-rerank-eval/
gemma-20261002/`), live blend settings, paired with the stored-only replay
(which reproduces the DB-embedding run exactly):

| blocks | P@12 stored -> +gemma | AUC stored -> +gemma | AUC blocks better |
|---|---|---|---|
| dev, 8 folds | 0.615 -> 0.635 | 0.779 -> 0.794 | 8/8 (p=0.008) |
| fresh, 4 blocks | 0.500 -> 0.583 | 0.860 -> 0.870 | 3/4 |
| pooled 12 | +0.042 (7/3/2) | +0.013 | 11/12 (p=0.001) |

Gemma is the only change in this round that holds on both development and
unseen votes (dense LR 0.4 adds nothing on top of it).

Cost and shorter text (same 12 blocks; CPU = laptop i7-10510U, onnxruntime,
2 threads, 24 stories; iGPU = OpenVINO f16):

| gemma input | CPU s/story | iGPU s/story | dP@12 (w/t/l) | dAUC (better, p) |
|---|---|---|---|---|
| 512 tokens | 4.8 (int8 4.4, q4 5.2) | 0.49 | +0.042 (7/3/2) | +0.013 (11/12, 0.001) |
| 256 tokens | 2.3 | 0.23 | +0.042 (7/3/2) | +0.009 (10/12, 0.021) |
| 128 tokens | 1.06 | 0.11 | +0.056 (7/4/1) | +0.011 (9/12, 0.027) |

Int8/q4 weights do not speed up this AVX2-only CPU (no VNNI; the VPS's EPYC
Rome has none either); input length is what costs. At 128 tokens the gain is
the same within noise. Measured on the VPS (2026-10-02 12:04 UTC; venv
onnxruntime 1.27, f32, 2 threads, nice 19, 48 of 151's stories): 0.218 s/story
at 128 tokens, 5x the laptop CPU; vectors match the laptop iGPU encodes
(cosine 1.0000). So ~40 min one-time for the ~11k-candidate pool and ~1.3
min/day for ~350 new stories; load rose 1.2 -> 2.0 during the run. The model
(1.2 GB) stays in the VPS's `~/.cache/huggingface`; the temp dir was removed.
Rollout pieces: an additive gemma vector table (the `embeddings` primary key
is story_id), a niced background encoder, and the ranker fed each story's
stored and gemma vectors concatenated, each part scaled 1/sqrt(2) (what the
replay does).

Per feed (impression pool; slices by the feed of each story's first
impression, `raw_feed_*`), live model vs the feed's own order:

| feed (cards, ups, downs over 4 blocks) | base up / down rate | live top-12 up / down | AUC up vs all / vs down |
|---|---|---|---|
| Popular (175, 8, 95) | 4.6% / 54% | 14.6% / 31% (gravity order 8.3% / 50%) | 0.875 / 0.921 |
| Explore (158, 8, 75) | 5.1% / 48% | 14.6% / 29% | 0.913 / 0.937 |
| Recommended (193, 41, 59) | 21% / 31% | 41.7% / 4.2% | 0.688 / 0.811 |

The model separates what 151 likes inside Popular and Explore far better
than their own orders do. Popular is unpersonalized HN gravity, and most
of its votes are downvotes. Ordering or filtering it with the model would
roughly halve its top-12 downvotes. Few ups per block (0-5), so the up
rates are rough; the downvote result rests on 95 downvotes.

Popular blended with its own order (`gravity_weight`, window clock 8 h,
aged at each block's median vote), top 12 of the Popular slice:

| order | up | down | AUC up vs all / vs down |
|---|---|---|---|
| gravity only (Popular today; 4 h clock alike) | 8.3% | 54% | 0.72 / 0.69 |
| 30% gravity + 70% live blend | 14.6% | 40% | 0.887 / 0.940 |
| 50% gravity | 12.5% | 38% | 0.858 / 0.900 |
| 70% gravity | 10.4% | 40% | 0.841 / 0.866 |
| live blend only | 14.6% | 31% | 0.875 / 0.921 |

Re-ordering Popular's own candidates (the top HN stories by gravity) with
30% gravity / 70% model keeps it a popularity view and cuts its top-12
downvotes from about half to 40%; model-only order cuts them to 31%.

Live yield since 2026-09-30, profile 151, first impression per story
(`scripts/badge_yield_report.py`): Recommended 15.4% upvoted (22/143, 82%
non-HN), Popular 2.4% (3/123, 72% of votes down), Explore 1.2% (1/85). Badges:
Hot 2.5% (79), Interest 8.7% (23), Unsure 0/19, Novel 0/17. Recommended
downvotes cluster on Reddit personal-finance/nomad subs and LessWrong;
since the blend Reddit votes are 10 up / 42 down, HN 17 / 137, other 30 / 54.

## TLDR follow-up diagnosis — 2026-09-30

User selected investigation of discussion failures, PDF handling and archive
duplicates after the review follow-through, then chose "Save diagnosis only;
stop here." Diagnosis is saved; fixes are parked. No runtime fix or deployment
was applied in this follow-up.

- Reproduced the live Mistral discussion failure on `49913192` (EDG C++
  front-end goes public) through the read-only diagnostic
  `scripts/inspect_tldr_failures.py`. Both discussion samples finished with
  `finish_reason=stop`, using 185/206 of 450 output tokens, but answered as
  prose. The identical-prompt retry did not correct the missing bullets;
  generation salvaged the Article half without caching it. This example is
  a formatting rejection, not a timeout or token-limit truncation.
- Replayed `46108780` (DeepSeek-V3.2 paper): `ch_seed`, 15,000 stored article
  characters beginning `%PDF-1.5`. Both article attempts explained that the
  supplied binary data contained no readable article; the Discussion half
  succeeded and was returned uncached. Current fetch eligibility excludes
  `.pdf`, but generation/cache key construction trust pre-existing article
  bodies. The urllib 403 fallback also loses Content-Type information.
- Archive duplicates are excluded in the regen callback, worker submission,
  due-candidate query, canonical replacement and title-feedback matching:
  each checks only source `hn`, while the shared `is_hn_source` contract
  includes `ch_seed` and `bq_seed`. Existing tests cover live HN and non-HN
  bypass, not the archive path. Any fix should retain the existing 250-item
  batch, low-comment filter, cache/backoff and title/target validation.
- The diagnostic uses SQLite `mode=ro`; it does not instantiate `Database`,
  generate embeddings, record usage into SQLite or cache its replay results.
  Database rows and existing WIP remain unchanged.
- If resumed: make the format retry explicitly require Markdown bullets,
  reject raw PDF content at fetch/prompt/cache-key boundaries, and extend the
  existing bounded duplicate path to both archive sources. Add applicable
  regressions before fixes and complete the project verification protocol.
  One live discussion failure was explained; other intermittent failures
  have not been exhaustively classified. PDF text extraction is not supported.

## Review follow-through — 2026-09-30 (items 1, 2 and 4)

User selected commit/deployment, actual reader freshness checks and corrected
ranking evaluation. The separate Codex shortcut and reader mockup remain WIP.

- Committed/pushed the 28 review-only files as `04d830d`. Inventoried the VPS's
  six-file deployed patch, retained it in a full WIP stash, and fast-forwarded
  the clean checkout. All six file hashes matched before/after reconciliation.
  No database maintenance, pruning, schema change or destructive operation.
- Standalone CI passed all 162 TUI tests on Linux/macOS/Windows, then exposed
  seven style diagnostics from unpinned Ruff 0.16 (local lock: 0.15.17).
  `16b46ed` fixes only import order and string parentheses. Backend and all
  standalone test/lint/type/build/wheel smoke jobs then passed.
- Live Chrome: scheduled hot refresh at 02:53:15 UTC (2026-10-01) probed 20,
  changed 7, took 864ms. Normal polling updated Gemini 4 Argon (`49913571`)
  from 1038 points / 695 comments to 1050 / 702 with the same deck version
  `1790822585638` and active card; selected/neighbor feeds all had fresh counts.
  This check also found a gap in the original regression: the outer summary
  survived, but its child details nodes were replaced by `showSummary`.
- Tightened Chrome coverage to close Article and require the same details node
  and closed state after count-only polling. It failed before the fix.
  `4071aec` skips rendering unchanged cached text/provisional status, clearing
  that memo on loading/errors. The same test verifies that a changed generated
  summary still replaces nodes. Backend 1040 passed / 18 skipped (73.78s),
  Chrome 2 passed (12.77s), Ruff/format/ty clean; backend CI passed.
- Clean VPS after `4071aec`; restart 03:01:48 UTC, active, Result=success.
  Deployed template SHA matches local. Live Article stayed collapsed through
  ordinary polling from deck `1790823710351` to `1790823710352` and subsequent
  polls. Final retained-cookie smoke: dashboard 200 / 0.01s, feed 27 stories,
  cached TLDR 200 / <0.01s, forced generation 200 / 4.66s / 1054 points and
  702 comments, invalid batch rejected=4 / inserted=0. An earlier miss-to-hit
  prefetch race invalidated the smoke helper's `cached=false` assumption;
  forcing one known thread made the generation check deterministic.
  Final minute: no application errors/tracebacks. The previously recorded
  intermittent article-only discussion failure remains outside items 1/2/4.
- Final native Chrome count-only proof: scheduled hot refresh at 03:11:56 UTC
  probed 19, changed 13, took 741ms. Passive polling kept deck version
  `1790823710352` while counts revision reached 15. Gemini 4 Argon updated
  from 1054 points / 702 comments to 1063 / 712; 1w/1m/1d responses agreed.
  The original active card and Article details node remained attached, with
  Article still collapsed. No browser reload or forced poll during this check.
- Status save at 03:17 UTC (2026-10-01): local and clean VPS HEAD `4d1ff85`,
  service active, PID 919091, Result=success, same restart timestamp. Latest
  [backend/Chrome CI run](https://github.com/djv/hn-rerank-v2/actions/runs/36809284021)
  completed successfully. Reader mockup service active; preview HTTP 200.
  This status save changes documentation only; unrelated WIP remains intact.

Corrected evaluator replay: frozen user-1 2026-09-25 database, all 5189 stored
384-d embeddings, no new encoder; same evaluation time as the earlier replay.
Development uses the oldest 80% (8 folds); confirmation uses newest 20% (4
blocks). Compare `prod[svm_c=0.1;linear_blend_enabled=false]` against
`prodlr[svm_c=4.0;lr_weight=0.2;tfidf_weight=0.3]`. Live `production` and the
corrected blend have exactly matching per-fold raw metrics in both runs.

| Replay | P@12 legacy → blend | Pooled AUC legacy → blend |
|---|---|---|
| Development, 8 folds | 0.625 → 0.708 | 0.726 → 0.786 |
| Newest 20%, 4 blocks | 0.667 → 0.667 | 0.739 → 0.753 |

Newest-block mean AUC delta is +0.0138, descriptive paired interval
[-0.0173, +0.0449]; top-12 delta is zero. Training windows overlap and these
newest votes have already informed tuning, so this is reused historical
evidence, not independent confirmation. The judged-only pool explains the
high NDCG and cannot establish live HN superiority. Recommended 1w has zero
judged cards in all development folds and 0/0/0/16 in confirmation; no served
deck gain is established. No ranking settings changed from this replay.

Durable artifacts: `~/.local/state/hn-rerank-eval/review-20261001/`
(`run.py`, `eval.toml`, full/confirmation JSON and logs, `compare.py`,
`comparison.json`). Both database backups hash to
`75ae7f3371d7b60de4593540d6ca57c366126c1e848a3ed52c1d76018da38b93`;
reports record `16b46ed`, whose evaluator is unchanged from `04d830d`.
Evaluator SHA: `de84d8f492b651b2f25f719504389ad6bd9bd9de2b24b154108971078f808712`.

## Review fixes and verification — 2026-09-30

Implemented all eight defects from the Review handoff on top of `fe38d9f`.
Preserved existing review WIP, original worktrees, `kernel.errors.txt`, the
reader mockup and the concurrent Codex shortcut change. No commit/push,
database maintenance, schema migration, or destructive database operation.

- Tightened the two regressions first. An unprotected SQLite snapshot requires
  the fresh writer to commit before stale release; a protected snapshot requires
  a verified fresh attempt. Timeout is a setup failure. Evaluator parity now
  compares complete ordered Recommended IDs in every window and keeps the
  Interest assertion; the production control requires 31 Recommended entries.
- `upsert_story` reserves the SQLite writer with `BEGIN IMMEDIATE` before its
  preservation read, making merge/write atomic across pooled connections.
  Existing merge rules remain intact; the real temporary-SQLite race passes.
- Web/TUI capture a per-story revision when queuing each optimistic action.
  Older failed votes or undos cannot reverse newer actions. Generated web
  sequences check an independent acknowledged-write model and exact request
  order; event-gated TUI tests cover both vote and undo failures.
- The live-comments probe compares against `comment_count_at_fetch`, including
  when stored metadata already records the growth. Generated relationships
  exercise tap and regen helpers without mocking away the faulty filter.
- Web refresh patches the active card's header while keeping its summary DOM.
  Count-only polls invalidate neighboring caches and obsolete in-flight replies.
  Generated TLDR counts patch cards and cached feeds. TUI count changes also
  invalidate/re-prefetch neighbors. Changed count revisions are consumed only
  after successful refreshes so failed reads can retry.
- Evaluator deck assembly always receives fold-local training upvote embeddings,
  including with cached similarities. Interest/dedup parity matches serving;
  no held-out labels or raw-embedding scaling were introduced. Real-user metric
  gains were not remeasured.
- RSS/Atom dates use `calendar.timegm`. Generated dates verify host timezone
  independence. Interaction events enforce signed-64 story IDs, nonnegative
  signed-64 versions/positions and finite timestamp conversions; generated mixed
  batches retain valid neighbors even when another integer overflows `float()`.
- Strengthened limiter decisions/bucket consumption with an independent model,
  exact retained token prefixes and realistic CH `kids`/attached-child checks.
  Controlled clocks/events replace timing windows for Reddit spread, TUI
  impressions and feedback debounce. The navigation-prefetch test drains actual
  background work before its assertions. Browser refresh awaits the promise;
  Chrome and rendered-feed contracts verify count/summary preservation.

Final verification:

- Backend: `batch uv run --no-sync pytest tests/ -n 4 -q` — **1040 passed,
  18 skipped, 144.26s**. The preceding run exposed the existing wall-clock
  debounce failure; the controlled-clock replacement passes. Resolved regression
  `xfail` marks were removed.
- TUI: **162 passed, 1 skipped, 48.03s** on an isolated current `src`/`tests`
  snapshot in `/tmp/hn-review-MCJ0r0/tui` on the VPS. Used the existing project
  venv with `uv run --no-sync pytest tests -n 8 -q`, explicit snapshot
  `PYTHONPATH`, idle priority, a two-CPU quota and 4GB memory cap; no environment
  sync or deployed-client replacement. Checksums match the final local files.
  Earlier laptop gates exposed the now-corrected neighbor-fetch expectation and
  a known three-second prefetch timeout under concurrent sweeps. Each corrected
  test also passed alone. The concurrent shortcut task separately reports a
  green laptop TUI/backend run in its section below.
- Chrome: `batch uv run --no-sync --group browser pytest tests/test_browser.py
  -m browser -q` — **2 passed, 12.03s**. Throwaway server and mocked network/LLM.
- Ruff, formatting of all 24 touched Python files, ty and diff whitespace pass.

Live verification:

- Inspected a clean VPS checkout at `fcbfc9a`, checked/applied only the six-file
  backend/evaluator patch, and restarted `hn_rewrite.service` at
  **2026-10-01 00:49:21 UTC**. Final state: active, `Result=success`, PID 886763.
  All six deployed file hashes match the verified local sources. The deployed
  patch remains uncommitted; no unrelated remote files changed.
- One retained cookie session: dashboard **200 / 0.16s**, 1w feed **200 / 28
  stories**, readiness with valid count revision **0**; cached TLDR
  `-1569532955` **200 / 0.08s**; uncached TLDR `49915082` **200 / 3.14s**,
  `cached=false`, complete/non-stale, **105 points / 29 comments**.
- Live oversized story/version/position/timestamp events returned **200,
  rejected=4, inserted=0**. Valid-neighbor persistence is covered against real
  temporary SQLite in the generated regression; no production feedback votes
  were submitted for smoke testing.
- Bounded last-minute journal scan plus the one-minute smoke window: no
  application errors/tracebacks. Expected malformed-event rejection warning;
  regeneration also logged `embedding_slow` (13 texts, 20.4s), still a performance
  limit. The first smoke helper used the wrong readiness query name (400);
  corrected to `min_version` before the successful full smoke.
- Current local reader PID 2414349 started at 20:32:25 EDT, after the final
  client source edit at 20:29:56. Its reload/render was verified by the concurrent
  shortcut task. Open web tabs need a page reload to load the new inline script;
  passive count updates on the user's physical display were not observed here.

## TUI shortcut back to Claude Code — 2026-10-01

- User asked to switch `a` back from Codex to Claude Code. The Codex change was
  never committed, so `app.py`, `README.md` and `test_client.py` were restored
  from HEAD (default `claude`, label "Ask Claude"); kept the new
  `Opened Claude on:` status assertion. Focused tests 3 passed; Ruff clean.
- Reader in pane `%58` (`HN_RERANK_BROWSER=surf-tall`, no `HN_RERANK_AGENT`)
  was restarted with the venv `hn-rerank` and renders the feed. Not verified:
  an actual `a` press opening a Claude pane.

## TUI Codex shortcut — 2026-09-30

- User requested that `a` open the selected article in Codex with the default
  model. `open_agent_session` now defaults to `codex`, passing the existing
  article/discussion prompt as its only argument; no model override. Updated
  the key label, help, status messages, README and existing behavioral tests.
  The custom `HN_RERANK_AGENT` command and clipboard fallback are preserved.
- Installed `codex --help` confirms an optional initial prompt and model flag.
  Running reader had no `HN_RERANK_AGENT` override and Codex on its PATH.
  Reloaded only reader pane `%21`; new PID 2414349 rendered the feed and summary.
  No backend service restart was needed for this client-only change.
- Checks: focused launch/key-flow tests 3 passed; full TUI 162 passed / 1 skipped
  (204.12 s); full backend 1,040 passed / 18 skipped (116.54 s); Ruff, touched
  Python formatting and ty clean; CLI help boot passes. All suites used `batch`.
  Initial backend run: 1,034 passed / 18 skipped / 6 timing failures (427 s);
  all six passed alone. The successful full rerun set `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` to reduce contention.
- Shortcut execution is covered by tests; verification did not start an actual
  Codex conversation. Unrelated review/fix WIP and accumulated DB are preserved.

## Bloomberg-inspired reader mockup — 2026-09-30

User requested Bloomberg Terminal UI research for HN, then interactive HTML/JS
mockups to review. Delivered `docs/mockups/bloomberg-reader.html`, README and
`preview.jpg`; prototype is self-contained, with illustrative stories/counts,
no production API requests or database access. It follows the existing reader's
charcoal/ivory/orange styling and keyboard-first interaction.

- Concepts: command/search palette; named sort/window/layout views;
  "What changed" discussion briefing; linked Article/Discussion/Related tabs
  with return navigation; desktop Scan and expanded Focus layouts.
- Browser verification during delivery: palette search/Enter, custom-view
  persistence after reload, mark discussion changes read, related-story
  return with exact tab/scroll restoration, Focus/Scan, vote/undo, reset.
  Desktop inspected at 1440x900 CSS pixels and mobile at 390x844; browser
  console was clear. Screenshot: `docs/mockups/preview.jpg`.
- Custom views/theme persist in localStorage; demo votes/read state are
  session-local. Reset restores sample data. No app integration is deployed.
- Preview: <http://127.0.0.1:8766/bloomberg-reader.html>, served by transient
  loopback-only `hn-reader-mockup.service`. On this `ss` save, service status
  was `active` and the page returned HTTP 200. It is retained for review,
  not enabled at boot. Stop with `systemctl --user stop hn-reader-mockup.service`.
- Delivery checks: inline JavaScript syntax, Ruff, ty and diff whitespace
  passed. Backend suite under `batch`: 1024 passed, 18 skipped, 10 xfailed,
  2 failed in 152.16s (`/tmp/hn-reader-mockup-pytest.log`). Failures:
  `test_enqueue_spread_distributes_evenly` and
  `test_feedback_idle_threshold_queues_latest_warm`; both failed again alone
  under `batch`. Those wall-clock tests were outside this prototype change.
  Concurrent application/test edits are now present, so this historical run
  does not establish the current tree's test status. No test sweep rerun for
  this documentation-only save.

Research used Bloomberg's official navigation/autocomplete guide, Launchpad
workspace article and news activity/briefing page; source links are in
`docs/mockups/README.md`. The existing TUI already has split panes and keyboard
navigation; the new concepts add domain commands, saved views and linked reading.
Next: user reviews and chooses concepts; refine or integrate only the requested
scope while preserving unrelated WIP.

## Review handoff — 2026-09-30

Historical starting plan; completed implementation and verification are above.

Work in `/home/d/code/hn-rerank`. Read `/home/d/AGENTS.md`,
`/home/d/AGENT-ACCESS.md`, project `AGENTS.md`, and `STATUS.md` first.
This handoff prepares the next implementation pass; no production fixes,
commit, push, production database access, or deployment occurred in this review.

### Starting state and preservation

- Last inspected: `main` at `fe38d9f`. Recheck the live tree before editing.
- Review WIP is the three modified docs (`STATUS.md`, `FINDINGS.md`,
  `WORKLOG.md`) and seven untracked test files listed in the next section.
  Preserve unrelated `kernel.errors.txt` and `docs/mockups/`.
- The `hn-rerank-review-{clients,ranking,server}` worktrees still contain
  untracked original review tests. The integrated versions are in the main
  checkout; inventory before any cleanup. Leave `hn-rerank-window` alone.
- Use in-memory or temporary test databases. Preserve all accumulated
  databases and feedback. Do not run maintenance or migrations as part of
  these fixes. Keep raw embeddings unscaled and add no unnecessary deps.

### Implementation order

1. **Repair the two new regression tests first.**
   `tests/test_database_concurrency.py:127` ignores the result of
   `fresh_done.wait(1)` and resumes the stale writer. An in-process scheduling
   check delayed the fresh writer until the stale writer finished: the test
   body passed on unchanged buggy production code. With its current strict
   mark this is a false XPASS, not proof of a fix. Establish and verify the
   interleaving explicitly; if the fix serializes writers, deliberately
   adapt the coordination rather than accepting an unexplained timeout.
   `tests/test_eval_deck_parity.py:199` only checks that story 1 is absent.
   Replacing the evaluator result with a production deck whose Recommended
   views were emptied passed both new evaluator assertions; production has
   31 Recommended entries in this fixture. Compare complete ordered IDs
   against `_production_deck`, retaining the Interest/badge check.
2. **Prevent SQLite lost updates** in `database.py:572` (`upsert_story`).
   Make the preservation read/merge/write atomic across pooled writers so
   routine ingestion cannot overwrite newly hydrated content and counts.
3. **Fix optimistic vote/undo rollback** in web `submit`
   (`templates/index.html:1517`) and TUI `Reader.submit`
   (`clients/tui/src/hn_rerank/app.py:1786`). A failed earlier request must
   not undo a later same-story action or duplicate history. Preserve request
   ordering and test both vote and undo failures.
4. **Fix the three count/summary freshness defects.** The tap probe must
   compare live comments with the fetched marker even when the stored count
   already reflects growth (`pipeline/__init__.py:545`, `server.py:2645`).
   Web refresh must patch the preserved active card and observe count-only
   changes (`showFeed`, `pollFeedVersion`). TUI count refresh must invalidate
   or update prefetched windows as well as the selected window.
5. **Restore evaluator Interest parity** in
   `scripts/eval_ranker_variants.py:328` (`_recommended`): provide fold-local
   upvote embeddings even when similarities are cached. Preserve train/test
   isolation. The reproduction proves deck divergence, not inflated metrics
   or a measured effect on real user rankings.
6. **Parse feed dates as UTC**, independent of host timezone; replace local
   conversions in `pipeline/enrichment.py:686,690` with UTC conversion.
7. **Validate interaction integer bounds at the HTTP boundary**
   (`server.py:2412`, `_parse_interaction_event`). Reject malformed individual
   events while retaining valid neighbors; check all persisted integer fields,
   including signed story IDs. `2**63` currently yields HTTP 500 and loses
   the valid event in the same batch. Keep tests meaningful and structured.
8. **Strengthen existing tests:** independent per-key/global limiter decisions
   and bucket consumption; exact retained chunk prefix; realistic CH `kids`
   responses and attached child IDs. Stabilize the Reddit queue and TUI
   impression tests using controlled clocks/events. Correct the browser
   synchronization that awaits `reloadInFlight` instead of its `.promise`.

### Validation and completion

- New tests currently use `xfail(strict=True, raises=<specific mismatch>)`.
  Run affected tests with `--runxfail` to inspect the original assertion;
  remove each mark with its fix. Do not broadly xfail new failures.
- Latest focused rerun: backend 3 passed / 10 xfailed (8.76 s); TUI 2 xfailed
  (12.70 s). The mutation/scheduling experiments above were in-process only;
  no production or test source was changed by the follow-up review.
- Earlier full runs remain the latest full-suite evidence: backend
  1025 passed / 18 skipped / 9 xfailed / 1 existing queue timing failure
  (before the extra integer property); TUI 158 passed / 1 skipped / 2 xfailed /
  1 existing impression timing failure. Both failing tests passed alone.
  Ruff, changed-file format, and ty passed then. These are not clean full gates.
- Run affected files after each logical edit, then sequentially:
  `batch uv run --no-sync pytest tests/ -n 4` from the root;
  `batch uv run --no-sync pytest tests -n 8` from `clients/tui`;
  root browser tests via
  `batch uv run --no-sync --group browser pytest tests/test_browser.py -m browser`.
  Browser dependencies must be installed for `--no-sync` to work.
  Run `uv run --no-sync ruff check .`, changed-file
  `uv run --no-sync ruff format --check <files>`,
  `uv run --no-sync ty check`, and `git diff --check`.
- Check for other heavy jobs before starting a sweep; run one at a time.
  Update template contracts in `tests/test_server.py` when changing the web UI.
  Update these docs with actual results and remaining limits.
- For runtime verification, inspect the actual service host and deployed
  checkout first. Follow AGENTS.md's restart and dashboard/cached/uncached TLDR
  smoke protocol, keeping one cookie session and scanning the last minute of
  `hn_rewrite.service` logs. Verify client count/vote behavior as well. A local
  test pass is not deployment evidence; preserve remote WIP before any deploy.

## Review recheck and test quality — 2026-09-30

Scope: live checkout `fe38d9f`; existing property tests and relevant backend,
ranking, web/browser, and TUI integration tests. Test-only additions; no
production database access or service restart. All seven original findings
and an additional interaction-batch boundary bug reproduced with unmarked
intended-behavior assertions. The evaluation reproduction proves
deck divergence, not the frequency or size of an effect on real ranking metrics.

Added coverage:

- `tests/test_database_concurrency.py`: controlled real SQLite pooled-writer
  interleaving; an old writer overwrites the newer comments, body and counts.
  Scheduling matters more than generating arbitrary text or numbers; the
  follow-up handoff above identifies a timeout that still needs tightening.
- `tests/test_tldr_probe_properties.py`: generated count relationships use
  both real probe helpers, mocking only Firebase. Shrunk failure: fetched=1,
  stored=2, live=2 returns no growth. Checks count preservation independently.
- `tests/test_client_state_regressions.py`: variable-length vote/undo chains
  with an earlier request failure and an independent acknowledged-write model.
  An earlier failed vote loses the final vote; a failed undo duplicates history.
  Deterministic tests also prove the active web card retains old counts and
  count-only polling starts no refresh.
- `tests/test_rss_date_properties.py`: generated UTC dates and fixed local
  timezone offsets, for RSS publication and Atom update fields. A one-minute
  offset already changes the stored instant; UTC controls pass.
- `tests/test_eval_deck_parity.py`: a bounded fold-local deck where Interest
  affects Recommended semantic deduplication. A production control passes;
  evaluation omits the Interest crosspost and retains the duplicate.
- `clients/tui/tests/test_client_state_regressions.py`: event-gated vote
  rollback and prefetched-window count regressions. Current window shows
  444/218; selecting the cached neighbor restores 100/10.
- `tests/test_interaction_integer_properties.py`: generated signed 64-bit
  overflow story IDs and mixed-batch ordering exercise Flask parsing through
  real SQLite insertion. At `server.py:2453`, IDs are not storage-bounded;
  `database.py:1437` raises on binding `2**63`. Verified HTTP 500 and no
  inserted events, losing the valid neighbor instead of rejecting only the
  malformed event. This is an additional P2 finding.

The known failures use `xfail(strict=True, raises=<specific mismatch type>)`:
only the identified invariant failure is expected. Remove each mark with its
implementation fix. To see the original failing assertions, run the new files
with `--runxfail`. No Hypothesis property was added where generated values would
merely decorate a fixed missing-field or scheduling regression.

Existing tests worth preserving:

- Scalar kNN oracle (`test_pipeline.py:6605`), embedding-cache states and
  permutations (`test_data_invariants.py:113`), exhaustive comment packing
  (`test_comment_packing.py`), chronological splits / URL isolation, and the
  deck-version state machine check useful independent contracts.
- Markdown properties generate structured documents, assert idempotence,
  bounds and ordered subsequences, and execute the JavaScript mirror. TUI
  boundary properties generate valid feeds and use an independent Unicode
  sanitation oracle. Finite tokens inside variable structures are useful
  generators, not just parametrization.

Concrete quality gaps, verified without editing existing tests:

- `tests/test_server.py:2493` mocks `_probe_live_counts`, so its healed-count
  case bypasses the production filter that causes the stale-summary bug.
- `tests/test_eval.py:43,146` asserts whole-deck parity with a pool exhausted
  by Recommended, Unsure and Novel; Interest never gets exercised. The new
  deck topology reaches that branch.
- `tests/test_server_properties.py:167-202` caps its valid numeric generators
  and stops at parsing. It never tests parsed values against SQLite storage,
  so it misses the oversized-integer batch loss covered by the new property.
- `tests/test_server_properties.py:70-83` claims atomic multi-bucket denial
  but checks only the global admission ceiling. An always-deny limiter passes
  its oracle. Compare each decision and bucket consumption with an independent
  per-key/global model.
- `tests/test_embedding_bakeoff.py:118-123` allows an empty prefix for text
  beyond the chunk budget. Returning `[""]` passes the 60-word / one-token
  case. Assert the exact prefix that fits the available budget.
- `tests/test_ch_client.py:322-370` checks only that `children` exists; its
  mock has no story `kids`. Returning no comment trees still passes. Use
  separate realistic responses and assert attached child IDs.
- `tests/test_browser.py:229` awaits the `reloadInFlight` object, not its
  `.promise`; the polling loop eventually observes readiness, but this does
  not establish the intended synchronization.
- `clients/tui/tests/test_impressions.py:17-26` includes a legitimate initial
  impression if startup takes long enough; `settle` deliberately does not
  await impression workers. This explains the double-event failure under
  load. `tests/test_reddit_fetch_queue.py:190` also assumes tight wall-clock
  scheduling. Use controlled clocks or observable events rather than narrow
  timing windows for these contracts.
- Production raw-embedding preservation lacks a direct assertion on estimator
  inputs; existing feature-concatenation and eval-scaler tests cover neighboring
  layers. Node feed tests intentionally stub summary rendering and cancellation,
  so those behaviors require browser coverage.

## Hot-thread counts — 2026-09-30

- Symptom: Gemini 4 Argon (49913571, posted 20:04 UTC) showed 419 pts /
  165 comments in the TUI at 21:00 while Firebase had 444 / 218. `r` at
  21:01 and 21:02 regenerated the summary (Algolia hydration: 165, then 183
  comments) and stored 449 / 183, but the TUI still showed 419 / 165.
- Cause: `/api/feed` copied counts from `CandidatePool` story snapshots
  (`pipeline/candidate_cache.py`), rebuilt only by the hourly regen
  (`_pool_changed`); tldr-detail wrote the DB and nothing else. The TUI
  refetched the feed on `r` before the ~4 s regeneration finished, and the
  summary reply carried no counts.
- Algolia lags Firebase on a fast thread: at 21:01 it returned 165 comments
  while Firebase said 218. The hourly CH pass could also lower a fresher
  stored score (`upsert_story` wrote `score` unconditionally).
- After `fcbfc9a`: `hot_refresh` 22:05 probed 20 / changed 18 (729 ms),
  22:15 probed 20 / changed 8 (813 ms); /api/feed 667 / 412 = Firebase.
  tldr-cache returned 204 (`prefetch_cache_stale`) for Argon: 294
  summarized vs 399 stored comments, past max(294 // 3, 5) = 98.
- Unrelated, pre-existing: `tldr: discussion call failed (status=None),
  salvaging article-only` 54 times in 2 days (also 21:40, before the
  deploy); the partial summary is not cached, so the next tap retries.

## Explore categories — 2026-09-30

- Live yield, profile 151 since 09-26: 5.6% of shown Explore cards upvoted vs
  26.8% of Recommended. Badges were not logged, so Explore cards were split
  into thirds by similarity to earlier votes: far (Novel-like) 1/78 up, 92%
  of its votes down; middle (Unsure-like) 1/78; near an upvote (Similar-like)
  14.1%. Similar used the weak full-text nearest-upvote signal.
- Interest clusters, KMeans k=10 on 151's 735 upvotes, read coherent: AI
  policy, AI news, open models, AI research, AI coding, nomad/FIRE, health,
  investing, gadgets, maps/transit. Warm-started refit 0.02 s vs ~1 s cold on
  the VPS.
- Live after `34051df`: 1w 🧭 AI buildout, LeanFIRE in Bangkok, Opus 5.5
  prompting guide, city shape, Live Avatar; 1d 🧭 "AI Is a Collective
  Disaster", Frog and Toad, month three as a nomad, "AI Gave my brother
  independence", Minitel. 🤔 on 1w took the investing/health stories
  (private companies, a health dashboard, a 150k inheritance), which overlap
  with interests.
- Badges are logged per impression from `34051df` on; judge each badge by
  its upvote rate around 2026-10-14.

## Do Unsure votes teach the ranker more? — 2026-09-30

`scripts/eval_unsure_votes.py` on a live snapshot (profile 151, 3091 votes in
time order, production ranker; results
`~/.local/state/hn-rerank-eval/unsure-votes-20260930.json`). Train on the
oldest votes, add N votes from the next 618 by strategy, test on the next 20%.
AUC up-vs-rest (random: mean ± sd of 5 seeds):

| split | N | entropy (Unsure) | random | top-scored | none / all |
|---|---|---|---|---|---|
| 0.6:0.2 | 100 | 0.856 | 0.861 ± 0.002 | 0.865 | 0.861 / 0.866 |
| 0.6:0.2 | 300 | 0.860 | 0.862 ± 0.003 | 0.867 | |
| 0.4:0.2 | 100 | 0.770 | 0.771 ± 0.002 | 0.769 | 0.767 / 0.798 |
| 0.4:0.2 | 300 | 0.783 | 0.784 ± 0.003 | 0.787 | |

- Entropy-picked votes never beat random (AUC up-vs-down and recall@40 agree);
  top-scored votes, what Recommended already shows, help most in 3 of 4.
  NDCG@12 swings ±0.03-0.05 between random seeds, too noisy to read.
- Caveat: the pool is votes the user chose to cast, mostly on Recommended
  cards, not the whole candidate set Unsure picks from.
- With the live rate (Unsure-like Explore cards 1/78 upvoted), Unsure shows no
  learning or reading value here.

## "Because you upvoted" coherence — 2026-09-30

Report: 49908757 ("Ubuntu 26.04.1 LTS") said "Because you upvoted: NInfer
Qwen 3.8-27B uncensored on RTX 5090". Read-only on the VPS DB, profile 151
(735 upvotes), stored full-text embeddings (mxbai-embed-xsmall, title +
self + article + up to 6000 chars of comments):

- Ubuntu's nearest upvote 0.70; the next seven 0.63-0.67. By title
  embedding (64 tokens) its best is 0.33 ("The $60 Gaming PC").
- Live 1w deck, 47 cards: nearest-upvote similarity min 0.23, median 0.77,
  median lead over the runner-up 0.015; the 0.35 floor passed 42.
- Clear misses: SR-71A <- "Things we learned about LLMs in 2024" (0.64),
  Tank Body Problem <- a GPT-6 Astra post (0.65), Factorio <- "Small Models
  Have Arrived" (0.76), Nvidia watchdog chip <- "Anthropic CEO says AI swarm"
  (0.79), ESP32 BitNet <- Jamesob's local-LLM guide (0.81). Good matches
  also sit low (creatine <- "Does creatine make you smarter?" 0.69, Opus 5.5
  <- Opus 5 0.82) and weak ones high ("Withdrawal phase" 0.88, "The
  Education of a Doomer" 0.89): no cutoff separates them.
- Title-only nearest (median best 0.53) read far better: creatine 0.84,
  Opus 0.93, SNL/Dario 0.61, Post-AGI 0.56; below ~0.5 it is noise.
- Shipped (user's choice, `2190914`): cards with a Hot/Top/Talk/Unsure/Novel
  badge need 0.85, picked by eye above the clear misses. On that deck 22 of
  42 lines stay (all 20 dropped were 🔥 or 🤔). Live after restart: 1d deck
  17/38 attributed, Ubuntu (🔥🎯) none.

## TUI review — 2026-09-30

Rendered the reader on the live 1w feed (mocked API; Textual SVG via
headless Chrome, ImageMagick drops the text). Found: `▲ 0` on 12/47 non-HN
cards (unknown, not zero), `x.com` for the 5 AINews items, the parsed
`best_match_title` (42/47) never shown, a static "Loading summary…" during
~15 s generations. Fixed in `9c17ac0`. Not done: below 100 columns the list
gets 1fr vs the TLDR's 3fr (2 headlines at 90x39); badges are bare icons
(labels only in the `b` legend).

## Live blend, first read — 2026-09-30

Profile 151, stories by first impression (read-only on the VPS DB at 06:55
UTC; last 151 vote 01:04). Blend live since 2026-09-29 17:17 UTC.

| period | shown | up | up/shown | up/voted |
|---|---|---|---|---|
| 09-22..25 | 287 | 61 | 21.3% | 26.3% |
| 09-26..merge (09-29 14:01) | 420 | 58 | 13.8% | 19.6% |
| merge, no blend | 65 | 10 | 15.4% | 20.0% |
| blend | 211 | 22 | 10.4% | 15.1% |

By source since 09-26 (pre vs blend): HN 10.0% (269) vs 8.9% (101); RSS and
other 19.0% (216) vs 11.8% (110), with mean story age at impression 132 ->
286 days (Archive-window browsing). The drop is in old non-HN stories, not
in the ranked HN deck, and pre vs blend is not significant (z ~ 1.2). The up
rate was already falling before the blend (21% -> 14%). Verdict: no evidence
either way; no rollback. Note that SQLite compares `strftime` text above any
number unless cast (`CAST(strftime('%s', ...) AS INT)`).

## Embedding model hill-climb paused — 2026-09-30

Setup: `encode_replay_embeddings.py --device gpu`, small test (300 votes/class,
3 folds), vs stored + gemma best (step 10: P@12 0.556, AUC 0.721, n.s. on
P@12 but 7/8 folds better AUC).

- **harrier-270m** (Microsoft 2026-03-30, Gemma 3, MMTEB 66.5/69.0): GPU f32
  encoding 0.65 s/story (no NaN at f16 after fixing RMSNorm overflow with
  ACTIVATIONS_SCALE_FACTOR=8+); small test P@12 0.389 (vs best 0.556, n.s.),
  AUC 0.708 (vs best 0.720, n.s.). Not promising alone. Included in full-eval
  reject list (status).
- **Qwen3-0.6B** (Alibaba 2026-09, 768-d, decode-embed, no instruct):
  Encoding batch 8 on GPU, 900-story sample (300/class). Process killed by
  system memory pressure 15:28 UTC at story ~100 (685MB encoder + 10GB
  background, 11.5GB total). Did not reach checkpoint at 200 stories; no
  partial reuse. Deferred pending memory recovery (3GB free now).

Both models hit CPU/GPU constraints or memory limits on the laptop. None beat
step 10. Hill-climb paused until memory stability confirmed.

## Code review — 2026-09-30

Three review agents over 45882a2..9f97224 (server/TLDR/CH, ranking/eval,
TUI/AINews). Fixed (WORKLOG 2026-09-30): pointer-follow retries and stale
fallback, pointer false positives, eval double blend and dense-model
mismatch, blend at the 20/20 gate, unbounded blend caches, AINews regen
abort, TUI `a` guard, tmux agent check, status tail.

- Eval mismatch, confirmed: `prodlr`'s dense logreg had 5 meta columns
  (live: the SVM's 10, plus zero rows for absent classes) and took C from
  `svm_c`. C matched only while config.toml had `svm_c = 0.1`, i.e. for
  every run before 26b9474; the columns never matched. So the reported
  prodlr numbers measured an approximation of the dense part; the TF-IDF
  part matched live. Now `prodlr[svm_c=4;lr_weight=0.2;tfidf_weight=0.3]`
  equals the live blend exactly for a heavy voter (test). Not rerun yet.
- Pointer rule recheck against the live DB: 373 HN stories with 1-2
  comments but several stored comments (the followed signature). Algolia
  source threads: 109 match the committed rule, the rest do not or are
  missing. A first tighter draft rejected 5 real pointers ("Discussed 2
  days ago with 280 comments: <link>", "Previous discussion 11 days ago:
  <link>"), so "discussion" is now rejected only when followed by
  "of/about" something other than this/it. Final rule: all 109 still
  match; none of the live follows was a misfire.
- ClickHouse playground quota: 21:43 and 21:54 UTC 2026-09-29, prewarm
  hit `queries_per_normalized_hash = 101/100` per hour during repeated
  restarts (each restart prewarms; the comment walk is one query per
  level per chunk). None after the 22:06 restart. Cause (2026-09-30):
  every regen re-fetched ~330 small threads whose comments are all
  one-word replies (49809698: 9 comments of 3-15 chars), since an empty
  selection wrote nothing back. Fixed (WORKLOG 2026-09-30).
- Fixed 2026-09-30: prefetch's follow overwriting comments a concurrent
  hydration just wrote; live-window retry of deterministic errors.
- Open, not fixed: prose-reply
  retry never fires for Responses-API providers (`finish=None`); tweet
  `internal_exception` skips backoff; `_build_story_kids_query` filters
  `deleted/dead` per row version; AINews cards sharing a tweet rebuild
  each other partially, capped cards never refill, two topics leading
  with the same tweet collapse in render dedup; warm start makes fits
  depend on history (max 0.005 difference in P(up) - P(down)).

## Rerank latency — 2026-09-29

- Before: `linear_blend_fit_ms` 5-18 s per vote (19:50-20:41 UTC, 10k
  candidates); the 7.7 s "warm" figure was a best case. After the cap raise
  (11,360 candidates) warm reranks were 15-22 s.
- VPS probe, 151's 2,883 votes: hashed rows cold 2.4 s / warm 0.11 s; dense
  LR 1.4 s; TF-IDF logreg C=4 ~44-50 lbfgs iterations, 7-14 s (229k kept
  columns). Warm start from the fit one vote earlier: 12 iterations, 2.2 s
  vs 13.9 s, Spearman 0.99999.
- Spikes at 21:03 and 21:20 (`tier2_ms` 12-17 s, `dedup_ms` 8-12 s, normally
  under 1 s) coincided with post-regen article fetch embedding one story at
  a time and regen prewarm embedding ~1,000 stories.
- 21:38 rerank 56 s: `candidate_sql_ms` 49 s behind the pool lock while the
  rebuild embedded 51 stories (45.6 s). Those were articles fetched from
  deck ranking copies (no comments): embedded under a hash of title +
  article, then `upsert_story` merged comments into `text_content`.
- After all fixes, 12 reranks while voting (21:43-21:48): total 5.7-11 s
  (one 17.9 s), fit 0.3-1.3 s (occasionally 3-5 s), pool wait < 2 ms,
  SVM decision 1.1-2.7 s, feature prep 0.75-3 s.

## TLDR providers and quality — 2026-09-29

- `gofree` (longcat) since 17:02 UTC: 132 TLDRs, llm_ms p50 45 s, p90 61 s,
  max 76 s; 25 llm_error, 35 half-only. Hydration under 2 s.
- Bakeoff, 3 stories each (VPS `scripts/bakeoff_tldr_providers.py`):
  gemini flash-lite 1.6-6 s 3/3; groq <1 s, 429 on the 3rd; cerebras 402;
  gofree 20-52 s 3/3; mistral 2.2-3.9 s 3/3. Gemini free tier live: 4 taps,
  then `GenerateRequestsPerDayPerProjectPerModel-FreeTier` quotaValue 20.
- Key probe 20:28 UTC: mistral 200, groq 200, cerebras 402, zen 402
  (funds), go 429 (monthly limit, resets 2026-10-06 16:28 UTC), openrouter
  402 ($10.21 used of $10).
- Mistral vs longcat on NSL, Pac-Bench, HN.watch: same facts, Mistral drops
  names, versions and quotes. Mistral answered 1 of 3 discussion calls on
  30230620 as prose (no bullets), which `_valid_llm_completion` rejects.
- OpenRouter prices ($/day at ~520K in / 350K out): gpt-6-luna 0.23,
  gemini-2.5-flash-lite 0.19, gpt-5-nano 0.17, gpt-oss-120b 0.08,
  claude-haiku-4.5 2.27.
- 1,104 stories link a tweet, 1,034 without article text.
- Pointer threads: "<= 400 chars with an HN link" matched 465 rows (402
  live `hn`), mostly real threads citing others ("Related: <other story>",
  "also saw it here", two comments). One comment opening with
  moved/dupe/discussion wording or "moved the comments": 162 (124 `hn`, 34
  `ch_seed`, 4 `bq_seed`); 160 followed. Under the loose rule (20:44-22:06
  UTC) the service followed only 32148318 and 38507672 and cached no TLDR
  for the 303 loose-only stories.

## ClickHouse source review — 2026-09-29

Read-only review by a subagent, spot-checked:
- Healthy: 5 `CH live_window failed` since Sep 23 (4 DNS, 1 "Too many
  simultaneous queries"); none of MEMORY_LIMIT. Freshness: CH max(id) equal
  to Firebase maxitem, newest item 26 s old at 20:36 UTC.
- Sep 25 20:50 failed call: regen logged 4,068 candidates. The deck reads
  live HN from the DB (`load_production_candidate_stories`), so the feed kept
  its rows; scores went stale for one regen.
- Comments were flat in GROUP BY order: on 49892245 every comment had
  depth 0; top-level order differed from Firebase on 5/5 stories. After the
  fix: 226 nodes vs 213 descendants, depth 12, order matches, 172/189
  selectable comments below top level. 140 stories in 8.1 s at depth 30.
- Depth 5 missed 10-22% (128 vs 165 of 210; 518 vs 573). Row-level
  `deleted = 0` kept 30 of 5,428 comments deleted later.
- CH had 7,511 live stories with score >= 5; `limit=5000` cut 2,511 (all
  score 5-7, 56 from the last 24 h). DB held 5,148 live `hn` rows against a
  5,000 deck cap.
- Not fixed: `query_single_story` is test-only; ~140 HN stories re-prewarm
  every regen (149 have <= 9 comments, likely under the 60-char minimum).

## TUI headline dividers — 2026-09-29

Tried on the TUI list (`#headlines`, Textual 8.2.8 OptionList). Kept: the
`None` separator rows (user: "faint divider on its own line was ok").
- CSS `border-bottom` on `.option-list--option`: ignored by OptionList.
- `None` between options: a faint `─` row after each story (style via
  `.option-list--separator`); takes no option index, so `highlighted` still
  indexes `self.stories`. Costs one row per story; kept.
- Underlining the meta line: zero rows, but Rich/Textual have no underline
  colour, so it is drawn in each segment's colour (blue domain, green points)
  and reads as links; stops at the age, not the row edge. Full-width padding
  would go stale on resize (rows re-render only on highlight). User: revert.
- Not tried: zebra background on alternate rows (zero rows; needs per-option
  background, which OptionList does not expose without subclassing).

## Full-eval reruns of near-tie ideas — 2026-09-29

The small test (900 votes, 3 folds) is too noisy for changes of ~0.01, so
every near-tie went to the full eval (user 1, all votes, 8 folds). Base
`prodlr[svm_c=4.0;lr_weight=0.3]`, stored + gemma: P@12 0.708, AUC 0.791.

| variant | P@12 | AUC | folds better (AUC) |
|---|---|---|---|
| + TF-IDF 0.2 (`tfidf_weight=0.2`) | 0.729 (+0.021, n.s.) | 0.800 (+0.009, p=0.015) | 7/8 |
| + TF-IDF 0.3 (lr 0.2) | 0.729 (+0.021, n.s.) | 0.801 (+0.011, p=0.021) | 7/8 |
| one SVM per embedding (`dims=384+768`) | 0.656 (-0.052) | 0.801 (+0.011, p=0.004) | 8/8 |
| logreg up vs rest (`lr_target=up`) | 0.698 | 0.790 | 3/8 |
| stored only, base | 0.698 | 0.774 | - |
| + skipped stories as down | 0.667 | 0.775 | 2/8 |
| + skipped stories as neutral | 0.667 | 0.772 | 0/8 |

TF-IDF (word 1-2 grams + domain/source tokens, logreg fit per fold) is the
first idea to beat the base on AUC without costing the top 12; sklearn
only, so deployable without gemma.

Unseen votes (base -> `lr_weight=0.2;tfidf_weight=0.3`):

| check | top-12 up | AUC |
|---|---|---|
| user 1 reserved newest 20% (`--confirmation --folds 4`, stored+gemma) | 0.646 -> 0.667 (P@12) | 0.768 -> 0.774 |
| 151 alone, votes after 09-26 (stored) | 9 -> 10 | 0.848 -> 0.857 |
| 151 + user 1 since Jul, after 09-26 | 8 -> 10 | 0.879 -> 0.890 |
| 151 alone, after 09-27 | 7 -> 7 | 0.836 -> 0.861 |
| 151 + user 1 since Jul, after 09-27 | 7 -> 7 | 0.913 -> 0.921 |

Better or equal on every check, never worse; each gain alone is within
noise. (151 runs after the tie-aware AUC fix, confirmation before it.) Per-embedding SVMs improve AUC but lose
top-12 precision; skipped-story negatives and the binary target do nothing.

## MMR diversity on the top 12 — 2026-09-29

Production has `enable_mmr = false`. Full eval (user 1, all votes, 8 folds,
stored embeddings), same run's raw ranking vs MMR-filtered, top-12 upvotes
per fold (mean):

| threshold | production raw -> MMR | prodlr C=4 + 0.3 raw -> MMR |
|---|---|---|
| 0.65 | 7.50 -> 6.12 | 8.38 -> 7.62 |
| 0.75 | 7.50 -> 7.25 | 8.38 -> 8.00 |
| 0.85 | 7.50 -> 7.38 | 8.38 -> 8.25 |

MMR only costs upvotes (fewer at looser thresholds) and raises the neutral
share; downvotes unchanged. Keep it off. Caveat: judged-only pool, so the
metric cannot reward variety for its own sake.

## Merging user 1's votes into the live profile (151) — 2026-09-29

Snapshot `snapshot-20260929.db` (read-only VPS copy; 151 has 459 votes,
2026-09-24 to 09-29). `scripts/compare_profiles.py --old 1 --new 151`:

| | votes | up/neu/down | HN | archive (>30 d old when voted) | median story age |
|---|---|---|---|---|---|
| user 1, June | 2,713 | 39/29/31% | 85% | 36% | 20.7 d |
| user 1, Jul-Sep | 2,476 | 20-30% up | 49-82% | 1-8% | 0.9-2.5 d |
| user 151 | 459 | 23/32/45% | 67% | 2% | 5.5 d |

- June is the archive "best of" deck: user 1 upvoted 59% of archive
  stories vs 26% of live ones. 151 has barely seen archive stories.
- 151 is harsher on HN (16% up vs user 1's 31%); non-HN up rates match
  (37-38%).
- Same person, stable taste: on 147 stories voted by both, 69% same label,
  up->down 2, down->up 0; disagreements are about neutral.

Eval: `merge_profiles_snapshot.py --variant ID[:DATE][:live]` builds
merged users (151's votes plus user 1's on other stories) in
`snapshot-20260929-merged2.db`; 6 user-1 votes whose stored embedding no
longer matches the story text were dropped from that copy. Holdout = 151's
votes after the cutoff (`--holdout-after`), `prodlr[svm_c=4.0;lr_weight=0.3]`
on stored embeddings:

| trained on | AUC, cut 09-26 (≈257 votes) | cut 09-27 (145-146) | top-12 up (26 / 27) |
|---|---|---|---|
| 151 only | 0.848 | 0.837 | 9 / 7 |
| + all of user 1 | 0.869 | 0.888 | 8 / 7 |
| + user 1 since Jul | 0.879 | 0.913 | 8 / 7 |
| + user 1 since Aug | 0.853 | 0.893 | 6 / 6 |
| + user 1 live only | 0.869 | 0.903 | 9 / 7 |
| + user 1 live, since Jul | 0.879 | 0.914 | 9 / 7 |

Codex review (read-only, `/tmp/hn-eval-local/codex-ml-review.md`): the
AUC gain stands (at 09-27, the four dated/live merges keep the identical
146-vote test set: 0.837 -> 0.89-0.91), but choosing among the merges does
not: "since Jul" and "live since Jul" differ by one ordered pair; the
cutoffs overlap; top-12 counts are unchanged or (since Aug) worse. URL-group
isolation drops 1-2 of 151's test upvotes in some merged runs. Merge also
removes user 1's vote whenever 151 ever voted on the story (retrospective).
Conclusion: merging raises AUC on 151's later votes by ~0.03-0.08 without
moving the 12-card top; which slice of user 1 is not decided by this data.
A July-onward merge drops the archive-heavy June votes, matching the
profile comparison. The live merge (writing 151's training votes in the
VPS DB) is not done: it needs the user's decision.

## Feed yield check — 2026-09-29

Read-only on the VPS (`26c3736`). The scheduled check assumed user 1, who
stopped voting on 2026-09-24; the live profile is user 151.
- Reddit: `reddit_limiter 429` 63/day (09-27), 54 (09-28), was ~150/day;
  full refreshes ~2h apart. Every configured subreddit refreshed within
  ~2h except r/ocaml (7 failures, "fetch returned no snapshot", last ok
  09-28 07:25). Stale rows in `reddit_feed_state` are dropped feeds.
- Article text: 44 regen runs fetched 220/260 RSS article bodies; 531
  recent rows of configured non-Reddit RSS feeds still have none.
- Errors: none beyond handled feed read timeouts / server disconnects. The
  15 "Failed with result 'exit-code'" lines are restarts (SIGTERM -> 143).
- Yield since 09-27, user 151 (up/shown): HN 3/111 (38 of 53 votes down),
  Latent Space 3/9, Zvi 3/5, r/LocalLLaMA 1/6, LessWrong 0/15. New feeds
  mostly not shown yet. r/transit 0 up / 1 down, r/MachineLearning 1 up
  of 2 shown: too few to decide, keep both.

## Stacked ranker — 2026-09-29

Second stage (`stack[...]`) over out-of-fold content scores (RBF SVM C=4
margin, 3-class logreg P(up)-P(down); 5 inner folds, candidates get the
mean of the inner models) and/or metadata (source and domain up/down rates
shrunk toward the global rate, log points, log comments at fetch, log
length, log age at vote, Show/Ask HN, HN vs other). Embeddings: stored +
gemma.

- Leak found and fixed: leave-one-out source/domain rates let gradient
  boosting learn an inverted label signal (within a key the rate drops
  exactly when the row's own vote is up); small-eval AUC fell to 0.58 and
  metadata alone to 0.50. Out-of-fold rates fixed it
  (`test_out_of_fold_label_rates_ignore_own_vote`).
- Small test (300/class, 3 folds; current best P@12 0.528, AUC 0.721):
  gbm both 0.528/0.717, ordinal both 0.500/0.714, pair both 0.556/0.730,
  lr both 0.583/0.728; metadata alone AUC 0.57-0.59 on temporal folds.
- Full eval (all votes, 8 folds; current best P@12 0.708, AUC 0.791,
  reproduced exactly): pair both 0.583 / 0.786 (better in 1/8 folds), lr
  both 0.521 / 0.779 (P@12 p=0.04). Both also raise the downvote share of
  the top 12 (0.010 -> 0.062). Rejected as implemented.
- Codex review (2026-09-29) found the implementation flawed, so this does
  not rule out stacking: inner folds reuse kNN-similarity features built
  from all outer-training labels (inner leak), the age feature uses each
  candidate's own vote time, `comment_count_at_fetch` is always 0 in
  heldout-feedback rows, and the content cache ignores gamma/kernel. Also
  repo-wide: AUC counts tied pairs as 0/1 instead of 0.5.

## What others do for small-data personal ranking — 2026-09-28

Web survey (no code). Ranked by expected value per effort for ~5k explicit
votes on one user, CPU only:
1. Word TF-IDF (1-2grams + domain/submitter tokens) linear model next to
   the dense SVM. Semantic Scholar feeds average a TF-IDF SVM and a SPECTER
   SVM (arxiv.org/html/2301.10140v2); Scholar Inbox found TF-IDF lower AUC
   but higher nDCG than embeddings (arxiv.org/html/2504.08385);
   github.com/fredrik/rekorderlig ranks HN on words/domain/submitter.
   Fit the vectorizer inside each fold.
2. Unvoted stories as low-weight negatives (both systems above; Scholar
   Inbox tunes count and weight). Draw only from before the fold cutoff.
3. TabPFN as the stacker (~15 features, ~3k rows is its regime; ties
   logreg on raw frozen embeddings, arxiv.org/html/2607.11007). Needs
   torch in an opt-in group; check the license.
4. Boosted lambdarank/pairwise (being tested as `stack[...]`); large gains
   reported only at millions of rows (arxiv.org/abs/2608.13874).
5. Exploration: uncertainty sampling (Scholar Inbox), LinUCB
   (github.com/permacommons/habitfeed); needs a learning-curve simulation,
   not AUC.
6-8. Low value here: SetFit/contrastive fine-tuning (gains at 8-64
   examples/class), LLM rankers or LLM-written profiles (popularity and
   position bias, cost), two-tower/user embeddings (one user = the linear
   model). Working personal feeds use linear models on embeddings
   (adamwiggins.com/posts/a-bluesky-feed-for-one, scour.ing/docs/ranking).
Scale check: Scholar Inbox embedding swaps moved AUC ~0.004; treat any
single change worth >0.03 AUC here as a possible leak.

## Incremental ranker hill-climb — 2026-09-28

Setup: `eval_ranker_variants.py --candidate-pool heldout-feedback`, user 1,
snapshot 2026-09-25 (same development folds as the 2026-09-25 study; the
newest 20% stays reserved). Each step changes one thing from the current
best: first a small test (300 votes/class, 3 folds, ~1 min), then the full
eval (all 5,189 votes, 8 folds, ~5 min) only if the small test is no
worse on the top-12 upvote rate and better on AUC. Primary metric: upvote
rate in the top 12 (the clients' 12-card view); also AUC(up vs rest),
downvote/neutral share of the top 12, discovery upvotes (top-12 upvotes
below the fold's median similarity to training upvotes) and non-HN
upvotes. Reports: `/tmp/hn-eval-local/{small,full}-*.json` (not kept).

| Step | Change | Small test | Full eval (vs previous best) | Kept |
|---|---|---|---|---|
| 1 | up-minus-down margin scoring | worse | – | no |
| 1-2 | SVM C 0.1 → 2 | P@12 0.306→0.389, AUC 0.577→0.657, 3/3 folds | P@12 0.625→0.646 (p=0.52), AUC 0.726→0.759 (p=0.046), discovery 0.25→1.12 | yes |
| 3 | mxbai-xsmall text cut at 512 tokens | 2/3 folds better | P@12 0.646→0.615 (worse 7/8), AUC +0.009 | no |
| 4 | γ 0.015 / 0.05 at C=2 | P@12 lower | – | no |
| 5 | + logreg rank blend, weight 0.3 (0.6 tested) | P@12 tie, AUC 0.657→0.674 | P@12 0.646→0.688 (p=0.10), AUC 0.759→0.769 (p=0.07) | yes |
| 6 | body + comments as two 512-token vectors (mxbai-xsmall) | P@12 0.389→0.500 (1/3 folds), AUC 0.674→0.700 | P@12 0.688→0.635 (n.s.), AUC 0.769→0.788 (p=0.025, 7/8), discovery 1.25→0.12 | no (top 12 is the target) |
| 7 | embeddinggemma-300m (512 tokens, classification prefix) instead of stored | P@12 0.389→0.611 (3/3), AUC 0.674→0.710 | P@12 0.688→0.594 (worse 6/8), AUC 0.769→0.794 (p=0.011, 7/8) | no |
| 8 | stored + gemma side by side (50/50) | P@12 →0.500, AUC →0.706 (3/3, p=0.003) | P@12 0.688→0.667 (n.s.), AUC →0.785 (8/8, p=0.002), downvotes 0.031→0.010 | close |
| 9 | stored + gemma, 30% gemma (70%: small AUC 0.716) | AUC 0.683 | P@12 0.656, AUC 0.782 | no |
| 10 | step 8 retuned: C=4 (γ 0.015/0.06, blend 0.5 tested) | C=4 and blend 0.5 ≥ step 8 | **P@12 0.688→0.708 (4/8 better, n.s.), AUC 0.769→0.791 (8/8, p=0.002)**, downvotes 0.031→0.010 | yes |

Step 10 attribution: C=4 on stored alone gives P@12 0.698, AUC 0.774
(+0.005); adding gemma at C=4 adds AUC +0.016 (8/8, p=0.004). On the
fresh votes (below) step 10 gets AUC 0.885 ± 0.07 vs 0.859 and 9 vs 10
upvotes in the top 12 (one card; noise). Shipping it needs gemma
embeddings for every candidate on the VPS (not measured there; 0.45-0.6
s/story on the laptop iGPU). The small test predicts AUC direction but
not the top 12 (step 7: +0.22 small, -0.09 full).

Best on stored embeddings (no re-encoding), `prodlr[svm_c=2.0;lr_weight=0.3]`, vs production: top-12
upvotes 7.50→8.25 of 12 (p=0.20), AUC 0.726→0.769 (p=0.034), downvotes
0.62→0.38 and neutral 3.9→3.4 per top 12, discovery upvotes 0.25→1.25,
non-HN upvotes 1.25→2.75. Caveats: the small test misled once (step 3:
~1 story per fold at top 12); all steps share the development folds; the
2026-09-25 C=4 + blend challenger failed on the newest votes and hand
orderings, so votes after 2026-09-25 (a fresh snapshot) are the real test.

Fresh votes: user 1 stopped on 2026-09-24; the user's new default profile
is user 151 (363 votes, 2026-09-24 to 2026-09-28; 135 stories voted by
both). Snapshot `snapshot-20260928.db` (read-only VPS copy);
`scripts/merge_profiles_snapshot.py` builds a copy with user 900001 = 151's
votes plus user 1's on other stories. `--holdout-after 1790380800` trains on
votes before 2026-09-26 and tests on 151's 161 later votes (44 up):

| Trained on | Ranker | Upvotes in top 12 | AUC ±95% |
|---|---|---|---|
| 151 only (live today) | production | 8 | 0.844 ± 0.08 |
| 151 only | C=2 + blend 0.3 | 9 | 0.862 |
| merged 1+151 | production | 7 | 0.837 |
| merged 1+151 | C=2 / C=2 + blend 0.3 | 10 / 10 | 0.852 / 0.859 |

Direction holds on unseen votes, within noise (one 12-card block). AUC is
higher than on the development folds (0.73-0.77); not a text leak: text
length, article presence and comment count alone give AUC 0.36-0.53.
The live profile trains on 363 votes, not 5,500: merging needs the user's OK.

Taste drift check (2026-09-28 snapshot): old votes stay useful. On the
135 stories both profiles voted on, the same person agreed with their
earlier vote 70% of the time (up to down: 2; down to up: 0). A logistic
model on the stored embeddings, trained on 600 of user 1's votes from
one month, predicts user 151's votes on the 228 unseen stories (48 up)
about equally well:

| Month voted | AUC |
|---|---|
| June | 0.83 |
| July | 0.83 |
| August | 0.75 |
| September | 0.81 |

All 5,189 votes give AUC 0.82 (95% CI 0.75-0.89); August and September
alone give 0.79. Every vote was cast between June and September 2026, so
this covers four months of taste; old stories are not the same as old
votes.

The earlier embedding screen (700 votes/class, 8 folds, same day) found
body + comments as two 512-token vectors best at top 12 (0.656 vs 0.594
full text at 512), within noise; bge-base, arctic-m-v2 and single-vector
variants (averaged, weighted, split budget, chunked comments) did not
beat it. On the laptop iGPU (OpenVINO f16, `--device gpu`) encoding is
~4x faster than CPU (bge-base 0.21 s/story, mxbai-large 1.05 s/story at
512 tokens); `scripts/bench_embed_gpu.py` times a model.

## Untuned embedding probe and 2026 models — 2026-09-28

The hill-climb eval compares new embeddings with a ranker (C, gamma, blend)
tuned on the stored ones, and against the tuned stored+gemma combo, so a
new model can lose on fit alone. `scripts/probe_embeddings.py` scores each
embedding file the same way, hyperparameters picked by inner CV per
embedding (logreg, RBF SVM, cosine kNN), plus a taste-free check: source
classification (macro-F1) and k-means V-measure on non-HN stories. `meta`
= source one-hot + log points/comments/length, no text. 900-story sample
(300/class), 5 folds, score P(up)-P(down):

| Embedding | best AUC up vs rest | up vs down | up vs neutral | up in top 12 |
|---|---|---|---|---|
| meta (no text) | 0.661 (svm) | 0.696 | 0.626 | 0.50 |
| stored mxbai-xsmall | 0.756 (svm) | 0.866 | 0.645 | 0.63 |
| embeddinggemma-300m | 0.767 (svm) | 0.859 | 0.674 | 0.67 |
| harrier-oss-v1-270m, no prefix | 0.752 (logreg/svm) | 0.850 | 0.654 | 0.63 |
| harrier-270m, classify prefix | 0.729 (logreg) | 0.838 | 0.621 | 0.58 |

Pairs (stored+X) and the remaining rows: `probe-s300*.log` in the saved
eval dir (below). Up vs down is easy (0.85+); up vs neutral is where every
embedding is weak (0.63-0.67), and metadata alone reaches 0.66 overall,
signal the SVM-on-embeddings ranker barely uses.

harrier-oss-v1 (Microsoft, 2026-03-30; MMTEB v2 66.5 / 69.0) is Gemma 3
(270m) and Qwen3 (0.6b). 270m on the iGPU gives NaN at f16:
`scripts/debug_fp16_overflow.py` traced RMSNorm-input Adds up to 2.7e7
(f16 max 65504). OpenVINO's `ACTIVATIONS_SCALE_FACTOR` of 8+ on GPU fixes
it (cos 1.0000 vs CPU f32); not yet wired into the encoder (it ran at f32,
0.65 s/story). 0.6b: f32 ran out of GPU resources; f16 passed the check,
then failed at 600/900 stories (batch 8); resumed with batch 2. Small
tuned-eval (3 folds) vs stored+gemma: 270m ties at best (AUC 0.708-0.726
vs 0.720, n.s.). Research shortlist (classification-heavy MTEB, fits the
UHD 620): jina-v5-text-nano (classification variant), Qwen3-Embedding-0.6B,
KaLM-mini-v2.5, mdbr-leaf-mt (cheap VPS option).

### Extra probes (`--extra`) and harrier-0.6b — 2026-09-28

Same 900-story sample (300/class). Few-shot = AUC up vs rest from N votes
per class (20 draws, ±SE); temporal = oldest 70% train, newest 30% test;
centroid and nn10 (share of 10 nearest voted neighbours with the same
label, chance 0.33) involve no fitting.

| embedding | few 25 | few 50 | few 100 | temporal | centroid | nn10 up | nn10 dn | k-means V |
|---|---|---|---|---|---|---|---|---|
| stored | 0.662 | 0.694 | 0.716 | 0.718 | 0.695 | 0.461 | 0.384 | 0.669 |
| gemma | 0.676 | 0.701 | 0.730 | 0.709 | 0.692 | 0.518 | 0.366 | 0.839 |
| harrier-270m (no prefix) | 0.670 | 0.699 | 0.720 | 0.698 | 0.692 | 0.478 | 0.385 | 0.679 |
| harrier-0.6b (prefix) | 0.661 | 0.681 | 0.700 | 0.684 | 0.676 | 0.473 | 0.415 | 0.661 |
| harrier-0.6b (no prefix) | 0.670 | 0.696 | 0.720 | 0.694 | 0.682 | 0.510 | 0.379 | 0.769 |

- gemma's edge is real but small: few-shot (+0.014 at 25/class, SE 0.005),
  purer upvote neighbourhoods (0.52 vs 0.46) and much cleaner source
  clusters (V 0.84 vs 0.67, only 74 stories / 5 sources). Stored keeps the
  best temporal (newest-votes) AUC.
- harrier-0.6b is worst on every metric except downvote neighbourhoods.
  But it was encoded with the instruct prefix, and on harrier-270m the
  prefix cost ~0.02 AUC (0.729 with vs 0.752 without), so its
  no-prefix rerun is the fair test.
- No-prefix rerun (2026-09-29): the prefix was the whole deficit. Taste AUC
  0.760 logreg (best logreg of all) / 0.751 SVM, up vs down 0.869 (best),
  top-12 up 0.70; few-shot and neighbourhoods level with gemma, topic
  clusters between (V 0.77); temporal 0.694 still below stored (0.718).
  Net: ties gemma, beats neither clearly, and is 2x its size (1.35 s/story
  on the GPU). Encode decoder embedders (harrier) without the instruct
  prefix on documents.

## TUI simplification — 2026-09-26

- Review (code-review, high) of `clients/tui/src`: a passive refresh could
  strand "Loading summary…" (it cancelled the prefetch the selection had
  joined); a failed `r` left a forced regeneration armed; reverse sort
  logged reversed list positions as impressions; 45s client timeout vs
  ~2 min server generation; post-vote loop refetched the feed every 1s for
  up to 30s. All fixed; reverse sort (`v`) removed.
- Refactor in 4 commits (2592706, 4274de9, ff85910, d1b4bea). State removed:
  prefetch queue/worker/`prefetching`/`prefetch_requests`/cache-miss set,
  `target`, `force_summary_id`, `pending`, `can_read` field + 3 timers + 1s
  interval. `app.py` 1680 -> 1625 lines (docstrings ate most line savings).
- Test speed: 54-60s -> ~39s at `-n 4`, ~25s at `-n 8`. Floor is Textual,
  ~0.5s CPU per app lifecycle (1.0s with the 5-section markdown fixture,
  0.86s at 3 sections). Fixed sleeps became `tests/_settle.py::settle`.
- Two undo-during-stale-deck tests had passed without reaching the
  `restored` path once votes stopped refetching; they now reload the stale
  deck and fail if that path is removed. Each new regression test was
  checked to fail without its fix.
- Live (read-only, `--prefetch-generate 0`, no votes): feed, prefetched
  summaries on j, sort switch, zoom and quit all worked against the VPS.
- Diagrams (today vs proposed): `/tmp/hn-tui-diagrams/` (not in git).

## Source yield review — 2026-09-26

Per source, all users (stories fetched in 30d / shown / % of shown upvoted).
HN baseline: 6031 / 1696 / 21%.

- Above HN: latent.space 43/69/59%, Slashdot 251/69/52%, r/digitalnomad
  160/69/47%, r/Bogleheads 159/67/37%.
- Dropped (0-13%): Tildes 515/74/6% (36 downvotes), r/awardtravel,
  r/programming, r/ProgrammingLanguages, r/Urbanism, r/selfhosted,
  r/dataengineering, r/CreditCards.
- Added (similar to the winners): understandingai, dwarkesh,
  pragmaticengineer, oneusefulthing, thezvi, The Register, r/expats,
  r/eupersonalfinance. Recheck their yield after a few weeks.
- Fetched but rarely shown: text is thin, so the ranker has little to go on.
  lobste.rs: 75/120 URLs are also HN stories and collapse into them.
  LWN (41 article 403s, subscriber pages) and Aeon (38 article 429s)
  keep only RSS snippets (~500-1000 chars vs 4-9k for Slashdot and
  latent.space). r/Compilers has text but no upvotes from anyone.
- Silent blogs are healthy feeds that post rarely (aphyr, jvns, eugeneyan,
  pudding, earlyretirementnow); huyenchip last posted 2025-01.
- Fetch errors 2026-09-23..25 were a VPS DNS outage (157 errors, all
  feeds, 4 empty HN live windows; LessWrong's 18 "timeouts" are all in it).
  Cause: all VPS egress, DNS included, goes through the Proton WireGuard
  tunnel (resolver 10.2.0.1, fallbacks 1.1.1.1/9.9.9.9 also via the tunnel),
  so fallback resolvers cannot help. Errors stopped after proton.conf's
  routing rules were rewritten on 2026-09-26 (system-setup, not this repo).
  An empty live window only skips that regen's HN score refresh; ranking
  reads stories from the DB, so the deck keeps its HN rows.
- Follow-ups (2026-09-27): LWN, lobste.rs, r/Compilers, huyenchip dropped.
  Regen now fetches article text for up to 30 RSS snippet rows per run
  (1167 of 1669 recent rows had none); the first run got 30/30. Per-source
  report: `scripts/source_yield_report.py --since YYYY-MM-DD`.
- Backfill 2026-09-27 (`scripts/backfill_rss_articles.py`, configured feeds
  only): article text for 464/478 RSS rows (plus 185 before the
  configured-feeds filter, mostly Tildes).
- LessWrong: upvote rate is flat across karma (<50: 15%, 50-99: 24%,
  100-199: 14%, 200+: 18%; n=16-39), so the karmaThreshold=20 feed stays.
  thezvi.substack.com shares no titles with LessWrong.
- Legacy source labels: 272 pre-June rows (tildes, digg, bare `rss`,
  reddit_*, ...) had no source category, or RSS instead of Reddit, in the
  ranker's one-hot. Relabelled with `scripts/relabel_legacy_sources.py`
  after an 8-fold heldout-feedback eval on a DB copy: composite 0.669 ->
  0.675 (MAP 0.484 -> 0.491, NDCG@40 0.527 -> 0.548, top-40 downvotes
  5.9% -> 5.3%; 5/8 folds up), i.e. no drop. Rollback list (id, old source):
  VPS `~/hn-rewrite/relabel_rollback_20260927.tsv`; reports
  `~/.local/state/hn-rerank-eval/relabel-*-20260927.json` on the VPS.
- Feed picks by user 1's past votes per URL domain (HN stories; overall
  up rate 32%): AI lab announcements lead (blog.google 26/33 up,
  anthropic.com 29/46, openai.com 29/49, mistral.ai 5/8). Their feeds were
  added, then removed at the user's request: no AI provider blogs. The Register, added the day before, had 4 up / 14 down of
  28 and was dropped; arstechnica.com (6/22) was not added. Query: feedback
  joined to stories, grouped by URL host.
- Same scoring over every configured feed (its own stories, plus HN
  stories from its host): dropped r/ChatGPTCoding (11 votes, 18% up, 36%
  down), blog.cloudflare.com (0 up of 3 own + 4 HN), github.blog (0/4 HN),
  pluralistic.net (0/5 own, 3 down), danluu.com (0/5 HN). Borderline, kept:
  r/transit (27% up / 36% down), r/MachineLearning (34% / 38%).
- No AI provider blogs (user preference, 2026-09-27): deepmind.google,
  research.google and huggingface.co feeds dropped too.
- Independent sites user 1 upvotes on HN, added 2026-09-27: dynomight.net
  (3/5 up), borretti.me (3/4), martinalderson.com (3/4), vickiboykis.com
  (3/4), e360.yale.edu (3/5), sciencedaily.com (5/9). Small samples; judge
  with `source_yield_report.py` after a few weeks. Skipped: newyorker
  (paywall), macrumors (3 down of 7), seangoedecke (2 down of 6).
- Reddit 429s ~150/day: mostly one retry, circuit opened once in 3 days,
  no feed failed. Cause: each regen refreshes every subreddit, and votes
  trigger a regen about every 20 minutes; only ~4 of 21 weekly-top feeds
  change per refresh.

Query: stories x interaction_events (impression) x feedback, grouped by
`stories.source`.

## Ranking-quality study — 2026-09-25

Setup: `eval_ranker_variants.py --candidate-pool heldout-feedback`, user 1
(5,189 votes: 1,707 up / 1,813 neutral / 1,669 down), temporal expanding
folds over the development 80%; newest 20% reserved. Composite = mean of
AUC(up vs rest), MAP, NDCG@12, NDCG@40, 1 − downvote share of top 40.
Reports: `~/.local/state/hn-rerank-eval/*-20260925.json` (private).

- Baselines (3 folds): production AUC 0.697 / MAP 0.428; up−down centroid
  0.715 / 0.424; random 0.52 / 0.27; HN gravity 0.39 / 0.22 (worse than
  random: upvotes lean away from high-points stories). Shuffled-label
  controls sit at AUC ≈ 0.50 for every scorer (5 seeds), incl. new variants.
- Plain logistic regression beat production on dev (AUC 0.78 vs 0.70) but
  the pre-declared confirmation on the newest 20% tied/lost (AUC 0.739 vs
  0.737, NDCG@12 0.645 vs 0.720, top-40 downvotes 12.5% vs 2.5%). Not adopted.
- Hill climb (8 folds, composite; production 0.669): SVM C 0.1→4 with
  γ 0.05 → 0.705 (8/8 folds); + percentile-rank blend with
  logreg P(up)−P(down) → 0.714. Flat/negative: knn_k, positive_cluster_k,
  training dedup, P(up)−P(down) for the SVM, kNN in the blend, HN
  points/comments features (`engagement_features_enabled`, opt-in, 0.635 at
  C=0.1). Confirmation second look (already used once, so contaminated):
  blend 0.717 vs production 0.715 — within noise.
- Embeddings (replay-only `--replay-embeddings`, 512 tokens, blend
  composite): mxbai-xsmall@4096 0.714, mxbai-xsmall@512 0.732, bge-base-
  en-v1.5@512 0.743 (C=4, γ=0.025, logreg weight 0.6; neighbours
  0.733–0.741). Shorter context does not hurt; a stronger model helps.
  Concatenated bge-base+mxbai@512 (each /√2): 0.752–0.754, 8/8 folds, best
  on every metric (AUC 0.804, MAP 0.58, NDCG@12 0.77, top-40 downvotes 2.2%);
  arctic-embed-m-v1.5@512 alone 0.747, arctic+bge+mxbai 0.746, nomic-embed-
  text-v1.5@512 ("classification: " prefix) 0.744; stored@4096+bge 0.723. Stronger
  embeddings cluster at 0.745–0.754 (differences within fold noise).
- Short context / titles (blend, dev): mxbai@128 0.699, @256 0.719, @512
  0.732, @4096 0.714; bge title-only 0.681, title+body concat 0.713 < body
  0.743. 512 tokens of body is the sweet spot; titles alone lose signal.
- Confirmation (newest 20%, third look, so contaminated; 3 blocks, ~100
  upvotes each): production 0.715 composite; blend on stored embeddings
  0.717; bge+mxbai blend 0.678 and arctic blend 0.662 despite AUC
  0.766/0.765 vs 0.737. Top-of-list metrics swing 0.18–0.82 per block
  (NDCG@12) and top-40 downvote gaps are 1–3 cards, so the newest block
  cannot separate candidates at the top; whole-list AUC favours the new
  embeddings in the two later blocks.
- Hand orderings (`calibrate_rankings.py`, stored mxbai embeddings,
  challenger = C=4/γ=0.05 + logreg weight 0.4; 8 batches of 5 stories on
  which the two rankers order every pair differently): production agrees
  with the user on 43/80 pairs (54%), challenger 37/80; batches 5–3. First
  3 batches 20–10 for production, next 5 batches 23–27: noise. Every
  ordered story sat in both rankers' top 5%, so they differ only in fine
  order at the top. Decision: keep production; the offline gain does not
  show up in the user's own judgement. Clean holdout: votes after
  2026-09-25.
- Score correction from the orderings (`scripts/fit_score_adjustment.py`:
  production logit + weighted points/comments/age/HN-source/challenger,
  pairwise logistic, leave-one-batch-out, C 0.1–1): production alone 43/80;
  +points 43/80; every other set worse (23–38/80, +all 29/80). Fitted
  weights are tiny and flip between folds: 8 batches carry no usable
  correction. Revisit only with several dozen batches.
- Tooling: `scripts/summarize_eval_report.py` (composite + fold wins),
  `scripts/encode_replay_embeddings.py`, `scripts/calibrate_rankings.py`
  (hand-order 5 disputed stories; pairwise agreement per ranker).

## TUI review and narrow-pane evidence (2026-09-24)

- `--server` bug: the argparse default made `explicit_server` always set,
  contradicting the WORKLOG intent ("mismatch guard only for explicit
  `--server`"). No effect on the user's own profile, which already uses
  `DEFAULT_SERVER`.
- Live 51-column pane (`work:3.2`): the hints `Static` (`width: auto`)
  consumed the whole footer, leaving `#status` (`1fr`) at 0 cells. Status and
  error messages were never visible in narrow mode.
- Half-box frame: `#reading-pane.has-story:focus-within` (id + class +
  pseudo) outranks `.narrow #reading-pane` (id + class), so its left border
  leaked into narrow mode.
- Test time is spread across Textual pilot tests with real `pilot.pause`
  calls; no single slow test (max 3.5 s). xdist `-n 4`: 90 s → 25 s.
- Screenshot rig: `/tmp/tui-shots/shots.py` (EditorialServer, no profile
  access). Pilot `save_screenshot` produces SVG; `google-chrome --headless=new
  --screenshot --window-size=<viewBox>` produces PNG.
- `hn` is an alias to the editable `clients/tui/.venv/bin/hn-rerank`, so the
  checkout is live without a reinstall.
- The earlier one-off TUI flake did not recur across roughly 8 full-suite
  runs today.

## TUI focus-visibility evidence (2026-09-24)

- Screenshot rig: `clients/tui` pilot harness (`export_screenshot` → SVG) +
  `google-chrome --headless --screenshot` → PNG; outputs in `/tmp/tui_*.png`
  (narrow/wide × headlines/sort/summary focus). Rig script removed after
  use; recreate from WORKLOG recipe if needed.
- Key results: orange pane frames only on focus; amber `#5A3A12` selected
  row (full `#FF914D` washed out metadata, `#3D2A17` too faint);
  `Select:focus` never fires (focus lands on inner `SelectCurrent`) — fixed
  with `:focus-within`; `outline` spills on 1-row widgets — permanent
  invisible border recolored on focus instead; `SelectCurrent` needs
  explicit `height: 3` once bordered (border-box squeezes text otherwise).
- `wmctrl -x -a Navigator.firefox` verified live for focus pull.
- Flaky TUI test observed once (fail → pass on rerun, same code); test name
  not captured — rerun full TUI suite before committing the UI batch.

## Client-side heavy work (2026-09-23)

- Compute is not the bottleneck anywhere: local ONNX embed of 68 stories =
  0.18s (3ms/story) on 8 CPU cores; server (4 cores) likewise trivial.
  Ranking/SVM cost is rounding error.
- Real server "heavy" = LLM TLDR API calls ($/rate-limit bound, Groq
  default; ~155 tldr journal lines/hr) + network I/O (CH/LW/Reddit).
  Feedback DB must stay server-side (multi-device + backup).
- Client: 8 cores/11GB, no GPU, ollama binary present but no daemon/models.
  A 3-8B q4 local model would fit RAM at ~10-30 tok/s (5-15s/summary),
  quality below Groq-70B/Mistral.
- Recommendation: (1) client-side TLDR via BYO Groq key with server
  fallback — removes all server quota pressure, trivial HTTP reuse;
  (2) local TUI disk cache for summaries; (3) ollama only as offline
  fallback experiment. Do NOT move ranking/embeddings client-side:
  no measurable gain, splits model/feedback source of truth.

## LessWrong score attribution — snapshot counterfactual

The primary profile's feed had 10 LessWrong stories at Recent Recommended
positions 1–10. An offline SQLite backup (private on VPS) contained 114 eligible
LessWrong candidates. `scripts/diagnose_source_scores.py` held training rows and
candidate embeddings fixed while altering only candidate source/length metadata:

| Candidate metadata | LW median score | Median paired change |
|---|---:|---:|
| Actual | 0.9587 | 0 |
| Source category changed to HN | 0.2892 | -0.4544 |
| Text length changed to HN median | 0.9704 | +0.0115 |
| Both changed | 0.9360 | -0.0193 |

HN median text length was 3,520 characters; LW median was 9,827.5. This exposes
strong nonlinear source/length interactions, not a client sorting bug or simply
an additive LessWrong boost. Source features group RSS together, not LessWrong
alone. Single-feature ablations can be out-of-distribution; these are model
sensitivities, not causal evidence of user preference or quality probabilities.
Do not tune against the consumed confirmation set. No ranker changes made.

## LessWrong concentration — model-driver attribution (snapshot-only)

`scripts/diagnose_rank_drivers.py` reuses the production-trained SVM/scaler
via the model cache (reconstruction fidelity: top-100 overlap 100/100, rank
correlation 1.0) on a disposable read-only snapshot for user 1.

- Tier math: with 5003 feedback (1671 up / 1569 down), alpha_2 = alpha_3 = 1,
  so final scores are 100% SVM decision. HN gravity and centroid similarity
  have exactly zero weight for this profile.
- Raw SVM margins: LW median +2.15 vs HN/other-RSS median -0.21.
- Neutralizing all 10 meta features collapses the LW-vs-HN gap (+2.36 → +0.01)
  and sinks mean LW rank 1758 → 4688. Zeroing all 384 embedding dims barely
  moves it (gap 2.35, rank 1859). The advantage lives in meta, not raw text.
- No single meta column explains it (per-column drops shift the gap only
  ~0.03–0.16): the RBF SVM learned a joint pattern, not one dominant feature.
- Training revealed preference matches: RSS up-rate 43.8% (highest), HN-live
  30.6% (lowest). Within RSS, upvoted items are ~3.6x longer (median 6033 vs
  1676 chars); HN shows no length split. Decided LW votes are 71% up (25/35).
- LW candidates combine all three learned signals: is_rss=1, very long
  (median 9827 chars), high similarity to upvoted content (0.70 vs 0.48–0.55).

Conclusion: the concentration is faithful personalization of recorded votes,
not a client bug or an additive source boost (the SVM cannot even distinguish
LW from other RSS — same one-hot bucket). Concentration rotates among RSS
sources as candidate sets change. Any diversity intervention is a product
decision requiring explicit authorization and an evidence plan; the
confirmation set is consumed and must not be tuned against.

## Authorized duplicate Import AI vote cleanup — 2026-09-22

- User explicitly chose cleanup of duplicate feedback records. Scope limited
  to the three previously identified Import AI 458/459/460 pairs for user 1.
- Deleted only older feedback rows for story IDs -9281630763986,
  -78711871060061, -129256259946967. Retained newer upvotes on
  -623350225, -851489639, -1662778741. Verified exact matching URL/action,
  newer timestamp, and no concurrent changes since backup before deletion.
- Full SQLite/WAL-consistent backup (quick_check ok), plus exact affected-row
  manifest, retained privately on VPS under
  `/home/dev/hn-rewrite/shared/feedback-cleanup-backups/20260922T201813Z/`.
  Files: `before.sqlite`, `affected-votes.json`. Restore individual rows if
  required, not the whole database over later feedback.
- Transaction verification: total feedback decreased by exactly 3; story
  count unchanged; all six article rows and all retained votes intact.
  Read-only post-check: jack-clark.net has 13 upvotes, no neutral/downvotes.
- Service restarted to clear cached profile/model state; active, dashboard
  200, bounded journal scan clean. Training-dedup experiment remains off.
  Earlier evaluation artifacts describe the pre-cleanup feedback snapshot.

## Reserved training-dedup confirmation — 2026-09-22

- Ran only production vs deduplicated, no tuning, isolated nice-19 single
  CPU-thread job. `--candidate-pool heldout-feedback --confirmation --folds 3
  --variants deduplicated`. Verified frozen configuration and confirmation
  boundary match the preceding development run. 332/332/331 held-out items,
  all retained after normalized-URL group isolation, 83/98/120 positives.
- Production/dedup means: NDCG@10 0.76029/0.71383; NDCG@40
  0.56584/0.54434; MAP 0.49923/0.49562. Per-fold NDCG@10:
  0.6620/0.5837, 0.8390/0.7779, 0.7799/0.7799. The development
  top-10 gain did not replicate. Keep dedup disabled; no production changes.
- This remains retrospective judged-only discrimination, not live retrieval
  quality. Three folds are not a significance guarantee, but these results
  do not justify rollout. The reserved period has now been used and must
  not be described as untouched or repeatedly tuned against.
- Report: `/home/d/.local/state/hn-rerank-eval/dedup-confirmation.json`;
  VPS `/tmp/hn-publication-eval/confirmation.json`. No eval job left running.

## Commit and deployment verification — 2026-09-22

- `f03c34e` pushed and deployed to VPS. Only pre-deploy remote change was
  server.py, byte-identical to the committed v12 fix (SHA-256
  `91fa1931ee4963ded153317f25ff8111e425d6564513d10b6c12a57449c0e510`).
  Preserved via named stash `pre-f03c34e-deploy-identical-detail-v12`; clean
  fast-forward then service restart. No stash dropped, no feedback modified.
- Dashboard HTTP 200; Import AI 473 returned cached v12 output (1030 chars);
  uncached story -2028198065 generated 505 chars via Muse Spark in ~13s.
  Both ok, neither stale/retryable. Bounded journal scans free of
  ERROR/Traceback/Exception. Deployed experiment flags both false.
- Restarted existing TUI pane work:1.1, default profile hash unchanged. SQL
  read-only verification found new ledger event: impression, story 49803863,
  position 0, recommended/recent, ranker_arm tui_observed. This verifies the
  real client-to-server insertion path, not just a mocked endpoint.
- Exact staged tree passed 795 backend tests; full client suite 63 passed,
  one Windows-only skip. Deferred embedding code and tests were excluded
  from staging/deployment and preserved locally. No live ranking experiment
  enabled. Documentation-only wrap-up follows the code commit.

## Five-seed controls and training deduplication — 2026-09-22

- User narrowed scope to eval controls + training deduplication only. No
  deployment. Deferred section-embedding and TUI-impression WIP preserved.
- Completed isolated single-thread/nice-19 VPS replay: 3 development folds,
  five independent shuffled-label seeds. Four already-started variants ran;
  no further publication tuning. Expected shuffled NDCG@10 averaged 0.3402.
  Production shuffled mean 0.3352 (seed means 0.2766–0.4024); publication
  0.3545 (0.2613–0.4498); dedup 0.3211 (0.2829–0.3863); combined 0.3725
  (0.2604–0.4822). Prior one-seed elevation is not consistently reproduced;
  five seeds do not prove absence of leakage. These are descriptive seed
  means, not independent-fold confidence intervals.
- Real production/dedup: NDCG@10 0.80537/0.85016, NDCG@40
  0.61111/0.61219, MAP 0.46820/0.46637. Dedup per-fold NDCG@10:
  0.7530/0.7975/1.0000 versus 0.8205/0.7949/0.8007. Gain concentrated in
  fold 3; mixed outcomes do not justify claiming a general improvement.
- Dedup uses latest vote per production-normalized URL (ID tie-break), only
  in training memory; original feedback rows remain intact. Disabled by
  default. Publication canonical URLs now also use production normalization.
- Artifact: `/home/d/.local/state/hn-rerank-eval/publication-multiseed.json`
  (VPS `/tmp/hn-publication-eval/multiseed.json`, report schema 4). Confirmation
  period untouched. Next scoped step: predeclared production-vs-dedup
  confirmation evaluation, no hyperparameter search.

## Group-isolated judged-feedback replay — 2026-09-22

- Added `--candidate-pool heldout-feedback` to retain past rated stories,
  with all three labels; current-pool evaluation remains a separate mode.
  Shared production URL normalization excludes training cross-posts and
  deduplicates test/candidates. Effective test counts: 657/661/662 from
  664/663/663 (ten overlapping/repeated test rows removed). All replay
  candidates judged; average 175 eligible positives per fold.
- Low-priority, single-thread isolated VPS run, 3 development folds plus
  shuffled-label control. No production changes. Replay report:
  `/home/d/.local/state/hn-rerank-eval/publication-replay.json` locally and
  `/tmp/hn-publication-eval/replay.json` on VPS (schema 2 artifact immediately
  before the new report contract was versioned to 3).
- Production vs affinity: NDCG@10 0.80537 / 0.80493;
  NDCG@40 0.61111 / 0.61074; MAP 0.46820 / 0.46231.
  No demonstrated benefit from enabling affinity. High levels reflect the
  judged-only, exposure-selected pool and cannot be compared to live-pool
  NDCG (~0.03). Stored content is current, not historical.
- Shuffled control: NDCG@10 0.27660 / 0.43605;
  NDCG@40 0.32250 / 0.41762; MAP 0.32604 / 0.35010.
  Affinity control is elevated: one seed/three folds does NOT establish
  leakage absence. Investigate with multiple permutations and null baselines
  before treating small differences as meaningful. Latest confirmation data
  remains untested. Semantic duplicates beyond normalized URL remain possible.
- New metrics include @10, coverage warnings, group-isolation counts; SVM
  Brier reporting disabled. See docs/RANKER_EVALUATION.md for mode semantics.

## Publication-affinity initial evaluation — 2026-09-22

- Ran production vs `publication_affinity` on a read-only SQLite backup in
  `/tmp/hn-publication-eval` on VPS; production source/config unchanged. One
  thread, nice 19, cached embeddings only. Command: `uv run python
  scripts/eval_ranker_variants.py --config eval.toml --user-id 1 --variants
  publication_affinity --folds 3 --output result.json` (via production venv,
  isolated source). 12,671 candidates, 3,981 development votes; latest 20%
  timestamp groups reserved, no confirmation or label-shuffle run.
- Mean raw NDCG@12: 0.01826 → 0; NDCG@40: 0.03135 → 0;
  MAP: 0.02133 → 0.01321. No evidence for enabling this implementation.
- Major limits: only 85/20/2 eligible positive judgments across folds;
  current candidate pool contains only 314/1664 saved positive stories.
  Recommended Recent mixed had ZERO judged cards across development folds,
  so this does not measure current newsletter recommendation quality.
- Separate exact normalized URL overlap audit found 3/2/0 test stories with
  same-article training history. Not yet a duplicate-group-clean evaluation.
  Identity excludes ambiguous shared hosts, so this is a lower-bound audit.
  Snapshot retrieval is retrospective, not historical exposure evaluation.
- Initial scoring finished but report writing failed because archived source
  had no Git metadata. Initialized an isolated source-only Git snapshot and
  reran successfully. Report: VPS `/tmp/hn-publication-eval/result.json`;
  local `/home/d/.local/state/hn-rerank-eval/publication-first.json`.
  Private report stays outside Git. Treat affinity Brier output as invalid:
  SVM decision softmax is uncalibrated, despite evaluator's generic label.
- Decision: keep feature disabled. Improve group isolation and held-out
  candidate coverage before tuning; retain the current production ranker.

## Import AI 473 TLDR drops the tail of a long newsletter — 2026-09-22

- Story `Import AI 473` (jack-clark.net, VPS `stories.id = -1607225291`) headlined
  three topics: US superintelligence strategy, brain chimeras, machine
  hermeneutics. The cached TLDR (824 chars, 4 bullets) covers the RAND strategy
  plus one chimera bullet; machine hermeneutics is absent entirely.
- Not a fetch bug for this story: stored `article_body` is 24,067 chars (under
  the 30k `ARTICLE_BODY_CHAR_LIMIT`) and contains all three topics (RAND @ch15,
  chimera @ch8150, hermeneutics @ch21382). The full body goes into the prompt.
- Structural cause is prompt/budget: `_section_budget()` caps output at
  "3-4 bullets, max 90 words" for ANY input over 5k chars, and
  `prompts/article_v4.txt` says "Summarize the article" with no
  cover-each-section instruction — so the model writes a lead-biased summary
  and the section starting 88% into the text is dropped.
- Caveats: jack-clark.net now serves a JS browser-check to curl, so the source
  page could not be re-verified; the stored text ends with a personal-dream
  coda that reads like a natural ending. The TLDR's freeform "### Summary"
  heading was model-chosen (freeform headings allowed since prompt v7).
- Fix needs a product call: `afafefc` deliberately targets one-screen summaries,
  which conflicts with covering every section of a 24k-char newsletter. Any
  prompt change bumps `TLDR_PROMPT_VERSION` and invalidates the whole
  `tldr_cache` — plan around the 120/hr uncached limit and the Mistral $10 cap.
- Read-only investigation: VPS SELECTs plus one curl; no code or data changed.

## Terminal reader — read mode gated on overflow — 2026-09-20

- Article + Discussion fit one screen for most stories, so the read mode is
  offered only where it is needed: Enter is enabled — and the footer hint
  shown — only while the summary overflows its pane. Enter expands the summary
  (full height below 100 columns; focus plus `j/k` scrolling above) and Escape
  leaves it. Fitting summaries keep the plain layout with Enter disabled and
  no hint. Read mode is frozen while active so the expanded pane cannot flip
  the state, and the overflow check re-runs after every render and resize.
- Verified: client 40 passed / 1 Windows-only skip; standalone copy with fresh
  deps (ruff 0.16.8, ty 0.0.82) clean; in-tree ruff/format/ty clean; narrow
  80×30 render inspected.

## Terminal reader / origin/main merge — 2026-09-20

- Merged the 98-commit backend line into the terminal-reader branch (`354acd6`).
  Resolutions: backend files and tests take `origin/main`; `clients/tui/**`
  keeps the branch side (the newer feed models the server imports); WORKLOG
  keeps origin/main's full history plus the TUI entries and drops a stray
  committed `>>>>>>> theirs` marker; ROADMAP restores §6; ARCHITECTURE adds the
  client package section; AGENTS.md stays origin/main's long form.
- `uv.lock` then regenerated to include the `clients/tui` workspace member
  (`6a0dc99`); the origin/main lock predated the member.
- Verified: backend 769 passed with `HN_ONNX_MODEL_DIR`; client 38 passed /
  1 Windows-only skip; Ruff and ty clean.
- Open follow-ups (config path pinning, stale env-var docs, superseded WORKLOG
  vacuum and AGENTS slimming, stale remote branch, VPS pull + restart) are
  listed in [STATUS.md](STATUS.md).
- Rollback: `backup/tui-pre-merge` marks the pre-merge tip; the merge commit's
  first parent is the terminal-reader line and the second is `origin/main`
  `97e1e25`.

## Editorial terminal client — 2026-09-20 review

- Live review against the VPS deployment (throwaway profile) found and fixed
  four defects: selector focus could strand every keybinding below 100 columns;
  Escape out of help left the Shortcuts text in the reading pane; a feed
  refresh with an unchanged selection left the heading on stale points and
  comments; and `text-align` never overrode Textual's centered H1 content
  alignment. The footer now labels the vote keys (`1 up · 2 neutral · 3 down`)
  and Enter toggled back to headlines (read mode is now gated on overflow).
- Article + Discussion now fit one screen. Companion server work landed on
  `origin/main` as `detail-v9`: section budgets target 45/70/90 words and a
  deterministic cap keeps at most four bullets and one `####` heading per
  section. The reader renders Markdown blocks without per-block margins; the
  Textual pilot measured 8/8 sampled recommended stories inside the 27-line
  reading pane at 145×38 (4/8 before).
- Verified: client suite 38 passed / 1 Windows-only skip; Ruff and ty clean;
  live terminal sessions at 145×38 and 80×30 exercised filters, reading,
  voting/undo/neutral, help, empty and failure states. New tests cover the
  help exit, refreshed-heading sync, footer vote labels, Enter toggle,
  heading alignment and narrow-layout selector escape.

## Editorial terminal client — 2026-09-17 handoff refresh

- Current state: full visual polish is implemented on `feat/terminal-client` in
  the working tree, uncommitted, on top of the reading-focus cue: selection
  contrast, themed scrollbars, docked footer with per-filter counts and `✗`
  errors, age guard, headed empty/error notices (`show_failure()`/`last_error`
  with recovery), setup card, 100-col reading cap with headline hairline, and
  vote toast. ROADMAP §6 polish is fully done.
- Agent-tested: client suite 32 passed / 1 Windows-only skip; Ruff and ty clean;
  offline SVG renders inspected for populated/empty/error/setup states.
- Remaining: native terminal visual confirmation and credential-dependent PyPI
  publication.

## Editorial terminal client — 2026-09-13

- Implemented: charcoal/ivory theme, restrained orange tabs/focus/selection,
  explicit `>` marker, domain/points/comments/age metadata, summary heading,
  Markdown spacing, thin pane divider and context-sensitive shortcut footer.
  Setup separates import and creation. Existing credential/API code is unchanged.
- Agent-tested: 25 client tests passed / 1 Windows-only skip; backend 541 passed /
  1 skipped. Ruff and ty clean. Rebuilt wheel version and headless setup startup
  passed outside the checkout. New tests cover 60/80/100/140 columns, filters,
  focus, preserved reading scroll and selection, setup validation and empty/errors.
- Visual evidence: offline SVG renders under /tmp/hn-editorial-*.svg inspected
  for populated/code, empty/error and setup states, including a long headline.
  NO_COLOR=1 is set in the agent shell; color preview explicitly unsets it.
  OptionList vertical component padding clips metadata in Textual 8, so options
  use horizontal padding and a literal marker instead of a decorative border.
- Preview: `env -u NO_COLOR TERM=xterm-256color COLORTERM=truecolor uv run python
  -m clients.tui.tests.preview --headless`; omit --headless for a synthetic terminal
  exercise. No production profile or server is used. Graphical windows launched,
  but a successful capture of the actual preview window was not obtained.
- Resolved: guard fix is included in the pushed branch; latest `tui.yml` run
  (2026-09-14, `34805118943`) passed. Native terminal visual confirmation and
  credential-dependent PyPI publication remain pending.
  Earlier CI below covers the old revision.
- Rollback: editorial work and the setup guard are committed on
  `origin/feat/terminal-client`; revert with `git revert` if needed.
  Existing unrelated untracked files were preserved.
  Wheel/sdist can be rebuilt from the previous source if needed.
- CI 2026-09-13: run `34739788455` failed all OS on ruff 0.16 I001 import order
  (local ruff was 0.15.17 and passed); fixed in `45ca54f`. Rerun `34739910637`:
  ubuntu/macos green, Windows red on
  `test_setup.py::test_import_validates_then_persists_and_relaunches` —
  `on_option_list_option_highlighted` queried `#headlines` while the Setup
  screen was active (straggler highlight during teardown). Guarded in `763cc20`
  (`setting_up`/existence early return); local client suite green.

## Laptop project and package — 2026-09-13

- Agent-tested: checkout moved from `/home/d/hn-rerank-v2` to `/home/d/code/hn-rerank`;
  one worktree, no other cwd users at move time, original untracked `.opencode/`,
  `.playwright-mcp/`, `docs/MANUAL_TESTING.md` preserved. GitHub name remains
  `djv/hn-rerank-v2`. uv console scripts were reinstalled and verified at the new path.
- Agent-tested: `HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model uv run
  pytest tests/ -n 4`: 541 passed, 1 skipped. Real test model copied from the VPS;
  no production DB copied or used by development tests. Client: 16 headless tests
  passed, 1 Windows-only test skipped locally; Ruff and ty clean. A laptop PTY exercised navigation, feedback, undo,
  reading/help and clean exit at 120 columns; headless tests also cover 80 columns.
- Agent-tested: Hatchling wheel and sdist in `dist/`; installed wheel launched via
  `uvx --from` from `/tmp`. Isolated headless startup and config round trip passed;
  numpy, sklearn, ONNX Runtime and Flask absent. Default runtime dependencies are
  Textual, HTTPX, platformdirs and their transitive dependencies.
- Agent-tested: source revision `12f7b08` on `feat/terminal-client` passed Linux,
  macOS and Windows CI, including standalone build/install, headless startup and
  Windows owner-only credential ACL verification:
  https://github.com/djv/hn-rerank-v2/actions/runs/34738793586
- Configured and read back: updated only the project trust path in
  `/home/d/.codex/config.toml`, preserving all other TOML values. Private rollback
  copy: `/home/d/.local/state/hn-rerank-rename/codex-config-before.toml`.
  No active session restarted. Generated uv environment has no old-path matches.
- Publication pending: PyPI name lookup returned 404, which does not reserve the
  name. No local publishing token or OIDC identity configured; requested account
  setup through the question popup. No package uploaded and `uvx hn-rerank` from
  the public index is not yet verified. See [release instructions](docs/TUI_RELEASE.md).

## VPS feed deployment — 2026-09-13

- Agent-tested: live service `hn_rewrite.service` runs in `/home/dev/hn-rewrite/main`.
  Its baseline was `4ec6bc6`, ahead of its origin and with uncommitted database,
  ranking, enrichment, tests and documentation changes. Those changes were preserved.
- Agent-tested: ported only feed rendering/API/models and the new API test, using
  source-hash checks and a bounded remote flock. No other agent process had that cwd;
  the flock cannot arbitrate controllers that do not participate. Server policy,
  model configuration, deployment directory and production schema were untouched.
- The newer VPS Explore filter shuffles; the port preserves shuffled feed orders.
  Its stale HTML helper also updates latest-version metadata and now preserves the
  feed attachment. [Exact scoped patch](docs/TUI_VPS_PATCH.patch).
- Agent-tested: copied VPS source was tested on the laptop with temporary databases:
  220 API/server tests passed. VPS full suite before/after: 767/768 passed;
  Ruff and ty passed after deployment. Service restarted successfully.
- Agent-tested: live HTTPS `/hn/` integration created a dedicated profile, returned
  68 stories and a 3592-character summary, accepted an upvote, reached its ranking
  target with the story excluded, then cleared the vote and verified zero feedback.
  Importing the same profile preserved identity. The private test credential is in
  `/home/d/.local/state/hn-rerank-test/profile.json`; never commit its contents.
  A bounded post-restart journal scan found no ERROR/Traceback/Exception lines.
- Rollback: `/home/dev/hn-rewrite/shared/deploy-backups/20260913-tui-feed/` contains
  original `server.py`, `pipeline/render.py` and hash manifest. Recheck concurrent
  changes before restoring those two files, then restart only `hn_rewrite.service`.
  The new dependency-free model/test files may remain inert on rollback. No DB
  rollback is needed. Laptop rename rollback is in the release instructions.

## 2026-09-24 reader-plan coverage inspection (read-only, VPS DB)

Replicated docs/source-review.md 30-day content check: asterisk 0/4,
Hugging Face 0/23, OCaml.org 3/27, Lobsters 16/123 stories with >=300 chars
self_text/article_body or any top_comments. All-time fully-empty counts are
small (HF 12/104, OCaml 6/103, Lobsters/Asterisk 0). article_fetch_failures
holds exactly one row across all four sources (a transient Lobsters
ConnectTimeout) — the fetcher is not failing on these rows, it is not
attempting them (or attempts predate the table).

Nuance for the doc's starvation theory: is_summarizable() accepts ANY
nonempty text, so HF rows with 36–62 char RSS snippets pass the filter and
reach ranking; only fully-empty rows are filtered pre-rank. Sampled HF rows
have short self_text, zero article_body, zero comments, fetchable public blog
URLs, no failure rows. Whether they miss the 50-slot article-fetch budget on
rank or another gate is unproven — no scheduling change made.

Summary spot-check (mixed set): discussion-only HN thread 49555155 renders
6 bolded bullets and explicitly notes thin/author-only signal (~190 words,
appropriately under target, no padding). Combined Reddit case verified
previously (bold in 4/4 Discussion bullets). Latent Space article-only
overshoot (414 words vs 240 target) already recorded in STATUS.md. No new
prompt tuning demonstrated; prompts untouched.

## 2026-09-24 TLDR zoom

The existing read action was gated on summary overflow, and only the narrow
CSS hid headlines. Zoom now accepts any selected story and hides headlines
at every width. The follow-up reading-width refinement centers the zoomed
pane and retains the 100-column cap; narrow terminals use their full width.
Both Enter and Escape restore list focus. Tests verify dimensions, focus, selection and
footer controls for short/long content at 51, 80 and 140 columns.

Validation: TUI 116 passed, 1 skipped; backend 821 passed using the local
`HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model` override. The initial
backend run had 18 setup errors because its default model path points at
the VPS. Ruff/format/ty clean. User independently reported fullscreen works.
No backend deployment or database changes were needed.

## 2026-09-24 Vote-to-advance footer

Adding `→ next story` exposed a cramped wide footer at 100 columns: status
shrunk to seven cells. Stacking status above shortcuts at all widths keeps
counts readable. At 51 columns the shortcuts wrap onto two lines.

## 2026-09-24 Navigation-aware prefetch

Previous default: ten cache-only lookups in sequence after foreground completion.
Missing server summaries were not generated until selection. New rolling window:
20 forward cache targets, previous three, other-sort first three; generation for
next three plus previous/other-sort neighbors, four background requests maximum.
The selected story reuses an in-flight request. Existing server cache/generation
endpoints suffice. Tests verify navigation readiness and one request per generated
selection; live provider latency can still exceed rapid navigation.

Live verification after relaunch: 44 stories in `work:3.1`. A bounded VPS
journal read showed concurrent cache hits/misses and successful generation
(2026-09-24 20:57:05 UTC, story 49525378, HTTP 200, 9280 ms total). This
confirms live cache access and generation, not guaranteed readiness for every
possible rapid navigation path. No server mutation/restart was performed.

## 2026-09-24 Pre-commit review

Reproduced and fixed one issue: at 51 columns in zoom mode, a three-line
error plus the two-line legend overran the footer by one row. A new layout
test failed with `hints.bottom=38`, `footer.bottom=37`; removing the overall
footer height cap makes all rows visible. Status remains limited to three
lines. Reviewed navigation prefetch bounds, cache-only opt-out, cancellation,
rate-limit handling, request reuse and documentation; no other blocker found.

## 2026-09-24 Post-push CI status

Head `71de3b4c3049d36d22a15b3745bfd1dfac1e3b81` is pushed to `origin/main`.
Backend CI succeeded: https://github.com/djv/hn-rerank-v2/actions/runs/36059751527
Terminal client failed: https://github.com/djv/hn-rerank-v2/actions/runs/36059751541

All terminal-client test jobs passed before lint: Windows 122 passed; Linux
and macOS 121 passed/1 skipped. Each Ruff step reports eight errors, including
I001 imports and SIM102 nested conditions (`tests/test_client.py:566`).
`.github/workflows/tui.yml` copies the package outside the backend workspace
before running `uv run ruff check src tests`; reproduce that isolation for
the next fix. No code changes or CI reruns were performed for this status save.

## 2026-09-24 Quota doubling evidence (240/hr, deployed)

Pre-restart journal showed `tldr_detail ... result=stale_fallback
reason=quota_denied` under the old 120/hr limits. After pulling `247f0c1`
to the VPS `main` worktree and restarting `hn_rewrite.service` (22:04:31
UTC, new PID 597572), host `config.toml` confirms 240/240 and zero
`quota_denied` lines appear post-restart; dashboard returns 200.

VPS layout for future deploys: `/home/dev/hn-rewrite` repo with worktrees
`main` (live, `WorkingDirectory` of `hn_rewrite.service`, runs
`uv run python server.py`) and `hn-rewrite-explore` (separate branch,
left alone). Deploy = push, `git pull --ff-only` (must be clean),
`systemctl --user restart hn_rewrite.service`, smoke test dashboard +
`api/ranking-ready`, scan journal for `quota_denied`/tracebacks.

Backend suite note: 803 passed with 18 `test_pipeline.py` errors that are
pre-existing/environmental (HF repo-id validation); verified identical
with the quota change stashed, so they do not gate config-only changes.

## 2026-09-29 TF-IDF tuning sweep (full eval, 5,189 votes, 8 folds)

Reference: `prodlr[svm_c=4;lr_weight=0.2;tfidf_weight=0.3]` (P@12 0.729, AUC 0.801). Deltas are vs that reference.

| Variant | P@12 | AUC | AUC delta |
|---|---|---|---|
| tfidf_text=title / titledom | 0.635 / 0.615 | 0.789 / 0.792 | -0.011 / -0.009 (worse; body text matters) |
| tfidf_char=1 | 0.708 | 0.803 | +0.001 (noise) |
| tfidf_c=1 / 16 | 0.719 / 0.719 | 0.799 / 0.802 | -0.002 / 0.000 |
| tfidf_weight=0.4 | 0.740 | 0.804 | +0.003 (p=0.15, noise) |
| lr_target=joint (with tfidf 0.3) | 0.708 | 0.805 | +0.004, better in 8/8 folds, P@12 -0.021 |
| half_life=60 / 120 | 0.708 / 0.719 | 0.801 / 0.802 | 0.000 |
| source_weight=0.1 | 0.677 | 0.806 | +0.004 (p=0.22); P@12 -0.052, non-HN 3.5 of top 12 |

Conclusion: nothing beats the TF-IDF 0.3 blend beyond noise. Title-only text is clearly worse. Half-life, char n-grams and C do not matter. The joint target is the only consistent AUC gain (+0.004) but costs P@12, so it is not a win. Source prior trades top-12 hits for more non-HN. Keep word 1-2 grams over title+body+domain, C default, weight 0.3.

Holdout check of the two near-winners (`/tmp/hn-eval-local/sweep1-holdout.sh`,
`s1h_table.py`; merged 151 and user 900002, cutoffs 09-26/09-27, upvotes in
the top 12 and AUC):

| Holdout | TF-IDF 0.3 | TF-IDF 0.4 | 0.3 + joint |
|---|---|---|---|
| 09-26, 151 | 10, 0.857 | 11, 0.859 | 10, 0.859 |
| 09-26, 900002 | 10, 0.890 | 11, 0.892 | 10, 0.888 |
| 09-27, 151 | 7, 0.861 | 8, 0.866 | 7, 0.870 |
| 09-27, 900002 | 7, 0.921 | 8, 0.926 | 8, 0.922 |

Weight 0.4 is ahead on all four (+1 top-12 upvote, AUC +0.002 to +0.005).
The holdouts overlap, so this is weak evidence, but it points the same way as
the full eval (+0.003 AUC, P@12 +0.010). 0.3 vs 0.4 is a coin flip within
noise. For production, 0.4 is no worse and possibly slightly better.

### Final full eval and Codex review of the TF-IDF proposal — 2026-09-29

Proposal: `prodlr[svm_c=4;lr_weight=0.2;tfidf_weight=0.3]` vs base `prodlr[svm_c=4;lr_weight=0.3]`, stored+gemma embeddings, user 1, tie-aware AUC.

| Run | P@12 base → new | AUC base → new | AUC delta |
|---|---|---|---|
| Full, 8 folds | 0.708 → 0.729 | 0.791 → 0.801 | +0.011 [+0.002,+0.019] p=0.021, 7/8 folds |
| `--confirmation` (newest 20%), 4 folds | 0.646 → 0.667 | 0.768 → 0.774 | +0.006 [-0.008,+0.019] p=0.28, 2/4 folds |

P@12 gain is two extra hits in 96 slots (not significant). Codex found no TF-IDF vocabulary/IDF leakage; verdict: ship only behind a default-off, profile-scoped flag. Its caveats: the 151 "unseen" checks overlap and informed tuning, so they are not independent replications; the model is really 50% prod rank + 20% dense LR + 30% TF-IDF (kNN weight 0); the evidence uses stored+gemma but production has stored only; TF-IDF text is title + first 5,000 chars of `text_content` (may include comments, ignores component fields when empty); the scorer has no cold/sparse-profile fallback (needs both classes) and per-profile fit cost is unmeasured; `cmp.py` pairs by position and mixes pooled and mean-of-fold AUC. Full review: `~/.local/state/hn-rerank-eval/hn-eval-local/codex-tfidf-review-out.md`.

### TF-IDF blend on stored embeddings only (what production has) — 2026-09-29

Same comparison as above, `--replay-embeddings all-stored.npz` only (no Gemma).

| Run | P@12 base → new | AUC base → new | AUC delta |
|---|---|---|---|
| Full, 8 folds | 0.698 → 0.719 | 0.774 → 0.790 | +0.016 [+0.006,+0.026] p=0.006, 7/8 folds |
| `--confirmation`, 4 folds | 0.604 → 0.667 | 0.737 → 0.751 | +0.014 [-0.003,+0.030] p=0.084, 3/4 folds |

The gain holds without Gemma, and is slightly larger (stored embeddings alone are weaker, so TF-IDF adds more). Caveats from the Codex review still apply.

### Proposal vs the actual live ranker — 2026-09-29

Correction: the "base" above (`svm_c=4` + 0.3 LR) is not live. Live is `production` with `svm_c=0.1` (config.toml), no LR or TF-IDF. Stored embeddings only, user 1:

| Run | P@12 live → proposal | AUC live → proposal | AUC delta |
|---|---|---|---|
| Full, 8 folds | 0.625 → 0.719 | 0.726 → 0.790 | +0.063 [+0.015,+0.111] p=0.017, 7/8 folds |
| `--confirmation`, 4 folds | 0.667 → 0.667 | 0.739 → 0.751 | +0.011, p=0.38, 2/4 folds |

Most of the full-eval gain over live comes from `svm_c` 0.1 → 4 plus the LR blend, not from TF-IDF alone (TF-IDF adds about +0.016 AUC on top). The newest-20% confirmation is flat, so treat the size of the gain as unconfirmed.

### Linear blend live: rank latency — 2026-09-29

Deployed `26b9474` (`svm_c = 4.0`, `linear_blend_enabled = true`), restart 17:17 UTC; server not listening until 17:18:45 (a refresh in that window gets "Connection failed"). No errors after restart. `rank_perf` for user 151 (2,845 votes, ~10k candidates):

| | rank_total_ms |
|---|---|
| Before (12:00–17:17 UTC, 8 ranks) | 4,240–12,842 |
| After, model cache miss | 29,259 (`linear_blend_fit_ms` 14,412, `linear_blend_score_ms` 7,815) |
| After, model cache hit | 21,755 (`linear_blend_score_ms` 9,238) |

Reads stay fast (`/api/feed` 8-10 ms on the VPS, ~0.5 s via Tailscale, all windows 200), since they serve the current deck while a rerank runs; the reranked deck after a vote arrives ~15-20 s later than before.

### Hashed TF-IDF (per-story cached) vs live — 2026-09-29

Production TF-IDF now hashes word 1-2 grams into 2^18 columns so a story's row is cached across retrains and ranks (first live rank: fit 14.4s, scoring 7.8s on 10k candidates). Columns seen in <2 training stories are dropped; idf and sublinear tf are fit on training votes. Same eval as above (stored embeddings, vs live `production`): full AUC 0.726 -> 0.788 (was 0.790), P@12 0.625 -> 0.708 (was 0.719); newest 20% AUC 0.739 -> 0.750 (was 0.751). Same gain within noise.
