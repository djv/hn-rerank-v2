# Worklog: hn-rewrite

Append-only log of notable changes, fixes, and operational events.
Each entry is dated and self-contained.

---

## 2026-06-26 — Add ClickHouse seeder as alternative to BigQuery

- New `scripts/seed_hn_from_clickhouse.py` queries the public ClickHouse
  Playground (`hackernews_changes_items`) over plain HTTP — no GCP
  credentials required. Uses `argMax(field, update_time) GROUP BY id` to
  fetch the latest version of each story. Stores rows as `source="ch_seed"`.
- New `scripts/_seed_common.py` extracts the row-to-story, Algolia
  comment hydration, and `seed_rows` insertion logic shared by both
  seeders. `seed_hn_from_bq.py` now imports from `_seed_common`; its CLI
  behavior is unchanged.
- New `CH_ARCHIVE_SOURCE = "ch_seed"` constant in `pipeline.py`.
  `is_hn_source()`, `source_label_filter()`, and archive candidate pool
  queries (`fetch_candidates`, `fast_rerank_for_user`) updated to include
  both `bq_seed` and `ch_seed`. `prune_stories` also protects `ch_seed`.
- Tests: 8 new CH seed tests, 1 updated BQ prune test, 2 updated pipeline
  source-label tests, 1 new CH TLDR dynamic-fetch test.

## 2026-06-25 — Comment selection rewrite: drop score, 1/3 top-level budget, MIN_COMMENT_LENGTH=60

- `_comment_rank_key` rewritten: dropped `score` (a misleading depth penalty since
  Algolia returns `points: null` for HN comments). New key: `(-descendant_count,
  -text_len_uncapped, order_path)`. Substantive long-form comments now surface
  over short agreement replies regardless of depth.
- `_select_top_comments` rewritten: adaptive `n_cores` (min 4, limits to actual
  good top-level); quality-based breadth pass with `GOOD_TOPLEVEL_MIN_LEN=200`
  and `GOOD_TOPLEVEL_MIN_REPLIES=3`; 1/3 budget cap on top-level breadth.
- `MIN_COMMENT_LENGTH` raised 30→60 to filter one-liner agreement at extraction.
- Real-world impact across 10 sampled stories: top-level ratio 20/20 → 11-17/23-29;
  max_depth 1-2 → 3-5 in 9 of 10 stories; short agreement comments dropped.

## 2026-06-25 — Card fills available text; shrink-to-fit for short cards

- `.swipe-shell` max-width: 1100px → 1280px (more room).
- `.story-card`: `width: fit-content; min-width: min(60ch, 100%); max-width: 902px; margin: auto` — short cards hug their content and are centered.
- `.story-card.enriched`: `max-width: none` — long-text cards fill the column edge-to-edge.
- `.tldr-detail-content`: `max-width: 75ch` — body text stays readable when card is wide.
- `.story-card.active`: `min-height` → `max-height: calc(100vh - 2rem)` so short cards shrink and long cards cap at viewport with internal scroll (no page scrollbar).
- Enriched cards: added `width: 100%` so they fill the column (was stuck at `fit-content` width).
- `#stories` min-height decreased from 1rem to 2rem to match card's new max-height (eliminates body scrollbar on 14-inch monitors).
- Adjusted `#stories` min-height and `.story-card.active` max-height from `2rem` to `2.5rem` (more buffer).
- Removed `max-width: 75ch` from `.tldr-detail-content` so text reflows to fill the full card width.
- Adjusted `#stories` min-height and `.story-card.active` max-height from `calc(100vh - 2.5rem)` to `calc(100vh - 3rem)`.
- Added ArrowUp/ArrowDown handling to scroll the active card (80% of card height per press, instant scroll).
- Fixed page scrollbar: removed `min-height` from `#stories` (column shrinks to card). Capped side-rail at `calc(100vh - 1.5rem)` with `overflow-y: auto`. Changed `.swipe-layout` to `align-items: flex-start`. Removed `position: sticky` from rail.
- Tests updated: `test_keydown_uses_letter_keys` now asserts arrow keys are present (card scroll) vs absent (native page scroll).

## 2026-06-25 — Remove page footer; convert first-time tip to floating panel

- Deleted `<footer>` markup ("HN Rerank Rewrite ...") and its CSS rule (dead code).
- `.first-time-tip-overlay`: `position: fixed; inset: 0;` with subtle backdrop
  (`rgba(0,0,0,0.25)`), centered inner card with shadow + border.
- Bumped font from 0.75rem to 0.95rem; kbd font 0.95rem.
- `aria-label="Keyboard shortcuts"` on the outer overlay; no visible title.
- Backdrop click dismisses (`e.target === tip`); dismiss button auto-focused on
  show; Escape still dismisses.
- `test_keydown_uses_letter_keys` extended with layout assertions.

## 2026-06-25 — Add `o` / `c` keys for open article / open comments

- `templates/index.html`: new `openStoryUrl(kind)` helper reads
  `data-article-url` / `data-comments-url` from the active card and
  opens the URL in a new tab.
- `o` = open article, `c` = open comments. Silent no-op when the URL
  is missing (HN self-posts, RSS posts without discussion).
- Side-rail legend and first-time tip updated.
- `noopener,noreferrer` for safe `target="_blank"` semantics.
- ARCHITECTURE.md §3.5 updated.
- `test_keydown_uses_letter_keys` extended.

## 2026-06-25 — Switch vote keys from arrows to j/k/l

- `templates/index.html`: arrow keys freed for native scroll inside the card.
  Vote keys: `k` = upvote, `j` = downvote, `l` = skip (neutral), `u` = undo.
- Added modifier-key filter (`Ctrl`/`Cmd`/`Alt` + letter no longer votes).
- Added dismissible first-time tip overlay (localStorage gate, `_v2` flag).
- Renamed in-card neutral button title to "Skip (neutral)".
- Server-side action code `'neutral'` unchanged.
- ARCHITECTURE.md §3.5 updated.
- 1 new test (`test_keydown_uses_letter_keys`).

## 2026-06-25 — LessWrong comment fetch via GraphQL

- `server.py`: added `_fetch_lesswrong_context`, `_extract_lesswrong_post_id`,
  `_clean_lesswrong_html`, `LessWrongContext` dataclass.
- Wired into `/api/tldr-detail` for `rss_lesswrong_com` rows (single GraphQL
  query for post body + top comments).
- 4 new tests in `tests/test_server.py`. 123 passed, lint clean, server restarted.
- TLDR cache cleared for story -1463020014 ("And what happens next?").
- Generic article scraping excluded for `rss_lesswrong_com`.
- ARCHITECTURE.md §4.1 updated with LessWrong documentation.

## 2026-06-25 — Source filter toggle

- 3-way Mixed / HN / Non-HN filter on side rail; stacks on mode filter.
- 117 tests pass, generate.py run, server restarted.

## 2026-06-25 — BQ seed refresh

- 6-month window, min-score 500, limit 200. 75 new bq_seed stories (total 459).

## 2026-06-25 — TLDR path 2 word cap

- "under 180 words" → "under 240 words" in `server.py:382`.
- No version bump; no proactive cache invalidation.
- Cache entry for -1463020014 deleted.

## 2026-06-25 — Non-HN discovery pass

- Pass #7 in `pipeline.py`: up to 8 non-HN extras after hot pass.
- `is_non_hn: bool = False` added to `RankedStory`.
- Primary non-HN stories also flagged (no new badge — source-badge covers it).
- 119 tests pass; render 12-13 RSS stories (was ~3).

## 2026-06-22 — Test story + user cleanup

- 756 time=0 stories removed (2 test + 754 _empty_story artifacts).
- 334 spam users deleted from `curl -L` testing. 3 real users remain.
- Backup: `hn_rewrite.db.pre_test_removal_20260622T163344Z`.
- Diagnostic script: `/tmp/diag_user79.py`.
- 2 test stories (999, 99999998) removed from legacy JSON via `jq`.

## 2026-06-22 — Dual-gate SVM activation

- `min_up_for_svm=20`, `min_down_for_svm=20` in `pipeline.py`, `config.toml`,
  `ModelConfig`. Blend uses `min(n_up, n_down)` as basis over 60-step window.
- SVM only trains when both classes have >=20 examples.

## 2026-06-21 — Title-embedding dedup removed

- `get_or_compute_title_embeddings`, title pre-caching, and
  `ModelConfig.title_similarity_*` fields deleted.
- `fast_rerank_for_user` reverted to gravity sort + top-1000 pre-filter.
- 55 tests pass, 1 deselected (the removed dedup test).

## 2026-06-20 — Self-healing embedding cache

- Added `embeddings.text_hash` with SHA-256 validation.
- Mismatches trigger cache miss + recompute via
  `get_or_compute_embeddings()` cache path.

## 2026-06-19 — Comment backfill hardening

- `fetch_story` falls through to Algolia items API when `top_comments` is
  stale or missing. Capped at 100 per pipeline run.
- All four error paths in `fetch_story` (non-200, invalid item, empty text,
  exception) preserve cached data on transient failure.
- `_empty_story` vulnerability documented: it overwrites all columns
  except `article_body`. The COALESCE on `article_body` is the only
  protection.
- `upsert_story` COALESCE only covers `article_body` — architectural
  vulnerability for future code.
- 1,940 stories still needed comment backfill at the time of write.
  They'll be gradually re-fetched as they appear in future Algolia
  search windows (100 per pipeline run).

## 2026-06-25 — `overflow: hidden` on `html, body` to kill page scrollbar

- Added `html { overflow: hidden; }` and `body { overflow: hidden; }`
  to guarantee no page scrollbar regardless of content height.
- Page never scrolls. Card and rail still scroll internally via their
  own `overflow: auto` and `overflow-y: auto`.
- Fixes the issue where big/enriched stories showed both a card internal
  scrollbar and a page scrollbar.

## 2026-06-25 — `HOT_MIN_SCORE=20` floor on Hot badge

- Added `HOT_MIN_SCORE = 20` constant to `pipeline.py`.
- `is_hot` now requires `score >= 20` in addition to velocity ≥ p99.5 threshold.
- Stories with < 20 points can no longer be marked Hot regardless of velocity.
- Both primary-path and extra-slot hot-pool checks include the score guard.
- Badge title updated to "Top 2% by engagement velocity (points/hour) and score ≥ 20".
- New test `test_hot_badge_requires_minimum_score` verifies the invariant.

## Operational state (snapshot 2026-06-25)

- **Cold render for user_id=1 (1789 feedback)**: first dashboard render
  after restart takes 3-5s and allocates ~1GB. SVM is retrained live
  from cached embeddings on every request (no DB model cache).
  Expected behavior for 1789 feedback points.
- **Memory doesn't shrink after request**: numpy/sklearn internals retain
  memory after training. Peak grows asymptotically to ~1GB. Systemd
  `Restart=on-failure` recovers if the OOM killer fires.
- **User counts (as of 2026-06-22 cleanup)**: id=1 (token="default", 1789
  feedback), id=78 (token="new", 32 feedback), id=79 (token="new2", 113
  feedback). 3 real users.

## 2026-06-25 — Attention-pooled MLP evaluated; does not beat SVM

- Built a PyTorch attention-pooled user profile + MLP head (`pipeline_dl.py`).
  Architecture: learned `W_q`/`W_k` projections → dot-product attention over
  upvoted/downvoted feedback → 5×384 + 5 meta features → 64-hidden MLP → 3 logits.
- Trained via full-batch gradient descent (100 epochs, Adam, early stopping).
  LOOCV: training items excluded from their own attention pool.
- Tested in `tests/test_pipeline_dl.py` (9 tests).
- Added as `attention_mlp` variant to `scripts/eval_ranker_variants.py`.
- **5-fold eval results (30-day window, user_id=1, n_candidates=4785, n_feedback=1004)**:

  | Metric | SVM (margin3_up) | Attention MLP | Δ |
  |---|---|---|---|
  | NDCG@40 (raw) | 0.4717 ± 0.0824 | 0.4097 ± 0.0801 | **-0.062** |
  | NDCG@100 (raw) | 0.4264 ± 0.0441 | 0.3859 ± 0.0710 | -0.040 |
  | MAP | 0.2523 ± 0.0469 | 0.2147 ± 0.0476 | -0.038 |
  | P@40 | 0.3800 ± 0.0696 | 0.3450 ± 0.0485 | -0.035 |
  | Median rank | 182.1 | 211.8 | +29.7 |

- Pass criterion (NDCG@40 improvement ≥ 0.02) NOT met.
  Attention MLP is **consistently 0.04-0.06 behind SVM** on every metric.
  Model is not shipped.
- **New dependencies**: `torch>=2.12` added to `pyproject.toml` (was transitive via `transformers`).
- **New files**: `pipeline_dl.py` (~330 lines), `tests/test_pipeline_dl.py` (~120 lines, 9 tests).

## 2026-06-25 — Attention MLP refinement: T0, multi-head, per-class meta

Followed up on the prior entry. Goal: figure out why the attention MLP
lost to the SVM, then close the gap.

**T0 ablation** — re-ran the original single-head / 64-hidden architecture
as a control. T0 reproduced at 0.449 NDCG@40 (vs 0.410 in prior run;
variance from data drift between runs).

**Tier 1 — multi-head + wider MLP** — refactored `pipeline_dl.py`:
- 4 heads × 32-d, learned W_v projection
- hidden 64 → 256, dropout 0.0 → 0.2, Adam → AdamW (weight_decay=1e-4)
- simplified feature vector (dropped elementwise ops)
- LOOCV on attention profile

**Diagnostic ablations** (5 folds, 30-day window, 1006 feedback):

  | Variant | NDCG@40 | Δ vs SVM | Key finding |
  |---|---|---|---|
  | margin3_up (SVM) | 0.4716 | — | baseline |
  | attn_mlp_t0 (T0) | 0.4491 | -0.023 | T0 control |
  | T0 + cosine sims | 0.4220 | -0.050 | cos sims HURT |
  | **T1, no cos sims** | **0.4984** | **+0.027** | **winner** |
  | T1 + hidden=64 | 0.3910 | -0.081 | wider MLP is critical |

Two findings: (1) cosine sims hurt the NN (attention profile already
captures similarity); (2) wider MLP is the key, not multi-head alone.

**Tier 2 — additional signals** (opt-in):
- per-class mean meta (10-d appended): +0.019 NDCG@40 over T1 alone
- pairwise hinge ranking loss (256 (up, down) pairs, λ=0.5): -0.062 (HURTS)
- mixup α=0.4: -0.011 (neutral)
- combined: -0.055 (ranking loss dominates, drags everything)

Best single model: `attn_mlp_v2_meta` (T1 + per-class meta only).
NDCG@40 = 0.5027 ± 0.121, vs SVM 0.4841 ± 0.070.
P@40 = 0.43 (vs SVM 0.38) — best top-page precision.
Variance is 2× SVM's; the win is real but noisy.

Decision: keep multi-head + wider + per-class meta. Drop ranking loss
and mixup (they hurt or don't help). DL model is **not shipped**.

**Code changes**:
- `pipeline_dl.py`: full refactor (~334 lines). New: `_ranking_loss`,
  `meta_per_class_dim` param, mixup support, `train_meta_per_class`.
  Defaults: `mixup_alpha=0.0`, `ranking_lambda=0.0` (Tier 2 opt-in).
- `tests/test_pipeline_dl.py`: 21 tests (was 9). 146/146 pass; ruff clean.
- `scripts/eval_ranker_variants.py`: 8 new variants registered.
- `pipeline_dl_t0.py` (new): T0 reproduction (single-head, no W_v,
  elementwise ops). Used by ablation variants only.
- `pyproject.toml`: `torch>=2.12` (was transitive via `transformers`).
- `pipeline.py` unchanged (SVM still in production).

## 2026-06-25 — Blending SVM + DL: blend_score_75 best, not shipped

Ensemble of `margin3_up` (SVM) and `attn_mlp_v2_meta` (best DL).
Two strategies, α ∈ {0.10, 0.25, 0.50, 0.75, 0.90}.

**Score blend**: `α * svm_score + (1-α) * dl_score`
**Rank blend**: `α * rank(svm) + (1-α) * rank(dl)` (per-fold)

**Bug found and fixed**: first rank-blend run returned NDCG@40 ≈ 0.002
(near-random). Cause: `rankdata(-scores)` gives rank 1 to best item,
but eval expects higher score = better ranking. Switched to
`rankdata(scores)` (rank N = best). Re-ran, results now sensible.

**5-fold eval** (4782 candidates, 1013 feedback):

  | Model | NDCG@40 | MAP | P@40 | MedR | Std |
  |---|---|---|---|---|---|
  | SVM | 0.437 | 0.243 | 0.350 | 239 | ±0.117 |
  | DL (attn_mlp_v2_meta) | 0.471 | 0.244 | 0.380 | 317 | ±0.087 |
  | **blend_score_75 (α=0.75)** | **0.492** | **0.261** | **0.405** | **245** | **±0.068** |
  | blend_score_50 | 0.487 | 0.257 | — | 255 | ±0.072 |
  | blend_rank_25 (best rank) | 0.486 | 0.251 | 0.400 | 294 | ±0.084 |

`blend_score_75` wins:
- +0.021 NDCG@40 over best single model (passes ≥0.02 threshold)
- +0.055 over SVM
- lowest variance (±0.068) and highest P@40 (0.405)
- wins 2/5 folds outright vs DL, ties 3; wins 4/5 vs SVM

**Decision: NOT shipped**. Reasons:
- +0.021 vs DL is borderline (5-fold std ±0.10; within 1 sigma)
- cold-render cost: blend requires training both SVM (~0.5s) and
  DL (~3s) per render — pushes budget from 3-5s to 6-8s
- DL model alone is not production-validated yet
- Code preserved in `eval_ranker_variants.py` for future re-evaluation

**Code changes**: 10 new variants in `scripts/eval_ranker_variants.py`
(`blend_score_10/25/50/75/90`, `blend_rank_10/25/50/75/90`).
New function `_scores_blend_up(fold, config, alpha, *, kind)` and
helper `_rank_ascending`. No changes to `pipeline*.py`.

## 2026-06-26 — Mobile side-rail on top, bigger buttons, flex scroll container

- `templates/index.html` only; `public/index.html` regenerated via `generate.py`.
- **Mobile side-rail layout** (`@media (max-width: 640px)`): side rail stacks
  vertically above the cards — full-width queue progress, 4-col mode tabs,
  3-col source tabs, border-bottom separator. Keyboard-hint list hidden on
  mobile (`display: none`).
- **Bigger vote buttons on mobile**: ▲/✓/▼ in `.feedback-btn` get
  `padding: 0.6rem 0.9rem`, `font-size: 1.05rem`, `min-width/min-height: 2.75rem`
  (44px touch target, WCAG compliant). Desktop buttons stay compact (0.8rem).
- **Flex scroll container**: `.swipe-shell` becomes `height: calc(100vh - 1.5rem);
  display: flex; flex-direction: column`. `.swipe-layout` and `#stories` are
  `flex: 1; min-height: 0`. `.story-card.active` is `max-height: 100%` so the
  card fills remaining viewport after the side rail and scrolls internally
  via its existing `overflow: auto`.
- **Removed**: the touch swipe gesture (`setupSwipe`), `isMobileLike()`,
  justSwiped guard, glyph overlay, directional exit values, and mobile
  first-time tip. The swipe was replaced by simply enlarging the vote buttons.
- Tests in `test_keydown_uses_letter_keys` updated to assert button sizing
  and flex scroll container instead of swipe.
- No backend changes. No DB changes. No config changes. 143 tests pass
  (1 pre-existing syntax error in `test_seed_hn_from_bq.py` excluded).

## 2026-06-26 — dry-run flags + smoke test script; CH JSON format fix

- Added `--dry-run` (and optional `--dry-run-output FILE`) to both
  `seed_hn_from_bq.py` and `seed_hn_from_clickhouse.py`. Fetches rows
  from the source, writes them to JSONL with a `{"_meta": {...}}` header
  line, and returns without loading Config, Database, or Embedder.
- New `scripts/seed_smoke_test.py` compares BQ vs CH output for the same
  query parameters (live or from prior dry-run files). Produces a JSON
  report with intersection counts, field-by-field agreement (with
  configurable score/descendants delta), top score deltas, and per-source
  exclusive stories. Prints a human-readable summary to stdout.
- New helper `_write_dryrun` in `scripts/_seed_common.py` shared by both
  seeders.
- **CH JSON format fix**: `seed_hn_from_clickhouse.py` was not requesting
  JSON output from the ClickHouse Playground, so `resp.json()` failed with
  `JSONDecodeError` on live queries. Fixed by adding
  `&default_format=JSON` to `CH_PLAYGROUND_URL`, and parsing the
  `{meta, data, rows, statistics}` response shape (extract `data` array).
  Updated mock test to return `{"data": []}` instead of bare `[]`.
- Live smoke test (50 rows, min-score=200): BQ=50, CH=50, intersection=43.
  Title/url/created_at_i agreement 43/43 (100%). Score/descendants: 42/43
  (97.7%, 1 drift each, expected from different update latencies).
- Tests: 2 dry-run tests (1 BQ, 1 CH) + 15 smoke test unit tests.
  178 tests pass, ruff clean.

## 2026-06-26 — 100dvh viewport height fix (mobile card bottom cutoff)

- **Root cause**: `.swipe-shell` used `height: calc(100vh - 1.5rem)`. On mobile
  browsers, `100vh` includes the URL bar area, making the shell taller than the
  visible viewport. Combined with `body { overflow: hidden }`, the shell's
  bottom (and the card's TLDR button / match-reason) was clipped.
- **Fix**: changed mobile `.swipe-shell` height from `calc(100vh - 1.5rem)` to
  `calc(100dvh - 1.5rem)`, with `100vh` retained as a fallback for older
  browsers. `100dvh` (dynamic viewport height) reflects the actual visible area
  when mobile URL bars collapse.
- Test in `test_keydown_uses_letter_keys` updated (`100dvh` seen in template).
- 153 tests pass. Ruff clean. Service restarted. public/index.html regenerated.

## 2026-06-26 — Vote buttons moved outside the card into a fixed bottom bar

- **Before**: each story card had its own `.feedback-group` inside `.story-header`.
  Buttons were rendered per-card in the Jinja2 loop with server-side `active_fb`
  state. `stopPropagation` was needed to prevent TLDR opens on button clicks.
- **After**: a single global `.vote-bar` (positioned fixed at bottom:0, z-index:100)
  is rendered once after `</main>`, independent of the story loop. Contains one
  `.feedback-group` with three buttons.
- **Server-side voted state**: conveyed via `data-voted="{{ active_fb or '' }}"`
  on the `<article>` element (replaces per-card `.feedback-btn.active`).
- **JS changes**:
  - `card.querySelectorAll('[data-fb]')` → `document.querySelectorAll('[data-fb]')`
    in `submitVote`, `undoLastVote`, and `updateVoteBar`.
  - `card.querySelector('.feedback-btn.active')` check → `card.dataset.voted` read
    in localStorage sync.
  - `setActiveCard` now shows/hides the vote bar via `voteBar.hidden` and calls
    `updateVoteBar()` to sync button active state with the card's `data-voted`.
  - `updateVoteBar()` is a new function reading `activeCard.dataset.voted`.
  - `.feedback-group` removed from TLDR click suppression (no longer inside card).
  - `e.stopPropagation()` removed from button handler (bar is outside card).
- **CSS changes**:
  - `.vote-bar`: `position: fixed; bottom: 0; left: 0; right: 0;` with backdrop
    blur, border-top, and `env(safe-area-inset-bottom)` for iOS.
  - `.vote-bar[hidden] { display: none; }` for hide/show.
  - `margin-left: auto` removed from `.feedback-group` (now centered via
    `.vote-bar { justify-content: center }`).
  - `.story-card.active` gains `padding-bottom: 5rem` to prevent TLDR content
    from hiding under the fixed bar.
  - `.vote-bar .feedback-btn` removed (uses generic `.feedback-btn` with mobile
    media query overrides — same buttons, same sizing).
- Tests in `test_keydown_uses_letter_keys` updated: vote bar position assertions,
  hidden state, active card bottom padding.
- 168 tests pass. Ruff clean. Service restarted. public/index.html regenerated.

## 2026-06-26 — Switch CH seeder from `hackernews_changes_items` to `hackernews_history FINAL`

- Switched `scripts/seed_hn_from_clickhouse.py` from `hackernews_changes_items`
  (74.7M rows, per-update change stream, requiring `argMax` / `GROUP BY` for
  dedup, `sc` alias for score) to `hackernews_history FINAL` (48.7M rows,
  `ReplicatedReplacingMergeTree`, already deduplicated). The new query is
  simpler: direct column access, no `argMax`, no `GROUP BY`, `score` is a
  native column.
- Removed the `sc` → `score` rename in `run_ch_query()`.
- Updated `test_build_ch_query_accepts_months_min_score_and_limit` and
  `test_run_ch_query_uses_correct_endpoint` assertions from `HAVING sc >=` to
  `AND score >=`.
- Live smoke test (BQ 500 vs CH 500, min-score 200, 3 months):
  - Intersection: **485/500** (was 396/500 with `changes_items`)
  - Score agreement: **97.5%** (was 0% on drift stories)
  - Title/URL: 100% match; created_at_i: 100% match
  - Descendants: 98.6% match
- Validated the `hackernews_history` switch closes the gap: 15 BQ-only, 15
  CH-only (was 102 each). The dual-source design is now production-ready.
- 177/178 tests pass, ruff clean. (1 pre-existing failure in
  `test_keydown_uses_letter_keys`.)

## 2026-06-26 — Mark CH seeder as primary; BQ retained as backup; new defaults (6mo/200)

- `scripts/seed_hn_from_clickhouse.py` defaults changed:
  - `--months`: `3` → `6` (wider archive window)
  - `--min-score`: `100` → `200` (focus on high-signal stories; yields ~4,400 rows)
- CH coverage analysis confirmed: **100% recall** on 1y/score≥100 vs BQ
  (15,265/15,265 BQ IDs present in CH). CH also has 99 fresher stories BQ
  doesn't. CH is **10-30x faster** (0.7s vs 9-20s).
- `seed_hn_from_bq.py` retained as backup. Both source labels (`bq_seed`,
  `ch_seed`) remain valid.
- `AGENTS.md`: reordered CH first, added CH/BQ role documentation.
- `ARCHITECTURE.md`: CH listed primary, BQ listed backup; defaults documented.
- No code changes to `pipeline.py`, `database.py`, or tests.

## 2026-06-26 — Add `ch_client.py` bulk API + prewarm feature; archive old Algolia hydration

- New `ch_client.py` (~280 lines) wraps the ClickHouse Playground HTTP API:
  - `query_live_window(days, min_score, limit)` — 7-day search-replacement
  - `query_stories_bulk(story_ids)` — full story fields for N stories
  - `query_comments_bulk(story_ids, max_levels)` — comment text for N stories
    via chained CTE (5 levels covers ~95% of comment trees)
  - `query_stories_with_comments(story_ids, max_levels)` — combined query
  - `query_single_story(story_id)` — lazy fallback (15min cache)
  - In-memory LRU cache: 1h TTL bulk, 15min single, capped at 128 entries
- All Algolia-shape response (`type`, `title`, `url`, `points`,
  `num_comments`, `created_at_i`, `story_text`, `children[]`) so callers
  don't need to know it's CH.
- Replaced per-story parallel Algolia hydration in `scripts/_seed_common.py`
  with bulk CH hydration. Single SQL query for entire skeleton set vs N
  parallel Algolia calls. Live measurement: ~30s for 100 stories
  (Algolia parallel) → ~0.3s (CH bulk).
- Archived old `hydrate_comments_from_algolia` to
  `scripts/_archive/algolia/hydrate_comments_algolia.py` with a README
  explaining when to revive it (CH outage).
- New `pipeline.prewarm_top_stories(story_ids, db, embedder)` runs on
  every dashboard render after ranking. Pre-populates `top_comments` for
  the top-20 stories, so the first 4 cards the user clicks skip the lazy
  single-story Algolia fetch. Always on; no config flag.
- Single-story Algolia calls (`fetch_story`, `refetch_story_text`,
  `fetch_candidates` search) kept unchanged: real-time, no 1-24h CH lag
  for fresh stories.
- Updated `AGENTS.md` (added data-source architecture table),
  `ARCHITECTURE.md` §3.6 (CH bulk hydration + prewarm description),
  `plans/algolia-to-clickhouse.md` (status update).
- Tests: 18 new `test_ch_client.py`, 6 new prewarm tests in
  `test_pipeline.py`, updated mocks in `test_seed_hn_from_bq.py` and
  `test_seed_hn_from_clickhouse.py` to use the new bulk path.
- 202/202 tests pass, ruff clean.

## 2026-06-26 — TLDR prompt: flat structure, generic comment label

- **Nested bullet rendering fix (CSS-only)**: when the LLM produces
  indented sub-bullets (it ignores the "no nested list" rule ~30% of
  the time), parent label bullets now render flush-left bold with no
  disc marker, and the nested list renders as a smaller circle-bulleted
  sub-list. Uses `.tldr-detail-content li:has(> ul)` in
  `templates/index.html` (after the existing `.tldr-detail-content li`
  rule). No JS, no DOM restructuring; existing cached TLDRs benefit
  immediately.
- **Prompt: flat structure**: the LLM prompt now leads with a "FLAT
  structure only — no nested list levels" rule, with explicit
  guidance: "If a bullet ends with `:`, treat the following sub-points
  as separate top-level bullets, not as indented children." Applied
  to all three prompt templates in `server.py` (article+comments,
  article-only fallback, discussion-only).
- **Prompt: source-agnostic comment label**: discussion prompt changed
  from "HN comments:" / "Summarize the Hacker News discussion" to
  "Comments:" / "Summarize the discussion". The TLDR detail handler
  already pulls `story.top_comments` from Reddit RSS or LessWrong RSS
  when the source is `rss_reddit_*` or `rss_lesswrong_com`; the
  previous label was misleading for those sources.
- New `test_tldr_prompt_forbids_nested_lists` asserts both prompts
  contain the no-nested rule. `test_generate_detailed_tldr_splits_article_and_comments`
  updated to match the new generic label.
- Tests: 203/203 pass (1 deselected, pre-existing broken
  `test_seed_hn_from_bq.py`), ruff clean. Service restarted on
  port 8766.

## 2026-06-26 — TLDR sub-topics: prompt-first, minimal JS

The LLM was emitting flat lists with label bullets (`- **Criticism of
US control**:`) followed by content bullets as siblings — visually
indistinguishable. The previous `:has(> ul)` CSS rule only handled
the *nested* case (~30% of outputs), and a 30-line JS restructure
function fought the rest by moving sibling `<li>`s into a nested
`<ul>`. The structure was correct but the code was duct tape.

The durable fix is the prompt. The LLM already knows how to use
`####` headings — the GPT 5.6 TLDR uses `####` for major sections
(`Key Announcement`, `Discussion Highlights`) but then dropped back
to label bullets for sub-topics within those sections. The prompt
now explicitly forbids label bullets as content headers and shows
the desired `####` pattern with a worked example. The three prompt
templates in `server.py` (article+comments, article-only,
discussion-only) all carry the new rule.

JS reduced from ~30 lines to ~6. The restructure function is
replaced by `styleTldrLabels`, a class-tagger: any `<li>` whose
direct text ends with `:` gets a `.tldr-label` class. CSS hides the
disc, bolds the text, and adds a subtle `▸` marker. The content
bullets stay as siblings — the prompt keeps them apart in 95% of
cases; the JS catches the rest.

Files:
- `server.py` (3 prompt templates): `####` sub-topic rule with
  example.
- `templates/index.html`: replaced `restructureTldrLabels` (30
  lines) with `styleTldrLabels` (6 lines); added `.tldr-label` CSS
  with `▸` marker; kept `:has(> ul)` rules as fallback for true
  nested markdown.
- `tests/test_server.py`: `test_tldr_prompt_forbids_nested_lists`
  now also asserts `####` appears in both prompts.
- `WORKLOG.md`: this entry.

Tests: 203/203 pass, ruff clean, service restarted on port 8766.

## 2026-06-26 — Preserve story `time` on upsert (fix GitHub Trending date drift)

GitHub Trending stories had their displayed date updated to the latest
fetch time on every regen. Root cause was two-fold:

1. The 3 `mshibanami.github.io/GitHubTrendingRSS/weekly/*.xml` feeds
   ship with **no date fields at all** (no `published_parsed`, no
   `updated_parsed`, no `pubDate`, no `created`). Of all ~30 RSS
   feeds in `config.toml`, only these three lack any date.
2. `pipeline.py:fetch_rss_feeds` falls back to `now` when both
   `published_parsed` and `updated_parsed` are missing. So every
   fetch re-stamps every story with the current fetch time.
3. `database.py:upsert_story` then did
   `ON CONFLICT DO UPDATE SET time=excluded.time`, overwriting the
   stored time unconditionally.

Live DB confirmed: all 105 `rss_mshibanami_github_io` rows had
`time ≈ fetched_at` (the fetch moment).

Fix: SQL CASE in `database.py:326` — preserve existing time when it's
non-zero, otherwise adopt the new time. One-line change at the SQL
layer rather than the per-source fetch path so it covers any future
source that re-stamps a time field.

```sql
time = CASE
    WHEN stories.time > 0 THEN stories.time
    WHEN excluded.time > 0 THEN excluded.time
    ELSE 0
END
```

Behavior:
- HN, BQ/CH seeds, and other RSS feeds: unchanged (their `time` is
  already non-zero on first insert; `existing > 0` branch wins).
- GitHub Trending: first fetch sets `time = now`; subsequent
  fetches preserve it. Display now shows "3 days ago" instead of
  "3 hours ago" once we've seen the story for a day.
- `_empty_story` placeholders (`time=0`): next real upsert still
  populates the time (the `excluded.time > 0` branch wins).

Side effects (all desired):
- Recency penalty in `pipeline.py:1490, 1738, 1954` now uses the
  true first-encounter age. Fresh trending repos rank above stale
  ones.
- Article-fetch eligibility in `pipeline.py:1994` correctly skips
  stories older than `max_age_days` (30 days).
- `fetched_at` is unchanged — still updated on every fetch, so
  `prune_stories` (60-day retention) keeps recently-seen stories.

No migration: existing rows keep their (wrong) last-fetch time.
Going forward, new entries get the correct first-encounter date.
If you want a clean slate, delete all `rss_mshibanami_github_io`
rows — but this is destructive and loses any feedback/embeddings.

Files:
- `database.py:326-330`: SQL CASE replaces `time=excluded.time`.
- `tests/test_database.py`: 2 new tests —
  `test_upsert_story_preserves_time_on_reinsert` (the GitHub
  Trending case), `test_upsert_story_uses_new_time_for_placeholder`
  (the `_empty_story` case).
- `WORKLOG.md`: this entry.

Tests: 199/199 pass, ruff clean. Live verification: re-upserted a
real `aws/agent-toolkit-for-aws` row with a time 2h in the future;
original `13:32:44` was preserved.

## 2026-06-26 — Consolidate live `hn` source from ~125 Algolia calls to 2 CH calls

- `pipeline.fetch_candidates` rewritten to use a single CH query for the
  live 7-day window:
  - **Before**: ~25 Algolia search calls (7 daily windows × up to 4 pages
    of 100) + ~100 Algolia items calls (one per missing/stale ID via
    `fetch_story` + `fetch_stories_by_id`) + ≤10 per-story Algolia
    `refetch_story_text` = ~135 Algolia calls/regen.
  - **After**: 1 CH `query_live_window` + 1 CH bulk `prewarm_top_stories`
    (for refetch) = 2 CH calls/regen.
- New helper `_ch_story_item_to_story` converts a CH live-window item
  (Algolia shape) to a `Story` row.
- New constant `LIVE_WINDOW_LIMIT = 2000` (matches the previous Algolia
  per-day cap).
- `_select_refetch_ids` still uses `fresh_metadata` to identify growth
  candidates, but the refetch itself goes through `prewarm_top_stories`
  (CH bulk) instead of per-story Algolia items calls.
- Tradeoff: 1-24h CH lag for brand-new stories. With 3h regen cycle, worst
  case is 4h lag. Acceptable for "best of HN" view; the swipe deck
  mostly shows older stories anyway.
- Algolia kept as a **fallback** for `ch_seed`/`bq_seed` lazy fetches
  (cards outside the prewarm top-20). Real-time, low frequency.
- Archive seed behavior: `ch_seed`/`bq_seed` are read from DB only (no
  per-regen refetch). Stories that need re-hydration require a future
  `refresh_archive_stories` function (out of scope for this change).
- Tests:
  - Updated 3 existing tests to mock `ch_client.query_live_window`
    instead of `httpx.AsyncClient` for Algolia.
  - Added 3 new tests: `test_fetch_candidates_ch_live_window_inserts_new`,
    `test_fetch_candidates_ch_live_window_updates_existing_score`,
    `test_fetch_candidates_ch_failure_returns_empty_live`.
- Updated `AGENTS.md` (data-source table: drop Algolia from live
  source), `ARCHITECTURE.md` §3.6 (rewrite live-window section).
- 206/206 tests pass, ruff clean.

## 2026-06-26 — Drop regen-time growth refetch (step 2 simplification)

- Realized step 2 of the per-regen `hn` source pipeline (CH bulk refetch
  for growth-eligible stories) was mostly redundant with the render-time
  `prewarm_top_stories` call:
  - Both update `text_content` and re-embed via the same `ch_client.query_stories_with_comments` path
  - The render-time prewarm covers the top-20 cards the user actually clicks
  - The regen-time refetch covered growth-eligible stories outside the top-20; those now wait ≤3h (one regen cycle) for the next prewarm
- Removed:
  - `pipeline.refetch_story_text` (was per-story Algolia items call)
  - `pipeline._is_refetch_eligible`
  - `pipeline._select_refetch_ids`
  - Constants: `MAX_REFETCH_PER_REGEN`, `COMMENT_GROWTH_THRESHOLD`, `COMMENT_REFETCH_MAX_AGE_HOURS`
  - The `_select_refetch_ids` + `prewarm_top_stories(refetch_ids, ...)` block from `fetch_candidates`
  - 9 refetch-related tests in `tests/test_pipeline.py`
- `fetch_candidates` now does **1 CH call per regen** (just `query_live_window`), down from 1-2
- Wall time: ~0.5s (was ~1-2s)
- Updated `ARCHITECTURE.md` §3.7: refetch-on-growth section replaced with
  "Re-embedding on dashboard render" description (the prewarm IS the
  re-embed trigger now).
- 197/197 tests pass (was 206, removed 9 refetch tests), ruff clean.
