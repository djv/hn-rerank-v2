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
