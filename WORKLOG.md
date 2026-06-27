# Worklog: hn-rewrite

Append-only log of notable changes, fixes, and operational events.
Each entry is dated and self-contained.

---

## 2026-06-27 — Badges + UI cleanup (6 items + novel pass simplification)

Big readability sweep across `pipeline.py`, `server.py`, `templates/index.html`, and supporting modules. The user said the system was "becoming complicated" — this addresses the accidental complexity while leaving the conceptual structure intact.

### 1. Fix `ch_seed` source invisible to client filter (latent bug)
- `ch_seed` source was treated as neither HN nor Non-HN by the client filter (only `hn` and `bq_seed` matched the HN bucket). Fixed by emitting a server-side `data-is-hn` flag on every card (derived from `is_hn_source`) and switching the client filter to read it.
- `templates/index.html:836` now emits `data-is-hn="{{ '0' if item.is_non_hn else '1' }}"`.
- New test `test_story_cards_emit_is_hn_attribute` enforces the new contract.

### 2. Add 5 missing badge/weight config knobs (consistency)
- New `ModelConfig` fields: `hot_badge_percentile=99.5`, `similar_badge_percentile=97.0`, `novel_badge_percentile=10.0`. Defaults match the old hard-coded magic numbers; behavior unchanged unless config overrides.
- Tooltips for the Hot, Similar, and Novel badges are now template-driven: `"Top {{ hot_badge_percentile }}% by engagement velocity..."`, `"Top {{ similar_badge_percentile }}% most similar to your upvoted stories"`, `"Bottom {{ novel_badge_percentile }}% by similarity to your feedback, but model scores it high"`.
- New test `test_hot_badge_threshold_uses_config_percentile` verifies the knob is honored.

### 2b. Make Novel badge purely distance-based (no score blend)
- The Novel extra-slot pass previously ranked by `0.7 * score_pct + 0.3 * novelty_pct` — a blend that let high-score stories crowd out genuinely novel ones. Now it sorts by distance only: `argsort(-novel_distances)`.
- The `novel_score_weight` / `novel_distance_weight` config knobs are removed (no longer meaningful). Net behavior: a story that is semantically distant from your feedback will be surfaced regardless of how the model would have ranked it.
- New test `test_novel_pass_ranks_purely_by_distance_not_score` verifies the cut.

### 3. Move `_augment_features` to `legacy_features.py` (research-code isolation)
- 108-line `_augment_features` was dead code in the production path but still used by 4 offline eval scripts (`eval.py`, `eval_rss.py`, `eval_no_hn_features.py`, `scripts/feature_ablation.py`). Moved to a new top-level `legacy_features.py` module with its own log-scale constants; production `pipeline.py` now only contains `_svm_personalization_features` (the slim version actually called by `_score_and_rank`).
- Also removed the unused `_AGE_DAYS_SCALE` constant.
- `tests/test_pipeline.py::test_augment_features_properties` now imports from `legacy_features`. All 4 eval scripts updated to do the same.

### 4. Collapse 7 discovery passes to a table-driven loop
- The 7 numbered passes in `rerank_candidates` (uncertain / novel / similar / discussion-rich / high-engagement / hot / non-hn) were 95% identical boilerplate (filter → sort → take K → extend → prune). Replaced with a `DiscoveryPass` dataclass and a single loop over a list of 7 entries. ~130 lines → ~50 lines, and the 7 slot caps + predicates are visible in one place.
- Public behavior unchanged; all 243 tests pass.

### 5. Extract inline `<script>` to `static/dashboard.js`
- 758-line inline `<script>` block extracted from `templates/index.html` to `static/dashboard.js`. Template shrank from 1726 → 968 lines; the script is now a real, syntax-highlightable, grep-able file. No build step.
- Added a `/static/<path>` route handler in `server.py` (mimetype per `.js`/`.css`/`.svg`/`.png`/`.ico`, `no-cache` headers to match the dashboard HTML).
- New helpers in `tests/test_server.py`: `_read_template_and_static()` and 2 new tests (`test_dashboard_js_loaded_via_static_endpoint`, `test_static_dashboard_js_has_no_jinja`).

### 6. Extract prompt strings to `prompts/*.txt`
- The 5 inline LLM prompt strings inside `generate_detailed_tldr` (server.py) are now 5 files in `prompts/`: `article_v4.txt`, `discussion_v4.txt`, `article_only_v4.txt`, `discussion_only_v4.txt`, `combined_v4.txt`. Loaded via a small cached `_load_prompt(name)` helper. Filenames are pinned to `TLDR_PROMPT_VERSION = "detail-v4"` so the cache key and file name stay in sync.
- `server.py` shrank by ~150 lines.

### 7. Consolidate HTTP fetch fallback
- `_urllib_fetch` (Cloudflare TLS fingerprint workaround) and the "try httpx, fall back to urllib on 403/503" decision were duplicated in `pipeline.py` (RSS fetch) and `server.py` (article fetch). Extracted to a new `http_fetch.py` module with `urllib_fetch(url, ua)` and `fetch_with_urllib_fallback(client, url, headers)` helpers. Both call sites use the helper.
- `_urllib_fetch` is re-exported from `pipeline.py` for backward compatibility with `server.py` and any other callers.

### Final state
- 243 tests pass, ruff clean, ty clean, server restarted cleanly.
- LOC change: `pipeline.py` -108, `server.py` -150, `templates/index.html` -758, `eval*.py` +4 imports each, `legacy_features.py` +113 (new), `http_fetch.py` +60 (new), `static/dashboard.js` +760 (new), `prompts/*.txt` +145 (new).

## 2026-06-27 — Config-driven Discussion-rich badge threshold (default 90th pct)

- **New config knobs `discussion_badge_percentile` (default 90.0) and `discussion_badge_min_comments` (default 0)** in `ModelConfig` and `config.toml`. The Discussion-rich threshold is now `max(np.percentile(nonzero_comments, pct), float(min_comments))` instead of hard-coded `np.percentile(nonzero_comments, 98)`.
- **Template tooltip now dynamic**: renders "Top {{ discussion_badge_percentile }}% by HN comments" instead of the stale "top 7%".
- **ARCHITECTURE.md:121** updated to reflect the config knobs.
- New test `test_discussion_badge_threshold_uses_config_percentile_and_floor` verifies the threshold responds to both knobs.

## 2026-06-27 — Rename `rank_stories` → `_score_and_rank` for clarity

- Renamed internal scoring function from `rank_stories` to `_score_and_rank` to signal that it's private and to distinguish it from the public `rerank_candidates` (which wraps it and adds badges + discovery passes).
- Added docstring layering note to `rerank_candidates`.
- Updated all call sites in `pipeline.py`, `tests/test_pipeline.py`, `ARCHITECTURE.md`, and `WORKLOG.md`. Skipped `plans/` (historical design docs).
- Zero behavioral change; all 237 tests pass.

## 2026-06-27 — Config-driven Top badge threshold (default 90th pct, min 100)

- **New config knobs `top_badge_percentile` (default 90.0) and `top_badge_min_score` (default 100)** in `ModelConfig` and `config.toml`. The engagement_threshold is now `max(np.percentile(scores, pct), min_score)` instead of hard-coded `np.percentile(scores, 98)`.
- **Template tooltip now dynamic**: renders "Top {{ top_badge_percentile }}% by HN points" from config instead of the stale "Top 5%".
- **ARCHITECTURE.md:122** updated to reflect the config knobs.
- New test `test_top_badge_threshold_uses_config_percentile_and_floor` verifies the threshold responds to both knobs.

## 2026-06-27 — Filter unsummarizable stories; expand regen prewarm (HN + Reddit full mode)

- **Filter unsummarizable stories from the dashboard:** `fetch_candidates` now drops stories with no self_text, no top_comments, no article_body, and (for HN) zero comments. They would only ever produce a "No article body or discussion available to summarize for this story." placeholder in tldr-detail. Covers ~965 HN stories with `comment_count==0` and ~180 non-HN stories with no content.
- **Expand regen prewarm to all candidates (not just top-N by score):** `fetch_candidates_only` now prewarms `top_comments` for all HN candidates with `comment_count > 0` and empty `top_comments` (~2149 stories), all Reddit RSS candidates with empty `top_comments`, and all LessWrong RSS candidates with empty `top_comments`. New config knobs `prewarm_hn_full`, `prewarm_reddit_full`, and `prewarm_lesswrong_full` (default true) control scope; set false to revert to top-by-score prewarm.
- **Purged 20 orphan tldr_cache rows** holding the literal "No article body..." placeholder (one-off SQL DELETE).
- Added `is_summarizable(story)` helper in pipeline.py.
- Expanded test coverage: 8 new tests for `is_summarizable`, candidate filter, and full-mode prewarm; fixed 1 existing test fixture (archive seed needed `self_text` to survive the filter).

## 2026-06-27 — TLDR prompt fix: structural separation of article-only and comments-only code paths

- **Bug:** The TLDR prompt had overlapping triggers for the "article + discussion" and "article-only" output branches. Both branches fired on the same input (article body present, no comments). The LLM always chose the richer "two sections" branch and hallucinated a Discussion section from nothing.
- **Fix (code):** The unified `else` branch in `generate_detailed_tldr` was split into two separate code paths with two separate prompts:
  - **Article-only path:** prompt never mentions "Discussion", "Consensus", "Disagreement", or "Caveat". The words physically do not appear.
  - **Comments-only path:** prompt only mentions "Discussion" and discussion labels, effectively unchanged.
  - **Both (defensive):** preserved for the edge case where both arrive via the wrong branch.
- **Cache clear:** `tldr_cache` deleted (801 rows). Version bumped to `detail-v4`. All existing TLDRs will regenerate on next request with the new structural prompt.

## 2026-06-27 — Reddit RSS comment pre-warm in regen, plus RSS self_text fix

- **Pre-warm:** New `pipeline.prewarm_reddit_top_stories(story_ids, db, embedder)`
  fetches Reddit RSS comments for top-N (default 20) Reddit candidates during
  regen, matching the HN prewarm pattern. Called from `fetch_candidates_only`
  after the HN prewarm loop, serialized to avoid 429 rate limits.
- **Config:** `Config.reddit_prewarm_top_n: int = 20` added.

## 2026-06-27 — fix RSS TLDR stub; populate self_text at ingestion

- **Bug:** Reddit RSS stories showed "No article body or discussion" in the
  TLDR card. Root cause: RSS pipeline (`pipeline.py:fetch_rss_feeds`) stored
  the post body in `text_content` but left `self_text` empty. The `/api/tldr-detail`
  endpoint feeds only `self_text + top_comments + article_body` to the LLM,
  so when the runtime Reddit RSS context fetch failed (429 rate limit), all
  inputs were empty and the stub was returned.
- **Fix:** RSS pipeline now populates `story.self_text` from the feed body
  and derives `text_content` via `compose_story_text(title, self_text)`,
  matching the HN pipeline convention.
- **Backfill:** `scripts/backfill_rss_self_text.py` — one-shot idempotent
  backfill that strips the `"{title}. "` prefix from existing `text_content`
  to populate `self_text` for 1709 RSS stories (44 edge cases with no real
  body content flagged as warnings). 3 stories with unicode/emoji title
  normalization mismatches fall back to `clean_text(title)` — all covered.
- **New regression test:** `test_fetch_rss_feeds_populates_self_text` in
  `tests/test_pipeline.py`.

## 2026-06-27 — SWR render path + SVM model cache; fix stale-cache pop bug

- Stale-while-revalidate dashboard rendering: first request for a user
  returns a 3s meta-refresh skeleton immediately, background thread
  renders the real dashboard. Subsequent requests within the same version
  return cache hit; version bump (from feedback or regen) returns stale
  data while triggering a fresh warm.
- SVM model cache (`pipeline.py`): module-level `OrderedDict` keyed on
  `(user_id, feedback_signature)`. `feedback_signature` is SHA-256 of
  `(story_id, action, updated_at)` tuples. Cache hit skips the 3-5s
  `SVC.fit()` and reuses the trained model; LOOCV k-NN is still
  recomputed. `max_cached_models=20` (LRU eviction). Eliminates the
  per-render retrain cost that was OOMing the 8GB box with 10 concurrent
  high-feedback users.
- Prewarm moved from render path to regen path. `fetch_candidates_only`
  now prewarms the top-50 by score; `_render_dashboard_for_user` no
  longer calls `prewarm_top_stories`. The first dashboard render after
  regen finds the top-scored candidates already populated.
- Regen bumps all cached-user versions via `_bump_all_cached_versions`,
  so every cached user gets a fresh warm after each candidate fetch.
- `_render_dashboard_for_user` now writes structured logs: result
  (`cache_hit` / `stale_hit` / `skeleton`), version, elapsed_ms,
  cache_age_s.
- **Bug fix**: `_invalidate_dashboard_cache` was popping the cache
  entry, which broke SWR semantics — after feedback, the next render
  got a skeleton instead of stale data while the warm ran. Fix: drop
  the `pop`; the warm thread now overwrites the cache entry under
  the render lock, which is what the dedup check at `server.py:723`
  was already designed to handle.
- Tests: 3 new SWR tests (skeleton, stale hit, cache hit), 1 warm
  dedup test, 1 different-versions-not-deduped test, 1 cache cap test,
  1 bump-all-versions test, 1 SWR cache-uses-versions test, 1 stale
  warm overwrites cache test, 1 version invariant property test
  (hypothesis). The property test now uses `monkeypatch` fixture with
  the `function_scoped_fixture` health check suppressed.
- Test fixture hardening: `test_env` teardown now drains
  `_warmup_in_flight` before tearing down the server. Without this,
  warm threads from `test_feedback_post` (1s debounce) would outlive
  the fixture, monkeypatch onto the next test's `pipeline` functions,
  and inflate the next test's call counts. Diagnosed via
  `threading.get_ident()` — saw 2 different thread IDs appending to
  the same `calls` list with the same stack frame.
- **Test speed optimization**: replaced hardcoded `time.sleep(1.0)` in
  `_trigger_warm` with class attribute `_WARM_DEBOUNCE_S = 1.0`. Test
  fixtures override to 0.01, cutting the warm-thread sleep from 1000ms
  to 10ms. Combined effect: test suite 76s → 15s (5x faster).

## 2026-06-26 — detail-v3: tighten unified fallback prompt; bump discussion budget to 150w

- Bug report: story 48689028 ("Previewing GPT-5.6 Sol") yielded a TLDR dominated by
  a fabricated "Article Overview" (synthesized from the title alone) with 11+
  discussion bullets organized into 5 sub-categories. Root cause: 1) `article_body`
  fetch returned 403 from openai.com; 2) the dual-prompt path is gated on
  `article_section and comments_section`, so control fell through to the single
  unified fallback prompt; 3) the unified fallback had no structural enforcement
  (no bullet limit, vague "under 240 words" budget, no detection of missing
  article text). 3,952 / 5,153 HN stories (77%) have no `self_text` or
  `article_body` and were affected.
- User preference: discussion-only stories must not produce any article/story
  section. The `### Article` section was being fabricated from the title alone.
- **Fix**: replaced the unified fallback prompt (`server.py:482-498`) with a
  conditional prompt that has three explicit cases:
  - Article text + comments → `### Article` (120w) + `### Discussion` (150w)
  - Only comments → `### Discussion` only (150w, 3-5 bullets)
  - Only article text → `### Article` only (120w, 3-5 bullets)
  The discussion-only path explicitly instructs the model not to write an
  Article or Story section and not to summarize the title as if it were
  article content.
- Added a stub short-circuit at `server.py:408-409` — when both article_section
  and top_comments are empty, return a self-explanatory stub string instead of
  calling the LLM at all.
- Bumped the dual-prompt discussion budget from 100w to 150w (`server.py:428`),
  giving richer comment threads more room to breathe.
- `TLDR_PROMPT_VERSION` bumped from `"detail-v2"` to `"detail-v3"` so the old
  cache keys are invalidated and the new prompt takes effect on next click.
  ~3,952 cached entries will be regenerated lazily.
- `max_tokens` left at 2000 for the unified fallback (rely on prompt word
  limits for enforcement).
- New tests: `test_unified_fallback_omits_article_when_no_article_body`
  (asserts prompt has the conditional instruction, no article/story section in
  output), `test_generate_detailed_tldr_returns_stub_when_no_content` (zero
  LLM calls for empty content). 180/180 tests pass, ruff clean.
- ARCHITECTURE.md §4.2 updated with the 4-path TLDR table.

## 2026-06-26 — Dep cleanup: torch → optional group; drop unused duckdb + matplotlib

- `pyproject.toml`: `torch>=2.12` moved out of runtime `dependencies` into a
  new optional `[dependency-groups] dl-experiment = ["torch>=2.12"]` group.
  The live path (server / pipeline / database / ch_client / generate) does
  not import torch; only the unshipped `pipeline_dl.py` + `pipeline_dl_t0.py`
  experiment and the `scripts/eval_ranker_variants.py` offline eval do.
  See 2026-06-25 below for why the experiment is unshipped.
- `pyproject.toml`: removed `duckdb>=1.0.0` and `matplotlib>=3.11.0`. Audit
  found zero `import duckdb` or `import matplotlib` in the repo — both
  pins were stale from prior analytics experiments and never wired up.
- `tests/test_pipeline_dl.py`: added `pytest.importorskip("torch")` at
  the top of the file. The 21 DL-experiment tests now skip (not fail) in
  the default `uv run pytest` run, and execute normally with
  `uv run --group dl-experiment pytest tests/test_pipeline_dl.py`.
- `scripts/eval_ranker_variants.py`: added a friendly `sys.exit(...)` at
  the top of the script that explains how to install the missing group
  (`uv sync --group dl-experiment`) instead of crashing with
  `ModuleNotFoundError: No module named 'torch'`.
- `uv.lock` regenerated. Default `uv sync` no longer pulls in the
  `torch` (532MB) + `triton` (198MB) + 12 `nvidia-cu*` wheels. Venv
  shrunk from 91 → 66 packages for new clones.
- `AGENTS.md` new "Dependency groups" section documents the policy
  and the install command.
- Verification: `uv run pytest tests/` (default group) → 178 passed, 1
  skipped (the DL group), 1 deselected (slow marker). `uv run --group
  dl-experiment pytest tests/test_pipeline_dl.py` → 21/21 passed.
  `uv run ruff check .` → clean. The DL-experiment entry point
  (`scripts/eval_ranker_variants.py`) prints the friendly error and
  exits 1 when torch is missing.

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

## 2026-06-26 — Revert: return to detail-v2 markdown TLDR pipeline

After a day of experimenting with Pydantic-validated JSON TLDR output, the
server curl test timed out (cold cache + json_object latency spike). Reverted
`server.py`, `templates/index.html`, `tests/test_server.py`, `pyproject.toml`,
`AGENTS.md`, `WORKLOG.md`, and `uv.lock` to the pre-Pydantic (detail-v2)
markdown era. The dashboard now uses the original dual-prompt `asyncio.gather`
(article + discussion), `_normalize_tldr_markdown`, and client-side Snarkdown
parsing. New scripts from the benchmark/eval effort (`benchmark_tldr_llms.py`,
`eval_top_models.py`, `eval_results/`, `benchmark_partial.json`) are deleted
from the working tree.

Tests: 178/178 pass (was 188; Pydantic tests removed, markdown tests restored),
ruff clean.

## 2026-06-26 — Recreate TLDR benchmark script (OpenRouter)

- New `scripts/benchmark_tldr_llms.py` — standalone offline benchmark for
  the detail-v2 TLDR prompt against any OpenRouter model.
  - Reads `hn_rewrite.db` directly (read-only), samples 25 deterministic
    mixed-source stories (HN, RSS, archive seed).
  - Mirrors the dual-prompt `asyncio.gather` (article + discussion) and
    fallback single-prompt paths from `server.py`. Asserts
    `TLDR_PROMPT_VERSION == "detail-v2"` at import to detect drift.
  - Implements format-compliance scoring: no nested lists, word caps,
    bullet count ranges, heading structure, normalization idempotency.
  - Stores raw model responses (`raw_article_response`,
    `raw_discussion_response`, `raw_fallback_response`) alongside the
    normalized `tldr` in each result entry.
  - Partial results saved after each story; resume with `--resume`.
  - Retry + exponential backoff on 429/5xx; skip-and-continue.
- New `tests/test_benchmark_tldr_llms.py` — 22 tests covering sample
  selection, prompt building, OpenRouter call, compliance scoring, partial
  cache, and generate_tldr_for_story dual/fallback paths.
- `AGENTS.md` — new "TLDR benchmark" section with command.
- `.gitignore` — add `eval_results/`.

## 2026-06-27 — tighten novel criterion (3-way max, 10th-pct) + client shuffle for popular/explore

- **Pipeline** (`pipeline.py:1707-1730`): novel `cand_max_sim` now includes neutral
  similarity (`cand_closest_neutral`), making the 3-way `max(up, down, neutral)`.
  Threshold lowered from 15th to 10th percentile. Both changes make the Novel
  badge stricter — a story must be semantically distant from ALL feedback (up,
  down, AND neutral) to qualify.
- **Client** (`templates/index.html`): added `shuffleStories()` (Fisher-Yates)
  that fires once on initial page load and once after each dashboard refill.
  Mode switches (default/popular/explore/date) no longer re-order the DOM; the
  shuffled order is preserved across mode changes. Date mode still sorts by time.
- **Tests** (`tests/test_pipeline.py`): added `test_candidate_similar_to_neutral_is_not_novel`
  and `test_no_neutral_feedback_uses_up_down_only_for_novel` using controlled
  embeddings via monkeypatch. Updated existing test comments from "15th-pct" to "10th-pct".
- Diagnostics: `uv run pytest tests/ -q` = 182 passed, 9.25s.
  `ruff check .` = all clear. No new `ty` diagnostics.

## 2026-06-27 — Refresh button hidden until preloaded doc is ready

**Problem**: The refresh button at the top of the dashboard was shown as soon as the
server confirmed `ranking_refresh_queued: true`, but the preloaded refill document
wasn't ready until 2.5s + fetch later. Clicking during this window awaited the
in-flight preload, producing a 10s+ perceived delay (worse in the cold-cache case).

**Fix**: Refined `updateRefreshBanner` in the HTML's inline JS to a four-state
machine based on `preloadedRefillDoc !== null`:
- "X votes syncing" — feedback in flight, button hidden
- "Preparing refresh..." — server confirmed, preload in progress, button hidden
- "New ranking ready" — preload done, button visible (click is instant)
- "Refilling queue..." — refill in progress, button hidden

`refreshNowBtn.hidden` now tracks `preloadedRefillDoc !== null` exactly. Error
branches (vote save failure, undo failure) explicitly hide the button to prevent
the user from clicking into a broken state.

Single 30-line diff in `templates/index.html`. No backend changes.

## 2026-06-27 — Remove static-export code path (`generate.py`, `public/`, `run_pipeline`)

**Motivation**: The queue-based app (server.py) renders `templates/index.html` via
Jinja2 per request using `generate_dashboard_bytes`. The one-shot static-export
path (`generate.py` → `run_pipeline` → `generate_dashboard` → `public/index.html`)
was dead code — not used by the running service, not served, not part of deployment.

**Deleted**:
- `generate.py` — the one-shot CLI entrypoint
- `public/index.html` and the empty `public/` directory (gitignored; only residual)
- `PLAN.md` — historical v1 design doc; current architecture is queue-based

**Removed from `pipeline.py`**:
- `output` field from `Config` dataclass (and its `config.toml` key)
- `generate_dashboard()` function (wrote HTML to disk; `generate_dashboard_bytes`
  is unchanged and used by the live server)
- `async def run_pipeline()` function (orchestrator for the one-shot path; no callers
  after `generate.py` deletion)

**Removed from tests**:
- `test_run_pipeline_badge_assignment` and `test_primary_story_gets_qualifying_badge`
  in `tests/test_pipeline.py`. Both tests existed solely to exercise badge assignment
  through the `run_pipeline` → `generate_dashboard` path. Without those functions,
  the tests were dead. Badge logic is covered elsewhere via `_score_and_rank`/`rerank_candidates`.

**Updated**:
- `scripts/test_svm_variants.py` — removed DASHBOARD constant, generate.py subprocess call,
  and score-spread scrape block
- `AGENTS.md` — removed `public/` and `generate.py` mentions
- `ARCHITECTURE.md` — simplified mermaid diagram (removed `generate_dashboard`/`run_pipeline`)

**Verification**: `pytest tests/ -q` = 180 passed (2 removed). `ruff check .` = all clear.
`ls generate.py public/` = both gone. Server restarted cleanly. `curl` confirms
refresh button bug fix is live.

## 2026-06-27 — Move prewarm from render-path to regen-path (fresh user first load 15.6s → ~2s)

**Problem**: Fresh session first load took 15.6s. The render-path prewarm
`prewarm_top_stories` (called on every dashboard render) dominated at 13.5s
(87%): one CH bulk comments query for the personalized top-20, then 8-17
`upsert_story` + `upsert_embedding` single-row writes + embedding re-encode.

**Fix**: Prewarm happens at regen time, not render time. The regen loop
(`fetch_candidates_only`) now accepts an `embedder` and prewarms the top-N
(default 50) HN candidates by score descending before the regen completes.
All candidate rows in the DB already have `top_comments` populated when any
user's first render runs.

**Changes**:
- `pipeline.py:Config` — added `regen_prewarm_top_n: int = 50` field
- `pipeline.py:fetch_candidates_only` — accepts `embedder` and `prewarm_top_n`;
  sorts HN candidates by score desc, takes top N, calls `prewarm_top_stories`
- `server.py:regen_loop` — passes `Handler.embedder` to `fetch_candidates_only`
- `server.py:_render_dashboard_for_user` — **removed** per-render
  `prewarm_top_stories` call (was 13s of the fresh-user first load)
- `config.toml` — added `regen_prewarm_top_n = 50`
- `tests/test_pipeline.py` — 4 new tests: prewarms top N by score, skips
  when n=0, skips when embedder=None, handles empty candidate list

**Impact**:
- Fresh user first render: **15.6s → ~2.0s** (just the ranker)
- CH bulk calls: 1/render → 1/3h (shared across all users)
- Embedding recompute: 8-17/render → 8-17/3h (amortized)
- Per-render `prewarm_top_stories` removed from every dashboard render

**Verification**: `pytest tests/ -q` = 184 passed. `ruff check .` = clean.
New `ty` diagnostics: 0 (all 44 seen are pre-existing in other files).

## 2026-06-27 — Type-discipline cleanup: 66 → 0 ty diagnostics

**Context**: The codebase had accumulated 66 `ty` type-checker diagnostics across
12 files (44 pre-existing + 22 from recent changes before this cleanup). The
AGENTS.md requires zero new diagnostics.

**Bug fix (real behavior change)**:
- `ch_client.py` and `scripts/seed_hn_from_clickhouse.py`: changed `httpx.post(url,
  data=query, ...)` to `content=query`. At runtime httpx accepts `str` in `data=`
  (sends as form-urlencoded body), but `data=` is typed for form-field dicts.
  `content=` is the correct parameter for a raw SQL body. CH Playground is
  lenient about content-type, so behavior is identical.
- Test mocks in `tests/test_ch_client.py` and `tests/test_seed_hn_from_clickhouse.py`
  updated to check `kwargs.get("content", "")` instead of `kwargs.get("data", "")`.

**Type-cleanup patterns (across 12+ files)**:

| Pattern | Files fixed | Count |
|---|---|---|
| `Story \| None` — add `assert story is not None` before field access | test_pipeline.py, seed_hn tests | ~22 |
| `object()` → `MockEmbedder(Embedder)` for embedder fixture | test_server.py, test_pipeline.py | ~5 |
| `DummyEmbedder` → `DummyEmbedder(Embedder)` | seed_hn test files | 2 |
| `-> ...` → `-> Any` | test_server.py | 1 |
| `log_message(self, *a)` → match parent sig | test_fetch.py | 1 |
| `embedder: object` → `embedder: Embedder` | eval_ranker_variants.py | 2 |
| `# type: ignore` for torch imports (dl-experiment) | pipeline_dl.py, pipeline_dl_t0.py | 7 |
| `SGDClassifier.classes_` — `# type: ignore` where ty can't resolve | eval_ranker_variants.py | 1 |
| Nested dict access — `# type: ignore` for ty union resolution limit | eval_rss.py, eval_no_hn_features.py | 6 |
| Lambda assignment — `# type: ignore` for ty function-type limitation | test_server.py | 2 |
| `np.ndarray \| None` subtraction guards | eval_rss.py | 3 |
| `Embedder.__init__` — `self.tokenizer: Any` annotation | pipeline.py | 1 |

**Verification**: `ruff check .` — clean. `ty check` — 0 diagnostics (was 66).
`pytest tests/ -x -q` — 201 passed, 1 skipped (torch-dependent), 0 failed.

## 2026-06-27 — Fresh-user first load 5-10s → ~1.5-2.0s

**Problem**: Brand-new user dashboard load took 5-10s. Profiled warm-thread
work in `fast_rerank_for_user` with 6425 candidates (30d window) and
identified four optimizations.

**Root causes**:
- `ranked_decorated = [replace(r, is_non_hn=...) for r in ranked]` — 6425
  `dataclasses.replace` calls per render (~200ms). Verified dead after
  tracing every consumer: `is_non_hn` is never read; primary items get it
  re-set in the badge pass, discovery pass #7 calls `is_hn_source` directly,
  and the template doesn't read `.is_non_hn`. A final ~80-item `is_non_hn`
  pass was added to maintain correctness for discovery items.
- `_WARM_DEBOUNCE_S = 1.0` — 1s sleep at start of warm. Reduced to 0.2s.
- Skeleton `meta http-equiv="refresh" content="3"` — forced 0-3s wait
  between warm completion and dashboard visible. Reduced to `content="1"`.
- `pico.min.css` re-read on every render. Moved to module-level lazy cache.

**Changes**:
- `pipeline.py:rerank_candidates` — removed `ranked_decorated` construction;
  filter `ranked` directly; added final `is_non_hn` pass on final items.
- `pipeline.py:generate_dashboard_bytes` — `pico.min.css` read moved to
  module-level `_get_pico_css()` lazy cache.
- `server.py:Handler._WARM_DEBOUNCE_S` — 1.0 → 0.2.
- `server.py:SKELETON_HTML` — `content="3"` → `content="1"`.

**Impact**:
- Warm-thread (6425-candidate render): 1.3s → 1.1s
- End-to-end fresh user: 5-10s → 1.4-2.1s

**Verification**: `pytest tests/ -x -q` = 209 passed, 1 skipped.
`ruff check .` clean. `ty check` 0 diagnostics.
