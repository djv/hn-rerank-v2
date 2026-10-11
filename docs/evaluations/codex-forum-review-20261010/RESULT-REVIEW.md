# Independent review: Codex Forum content fixes (base 62af20e)

**Mode:** static review only. I did **not** run the test suite. Project dependencies aren't installed here (`feedparser` is missing), and the prompt forbids installing them. I ran two things:
- A standalone copy of the two HTML footer regexes, with `python3 -I`.
- `ruff`. The result proves nothing, because the repo's ruff config isn't in the snapshot and a global config added unrelated rules.

There is no evidence for lint, type checks, the focused tests or the full suite. The "96 passing" claim has not been verified.

## Verdict

**Approve with changes.** The core design is sound:
- A pure, idempotent effective view.
- Article suppression in one place.
- Footer stripping in the RSS parser before `clean_text`.
- A Codex-only versioned key.

I traced every reader that matters:
- **Embeddings:** these are safe even without the explicit wrappers. `story_embedding_text` normalizes internally (`pipeline/ranking.py:655-658`). Stored vectors are reused only when the hash matches (`ranking.py:910-920`), and the only other embedding writer (`enrichment.py:1396`) never runs for Codex.
- **All 7 `_tldr_cache_key` call sites** pass `source`. LLM inputs and the key are both built from the same effective fields (`server.py:3372-3396`, `3585-3591`, `1665-1683`).
- **Write-back:** the effective view is never written back. Every tap lane is closed to this source: HN lanes need `is_hn_source`, the source lane only runs for Reddit/LW, and the article lane now excludes Codex. The upsert longest-text merge keeps the raw data, and reads normalize it again.
- **Copies without raw fields:** I found no realistic producer for Codex. `hn_dupes.build_feedback_context` builds copies only for `source == "hn"`. Every DB read path selects the full fields, including `get_feedback_stories` after this patch.

One behavior issue needs a decision before merge (finding 1). One change to non-Codex behavior should be reverted or explicitly accepted (finding 2).

## Confirmed code issues

### 1. Medium — Stale, cooldown, busy and no-session fallbacks are effectively disabled for all Codex summaries, not just legacy ones
`server.py:2988-3001`

**Data flow:**
- The Codex branch serves a stale row only when the stored key equals the *current* key.
- `_handle_flask_tldr_detail` already returns any exact match as a normal `cache_hit` (`server.py:3611-3622`) before it reaches the provider-cooldown fallback (`3640`) or the no-session fallback (`3650`).
- Busy and quota fallbacks (`3126`, `3137`) are reached only after that same lookup.

So for Codex the fallback can fire only on the `force_refresh` path. Anything that changes the key turns every Codex summary into "no fallback". That includes:
- the `_llm_cache_identity()` provider/model,
- `TLDR_PROMPT_VERSION` (`server.py:1294-1296`),
- any change to the opening post or comments.

**Impact:** during a provider cooldown, or for clients without a session, after any model or prompt change:
- Codex cards return 429 or 401.
- Every other source serves its stale summary, as it did before.

Clean summaries generated after this deploy also lose stale serving. That goes further than the objective, which is to reject *incompatible legacy* caches.

**Fix without migration:**
- `tldr_cache.created_at` already exists (`database.py:359`).
- Add a fixed constant, e.g. `CODEX_INPUTS_V1_SINCE = <deploy epoch>`.
- Reject a Codex stale row only when `created_at < cutoff`, and otherwise keep normal stale semantics.
- Alternatively, keep the exact-match rule but document it as intentional and accept the availability cost.

**Repro test:**
1. Insert a Codex row and cache a summary under its current normalized key.
2. Monkeypatch `_llm_provider_config` to return a different model.
3. Assert that `_stale_tldr_fallback(db, id, "provider_cooldown")` is `None` today. The equivalent setup for a generic source returns the summary.

### 2. Low–Medium — Tap article-lane change for a non-Codex source (LessWrong)
`server.py:3180`

The old condition was `story.source != AINEWS_SOURCE`. The new one is `story.source not in _NO_ARTICLE_SOURCES`, which also excludes `rss_lesswrong_com`.

**When this changes behavior:**
- LessWrong reaches `article_eligible` with `src_kind is None` whenever `self_text` and `top_comments` are both non-empty (`server.py:3164-3169`).
- Before this patch, a short (<500 chars) LessWrong post with comments and no `article_body` fetched its URL on tap. Now it doesn't.

This matches the proactive selectors, which already excluded LessWrong, and it may well be what you want. But it contradicts "preserve other sources' behavior", nobody mentions it, and no test covers it.

**Options:**
- Use `story.source not in {AINEWS_SOURCE, CODEX_FORUM_SOURCE}`, or
- keep the shared set, record the LessWrong change as deliberate, and add a small test.

(Importing the private `_NO_ARTICLE_SOURCES` across modules is a minor nit.)

### 3. Low — New files appear at absolute-looking paths in the reviewed diff; risk of an incomplete commit
`REVIEWED.diff:529` and `:673` show `pipeline/codex_forum.py` and `tests/test_codex_forum.py` under `home/dev/hn-rewrite/hn-codex-forum-fixes-20261011/...`. That suggests they were generated with `--no-index` and are untracked.

`server.py`, `pipeline/enrichment.py`, `pipeline/__init__.py` and `pipeline/ranking.py` all import `pipeline.codex_forum`. If the branch is committed without `git add pipeline/codex_forum.py`, the app and the whole test suite fail at import time.

Before committing, check `git status --porcelain` on the feature branch.

## Test quality and missing coverage

**Good:**
- Footer tests cover both layouts, near-misses and idempotency.
- The parse test proves stripping happens only for Codex.
- The article-gate test includes a generic control.
- The two tap tests would have failed before the patch: their URL is fetchable, and the old check excluded only AINews.
- The legacy key and stale-fallback unit tests are useful.

**Important gaps** (one test each, not test matrices):

- **Feedback and training wiring isn't tested.** `test_feedback_reads_are_effective` calls `effective_codex_story` itself. Deleting the wrappers at `ranking.py:1260`, `ranking.py:2608` or `pipeline/__init__.py:1197` fails no test.
  - Embeddings stay clean anyway through `story_embedding_text`.
  - What isn't protected is `len(s.text_content)` in the text-length features (`ranking.py:1352`, `1422`, `1525`) and the dedup inputs.
  - Suggest one `_score_and_rank` test with `training_feedback=` containing a legacy row, asserting what the model or features actually receive.
- **`fetch_candidates` step 6 and the `fetch_candidates_only` refresh (`pipeline/__init__.py:1001`, `1255`) aren't covered.** The candidate-pool test covers `load_production_candidate_stories` only.
- **The hot-summary endpoint (`server.py:3913`) and the handler-level cooldown/no-session fallbacks are untested for Codex.** A legacy row with a legacy-keyed summary should return a miss or 204 there.
- **Nits:**
  - The `"giraffes".replace(...)` parametrize case is confusing; write the literal.
  - `[s for s in select_rss_article_prewarm(...)]` is an identity comprehension.
  - `_FakeEmbedder.captured` is a mutable class annotation; harmless.

## Uncertainties (not confirmed defects)

- **Count format.** Both regexes need plain integers. When I probed, `1,234 posts - 2 participants` was **not** stripped, so the whole footer is preserved. That is safe and conservative, but check whether Discourse ever formats large counts like `1,234` or `1.2k` in RSS. Codex topics rarely pass 999 posts.
- **Real feed whitespace.** Newline-indented Discourse output (`<p>Body</p>\n    <p><small>…</small></p>\n    <p><a …>Read full topic</a></p>\n`) stripped correctly in my probe. I didn't check the real feed through feedparser's sanitizer.
- **Regex cost.** Both HTML regexes are anchored with `$` and are quadratic on pathological input: 20k unclosed `<p ` tokens took about 0.9 s. Realistic descriptions are small, so this is speculative.
- **Code outside the snapshot.** `clients/tui` and any scripts aren't in the snapshot, so I couldn't audit other `_tldr_cache_key` or `get_tldr_cache` callers. The server-side endpoints are all covered.
- **Deliberate trade-off.** The longest-text upsert merge keeps a dirty `self_text` (with footer) and a dirty `article_body` forever. That's intended (raw is preserved), so every future reader must keep going through the effective view. `story_embedding_text` enforces this for embeddings only. Consider a short comment at `Database.upsert_story`.

## Verification gaps to close before merge

1. Run the focused tests yourself (`tests/test_codex_forum.py` and the two new `test_server.py` tests), then the full default suite, `ruff check`, `ruff format --check` and `ty` with the repo config. Record the results in CHECKS.md.
2. Confirm the two new files are tracked at repo-relative paths (finding 3).
3. Decide on finding 1 (cutoff vs. exact match) and finding 2 (LessWrong scope).
