# Review: remaining Codex Forum fixes plan

**Verdict: approve with adjustments.** No blockers. The plan is the right size and its two corrections are correct. What follows tightens the integration step and adds a few details to the test and docs steps.

## What I inspected (static only, nothing executed)

- PLAN.md.
- A full `diff BASELINE-server.py server.py`.
- `server.py` 1294–1330 (cache key), 2988–3001 (parked fallback, not reviewed further) and 3110–3185 (the tap predicate).
- `pipeline/enrichment.py` 1173–1240: `_is_fetchable_article_url`, `_NO_ARTICLE_SOURCES` and `_article_fetch_eligible`.
- `pipeline/codex_forum.py`.
- `pipeline/ranking.py`: `Embedder` (707–850) and `get_or_compute_embeddings` (900–935).
- `tests/test_codex_forum.py` 1–60, 215–300 and 333–515.
- `tests/test_server.py` 5422–5485 (LessWrong tap) and 7363–7420 (Codex tap).

I did not run any tests, Ruff, ty or diff checks. The evidence in PLAN.md is about the unchanged implementation and does not prove the proposed changes work.

## Confirmed findings

**Step 1: the predicate is correct (confirmed).**
- The only change to tap behavior relative to the baseline is at server.py:3177–3180. LessWrong is in `_NO_ARTICLE_SOURCES` (enrichment.py:1192), so the current code stops a LessWrong row from fetching an article when `src_kind is None`, which happens when both `self_text` and `top_comments` are non-empty. The baseline allowed that fetch.
- Reddit and generic sources are unchanged: the server predicate never had the Reddit `startswith` check.
- `effective_codex_story` returns non-Codex stories unchanged, and `_tldr_cache_key` keeps `_llm_cache_identity()` for non-Codex sources. So `story.source != AINEWS_SOURCE and not is_codex_source(story.source)` brings back exact baseline behavior for every non-Codex source.
- Codex still needs its explicit exclusion. The effective view sets `article_body=""`, and line 3121 turns that into `None`, so without the exclusion a Codex tap would qualify for an article fetch.
- After the change, `_NO_ARTICLE_SOURCES` has no other use in server.py, so dropping that import is safe.

**Step 2: the fake subclass is safe (confirmed).**
- `Embedder.__init__` (ranking.py:716) is where the tokenizer and ONNX session are loaded. A subclass that overrides `__init__` without calling `super()` loads nothing.
- `Embedder` has no `__del__` or `__getattr__`.
- `get_or_compute_embeddings` only reads `model_version`, `embedding_dim`, `model_onnx_sha` and `encode`. The fake overrides all four, and it inherits `model_manifest`/`max_tokens` as class attributes. The missing instance attribute `batch_size` is never read.
- The override signature `encode(self, texts: list[str], batch_size: int | None = None) -> NDArray[np.float32]` matches ranking.py:812–814.

## Adjustments

1. **Low, step 2: the annotation needs module-level imports.** `numpy` is currently imported inside `encode` (test_codex_forum.py:345), and the file has no `numpy.typing` import. `from __future__ import annotations` (line 7) stops this failing at runtime, but ty and Ruff still need the names to resolve.
   - Add `import numpy as np` and `from numpy.typing import NDArray` at module level, or under `TYPE_CHECKING` for `NDArray`.
   - If ty narrows the return of `np.array(..., dtype=np.float32)` to `NDArray[Any]`, accept that as assignable. Do not add a cast or a new ignore.
   - Also delete the "deliberately not an Embedder subclass" comments at lines 369 and 507–508, not just the ignores.

2. **Low, step 3: pin the regression to the condition that distinguishes old from new.** Model it on test_server.py:7363.
   - Use a real `lesswrong.com/posts/<id>/…` URL so it passes `_is_fetchable_article_url` and `_extract_lesswrong_post_id`.
   - Use `self_text` shorter than 500 characters and non-empty `top_comments`, which keeps `src_kind is None`.
   - Make `_fetch_lesswrong_context` raise, which proves the LessWrong context lane is not used.
   - Assert `article_calls == [url]` and that the stub's body reaches `generate_detailed_tldr`.
   - With these settings the test fails on the current code (0 fetches) and passes on the fix (1 fetch), so it is meaningful.
   - Keep test_server.py:5422 as it is: it covers the complementary case where `src_kind == "lesswrong"`.
   - Nothing more is needed for AINews or Reddit, since neither changes.

3. **Low, step 5: name the stale comments explicitly.**
   - server.py:3177–3179 says "Same exclusion set as the proactive selectors". Replace it with a comment saying the tap excludes only AINews and Codex, and why LessWrong is deliberately left out.
   - The `codex_forum.py` docstring (lines 8–9) points to `_NO_ARTICLE_SOURCES` as the mechanism for "every article lane". The tap now uses `is_codex_source`, so that reference should mention both.

4. **Medium, step 5: integration lacks a post-merge check and a clear location (gap in the plan, not a confirmed error).**
   - Step 4 runs the gates against base 62af20e, but the merge target is fa9f6f9. "Check integration diff, then CI" assumes CI exists and covers backend and ty, which the plan does not show.
   - Recommendation: merge main into the feature branch inside the isolated worktree, not in the main checkout. Resolve conflicts there, then re-run only the affected tests plus ty and Ruff on the merged tree. Repeat the full suite only if the merge touched server.py, pipeline or tests. Then fast-forward main.
   - Before any fast-forward, confirm `git status` in the main checkout is clean or leave it alone; "preserve all existing WIP" has to cover that checkout as well.
   - Use `git diff fa9f6f9..HEAD --stat` to confirm the TUI commits and the parked note survive, and that STATUS/WORKLOG show only additions.

5. **Low, steps 4–5: make the commit manifest explicit.**
   - Before committing, list the intended paths (server.py, the two test files, docs, CHECKS.md) and stage them by name rather than with `-A`.
   - The worktree holds uncommitted implementation work, and step 4 may leave artifacts behind (caches, fixture DBs).

## Unverified or hypothetical

- I have not confirmed that ty clears both diagnostics once the subclass change is made. The change is likely to clear them, but that needs the ty run in step 4.
- I did not re-review the parked clean-summary fallback (server.py:2988–3001); it is kept as a constraint, as instructed.
