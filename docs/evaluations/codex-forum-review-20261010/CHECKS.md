# Codex Forum integration checks — 2026-10-11

The user chose **Fix and integrate** after the remaining-fixes plan review.
Scope: the existing Muse source-safety implementation plus restored LessWrong
tap behavior and typed test fake. The conditional clean-summary fallback stays
[parked](PARKED-FALLBACK.md). No production rollout is authorized here.

## Source and integration

- Muse's original isolated branch: `agent/codex-forum-fixes-20261011`, base
  62af20e, `/home/dev/hn-rewrite/hn-codex-forum-fixes-20261011`. Its uncommitted
  work was preserved. [Receipt](MUSE-ACK.md).
- Verified integration checkout: `/tmp/hn-forum-integrate-20261011`, branch
  `agent/codex-forum-integrate-20261011`, based directly on current main
  fa9f6f99d27bfec8502a9b751618c1880c9e1f61. This avoids replacing newer TUI,
  STATUS, WORKLOG and parked-note changes with the older branch copies.
- The six previously tracked Python paths and ARCHITECTURE.md had no intervening
  main changes since the Muse base. Copied the eight changed/new Python files
  plus the architecture change and receipt, then applied the reviewed remaining
  corrections. Updated current main's documentation rather than copying Muse's
  stale STATUS/WORKLOG. No merge conflict or history rewrite was needed.
- [SOURCE-SHA256.txt](SOURCE-SHA256.txt) records the exact eight final Python
  files. Laptop working sources and the tested VPS integration sources match.
  `clients/tui/`, config.toml and PARKED-FALLBACK.md are unchanged from fa9f6f9.

## Behavior

Codex Forum is excluded from proactive article fetches and short-post article
taps. Generated terminal Discourse footers are removed before RSS text
conversion. Candidate, training/feedback, embedding/hash and summary inputs use
a source-specific effective view that omits legacy mixed article bodies and
recognized footers. Raw rows, identities and votes stay intact. Summary keys
are versioned for Codex, with legacy-key rejection including stale fallback.
No DB migration or cache deletion.

The tap excludes AINews plus Codex only; existing LessWrong behavior is
restored. Its new regression verifies one article fetch, zero context fetches
and the fetched article reaching the stubbed generator. The lightweight test
embedder now subclasses Embedder with an overridden constructor and matching
encode signature; it loads neither a tokenizer nor ONNX model.

## Reproductions and exact checks

Environment: a new integration-worktree environment from `uv sync --frozen`;
Python 3.12.3, Ruff 0.15.17, ty 0.0.54. No production environment modified.
Checks ran on the VPS with CPUQuota 200%, MemoryMax 3G, Nice 10 and OMP,
OPENBLAS and MKL threads set to one. `uv run --no-sync` below uses that environment.

- Before restoring the tap predicate,
  `pytest tests/test_server.py::test_tldr_detail_short_lesswrong_post_keeps_article_fetch -q`
  **failed as expected**: article_calls was [] instead of [url], 7.23s.
  The two original fake-embedder type errors were already reproduced during
  result review against the unchanged source manifest.
- `pytest tests/test_codex_forum.py tests/test_server.py -k 'codex or lesswrong or tldr_cache or stale_tldr or prefetch_tldr' -q`:
  **51 passed, 226 deselected, 4.00s**.
- `pytest tests/ -n 4 -q`: **1,301 passed, 1 skipped, 45.67s**. One existing
  sklearn MLP convergence warning; no failing tests.
- `ruff check .`: **passed**.
- `ruff format --check` on all eight changed/new Python paths: **passed**,
  eight already formatted.
- `ty check`: **passed**, both introduced diagnostics resolved.
- `git diff --check`: **passed**.

Full backend validation was run once after the corrections, against current
main plus this change. No repeat local full sweep or broad TUI sweep was needed.
The CI workflow separately runs backend tests, browser checks, lint, changed-file
formatting and ty on push/PR; completion requires its exact integration commit
to pass. These are pre-push check results, not a claim of a CI run from this file.

Peak sampled test-unit memory was about 1.1 GiB; sampled VPS load 3.22 during
the bounded suite and 1.60 afterward. Check units exited inactive successfully.

## Reviews and limits

- One authorized Claude cloud result review: [report](RESULT-REVIEW.md),
  session https://claude.ai/code/session_01Y5WngjCSN5NmPQTejRg8tG. Its reviewed
  source manifest is [REVIEWED-SOURCE-SHA256.txt](REVIEWED-SOURCE-SHA256.txt).
  The reviewer was static-only; supervising fixtures verified its findings.
- One subsequently requested remaining-fixes plan review:
  [report](REMAINING-PLAN-REVIEW.md), session
  https://claude.ai/code/session_01FbSqdUUM8GZufNDiL2mHrE. Approved with
  integration/test/import adjustments, applied within the reviewed scope.
- The raw result report's original medium fallback rating is historical:
  the user parked it after clarification as low. It requires both changed
  inputs/model identity and unavailable regeneration. Unchanged-input outages
  still serve the cache. No live occurrence was established.
- The report's absolute-path/untracked warning was a review-bundle artifact;
  both new files are included in the explicit integration manifest.
- No live production DB read/write, service restart, provider request, profile
  or vote mutation, model experiment, data cleanup or reply/count adapter.
  Tests use temporary databases and stubbed external helpers. Existing source
  policy tests do not exercise every production wrapper independently; the
  read-only call-site review and full suite supplement focused regressions.
- Production still has the old behavior until separately approved deployment.
  Natural feed refresh and real summary generation after rollout remain live
  verification steps; no deployment success is claimed here.
