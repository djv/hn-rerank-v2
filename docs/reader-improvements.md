# Reader improvements: small, verified scope

## What the code actually does

- `server.py` already distinguishes cached/generated responses, marks stale
  fallbacks, and marks salvaged half-summaries retryable. `API.Summary`
  already has `provisional`. No new status framework is needed.
- A cooldown during a requested refresh returned an exact-key cached response
  as an ordinary hit, concealing that refresh did not happen. Locally fixed:
  reuse `retryable`, plus a reason and the existing cooldown delay.
- TUI manual refresh cleared every cached summary. `load_summary` replaced
  the pane with Loading, then hid the story on API error. Locally fixed:
  preserve the selected cached summary during manual regeneration and keep
  it/selection on failure. Show a short message for provisional responses.
  Invalidate/cancel older summary work immediately when refresh starts.
- `is_summarizable` accepts any nonempty source text, while generation ignores
  self text below 300 characters (and can fold a thin article into discussion).
  This mismatch exists, but changing selection thresholds could hide useful
  short posts. Do not tighten eligibility without representative cases.
- Contentless RSS stories are filtered before ranking; background article
  fetching selects from ranked stories. Such rows can miss background enrichment.
  This verifies a possible path, not the cause of every low-coverage source.
- `select_article_fetch_candidates` skips any nonempty article body, even a
  short one. Low coverage counts alone do not distinguish scheduling, fetch
  failures, short legitimate content, or inactive sources.

## Tightened plan

1. Refresh: keep existing controls and API shape. Ship the small fixes above.
   No new key scheme, new metadata table, or refresh state machine.
2. Summaries: retain fixed 240-word single / 120+120 combined targets and bold
   instructions. Review a small mixed set of actual outputs; tune prompts only
   for demonstrated issues. No viewport parameters, automatic repair calls,
   or new extraction rules. Word targets are not guaranteed screen fit.
3. Coverage: inspect a few missing-content rows from Hugging Face, Lobsters,
   OCaml.org and Asterisk alongside existing fetch-failure records. If scheduling
   is the cause, add a bounded missing-content selection to the existing worker
   within its current budget. Do not introduce another queue or raise load first.

No feed removals, ranking changes, DB cleanup, or automatic mass regeneration.
Local changes are not deployed or loaded into the user's running TUI.

## Checks

Cooldown regression verifies retryable/reason/delay without an LLM call.
TUI integration tests exercise manual refresh with an existing summary followed
by either HTTP failure or a retryable response; text and selection survive.
Backend: 821 passed. TUI: 94 passed, 1 skipped. Ruff/format/ty clean.
Live verification remains pending an explicitly requested deployment/restart.
