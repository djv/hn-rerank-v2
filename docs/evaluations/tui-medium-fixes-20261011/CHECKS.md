# TUI medium fixes — 2026-10-11

## Scope and behavior

The user requested the two medium TUI fixes: queued votes must survive normal
quit, and HTTP 5xx must not hide otherwise readable stories. Polling/counts
findings and Muse's separate Codex Forum implementation are outside this patch.

- q/Ctrl+q capture all unfinished vote workers, including queued undo, and
  observe completion for at most five seconds. Existing FIFO serialization is
  unchanged. No `Worker.wait` or cancellation-producing waiter is used.
- New vote/undo actions are blocked during the drain and through exit teardown.
  Timeout keeps the reader and requests alive. Failed/cancelled requests keep
  the reader open; a later explicit quit can exit. Failed votes are not retried.
- A failure during draining retains the latest failed story ID and transport
  error even if later queued votes succeed. An already-sent request may have
  reached the server despite an error; no server-side loss claim is made.
- All HTTP 500–599 statuses raise the existing `TransientError`. Initial summary
  failures keep the selected story and offer manual retry. Existing summary
  text survives refresh failure. HTTP 404 remains a permanent failure control.
- Background cache 500/502/504 now match the pre-existing 503 behavior: selected
  readers receive a retry message instead of immediately attempting generation.
  Older-server 404 cache responses still fall back to direct summary retrieval.

## Independent review and verification

The user approved one read-only Claude Opus cloud result review. Session:
https://claude.ai/code/session_01QgJhf8zZsvsMZV5RDBu1y3
Private review-drop branch: `r/20261011-042326-hn-tui-medium`.
Only code, diff, tests and a prompt were bundled. No production data, profile
credentials or private context. [Report](CLAUDE-REVIEW.md),
[prompt](REVIEW-PROMPT.md), [reviewed tracked diff](REVIEWED.diff).
Blank context lines in the saved diff omit their single-space marker for the
repository whitespace gate; `git apply --numstat` still parses the patch.

The reviewer approved the approach with one small correction and identified
two introduced low issues. Both were verified against the actual patch:

1. The quit guard reset after exit request. A deterministic Pilot regression
   called vote before teardown and showed the optimistic rated set change from
   empty to `{1}` after exit. Keeping the guard set fixes that failure.
   Textual 8.2.8 `App.exit` sets `_exit` and posts `ExitApp`; the reviewer had
   not inspected its installed source. The regression confirms the action-body
   exposure, rather than proving a physical terminal keypress race.
2. The drain replaced the vote failure with a generic notice. A regression
   failed because story/transport detail was absent. The latest failed-story
   message is now retained and included in the final quit warning.

The report also notes pre-existing profile/setup cancellation and setup/input
quit gating. They are unchanged and are not claimed fixed. Review was static;
the reviewer did not run tests. The final small corrections were verified by
the main agent's regressions, not a second independent review.

## Test evidence

Base: `bfc24944857a411460efd8b55ed34ff085bfbd9d`.
Isolated VPS worktree: `/tmp/hn-tui-medium-20261011`, branch
`agent/tui-medium-fixes-20261011`. No production DB, provider, profile or vote
operation. Full suites run with two-CPU/3 GiB limits and one-thread libraries.

- Original regression reproduction against unchanged source: 7 failed / 3
  passed. It demonstrated early quit/cancellation and 500/502/504 hiding.
- Initial focused gate: 53 passed / 1 skipped.
- First full TUI gate: 216 passed / 1 failed / 1 skipped. The failed older test
  expected automatic cache-500 fallback; expanded it to assert 404 fallback
  and cache-500 retention/manual retry. Next full gate: 218 / 1 skipped.
- Review regressions failed before their respective corrections.
- Final focused client/reliability/prefetch tests: 69 passed / 1 skipped.
- A subsequent full gate found a pre-existing editorial test expecting the
  vote-error prefix. Kept that prefix while adding the failed story ID; final
  reliability/editorial focused gate: 66 passed.
- Final full TUI gate: 219 passed / 1 skipped, 62.19 s (Windows ACL test).
- Backend full suite: 1,264 passed / 1 skipped, 72.06 s. Existing sklearn MLP
  convergence warning. Backend source/config unchanged by later review fixes.
- Repository Ruff and five touched-Python format checks passed. Root ty with
  `--extra-search-path clients/tui/src` and TUI ty passed. The explicit search
  path avoids resolving the installed production client beside scratch sources
  (otherwise an archived prototype gets a duplicate-API-type diagnostic).
- [Source hashes](SOURCE-SHA256.txt) match laptop and tested VPS client files.
  Diff whitespace check passed.

## Runtime bounds and handoff

### Standalone CI follow-up

Backend CI for `3875d82` passed, including browser tests. All three standalone
OS jobs reached their static checks after tests: Linux/macOS reported
219 passed / 1 skipped; Windows 220 passed. Their newer Ruff rejected the new worker import order
and an existing editorial-test import order. Corrected those two blocks.
Reproduced with an outside-workspace VPS package at
`/tmp/hn-tui-standalone-SD2aku` using Ruff 0.17.0 and ty 0.0.86.
The newer ty also caught three existing BINDINGS overrides using an invariant
narrow list type; changed their annotations to Textual's public BindingType,
matching its superclass declaration. No binding values or behavior changed.
Standalone lint/type/format passed after those corrections; full suite 219
passed / 1 skipped (65.96 s), final-source reliability/editorial 66 passed,
and source distribution/wheel builds passed. Root Ruff and ty also pass.
The built wheel's outside-source entrypoint reports version 0.1.0.
These are import-order and annotation changes only; backend checks are reused.
Replacement CI is triggered by the follow-up commit; inspect the commit's runs
for the final cross-platform wheel-install/headless results.

`da91687` backend CI and Linux/Windows client test/lint/type/build/install/
headless jobs passed. macOS found a race in the expanded cache-failure test:
the summary cache was filled before the selected pane's asynchronous update,
so the final assertion still saw "Loading summary…". The test now waits for
both the cache result and rendered selected summary. The fetched CI failures
reproduce that distinction; no runtime code change or added sleep. The updated
404/500 cases and standalone lint/type/format pass on the VPS.

`4c17ed7` backend, Linux and macOS CI passed, including wheel install/headless
startup. Windows failed the unchanged narrow filter-order assertion twice
(original run and one failed-job retry); the same runtime code passed Windows
in `da91687`. The test assumed one idle turn completed Select's asynchronous
filter change. It now waits, bounded to five seconds, for selection of the
expected first story and retains the highlight/ID/scroll assertions. No filter
implementation change. Replacement cross-platform CI verifies that test change.

No personal reader was restarted. The final process check found no local reader
running. The local root-venv entrypoint's editable path resolves to
`/home/d/code/hn-rerank/clients/tui/src`; its next launch loads this code.
Physical terminal voting/quit/undo and natural
HTTP 5xx retry remain unverified. No backend restart is required for this
client-only change, and no backend deployment is claimed.

The separate Muse source-fix runner exited with status 0 after reporting 96
focused passes. Its full-suite tool call was rejected; no CHECKS.md or completed
commit was present at the checkpoint. Its isolated WIP is preserved and still
needs supervising validation before integration. Ranking research and the
existing interleaving trial remain unchanged.
