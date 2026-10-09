# Seven review fixes: checks and bounds

User authorized Muse to fix all seven issues after two independent Codex
reviews of commit 5129813. Actual reports: CODEX-RESULT-REVIEW.md and
CODEX-SECOND-REVIEW.md. No additional reviewer launched. The user subsequently
authorized save/push and deployment; deployment evidence will be recorded
separately after the rollout. These results describe the pre-deployment gates.

Implemented: strict hydration shape and matching identity validation before
writes (missing identity tolerated only for non-strict legacy callers);
repeat-r joins consume forced intent; vote/undo acknowledgements invalidate
older feed replies without losing pending targets or restored rows; stale
reply paths drain a queued sort refresh; reconnect cancels/resets profile
stats and regeneration state; complete warm-flight replies include captured
snapshot and legacy alias while provisional replies remain unknown;
discussion-snapshot wording replaces coverage claims; hidden-story notice
names reconnect instead of r.

Red-first regressions in isolated VPS scratch:
- Original baseline: 12 backend failures / 5 passed and all 4 new TUI cases
  failed. Covered malformed hydration, warm reply metadata, repeated r,
  delayed constructed feed after acknowledgement, profile cleanup and hint.
- Follow-up: 3 strict missing/null identity cases and 1 stranded sort follow-up
  failed against first-pass fixes; wrong identities, valid/legacy identity
  variants, target-covering feed and restart cases already passed.
- Final focused: 25 hydration cases; 7 new TUI cases pass. Existing flight/sort
  regressions also passed in the first focused checks.

Final full suites, sequentially in /tmp/hn-section-seven-fixes-20261009:

Backend: VIRTUAL_ENV=/tmp/hn-section-refresh-impl/.venv
/home/dev/.local/bin/uv run --active --no-project pytest tests/ -n4 -q -rs
-p no:cacheprovider
Result: 1264 passed / 1 skipped / 1 existing MLP convergence warning, 28.48 s.
Skip: browser module cannot import playwright.sync_api.
VPS log: /tmp/hn-seven-final-backend.log.

TUI: working directory clients/tui;
VIRTUAL_ENV=/tmp/hn-save-20261009-sYcAQL/.venv;
PYTHONPATH=/tmp/hn-section-seven-fixes-20261009/clients/tui/src;
/home/dev/.local/bin/uv run --active --no-project pytest tests/ -n4 -q -rs
-o asyncio_mode=auto -p no:cacheprovider
Result: 202 passed / 1 skipped, 52.27 s.
Skip: Windows DACL verification on Linux.
VPS log: /tmp/hn-seven-final-tui.log.
Both commands used OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.

Root verification: all six changed source/test hashes match laptop and VPS;
SEVEN-FIXES-SOURCE-SHA256.txt additionally covers all tracked executable Python,
templates and project config/locks plus new regression modules, excluding
historical docs prototypes. sha256sum -c passed for every entry on VPS.
Local uv run --no-sync ruff check ., ty check, format check of all six changed
Python files, and git diff --check pass. No test-source or runtime change after
these gates; subsequent changes are documentation only.

Bounds: simulated/mocked regressions and exact-source suites, not physical TUI
or live source/provider smoke. No live DB migration, service restart, deployment,
provider calls, ranking experiment or production trial change. Vote ack rejects
any already-issued feed reply; absent a queued sort refresh, later polling or
navigation fetches the next feed. Ordinary summaries and count provenance remain
unchanged outside the seven fixes.
