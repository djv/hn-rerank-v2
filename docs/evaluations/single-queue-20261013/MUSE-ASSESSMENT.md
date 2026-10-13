# Independent Muse result review — 2026-10-13

Target: `5b3b996` against `4c17918`, using Muse Spark 1.3 contributor,
xhigh, through pi on the VPS with read/grep/find/ls tools only. The prompt
excluded earlier reviews, project history, databases, secrets and personal
context. Raw report: `REVIEW-step1-muse.md`. Verdict: PASS, no High issues.

## Agent verification

- M1: the source skips starting a new dwell segment when a carried views
  row has no rank. The claimed attribution to a *different* prior story
  was not reproduced: the carry path retains the currently open story.
  The older version's segment may continue. Further isolated reproduction
  is needed before calling this a false-story event.
- M2: filtering restored IDs before enumeration compresses ranks, reproduced
  with order `[1,2,3]` and restored `{2}`: `(1,0),(3,1)`. This is deliberate
  for client-inserted undo rows (existing regression test). A ready feed is
  rebuilt in `set_feed` before `restored` is cleared in `refresh_feed`, so
  a new round built in that interval can also filter a genuinely served
  restored card. This remaining Medium affects position analytics; version
  and story attribution remain separate and intact. No behavior edit made.
- M3: session storage grows with earlier rounds, as required for stable
  slots and undo. The suggested prune to `entries` would retain all those
  IDs anyway. No demonstrated resource failure; long-session growth remains
  a limitation.
- M4: confirmed new-client/old-server interaction payload rejection.
  Deploy the new server before the user restarts the TUI. Backward
  compatibility for older clients against the new server is tested.
- L5: retaining unknown-time cards is the explicit current contract.
- L6: counts patching omits history/restored objects; a counts update can
  be replaced by an older undo object. Low cosmetic issue, left unchanged.
- L7: injected naive datetimes use the local timezone; production uses an
  aware UTC clock. No production startup-sort failure demonstrated.
- L8: the window and selection guards reject a delayed impression after
  switching windows. No false attribution demonstrated.

The user's existing instruction authorizes merge/deploy when no High issue
remains. Remaining Medium/Low observations are recorded, not silently fixed.
Only the separately approved import sorting in `test_queue.py` was changed.

## Verification and rollout gate

Branch backend GitHub CI passed. Terminal client CI ran 272 passing tests
and one skip, then failed on two I001 import-order errors in `test_queue.py`.
Those imports are now sorted using the standalone package configuration;
root and standalone Ruff/ty checks passed.

Independent VPS backend rerun: 1,445 passed / 1 skipped (123.88 s).
Standalone TUI at eight workers while backend checks also ran: 271 passed /
1 skipped / 1 failure, the existing short-deadline quit test. It reached
the exit cap before the assertion that the held vote was not yet cancelled.
A lower-concurrency full rerun is the final TUI gate; no test weakened.
That four-worker standalone rerun passed: 272 passed / 1 skipped in 66.45 s.
Root Ruff/ty and formatting of all 14 branch-touched Python files passed.

Private online pre-deploy backup:
`/home/dev/backups/hn-rewrite/single-queue-20261013T2057Z/hn_rewrite.db`.
`quick_check=ok`, SHA256
`4fb37049f9ac19cf42020764d5f679d965fd7d112aa902ee87927928d5906669`.
Backup validation is not restore verification.

Rollout: preserve config/lock/unrelated archive hashes, merge without history
rewrite, fast-forward VPS main, restart `hn_rewrite.service` once, then check
existing-profile dashboard/feed/readiness, cached and uncached summary
requests, additive columns, Import AI gate and bounded journal. Prior code:
`b1897f4c327c2de298ae443213b5344d1e77e2d4`. On smoke failure, stop further
rollout and report before any additional restart or rollback; a code-only
rollback retains the additive columns and all accumulated feedback. Never
restore the pre-deploy database over subsequent user data.
