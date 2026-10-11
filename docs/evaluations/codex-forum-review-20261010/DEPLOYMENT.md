# Codex Forum source-safety rollout — 2026-10-11

The user explicitly approved the reviewed pinned rollout in the choice popup.
Deployed exactly `794caa4671d7b9276eb4843cb8e69f2a601a0892` to
`/home/dev/hn-rewrite/main`. Prior code-only rollback target:
`e1d3fe30301c15ce7a93c12ccf0f9e3a99003ff3`.

## Preconditions and backup

At 16:46 UTC the tracked tree/index were clean; the only untracked file was
`VPS-COLD-ARCHIVE-20261011.json`. Git's primary-worktree inventory lists the
repository parent, but `core.worktree` and `--show-toplevel` both identify the
actual service checkout `/home/dev/hn-rewrite/main`; this was verified before
the fast-forward. Other worktrees, including original Muse WIP, were preserved.

The pinned object was fetched from the user's origin and ancestry checked.
Exact target CI 38155643087 succeeded. Eight changed-source hashes match
ROLLOUT-SOURCE-SHA256.txt. Previously verified target: 1,304 backend tests /
one skip, 37 focused Codex tests, Ruff/ty/format clean; no repeat sweeps during
deployment. Config and lock hashes below were unchanged. All required primary
and side-model graph/data/tokenizer artifacts existed.

Fresh target-code read-only coverage at 16:46:59: 12,560 candidates, 29 Codex,
zero Codex feedback; all 17 profiles with feedback projected side coverage
99.5143–99.6461%, passing 98%. Twenty-eight Codex primary vectors required new
hashes; 29 Codex side vectors awaited the ordinary timer. The timer was active,
last successful at 16:35 and next scheduled at 17:05, using this service checkout.
No extra side job was launched.

Private online SQLite backup:
`/home/dev/backups/hn-rewrite/codex-inputs-20261011T164815Z/hn_rewrite.db`.
Backup size 1,196,339,200 bytes, file 0600 / directory 0700. Stepped read-only
source backup completed in 4.595 seconds; `quick_check=ok`; SHA256:
`8b0fb56016b17e37025866afd4db9591952c203738532d79d5cc47644f09f6f4`.
Free space 26,347,327,488 before / 25,150,980,096 after, comfortably over the
2 GiB reserve. Deadlines and disk-floor checks were active. No upload,
credential copy, retention deletion or restore. Validation is not restore proof.

## Restart and API behavior

Fast-forwarded only to the pinned target. The archive inventory's SHA256
matched before/after. Source manifest, config and lock matched before restart.
Restarted only `hn_rewrite.service` at **16:51:50 UTC**; MainPID 759874,
active/running, working directory `/home/dev/hn-rewrite/main`.
No personal TUI or other service was restarted.

Actual configuration: initial regen delay 30 seconds, startup-warm wait 180,
regen interval 3,600; side model enabled with 98% minimum coverage.
Initial primary encoding logged 31 texts in 3.805 seconds. Startup personalized
warm completed in 36.481 seconds. First regen waited seven seconds for it,
started 16:52:39 and completed 16:53:31. The following deck warm completed
16:53:44 in 13.359 seconds; trace `side_embeddings=on`.

Existing-profile localhost GETs only; credentials remained in memory and no
response body was saved/shared. Baseline at 16:52:03:

- Dashboard 200, one-week feed 200 (cold served version zero / not ready),
  readiness `min_version=0` 200 (liveness only).
- Target version 1791737512589, sampled before first normal regen.
- Codex cache GET for -1809311842: 204.

At 16:53:54, readiness for baseline+1 returned 200 / true. Feed 200 / ready,
served and target version 1791737512590, 33 unique cards including two Codex
cards. Readiness/feed/cache latencies 9.59 / 20.06 / 4.96 ms. The normal
refresh and deck advancement were observed without a forced regen or vote.

## Source policy, vectors and logs

The deployed pure helper on stored Codex sample -1809311842 shortened
self_text 7,058 → 7,019 characters by removing the footer. Stored article body
was empty, effective article body remained empty; raw object and reread DB row
were unchanged. Normalization was idempotent. A separate in-memory synthetic
reply-body fixture was omitted and its original copied object preserved.
This is deployed-source simulation, not a naturally contaminated-row test.

Known pre-policy cached topic -481200819 retained the previously verified
summary SHA256 `09b59cc2cad777d3de11227265f1be7fd90fac7469245c5d016b2d8b0578ba9c`;
its cache GET now returned 204. The old row was preserved and was not served.
This demonstrates the live legacy-cache miss; no independent exact current-key
calculation or fresh Codex generation was performed.

Read-only post-start coverage at 16:56:00: 12,559 candidates / 29 Codex.
All checked profile spaces remain 99.4267–99.5822% covered. Codex primary
missing count is zero; 29 side vectors still await the normal timer. Natural
other-source ingestion affects the overall coverage figures; they are snapshots.

Bounded journal scan from 16:51:49 through the verification probe: 362 records,
zero ERROR/traceback/database-full/side-low-coverage signals, five rank traces
with side model on, zero off, one completed regen and no provider-warning
matches. The first local journal parser rejected a JSON MESSAGE array; it was
corrected to decode binary/repeated messages and rerun. This was a verification
harness error, not a service error. Final source/archive/config checks passed,
service remained active and disk reserve/load/memory checks passed. No rollback.

Config SHA256:
`750bb70512febdcd9cbf3ecfb1d7dfe3cc5d6a2b2058999351dcae029c458446`.
uv.lock SHA256:
`0fb6ffc27b3202f1a94d5f0d4f47baf552f6228d9fc6509e5c524f637fe3e198`.
Execution receipts/scripts remain in `/tmp/hn-forum-deploy-20261011` on the
agent's laptop/VPS; backup contents and credentials were never bundled.

## Boundaries and next step

Ordinary startup/refresh/prefetch can use configured providers and update
derived caches/vectors. No extra generation POST, forced summary, new profile,
vote, cache purge, cleanup or embedding job was issued. No schema migration.
The parked fallback and two low implementation findings remain unchanged.

Fresh Codex summary generation and physical TUI rendering remain unverified.
The next scheduled side-vector refresh was verified in the read-only follow-up
below. A cache miss alone is expected, not a request to reopen the parked
fallback decision.

## Scheduled side-vector follow-up

Read-only follow-up on 2026-10-11: production HEAD remained 794caa4 and
`hn_rewrite.service` remained active/running, MainPID 759874, with its original
16:51:50 UTC start. The ordinary `hn-rewrite-side-embed.timer` fired at
17:05:26 UTC. Its service found 83 of 17,523 stories needing vectors, logged
`encoded 83/83: 0.245 s/story`, and exited successfully at 17:06:01 UTC
(`Result=success`, `ExecMainStatus=0`). No extra embedding job was launched.
The tokenizer-only PyTorch-not-found notice did not prevent ONNX encoding.

The existing read-only aggregate probe, without model loads or DB writes,
completed at 17:06:36 UTC. Across 17 profiles with feedback, current side-vector
coverage was 99.992039–99.994199%, above the 98% cutoff. All 29 Codex topics
had current primary and side vectors: both maximum missing counts were zero.
These are post-job snapshots; the earlier 17:05:57 probe overlapped encoding
and showed changing coverage, so it is not the final coverage evidence.

No service restart, forced generation, profile/vote mutation, cleanup or runtime
change was issued during the follow-up. Fresh summary generation and physical
TUI behavior remain untested; the parked fallback and low findings are unchanged.
