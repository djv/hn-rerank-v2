# Codex Forum rollout plan — 2026-10-11

Review target only. Deployment has not been authorized. The user requested
one independent review of this plan after the medium footer-search fix.

## Pinned target and current evidence

- Deploy exactly `794caa4671d7b9276eb4843cb8e69f2a601a0892` from the user's
  repository `djv/hn-rerank-v2`; do not substitute a newer main.
- Production checkout and service working directory are
  `/home/dev/hn-rewrite/main` (a real Git worktree, not the repository parent).
  Git common directory is `/home/dev/hn-rewrite/.git`.
- Read-only preflight at 16:30 UTC: production HEAD
  `e1d3fe30301c15ce7a93c12ccf0f9e3a99003ff3`, tracked tree/index clean.
  Untracked `VPS-COLD-ARCHIVE-20261011.json` is unrelated and must be preserved;
  it is absent from the target. Inventory again before changing Git state.
- `hn_rewrite.service` active/running, process started 03:26:55 UTC.
  ExecStart `/home/dev/.local/bin/uv run python server.py`, port 8766.
  Checkout HEAD alone does not identify the already-loaded process code.
- Disk available 23,755,079,680 bytes; DB 1,196,064,768 bytes, WAL 8,709,712.
  Approximately 4.9 GiB available RAM, load 0.29. Primary ONNX model and
  tokenizer exist. Fresh preflight must verify every configured model artifact
  needed by startup, including enabled side-model files; never read/upload secrets.
- Config SHA256
  `750bb70512febdcd9cbf3ecfb1d7dfe3cc5d6a2b2058999351dcae029c458446`;
  uv.lock SHA256
  `0fb6ffc27b3202f1a94d5f0d4f47baf552f6228d9fc6509e5c524f637fe3e198`.
  config.toml, pyproject.toml, uv.lock and TUI sources are unchanged in the
  production-to-target range. The range also carries the previously saved
  Terminal-client CI workflow change and project documentation.
- Exact target backend 1,304 passed / 1 skipped, 37 focused Codex passes,
  clean Ruff/ty/touched-file format. Existing MLP convergence warning only.
  [Exact target CI](https://github.com/djv/hn-rerank-v2/actions/runs/38155643087)
  completed successfully. Eight changed Python hashes are in the review manifest.

## Intended behavior and limits

Codex Forum topics leave article-fetch lanes. Generated terminal footers are
removed; legacy read views omit mixed article bodies without rewriting raw
stored fields, IDs or feedback. Derived vector hashes and summary inputs may
change and ordinary operation can recompute them.

The `codex-inputs-v1` marker causes **all pre-policy Codex summary keys to miss
once, even old summaries whose text happened to be clean**. Current normalized
keys can be reused. Other sources retain their existing key policy. Footer
matching now requires a separate count token and scans long digit/space runs
without repeatedly rescanning their tails.

The conditional clean stale-summary fallback remains parked at the user's
request: an old clean summary with a different current key may be unavailable
when fresh generation cannot run. A miss by itself is expected and is not a
reason to change that decision. Reply/count adapters and ranking research
remain outside scope. The two low implementation-review findings are unchanged.

## Execution after separate approval

1. Repeat host, exact HEAD, tracked/index/untracked inventory, unit working
   directory, resources, required model artifacts, config/lock hashes and target
   source checks. Stop on unexplained drift, tracked WIP, a colliding untracked
   path, missing startup artifacts or failed CI. The known non-colliding archive
   inventory must remain untouched; never reset, clean or stop another job.
2. Require snapshot-size estimate plus 2 GiB operating reserve before a backup.
   This is this plan's conservative gate, not a standing project policy.
   Use a timestamped location outside any Git checkout with restricted access.
   Take an online SQLite backup with bounded lock/overall timeouts, validate the
   backup with `PRAGMA quick_check=ok`, and record its size/hash locally. Abort
   on timeout/incomplete backup or less than 2 GiB remaining space. No retention
   deletion, credential copy or upload; the generic backup helper does those
   additional operations and must not be used. Validation is not restore proof.
3. Record prior exact HEAD as the code-only rollback target. Confirm target
   ancestry and fast-forward production with `git merge --ff-only TARGET`.
   Verify the eight-file manifest, config/lock hashes and tracked/index state;
   preserve the unrelated untracked file. Restart only `hn_rewrite.service`.
   Do not restart the user's TUI, change limits/config, or force global regen.
4. Verify a new PID/start time, active/running status, pinned checkout and unit
   working directory. Observe at most ten minutes of normal startup/refresh.
   Default initial regen delay is 30 seconds and startup-warm wait can be
   180 seconds; interval is 3,600 seconds. Read back actual configured timing.
   No extra provider or forced-summary request. Startup and authorized ordinary
   warming can invoke configured providers and write derived vectors/caches;
   approval must acknowledge that ordinary behavior.
5. Use the existing profile privately for localhost dashboard, one-week feed,
   readiness (include `min_version=0` initially) and Codex TLDR-cache GETs.
   Keep credentials and response bodies out of logs/shared artifacts. Do not
   create a profile, vote or call forced/uncached summary generation. Dashboard
   and feed GETs may queue ordinary warming. Record status, timing and readiness;
   accept a Codex cache 204 while its legacy key is invalid or no current cache
   exists, not as proof that fresh summary generation succeeded.
6. Included read-only source-policy checks: select minimal fields for one
   existing Codex row privately; hash its raw fields before/after normalization.
   Run the deployed pure helper on a copied Story to verify that the effective
   view omits article text and generated footer while leaving the raw object
   untouched. Calculate its deployed current key privately and compare a
   cache GET with the exact-key DB lookup, allowing legitimate concurrent
   prefetch changes by rechecking. Do not write rows, purge caches or hydrate
   the topic. A current-key summary, if normal warming produced one, can be
   inspected without generation. Successful HTTP status alone is insufficient.
7. Scan the bounded journal since this restart, disk/load/memory and ordinary
   refresh progress. Accept when endpoints/readiness work, pinned source/config
   checks hold, reserve remains and no new local exception, database-full error
   or repeated source-policy failure appears. Separate known upstream/provider
   failures from new local regressions. If normal refresh has not completed in
   ten minutes, report that live gap; do not force generation or claim full
   refresh/cache migration verified. Endpoint regressions persisting beyond
   startup's observation window trigger rollback.

## Code-only rollback and records

For failed start, persistent endpoint regression, new local/DB-full error or
resource-reserve breach: inventory Git/WIP again, then switch the service
checkout to a detached saved prior commit without force/reset/clean and restart
only this service. If intervening WIP prevents a safe switch, stop and coordinate;
never discard another agent's work. Repeat bounded service/GET/log verification.
No schema migration is introduced. New clean derived summaries/vectors can
remain; old code may recompute their hashes again. Do not restore the DB backup
over intervening feedback; DB restore is a separate recovery decision.

Record pinned/prior commits, source/config hashes, backup validation and path,
new start/PID, observed normal refresh/cache results, bounded log findings and
rollback status in project deployment evidence. Distinguish source simulation,
API behavior, normal generation and physical TUI rendering. No secret or raw
personal DB content belongs in Git or the cloud-review bundle.
