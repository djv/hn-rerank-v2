# Codex Forum rollout plan — 2026-10-11

Deployment has not been authorized. One independent review of the
[initial plan](ROLLOUT-PLAN-REVIEWED.md) completed after the medium fix.
The agent verified its findings and incorporated the corrections below;
the amended plan has not received a second independent review.
[Assessment](ROLLOUT-ASSESSMENT.md), [report](ROLLOUT-REVIEW.md).

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
  needed by startup, including enabled side-model files; never print or upload
  credentials. Side ONNX graph and its external data file were also verified.
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
- Read-only target-code coverage check at 16:38 UTC: 12,550 candidates,
  28 Codex candidates and zero Codex feedback rows. After per-profile feedback
  exclusion and configured training deduplication, all 17 profiles with feedback
  retain target-hash side coverage 99.7769–99.8374%, above the configured 98%.
  Exactly 28 Codex side vectors and 28 primary vectors need new hashes in each
  checked space. No model inference or production writes were run for this check.
- A separate cold/shared-pool read during save (16:44 UTC) found 12,560
  candidates and 12,499 matching side vectors: 99.5143% coverage, still above
  98%. Normal ingestion changed both denominator and missing-vector count;
  the 16:38 per-profile values are a snapshot, not an execution prerequisite.
- Side timer is active, uses `/home/dev/hn-rewrite/main`, and invokes
  `scripts/embed_side_vectors.py --limit 2000 --threads 2`. Last run succeeded
  16:35:27–16:35:36 UTC; next was 17:05 UTC. Its normal next run will load the
  then-current checkout. Refresh schedule and coverage immediately before rollout.

## Intended behavior and limits

Codex Forum topics leave article-fetch lanes. Generated terminal footers are
removed; legacy read views omit mixed article bodies without rewriting raw
stored fields, IDs or feedback. Derived vector hashes and summary inputs may
change and ordinary operation can recompute them.

Primary Codex vectors recompute synchronously in the server. Side vectors are
updated by the existing timer, not server startup. Missing side vectors use zeros
while coverage stays at least 98%; below that threshold ranking falls back to
primary vectors only. Current projected coverage passes, so this plan waits for
the normal side timer and does not authorize an extra embedding job. If refreshed
target-code coverage falls below 98% for any checked profile, stop before rollout;
accepting fallback or running an extra side job requires a separate decision.
New anonymous/cold profiles use the shared candidate pool; the latest measured
coverage was 99.5143%, and this gate must also be refreshed at execution.

The primary recompute is currently 28 Codex inputs, batch one, capped at 4,096
tokens. Bounded existing logs showed 0.31–1.57 seconds per unrelated single-input
encode, suggesting roughly 9–44 seconds for 28 comparable inputs; this is an
estimate, not a Codex benchmark or restart SLA. Record actual `embedding_perf`
and `embedding_slow` events. A slow first refresh alone is a live verification
gap; persistent endpoint unavailability still triggers the ten-minute stop rule.

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
   Recompute the aggregate target-hash side coverage using read-only DB access,
   production candidate loading and configured training deduplication; require
   every checked profile and the cold candidate pool to remain at least 98%.
   Record Codex candidate/feedback counts and timer state, checkout and last run.
2. Require snapshot-size estimate plus 2 GiB operating reserve before a backup.
   This is this plan's conservative gate, not a standing project policy.
   Use a timestamped location outside any Git checkout with restricted access.
   Re-read free space immediately beforehand. Use a read-only Python sqlite3
   source connection with 5-second busy timeout, `src.backup(dst, pages=256,
   sleep=0.1)`, and a progress callback enforcing a 120-second elapsed deadline
   and 2 GiB free-space floor. Wrap backup in a 180-second hard process timeout
   (five-second kill grace), so write-induced restarts or blocked calls cannot
   run indefinitely. The source remains live in WAL mode; its WAL may grow.
   Create the destination exclusively with mode 0600 under a mode-0700 directory
   `/home/dev/backups/hn-rewrite/codex-inputs-<UTC>/`. Run backup `quick_check` and
   SHA256 separately with 120-second hard caps each, recording size/hash locally.
   Abort on timeout/incomplete backup or less than 2 GiB remaining space. No retention
   deletion, credential copy or upload; the generic backup helper does those
   additional operations and must not be used. Validation is not restore proof.
3. Record prior exact HEAD as the code-only rollback target. Confirm target
   exists with `git cat-file -e TARGET^{commit}` (fetch own origin if absent),
   its ancestry, the production branch and worktree inventory. Fast-forward
   production with `git merge --ff-only TARGET`.
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
   `min_version=0` is liveness only. Record the pre-refresh current version,
   then await a normal refresh and request readiness for that baseline plus one;
   check the feed serves at least that version. Do not take a new baseline after
   the refresh and accidentally wait for a second hourly refresh. If the first
   refresh precedes sampling, use the bounded restart/refresh trace and current
   ready deck, and explicitly mark version advancement unverified.
6. Included read-only source-policy checks: select minimal fields for one
   existing Codex row privately; hash its raw fields before/after normalization.
   Run the deployed pure helper on a copied Story to verify that the effective
   view omits article text and generated footer while leaving the raw object
   untouched. Cache GET verifies only the observable hit/miss, not an independently
   calculated exact key: calculation needs the live provider/model environment,
   and this plan does not inspect API-key-bearing environment or secret files.
   Do not write rows, purge caches or hydrate
   the topic. A current-key summary, if normal warming produced one, can be
   inspected without generation. Successful HTTP status alone is insufficient.
7. Scan the bounded journal since this restart, disk/load/memory and ordinary
   refresh progress. Accept when endpoints/readiness work, pinned source/config
   checks hold, reserve remains and no new local exception or database-full error
   appears. Concrete policy signals are tracebacks mentioning `codex_forum` or
   `effective_codex_story`, `tldr_detail` result events for the sampled topic,
   and the step-6 helper assertions. Watch `side_embeddings_low_coverage` and,
   when emitted, the `side_embeddings` trace label. Recheck aggregate coverage;
   persistent coverage below 98% or a confirmed model-space `off` beyond startup
   is a rollback criterion, not an accepted model change. Separate known upstream/provider
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
remain; old code may recompute their hashes again and side coverage may dip.
Old code reopens Codex article lanes and can reuse legacy summary keys or write
new mixed article bodies/summaries. Record that restored source behavior.
The branch ref remains at TARGET while rollback HEAD is detached; record this
explicitly so a future checkout/pull does not inadvertently redeploy TARGET.
Do not restore the DB backup
over intervening feedback; DB restore is a separate recovery decision.

Record pinned/prior commits, source/config hashes, backup validation and path,
new start/PID, observed normal refresh/cache results, bounded log findings and
rollback status in project deployment evidence. Distinguish source simulation,
API behavior, normal generation and physical TUI rendering. No secret or raw
personal DB content belongs in Git or the cloud-review bundle.
