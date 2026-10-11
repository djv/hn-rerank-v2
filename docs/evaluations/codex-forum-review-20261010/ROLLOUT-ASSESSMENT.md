# Verified rollout-plan review — 2026-10-11

One authorized independent Claude cloud review completed:
https://claude.ai/code/session_01NNptS9EXt9DtptFpFnPAMp
Mailbox `r/20261011-163226-hn-forum-rollout-final`.
The initial plan is preserved as ROLLOUT-PLAN-REVIEWED.md; the full report is
ROLLOUT-REVIEW.md. The reviewer rehashed all eight sources and ran no tests.

The reviewer rated the original plan not ready, with one conditional high,
four medium and three low observations. The agent verified against the actual
source and live read-only state, then amended ROLLOUT-PLAN.md. No code fix,
deployment or second independent review was performed in this task.

## Conditional high: side-vector coverage

The mechanism is real: ranking.py normalizes Codex input hashes; side vectors
use the same hashes, and side_embeddings.py:171-178 falls back below 98%.
Server startup does not write these vectors. scripts/embed_side_vectors.py
updates them from the timer's checkout. The report's all-user fallback is a
conditional scenario, not a demonstrated current production failure.

A read-only target-code check used Database(read_only=True), the production
candidate loader, per-profile feedback exclusion, configured URL training dedup,
SideVectorCache.get and primary embedding-cache reads. It loaded no model,
wrote no DB data and printed no profile identifiers, text, URLs or credentials.
At 16:38 UTC: 12,550 candidates, 28 Codex candidates, zero Codex feedback rows;
all 17 profiles with feedback retain 99.7769–99.8374% target side coverage.
Exactly 28 missing target side vectors in those checked profile spaces are
Codex; 28 primary Codex vectors also need recomputation.
These pass 98%; an immediate extra side job is unnecessary under this snapshot.

A separate direct cold/shared-pool read during save at 16:44 UTC found 12,560
candidates, 12,499 matching side vectors and 99.5143% coverage, still above
98%. Normal ingestion changed the pool and created additional side misses;
the earlier per-profile results are historical. Both gates require a fresh
check at deployment rather than treating this snapshot as fixed.

The timer was active, next scheduled 17:05 UTC, with successful 16:35:27–36 run,
working directory `/home/dev/hn-rewrite/main` and limit 2,000 / two threads.
The amended plan refreshes coverage/timer state, aborts before deployment if
coverage fails, and watches the actual warning/trace and post-start coverage.
No temporary model fallback or extra embedding job is authorized.

## Medium corrections

- Backup now specifies read-only source, five-second busy timeout, stepped
  Python backup, 120-second progress deadline, 180-second outer timeout and
  separate capped quick_check/hash operations. Free-space checks cover both
  destination and possible WAL growth. Interrupted partial backups remain private
  and are not treated as valid; no retention deletion is authorized.
- Exact-key DB comparison was removed: the deployed key depends on live
  provider/model environment. Cache GET remains a bounded hit/miss check;
  exact-key verification and fresh generation are explicitly unverified.
  No broad environment or secret-file read is needed for that smoke.
- An unobservable generic "source-policy failure" log criterion was replaced
  with named traceback/result signals and pure-helper assertions.
- Primary re-embedding is now counted: 28 missing Codex inputs. Existing
  unrelated one-text perf records give only an illustrative 9–44 second
  estimate, not a benchmark/SLA. Actual encode events and endpoint behavior
  decide the bounded startup outcome; slow refresh alone is reported as a gap.

## Low corrections and limits

Target existence/fetch, branch/worktree inventory and detached rollback ref
semantics are explicit. Rollback restores old article lanes and legacy key
reuse as well as possible further vector churn. No DB restore overwrites votes.

Readiness zero is only liveness. The amended plan samples a baseline before
normal refresh and then waits for baseline+1; the review's suggestion to take
current+1 after the first refresh could wait an extra hour, so it was corrected.
If sampling is too late, version advancement stays explicitly unverified.

Ready for a separate deployment decision subject to the amended live gates,
based on the agent's verification. This is not a second independent approval.
Source 794caa4 already passed 1,304 backend tests / one skip, 37 focused tests,
Ruff/ty/format, and exact-target CI 38155643087. No repeats were needed for
this documentation-only plan correction. User `ss` finalizes this handoff;
rollout remains deferred until explicit authorization.
