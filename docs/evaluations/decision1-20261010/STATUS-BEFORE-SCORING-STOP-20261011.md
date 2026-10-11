# HN paired scoring — 2026-10-11

## Objective
The user explicitly requested `rv and do the scoring`, authorizing complete
frozen held-out real/empty scoring. Run the reviewed free VPS CPU route; no paid
provider/GPU job, Colab, reserved confirmation access or deployment.

## Verified result
Independent Claude cloud plan review completed at
https://claude.ai/code/session_01AZ89UDNykQTTN6DWKokHaQ. Its required prerequisites
were verified: exact51package pins, complete immutable Hub snapshots, same-
filesystem installs, bound xet cache, inspected right padding / tail-mask pooling,
byte-exact pilot subset, explicit bf16w-fp32 and resource caps. LFS blob bytes
passed SHA256 verification. Source worker matches saved7649bc9 exactly.

The batch4/check4 pilot completed successfully: all16comparisons passed,
max difference0.00004637384414674095 against tolerance0.0001. Receipt audit found
exactly16valid scores, zero invalid; each pilot chunk has4requests/4checks.
Pilot wall time1159s, reported cgroup peak4.3G and swap peak128M; observed memory
throttle/max/OOM events zero. Full coverage remains incomplete, so no efficacy
metrics were computed. The original104legacy scores passed retained checksums.

Owned hn-strands-full-20261011.service is running (launched12:45UTC). It processes
the complete unchanged4132-job v3 plan, reusing16validated pilot scores. Args:
batch4/check1/threads2/tolerance1e-4/bf16w-fp32, no length bucketing. Nice15,
CPUQuota200%, MemoryHigh4500M, MemoryMax5G, MemorySwapMax128M, RuntimeMaxSec48h,
no restart. Its sequential pipeline runs audit and then report only after scorer
success; audit must prove2066real+2066empty complete before quality metrics.
Live hn_rewrite.service PID531386/start03:26:55UTC unchanged.

## Blocker
CPU inference is likely about one to two days; full numerical coverage and
held-out quality remain pending. Parity is sampled per full-run chunk. The plain
reference uses the same worker numerical path, not certification of stock model
CPU/GPU numerics. This historical held-out result is exploratory, not untouched
confirmation or permission to deploy.

## Next step
Monitor hn-strands-full-20261011.service and
/tmp/strands-eval/scoring-20261011-full.log; verify prefix reuse and initial chunk.
Evidence: /tmp/strands-eval/scoring-20261011-evidence/. On success, inspect full
cache_audit_v3.json/report_strands_v3.json in /tmp/decision1-eval/work_heldout.
On guard failure, investigate before any restart. At a clean runtime-limit stop,
resume the same reviewed job only after revalidation; valid scores are skipped.
Report completion only after complete audit and actual report readback.
