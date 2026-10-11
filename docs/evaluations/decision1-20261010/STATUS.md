# HN paired scoring — explicitly paused, 2026-10-11

## Objective
The user selected "Pause speed work". Retain results and evidence; do not continue
CPU investigation, conversion, pilot retries or full scoring until requested.

## Verified result
Full scorer inactive/MainPID0; bounded ONNX unit stopped/MainPID0 (resource-stop
exit143, not OOM). Latest check confirms both have no running main process.
The last validated original cache audit was56valid/0invalid/4076missing of4132;
diagnostics published zero scores. No complete HN quality result.

Same-model favorable real batch16/check1 measured8.991s/score with packedFP16,
excluding loading/prefix; no representative paired under5s result exists.
The subsequently approved eight-request older hobson-v19 q8 ONNX check loaded on
CPU and passed public fixtures (maxdiff0.0258148 at diagnostic tolerance0.03).
It stopped at the approved3.5GiB cap before its first real batch completed;
private latency, batch/standalone parity and empty timing remain unmeasured.
Result review found no serious safety/scope defect; wheel hash enforcement and
memory/phase telemetry are incomplete. No fixes or retry applied. Production
source and original scoring environment are unchanged by this task. At save,
live HN PID759874/start16:51:50UTC supersedes the earlier PID531386 checkpoint;
this agent performed no restart or deployment. Both scoring jobs remain stopped.

## Blocker
No verified route meets below5s including parity overhead. Charged memory was not
split into anonymous/file cache; the bounded stop does not prove VPS impossibility.

## Next step
Wait for the user's explicit resumption. No automatic retry, cap increase, new
model/conversion work, paid route, deployment or full scoring. Evidence:
CPU-SPEED-20261011.md and onnx-check/VERIFIED.md and REVIEW.md. Raw runtime
artifacts/models remain at VPS /tmp/strands-eval/onnx-cpu-check-20261011/;
private token payloads remain there at0600 and were not copied into this repository.
The older unapproved16-row proposal is superseded by the executed smaller check.
