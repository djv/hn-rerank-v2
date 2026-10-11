# Concrete bounded CPU feasibility check — proposed, not executed

Scope selected by the user: keep the existing VPS. Execution of an alternate model
remains a separate decision. No paid services, production changes or published scores.

Candidate: onnx-community/strands-decider-2B-hobson-v19-ONNX, q8, pinned revision
31a83e0244b26d939ae33f87189d6c8204003e53. This replaces the current adapter for
this diagnostic only. The fused graph has no external prefix cache; every test
must measure full fitted prompts and include explicit standalone-check overhead.

Verified public artifact metadata:
- onnx/model_quantized.onnx: 515126 bytes; SHA256
  d156ecf442aaceccd891864b709a7acbea558e382fb1253a7eb8e53c48f46b23.
- onnx/model_quantized.onnx_data: 1800931328 bytes; SHA256
  0b92e9b7ff19b21c478ae056af397addd0f674180637b4850967fbe8849b21cf.
- tokenizer.json: 19989592 bytes; SHA256
  a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86.
- Fetch config/tokenizer_config and small public fixtures at that revision; record
  their SHA256. Total model/tokenizer payload about 1.822 GB, excluding environment.
- Isolated uv environment, Python3.12 CPU onnxruntime1.31.0; official wheel
  onnxruntime-1.31.0-cp312-cp312-manylinux_2_28_x86_64.whl, SHA256
  5e129d6c56abd53e659cb70f00a108d6824086470ff99c2e47a82e5786563db3.
  Pin actual dependency resolution. Package presence does not certify graph support.

Bounded procedure if authorized:
1. Refresh host load, MemAvailable, disk and service identity. Require at least
   4.5 GiB MemAvailable before inference and artifacts/environment plus 1.5 GiB
   remaining disk before download. No deletions or stopping other workloads.
2. Download only the listed q8 artifacts/config/tokenizer/fixtures on the VPS,
   hash-check them, and create a separate temporary environment. Do not modify
   existing scorer environment, frozen inputs, cache identity, database or service.
3. Own transient cgroup: CPUQuota300%, Nice15, MemoryMax3.5GiB, SwapMax128MiB,
   no restart; total task timeout20min. Explicit ORT intra3/inter1/sequential,
   spinning disabled. Terminate only the owned task on failed prerequisites,
   timeout, parity failure, unsupported operators or sustained resource pressure.
   Host MemAvailable below1GiB stops the task. The smaller cap may reject the graph;
   report that outcome without relaxing it automatically.
4. Use candidate tokenizer/config/noul calibration; reproduce public fixtures
   without private data and establish exact graph/provider compatibility first.
   Right padding only. Require finite normalized output, no public fixture top
   answer changes, and maximum probability difference0.03 against published
   upstream fixtures. This explicit diagnostic threshold is informed by the
   published0.026 WebGPU drift, not CPU certification or faithful parity. Stop on
   missing usable references or a failed fixture. Record all drift values; do not
   demand current-model equivalence or transfer this threshold to existing caches.
5. Select four stories from the frozen plan using lengths alone (short, median,
   long and maximum); retain each real/empty pair, eight requests total. No labels,
   mutable database, confirmation cohort or phone data. Use existing prompt fitting
   while preserving candidate-specific tokens/config/positions. Log technical
   lengths/times/probability differences only, not private text.
6. Run batched real and empty conditions separately; independently evaluate each
   of eight requests as batch1. Compare within-runtime batch/standalone probabilities
   at1e-4. Include all standalone work in total inference time/8. Record loading,
   any warmup, both condition timings and host/cgroup counters separately.
7. If the bounded sample misses5s including checks, stop this route's investigation
   and report that measured sample result, not theoretical impossibility. If it
   passes, report preliminary feasibility only. Representative weighted full-plan
   throughput, repeat-based variance, later production check policy and HN efficacy
   still require separate work/authorization. No automatic expanded pilot or run.

Live preflight snapshot 2026-10-11 16:31UTC: VPS MemAvailable5104MiB, root disk
23754948608bytes available, load0.22/0.33/0.27. Full scorer inactive/MainPID0.
These values are volatile and must be rechecked before execution.

Sources:
https://huggingface.co/api/models/onnx-community/strands-decider-2B-hobson-v19-ONNX/revision/main?blobs=true
https://pypi.org/pypi/onnxruntime/json
https://huggingface.co/onnx-community/strands-decider-2B-hobson-v19-ONNX

No execution, download, installation or model change performed while preparing
this proposal. Approval authorizes only the bounded diagnostic described above.
