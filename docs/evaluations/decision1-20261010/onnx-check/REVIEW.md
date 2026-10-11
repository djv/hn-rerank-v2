# Independent review: bounded ONNX CPU diagnostic (hobson-v19 q8)

Reviewer scope: read-only review of the supplied `run.py`, `prepare.py`, `infer.py`,
`task.log`, `result.json`, `resources.json`, `provenance.json`, `packages.txt`,
`FINAL-STATE.txt` and the approved `SMALL-CPU-CHECK.md` (all under
`files/tmp/hn-onnx-result-rv-20261011/`). Nothing was run against the model, no
packages were installed. The one network action was fetching the public upstream
`conversion/parity_fused.py` at the pinned revision to check that the fixture
procedure matches it.

Paths below are relative to `files/tmp/hn-onnx-result-rv-20261011/`.

## Verdict

The diagnostic stayed **inside the approved scope and stopped the way the approval
said it should**. The transient unit's limits match the approval, the owned process
tree is gone, nothing was published, and the task stopped on the watchdog's
resource rule without OOM kills. Public fixtures passed against the upstream
references with right-padded batches (worst |Δp| 0.0258 against the 0.03 threshold,
no argmax flips). This shows the graph and provider are compatible with the published
fixtures on this CPU. It does not certify numerical parity, and it says nothing about
HN efficacy.

The run **produced no private-request result**. No real or empty row was timed, and no
batch/standalone comparison ran. So the run supports neither "fast enough" nor "too
slow / cannot run". It shows one thing: under the approved 3.5 GiB cgroup, with this
implementation (download charged to the same cgroup, all four real rows of about
2.2k–2.9k tokens in one batch), the charged memory reached `memory.max` and the
approved sustained-pressure rule fired. The evidence cannot separate anonymous
runtime memory from reclaimable file cache, so it does not show that the model
cannot fit.

I found no serious compliance or safety defect. There are two moderate provenance and
observability gaps (F1, F2) and a few hypothetical gaps (F3–F5). These limit how the
result can be read. They do not invalidate the stop.

## Q1 – Scope, limits, private data, publication, stop semantics

**Reproduced from supplied files**

- Unit limits match the approval (`FINAL-STATE.txt:1-16`). `Restart=no`,
  `RuntimeMaxUSec=20min`, `CPUQuotaPerSecUSec=3s` (300%), `MemoryMax=3758096384`
  (exactly 3.5 GiB), `MemorySwapMax=134217728` (128 MiB), `Nice=15`,
  `OOMPolicy=stop`, `KillMode=control-group`.
- ORT session settings match the approval (`infer.py:36-42`): intra 3, inter 1,
  sequential, spinning disabled on both pools, CPU EP only, provider list asserted
  (`infer.py:46-47`).
- Preflights. MemAvailable ≥ 4.5 GiB is checked at start, before preparation and
  before inference (`run.py:71,106,109`). The disk check (`run.py:73`) requires
  artifacts + 2.5 GiB, which is stricter than the approved "artifacts/environment +
  1.5 GiB". The scripts do not check host load or HN service identity; the prompt
  asserts the operator checked those outside the scripts, and I could not verify that
  from the files.
- Download scope: only the seven listed paths, at the pinned revision (`run.py:17-29`).
  Each is size-checked. The three large files are also checked against the
  approved SHA-256 values (`run.py:95`). The small files have their hashes recorded
  (`provenance.json:16-31`), as the approval asked. Each file is written to `.partial`
  and renamed only after it verifies.
- Script integrity: I recomputed the SHA-256 of the supplied `run.py`, `prepare.py`
  and `infer.py`. All three match `provenance.json:34-36`, so the reviewed code is
  the code that ran.
- Isolation. The new venv lives under `CHECK_ROOT` (`run.py:101`, `task.log:9`) and
  borrows only the base interpreter of `/tmp/strands-eval/.venv`. `prepare.py` runs on
  the old interpreter with `uv run --no-project` (`run.py:108`), which imports the
  pinned `strands_decider` package and does not install into it. Model files go to
  `CHECK_ROOT/model`, not to an HF cache. There is no database, cache-identity or
  service write path in any script.
- Zero publication: `published_scores: 0` (`result.json:2`). `infer.py` writes only
  `result.json`.
- Private data. `umask 077` (`run.py:70`), and `prepared-private.json` and
  `result.json` are written with mode 0600 (`prepare.py:69-71`, `infer.py:22`). Logs
  carry only token lengths and length classes (`task.log:26`). There is no
  text, there are no IDs and no labels.
- Stop path. The watchdog fired its "sustained hard-limit pressure" rule
  (`run.py:55,62-66`, `task.log:35`). It sends SIGTERM to `run.py`, which exits
  with 143 (`FINAL-STATE.txt:9`). systemd's `KillMode=control-group` then removes the
  `uv`/`infer.py` children: `MainPID=0` and an empty `ControlGroup=` (`FINAL-STATE.txt:3,10`).
  `oom`/`oom_kill` = 0 (`resources.json:7-8`). The last host MemAvailable sample was
  2.84 GB, above the 1 GiB stop line (`resources.json:2`). The run started at 16:36:55
  and exited at 16:39:31, well inside the 20-minute limit.

**Findings**

- **F1 (moderate, reproduced): the approved onnxruntime wheel hash was not
  enforced or recorded.** `SMALL-CPU-CHECK.md:20-23` names the exact wheel and its
  SHA-256 and says "Pin actual dependency resolution". `run.py:102-103` installs
  `onnxruntime==1.31.0 numpy==2.3.3` by version only, with no `--require-hashes` or
  hash pin. `packages.txt` records versions only, and the transitive packages are
  unpinned. The version matches, but the artifact identity the approval named cannot
  be confirmed from the evidence. This does not change the outcome. It should be fixed
  before any retry is cited as provenance-complete.
- **F2 (moderate, reproduced): the stop record is thin.** Log lines have no
  timestamps (`task.log`), so it is unknown how long `real_batch4` ran before STOP,
  or whether `memory.events.max` was already rising during the fixtures. Only the last
  watchdog sample survives (`run.py:57-61` overwrites `resources.json`). SIGTERM uses
  the default handler, so `failure.txt` is not written and `result.json` does not
  carry the stop reason (`result.json:3` only says `complete: false`). The stop is
  still correct; the result is just hard to interpret afterwards.
- Residual retention (observation, not a defect): `prepared-private.json` (private
  token IDs, which can be decoded back to text) and the 1.8 GB model stay under
  `/tmp/strands-eval/onnx-cpu-check-20261011/`. They are mode 0600. The approval
  did not specify cleanup, so the user should decide whether to keep them.
- Hypothetical: the watchdog thread has no exception guard (`run.py:43-66`). If a
  cgroup file read failed, monitoring would stop silently, including the host
  MemAvailable guard. The kernel would still enforce the cgroup limits. This did not
  happen here; samples were written through to the STOP.

## Q2 – Validity of the fixture and comparison logic

- **Fixtures.** `infer.py:81-95` matches the upstream `parity_fused.py`: one
  right-padded batch per record, `answer_pos = len-1`, `option_pos` zero-padded, and
  softmax over `logits[:, :k]` divided by `temperature_by_kind[kind]`. The
  references are therefore post-temperature probabilities, and comparing them is
  logically valid. There are two harmless differences. `infer.py` pads with
  `tokenizer_pad_id` where upstream pads with 0. It also raises a KeyError instead of
  falling back to `cfg['temperature']` for an unknown kind, which fails loudly
  rather than silently. Before inference, `prepare.py:41-49` checks that the
  candidate tokenizer and the existing renderer/`_fit` reproduce the published
  `state_ids + question_ids`, `opt_idx` and `kind` exactly. That is a strong check
  for the fixture texts.
- **Result.** The worst value, 0.02581, sits inside 0.03 but close to it, and close
  to the published 0.026 WebGPU drift. That is consistent with q8 quantization drift,
  not with an obvious CPU-EP defect. As the approval says, it is a diagnostic gate,
  not CPU parity certification. It does not transfer to existing caches and says
  nothing about HN efficacy.
- **Right padding and positions.** The graph has no `position_ids` input, so the graph
  derives positions itself. With right padding, both arange- and mask-cumsum-derived
  positions are correct for real tokens. Under causal attention, trailing pad tokens
  cannot reach earlier positions, and logits are read at `answer_pos`, not at the last
  column. Right padding is therefore sound by construction. The fixture batches
  already exercised padded rows of unequal length and passed.
- **Batch vs standalone.** The design is valid: same session, 1e-4 on probabilities,
  standalone time counted in the per-score total (`infer.py:100-115`,
  `SMALL-CPU-CHECK.md:51-53`). Because of the causal argument above, this check mostly
  tests kernel determinism across batch shapes, not masking. **It never ran.**
- **Temperature and validation.** Temperature scaling is per kind. The code checks
  for finite logits, the logits shape, and a probability sum within 1e-10
  (`infer.py:69-77`).

**Untested because no private row completed:** latency at real lengths (2248–2907
real tokens, 198–723 empty, `task.log:26`) on this host; batch/standalone agreement on
any private row; empty-condition timing; whether the 1e-4 gate holds at about 3k
tokens; peak memory for a single 2.9k-token row compared with the batch of four; and
the 5 s/score criterion in either direction.

Hypothetical gaps in `prepare.py` that I could not check without the plan:

- **F3.** `paired` is keyed by `(fold, story_id, variant)` (`prepare.py:52`). If the
  plan holds more than one job per key, the "empty" partner may come from a different
  job than the selected real row.
- **F4.** `prepare(...)[0]` (`prepare.py:55,64`) keeps only the first question. If a
  job carried more than one question, per-request work would be undercounted.
- **F5.** `task.log:25` shows transformers' "incorrect regex pattern" tokenizer
  warning. The fixtures tokenized exactly as upstream, so the warning does not affect
  the reference comparison. Whether the same tokenization matches training on private
  text with other character classes is unverified.

## Q3 – Does the 3.5 GiB stop show that ONNX is too slow or that the VPS cannot run it?

**No, on either count.**

- **Runtime speed.** Nothing at private lengths was timed. The six fixture records took
  7.6–16.3 s each (`task.log:28-33`). The log does not record their case counts or
  token lengths, and the times may already include reclaim or refault stalls, so they
  cannot be turned into a per-score figure. No throughput claim, for or against, is
  supported.
- **Memory.** `memory.current` and `memory.peak` were 3.758 GB, which is the limit
  (`resources.json:11-12`). There were 652 `max` events, swap was 89 MB of 128 MB, and
  there were no OOMs. `memory.max` events count every time the kernel had to reclaim
  at the limit, including routine eviction of clean page cache. The STOP rule (more
  than 4 consecutive 2 s samples with a rising `max` count) therefore fires during
  ordinary cache reclaim as well as real anonymous exhaustion. The charged total
  includes at least three things:
  1. File cache from the 1.8 GB `onnx_data` download, which ran in the same cgroup
     (`run.py:83-99`).
  2. Weight pages ORT either memory-maps or copies or prepacks into anonymous memory.
     Which one is unknown, and a copy would double-count against cached file pages.
  3. Activations for a 4 × 2907 batch.

  No `memory.stat` (anon, file, file_mapped) and no PSI were captured, so these cannot
  be separated. Non-zero swap shows that some anonymous pages were reclaimed. That
  points toward real anonymous pressure but does not prove it, because the kernel also
  swaps while file cache remains.
- **Host headroom.** MemAvailable was about 5 GiB before launch and 2.84 GB at the last
  sample (`resources.json:2`). The host was not near exhaustion. The binding constraint
  was the approved cgroup cap, which must not be relaxed automatically
  (`SMALL-CPU-CHECK.md:36-37`).
- **Accurate statement of the result:** *"Under the approved 3.5 GiB / 128 MiB-swap
  cgroup, with the download in the same cgroup and all four real rows in one batch,
  charged memory hit the cap and the sustained-pressure rule stopped the run before
  the first private batch finished. Speed was not measured."* That does not mean the
  run was under 5 s, or that the model cannot run on this VPS, or anything about a
  cohort average.

## Q4 – What to fix before any separately authorized retry

No retry, cap change, batching change or further model run is authorized. The changes
below are limited to provenance and observability. None of them alters the approved
procedure's limits or batching.

1. Enforce and record the approved wheel SHA-256, and pin resolution with hashes
   (F1).
2. Add a wall-clock timestamp to every log and phase line. Append watchdog samples
   instead of overwriting them. Each sample should include `memory.stat`
   (anon/file/file_mapped/swap) and `memory.pressure`. Write the stop reason into
   `result.json` (F2). Without this, a second stop would be just as uninterpretable
   as this one.
3. Before any retry, decide whether file cache charged by the download should count
   toward the pressure gate. The download could run outside the measured unit, or the
   gate could be defined on anonymous memory or PSI instead of the `memory.events.max`
   count. Either is a change to the stop definition, so it needs explicit approval.
   I am not recommending one over the other.
4. Confirm, from the plan's structure alone, that `(fold, story_id, variant)` is
   unique and that each job holds a single question (F3, F4).

## Remaining unknowns

- Private-length latency and the 5 s gate.
- Batch/standalone agreement.
- How peak memory splits between anonymous and file memory, and whether a single
  2.9k-token row fits under 3.5 GiB.
- How the AVX2-only shared vCPUs affect q8 MatMul throughput.
- Fixture token lengths and case counts, needed to normalize the fixture timings.
- Whether the transformers tokenizer warning matters for private text.
- HN efficacy of the changed model. This was never in scope.
