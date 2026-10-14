# Gemini versus Muse worker benchmark — 2026-10-14

Recommendation: retain Muse as the default economical worker, with Gemini
available for selected repairs and a second attempt on difficult tasks.
Gemini produced cleaner artifacts in this sample, but no broad superiority
was established. No persistent model default or production code was changed.

## Results

One attempt per model per task; worker deadline ten minutes. Times include
the worker's own tools and checks, excluding independent grading.

| Task | Gemini 4 Argon, high | Muse Spark 1.3, xhigh |
| --- | --- | --- |
| Repair three seeded queue bugs | 2:14; all gates pass | 3:11; behavior passes, new test needs formatting |
| Add generated property coverage | Timed out at 10:00; partial artifact passes gates | Timed out at 10:00; partial artifact leaves a source mutation and fails suite/format |
| Simplify queue and investigate four client contracts | 8:41; all gates pass | 3:46; all gates pass |

Both completed two tasks; neither completed test writing by the deadline.
Gemini's completed tasks both passed every code gate. Muse's repair had a
cosmetic formatting failure; its simplification passed every gate.
All four completed code artifacts matched the baseline oracle in 1,040
structured cases. This counts generated inputs, not independent real-world tasks.

Full TUI suites: Gemini repair 296 passed / 1 skipped; Muse repair 294 / 1;
Gemini partial tests 292 / 1; Muse partial tests 288 passed / 5 failed / 1
skipped; Gemini simplification 292 / 1; Muse simplification 294 / 1.
Ruff and ty passed for all six; formatting failed on Muse repair and partial
tests only. Gemini simplification changed 19 lines (8 added, 11 removed);
Muse changed 31 (18 added, 13 removed). Both directly construct the lanes.

### Timed-out test artifacts

The saved Muse artifact removed Explore badge grouping, causing 652 oracle
mismatches and five test failures. Its last tool call was a mutation probe:
it patched source, ran multiple seeded tests, and intended to restore the
backup afterwards. Deadline termination interrupted that sequence. This is
an unsafe unfinished artifact, not evidence that Muse chose the mutation
as its final implementation. The tests-only brief prohibited production
source changes, including these temporary changes. Muse also used direct
`python3` for patching despite the uv instruction.

The agent independently copied Muse's new tests into a temporary directory
with correct baseline source: all four tests passed in 2.46 seconds. Its
partial tests therefore work on correct code, but the returned checkout
cannot be accepted without inspection and cleanup. Neither timed-out task
produced the requested NOTE.md. No production checkout was involved.

The new Gemini file detected four of six seeded faults within a 45-second
per-mutant deadline; Muse detected two. The other runs timed out, so those
faults are unresolved, not proven undetected. Faults covered exclusions,
restorations, zero version, Popular cap, badge priority, and frozen rounds.
The original suite was not scored against these same mutations: these
counts do not establish additional coverage over existing tests. Both
files contain expected-output logic resembling the queue implementation,
so passing does not by itself establish an independent behavioral oracle.

### Research assessment

Both correctly answered the four client-source questions: restorations
filter before served-rank enumeration; returned rows lose local-restoration
status before the readiness branch; old rounds retain their origin/rank/
version; Queue telemetry uses saved origin metadata with a separate Queue
position. Both marked absent backend evidence as uncertain.

Gemini's report contains one incorrect extra locator: API.interaction is
at baseline api.py:430–450, not 390–409. The payload claim is correct at the
verified location. The original reports are preserved without correction
in GEMINI-EVIDENCE.md and MUSE-EVIDENCE.md. This was local code research,
not a benchmark of internet research or factual recall.

## Usage

Pi's completed assistant-message usage events reported catalog-equivalent
costs of $2.9435 Gemini and $0.0580 Muse across the three tasks, about 51×
different. By task: Gemini $0.2315 / $1.2516 / $1.4604; Muse $0.0109 /
$0.0255 / $0.0216. These are adapter estimates, not measured Go quota
debits or additional charges. Timed-out usage can omit unfinished output.
The snapshot's promotional quotas and catalog prices may change.

Muse was 2.3× faster on simplification/research; Gemini was faster on repair.
This small sample cannot establish general latency, reliability, coding
quality, or value across tasks. Cost estimates alone do not determine
which subscription allowance is better to spend.

## Protocol and reproducibility

- Source: clients/tui only, commit
  `11ec7b820c4f3c2a6ffefb4750e493468f13798f`.
- Host: VPS; laptop was on battery. Six isolated synthetic directories,
  Python 3.12, independent virtual environments, candidate-specific
  PYTHONPATH, BLAS/OMP/MKL thread counts one. No production data or DB.
- Same pi 1.1.0 harness, same corresponding task brief and system guard,
  fresh context per task, no ancestor instructions, skills, MCP, session
  persistence, or Git remote. Models ran concurrently, sequential tasks
  within each model. This is a comparison of their supported configured
  effort settings, not an identical reasoning budget.
- Models: `opencode-go/gemini-4-argon-contributor` high and
  `opencode-go/muse-spark-1.3-contributor` xhigh.
- Gemini used a temporary native Google API provider adapter, retaining
  Go authentication and the session header. Live response and tool use
  succeeded. Pi 1.1.0 and the inspected 1.2.0 package lacked that adapter;
  upstream main had support. Availability was checked against the
  [Go documentation](https://opencode.ai/v2/docs/console/go/) and
  [upstream pi provider](https://github.com/earendil-works/pi/blob/main/packages/ai/src/providers/opencode-go.ts).
- Repair brief: restore zero version, five unique Popular cards, and
  future-date window behavior; minimal queue.py edits plus new regressions.
- Tests brief: at least two generated Hypothesis properties covering
  ownership, order/cap, restored/excluded rank semantics, zero version,
  frozen rounds, and no input mutation; new tests and notes only.
- Simplify/research brief: construct lanes directly, preserve every
  behavior, retain explicit types, and answer the four questions above
  with file:line evidence and uncertainty. No telemetry/ranking edits.
- Worker checks: focused tests. Independent gates: full TUI suite,
  Ruff, ty, changed-file formatting. Full suites used eight workers except
  the two repair grades, which used two. Oracle seed 710149: 1,000 generated
  feed cases, checking queue entries/build_round and input mutation,
  plus 40 timestamp/window cases. Grader and worker implementation differ;
  no oracle results were supplied to workers.
- Discarded pilot: a shared editable environment resolved imports to the
  baseline and PATH lacked uv. It was stopped, environments corrected,
  and all six scored cases restarted. No pilot result is included.

Detailed grades: grades.json. Raw runs and reproducible task/grading
artifacts remain outside Git on the VPS at
`/home/dev/agent-benchmarks/gemini-worker-20261014-02/`; coordinator scripts
are `/tmp/run_gemini_worker_benchmark.py`,
`/tmp/grade_gemini_worker_benchmark.py`, and
`/tmp/check_worker_partial.py`. Temporary scripts are not durable installs.
No benchmark patch was applied to the project. Worker deadlines require
isolated copies and a final diff check; temporary mutation probes should
operate on separate copies rather than editing the candidate source.
All own worker, grader, and pytest processes were verified exited. Final
VPS load was 0.83 with 4.6 GiB available RAM; other jobs were left untouched.

## Decision

Keep Muse's default for scoped routine work; inspect diffs and enforce
checks before accepting its output. Use Gemini as an available alternative
where repair correctness or a smaller patch justifies more quota use.
Neither should be an unattended default for open-ended test improvement
on this evidence. Pi is the user's preferred interactive TUI; OpenCode
CLI remains acceptable for headless research/benchmarks. No global
default change was authorized or made by this test.
