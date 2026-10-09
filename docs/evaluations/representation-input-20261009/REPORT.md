# Primary embedding input comparison — completed 2026-10-09

The comment-free primary embedding failed the frozen improvement gates. Stop
this representation branch; spend no Muse canonicalization annotations on it.
The broader goal remains systematic Muse advantages that improve ML upvote discovery.

| Endpoint | Native baseline | Comment-free primary |
|---|---:|---:|
| Pooled within-block UP-versus-rest AUC | .779843 | .778184 |
| Upvotes among four blocks' top 12 (48 total) | 29 | 29 |
| Downvotes among those 48 | 1 | 1 |

AUC delta −.001659; paired bootstrap 95% interval [−.007995, +.004818]
(2,000 resamples, conditional on fitted models). Two of four blocks improved.
Required delta ≥ .01, improvement in ≥3/4 blocks, and positive interval lower
bound all failed. Both top-12 guardrails passed. `metrics_report.json` records
all gates and per-block outcomes; runner exit 1 denotes failed gates.

## Design and execution

- Historical cohort: 3,848 rows; expanding-window tests on 3,079 rows in blocks
  2–5. Only primary embeddings and derived similarities changed; lexical,
  metadata and side inputs stayed fixed. The 51 stored-text/composer mismatches
  retained original vectors; 509 cleaned-body-empty rows used title-only input.
  Encoded 3,237 changed inputs.
- Native baseline exactly matched the amended frozen one-thread reference:
  maximum probability and score differences 0.0. Original failed reference and
  successful amended validation are preserved. Thread limits were set before
  Python; amendment preceded challenger outcomes (PLAN.md).
- Deterministic parity20 passed, minimum cosine .999999821. Encoding took
  891.6 seconds at batch 1. Separate task DB copies and fresh processes for
  baseline/challenger fits took 24.6/28.8 seconds. Warm-start persistence off.
- Completed 15:52 UTC; task-owned encode and runner processes verified exited
  at 15:56 UTC. No live DB/config/service or shared-cache changes.
- `postrun_integrity.json`: all 3,947 voted story rows, feedback rows and side
  vectors match across source and both copies; frozen config/encoder identities
  match. Runtime suite: 1,213 passed, one skipped. Artifact Ruff/format/ty and
  13 self-check groups passed.

## Replay identity

`harness_frozen.py` is the exact experiment source (SHA-256 prefix
`988f127a212e9e2d`). Current `harness.py` has post-run hardening to reject baseline
mismatch and correct an unsupported batch-speed comment; those edits were not
used for these results. Sample `d8e996f30a64cdf4`, config `99a2ad40f7f9973f`,
model.onnx `9ba32b8568b5bf0a`, tokenizer.json `da0e79933b9ed517`.
Fixed scoring clock: 1791401987.9873054.

Local evidence: metrics, baseline/challenger, parity, validation and integrity
reports. Private prediction rows, vectors and snapshot remain on VPS under
`/tmp/opencode/representation-input-20261009/`. No new LLM ranking annotations;
Muse coding/orchestration and one Claude review were used.

## Limits and next direction

Historical outcomes are exploratory, not prospective validation. This result
concerns primary embeddings conditional on unchanged other inputs; it does not
establish that comments are irrelevant in every model. Bootstrap intervals do
not cover refitting uncertainty or repeated model exploration.

The user selected missed-upvote analysis using existing predictions and cached
Muse annotations. Identify patterns with denominators and downvote controls
before choosing another correction design. A useful historical pattern would
justify a frozen prospective test, not production deployment.
