# Representation-input two-arm comparison: frozen plan (2026-10-09)

Status: completed, improvement gates failed; current result in REPORT.md.
Below is the frozen protocol and dated execution history, including the
original preparation scope subsequently extended by the user's keep-working
instruction. Running/pending entries below are historical, not current status.

## Amendment 2026-10-09: parity passed, challenger encode running

- Parity20 gate PASSED on VPS (20/20 full-text rows cosine >= .9999 vs
  stored vectors, min 0.9999998212; 27.5 s for 20 rows, 4455 s projected
  for 3237 changed rows). Deterministic ID-strided selection, 4 per frozen
  block, matched rows only; explicit one-thread shell vars, ONNX intra_op=1.
- Challenger encode launched as a persistent task-owned resumable VPS job:
  PID 3989858, nohup + nice -n 10, log
  `/tmp/opencode/representation-input-20261009/encode.log`, out dir
  `.../challenger/` (npz + done-checkpoint + report). One job only;
  never stop other jobs; headroom guard (load > 4 or avail RAM < 2 GB)
  refuses before/during encoding.
- Required shared-box settings (not a bug): ENCODE_BATCH=1,
  ENCODE_CHUNK=16, single-thread encoder session; chosen for memory
  safety, no wider-batch claim made.
- Report stage adds the pre-result guardrail `guardrail_up_not_down`
  (pooled top-12 up fraction must not decrease) alongside the frozen
  down-not-up guardrail; both must pass with the primary/secondary gates.
- Post-parity harness delta is resource comments/helper/report guardrail
  only; encode path and parity result unchanged.
- Pending: encode completion, fresh-process per-arm compares on
  task-owned snapshot copies, `report` with frozen gates + conditional
  2000-rep bootstrap. No production changes, no LLM ranking annotations.

## Baseline execution amendment (2026-10-09, before challenger results)

The original saved OOF run imported NumPy before setting library thread limits.
The new harness sets limits before importing NumPy. The original script rerun
with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 reproduces the new
harness **exactly** (0.0 probability delta), while both differ from the older
saved run (max .4344247878, mostly tiny differences and one large block-4 row).
The earlier saved run remains preserved; this is not a challenger improvement.
Use the original script's frozen one-thread reference at
/tmp/opencode/representation-input-20261009/diag_threads1/oof_scores.json
(sha16 e40fae0c91941304). Against this reference a fresh harness process passed
all 3,079 rows at 0.0 probability and production-score difference, with no vector
drift and disk warm store disabled (validation_report_threads1.json).
Always set the three thread variables explicitly to 1 in the shell **before**
Python starts. Do not rely on setdefault if the host already exports a value.
This changes the execution reference before any challenger encoding/evaluation;
it changes neither the user-selected endpoint nor the go gates.
The exact numerical branch behind the exceptional row was not diagnosed.
Challenger and baseline must use the same environment and fresh processes.

Implementation limit: compare currently refuses to run. Encoding/parity and
two-arm evaluation plumbing are the next stage, not completed functionality.

Scope correction (overrides older report text): the next comparison
changes **primary embedding inputs only**. Stored story rows, lexical /
TF-IDF features, metadata (including comment-bearing `text_length`),
and side vectors stay identical between arms. Only primary vectors and
the vector-derived similarity/cluster features follow the arm.
`DATA-AUDIT.md` primary review and `NEXT-EXPERIMENT.md` user-selected
endpoint section override older sections on conflicts.

## User-selected endpoint: find more upvotes

- Primary: within-block UP-vs-rest AUC on each arm's **production
  ranking score**, pooled over test blocks by UP/non-UP pair count.
- Secondary: known-upvote fraction at top 12.
- Pre-result user-goal guardrail (added before any challenger result): known
  upvote fraction at top 12 must not decrease in the pooled descriptive result.
  An AUC improvement alone must not justify fewer upvotes at the top.
- Guardrail (pooled descriptive): known-downvote fraction at top 12 must
  not increase vs baseline.
- Ordinal AUC stays descriptive and cannot override the primary.
- Exploratory go gates (challenger minus baseline): primary pooled delta
  >= +.01; positive in >= 3/4 test blocks; paired story-bootstrap 95%
  lower bound > 0 (conditional on fits; repeated development use
  disclosed). All gates plus guardrail must pass.
- Splits: existing chronological full-vote blocks 1..5 (old cohort,
  3848 rows). Expanding-window fits on earlier blocks, tests on blocks
  2..5. Dev 99 reported separately, never a gate.

## Arms

- `T_full`: `story_embedding_text` (stored `text_content` verbatim;
  byte-identical to production, staleness included).
- `T_nocom`: `compose_story_text(title, self_text, "", article_body)`
  (same cleaner, same 6000/4000 caps, title-first order) **only on rows
  where stored text equals the full composer**. On mismatch rows the
  original primary vector is reused unchanged; those rows are reported
  as unmodified, never reconstructed.
- Encoder/settings frozen: production ONNX dir + tokenizer,
  `max_tokens=4096`, mean-pool, L2 norm, one CPU, capped threads.
- Resource constraint before parity/encoding: one input per ONNX batch. The VPS
  also runs the live service; do not allocate four long-context inputs together.
- Challenger vector domain: distinct model version
  `<prod>|nocomments` served by a precomputed embedder (no inference at
  fit time); source/input/vector hashes recorded per story in the
  manifest. Mismatch rows map to the stored vector and are counted.

## Isolation (both arms, including baseline validation)

- Existing frozen VPS snapshot / task cache only; never the live DB, no
  `upsert_story`, exact source `Story` rows.
- Real `_score_and_rank` with explicit `training_feedback`; story order
  preserved (chronological block lists as frozen).
- Isolated fresh process per arm; task-owned snapshot copies per arm;
  `warm_store.enabled()` asserted false; no shared warm caches (model /
  linear-blend in-memory caches start empty per process).
- Fixed scoring clock: `sample.json` `eval_now` patched over
  `time.time` during scoring in both arms, so production scores
  (tier-1 gravity) are comparable. Historical probability reproduction
  does not reproduce served scores; score drift vs old wall-clock
  `oof_scores.json` is expected and reported, never gated.
- Baseline gate first: reproduce old `oof_scores.json` probabilities
  within 1e-6 (max abs diff over prob_up/neutral/down on all test rows)
  before any encoding. On failure: stop, write report, no fits.
- Later encoding gate (deferred, not this step): ~20 full-text encoder
  checks against stored vectors (cosine >= .9999) before body encoding.

## Artifact-local files (this dir only)

- `harness.py`: `manifest` (read-only input hashes + changed counts),
  `validate` (baseline probability reproduction), `compare` (refuses
  until baseline gate passes and challenger vectors exist),
  `self-check` (pure behavioral checks, no DB/model).
- `STATUS.md`: step status + factual VPS result.
- `validation_report.json` + stdout/log: actual run artifacts, no
  private text (ids + floats only).

## Baseline validation command (VPS)

```bash
ssh hetzner "uptime; free -m | head -n 3"
scp docs/evaluations/representation-input-20261009/harness.py \
  hetzner:/home/dev/hn-rewrite/main/docs/evaluations/representation-input-20261009/harness.py
ssh hetzner "nice -n 10 /home/dev/.local/bin/uv run --project /home/dev/hn-rewrite/main \
  python /home/dev/hn-rewrite/main/docs/evaluations/representation-input-20261009/harness.py validate \
  --snapshot-src /tmp/opencode/cc-replay/snapshot.db \
  --work-snapshot /tmp/opencode/representation-input-20261009/snapshot_baseline.db \
  --sample /home/dev/hn-rewrite/llm-labels/facet-residual-20261009/sample.json \
  --oof-scores /home/dev/hn-rewrite/llm-labels/facet-residual-20261009/oof_scores.json \
  --config /home/dev/hn-rewrite/main/config.toml \
  --out-dir /tmp/opencode/representation-input-20261009/work"
```

One CPU via `OMP/MKL/OPENBLAS_NUM_THREADS=1` (set in-process, as the
prior OOF run did); niced task only; no laptop heavy work.
