# Bounded source diagnostic — frozen before new slice computation

Authorized: user questionnaire selected Run cheap diagnostic on2026-10-09.
Goal: decide whether the observed HN tail warrants a later audit, conserving
ranking annotation quota. All outcomes are reused development data, not a new
holdout. Stop after one report; no automatic intervention or further slice search.

## Fixed inputs and endpoints

Existing VPS baseline rows:
/tmp/opencode/representation-input-20261009/compare_baseline/baseline_rows.json
(SHA256 prefix8fce60bbf9109478);3079rows, blocks2–5. Source snapshot:
/tmp/opencode/cc-replay/snapshot.db, read-only immutable only after verifying no
nonempty WAL. Query cohort IDs only, columns id/source/time. No private text or
individual prediction rows copied locally. Do not read new votes or live DB.
Use source and story time from snapshot; vote time and production_score from
existing baseline JSON. Labels UP2/neutral1/DOWN0. Native score unchanged.

HN = hn/ch_seed/bq_seed; non-HN = every other source. Report non-HN composition,
per-block/source class counts and UP base rates, score summaries, UP-versus-DOWN
and UP-versus-rest AUC, plus source-specific DOWN placement in the lowest global
within-block rank20%. These are descriptive conditional metrics, not Muse scores.

## Fixed age controls and bands

Primary age = (frozen scoring clock1791401987.9873054 - story_time)/86400.
Single sensitivity: age at vote = (vote_time - story_time)/86400, same bins.
Both use age bins0–7days inclusive, >7–30, >30–90, >90. Missing/nonpositive
story time or negative age is unknown, separately counted; no silent clamping.
No alternative tertiles, source exclusions or optimized cut points.

Report all strata/counts; AUC undefined only when one label group is absent.
Flag an endpoint sparse when either of its two groups has <10rows; do not treat
this as evidence of failure. For common-support summaries retain only strata
with ≥10positive and ≥10negative rows in BOTH source groups for that endpoint.
Use identical label-independent weights from pooled all-row age-stratum counts,
renormalized over retained bins. Describe this as a weighted within-age AUC,
not unconditional AUC. Report support mass; if <50% of known-age pooled rows,
summary is inconclusive/undefined. Thresholds are diagnostic support conventions,
not calibrated statistical gates.

Use10common score-quantile bands per block, computed over ALL block rows without
labels/sources. Same numeric boundaries for HN and non-HN; preserve ties and
report empty bands. Report source class counts and observed UP rate in each band,
score endpoints and age composition. A raw score is not a predicted UP probability.
Shared bands are descriptive offsets and do not adjust away every confound.

## Uncertainty and stopping

No bootstrap, confidence claims, fitting or thresholds for a winner. Show
counts, sparse/common-support flags, direction per block and both age bases.
A lower HN AUC indicates conditional discrimination difference, not mechanism
or LLM superiority. Equal AUC does not validate cross-source calibration.
Different observed rates in shared bands are descriptive, not causal offsets.
Mixed directions, sparse bins or missing common support remain inconclusive.
Whatever the outcome: one report, stop. Any next model/feature design needs a
separate frozen protocol and fresh untouched eligible-pool/exposure validation.

## Execution and checks

Muse implements a small typed artifact-only script; remote run via project uv,
nice10, oneCPU/thread caps; no new dependencies. No annotations, fits, embeddings,
service/config changes, production reads/writes or full regression sweeps.
Inspect remote instructions/headroom; only task-owned output files writable.
Check input hash, unique IDs, finite scores, cohort counts, source/time coverage,
read-only source identity before/after and aggregate count/weight invariants.
Artifact Ruff/format/type checks and meaningful scorer/tie/support checks only.
Root reviews output before reporting conclusions. Record plan hash in results.
