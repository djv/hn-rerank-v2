# Bounded source diagnostic — report (2026-10-09)

Frozen protocol: `PLAN.md` (sha256
`46cc6591405a6e2dc6daa8f5816961023f2e9ef4cddae459be4c658a3206618f`).
Frozen endpoints unchanged; integration completed missing denominator and
integrity checks. No new slices or bootstrap/CI; no causal calibration claims.
One report, then stop. Independent protocol reviews: reviews/.

## Inputs (all pre-existing, reused development data — not a new holdout)

- Baseline rows: `/tmp/opencode/representation-input-20261009/compare_baseline/baseline_rows.json`
  (sha256 prefix `8fce60bbf9109478`, verified by script before run).
- Snapshot: `/tmp/opencode/cc-replay/snapshot.db`, read-only (`mode=ro`,
  `immutable=1`, `query_only=ON`; file size/mtime unchanged),
  cohort IDs only, columns id/source/time). WAL 0 bytes before and after.
- Script: `run_source_diagnostic.py` (dataclasses, explicit
  boundaries); remote run via project uv, `nice -n 10`, `taskset -c 0`,
  `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB` threads = 1. Ruff, format, ty clean.
- Full outputs: `aggregates.json` (this directory). Counts below are
  raw strata/report denominators.

## Validation (script asserts, all passed)

3079 rows; blocks 2/3/4/5 (770/769/770/770); 683 UP (label 2), 1331 DOWN
(label 0), 1065 neutral; all production_scores finite; 3079 unique story IDs;
source_meta coverage 3079/3079; zero unknown-age rows under either age basis;
score-band counts sum to block n per block; common-support weights sum to 1;
every defined AUC in [0, 1]. No unconditional endpoint sparse
(min group sizes well above 10 everywhere).

Labels: UP=2 / neutral=1 / DOWN=0. Score: native `production_score`, unchanged.
HN = {hn, ch_seed, bq_seed}; non-HN = all other sources.

## Composition

HN total 1867 (hn 1778, ch_seed 71, bq_seed 18). Non-HN total 1212 across
69 `rss_*` sources; largest: lesswrong 102, slashdot 105, reddit_digitalnomad
95, reddit_localllama 81, bogleheads 67, expatfire 63, latent_space 60,
fatfire 52, tildes 45, ainews 39; long tail of 1–30-count feeds
(full per-source counts in `aggregates.json` → `composition`).

## Per-block × source-class results

| block | class | n | UP rate | score q25/q50/q75 | AUC UP–DOWN | AUC UP–rest |
|---|---|---|---|---|---|---|
| 2 | HN | 449 | 0.189 | 0.234/0.397/0.640 | 0.914 | 0.752 |
| 2 | non-HN | 321 | 0.346 | 0.384/0.628/0.814 | 0.877 | 0.683 |
| 3 | HN | 454 | 0.256 | 0.190/0.354/0.626 | 0.911 | 0.808 |
| 3 | non-HN | 315 | 0.381 | 0.394/0.683/0.845 | 0.703 | 0.594 |
| 4 | HN | 495 | 0.143 | 0.198/0.363/0.581 | 0.954 | 0.908 |
| 4 | non-HN | 275 | 0.291 | 0.458/0.684/0.836 | 0.885 | 0.802 |
| 5 | HN | 469 | 0.083 | 0.202/0.346/0.566 | 0.915 | 0.854 |
| 5 | non-HN | 301 | 0.203 | 0.532/0.747/0.874 | 0.831 | 0.717 |

Direction is uniform: HN conditional AUC ≥ non-HN in all 4 blocks on both
endpoints. Non-HN rows score higher (median 0.63–0.75 vs 0.35–0.40) and have
higher UP base rates in every block. A lower HN AUC would have indicated a
conditional discrimination difference; the observed direction is the reverse.
Equal-or-higher AUC does not validate cross-source calibration — a raw score
is not a predicted UP probability.

## DOWN placement in lowest global within-block rank 20%

Tie-preserving bottom set (score ≤ cutoff; size ≥ floor(20%)):

| block | cutoff | bottom n | HN DOWNs in bottom / total (rate) | non-HN DOWNs in bottom / total (rate) |
|---|---|---|---|---|
| 2 | 0.201 | 154/770 | 101/176 (0.574) | 27/67 (0.403) |
| 3 | 0.199 | 153/769 | 97/169 (0.574) | 12/79 (0.152) |
| 4 | 0.209 | 154/770 | 119/286 (0.416) | 20/112 (0.179) |
| 5 | 0.212 | 154/770 | 131/322 (0.407) | 11/120 (0.092) |

This is joint placement: both within-source ordering and between-source score
location affect it. DOWN placement alone cannot establish a source offset.

## Per-block age-support findings (both age bases)

Primary age = (1791401987.9873054 − story_time)/86400; sensitivity = age at
vote = (vote_time − story_time)/86400. Exact boundaries: 0–7 inclusive, >7–30,
>30–90, >90 days; labels below abbreviate them as 8–30 and 31–90.
54/80 fine strata sparse (expected at this granularity); common-support rule
(≥10 pos and ≥10 neg in BOTH classes, pooled identical weights, 50% support
rule) provides sufficient support for all 16 summaries — mass 0.81–0.95,
reported as weighted within-age AUC, not unconditional AUC:

| block | basis | endpoint | retained bins | mass | HN wAUC | non-HN wAUC |
|---|---|---|---|---|---|---|
| 2 | primary | up_down | 31–90 | 0.93 | 0.932 | 0.877 |
| 2 | primary | up_rest | 31–90 | 0.93 | 0.763 | 0.683 |
| 2 | at_vote | up_down | 0–7 | 0.89 | 0.955 | 0.877 |
| 2 | at_vote | up_rest | 0–7 | 0.89 | 0.784 | 0.683 |
| 3 | primary | up_down | 8–30, 31–90 | 0.94 | 0.916 | 0.691 |
| 3 | primary | up_rest | 8–30, 31–90 | 0.94 | 0.816 | 0.590 |
| 3 | at_vote | up_down | 0–7, 8–30 | 0.94 | 0.915 | 0.706 |
| 3 | at_vote | up_rest | 0–7, 8–30 | 0.94 | 0.814 | 0.598 |
| 4 | primary | up_down | 8–30, 31–90 | 0.95 | 0.953 | 0.882 |
| 4 | primary | up_rest | 8–30, 31–90 | 0.95 | 0.908 | 0.795 |
| 4 | at_vote | up_down | 0–7, 8–30 | 0.95 | 0.951 | 0.875 |
| 4 | at_vote | up_rest | 0–7, 8–30 | 0.95 | 0.904 | 0.796 |
| 5 | primary | up_down | 0–7 | 0.81 | 0.889 | 0.823 |
| 5 | primary | up_rest | 0–7 | 0.81 | 0.827 | 0.742 |
| 5 | at_vote | up_down | 0–7 | 0.91 | 0.895 | 0.830 |
| 5 | at_vote | up_rest | 0–7 | 0.91 | 0.832 | 0.736 |

Direction agrees across all blocks, both bases, both endpoints: HN weighted
within-age AUC exceeds non-HN (gaps 0.05–0.23). Retained bins differ by basis
(cohort age concentrates in one or two bins per block), but the direction does
not flip anywhere. No missing-support or mixed-direction inconclusiveness
occurred. The JSON field `conclusive` means support-sufficient only, not
statistical certainty. Retained bins and weights are per block and endpoint;
do not interpret differences across endpoints as the same age mixture by default.

## Shared score-quantile bands

10 bands/block over ALL rows, identical boundaries per class, ties preserved,
no empty bands. Numeric boundaries use NumPy linear quantiles; a score exactly
on an interior boundary goes to its lower band. Tied scores stay together;
duplicate boundaries could leave empty bands. AUC ties get half-credit.
Shared-band directions are mixed: HN/non-HN/tied higher-UP-rate bands are
1/8/1 in block2, 6/4/0 in block3, 5/3/2 in block4, and 5/4/1 in block5.
Five of 80 band/source cells have <10 rows. All counts are retained, with sparse
flags and per-cell UP/neutral/DOWN counts; no band-selection rule was tuned.
These are descriptive offsets, not causal adjustments, and shared bands do not
adjust away every confound. Per-band counts, UP rates, endpoints, and primary
age composition are in `aggregates.json` → `score_bands`.

## Conclusion (descriptive only)

No HN discrimination deficit is observed: HN conditional AUCs meet or beat
non-HN in every block, endpoint, age basis, and common-support summary, and
HN DOWNs are over-represented in the bottom score quintile. Non-HN has higher
overall scores and UP base rates, while shared-band UP-rate directions are mixed.
Neither pattern proves that source score offsets are correct or incorrect.
This weakens the proposed broad HN discrimination audit and establishes no Muse
advantage. Per protocol: one report, stop here; any next model/feature design
needs a separate frozen protocol with fresh untouched eligible-pool/exposure
validation.

Root integration reran the same endpoints after adding per-band/age class
denominators, endpoint-specific sparse flags and read-only identity checks.
Unconditional AUCs, common-support summaries, DOWN placement and composition
match the initial run exactly. AUC ties/one-class cases, score ties, aggregate
class/age counts and weight bounds checked; artifact Ruff/format/ty passed.
Initial script SHA256: 92de311937b7cb8e7931f409192eeccb609cddbcd8757f2c5e28d82c927831d0;
final script hash in aggregates.json metadata. No production changes or new
ranking annotations, fits or embeddings.
