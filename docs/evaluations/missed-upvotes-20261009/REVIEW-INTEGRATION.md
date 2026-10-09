# Claude review integration — 2026-10-09

The user selected a bounded Claude review. `CLAUDE-REVIEW.md` contains the actual
read-only `claude -p` response, unedited. No new ranking annotations, fits,
database reads, tests or production changes were performed for this review.

## Adopted recommendation

A within-source discrimination check should precede the full score-component
audit. Using existing predictions, compare HN UP-versus-DOWN and UP-versus-rest
AUC against non-HN in each block, with source label denominators and an explicitly
defined age-stratified sensitivity check. This is a cheaper, exploratory check
of whether the HN tail pattern warrants component inspection. No computation of
these new slice metrics has happened yet. No claim of a Muse advantage.

The top-12 replay ranks historically voted rows, not all live eligible candidates.
Its 48-slot ceiling and 19 non-UP selections are arithmetic about that cohort;
they do not estimate dashboard gains. Historical exposure also shapes labels.
Preserve these caveats in every future comparison.

## Corrections and limits on the review

- Targeted Muse 11/12 was correctness against known user labels in a deliberately
  selected production-wrong/Opus-right stratum. It also agrees with Opus there;
  it was not merely an agreement metric. Overall 25/35 vs production20/35 is
  also selection-biased. This does not establish a general Muse advantage.
- The reported zero post-window votes was a past CC-replay snapshot check, not a
  fresh read of current votes. This review did not establish current availability.
- Equal within-source AUC cannot prove that the HN source prior is appropriate,
  or exclude narrower blind spots. It can weaken the argument for a broad HN
  component audit. Higher UP counts alone do not explain deep rank placement.
- Proposed .05/.03 gap cutoffs are heuristic exploratory stopping thresholds,
  not established power/error controls. Small age strata may be uninformative;
  report missing/low-count cells, never convert them to a pass/fail by default.
  Predefine age strata and treatment of mixed results before computing slices.
- These outcomes have already been repeatedly inspected. A new frozen slice
  rule remains development work; future untouched outcomes are necessary for
  confirmation. November5 interleaving end belongs to the existing experiment;
  it is not evidence that every future offline read must wait until then.

Next: specify one small within-source check, then let cheap Muse compute it on
existing VPS artifacts. Only evidence of useful discrimination failures would
justify component inspection and a separately frozen correction design.
