# Independent ranking research reviews — 2026-10-09

Codex, Muse and Claude Code all recommend one small within-source discrimination
check before any component audit or new LLM experiment. None finds an established
Muse advantage. This agreement is a review judgment, not experimental evidence.

## Actual responses and provenance

- CODEX.md: Codex CLI0.162.0, gpt-6.1-sol, high reasoning, read-only sandbox,
  ephemeral run, exit0; final response copied unchanged.
- MUSE.md: opencode-go/muse-spark-1.3-contributor, fresh session, pure mode,
  read/grep/glob allowed and other tools denied; exit0. Text response extracted
  from JSON events, including its short initial progress sentence.
- CLAUDE-CODE.md: direct claude-p, Read/Grep/Glob only, no session persistence or
  MCP tools, exit0; stdout copied unchanged. This is a new review, separate from
  the previous one in missed-upvotes-20261009/CLAUDE-REVIEW.md.
- PROMPT.txt: common review request. Reviewers were instructed not to read each
  other's answers or previous review opinions. Existing five aggregate reports
  were the evidence. No new ranking annotations, model fits, experiments or live
  changes. Raw tool logs remain temporary, outside project documentation.

## Agreement

1. HN's deep-UP-tail enrichment lacks corresponding within-source DOWN
   discrimination. Exposure, class composition and selection limit interpretation.
2. Top12 synthetic replay is over historically voted rows, not live eligible
   candidates. These UPs were exposed and voted on; deep rank here does not mean
   the production feed never found them. The19non-UP slots total all4blocks and
   do not establish live headroom.
3. Targeted Muse11/12 is true user-label correctness on a selected stratum, but
   cannot establish general superiority. Correction damage and failed facet/
   representation trials remain negative evidence for those tested designs.
4. Similar within-source AUC does not validate cross-source score calibration.
   Native production scores are ranking scores, not probabilities of an UP.
5. One bounded no-annotation check may clarify the HN lead. Stop after reporting;
   any subsequent intervention requires a separately frozen design and untouched
   future validation. No retuning spent outcomes into confirmation.

## Differences and corrections

- Codex proposes fixed age bins at the replay clock and common-support weighting;
  Claude proposes age-at-vote tertiles and extra source-exclusion sensitivities.
  Those are different controls. Define one primary age basis and one small
  sensitivity before computing; do not run whichever bins produce the best gap.
- Muse would stop additional historical component work regardless of the result;
  Codex treats the check as a bounded diagnostic before a possible audit; Claude
  routes findings into within-source versus between-source hypotheses. Adopt the
  shared minimum: produce one report, no automatic follow-on experiment.
- Lower HN AUC is a conditional discrimination difference, not proof of a semantic
  mechanism or that Muse can fix it. Equal AUC plus similar coarse-bin UP rates
  also cannot prove the tail is entirely a base-rate artifact.
- Report observed UP rates in common within-block score/rank bands shared across
  sources. Source-relative rank deciles are insufficient for assessing a
  between-source offset. Do not label raw score0.8 as an80% predicted UP chance.
- Any prior zero-future-vote claim is historical. An old timestamp is not sufficient
  to designate data untouched now: exclude all outcomes already inspected, and
  choose a new explicit cutoff when freezing a prospective design.
- These review thresholds, bins and uncertainty summaries are proposed diagnostic
  tools, not calibrated statistical gates. Sparse cells and missing common age
  support mean inconclusive, not pass/fail. Non-HN is a mixed-source population.

## Proposed next measurement — not performed

Use existing baseline predictions for blocks2–5, labels, normalized source,
recorded story/vote times and the frozen scoring clock. Cheap Muse does the work
on existing VPS artifacts; no new annotations, embeddings, fits or score changes.

Report per-block HN/non-HN class counts and UP base rates; UP-versus-DOWN and
UP-versus-rest AUC; score distributions and observed UP rates in shared rank/score
bands; per-source DOWN tail placement. Predefine age control, common-support
handling and sparse-cell treatment. Uncertainty, if computed, is conditional on
existing fits and exploratory only.

Stop after a single report. Similar discrimination weakens the broad component-
audit case; poorer HN discrimination suggests an error hypothesis; different UP
rates at comparable scores suggest a between-source calibration hypothesis.
Neither outcome establishes Muse superiority. No additional slice search or
automatic feature implementation. Future confirmation needs a frozen design,
untouched votes, eligible-pool/exposure records and an appropriate live comparison.
