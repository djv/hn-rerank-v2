**Verdict:** Run the cheap within-source discrimination and calibration check once, then stop slicing the spent labels. No result yet justifies a new LLM approach or the native component audit. No Muse advantage has been established.

**Key issues (evidence)**
1. **The HN tail pattern only looks at upvoted (UP) rows and uses a synthetic ranking.** The tail-by-source table (missed-upvotes §2) counts only UP rows; it has no per-source DOWN mirror. HN could be correctly ranked low overall and still crowd the UP tail. The top-12 is ranked among voted rows only, not the live eligible pool. Every voted row was shown to the user (cc-replay §1, exposure bias), so these "missed" UPs were actually found. The 19 non-UP picks are a total across all four blocks and are not live headroom (§Cohort).
2. **The Muse evidence does not show an independent advantage.** In targeted-muse, 11/12 is accuracy against user labels on a stratum picked because production was wrong and Opus was right. On regressions Muse scores 5/11 and leans toward Opus, so the stratum design cannot separate Muse skill from agreement with Opus. In cc-replay §3, fixes minus damage is 2−4, with confident picks of DOWN-voted stories. The facet-to-ML transfer failed (pooled Δ −.051). One design flaw there, C=1 also penalizing the baseline, is untested and cannot be retuned on spent outcomes.
3. **The component audit is premature.** Matching low-ranked HN UP/DOWN rows selects on rank, which conditions away the very offset in question. Without knowing first whether the HN problem is within-source or between-source, the audit is fishing.

**Is the source-AUC check worth doing?** Yes, but only to route the HN lead, not as a search for Muse wins. It costs zero annotations and can cheaply kill the lead. If within-HN AUC is equal to non-HN, that does not prove the source prior is right, so calibration must be measured alongside it.

**Next measurement (zero new annotations, descriptive)**
- **Inputs:** existing baseline predictions for blocks 2–5 (3,079 rows), the label, broad source, and age at vote.
- **Per block × {HN, non-HN}:** UP/neutral/DOWN counts and base rates; UP-v-DOWN and UP-v-rest AUC with story-level bootstrap intervals; observed UP rate by within-block rank decile per source; per-source DOWN tail rate.
- **Age controls:** age tertiles with cut points fixed before any outcome is read. Sensitivity runs: drop `ch_seed`/`bq_seed` rows, and drop LessWrong (n=27).
- **Reading the outcome:**
  - HN AUC similar to non-HN, and per-decile UP rates similar within age strata: the tail is a base-rate/slot artifact. Stop.
  - HN UP rate above non-HN at the same decile: a between-source or age offset. That would be an ML-side prior hypothesis, not a Muse target.
  - HN AUC lower in the same direction in every block: a within-HN discrimination hypothesis for a future test.
- **Stop rule:** report the result and end. Any follow-on needs a separate user decision.
- **Confirmation:** freeze the hypothesis and analysis code. Evaluate only on votes after 1791511537.886615, with logged exposure position and the live pool, ideally via interleaving or an exploration slot. That way low-ranked items get exposed rather than inferred.

**Proposed heuristics (not evidence):** that HN is under-ranked because of a semantic blind spot; that Muse would help on HN.

**Uncertainty:**
- Blocks 4 and 5 have only 11 tail rows each, and the per-block HN UP counts in those blocks are 39–71, so the intervals will be wide.
- Source and age are confounded.
- "Zero post-window votes" is historical (cc-replay §4) and has not been checked against current data.
- The cut points and decile bins are descriptive tools, not calibrated statistical gates.
