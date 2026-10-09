**Verdict: worth running after two MUST-FIX clarifications.** The design can separate within-source discrimination from a between-source score offset, but only if AUC is computed within blocks and the band rates are split by source.

1. **MUST-FIX: AUC must be computed within each block (PLAN.md:20–21, 33–40).** The common-support rules define strata by age only. If HN and non-HN rows are pooled across blocks, the AUC compares scores from different blocks, which the baseline report (REPORT.md:24) forbids. Use block×age strata, with weights taken from the pooled all-row count in each block×age cell, and apply the 50% support-mass rule to those cells. Separately, expect thin common support on the replay-clock age bins: HN seed stories sit in older bins than RSS stories, so an inconclusive result is likely and should be read as that.

2. **MUST-FIX: report the UP rate per source in each score band (PLAN.md:42–46).** The text says "observed UP rate in each band" but doesn't say per source. Only per-source rates (with UP, neutral and DOWN counts) at the same score bands show an offset. Apply the <10-row sparse flag to each band×source cell, since each block has only about 77 rows per band.

3. **Should-fix: pair the DOWN placement with UP placement (PLAN.md:21–22).** DOWN rows landing in the lowest 20% of ranks is correct ranking behaviour. A source offset would push both HN UP and HN DOWN rows down, so report per-source UP and neutral rates in that same 20% too. This also keeps the tail definition consistent with REPORT.md:22.

Implementation checks: how ties at band boundaries are assigned, block×age cell counts adding up to the totals, and the exclusion list for rows with unknown age.

**Next step, depending on the outcome:**
- If HN within-block AUC is about the same as non-HN but HN has a higher UP rate at the same score bands in most blocks: freeze a protocol to test a source prior on fresh votes.
- If HN AUC is lower and band rates are similar: freeze a test for a semantic blind spot (the case relevant to Muse) on fresh, untouched pairs.
- If directions are mixed or support is thin: stop.
