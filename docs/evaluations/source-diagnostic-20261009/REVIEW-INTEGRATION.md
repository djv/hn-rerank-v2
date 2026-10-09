# Protocol review integration — 2026-10-09

The user requested Codex, Muse and Claude Code review while the diagnostic was
running. Their independent reviews assessed the frozen PLAN and baseline report;
they did not review mutable code or these new results. Actual responses unchanged
under reviews/. All three calls completed exit0.

All support bounded descriptive measurement, with clarifications. Root checked
these against the implementation rather than changing the experiment after data:

- AUC, endpoint-specific support, pooled all-row weights and support mass are
  calculated separately within each block. No cross-block score pooling.
- Shared numeric quantile bands report each source group's rates separately.
  Root completed per-cell UP/neutral/DOWN counts and <10-row sparse flags.
- Ties get AUC half-credit; NumPy linear quantiles define bands; scores on an
  interior cut enter the lower band, so equal scores remain together. Duplicate
  cuts can create empty bands. Bottom20 uses floor(0.2*blockN), then includes
  every score tied at its cutoff. DOWN placement denominator is all DOWNs in
  that source/block. It is joint placement, not a pure source-offset measure.
- Common-support summaries are weighted within-age AUC. Separate endpoints can
  retain different age mixtures; 50% mass is a support convention, not certainty.
- Shared-band rate differences remain age/composition/exposure-confounded. Root
  corrected the initial report's erroneous claim that non-HN rates were higher
  in most bands: directions are mixed, and five source/band cells are sparse.
- Claude's optional UP/neutral bottom-tail addition was not made into another
  post-result slice. Existing shared bands already contain all class counts;
  the diagnostic stops as frozen.

Root also enforced immutable read-only/query-only snapshot access and before/
after input identity checks. A verification rerun left original AUCs, support
summaries, DOWN placement and source composition exactly unchanged. No ranking
annotations, embeddings, model fits or production changes.

Checked result: HN discrimination exceeds non-HN on both endpoints in all four
blocks, also in all16support-sufficient summaries over both age bases. This
weakens the broad HN discrimination-failure hypothesis. It does not establish
source calibration correctness, an LLM advantage or a deployable improvement.
Stop after this report; any later model/feature claim needs a separate frozen
protocol and genuinely untouched future eligible-pool/exposure validation.
