# Bounded source diagnostic status

Updated 2026-10-09.

Objective: clarify HN discrimination versus source score placement using existing
predictions, conserving Muse ranking quota. User authorized the diagnostic and
subsequently requested independent Codex/Muse/Claude Code protocol reviews.

Verified result: 3,079 rows. Native HN UP/DOWN AUC .914/.911/.954/.915 versus
non-HN .877/.703/.885/.831; UP/rest also higher in all four blocks. Both frozen
age bases retain this direction, with support mass .81–.95. HN UP base rates
are lower in every block; shared-band UP-rate directions are mixed, five sparse
cells. No broad HN discrimination deficit or demonstrated Muse advantage.

Checks: input hashes/unique IDs/counts; immutable/query-only read; unchanged
snapshot size/mtime and zero WAL; tie/one-class/count/weight checks; artifact
Ruff/format/ty. Integration rerun reproduced all original endpoints exactly.
All worker/review calls completed; reports and actual reviews saved locally.
No new ranking annotations, fits, embeddings, production changes or full suites.

Limits: reused/exposure-selected labels, heterogeneous non-HN, residual age/
composition confounding; support-sufficient does not mean statistically certain.
Reviews concern protocol; root checked implementation/results separately.

Decision: stop this diagnostic. Defer the broad HN component audit on this
hypothesis; no automatic model/feature trial. Future ranking improvement requires
another frozen design and untouched eligible-pool/exposure validation. No blocker.
