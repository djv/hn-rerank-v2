# HN AI ranking research status

Updated 2026-10-09.

Goal: identify a repeatable Muse advantage on the user's votes and turn it into
an ML ranking improvement. User-selected priority: find more upvotes; track
downvotes near the top as a guardrail. No production deployment authorized here.

Verified: completed facet trial failed all five original gates. Representation
audit read 3,947 voted snapshot rows (3,848 old + 99 dev): 3,896 stored texts equal
the full composer, 51 differ, 516 have no clean self/article body. Bounded search
did not locate reusable body-only vectors or results. Audit script passed Ruff,
format and ty checks. No new ranking annotations or embedding inference ran.

Current result: Muse implemented manifest/baseline validation under
../representation-input-20261009. Root diagnosed a thread-initialization
reference mismatch; the new harness reproduces the original script under the
same explicit one-thread environment exactly across 3,079 rows. Original failed
reference and successful amended reference are preserved. Full comparison and
encoding are not implemented yet. See that directory's STATUS.md and PLAN.md.
Each arm requires fresh process/copy, unchanged story/side inputs, no story
upserts, no cross-arm warm state and a fixed scoring clock.

Next: freeze script/config identities, verify full-text encoder parity on
20 rows, then run capped single-CPU primary-input encoding/fits on the VPS.
On mismatch or parity failure, stop and diagnose before new encoding. All reused
outcomes are exploratory; a Muse canonicalizer and prospective validation come
only after a promising clean-input comparison.
