# Queue mix review assessment — 2026-10-14

Objective: independently review commit485c76e, the user-selected Queue opening
mix, without changing reviewed files or live behavior.

One Claude cloud review was authorized by the user's `rv`:
https://claude.ai/code/session_01F3ahKvYvtmjAhMV5m4mtBq
Mailbox branch: `r/20261014-140210-hn-queue-mix`.
Completed review commit: `a29f16c`.
Original prompt, complete diff and reviewer output are retained alongside this
assessment. The reviewer inspected source; it did not run tests.

## Verified finding

**Low — stale in-app Queue explanation.**
`clients/tui/src/hn_rerank/app.py:2850` (Why this story, opened with `w` while
a story is selected) and `:2935` (Help, `?`) still describe Recommended, then
Explore, then Popular. Both conflict with the implemented repeating
Recommended → Explore → Recommended → Popular order. Source verification
confirmed both strings. The reviewer called severity Low/Medium; the agent
rates this Low because it misdescribes correct behavior rather than changing
card selection, order or telemetry. The review's suggested no-story trigger
was incorrect: `action_why_story` returns immediately without a selected story.

Proposed correction: update both explanatory strings to describe the chosen
mix, exhausted-section skipping, badge grouping and five additional Popular
cards. This correction is not implemented; `rv` authorizes review only.

## Other observations

No confirmed runtime correctness or regression bugs were found. The mix moves
existing frozen QueueEntry objects after selection. Membership, ownership,
original serving ranks/versions, per-lane order, rounds and undo remain intact.

The property test permits empty Recommended lists and already checks the
Popular subsequence. Thus the review's claim that empty Recommended is untested
is too broad; however, no dedicated sparse-lane assertion pins its exact
mixed order, and the property test does not check the complete mixed pattern
or Explore subsequence. Those are optional coverage improvements, not bugs.
Independent lightweight probes of the actual function passed for:
- No Recommended: Explore `[1,2,3]`, Popular `[4]` → `[1,4,2,3]`.
- No Explore: Recommended `[1,2,3]`, Popular `[4,5]` → `[1,2,4,3,5]`.
- All lanes empty → `[]`.

The import-order observation is a style nit. No fix or new test was added.
Earlier full verification results remain in the reviewed commit's WORKLOG;
the full suites were not repeated for this read-only review. Physical reader
rendering remains unverified. No service restart, database write, feedback,
synthetic telemetry or second review occurred. Reviewed files remain unchanged.

Next step: fix the two in-app descriptions only if separately authorized.
