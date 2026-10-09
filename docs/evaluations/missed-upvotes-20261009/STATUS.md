# Missed-upvotes analysis status

Updated 2026-10-09.

Objective: identify repeatable mistakes that Muse can help correct in the ML
ranker, prioritizing upvote discovery and conserving ranking annotation quota.

Result: existing 3,079 held-out rows analyzed. HN contributes87/137 deep-missed
UPs versus311/683 all UPs; overrepresentation direction holds in all4blocks.
This is an exploratory error pattern, not demonstrated Muse superiority.
Baseline top48 contains29UPs;19non-UP slots could in principle be replaced.
Cached facets cover61/683UPs and cannot establish a useful correction rule.
No new ranking annotations, fits or production changes.

Limits: reused labels/exposure, source/content/history confounding, sparse facets;
no confirmed ranking gain. See REPORT.md and aggregates.json for denominators.

Review: user-selected Claude review completed; actual response CLAUDE-REVIEW.md,
corrections and adopted direction REVIEW-INTEGRATION.md. Top12 replay is synthetic
over voted rows, not live eligible candidates; its ceiling is cohort-specific.
Subsequent independent Codex/Muse/Claude Code reviews all favor one bounded
source-discrimination check; current handoff ../three-review-20261009/STATUS.md.

The authorized source diagnostic subsequently completed: native HN discrimination
is stronger, not weaker, in all four blocks and both age controls. Broad component
audit deferred. Current handoff ../source-diagnostic-20261009/STATUS.md; stop after
that report. All reused data remains exploratory; fresh votes required for
confirmation, with recovery and damage measured together.
Representation rewriting branch remains stopped after failed frozen gates.
