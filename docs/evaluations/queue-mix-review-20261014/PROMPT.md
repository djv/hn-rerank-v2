# Independent read-only review: Queue opening mix

Review commit 485c76e against parent dd92d18. The supplied CHANGES.diff is the
complete change; source files are the current checkout at that commit.

The user requested Queue to differ from Recommended near the start, mixing
cards from other sections. The user chose the repeating pattern Recommended →
Explore → Recommended → Popular, skipping exhausted sections. Existing card
selection, duplicate ownership (Recommended before Explore before Popular),
Explore badge grouping and the five-additional-Popular cap should remain.

Assess concrete correctness or regression issues in this change, including:
- Mixed openings, missing/duplicate/excluded/restored cards, sparse lanes.
- Per-section order and origin rank/version preservation.
- Reader round freezing, refresh churn, undo, window switches, refills.
- Telemetry preserving original view/rank/serving version and mixed queue slots.
- Tests: whether they verify meaningful behavior and miss a consequential case.

Do not broaden the review into changes to server ranking, trial design or
unrelated client behavior. Treat source as authoritative. No database, private
profile, credentials or live interaction data are supplied or needed.

Validation already run by the implementing agent: focused Queue 23 passed;
full TUI 275 passed / 1 skipped; root and standalone Ruff/ty passed; changed
Python formatting passed. The unchanged backend suite on the VPS passed
1445 / 1 skipped, with an existing MLP convergence warning. Read-only existing
profile API feeds in all five windows composed 26 unique cards each, with the
chosen first-four origins and correct original ranks and versions; the feeds
were still warming. Physical reader rendering remains unverified. No server
restart was necessary for this client-only change.

Read the supplied code independently. Report only actionable findings with
severity, original repository file:line, concrete trigger, consequence and
suggested correction. Distinguish confirmed bugs from open questions or testing
limitations. If no issues are found, say so explicitly and note relevant limits.
Do not modify the reviewed files, launch another reviewer, or contact services.
Write only the requested REVIEW.md in the review mailbox and push that file.
