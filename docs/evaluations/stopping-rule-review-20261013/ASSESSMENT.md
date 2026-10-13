# Stopping-rule review assessment — 2026-10-13

User requested `rv` on continuing the ranking trial until enough evidence
exists, rather than requiring a calendar end. One independent Claude cloud
session reviewed the protocol, draft/report source, report tests and saved
aggregate October12 report, before seeing previous reviews:
https://claude.ai/code/session_015L8mTPGhnMKiBuqE9ugy7K
Original review is preserved unchanged in REVIEW.md. No DB, credentials or
private context supplied. No additional review or implementation authorized.

## Verified findings

- Medium methodological limitation: scripts/simulate_interleaving.py:160-181
  counts each credited vote in a pooled two-proportion z-test. The report
  calls it at scripts/interleave_report.py:168. Neither deck/session
  clustering nor continuous monitoring is accounted for. A fixed date
  preserves the chosen stopping rule; it does not establish nominal error
  control under dependence and adaptive retraining.
- Medium data limitation: scripts/interleave_report.py:241-249 reads final
  mutable feedback. Later changes/undo can remove historical credits. A
  consistent snapshot is required for repeatable boundary analysis; a new
  stopping rule alone does not solve this.
- Verified metric limitation: denominator is credited votes, not exposed
  cards. Skipped cards do not enter it. Exposure-normalized upvotes and
  position summaries are useful complements, but require an explicit
  definition of exposure, repeated impressions and attribution/window.
- Low transition behavior: scripts/interleave_report.py:236 loads only
  decks created since the start. Post-boundary votes on older decks are
  excluded. Stage2 must define this explicitly.
- Verified protocol: three arms; original November5 04:07:22 UTC boundary;
  already allows an inconclusive result. FAR_ARM skip is at report:124-125.

## Corrections to reviewer coverage and recommendations

- Rejected missing far-card test claim: tests/test_far_card.py:395-415
  directly exercises report credit exclusion; :368-389 checks storage.
  Those tests were outside the supplied bundle. No missing regression fix
  is warranted on that claim.
- Click/open/dwell telemetry exists: database.py:24,206-214,511-528 and
  clients/tui/src/hn_rerank/dwell.py; normal dwell/article delivery was
  verified in the current handoff. Reviewer correctly qualified its claim
  as limited to supplied files; no missing instrumentation claim accepted.
- The review contains inaccurate line references (including aggregate JSON
  lines beyond its length); verified references above supersede them.
- A deck-level unit does not automatically establish independence or the
  martingale assumptions of a sequential test. Adaptive methods can be
  valid under suitable conditional assumptions; adaptation does not
  universally invalidate them. Those assumptions must be established for
  the actual estimand and attribution scheme before choosing a method.
- Proposed150 credited decks, February1 cap and effect thresholds are
  examples, not validated sample sizing or approved choices. Conditioning
  solely on decks with credit also omits ignored exposure. Neither targets
  nor the suggested alpha-spending/combination method are accepted as ready
  to implement. Generic iid power estimates are illustrative, not a
  reliable schedule forecast for this trial.
- An interval excluding a positive useful improvement can rule out that
  improvement; it does not by itself prove a practical tie. A tie requires
  uncertainty contained within a prechosen equivalence region, with harm
  assessed too. No significant harm is not proof of safety.

## Recommendation

Revise the agent's earlier recommendation: preserve the original November5
snapshot/report and accept an inconclusive result. Collection can continue,
but any continuation should have a separately documented prospective rule.
Use earlier data as exploratory/contextual evidence; combining it into a
formal staged claim needs a justified method, not an ordinary pooled
p-value. A fresh stage is a simpler defensible path than retroactively
changing the original endpoint.

Before implementation, choose a practically useful improvement and define
the outcome/exposure unit, dependence treatment, stopping/cap, harm and
diversity guardrails, and handling of ties/futility. Continuous collection
is compatible with this; a calendar date need not force a winner. No user
choice, runtime change, capture job or code edit made by this review.

External primary-source checks support the need to verify sequential
assumptions and the distinction between clicked-credit interleaving and
voted-card quality, without validating this proposed design:
- https://arxiv.org/abs/1810.08240
- https://proceedings.mlr.press/v139/karampatziakis21a.html
- https://www.cs.cornell.edu/~tj/publications/chapelle_etal_12a.pdf

Verification was read-only source inspection and primary-source research;
no code changed and no tests rerun. Reviewed project files remain unchanged.
