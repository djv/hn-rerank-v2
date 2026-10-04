# HN Rerank status

Saved 2026-10-02 after the 21:22 UTC analysis. Previous handoff:
[Popular badge stacking](docs/status-archive/before-upvote-policy-analysis-20261002.md).

## Objective
Think how to improve profile 151's upvote rate and lower downvotes.
Analysis is complete; no ranking/subscription/runtime change was requested.

## Verified result
- Live `176cc59`, service active; 3,340 reactions, latest at 20:30 UTC,
  last impression 20:49. Since Sep 30: Recommended 25/206 up, Popular 6/177,
  Explore 1/90. Today Recommended/Popular are both around 4-5%.
- Fresh read-only, current-config replay (Gemma on): four chronological
  blocks, 494 evaluated reactions / 63 upvotes. In 48 Popular replay slots,
  day-clock gravity finds 5 up / 30 down; 70/30 preference/gravity finds
  9/18; model-only 9/16. The 12h clock favors personalization too.
- Extra downvote penalty changes only one disliked Popular card to neutral.
  Explore rejection returns three fewer disliked cards, with no lost likes,
  but fewer slots; quality-floor/discovery benefit remains unproven.
- Recommendation: trial HN-only Popular's existing shortlist with 70/30
  preference/gravity first; separately test Explore quality eligibility.
  Selective source controls may help; blanket source boosts previously did
  not. Evidence, safeguards and prospective trial criteria are in FINDINGS.md.
- Read-only analysis audited; batch finished with one CPU / 3G cap and
  runner exit verified. Private artifacts: laptop and VPS
  `~/.local/state/hn-rerank-eval/policy-analysis-20261002/`.
- Badge stacking remains deployed and complete; prior verification is in
  the archived handoff. Review added documentation only.

## Blocker / limits
- No blocker for the requested analysis. Offline reordering of exposed
  stories with current content/points is not proof of future live uplift.
- Only 48 Popular slots and 12 known likes in its replay pools. Live badge
  outcomes are still too sparse. Raw model scores are not quality probabilities.
- Capped encoder and archive-report classification findings remain parked,
  as do TLDR fixes. Unrelated TUI test, mockups, TLDR script and kernel log
  are preserved.

## Next step
- Implement the proposed Popular pilot only if the user requests it.
  Keep the existing HN shortlist and other features fixed; record the arm
  on both clients and compare up/down per unique first impression.
- Earlier reader, ranking and LLM work remains in
  `docs/status-archive/before-ranking-review-handoff-20261002.md`.
