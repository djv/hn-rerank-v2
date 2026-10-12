# Recommended diversity: a less same-y Recommended — plan (2026-10-12)

Status: the user chose this as the next step on 2026-10-12 ("Diversity prep"):
offline preparation only, ready to ship after the interleaving trial ends on
2026-11-05. Recommended does not change before then.

Result (2026-10-12): Phase 0 ran; no rule passed the gate (FINDINGS.md
"Diversity Phase 0"). Phase 1 does not start from this result.

## Phase 0b — re-test on new days (user chose 2026-10-12)

Fixed on 2026-10-12, before any of its data exists:
- Data: stories first shown from 2026-10-13 to 2026-11-04 UTC (inclusive);
  everything else as in Phase 0. Run on or after 2026-11-05.
- Only these two rules are tested against B; others are printed for context
  only:
  - Far 1, unchanged.
  - Combo: MMR 0.8 picks the first k − 1 stories. The last slot goes to the
    best-scored remaining story with max similarity < 0.60 to up-votes cast
    before the group's day. If there is none, it goes to MMR 0.8's k-th
    pick.
- Same gate as Phase 0. If both pass, the larger diversity gain per up-rate
  point lost wins. A winner goes to Phase 1 with the user's OK.
- Caveat: Discover (live in Explore since 2026-10-12) puts more far stories
  on screen, so the pool holds more far stories than in Phase 0.

## Why

- The user's worst failure is a same-y feed (STATUS.md "Goals").
- Recommended covers about as many topics as a random sample of the same
  week's stories. But its mean pairwise diversity is 0.15–0.21 lower: it
  picks close neighbours within topics (FINDINGS.md "Sameness and novel
  up-votes").
- Novel up-votes are rare: Recommended had 7 of its 140 up-votes. Far cards
  in Recommended still got 11% up, against 22% for near ones.
- Explore now has Discover (live 2026-10-12). This plan is the Recommended
  side.

## Candidate rules (applied to production's order, top k = 8 served)

- B: production order (baseline).
- MMR(λ), λ in {0.9, 0.8, 0.7, 0.6}. Pick greedily by
  `λ · s(i) − (1 − λ) · max_sim(i, picked)`, where `s` is the production
  score min-max scaled within the group. Similarity is the cosine on the
  stored embeddings.
- Cap(c), c in {1, 2}: production order, at most c picks per topic (40
  k-means topics), skipping the rest.
- Far(1): production order, but the last slot goes to the best-scored
  story far from every up-vote (max similarity < 0.60), if one exists.

## Phase 0 — offline (read-only, VPS, no paid compute)

Profile 151. For each UTC day since 2026-09-24 and each window: the pool is
the stories first shown that day in that window, in any view. Each is scored
with the production model trained on votes before that day, as in the
Discover Phase 0 and short-window checks. Outcome: up if the user's current
vote is up and was cast after the first showing (re-vote window excluded),
otherwise 0. Groups with fewer than 12 stories are skipped. Each rule picks
k = 8.

Report per rule, vs B, with a paired bootstrap over days (2,000 resamples):
- up-rate and down-rate of the picks;
- mean pairwise diversity (1 − cosine) and normalized topic entropy of the
  picks;
- novel up-votes caught (up-votes with max similarity < 0.60 to earlier
  up-votes);
- all of the above split short (12h, 1d) vs long (1w, 1m).

Gate, fixed before the run. A rule passes if all three hold:
- diversity rises by at least 0.05 (about a third of the gap to random);
- the up-rate point estimate is at most 2 points below B, and the lower CI
  bound is no worse than −5 points;
- novel up-votes caught are not fewer than B's.

Among passing rules, prefer the largest diversity gain per up-rate point
lost. If none pass, report and stop.

Limits:
- The pool holds only stories already shown, and shown-but-unvoted counts as
  not liked. A rule that reaches deeper into the ranking is judged on
  stories production put on screen, not on the whole candidate pool.
- Embeddings include top comments read at snapshot time.
- Small n: a directional read only.

## Phase 1 — build (after 2026-11-05, with the user's OK)

The winning rule becomes a ranking function on top of production's order.
It runs first as a team-draft interleaving arm against production on
Recommended, reusing the trial's arm machinery (`arm_rankings`,
`draft_recommended`), for about three weeks. The weekly sameness metrics
and novel up-votes per week are guardrails. Ship as production's order only
if interleaving does not lose and diversity rises. Each deploy needs the
user's restart OK.

## Out of scope

The ranker itself, Explore and Discover, Popular, the cold-start first feed
(parked).

## Far card — live now (user chose 2026-10-12)

The user lifted the Recommended freeze ("u can make changes to recommended
now") and chose Far 1, the Phase 0 near miss, to go live before 2026-11-05.
- Each window's served Recommended keeps its first 11 cards. Card 12 is the
  best-scored story whose max similarity to the user's up-votes is below
  0.60, taken from Recommended below card 11 or from Explore's Discover
  picks. It shows the ✨ Discover badge. If there is none, Recommended is
  unchanged.
- Trial safety: the card is tagged arm `far`, outside the team draft. The
  trial report credits only the arms it names, so votes on it count for
  neither production nor a challenger. The draft keeps 11 served slots.
- Logging: `deck_scores` cards carry their arm, so far cards stay
  identifiable after the trial ends.
- Read about 2026-11-02, next to Discover: the far card's up- and
  down-rate against Recommended cards 9–11, and novel up-votes per week
  against Phase 0's baseline (B caught 4 in 35 groups).
- Readout caveats (cloud review 2026-10-12):
  - The far card is often Discover's top pick, so from now on Discover's
    Explore up-rate measures its picks 2–4. Compare the far card with
    Discover on the far card plus Discover combined.
  - Use the served positions in impressions, not `deck_scores` positions,
    since head cards can age out between warms.
- Phase 0b still runs on 2026-11-05. The far card makes its Far 1 row
  partly live data, and Combo stays offline-only.
