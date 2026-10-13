# Single queue: one stream of cards instead of views — plan (2026-10-13)

Status: the user's original idea (2026-10-13): one queue of cards, no
separate Recommended / Popular / Explore sections. The user chose "plan +
independent review first". Nothing is built or deployed; the ranking trial on
Recommended runs unchanged until 2026-11-05.

## Why

- The user moves through cards by voting on every one (FINDINGS 2026-10-13),
  so sections only add a "which view now?" decision. One stream suits that.
- Each view today serves one goal (STATUS "Goals"): Recommended saves time,
  Popular keeps the user current and gives model-independent labels, Explore
  (Discover, Unsure, Interest) and the far card fight the same-y feed. A
  queue must keep all four jobs, as slots, not as places to visit.
- View yields (profile 151, votes after first impression, since 2026-09-24;
  up/neutral/down): Recommended 214/288/235 (29% up), Popular 65/193/539
  (8% up, 68% down), Explore 41/74/208. Popular costs the most votes per up.

## Design

Per time window (the window control stays; see open question 1), the server
adds one more order, `orders["queue"]`, built in `assemble_window_deck`
(`pipeline/ranking.py:2142`) from the orders it already computes. No new
model, no new scoring pass.

Slot pattern, repeated every 10 cards (starting mix; Phase 0 tunes it):

| Slots | Kind | Source | Goal |
|---|---|---|---|
| 6 | rec | Recommended order (trial draft and far card as today) | save time |
| 1 | pop | Popular order, skipping cards the model rates P(down) > τ | stay current |
| 1 | discover | Discover picks (max sim to up-votes < 0.60) | discovery |
| 1 | unsure | highest-entropy picks | labels where the model is unsure |
| 1 | pop-raw | Popular order, no model filter | model-independent labels |

- Fixed slot positions (e.g. pop at 3, discover at 5, unsure at 7, pop-raw at
  9), not random, so a session of any length sees each kind early.
- Dedup across kinds: a story takes the first slot that claims it; the next
  candidate of that kind fills the other slot. Empty kinds pass their slot to
  rec.
- Interest (🎯) is dropped from the queue at first: Discover covers the far
  side and Interest's yield (2/2/2 recently) is not distinct enough to earn a
  slot. Phase 0 checks this.
- Every card carries its slot kind. Impressions already carry `badges`; add
  the kind there (`queue_rec`, `queue_pop`, ...; badge kinds must match `[a-z_]{1,24}`, `server.py:2734`) and in `deck_scores` cards
  (`view="queue"`, `slot`), so yield per kind is measurable and pop-raw gives
  an unbiased (model-independent) label stream for evaluation.
- Clients: TUI and web gain a Queue sort and open on it; the other sorts stay
  reachable for the evaluation period (open question 2). Votes advance as now.

## Phase 0 — offline replay (read-only, VPS)

Same set-up as the diversity and Discover Phase 0 (FINDINGS): profile 151,
UTC days since 2026-09-24, pool = stories first shown that day in that
window, scored by the production model trained on votes before that day.
Outcome per shown-and-voted story. Rerun for each candidate mix:

- τ for the pop slot: on Popular-voted stories, ups kept vs downs removed for
  τ in {0.5, 0.6, 0.7, 0.8}.
- Mix variants: the table above; 7/1/1/1/0 (no pop-raw); 5/2/1/1/1.
- Measures per 10 cards, against Recommended alone and against the current
  "visit all views" baseline: expected ups, expected downs, novel ups
  (max sim < 0.60), topic coverage (40 k-means topics) and mean pairwise
  diversity, as in the diversity plan.
- Gate (fixed before running): the chosen mix keeps ≥ 90% of Recommended's
  expected ups per 10 cards, has ≥ 1.5× its novel ups and higher topic
  coverage, and has fewer expected downs per 10 cards than the user's
  current mix of views (time-weighted by impressions per view).
- Caveat: only shown-and-voted stories have outcomes; Unsure and pop-raw are
  scored on few labels. Report counts and intervals, not just means.

## Phase 1 — live (after 2026-11-05, with the user's OK)

- Ship `orders["queue"]` and the Queue sort as the default in TUI and web;
  restart once.
- Run two weeks. Compare per slot kind and overall: up-rate, down-rate,
  votes per up, novel ups per week, topics per session, time per useful card
  (dwell from impressions, 90 s idle cap). Baseline: the two weeks before.
- Interleaving later: future ranker trials team-draft only the rec slots
  (`draft_recommended` unchanged), and the credit report treats `queue_rec`
  impressions as Recommended.
- If the user then prefers the queue, remove the other sorts from the
  clients (server orders can stay for reports).

## Effects on existing work

- The 2026-11-05 trial endpoint and report are untouched (built only after).
- Far-card / Discover reads (about 2026-11-02) use pre-queue data.
- Diversity Phase 0b (2026-11-05) still runs; if Combo wins, it applies to
  the rec slots.
- Reports keyed on `sort_mode` (`interleave_report.py`, `badge_yield_report.py`,
  `narrowing_report.py`, `source_yield_report.py`) need the `queue` mode and
  slot kinds.

## User decisions (2026-10-13)

1. Keep the time windows: one queue per window.
2. Queue opens by default; the other sorts stay reachable during the test.
3. Popular: one model-filtered slot plus one raw slot, as designed.

## Independent review (2026-10-13)

Claude cloud review: `REVIEW-claude-cloud.md`, verdict "revise". The agent
checked its code claims against the checkout:
- H1 is correct: `WindowViews` has three fixed views (`pipeline/ranking.py`
  ~L2031), and `assemble_window_deck` runs before the draft, the far card
  and `serve_window`.
- M4 is correct: Popular does not exclude Recommended, while Explore does
  (~L2304–2320).
- M1 is correct: `deck_scores` is written at warm time
  (`server.py:2138`, `_record_deck_scores`).

H2's arithmetic matches the repo's yields (Recommended top-8 about 0.41 up
in Diversity Phase 0, Discover R1 about 0.12, Popular 0.08). With it, the
gate as written would fail on keeping ups and pass trivially on downs. The
plan is to be revised before any build.
