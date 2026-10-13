# Single queue: one stream of cards instead of views — plan (2026-10-13)

Status: the user's original idea (2026-10-13) is one queue of cards with no
separate Recommended / Popular / Explore sections. The first draft was
reviewed independently (`REVIEW-claude-cloud.md`, verdict "revise"). The
user then chose "client-first, then interleave", and asked for Step 1 to be
built now (2026-10-13).

## Why

- The user moves through cards by voting on every one (FINDINGS 2026-10-13),
  so the sections only add a "which view now?" decision.
- Each view has a job (STATUS "Goals"). Recommended saves time. Popular
  keeps the user current and gives labels the model did not choose. Explore
  (Discover, Unsure, Interest) and the far card fight a same-y feed. A queue
  must keep all of these jobs.
- View yields for profile 151 since 2026-09-24 (votes after the first
  impression; up/neutral/down):
  - Recommended 214/288/235 (29% up).
  - Popular 65/193/539 (8% up, 68% down; about 43% of all votes).
  - Explore 41/74/208.

## User decisions (2026-10-13)

1. Keep the time windows: one queue per window.
2. The queue opens by default; the other sorts stay reachable.
3. Popular gets one model-filtered slot plus one slot independent of the
   model.
   - Step 1 keeps raw top Popular.
   - Step 2 proposes a random whole-pool draw for the independent slot
     (review H3). That needs the user's OK, since it changes answer 3.

## Expected arithmetic (written down before any data, review H2)

From the repo's yields: Recommended top-8 is about 0.41 up, Discover R1
about 0.12, Popular 0.08, Unsure about 1 in 6.

- A 6/1/1/1/1 interleave gives about 2.9 ups per 10 cards, against 3.8–4.1
  for Recommended alone.
- Any mix with at most 2 Popular cards per 10 has fewer downs than today's
  Popular-heavy mix.

So neither "keep 90% of Recommended's ups" nor "fewer downs than today" is
a useful gate. The baseline is today's reading (all views, per session),
measured per fixed vote budget.

## Step 1 — client-only flow

A "Queue" sort in the clients, opened by default. In each window, it plays
the served views one after another:

1. Recommended: all served cards in served order, including the trial-draft
   arms and the far card.
2. Explore: Discover first, then Unsure, then Interest. Served order, not
   shuffled (review L7).
3. Popular: the top 5 by gravity, skipping stories already in this queue.
4. A "done" state: "Caught up for this window". The other sorts stay one
   key away.

Logging:
- Impressions keep the card's real `sort_mode` (recommended / explore /
  popular), so existing reports and trial credit work unchanged.
- Two new optional fields mark where the card was seen:
  - `surface` = `queue` or `views`;
  - `queue_position`.
- Not in `badges` and not in `deck_scores` (review M1).

Later, possibly in Step 1:
- Move the far card into the first 10 (review M5).
- Reorder Recommended so near-duplicates are not adjacent (review H4).

Live test (review M3): about three weeks with the default alternating by
UTC day. Compare per 20 votes, by day type, with bootstrap intervals over
days.
- Primary: up-votes and novel up-votes.
- Guardrails:
  - downs;
  - coverage of each window's top-5 Popular;
  - share of votes outside Recommended;
  - topic entropy of up-votes;
  - training-label mix by view.
- Secondary: dwell per useful card.

## Step 2 — server-side interleaved queue (only if Step 1 sticks)

Built at serve time from the outputs of `serve_window`, not in
`assemble_window_deck`: the trial draft, the far card, deduplication and
the age cap all happen after it (review H1, verified). Step 2 also adds:

- slot patterns 6/1/1/1/1 and a discovery-heavy 5/1/2/1/1;
- a label slot that claims first: a uniform random draw from the whole
  pool, with its propensity logged;
- priority for rec slots over other kinds (review M4);
- filtered pop by top-quartile P(down) (review L5);
- a cold-profile fallback (review L4);
- slot positions that rotate by day (review L1).

## Offline work (small, read-only)

- A τ / quartile sweep for filtered pop: ups kept against downs removed,
  with intervals.
- Yields per kind with intervals; unvoted counts as not up.

## Effects on existing work

- Trial credit uses each impression's real Recommended view, so Step 1
  does not break it.
- Shipping Step 1 before 2026-11-05 still changes how often and where
  Explore and Popular are seen. Read the far-card and Discover results
  (about 2026-11-02) knowing the queue's start date.
- Reports must say which `position` meaning they use:
  - TUI list index before 583d0bd;
  - served rank after it;
  - `queue_position` from Step 1.
