# Section auto-refresh on s/h/l: bounded comparison (2026-10-09)

Prototype only. No production code changed. Experiment:
`test_section_refresh.py` (250 lines, ruff/format/ty clean), run in an
isolated VPS scratch copy of the exact laptop TUI source
(`sha256 app.py f4484d2…`, reused save-checkout venv, `PYTHONPATH=<scratch>`,
`OMP/MKL/OPENBLAS_NUM_THREADS=1`, load 0.29, 8 passed in ~10.6 s).
Counts below are Pilot-fixture request counts, not timings or observed UI.

## What s/h/l does today (verified against live tree + Pilot)

- One shared time-window feed: `Feed.orders` holds recommended/popular/explore
  for the window; `action_cycle_sort` only flips the sort `Select`, and
  `on_select_changed` does a local `rebuild(select_id=-1)` — selection resets
  to the top, **zero feed GETs** (observed `ones() == 0` across switches).
- Cap: `VIEW_LIMIT = 8`; rated/unavailable stories filtered, rest slide in.
- Explore: `view_order` shuffles client-side per `<window>:explore`, stable
  within a visit (`explore_orders`), reshuffled on leaving because
  `on_select_changed` pops `view_key` (existing
  `test_explore_order_is_stable_within_a_visit` covers it).
- Freshness already comes from the 1-minute `poll_feed_version` (regen, vote
  elsewhere) and manual `r`; summaries stay cached across sort changes.

## Results (10-story deck, v0 → v1 = reversed orders + new points)

| Scenario | A (fetch on return) | C (fetch on leave) |
|---|---|---|
| Round trip, deck bumped mid-away | 1× `1w` GET (+1m/1d neighbour refetch on version change); new order + new points shown; selection top; summary matches; `forced == 0` | 2× `1w` GETs (leave **and** return each fetch) + neighbours once; same final state |
| Held/slow feed | Old deck stays, selection/summary stable; release applies v1 cleanly | Same (same held fixture) |
| Rapid 4× s, same version | Exactly 1× `1w` (stale-event guard + exclusive refresh group collapse) | Exactly 1× `1w`, same collapse; no echo on re-settle |
| B (explicit ML rerank on leave) | — | Unsupported: client's only POSTs are `feedback`/`interaction`/`tldr-detail`; no endpoint starts ranking without a vote (static source check; vote reranks cost live compute, not measured) |

Navigation itself was summary-silent in the warm-cache fixture (zero tldr
traffic after switches; `forced == 0` everywhere). Prototype subclasses only
overrode `on_select_changed` to schedule the existing `refresh_feed`; they
never reset summaries or set `force_refresh`.

## Bugs/races noted (prototype scope)

- Naive C refetches on the return too — a round trip costs 2 GETs, not 1.
  Same-version rapid switches collapse to 1 GET via the exclusive
  `refresh` worker group (no storm, but a cancelled first fetch is possible).
- Late held answers apply cleanly for sorts (same window, `rebuild` preserves
  selection); no stale-window guard needed, unlike the window-prefetch path.
- Inferred, not exercised: a refetch carrying grown comment counts drops
  non-open cached summaries (`forget_outgrown_summaries`), and new story IDs
  enter the prefetch/generate window — auto-refetch on navigation can induce
  regeneration traffic where s/h/l today costs nothing.

## Recommendation

Keep current local sort switching; do not auto-refresh on s/h/l. Sorts are
views of one deck, Explore already reshuffles on revisit, and the minute
poller plus manual `r` cover freshness. If navigation freshness is still
wanted, A dominates C (1 vs 2 GETs per round trip, no speculative fetch on
pass-through, identical consistency); ideally gate A's return-fetch on the
poller already knowing a newer deck, so it never fetches when nothing
changed. B needs a new server operation plus 10–20 s ranking compute per
switch — not viable for navigation.
