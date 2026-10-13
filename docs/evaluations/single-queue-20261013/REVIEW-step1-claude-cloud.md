# Review: single-queue Step 1 + TUI telemetry fixes (583d0bd, 0e8686e)

## Verdict: **revise** (TUI queue). The server and DB part can ship as is.

The pure queue composition, the server validator and the additive migration
are correct and well tested. The freeze/append design has one assumption that
does not hold against the real server. FIX1 treats "not in `feed.stories`" as
"aged out", but `serve_window` re-caps every view after each rerank. Frozen
Explore (and some Recommended) cards can therefore vanish from the queue
before the user reaches them. That is the starvation FIX1 was meant to
remove. Two telemetry and long-session issues also touch the live trial.
These fixes are small and local.

---

## High

### H1. Frozen entries disappear when the server stops *serving* them, not only when they age out
`app.py:1235-1253` (rebuild `lookup` / `if sid in lookup`), `queue.py:63-72`.

The queue keeps only `QueueEntry(story_id, origin, served_rank)`. A frozen
entry is shown only if its story is still in the *current*
`feed.stories` (or in `self.restored`). Per `ranking_serve_window_excerpt.py:61-75`,
each serve does two things:
- It rebuilds Explore as the top `EXPLORE_PER_BADGE` per badge from the
  freshly reranked scores.
- It caps Recommended and Popular to `VIEW_SIZE`.

**Scenario:** the queue is frozen with 12 Rec cards and 9 Explore cards
(3 Discover, 3 Unsure, 3 Interest). The user votes through the 12 Rec cards.
That is 12 reranks. The uncertainty and discover scores move, so the served
Explore set churns. If the feed JSON contains only served-view stories (the
TUI fakes all assume this), most of the frozen Explore entries are no longer
in `feed.stories`. They are silently skipped as "unavailable", and the user
jumps from Rec to the leftover Explore and Popular, then to the appended Rec.
Explore is still starved. It is now invisible too: no event records that the
slot was skipped. A Rec card at served rank 10–11 can be pushed out by a
better-scoring refill in the same way. Popular churns less, through gravity
over hours.

The new pilot test (`test_frozen_queue_reaches_explore_popular_then_appended_recommended`)
never exercises this. Its `RefillingServer` keeps every non-voted Explore and
Popular story in `stories`.

**Fix:** freeze the card data along with the slot. Store each
entry's `FeedStory` per window (for example, `queue_stories[window][sid]`, updated
from fresh feeds so the counts stay current). Use it as the lookup fallback, and skip an
entry only when it is voted or unavailable, or when its `story.time` is truly outside the window
(`in_window` / a client-side age check). First confirm what `/api/feed`'s
`stories` contains. If it includes the whole pool, this downgrades to Medium.
Add a pilot case whose refill drops a not-yet-shown Explore story from
`stories`. That story must still be reached.

### H2. Queue Recommended events pair a *frozen* served rank with the *current* `dashboard_version`
`app.py:1388-1396` (`_dwell_ctx`: `position=origin.served_rank`,
`dashboard_version=self.feed.version`).

The server reranks after every vote. By the time the user reaches queue slot 7,
the feed version is usually several versions past the deck that assigned
`served_rank` 7. The constraints say the A/B trial reads impressions by
`sort_mode`/`position`. If trial credit resolves `(dashboard_version,
position)` against the served deck, for example to find the draft arm or the
far card at that rank, queue Recommended impressions are credited to whatever
sits at that rank in a *different* deck. That is the arm-credit corruption the
spec set out to avoid.

**Fix:** add `served_version: int` to `QueueEntry`, set from `feed.version` in
`queue_entries` and `extend_queue`, and send it as `dashboard_version` for
queue-surface events. Or confirm in the trial-credit code that it keys only on
story id plus `sort_mode`, and note that in ARCHITECTURE 4.4. Also add an
assertion to the refill pilot test.

## Medium

### M1. Explore/Popular never refresh within a session (long-lived TUI)
`app.py:1238-1244`, `queue.py:63`. FIX1 says the snapshot is per window
"per session". The TUI polls and reloads on new versions from regens, not only
votes (`poll_feed_version`). It can stay open for days. After the first deck,
only new Rec cards are ever appended. New Explore and Popular cards (the jobs
the plan says the queue must keep) never enter the Queue until the user
reconnects. In the 12h window, the frozen Explore and Popular entries age out
within hours. Queue days then turn into Recommended days, which skews the
live comparison's "share of votes outside Recommended" guardrail.

**Fix (choose one):**
- Start a new snapshot epoch when the deck changes for a reason other than
  this client's vote (a version bump with no `_pending_votes`/ack, or more
  than N hours since the freeze).
- Or, once every frozen non-Rec entry is voted or unavailable, append the
  next Explore group and 5 Popular from the current feed.

### M2. "Caught up" is effectively unreachable in real use
`queue.py:63-72`. Every vote causes a one-card Recommended refill
(`serve_window` tops up to `VIEW_SIZE`), and each refill is appended. The tail
regrows on every vote until the window's pool runs out, so in 1w/1m/archive
the "done" state never appears. The tests reach it only because their fakes
stop refilling. This follows from the user's "Rec. appended" decision, but
PLAN's step 4 ("a 'done' state") and the guardrail metrics assume a finite
queue. Either cap appends, for example to at most `VIEW_SIZE` appended Rec per
snapshot epoch, or record the infinite tail explicitly in PLAN/README so the
live test is read with that in mind.

### M3. Startup sort is fixed for the life of the process; the day-alternation analysis can misassign days
`app.py:800`. `startup_sort` is computed once in `__init__`. A session started
at 23:50 UTC on a Queue day runs Queue through the next (Recommended) day, and
reconnecting does not recompute it. That is fine for UX, but the PLAN's
"by day type" bootstrap must classify by the session's assigned arm (or by
`surface`), not by the event's UTC date. Write that down in PLAN or README
now. Optionally, log the assigned default once per session, for example as
a `client_session_id`-scoped field, so that intent-to-treat analysis works even
when the user switches sorts at once.

## Low

- **L1. Persistent `restored` entries outlive their window.**
  `app.py:2016-2020` now keeps undone stories absent from a ready deck
  indefinitely. `rebuild` merges them into `lookup` for every window and sort
  (`app.py:1236`). An undone story that later truly ages out keeps showing in
  that window's queue until it is voted or the client reconnects. Drop
  `restored` entries whose story is outside the window, or cap them by age.
- **L2. Undo from a *views* sort now uses the queue's frozen rank.**
  `apply_vote` (`app.py:2474-2478`) adds the queue origin and "queue" to
  `vote_views` even when the vote came from Recommended, Popular or Explore.
  `restore_story` (`app.py:2576-2587`) then reinserts into the origin view at
  a stale `served_rank` instead of by score or gravity. If that origin view no
  longer serves the story (for example, it was displaced from Explore), undo
  puts it back into that view. Only add the queue metadata when the vote was
  cast in the Queue sort.
- **L3. Appended `served_rank` can be off by one on a not-ready deck.** On the
  not-ready path, `restore_story` inserts undone stories into
  `feed.orders["recommended"]` before `rebuild` → `extend_queue`, so
  `served_rank` for later appended cards counts a client-inserted row. Compute
  appended ranks from a copy of the server order, or skip `self.restored` ids
  when computing ranks.
- **L4. Dead branch / stale comment.** `app.py:1384-1386` ("carried open
  row has left the new queue") cannot trigger: carrying is disabled for
  queue at `app.py:1259`. Keep it as a guard, but fix the comment. Relatedly,
  in queue sort a refresh that drops the *open* story from `feed.stories` now
  jumps the reader to another card mid-read. H1's fix (frozen card data)
  also fixes this.
- **L5. Test comment mismatch.** `test_queue.py:354` says "Two replacement
  Rec cards", but the code refills one (`[:1]`). The pilot also asserts
  `position == 0` for every card, so a wrong `served_rank` on an appended card
  would still pass. Give the refill deck a non-zero rank for the new card
  (for example, a refill that lands at rank 1 behind an existing card).
- **L6. Mixed `position` meanings for `sort_mode="explore"`.** Queue events
  send the server's Explore index, and Explore-view events send the shuffled
  index. This is documented. Reports must split by `surface` and treat
  `surface=''` (old rows) as views. Add that line to whatever
  report or query reads Explore positions, not only the README.
- **L7. Positional SQL elsewhere.** The two test fixes
  (`tests/test_import_ai_gate.py`) show that positional
  `INSERT INTO interaction_events VALUES (...)` breaks with 17 columns.
  Grep `scripts/`, the backup/restore code and notebooks for `INSERT INTO
  interaction_events VALUES` and `SELECT * FROM interaction_events` with
  tuple unpacking before deploying. Also check for any table-rebuild
  migration that copies explicit column lists, which would silently drop the
  new columns.

## Earlier TUI fixes

- **583d0bd (position = full-order index):** correct for Recommended and
  Popular. Two nits:
  - In the Explore view, `view_ranks` is the client-shuffled index, not a
    served rank, so the commit message overclaims (now documented).
  - A carried open row in a views sort has no `view_ranks` entry, so
    `_dwell_ctx` falls back to `listed` = 0
    (`app.py:1381`). A focus-resume dwell (`on_app_focus`) on a carried row
    logs `position=0`. Return `None` for ids not in `view_ranks`, as
    queue already does.

  Positions also re-base when the reranked deck lands. That is correct (it is
  that version's served rank), but it is not "a vote never shifts positions".
- **0e8686e (idle cap 90 s):** correct; the constant and tests are consistent.
  Dwell data before and after 2026-10-13 11:30 is not comparable. Note the
  cutover alongside the position-meaning cutover in PLAN "Effects on existing
  work".

## Things that are fine

- `queue_entries` composition, dedup, missing-id skip, ≤5 Popular, no shuffle
  and origin and served-rank bookkeeping are correct. The Hypothesis
  invariants are good, and `feed.orders` is not mutated.
- `extend_queue`: exact-prefix, no-duplicate and Rec-only append, covered
  by a property test. It never re-appends a frozen id that comes back.
- No KeyError path: `rebuild` filters `sid in lookup`, and the restored story
  is merged into the lookup. No voted story is shown, since `rated` is
  checked against the queue order.
- `view_order("queue")` no longer has side effects. Snapshots are kept
  across window switches and cleared on reconnect, which is tested.
- Telemetry:
  - Queue events carry the origin `sort_mode`, the served rank and a stable
    `queue_position`.
  - Views events send `surface="views"` with no `queue_position`.
  - `record_impression`'s sort check correctly maps the queue surface.
  - A sort switch on the same story ends the old dwell with the old context
    and starts a new impression (`summary_view_key`), which is tested.
- Server validator: it rejects `""`, other strings, non-strings, bool or
  float or string `queue_position`, out-of-range values, and `queue_position`
  without `surface="queue"`. A missing `surface` maps to `""` for old
  clients. The 2**63 bound has a property test.
- Migration: additive only. It uses `ADD COLUMN ... NOT NULL DEFAULT ''` and a
  nullable `INTEGER` with a CHECK that existing NULL rows satisfy. This is
  valid on a STRICT table, idempotent, and tested on an old-schema DB with a
  pre-existing row preserved. Nothing drops or rewrites rows.
- Startup alternation, injectable clock and UTC conversion are tested. The
  conftest pins the day for older tests.
