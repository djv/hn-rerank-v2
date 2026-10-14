# Review: Queue opening mix (485c76e vs dd92d18)

Independent read-only review of `CHANGES.diff` against the supplied sources.
Paths below are repository paths (`clients/tui/...`).

## Verdict

The composition change is correct. I found **no correctness or regression bugs**
in `queue_entries`, round freezing, undo or telemetry. There is **one confirmed
user-facing issue**: the in-app help text is stale. Below that are a few small
test gaps and a nit.

## Confirmed findings

### 1. Low/Medium: in-app Queue explanations still describe the old sequential order
- `clients/tui/src/hn_rerank/app.py:2850` (the `reasons["queue"]` summary text):
  "Recommended, then Discover, Unsure and Interest from Explore, then up to five
  additional Popular cards."
- `clients/tui/src/hn_rerank/app.py:2935` (help/keys text): "each window's first
  shown Recommended, then Explore by badge, then up to five additional Popular
  cards".
- **Trigger:** select the Queue sort with no story open, or open help.
- **Consequence:** the reader says Queue is sequential when it now interleaves
  R → E → R → P. That contradicts the README and ARCHITECTURE text this commit
  updated, and it's the one description the user actually sees while reading.
- **Fix:** reword both strings to match the README, e.g. "mixes Recommended →
  Explore (by badge) → Recommended → Popular, repeating and skipping exhausted
  sections; up to five additional Popular cards". No test asserts these strings
  (I grepped `tests/`), so nothing else needs to change.

## Verified as correct (no action)

- **Selection unchanged:** the mix is a pure reordering pass after the old
  selection loop (`queue.py:68-77`). Membership, the `seen`/excluded/restored
  handling, duplicate ownership (R > E > P), badge grouping and the five-Popular
  cap all run before the mix and are untouched.
- **Per-lane order and metadata:** each lane is a `deque` filtered from
  `entries` in original order. `popleft` keeps per-section order, including
  Explore's badge-group order. `QueueEntry` objects are moved, not rebuilt, so
  `served_rank` and `served_version` are kept.
- **Termination and sparse lanes:** every pass of the `while any(...)` loop pops
  at least one entry, so it terminates. An empty lane is skipped. With no
  Recommended cards you get E, P, E, P, ... and then the leftovers of the longer
  lane. With all lanes empty the result is `[]`, so `build_round` still adds no
  round.
- **Round freezing, refresh, refill:** `app.py:1283-1292` still freezes the
  composed list per round, and `build_round` still excludes every earlier
  round's ids. The new order only affects a newly built round. Rounds already
  open in a running session keep the old order until the reader is relaunched,
  as documented.
- **Undo slot restoration:** `restore_story` (`app.py:~2650`) puts a card back
  into its *origin order* at `served_rank`, which doesn't depend on queue
  position. The queue slot comes from the frozen snapshot. Both still hold.
- **Telemetry:** `_dwell_ctx` (`app.py:1438-1448`) reports `sort_mode=origin`,
  `position=served_rank`, `dashboard_version=served_version`, and
  `queue_position` = index in the mixed queue. The updated test expectation
  (`("popular", 2, 2)`) is what the new interleave should produce.
- **Window switches:** composition is per window (`queue_rounds[window]`), and
  nothing in the change touches cross-window state.

## Test observations (limitations, not bugs)

1. **The property test lost its positional check.** `test_queue_invariants` used
   to assert the Recommended *prefix*. It now only checks the Recommended
   subsequence, so the only check on the R/E/R/P pattern is the one hand-written
   test. Consider adding a property that the origin sequence equals a reference
   round-robin over the three lanes, and that the Explore and Popular
   subsequences match the pre-mix order. That would catch, for example, a
   wrong pattern tuple or a lane that is built out of order.
2. **No test with an empty Recommended lane.** That is the realistic case for
   new profiles or a still-warming deck. A small case such as
   `recommended=[]`, Explore `[a, b, c]`, Popular `[x]` → `[a, x, b, c]` would
   pin down the skipping behaviour.
3. **The restored path is not checked alongside the mix.** `restored` ids are
   removed from served orders before selection, and existing app tests cover
   undo. The `test_queue_keeps_missing_cards...` expectations changed (`[3, 2, 4]`
   etc.), which looks consistent but is only checked end to end. Optional.
4. As stated in the prompt, physical rendering in the reader is unverified. My
   review didn't run code either; it's from source alone.

## Nit

- `clients/tui/src/hn_rerank/queue.py:5-6`: `from dataclasses import dataclass`
  comes before `from collections import deque`, which is not isort order. Ruff
  passed, so the `I` rules are probably not selected. Swap them for
  consistency anyway.
