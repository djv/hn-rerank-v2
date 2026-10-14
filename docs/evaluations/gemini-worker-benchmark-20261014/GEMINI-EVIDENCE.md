# Client Queue & Telemetry Research Evidence

## 1. How Locally Restored IDs Stop Counting in Served Ranks

### Confirmed Conclusions
1. **Local restoration mutates cached feed orders in place:**
   - When a user undoes a vote, `Reader.apply_undo` records the story in `self.restored[story.id] = story` and calls `self.restore_story(story)` (`src/hn_rerank/app.py:2741-2747`).
   - While a refreshed feed has `not feed.ready`, `Reader.refresh_feed` also re-applies `self.restore_story(restored)` for any undone stories still in `self.restored` (`src/hn_rerank/app.py:2272-2275`).
   - `Reader.restore_story` appends the story to `feed.stories` if absent (`src/hn_rerank/app.py:2821-2822`) and inserts `story.id` directly into each relevant non-`"queue"` view's `feed.orders[key]` list (`src/hn_rerank/app.py:2826-2857`, inserting at `min(saved.served_rank, len(order))` on `src/hn_rerank/app.py:2843` or by gravity/rank score on `src/hn_rerank/app.py:2857`).
2. **`rebuild` passes `set(self.restored)` into round construction:**
   - When `Reader.rebuild` starts a new Queue round (`if not rounds or all(entry.story_id in excluded for entry in rounds[-1]):`), it invokes `build_round(rounds, self.feed, excluded, set(self.restored))` (`src/hn_rerank/app.py:1468-1469`).
   - `build_round` forwards `restored` to `queue_entries(feed, excluded | seen, restored)` (`src/hn_rerank/queue.py:77-86`).
3. **`served_order` strips `restored_ids` before `enumerate` assigns `served_rank`:**
   - In `queue_entries`, `restored_ids = restored or set()` (`src/hn_rerank/queue.py:41`) is used by `served_order(view: View) -> list[int]` (`src/hn_rerank/queue.py:43-44`):
     ```python
     def served_order(view: View) -> list[int]:
         return [sid for sid in feed.orders.get(view, []) if sid not in restored_ids]
     ```
   - Each lane (`"recommended"` at `src/hn_rerank/queue.py:53`, `"explore"` at `src/hn_rerank/queue.py:57`, and `"popular"` at `src/hn_rerank/queue.py:66`) calls `enumerate(served_order(view))` and stores `rank` in `QueueEntry(sid, view, rank, feed.version)` (`src/hn_rerank/queue.py:50`).
   - Because `served_order` filters out `restored_ids` *before* `enumerate` runs (unlike `excluded`, which is checked inside `append` via `sid in seen` at `src/hn_rerank/queue.py:40, 47` *after* `enumerate`), locally restored IDs do not occupy a slot during enumeration and never inflate the `served_rank` of subsequent server-served stories.
   - Regression coverage in `tests/test_queue.py:658-662` (`test_restored_rows_do_not_count_in_served_ranks`) confirms that for `orders={"recommended": [1, 2, 3]}` with `restored={2}`, story `3` receives `served_rank=1` rather than `2`.

---

## 2. Why Returned Rows Must Stop Being Local Restorations Even While `ready=false`

### Confirmed Conclusions
1. **Unconditional pruning of `self.restored` on feed receipt:**
   - In `Reader.refresh_feed`, immediately after `self.set_feed(feed)` (`src/hn_rerank/app.py:2263`) and *before* checking `if not feed.ready:` (`src/hn_rerank/app.py:2272`), the client removes any story ID present in the newly fetched `feed.stories` from `self.restored` (`src/hn_rerank/app.py:2265-2271`):
     ```python
     present = {story.id for story in feed.stories}
     self.restored = {
         sid: story for sid, story in self.restored.items() if sid not in present
     }
     ```
2. **Why this must happen even when `feed.ready` is `False`:**
   - If the server includes an undone story in `feed.stories` (and `feed.orders`), that row is genuinely server-served in the returned deck snapshot (`feed.version`), even while a newer target deck version is still ranking in the background (`feed.ready == False`).
   - Only undone rows that are *missing* from `feed.stories` (`sid not in present`) are local client-side insertions that require `self.restore_story(restored)` while `not feed.ready` (`src/hn_rerank/app.py:2272-2275`).
   - If a story returned by the server remained in `self.restored` while `feed.ready == False`, `rebuild()` (`src/hn_rerank/app.py:2279`) would pass its ID in `set(self.restored)` to `build_round` (`src/hn_rerank/app.py:1469`), and `served_order` (`src/hn_rerank/queue.py:43-44`) would strip it from `feed.orders[view]`. That would both omit the server-returned card from new Queue rounds and compress (under-count by 1) the `served_rank` (`QueueEntry.served_rank` / telemetry `position`) of every story ordered after it in that view (`src/hn_rerank/app.py:2265-2267`).
   - This exact scenario is verified for both `ready=False` and `ready=True` in `tests/test_queue.py:665-707` (`test_refresh_returned_undo_card_keeps_server_ranks`): after story `2` is undone into `app.restored` (`tests/test_queue.py:686`) and then returned by a refresh in `orders={"recommended": [2, 3]}` at `version=7` with `ready=False` (`tests/test_queue.py:689-697`), `2 not in app.restored` (`tests/test_queue.py:706`), the new round is `[QueueEntry(2, "recommended", 0, 7), QueueEntry(3, "recommended", 1, 7)]` (`tests/test_queue.py:699-702`), and story `3` keeps server rank `ctx.position == 1` (`tests/test_queue.py:705`).

---

## 3. How Old Rounds Preserve Their Origin and Version Through Refreshes

### Confirmed Conclusions
1. **Immutable `QueueEntry` records:**
   - Each card placed into a round is represented by a frozen `QueueEntry(story_id, origin, served_rank, served_version)` (`src/hn_rerank/queue.py:12-17`), created with the `view` (`origin`), `rank` (`served_rank`), and `feed.version` (`served_version`) of the `Feed` at the time that round was built (`src/hn_rerank/queue.py:50`).
2. **Append-only round batches per window:**
   - `Reader` stores rounds per window in `self.queue_rounds: dict[str, list[list[QueueEntry]]]` (`src/hn_rerank/app.py:998`).
   - During `Reader.rebuild` (`src/hn_rerank/app.py:1467-1470`), `build_round` is called *only* when `not rounds or all(entry.story_id in excluded for entry in rounds[-1])`. As long as the latest round has at least one unexcluded card, a feed refresh does not modify `self.queue_rounds[window]` at all.
   - When the latest round is exhausted and `build_round(rounds, self.feed, excluded, set(self.restored))` runs (`src/hn_rerank/app.py:1469`), `build_round` (`src/hn_rerank/queue.py:77-86`) computes `seen = {entry.story_id for batch in rounds for entry in batch}`, calls `queue_entries(feed, excluded | seen, restored)` so no story from an earlier round can be re-queued with a newer origin/rank/version, and returns `[*rounds, batch] if batch else list(rounds)`. Earlier batches in `rounds` remain untouched.
3. **Origin map and frozen story cache across refreshes:**
   - On every `rebuild`, `entries = [entry for batch in rounds for entry in batch]` flattens the preserved batches into `self.queue_snapshots[window] = entries` and rebuilds `self.queue_origins = {entry.story_id: entry for entry in entries}` (`src/hn_rerank/app.py:1471-1476`). Because story IDs are unique across rounds (`src/hn_rerank/queue.py:84-85`), `self.queue_origins[story_id]` always resolves to the `QueueEntry` from the round that originally served that story.
   - Story objects for queued cards are cached per window in `self.queue_stories` (`src/hn_rerank/app.py:999, 1454-1456, 1472-1480`): if a story is still present in the refreshed feed, its `FeedStory` metadata (e.g., points/comments) is updated in `stored` (`src/hn_rerank/app.py:1455`), while if it churned out of the new feed, `stored` retains its frozen `FeedStory` so the card stays readable in Queue (`tests/test_queue.py:409-535`).

---

## 4. Where Queue Telemetry Chooses Served Rank/Version Instead of List Position/Current Version

### Confirmed Conclusions
1. **Dimension selection in `Reader._dwell_ctx` (`src/hn_rerank/app.py:1609-1635`):**
   - `self.stories.index(story)` (`src/hn_rerank/app.py:1613-1616`) only checks that `story` is currently in the visible `self.stories` slice (`[:VIEW_LIMIT]`); the visible list index itself is discarded.
   - `position = self.view_ranks.get(story.id)` (`src/hn_rerank/app.py:1617`) looks up the story's 0-based index in the full unrated view order `self.view_order(sort)` (`src/hn_rerank/app.py:1393-1399, 1484-1485`).
   - When `sort == "queue"`, `origin = self.queue_origins.get(story.id)` is retrieved (`src/hn_rerank/app.py:1621-1625`), and `DwellContext` (`src/hn_rerank/dwell.py:11-22`) is populated at `src/hn_rerank/app.py:1626-1635`:
     - **`position`:** `position=origin.served_rank if origin else position` (`src/hn_rerank/app.py:1628`) chooses `origin.served_rank` (the card's served rank in its origin view when its round was built) instead of the Queue slot `position` (`self.view_ranks[story.id]`) or the visible list index in `self.stories`.
     - **`queue_position`:** `queue_position=position if origin else None` (`src/hn_rerank/app.py:1632`) records the full-queue slot `self.view_ranks[story.id]` separately.
     - **`dashboard_version`:** `dashboard_version=origin.served_version if origin else self.feed.version` (`src/hn_rerank/app.py:1629`) chooses `origin.served_version` (the deck version when the card's round was built) instead of the current `self.feed.version`.
     - **`sort_mode` and `surface`:** `sort_mode=origin.origin if origin else sort` (`src/hn_rerank/app.py:1630`) and `surface="queue" if origin else "views"` (`src/hn_rerank/app.py:1631`) report the card's origin view (`"recommended"`, `"explore"`, or `"popular"`) on the `"queue"` surface.
2. **Propagation to all interaction events (`src/hn_rerank/app.py:1637-1658`):**
   - `Reader._interaction_event` copies `ctx.dashboard_version`, `ctx.position`, `ctx.sort_mode`, `ctx.surface`, and `ctx.queue_position` into `InteractionEvent` (`src/hn_rerank/app.py:1644-1658`; serialized by `API.interaction` at `src/hn_rerank/api.py:390-409`).
   - This applies to impressions (`schedule_summary` at `src/hn_rerank/app.py:1762-1769`), dwell segments (`schedule_summary` at `src/hn_rerank/app.py:1764`, `on_app_focus` at `src/hn_rerank/app.py:1742-1744`, `_dwell_event` at `src/hn_rerank/app.py:1671-1674`), and article/comment opens (`action_open_url` at `src/hn_rerank/app.py:2877-2882`).
3. **Impression guard after the 1 s delay in `Reader.record_impression` (`src/hn_rerank/app.py:1780-1814`):**
   - After `await asyncio.sleep(1.0)` (`src/hn_rerank/app.py:1783`), `record_impression` resolves `origin = self.queue_origins.get(event.story_id) if event.surface == "queue" else None` (`src/hn_rerank/app.py:1785-1787`) and computes:
     ```python
     shown_version = (
         origin.served_version
         if origin
         else self.feed.version
         if self.feed
         else None
     )
     ```
     (`src/hn_rerank/app.py:1788-1794`), then checks `shown_version != event.dashboard_version` (`src/hn_rerank/app.py:1804`) and `str(self.query_one("#sort", Select).value) != ("queue" if event.surface == "queue" else event.sort_mode)` (`src/hn_rerank/app.py:1805-1806`). Choosing `origin.served_version` instead of `self.feed.version` ensures a background feed refresh that advances `self.feed.version` during the 1 s dwell threshold does not invalidate a pending Queue impression from an earlier round.

---

## Uncertainty and Boundaries

- **Confirmed in local source:** All client-side queue round composition, restoration rank filtering, refresh state transitions, and telemetry payload construction are directly confirmed in `src/hn_rerank/queue.py`, `src/hn_rerank/app.py`, `src/hn_rerank/dwell.py`, `src/hn_rerank/api.py`, and `tests/test_queue.py` at the lines cited above.
- **External / Unverified (Server-Side):** This repository contains only the terminal client (`hn-rerank`). While client comments (`src/hn_rerank/dwell.py:8`, `src/hn_rerank/app.py:2265-2267, 2272-2273`) describe the server's `server.py` interaction event parser and the server's behavior when returning `ready=false` feeds while a new ranking version is being computed, the server source code itself is not present in this repository and its internal implementation cannot be inspected directly.
