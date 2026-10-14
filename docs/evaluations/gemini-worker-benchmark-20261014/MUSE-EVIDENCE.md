# Evidence: queue restored-rank, return, round-freeze, and telemetry behavior

Scope: read-only research of the existing client in this directory.
No telemetry or ranking logic was edited; only `src/hn_rerank/queue.py`
was simplified (direct lanes), plus this file, `NOTE.md`, and
`tests/test_worker_simplify.py`. All conclusions below cite the
current source `file:line`. Statements marked **Confirmed** have a
direct code citation; statements marked **Uncertain** are not provable
from this client code alone.

## (1) How locally restored IDs stop counting in served ranks

**Confirmed:** restored IDs are removed *before* `enumerate`, so the
`rank` passed to `QueueEntry` is the index in the filtered order, not
the index in the server order.

- `src/hn_rerank/queue.py:36` — `restored_ids: set[int] = restored or set()`
  normalizes the caller-supplied restored set.
- `src/hn_rerank/queue.py:38-39` — `served_order(view)` returns
  `[sid for sid in feed.orders.get(view, []) if sid not in restored_ids]`.
- `src/hn_rerank/queue.py:52` / `:56` / `:66` — each lane does
  `for rank, sid in enumerate(served_order(...))`, so `rank` already
  excludes restored IDs.
- `src/hn_rerank/queue.py:45-50` — `append_to` stores that filtered
  `rank` unchanged in `QueueEntry(sid, view, rank, feed.version)`.
- `src/hn_rerank/app.py:1469` — the only queue caller passes the live
  local set: `build_round(rounds, self.feed, excluded, set(self.restored))`.
- `src/hn_rerank/app.py:2265-2267` — comment states the invariant
  explicitly: “Only missing undo rows are local insertions; excluding a
  returned row as restored would compress subsequent Queue ranks.”

Concrete semantics: `recommended=[1,2,3]` with `restored={2}` yields
`served_order=[1,3]`, hence `QueueEntry(1,…,0)` and
`QueueEntry(3,…,1)` — `3` compresses from server index 2 to served rank
1. Popular behaves the same except its `rank` still counts
missing/duplicate positions that `append_to` later skips
(`src/hn_rerank/queue.py:65-70` only increments `popular` on success).

**Uncertain:** why the server omits undone stories from some decks
(e.g. whether omission is by version, readiness, or timing) is not
visible in this client; the client only assumes restored IDs must not
occupy served positions.

## (2) Why returned rows must stop being local restorations even while `ready=false`

**Confirmed:** `refresh_feed` prunes any restored ID that the new feed
already serves *before* the `ready` branch, so a returned card is
treated as server-served with its server rank even when the next deck
is still ranking.

- `src/hn_rerank/app.py:2268-2271` —
  `present = {story.id for story in feed.stories}` then
  `self.restored = {sid: story … if sid not in present}`.
- `src/hn_rerank/app.py:2272-2275` — only the *remaining* (still
  missing) restored rows are re-inserted:
  `if not feed.ready: for restored in self.restored.values(): self.restore_story(restored)`.
- `src/hn_rerank/app.py:2265-2267` — rationale comment: “A returned
  card is server-served even while the next deck is still ranking.”
- `src/hn_rerank/queue.py:38-39` — mechanism of harm: if a returned ID
  stayed in `restored`, `served_order` would filter it and every later
  card’s `served_rank` would compress by one.
- `src/hn_rerank/app.py:2741-2747` (`apply_undo`) vs
  `src/hn_rerank/app.py:2813-2828` (`restore_story` skips `"queue"`):
  undo creates a local insertion via `self.restored[id]=story` plus a
  server-order insertion; Queue itself is derived and never a server
  order, so the only correct Queue membership for a returned card is
  the next `build_round` from the feed, not the local set.

In short: prune-first preserves server ranks; re-insert-only-missing
preserves availability of still-absent undos while `ready=false`.

**Uncertain:** the server-side meaning of `ready`/`target_version`
beyond “reranked deck not yet landed” is not defined in this client;
we confirm only the client’s ordering (prune, then conditional
re-insert, then `rebuild` at `src/hn_rerank/app.py:2279`).

## (3) How old rounds preserve their origin/version through refreshes

**Confirmed:** rounds are append-only; each `QueueEntry` freezes
`origin`/`served_rank`/`served_version` at creation and is never
rewritten.

- `src/hn_rerank/queue.py:12-17` — `QueueEntry` is
  `@dataclass(frozen=True)` with `story_id/origin/served_rank/served_version`.
- `src/hn_rerank/queue.py:49` — version is frozen as `feed.version` at
  lane insertion.
- `src/hn_rerank/queue.py:85-94` — `build_round` collects
  `seen = {entry.story_id …}` from all prior batches, builds
  `batch = queue_entries(feed, excluded | seen, restored)`, and returns
  `[*rounds, batch]` (or `list(rounds)` when empty): earlier batches
  are never mutated or reordered.
- `src/hn_rerank/app.py:1467-1470` — `rebuild` only calls `build_round`
  `if not rounds or all(entry.story_id in excluded for entry in rounds[-1])`,
  i.e. refreshes never extend an unfinished round.
- `src/hn_rerank/app.py:1471-1476` — `entries = [entry for batch in rounds …]`,
  `queue_snapshots[window] = entries`, and
  `queue_origins = {entry.story_id: entry for entry in entries}` are a
  concatenation of frozen rounds.
- `src/hn_rerank/app.py:1454-1455` and `:1472-1474` — card *data* refresh
  is separate from slots: `stored.update(… if sid in lookup)` then
  `stored[entry.story_id] = lookup[entry.story_id]` when present, so
  present cards update counts while churned cards retain frozen data;
  `src/hn_rerank/app.py:1477-1480` keeps churned cards playable in
  Queue unless voted/unavailable/aged-out.

**Uncertain:** how long the server keeps a churned card out of later
decks (permanent vs transient churn) cannot be confirmed from the
client; the client only guarantees it does not shift old slots when a
refill arrives.

## (4) Where Queue telemetry chooses served rank/version instead of list position/current version

**Confirmed:** `Reader._dwell_ctx` substitutes the stored served
dimensions for Queue, and keeps list position only as `queue_position`.

- `src/hn_rerank/app.py:1617` — `position = self.view_ranks.get(story.id)`
  is the list position in the concatenated-rounds order
  (`src/hn_rerank/app.py:1484-1485` builds `view_ranks` from
  `view_order`, and `src/hn_rerank/app.py:1395-1399` builds Queue
  `view_order` from `queue_snapshots`).
- `src/hn_rerank/app.py:1622` — `origin = self.queue_origins.get(story.id)
  if sort == "queue" else None`.
- `src/hn_rerank/app.py:1623-1625` — Queue rows without an origin are
  dropped (`return None`).
- `src/hn_rerank/app.py:1626-1632` — the substitution:
  `position=origin.served_rank if origin else position`,
  `dashboard_version=origin.served_version if origin else self.feed.version`,
  `sort_mode=origin.origin if origin else sort`,
  `surface="queue" if origin else "views"`,
  `queue_position=position if origin else None`.
- `src/hn_rerank/app.py:1637-1658` — `_interaction_event` copies those
  fields into `InteractionEvent` unchanged.
- `src/hn_rerank/app.py:1785-1804` — `record_impression` re-validates
  against `origin.served_version` (not the current feed version) and
  drops the event on mismatch, preserving trial credit by
  (version, window, story).
- `src/hn_rerank/api.py:430-447` — wire payload sends
  `dashboard_version/position/sort_mode/surface` plus conditional
  `queue_position`.
- `src/hn_rerank/dwell.py:11-22` — `DwellContext` carries the same
  `position/dashboard_version/sort_mode/surface/queue_position` split.

Views (`surface="views"`) use the opposite branch: current
`feed.version` and list `position`, with no `queue_position`.

**Uncertain:** server-side interpretation of these fields (e.g. how
reports must split Explore `position` by `surface`) is documented in
`README.md` but has no additional client code to cite; the client-side
choice above is the complete confirmed mechanism.
