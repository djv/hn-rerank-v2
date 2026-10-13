Verdict: PASS — no High issues found. Queue composition, round stability, retention, attribution and migration match the stated objective on code read.

Scope: `/tmp/hn-single-queue-20261013.diff` + `clients/tui/src/hn_rerank/{queue,app,api,dwell,models}.py`, `database.py`, `server.py`, `tests/test_queue_interactions.py`, `clients/tui/tests/test_queue.py` excerpts. Read-only; tests/lint not re-run.

Correct on read:
- `queue.py:48-67` Recommended→Discover/Uncertain/Interest/Other→5 additional Popular, dedupes, skips missing, preserves served order; `build_round:70-79` preserves prefix, no cross-round dup via `seen`.
- `app.py:1283-1292` extends only when last round fully excluded; churned cards keep frozen data, present cards refresh; `queue.py:82-87` unknown/archive never ages, old cards in current feed stay playable.
- `_dwell_ctx:1425` + `record_impression:1596` keep origin view/served rank/version, `surface`/`queue_position`; `server.py:2835-2843` + `database.py:533-556,1903-1928` additive, backward-compatible for omitted fields.

Medium:

1. `app.py:1425,1562-1594` — stale dwell on `None` ctx.
   Trigger: open views story drops from new deck → carried row inserted, `view_ranks.get`=None → `_dwell_ctx`=None → `schedule_summary` skips `tracker.start`, old segment stays open.
   Effect: false dwell credit to prior story, none to shown row.
   Check: on `ctx is None`, `stop()` current segment instead of leaving open.

2. `queue.py:38-39,48-60` — compressed `served_rank` when restored present.
   Trigger: `restored={2}`, order `[1,2,3]` → entry `3` gets rank 1 not true index 2.
   Effect: `position` telemetry off-by-N vs server order; analytics drift, no crash (server only checks range).
   Check: enumerate unfiltered order then skip, or document post-filter rank contract.

3. `app.py:1271,1291-1292` — `queue_stories[window]` never pruned.
   Trigger: long session, many rounds/churn.
   Effect: unbounded growth; each `rebuild` iterates all-ever-queued in `frozen`/`excluded`.
   Check: prune `stored` to ids in current `entries ∪ restored`, or cap windows.

4. `clients/tui/src/hn_rerank/api.py:444-447` vs old server — one-way compat.
   Trigger: new TUI (`surface` always sent) against pre-patch server (`required|{duration_ms,badges}` rejects unknown).
   Effect: 400 on `api/interaction`, swallowed best-effort → silent telemetry loss, no crash. New server←old client is fine (`server.py:2835` defaults `""`).
   Check: note min-server version or gate `surface` send on capability.

Low:

5. `queue.py:85` + `app.py:1280` — unknown-time (`<=0`) churned cards retained forever.
   Trigger: `time=0` story dropped by server, `in_window`=True → never excluded.
   Effect: stale cards accumulate; Caught-up delayed. Per spec but unbounded.
   Check: expire unknown after N rounds or session cap.

6. `app.py:2120-2125` — `patch_counts` misses `self.restored`.
   Trigger: summary reply updates counts, then undo restores pre-patch object via `restore_story`.
   Effect: stale points/comments until next refresh.
   Check: also patch `restored[story_id]` and `history` entry.

7. `queue.py:19-22` — naive `now` flips day.
   Trigger: `utc_clock()` returns naive datetime → `astimezone(UTC)` interprets as local.
   Effect: wrong startup sort near midnight. Production passes aware/`None`; tests aware.
   Check: `if now.tzinfo is None: assume UTC` or require aware.

8. `app.py:1601-1612` — delayed impression re-looks up global `queue_origins`.
   Trigger: same story id in two windows, switch window within 1s sleep.
   Effect: safe drop via `feed.window` check, but version compared against wrong window's origin; no false credit.
   Check: capture `(version, surface)` at schedule time instead of re-lookup, or key origins by `(window,id)`.

