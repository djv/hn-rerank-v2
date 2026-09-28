# Time window: review amendments (2026-09-28)

**Status: applied 2026-09-28 (see WORKLOG.md).**

From a Codex (gpt-6-astra) review of plans/time-window.md; each point was
checked against the code before being accepted. These override the plan
where they differ.

1. **Membership at request time.** Views are built at warm with a margin;
   when serving, drop stories that aged out of the window at request time
   (a cached 12h view must not show 13h-old stories between warms). One
   `as_of` per build for all five windows. 1m = 30 days; archive = time <
   now - 30d; windows nested; clamp age at 0.
2. **Archive supply** is only `bq_seed`/`ch_seed` (pipeline/__init__.py
   302-314); apply the window predicate after loading. Empty or short
   windows return fewer stories, never widen; clients show an empty state.
3. **Explore** 5 each = 15; per window, exclude that window's Recommended
   ids from the Explore pool (as today's primary exclusion).
4. **Cold users** (0 votes, `build_cold_deck`): all five windows too;
   Recommended/Popular by the cold gravity score; Explore empty.
5. **Dedup before the cap, with refill:** select ~2N per view, run dedup +
   `canonicalize_hn_dupes`, recheck a canonical replacement is still inside
   the window, then cap at N. Don't rely on `selected_limit=config.count`.
6. **Atomic publish:** all windows in one `DeckState` under one version;
   `_deck_for_user` stays authoritative (version 0 valid). Voted ids masked
   across every cached window at once; undo/rollback as today. Temporary
   shortfall below 12 during a warm is acceptable.
7. **Explicit orders are authoritative;** no membership derived from badge
   flags. The feed is per window, so its story objects may carry
   window-specific Explore badges; don't merge flags across windows.
8. **Feed schema version 2;** clients reject a mismatch with a clear
   "update the client" message. Validate `window`; echo it in the response.
9. **Interaction events:** keep the DB column `age_filter` (no migration);
   the API field becomes `window`, stored in that column.
10. **TLDR prefetch** keeps its budget/cooldown/single-flight/stale refresh,
    walks the 1w views; neighbour-feed prefetch never generates summaries.
    Remove obsolete config knobs and callers.
11. **Eval script** deck metrics use `combo_keys`
    (scripts/eval_ranker_variants.py ~1310-1322): replace with the 1w
    Recommended deck or drop them, keeping raw metrics and tests working.
12. Update `specs/client-ux.allium` and `specs/ranking-feedback.allium`.
    Drop the serving `enable_mmr` branch; keep `mmr_filter` if the eval
    uses it.
13. **Client caches** keyed by (window, version) with a request-generation
    guard: a late neighbour prefetch must not overwrite the selected window
    or restore pre-vote data.
14. **Extra tests:** window boundary / empty window, canonical replacement +
    refill, cold start, rapid votes/undo across windows, out-of-order
    prefetch.

Rejected: none. Noted as acceptable: brief shortfall below 12 cards while
a warm runs after several quick votes.
