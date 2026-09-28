# Time window instead of Date sort and Recent/Archive (2026-09-28)

**Status: implemented 2026-09-28 (with the amendments in
`plans/time-window-review.md`); not deployed. See WORKLOG.md.**

User request: a window dropdown (12h / 1d / 1w default / 1m / Archive) that
filters Recommended, Popular and Explore; the Date sort and the Age axis go
away. The server sends only the requested window; clients prefetch the likely
next window in the background. Aim: simpler server code.

## Today (see the code map in the session; key refs)

- Each warm scores every candidate (`_score_and_rank`), then
  `_assemble_combo_deck` (ranking.py:1618) builds a ~45-card deck per age from
  three combos (recent_hn 12, recent_nonhn 10, archive_hn 16) plus the
  Hot/Top/Talk cascade and Unsure/Novel/Similar passes. Only that deck is
  cached (`DeckState`, server.py:1436); the scores are dropped.
- `render.py` then caps Recommended at 24 per age, derives 8 `"{sort}:{age}"`
  orders, and Date = the deck newest first. Combo keys (`recent_hn`,
  `*_mixed`) thread through ranking, render, server prefetch lanes
  (server.py:1263-1317), both clients and interaction events.

## Proposed

1. **One selection function per window.** At warm time, for each window
   `w` in (12h, 1d, 1w, 1m, archive), from the scored pool:
   - Recommended[w]: top N by model score.
   - Popular[w]: top N HN stories by HN gravity (badge icon 🔥/🏆/💬 kept as
     a label from the story's own metric, not a separate cascade pass).
   - Explore[w]: Unsure, Novel, Similar, N/3 each, as today.
   N = 16 (12 shown + 4 backfill for votes).
2. **Cache the five window views** in `DeckState` (≤ 5 × 3 × 16 ids plus
   the stories), not the scored pool: small, and a request just selects.
3. **Feed per window:** `GET /api/feed?window=1w` returns that window's
   stories and `recommended|popular|explore` orders (~40 stories instead of
   ~90). The page embeds the default window. Versions/ready unchanged.
4. **Clients:** Window dropdown replaces Age and the Date sort; key `d`
   (web) cycles windows. After showing window `w`, fetch the neighbouring
   windows in the background and cache them per feed version; a vote marks
   all cached windows stale as today.
5. **Removed:** combo keys and `*_mixed`, `RECOMMENDED_LIMIT`, the Date
   orders, `is_recent`, the per-combo quotas, the Hot/Top/Talk cascade, the
   date lane in TLDR prefetch (prefetch walks the default window's views).
   `age_filter` in interaction events becomes `window`.

## Decisions (user, 2026-09-28)

- Recommended: pure model score, no source quota.
- Popular: single pick, top HN stories by gravity; no Hot/Top/Talk cascade.
- Windows 12h / 1d / 1w (default) / 1m / Archive. Send only the requested
  window; clients prefetch the likely next window in the background.
- Aim: simpler server code.

## Verification

Full suite + TUI suite + browser test; new tests for window membership
(every story in a window's views is inside the window), per-window feed
shape, client window switch and background prefetch; live smoke after
deploy. Offline benchmark unaffected (it scores `_score_and_rank` output).
