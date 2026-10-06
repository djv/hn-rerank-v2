# HN Rerank status

Saved 2026-10-05. Previous handoff (TLDR section bullets):
[before-popular-1m-360](docs/status-archive/before-popular-1m-360-20261005.md).

## Objective
Show more older stories in Popular when the window is 1 month (user report
2026-10-05: the oldest card looked like 7 days).

## Verified result
- The served 1m Popular list was already correct (3 of 16 cards were 21–23d
  old, the first at #8). The user had voted on all of the top 49 stories by
  1m gravity, and at 240h this week's unvoted stories outranked older ones.
- `GRAVITY_TIME_SCALE["1m"]` changed from 240h to 360h in the server, web
  client and TUI. Deployed `3475235` (rollback tag
  `deploy-pre-popular-1m-360`). User 151's live 1m Popular now has 6 of the
  first 12 cards 21–27d old. Dashboard 200, no errors since restart.
  Evidence is in FINDINGS.md.

## Blocker / limits
- Clients show only 12 of the 16 Popular cards the server sends
  (`VIEW_LIMIT`), so ranks 13–16 are not shown.
- Unrelated TUI test edit, mockups, TLDR inspect script and kernel log are
  preserved and uncommitted. The inspect script has the one `ty` error.

## Next step
- The user checks whether the 1m Popular age mix feels right. Try 480h if
  more old stories are wanted, or 300h if fewer.
- The user still needs to check the TUI footer and tint from 2026-10-03 in a
  real terminal.
