# HN Rerank status

## Objective
TUI focus-visibility pass in tmux `work:3` (%11): every focused element reads
at a glance — pane frames, selected row, dropdowns — plus Firefox tab reuse
with focus pull. Preserve unrelated WIP; backend/VPS untouched.

## Verified result
- Focus frames: headlines list and reading pane show an orange frame only
  while focused (`test_focused_pane_shows_accent_border`).
- Selected row: amber-brown bold highlight (`#5A3A12`), tuned down from full
  orange after user review; dropdowns compact single-row with arrow-left
  `Dropdown` subclass, `Sort`/`Age` captions (narrow only), orange bg on
  focus instead of border.
- Narrow split 25/75 for the summary; brand label removed; `o`/`c` open in
  the running Firefox (`--new-tab`, launch if absent) and pull it to front
  via `wmctrl` (unit-tested, impressively untested live — user to confirm).
- Screenshot rig: pilot `export_screenshot` + chrome headless → PNG in
  `/tmp/tui_*.png`; verified narrow/wide, focused states.
- Live TUI restarted in place on every change (PID 3542401); user eyeballing.
- Checks: targeted TUI tests green, ruff/format/ty clean. Full suites last
  green at WIP commit `0d229f0` (backend 821, TUI 96+1).
- All UI work uncommitted (app.py, test_client.py, test_editorial.py,
  WORKLOG.md). No push/deploy since `0d229f0`.

## Blocker / limits
- One TUI test flaked once (failed then passed on rerun, same code) — watch
  for recurrence.
- Option rows render virtually, so the selected-row marker style is not
  unit-assertable; covered by screenshots + pane-border test only.
- Backend service/VPS unchanged since `0d229f0` deploy.

## Next step
User reviews live visuals; commit the UI batch on approval, then full-suite
gate + push (deploy only if server files get involved — currently TUI-only).
