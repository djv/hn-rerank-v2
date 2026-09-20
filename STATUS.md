# HN Rerank status

- Objective: maintain the shipped terminal reader and prepare uvx/PyPI release.
- Owner: none; save-state refreshed 2026-09-20, no resource locks held.
- Result: terminal client, feed API, project rename, editorial styling, and VPS
  integration are implemented and pushed on `feat/terminal-client` through
  `e7cfc0f`. The editorial polish and the 2026-09-20 review fixes are committed
  on local `main` (`d480344`, `5996c9e`, `f32f039`): selector focus no longer
  strands every keybinding, Escape out of help restores the story summary, the
  reading heading follows refreshed data, headings align left, the footer
  labels the vote keys (`1 up · 2 neutral · 3 down`), Enter toggles back to
  headlines, and Markdown blocks render without per-block margins so Article +
  Discussion fit one screen. Companion server change on `origin/main`: TLDR
  `detail-v9` targets 45/70/90 words and caps each section at four bullets and
  one `####` heading.
- Verification: client 38 passed / 1 Windows-only skip; backend 541 passed /
  1 skipped; Ruff and ty clean (also under ruff 0.16.7). Isolated rebuilt wheel
  startup passed from /tmp. Offline SVG renders of the populated, empty, error
  and setup states inspected; native terminal sessions at 145×38 and 80×30
  exercised filters, reading, voting/undo, help, empty and failure states, and
  the Textual pilot measured 8/8 live summaries inside the reading pane.
  Latest `tui.yml` CI run passed for the previous tip; the new commits are
  unpushed.
- Unresolved: credential-dependent PyPI publication remains.
- Next action: open a PR for `feat/tui-color-polish` (local `main` diverged from
  `origin/main`, merge has conflicts — resolving is deferred), then obtain PyPI
  publishing credentials, publish and verify `uvx hn-rerank`. See
  [FINDINGS.md](FINDINGS.md).
