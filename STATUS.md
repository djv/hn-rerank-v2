# HN Rerank status

- Objective: maintain the shipped terminal reader and prepare uvx/PyPI release.
- Owner: none; save-state refreshed 2026-09-17, no resource locks held.
- Result: terminal client, feed API, project rename, editorial styling, and VPS
  integration are implemented and pushed on `feat/terminal-client` through `e7cfc0f`.
  All visual polish (selection contrast, themed scrollbars, docked footer with
  per-filter counts and `✗` errors, age guard, headed empty/error notices,
  setup card, reading measure cap, status-line vote confirmation (toast removed))
  plus the reading-focus cue and restrained color accents are in
  the working tree, uncommitted; ROADMAP §6 polish is fully done.
- Verification: backend 541 passed / 1 skipped; client 32 passed / 1 Windows-only
  skip; Ruff and ty clean (also under ruff 0.16.7). Isolated rebuilt wheel startup
  passed from /tmp. Latest `tui.yml` CI run `34805118943` passed. Offline SVG
  renders of the populated, empty, error and setup states inspected 2026-09-17.
- Unresolved: native terminal visual verification and credential-dependent PyPI
  publication remain. Latest `tui.yml` run for the current branch passed.
- Next action: obtain PyPI publishing credentials, then publish and verify `uvx hn-rerank`.
  See [FINDINGS.md](FINDINGS.md).
