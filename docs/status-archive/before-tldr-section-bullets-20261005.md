# HN Rerank status

Saved 2026-10-03. Previous handoff (upvote/downvote policy analysis):
[before-tui-footer](docs/status-archive/before-tui-footer-20261003.md).

## Objective
Make the terminal client's footer one row and tint it and the top bar
slightly (user request 2026-10-03).

## Verified result
- Footer is always one row, with no top rule. Keys shrink to fit beside the
  status before the status is cut. Filter bar and footer use the new
  `chrome` color. Details in FINDINGS.md.
- TUI suite 162 passed; backend 1,077 passed; ruff clean; ty has no new
  diagnostics. The TUI is a local client, so no service restart is needed.

## Blocker / limits
- Not yet viewed in a real terminal (headless tests only).
- Unrelated TUI test edit, mockups, TLDR inspect script (which has one ty
  error) and kernel log are preserved and uncommitted.

## Next step
- The user should check the footer and tint in their terminal and adjust the
  colors if they want them stronger.
- Popular 70/30 pilot and Explore quality test await a request; see
  the archived handoff and FINDINGS.md.
