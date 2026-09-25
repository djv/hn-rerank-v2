# HN Rerank status

## Objective
Finish the TUI navigation improvements and get the standalone client CI green.

## Verified result
- Committed and pushed `71de3b4` (`Improve TUI zoom and navigation prefetch`)
  to `origin/main`, together with earlier local commits `c8651e5`, `23ebcb5`
  and `24784fd`.
- TUI: centered TLDR zoom; Enter/Escape returns to headlines; voting legend
  explicitly says next story; footer keeps errors and shortcuts visible.
- Prefetch: 20 forward cache targets, previous three, and other-sort entry
  points; generate missing nearby TLDRs with four background requests maximum.
  Selecting in-flight work reuses it. `--prefetch-generate 0` is cache-only.
- Client relaunched in `work:3.1`; 44 stories rendered. VPS logs confirmed
  cache hits and successful generation. Server/VPS code was unchanged.
- Local tests: TUI 121 passed/1 skipped, backend 821 passed with local ONNX
  model override; local Ruff, formatting and ty passed.
- Remote backend CI passed for `71de3b4`.
- Remote TUI tests passed: Windows 122; Linux/macOS 121 plus 1 skipped each.

## Blocker / limits
- Terminal client CI failed on eight Ruff errors on all three platforms,
  after tests passed. Do not report the push as fully green.
- Its workflow copies the standalone package outside the backend workspace;
  local checks did not reproduce this exact environment. Errors include
  unsorted imports and SIM102 in `tests/test_client.py`.
- Generation uses provider capacity and can still be outrun by rapid navigation.

## Next step
Resolved 2026-09-25: the terminal client lint failures came from the newer
ruff/ty that the lockfile-less standalone copy installs. Fixed, together
with a dead `is_mounted` prefetch guard; see WORKLOG.md. Confirm that
`.github/workflows/tui.yml` passes on the push.
