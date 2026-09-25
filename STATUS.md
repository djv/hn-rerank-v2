# HN Rerank status

## Objective
Ship the TUI sort/discovery improvements and relieve TLDR quota pressure
on the VPS (quotas 120/hr → 240/hr, deployed).

## Verified result
- Committed and pushed to `origin/main`: `5b82157` (TUI explore shuffles
  client-side per rebuild), `0dc1832` (`v` reverses any sort with focus-top
  plus footer `reversed` marker, sectioned `?` help), `df579d0` (prior WIP
  status save), `247f0c1` (uncached-TLDR quotas 120/hr → 240/hr per-user
  and global), `4f2feca` (deploy note). Tree clean.
- TUI: explore is a fresh random deck each visit; `v` flips rank order,
  focuses the new first item, shows `reversed` in the footer counts line;
  `?` help is sectioned (Move/Read/Vote/Sort/Other). Local TUI suite 123
  passed / 1 skipped; `ruff check`, `ruff format --check`, `ty check` clean
  via `uv run --frozen` in `clients/tui`.
- Backend: no `.py` changes for the quota bump. Full suite 803 passed;
  18 `test_pipeline.py` errors are pre-existing and environmental
  (identical with the change stashed); `ruff check` clean.
- VPS deploy: `main` worktree fast-forwarded clean to `247f0c1` (explore
  worktree untouched), host `config.toml` confirms 240/240,
  `hn_rewrite.service` restarted 22:04:31 UTC, dashboard 200, zero
  `quota_denied` after restart (old process denied under the 120 limits
  right up to the restart). Remaining journal noise is transient RSS
  DNS/fetch failures.

## Blocker / limits
- Remote CI for the new pushes (`0dc1832`, `247f0c1`, `4f2feca`) not yet
  checked — backend workflow and Terminal client workflow (tests + lint
  on 3 platforms) need a green confirmation.
- Provider-side caps (Mistral spend cap / Groq tier) remain the real
  ceiling: ours quotas no longer bind at 240/hr, so any further denials
  are provider cooldowns, and generation spend can now run ~2x hotter.
- 18 pre-existing `test_pipeline` errors (environmental) still open.
- The TUI itself needs a relaunch to pick up the sort changes; the VPS
  restart only affects the server.

## Next step
Check the GitHub Actions runs for the new pushes; if green, relaunch the
TUI and watch VPS `quota_denied`/spend under the 240/hr quotas.
