# HN Rerank status

- Objective: maintain the shipped terminal reader and prepare uvx/PyPI release.
- Owner: none; save-state refreshed 2026-09-20, no resource locks held.
- Result: the terminal reader line is merged with `origin/main` (merge commit
  `354acd6`, plus `6a0dc99` for the workspace lock). Backend code takes
  `origin/main` (server.py, pipeline/render.py, tests, uv.lock); the client
  package keeps the branch side (`clients/tui/**` with the newer feed models).
  The 2026-09-20 review fixes are in (`d480344`, `5996c9e`, `f32f039`,
  `a47b8de`): selector focus no longer strands every keybinding, Escape out of
  help restores the story summary, the reading heading follows refreshed data,
  headings align left, the footer labels the vote keys
  (`1 up · 2 neutral · 3 down`), Enter toggles back to headlines, and Markdown
  blocks render without per-block margins. Server `detail-v9` targets 45/70/90
  words and caps each section at four bullets and one `####` heading so
  Article + Discussion fit one TUI screen.
- Verification: backend 769 passed with
  `HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model`; client 38 passed /
  1 Windows-only skip; Ruff and ty clean. Offline SVG renders inspected;
  native terminal sessions at 145×38 and 80×30 exercised filters, reading,
  voting/undo, help, empty and failure states; the Textual pilot measured 8/8
  live summaries inside the 27-line reading pane.
- Unresolved: credential-dependent PyPI publication remains.

## Known issues / follow-ups (2026-09-20 merge)

- `config.toml` pins the VPS model directory
  (`/home/dev/hn-rewrite/shared/mxbai-embed-xsmall-v1`). Laptop test runs must
  set `HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model`. Docs still name
  the stale `HN_TEST_ONNX_MODEL_DIR` (`docs/TUI_RELEASE.md`, `AGENTS.md`).
- `origin/main`'s WORKLOG.md carried a committed `>>>>>>> theirs` conflict
  marker; this merge removed it. Worth a scan for other stray markers.
- The TUI branch's WORKLOG vacuum (digests + `WORKLOG-archive/` tarball, commit
  `3f3bb28`) is superseded by the merged long-form WORKLOG. The archive file
  still exists; re-run the vacuum or drop the refactor deliberately.
- The TUI branch's AGENTS.md slimming is likewise superseded by `origin/main`'s
  long guide. The slim version is preserved on `backup/tui-pre-merge`.
- `origin/feat/tui-color-polish` is now stale; delete it after this push.
- VPS deployment: pull the merge and restart `hn_rewrite.service` so the newer
  `clients/tui/src/hn_rerank/models.py` is imported; follow the backup/verify
  procedure in `docs/TUI_RELEASE.md`. Its "do not replace the laptop tree" note
  is now obsolete and should be updated.
- Summary fit was verified on 8 sampled stories at 145×38. A long title shrinks
  the reading viewport (heading `max-height: 8`); very long titles could still
  push a summary to a second screen.
- Untracked tool artifacts `.opencode/` and `.playwright-mcp/` sit in the repo
  root; consider `.gitignore` entries.

- Next action: push the merge to `origin/main`, delete the stale
  `feat/tui-color-polish` branch, then obtain PyPI publishing credentials,
  publish and verify `uvx hn-rerank`. See [FINDINGS.md](FINDINGS.md).
