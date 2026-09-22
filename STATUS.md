# HN Rerank status

- Objective: maintain the shipped terminal reader and prepare uvx/PyPI release.
- Owner: none; save-state refreshed 2026-09-22, no resource locks held.
- Result: the terminal reader line is merged with `origin/main` (merge commit
  `354acd6`, plus `6a0dc99` for the workspace lock). Backend code takes
  `origin/main` (server.py, pipeline/render.py, tests, uv.lock); the client
  package keeps the branch side (`clients/tui/**` with the newer feed models).
  The 2026-09-20 review fixes are in (`d480344`, `5996c9e`, `f32f039`,
  `a47b8de`): selector focus no longer strands every keybinding, Escape out of
  help restores the story summary, the reading heading follows refreshed data,
  headings align left, the footer labels the vote keys
  (`1 up · 2 neutral · 3 down`), Enter opens read mode only when the summary
  overflows its pane (otherwise the key and hint stay disabled), and Markdown
  blocks render without per-block margins. Server `detail-v10` targets
  45/70/90 words, caps each section at four bullets and one `####` heading,
  and bolds single-marker emphasis so Article + Discussion fit one TUI screen.
  The 2026-09-22 boundary hardening is in: deployment URLs must parse with
  httpx (ports, unprintable and non-IDNA hosts) and `Feed.parse` raises only
  `ValueError` for invalid payloads, covered by Hypothesis properties and
  server-side parity assertions in `tests/test_feed_api.py`. The reader also
  prefetches the next stories' summaries one at a time and caches them for the
  session (`--prefetch N`, default 2, 0 disables), pausing on rate limits and
  never caching stale/partial responses. Server `detail-v11` scales TLDR
  budgets for long multi-section pieces (up to 5-8 bullets/200 words at 20k+
  chars) and instructs full-piece coverage after Import AI 473 lost its tail;
  `LLM_PROVIDER` is gospark (muse-spark-1.3-contributor).
- Verification: backend 772 passed
  (full-suite run with `-n 2` on a loaded host, `HN_ONNX_MODEL_DIR` set); client 62 passed /
  1 Windows-only skip; Ruff, format, ty and `uv lock --check` clean. Offline SVG renders inspected;
  native terminal sessions at 145×38 and 80×30 exercised filters, reading,
  voting/undo, help, empty and failure states; the Textual pilot measured 8/8
  live summaries inside the 27-line reading pane.
- Unresolved: credential-dependent PyPI publication remains.

## Known issues / follow-ups (2026-09-20 merge)

- Summary fit was verified on 8 sampled stories at 145×38. A long title shrinks
  the reading viewport (heading `max-height: 8`); very long titles could still
  push a summary to a second screen.
- The TUI branch's WORKLOG vacuum (digests + `WORKLOG-archive/` tarball, commit
  `3f3bb28`) is superseded by the merged long-form WORKLOG. The archive file
  still exists; re-run the vacuum or drop the refactor deliberately.
- The TUI branch's AGENTS.md slimming is likewise superseded by `origin/main`'s
  long guide. The slim version is preserved on `backup/tui-pre-merge`.
- `docs/TUI_VPS_PATCH.patch` is the historical scoped-port patch, superseded by
  the merge into `origin/main`; keep for history or delete deliberately.

- Next action: obtain PyPI publishing credentials, publish and verify
  `uvx hn-rerank`. See [FINDINGS.md](FINDINGS.md).
