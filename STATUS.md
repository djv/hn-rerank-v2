# HN Rerank status

- Objective: eval controls and training deduplication; no ranking deployment.
  User authorized committing ranking/eval, deployed TLDR fix and TUI logging.
  Full-body embedding work remains deferred, uncommitted WIP. TUI logging is
  included but not live-verified; running reader was not restarted.
- Ranking experiment: read-only live audit found 16 jack-clark.net upvotes,
  13 distinct URLs (three duplicate issues). Added disabled publication
  affinity SVM metadata and eval variant; no production ranking change.
  791 tests pass, Ruff/ty clean. First 3-fold temporal eval completed in an
  isolated VPS checkout (one low-priority CPU thread, cached embeddings).
  Raw NDCG@40: production 0.03135 vs affinity 0; MAP 0.02133 vs 0.01321.
  Recent deck had zero held-out judgments, so no conclusion for that view.
  Five known duplicate-URL train/test overlaps remain; only 85/20/2 eligible
  positives per fold. Keep disabled.
- Improved eval: normalized-URL train/test isolation, deduplicated candidates,
  explicit judged-only historical replay, @10 metrics and coverage warnings.
  Replay retains 657/661/662 rated items: production/affinity NDCG@10
  0.80537/0.80493, MAP 0.46820/0.46231. No gain; flag remains off.
  Affinity shuffled-label control elevated (NDCG@10 0.43605 vs production
  0.27660); next investigate multiple permutations/null baselines. Do not
  compare judged-only levels to current-pool scores or claim leakage-free.
  793 tests passed; lint/format/types clean after a test annotation correction.
- Latest result: isolated 3-fold replay with five shuffled-label seeds done.
  Shuffled NDCG@10 means: production 0.3352, affinity 0.3545, dedup 0.3211;
  expected chance 0.3402. Earlier one-seed elevation is not persistent across
  seeds; this is not proof of zero leakage. Dedup real NDCG@10 0.8502 vs
  0.8054 production, but MAP slightly lower and gain concentrated in one fold.
  Keep flags off; next scoped step is a predeclared confirmation comparison
  of production vs dedup, not further feature work or tuning.
  Checks: backend 797 passed; client 63 passed/1 skipped; Ruff/format/ty clean.
- Owner: none; save-state refreshed 2026-09-22, no eval job left running.
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
- Current fix (deployed): detail-v12 shares prompt/output bullet
  limits so long articles and discussions retain six/eight bullets instead
  of being silently cut to four. Generation regressions pass; full suite
  776 passed with two workers, Ruff/format/ty clean.
- Live verification: Import AI 473 regenerated via Muse Spark in ~9 seconds;
  seven bullets cover all six major sections of the stored article, including
  uncensored models, RSI and machine hermeneutics. Repeat request returned an
  identical cached summary. Dashboard 200; bounded journal scan clean.
- Deployment: server.py copied with matching SHA-256, service restarted;
  VPS still has an uncommitted server.py patch; local fix included in commit scope.
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
