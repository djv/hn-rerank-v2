# HN Rerank status

Saved 2026-10-07. Previous handoff:
[1m Popular age mix](docs/status-archive/before-tui-sharing-reddit-handoff-20261007.md).

## Objective
Share the TUI with friends and preserve Reddit weekly/monthly top-post
coverage after the announced RSS shutdown. User: "we need to handle this soon".

## Verified result
- TUI CI fix committed and pushed as `86bd1db`; backend CI and TUI CI
  (Linux/macOS/Windows) passed. No runtime behavior change.
- GitHub `uvx` installation tested in an isolated Bubblewrap environment:
  first-run setup, live stories and summaries, navigation, scrolling,
  sorting, votes, undo and quit. Dedicated test profile ended with zero votes.
- Reddit ingestion uses subreddit RSS and thread RSS. Incidental generic
  HTML fetches are possible, but no replacement listing importer exists.
- Official Reddit announcement schedules RSS retirement for November 13,
  2026 and offers no replacement for feeds outside moderated communities.
  Research evidence and install commands are in FINDINGS.md.

## Blocker / limits
- Reliable automated Reddit weekly/monthly ingestion after shutdown is
  unresolved. Browser import is a proposal; Devvit external-reader access
  and export suitability are unverified. JSON API is also being retired.
- Existing VPS deployment was `3475235` when checked in this session;
  the TUI lint fix did not require a backend deployment.
- Unrelated TUI test edit, mockups, kernel log and TLDR inspect script remain
  untouched. The inspect script has the existing `ty` diagnostic.

## Next step
- Priority: investigate and test a Reddit replacement soon, ahead of
  November 13. Check supported Devvit access/export constraints and test
  weekly/monthly listing extraction with a normal signed-in browser.
  Choose a route from actual results before implementing the source adapter.
- Preserve existing stories and feedback; do not alter the production DB
  during research. No new scraper or migration has been implemented.
- Carry forward the prior handoff's user check of 1m Popular age mix and
  real-terminal footer/tint appearance.
- Gemma 2 future-vote recheck, 2026-10-21 (also in /home/d/TASKS.md):
  encode user 151's new votes with `scripts/encode_replay_embeddings.py`
  (`--reuse` the 2026-10-06/07 files; Gemma 2 on the iGPU needs
  `--repo ~/.cache/hn-rerank-embedding-models/onnx-community--embeddinggemma-2-ONNX
  --onnx-file onnx/model.positions.onnx --prefix "title: none | text: "`), run
  `scripts/eval_ranker_variants.py --candidate-pool heldout-feedback
  --holdout-after 1791396238 --holdout-blocks 2 --dump-scores ...` for
  stored+Gemma 1 and stored+Gemma 2, then `scripts/compare_eval_scores.py`.
  Decide by `~/.local/state/hn-rerank-eval/gemma2-full-20261007/decision-rule.md`.
