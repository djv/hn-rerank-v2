# Independent implementation review target

Review the saved section-refresh implementation against baseline 4e946e5.
Exclude the separate AGENTS-only commit 7706816 from the behavior review. Desired:
- Genuine s/h/l/tab/selector sort changes navigate locally and background-fetch
  the existing shared window feed. No explicit rerank. Coalesce in-flight work,
  preserve usable reader/selection/scroll and optimistic vote state. Reject late
  profile/window/teardown replies; handle deck/count versions honestly.
- r regenerates only the selected summary and background-checks its real stats.
  Keep prior text while waiting; preserve other summaries, feed order and hidden
  rows. Repeated r and positive popup answers coalesce paid work.
- Stats-only authenticated API: supported HN Firebase IDs only, bounded fetch,
  strict identity/type/count checks, per-field live/stored honesty. No provider,
  hydration, ranking, new profile or feedback changes from this endpoint.
- Known source-descendant baseline stored/read atomically with cache text.
  Legacy/provisional rows unknown; generated/cache/API metadata truthful and
  backward compatible. Snapshot describes source discussion size, not every
  comment being in the LLM input.
- Known growth updates counts and offers a story-specific Regenerate/Keep
  choice. Safe Keep/Escape default; no automatic spending or background vote
  keys. Decline suppresses same observation; renewed growth may offer again.
  Revalidate current profile/story; do not prompt redundantly during a force.
- Forced HN refresh obtains verified source input even at zero stored comments;
  upstream failure cannot become a fresh cache success or pointless LLM call.
  One ordinary generation plus one bounded shared fresh follow-up when needed,
  including late-waking waiters; no stale result overwrites, duplicate spending,
  process-long completed-flight history or indefinite retry loops.

Runtime files: server.py, database.py, single_flight.py, pipeline/__init__.py,
pipeline/enrichment.py, clients/tui/src/hn_rerank/api.py and app.py.
Regression files: tests/test_server.py, test_single_flight.py,
test_tldr_single_flight.py, test_pipeline.py; clients/tui/tests/test_sort_refresh.py
and updated client/editorial/prefetch/setup/windows tests. Behavior docs changed
in clients/tui/README.md, WORKLOG.md, STATUS.md and FINDINGS.md.

Final isolated matching-source VPS gates: backend 1,238 passed/1 skipped (browser
module lacks Playwright), TUI 195 passed/1 skipped (Windows DACL); existing MLP
convergence warning. Ruff/ty/touched-format/diff whitespace clean. Source hash
manifest and detailed test bounds: IMPLEMENTATION-CHECKS.txt, SOURCE-SHA256.txt.

Not deployed or physically checked. No live DB/provider/ranking/profile/trial
changes authorized. Ranking research remains paused. Review read-only: no edits,
tests/sweeps, provider calls, another reviewer, commit/push or live access.

Independence: assess current code/diff before any prior reviewer findings.
Do not read PLAN.md, CODEX-PLAN-REVIEW.md, FINDINGS.md, general/deployment review
reports, old evaluation reports or agent logs. Read AGENTS.md and the first 30
lines of STATUS.md for the current objective/gate state only. The standalone
criteria above provide the objective without previous findings.
