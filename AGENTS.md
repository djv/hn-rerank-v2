# hn-rewrite (Hacker News reranking)

Local-first reranking dashboard. Run Python only through `uv` (`uv sync`, `uv run python ...`, `uv run pytest tests/ -n 4`, `uv run ruff check .`, `uv run ty check`). Trust the live tree over stale docs or session logs.

Safety:
- Never delete or destructively modify `hn_rewrite.db` or any `*.db` (accumulated user feedback is the source of truth); no schema drops without asking; the pipeline's guarded `prune_*` operations are fine.
- Never lose uncommitted work: inventory with `git status --short` + `git diff --stat` before stash/checkout/restore/reset/clean; default to `git stash push -u`; use a worktree for parallel work.
- Do not standard-scale raw embeddings; be very skeptical of high metrics (>0.40 NDCG) — check for leakage.

Verification before reporting completion: full pytest at `-n 4`, clean ruff, zero new `ty` diagnostics; restart `hn_rewrite.service` before live smoke tests; update template assertions when templates change.

Keep `from __future__ import annotations`; type hints mandatory in core modules; `dict[str, Any]` only at parsing boundaries. Focused commits only. Details: WORKLOG.md, ARCHITECTURE.md, docs/BACKUP.md.
