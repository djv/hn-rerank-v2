# Muse implementation assignment

The user requested: "plan the fixes and handoff to muse". Implement FIX-PLAN.md
in this isolated worktree. The supervising Codex agent owns final verification,
main integration and any later production deployment. Read this project's
AGENTS.md, STATUS.md, FIX-PLAN.md and VALIDATION.md first. The raw REVIEW.md is
secondary: VALIDATION.md qualifies its overstated documentation claim and
reproduces the real extraction issue. Do not launch another reviewer or agent.

First write MUSE-ACK.md alongside this file with model, cwd, branch, HEAD and
accepted scope, then proceed. This lets the supervising agent verify receipt.
Finish with CHECKS.md containing changed files, failure reproductions, exact
test/check commands and outcomes, known limitations, and integration next step.
Keep chat output concise; put details in those files.

Scope is the two confirmed source-content defects, including nondestructive
handling of legacy inputs/caches. Keep code small and source-specific. The
planned effective-content normalization must not accidentally preserve mixed
text via story_embedding_text's text_content fast path, Database.upsert_story's
longest-text merge, get_stories_with_labels' independently constructed Story
objects, or summary stale fallback's get_any_tldr_for_story. Candidate reads
currently use Database._row_to_story. Audit actual call sites and choose a
shared pure policy boundary rather than scattering different sanitizers.
Use dataclasses/replace and typed helpers; no new runtime dependencies.
Don't reinterpret top_comments or introduce discussion-count claims.

Known context at base 557c0ee:
- config.toml contains the requested category RSS; readable source label is
  Codex Forum. Retain source ID and canonical URLs/IDs.
- _NO_ARTICLE_SOURCES is in pipeline/enrichment.py; server.py's article_eligible
  gate currently checks AINEWS_SOURCE separately. Warm and proactive selectors
  share _article_fetch_eligible. All relevant gates must agree for this source.
- Raw RSS descriptions end with generated count/Read-full-topic markup.
  Stored self_text may have that footer flattened. Recognize terminal metadata
  conservatively and preserve authored lookalikes without the complete footer.
- Embedding reuse validates text hashes. Summary keys have multiple warm/tap/
  batch call sites; reject legacy Codex keys in error/cooldown fallback too.
- Prior baseline checks: backend 1,264 passed / 1 skipped; 26 focused; Ruff,
  ty and format passed. This does not substitute for validating new changes.

Workspace/resources:
- Only edit this task's worktree and output files. No main checkout changes,
  merge, service restart, production requests or deployment.
- No production DBs, shared .env, credentials, private-assistant-context,
  agent-memory, personal transcripts or provider configuration reads. Tests
  may create isolated temporary fixture databases. Do not run real generation,
  live model fits, seeding, migrations, cache cleanup or data deletion.
- Use /home/dev/.local/bin/uv. Create the worktree's own environment with
  uv sync --frozen if needed; never mutate production's .venv. VPS has about
  8 GB free disk and 5 GB available RAM at launch; do not download model caches.
- Cap OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS at one. Run
  focused tests first, then uv run pytest tests/ -n 4 -q, uv run ruff check .,
  uv run ruff format --check on changed Python, uv run ty check, git diff --check.
  Inspect load before and after the suite. No broad TUI sweep unless changes
  unexpectedly touch client behavior; explain if that happens.
- This is unattended: don't block on questions. If scope or a tool is blocked,
  stop the affected step and write the concrete blocker in CHECKS.md.

Update STATUS.md/FINDINGS.md/WORKLOG.md in the worktree with isolated validation
only, explicitly not deployed. After checks pass, commit only this task's
implementation/tests/docs to the feature branch; never amend, force-push,
merge main or push implementation without the supervising agent's verification.
On a failed gate, preserve the diff and report; commit nothing broken.
The supervising agent will inspect the finished report and code. No extra
independent result review is authorized by this assignment.
