# OpenCode Instructions for `hn-rewrite`

## Project shape

- This repository is a minimalist, local-first Hacker News reranking dashboard rewrite.
- Use `uv run python <script>.py` or standard `uv` commands to execute scripts and run tests.
## Working rules

- **Trust the live tree, not stale docs or session logs.** When reviewing
  a plan or answering a "check" question, verify claims against the current
  checkout. Documentation and session history describe what was true
  at write time; the source tree is the ground truth.
- Make minimal, behavior-preserving changes unless the user asks for a broader refactor.
- Keep the runtime path local-first; do not add new external dependencies unless needed.
- **Be very skeptical of unusually high metrics** (e.g. NDCG > 0.40). We are unlikely to beat the Hacker News baseline by a large margin; high metrics often indicate feature leakage, train-test contamination, or metric saturation artifacts.
- **Do NOT standard-scale raw embeddings** (the production 384-d embeddings are L2-normalized; the current configured encoder is mxbai-embed-xsmall-v1). StandardScaler must only touch metadata columns from `emb_dim:` onward.
- **Never delete or destructively modify the local database** (`hn_rewrite.db`, `hn.db`, or any `*.db` file in the working tree). The DB holds the user's accumulated feedback and is the single source of truth for personalization. No `rm`, no `DELETE FROM` without a `WHERE` clause that excludes all rows, no schema migrations that drop tables or columns with data. `database.py`'s `prune_stories` has an explicit retention rule and an `id NOT IN (SELECT story_id FROM feedback)` guard, so it would be fine to run — but it is currently **dormant**: nothing in the live pipeline calls it (only tests do). Wiring it up would start deleting story rows from an 860MB+ DB; treat that as a decision requiring explicit user sign-off, not routine maintenance. When in doubt, ask before running any command that touches the DB file.
  - **Exception (2026-06-22):** 756 test/empty stories (time=0) were deleted with explicit user permission. This included 2 test stories (id=999 "Test", id=99999998 "Test regen live") that received 2 upvotes from user 1. Backup retained at `hn_rewrite.db.pre_test_removal_20260622T163344Z`.
- Keep test execution times optimized (target under 12 seconds total at `-n 4`). Run the full suite with `uv run pytest tests/ -n 4` (4 cores; `pytest-xdist` is in `dev`). Single-process takes ~32s; `-n 4` brings it to ~13s on this host (still a hair over the 12s target — no single dominant test, the slowest is a ~3s Hypothesis property test in `test_server.py`). Per-test ONNX model loads are avoided entirely by `MockEmbedder(Embedder)` in `tests/test_server.py:21` (overrides `__init__` to skip the `AutoTokenizer.from_pretrained` + `ort.InferenceSession` path) and `DummyEmbedder(Embedder)` in the two seed test files — they share a module-scoped `mock_embedder` fixture in `test_server.py`. Do not regress this: any new "mock" embedder that subclasses `pipeline.Embedder` MUST override `__init__` or it will silently reload ONNX per test.
- **Hypothesis profiles**: `tests/conftest.py` registers opt-in `dev` (50
  examples) and `ci` (300 examples, `deadline=None`, `print_blob=True`).
  Select with `HYPOTHESIS_PROFILE=ci uv run pytest tests/`; ordinary pytest
  runs retain Hypothesis' default settings. A test's own
  `@settings(...)` still overrides the profile's `max_examples`/`deadline`
  for that test specifically. When adding a `@given` property test, prefer
  a strategy that actually reaches the branch under test (avoid
  `st.sampled_from` over a handful of literals — that's `parametrize` in
  disguise — and avoid feeding raw `st.text()`/`st.floats()` where the
  interesting inputs are structured, e.g. HTML fragments or clustered
  embeddings) and draw any randomness used inside the test body (e.g.
  `np.random.default_rng(seed)` from a drawn `seed`) rather than calling
  global `np.random.*`, or hypothesis can't shrink or replay a failure.
- **Never silently lose uncommitted work.** The working tree can hold
  modifications from a prior session (Codex, codex, opencode, human
  hand-edits). Treat any pre-existing uncommitted change as load-bearing
  until proven otherwise. Three operational rules:

  1. **Inventory before any `git stash`, `git checkout`, `git restore`,
     `git reset`, or `git clean` operation.** Run `git status --short`
     and `git diff --stat` first, write down (or paste into the chat)
     the full set of modified and untracked files. If the operation
     targets a subset, confirm the subset is exhaustive — i.e. that no
     file outside the subset holds uncommitted changes you need.

  2. **Never `git checkout HEAD -- <file>` on a file that has uncommitted
     changes in the working tree unless the file is explicitly listed
     in a fresh stash and you've just verified it with `git diff <file>`
     showing no diff.** A bare `git checkout HEAD -- file` discards the
     file's working-tree state with no warning, no backup, no recovery
     short of `git fsck --unreachable`.

  3. **Default to `git stash push -u` with no file list** when the goal
     is "save the entire WIP before doing something different." If you
     need a partial stash, use `git stash push -- <file1> <file2> ...`
     *only after* step 1 confirms those are the only files with WIP.
     For surgical hunk selection, prefer `git stash -p` so every hunk is
     visible. Never combine a partial stash with a `git checkout HEAD`
     on the un-stashed files in the same logical operation.

  - **Use a worktree for parallel work on an existing WIP.** When you
    need to make a separate change on top of an uncommitted WIP, prefer
    `git worktree add ../hn-rewrite-<branch> -b <branch>` to a second
    working tree. The WIP stays untouched in the original tree, and
    you can `git stash`/commit at your own pace in the new tree without
    risking a partial-loss operation in the original.

  - **Recovery hint**: even after a bad `git checkout`, the pre-checkout
    blob may still exist in the object store as an unreachable object.
    `git fsck --unreachable --no-reflogs` lists candidates. `git show
    <sha>:<path>` recovers the file. This is best-effort — `git gc`
    prunes unreachable objects after the default 30-day window, and a
    subsequent `git add` may have overwritten the index entry.

  - **Codex/codex session logs are recovery hints, not source of truth.**
    If a prior codex or Codex session exists for a WIP, it may help
    reconstruct lost code, but: (a) the diffs are against an older
    snapshot of the tree, (b) line numbers and surrounding code may have
    shifted, (c) the session may show *attempted* patches that were
    never applied cleanly. Mirror structurally, not verbatim, and verify
    with tests. Record the recovery (and the loss) in WORKLOG.md so the
    git history is self-documenting — the next person reading the blame
    should see what happened and what was reconstructed.

- **Focused commits.** When asked to commit, stage only the intended
  files. Verify with `git status --short` and `git diff --stat` first.
  Leave unrelated workspace state (including `.opencode/`) alone.
  Do not amend or force-push unless explicitly asked.
- **Always update relevant documentation** (e.g., [ARCHITECTURE.md](ARCHITECTURE.md), [WORKLOG.md](WORKLOG.md)) after making code or behavior changes.

## Running scripts

- **Always run Python via `uv run python <script>.py`.** Never invoke `python` or `python3` directly — the project uses `uv` to manage the venv and dependencies. Direct invocations will use the system Python, missing project dependencies.
- For one-liners, use `uv run python -c "..."`.
- For ad-hoc tools, prefer writing a script in `scripts/` (tracked, testable) over `-c` one-liners (ephemeral, untracked).
- Interactive REPL: `uv run python` (drops you into the project venv).
- Do not bypass the venv. If you need a new package, add it to `pyproject.toml` and re-run `uv sync`.

## Type discipline

- **Strongly typed systems are the default.** Use explicit domain models
  (dataclasses, `NewType`, `Literal`, enums) for any value that crosses a
  module boundary, gets stored, or represents a meaningful domain concept.
- **No `dict[str, Any]` for core data flow.** Story rows, score maps,
  CH response rows, etc. should be `Story`, `dict[int, float]`,
  `list[ChStoryItem]`, or a typed dataclass. `Any` and untyped dicts
  are acceptable for:
  - Parsing JSON from external APIs (the boundary), as long as the
    parsed result is immediately normalized into a typed model.
  - Feature dicts in ML pipelines where keys are dynamic (but the
    *value type* must still be explicit, e.g. `dict[str, np.ndarray]`).
- **Type hints are mandatory for new code** in the `pipeline/` package,
  `ch_client.py`, `server.py`, `database.py`. Existing code without
  hints gets hints when you touch it.
- **Tests get hints too** — public test functions should have
  parameter and return type annotations, even if the body is short.
- **LSP errors are not optional.** If your editor (or `uv run ruff check`)
  flags a type mismatch, fix it. Don't `# type: ignore` to silence it
  unless the alternative is genuinely worse and the reason is documented
  inline.
- **Validation at the boundary, not deep in the code.** Functions that
  accept external data (CH responses, HTTP requests) should validate
  types/shapes and raise a clear error. Don't let `None` propagate
  through 5 layers.
- **The `from __future__ import annotations` directive is already at the
  top of every module** — keep it. New modules should add it too. It
  makes all annotations lazy strings, which avoids forward-reference
  issues and is required for Python 3.9+ compatibility (we target 3.12).

## Common commands

- Install or refresh the environment: `uv sync`
- Bootstrap the embedding model on a fresh checkout: `uv run python setup_model.py`
  (downloads the production ONNX model into `DEFAULT_ONNX_MODEL_DIR`,
  `pipeline/config.py`; no-ops if the files already exist)
- Run tests: `uv run pytest tests/`
- Run linting: `uv run ruff check .`
- Run type checking: `uv run ty check` (Astral's `ty`; pre-existing
  diagnostics are tracked in the baseline; new code must introduce
  zero new diagnostics. LSP errors must be fixed or `# type: ignore`
  with a documented reason)

## Verification protocol

Before reporting completion of any code or template change:

1. `uv run pytest tests/ -n 4` — full suite must pass
2. `uv run ruff check .` — lint must be clean
3. `uv run ty check` — zero new type diagnostics
4. If the change affects runtime behavior: restart the service, then
   live smoke test with endpoint requests plus a bounded
   `journalctl --user -u hn_rewrite.service --since '1 min ago'` scan
5. If UI templates changed: update corresponding assertions in
   `tests/test_server.py`

## Dependency groups

`pyproject.toml` ships three groups beyond the runtime deps. Default
`uv sync` installs only `dev` (linters, pytest, type checker). `dl-experiment`
and `embedding-experiment` are both opt-in.

- `dev` — pytest, pytest-asyncio, hypothesis, pytest-xdist, ruff, ty. Always
  installed by `uv sync`.
- `dl-experiment` — `torch>=2.12`. Pulls in the ~700MB torch +
  triton + nvidia-cu* wheels. **Required only by** `pipeline_dl.py`,
  `pipeline_dl_t0.py`, `tests/test_pipeline_dl.py`, and
  `scripts/eval_ranker_variants.py` — the unshipped attention-MLP
  ranker experiment (loses to SVM on every metric; see WORKLOG
  2026-06-25).
  - Install on demand: `uv sync --group dl-experiment`
  - Run the experiment tests: `uv run --group dl-experiment pytest tests/test_pipeline_dl.py`
  - Run the offline eval: `uv run --group dl-experiment python scripts/eval_ranker_variants.py ...`
  - Without the group, `scripts/eval_ranker_variants.py --help` still
    exits 0; the friendly error only fires when a DL variant is
    actually requested.
  - `tests/test_pipeline_dl.py` is a single `pytest.importorskip("torch")`
    at module scope, so pytest reports it as **1** skip (not one per
    test function) when the group is not active.
- `embedding-experiment` (`48185b7`) — `huggingface-hub`, `scipy`. Required
  only by `scripts/bakeoff_embedding_models.py` and
  `scripts/bench_qwen_embed_speed.py` (embedding-model comparison tooling,
  not the live ranking path).
  - Install on demand: `uv sync --group embedding-experiment`

If a future experiment is added that needs a different heavy
runtime dep (e.g. jax, tensorflow), give it its own
`[dependency-groups]` group with a descriptive name, not a runtime
  direct dep.

- Migrate feedback from legacy JSON: `uv run python migrate_feedback.py`
- **Primary archive seeder** — ClickHouse (no GCP auth, 10-30x faster, real-time scores):
  `uv run python scripts/seed_hn_from_clickhouse.py` (default: 12 months, score ≥ 200)
- **Backup archive seeder** — BigQuery (requires `gcloud`/`bq` auth, stale snapshot):
  `uv run python scripts/seed_hn_from_bq.py --months N --min-score N`
- Dry-run archive seeders (fetch rows to JSONL, skip DB/Algolia):
  `uv run python scripts/seed_hn_from_clickhouse.py --dry-run --limit N --min-score N`
  `uv run python scripts/seed_hn_from_bq.py --dry-run --limit N --min-score N`
- Compare ClickHouse vs BigQuery output: `uv run python scripts/seed_smoke_test.py --bq-file bq.jsonl --ch-file ch.jsonl`
  or live: `uv run python scripts/seed_smoke_test.py --limit 50 --min-score 200 --skip-bq`
- Run leakage-safe offline eval: `uv run python scripts/eval_ranker_variants.py --window-days N`

## Service & runtime

- Run persistent server: `systemctl --user {status|start|stop|restart} hn_rewrite.service`
  (or directly: `uv run python server.py`)
- **Restart before verifying**: live behavior can differ from the checkout until
  `hn_rewrite.service` is restarted. Deployed code and the working tree can diverge;
  verify with `git log`, `systemctl --user status hn_rewrite.service`, then
  `journalctl` before concluding runtime behavior from code alone.
- Live smoke test shape: after restart, hit the dashboard plus cached/uncached
  `POST /api/tldr-detail`, then scan `journalctl --user -u hn_rewrite.service
  --since '1 min ago'` for errors.

## Frontend notes

- **Data-attribute collisions**: `data-*` attributes on story cards (e.g.
  `data-story-source`, `data-story-id` in `templates/components/story_card.html`)
  can be accidentally matched by generic `querySelectorAll('[data-*]')` selectors
  in `templates/index.html`. Tab buttons use `data-source`, `data-sort`,
  `data-age` — scope selectors to `.tab-btn[data-*]`. When adding `data-*` to
  templates, verify no JS selector collides with them.
- **Template tests**: `tests/test_server.py` pins CSS/JS contracts via
  template-string assertions. Update them when `templates/index.html` changes.
  Avoid adding tests that only check exact strings without validating runtime
  behavior (the user will request their removal).
- **Version semantics**: `dashboard_version=0` is valid cold-deck data.
  Client-side version comparisons must use `Number.isFinite()`, not truthiness.

## HN data sources (architecture overview)

The dashboard uses two external data sources for HN stories. Algolia was
removed from the live `hn` source pipeline on 2026-06-26; CH is now the
sole source for the live 30-day window and bulk operations.

| Source | Used for | Why |
|---|---|---|
| **ClickHouse** (`hackernews_history`) | Live 30-day window (`query_live_window`), bulk comment hydration (archive seed), regen-time bulk prewarm (all eligible HN rows by default) | Single SQL query for N stories; 10-100× faster than per-story Algolia |
| **Algolia** (`hn.algolia.com`) | Single-story items fallback (lazy TLDR detail for stories outside prewarm) | Real-time, no CH equivalent for one-off fetches; used only as fallback |
| **BigQuery** (`bigquery-public-data.hacker_news.full`) | Backup archive seeder (manual) | Same data as CH; slower; requires `gcloud`/`bq` auth |

The live `hn` source pipeline (`fetch_candidates` in `pipeline/__init__.py`) now
issues **2 CH calls per regen**:

1. `ch_client.query_live_window(days=30, min_score=5, limit=5000)` — every
   live HN story from the past 30 days with all fields (title, url,
   score, descendants, time, text).
2. The prewarm (comment text for all HN candidates with `comment_count > 0` and
   empty `top_comments`), inside `fetch_candidates_only`,
at regen time — not on the render path. Every user's first dashboard render
finds the candidate rows already populated. The first cards any user sees
have `top_comments` already populated — no Algolia wait and no render-time
prewarm latency.

Stories with no content to summarize (self_text, top_comments, and article_body
all empty, and no HN comment_count > 0) are filtered out by `is_summarizable()`
in `fetch_candidates`. Config knobs `prewarm_hn_full`, `prewarm_reddit_full`,
and `prewarm_lesswrong_full` (default true) control prewarm scope; set to false
to revert to top-by-score prewarm (`regen_prewarm_top_n=50` default; Reddit
prewarm is now driven by `reddit_prewarm_top_per_sub=10` — top 10 hot per sub
from the topfeed cache, not by score from a DB query).

CH has 1-24h latency for brand-new content (vs Algolia's real-time).
With the default 4h regen cycle, worst case is 5h lag for stories posted in the
last hour. Acceptable for "best of HN" view; the swipe deck mostly
shows older stories anyway.

The CH bulk client lives in `ch_client.py`. The previous per-story parallel
Algolia hydration (used for archive seeding before 2026-06-26) was removed
in `f4b2000` ("Collapse eval scripts, remove legacy_features and archived
Algolia") — `scripts/_archive/algolia/` no longer exists. If CH becomes
unavailable, recover it from git history (`git show f4b2000^:main/scripts/_archive/algolia/...`)
rather than expecting it on disk.

**Comment tree fetches walk `kids` arrays, never join the comments table.**
`ch_client.query_comments_bulk` fetches a story's comment tree with one
`id IN (...)` lookup per level (via each row's `kids` array), N+1 cheap
queries instead of a single join. An earlier version joined against
`(SELECT * FROM hackernews_history FINAL WHERE type = 'comment' ...)` once
per level and reliably exceeded play.clickhouse.com's query memory limit
(`Code: 241 MEMORY_LIMIT_EXCEEDED`) regardless of batch size — this killed
HN discussion TLDR generation site-wide from 2026-07-23 to 2026-07-26 (see
WORKLOG.md). Do not reintroduce a full-table join or `FINAL` scan in a
comment query; `tests/test_ch_client.py::test_comment_queries_do_not_join_or_scan_full_table`
guards against it.

## See also
- [WORKLOG.md](WORKLOG.md) — recent changes and operational events

## Backup

The HN database is backed up daily to Google Drive via a systemd user timer.

- Script: `scripts/backup_hn_db.sh`
- Service: `~/.config/systemd/user/hn-rewrite-backup.service`
- Timer: `~/.config/systemd/user/hn-rewrite-backup.timer` (active)
- Target: `drive:hn-rewrite/backups/<YYYYMMDDTHHMMSSZ>/hn_rewrite.db`
- Retention: 30 most recent snapshots (env: `HN_KEEP_N=30`)
- Logs: `journalctl --user -u hn-rewrite-backup.service`

### Manual backup

```bash
./scripts/backup_hn_db.sh                          # default config
HN_DB_PATH=/path/to/other.db ./scripts/backup_hn_db.sh
HN_KEEP_N=7 ./scripts/backup_hn_db.sh             # keep 7
```

### Restore

```bash
LATEST=$(rclone lsf --dirs-only drive:hn-rewrite/backups/ | sort -r | head -1)
rclone copy drive:hn-rewrite/backups/$LATEST/hn_rewrite.db ./hn_rewrite.db
sqlite3 hn_rewrite.db "PRAGMA integrity_check;"
```

## Testing notes

- **Curl sessions**: first-visit `GET /` creates one user, sets `hn_token`, and serves the dashboard directly. `/u/<token>` only imports an existing profile onto a new device. Always use `-c cookie.txt -b cookie.txt` when testing live API flows with curl so subsequent requests keep the same profile.
