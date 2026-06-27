# OpenCode Instructions for `hn-rewrite`

## Project shape

- This repository is a minimalist, local-first Hacker News reranking dashboard rewrite.
- Use `uv run python <script>.py` or standard `uv` commands to execute scripts and run tests.
## Working rules

- Make minimal, behavior-preserving changes unless the user asks for a broader refactor.
- Keep the runtime path local-first; do not add new external dependencies unless needed.
- **Be very skeptical of unusually high metrics** (e.g. NDCG > 0.40). We are unlikely to beat the Hacker News baseline by a large margin; high metrics often indicate feature leakage, train-test contamination, or metric saturation artifacts.
- **Do NOT standard-scale raw embeddings** (384-d MiniLM vectors are L2-normalized; StandardScaler must only touch metadata columns from `emb_dim:` onward).
- **Never delete or destructively modify the local database** (`hn_rewrite.db`, `hn.db`, or any `*.db` file in the working tree). The DB holds the user's accumulated feedback and is the single source of truth for personalization. No `rm`, no `DELETE FROM` without a `WHERE` clause that excludes all rows, no schema migrations that drop tables or columns with data. The pipeline's own `prune_stories` and `prune_*` operations are fine — they have explicit retention rules and `id NOT IN (SELECT story_id FROM feedback)` guards. When in doubt, ask before running any command that touches the DB file.
  - **Exception (2026-06-22):** 756 test/empty stories (time=0) were deleted with explicit user permission. This included 2 test stories (id=999 "Test", id=99999998 "Test regen live") that received 2 upvotes from user 1. Backup retained at `hn_rewrite.db.pre_test_removal_20260622T163344Z`.
- Keep test execution times optimized (target under 10 seconds total).
- **Always update relevant documentation** (e.g., [ARCHITECTURE.md](file:///home/dev/hn-rewrite/ARCHITECTURE.md), [WORKLOG.md](file:///home/dev/hn-rewrite/WORKLOG.md)) after making code or behavior changes.

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
- **Type hints are mandatory for new code** in `pipeline.py`,
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
- Run tests: `uv run pytest tests/`
- Run linting: `uv run ruff check .`
- Run type checking: `uv run ty check` (Astral's `ty`; pre-existing
  diagnostics are tracked in the baseline; new code must introduce
  zero new diagnostics. LSP errors must be fixed or `# type: ignore`
  with a documented reason)

## Dependency groups

`pyproject.toml` ships two groups beyond the runtime deps. Default
`uv sync` installs only `dev` (linters, pytest, type checker). The
`dl-experiment` group is opt-in.

- `dev` — pytest, pytest-asyncio, hypothesis, ruff, ty. Always
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
  - Without the group, `scripts/eval_ranker_variants.py` exits 1 with
    a friendly error pointing at this command.
  - The 21 `test_pipeline_dl.py` tests are skipped (not failed) by
    `pytest.importorskip("torch")` at the top of the file when the
    group is not active.

If a future experiment is added that needs a different heavy
runtime dep (e.g. jax, tensorflow), give it its own
`[dependency-groups]` group with a descriptive name, not a runtime
direct dep.

- Run persistent server: `systemctl --user {status|start|stop|restart} hn_rewrite.service` (or directly: `uv run python server.py`)
- Migrate feedback from legacy JSON: `uv run python migrate_feedback.py`
- **Primary archive seeder** — ClickHouse (no GCP auth, 10-30x faster, real-time scores):
  `uv run python scripts/seed_hn_from_clickhouse.py` (default: 6 months, score ≥ 200)
- **Backup archive seeder** — BigQuery (requires `gcloud`/`bq` auth, stale snapshot):
  `uv run python scripts/seed_hn_from_bq.py --months N --min-score N`
- Dry-run archive seeders (fetch rows to JSONL, skip DB/Algolia):
  `uv run python scripts/seed_hn_from_clickhouse.py --dry-run --limit N --min-score N`
  `uv run python scripts/seed_hn_from_bq.py --dry-run --limit N --min-score N`
- Compare ClickHouse vs BigQuery output: `uv run python scripts/seed_smoke_test.py --bq-file bq.jsonl --ch-file ch.jsonl`
  or live: `uv run python scripts/seed_smoke_test.py --limit 50 --min-score 200 --skip-bq`
- Run leakage-safe offline eval: `uv run python scripts/eval_ranker_variants.py --window-days N`

## HN data sources (architecture overview)

The dashboard uses two external data sources for HN stories. Algolia was
removed from the live `hn` source pipeline on 2026-06-26; CH is now the
sole source for the live 7-day window and bulk operations.

| Source | Used for | Why |
|---|---|---|
| **ClickHouse** (`hackernews_history`) | Live 7-day window (`query_live_window`), bulk comment hydration (archive seed), bulk prewarm (top-50 ranked) | Single SQL query for N stories; 10-100× faster than per-story Algolia |
| **Algolia** (`hn.algolia.com`) | Single-story items fallback (lazy TLDR detail for stories outside prewarm) | Real-time, no CH equivalent for one-off fetches; used only as fallback |
| **BigQuery** (`bigquery-public-data.hacker_news.full`) | Backup archive seeder (manual) | Same data as CH; slower; requires `gcloud`/`bq` auth |

The live `hn` source pipeline (`fetch_candidates` in `pipeline.py`) now
issues **1 CH call per regen**:

1. `ch_client.query_live_window(days=7, min_score=5, limit=2000)` — every
   live HN story from the past 7 days with all fields (title, url,
   score, descendants, time, text).

The prewarm (comment text for top-50 ranked by score) is a second CH call
inside `fetch_candidates_only`, at regen time — not on the render path.
Every user's first dashboard render finds the top-scored candidates already
populated. The first 4 cards any user sees have `top_comments` already
populated — no Algolia wait and no render-time prewarm latency.

CH has 1-24h latency for brand-new content (vs Algolia's real-time).
With a 3h regen cycle, worst case is 4h lag for stories posted in the
last hour. Acceptable for "best of HN" view; the swipe deck mostly
shows older stories anyway.

The CH bulk client lives in `/home/dev/hn-rewrite/ch_client.py`. The
previous per-story parallel Algolia hydration (used for archive seeding
before 2026-06-26) is preserved in
`/home/dev/hn-rewrite/scripts/_archive/algolia/` as a fallback if CH
becomes unavailable.

## See also
- [WORKLOG.md](file:///home/dev/hn-rewrite/WORKLOG.md) — recent changes and operational events

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
## Testing notes

- **Curl and spam users**: `curl -L` without a cookie jar (`-c/-b`) creates one user per redirect hop. Every `GET /u/<token>` returns a 302 to `../`, and without cookie persistence the redirect chain creates a new user on each hop. Always use `-c cookie.txt -b cookie.txt` when testing with curl.
