# OpenCode Instructions for `hn-rewrite`

## Project shape

- This repository is a minimalist, local-first Hacker News reranking dashboard rewrite.
- Use `uv run python <script>.py` or standard `uv` commands to execute scripts and run tests.
- Treat `public/` as generated output unless a task explicitly targets it.

## Working rules

- Make minimal, behavior-preserving changes unless the user asks for a broader refactor.
- Keep the runtime path local-first; do not add new external dependencies unless needed.
- **Be very skeptical of unusually high metrics** (e.g. NDCG > 0.40). We are unlikely to beat the Hacker News baseline by a large margin; high metrics often indicate feature leakage, train-test contamination, or metric saturation artifacts.
- **Do NOT standard-scale raw embeddings**: StandardScaler must only be applied to metadata columns (indices `emb_dim:` onward), leaving the 384-dimensional unit-normed raw embeddings untouched. Scaling raw embedding dimensions independently distorts their semantic cosine similarity structure and collapses ranking performance.
- **Never delete or destructively modify the local database** (`hn_rewrite.db`, `hn.db`, or any `*.db` file in the working tree). The DB holds the user's accumulated feedback and is the single source of truth for personalization. No `rm`, no `DELETE FROM` without a `WHERE` clause that excludes all rows, no schema migrations that drop tables or columns with data. The pipeline's own `prune_stories` and `prune_*` operations are fine — they have explicit retention rules and `id NOT IN (SELECT story_id FROM feedback)` guards. When in doubt, ask before running any command that touches the DB file.
  - **Exception (2026-06-22):** 756 test/empty stories (time=0) were deleted with explicit user permission. This included 2 test stories (id=999 "Test", id=99999998 "Test regen live") that received 2 upvotes from user 1. Backup retained at `hn_rewrite.db.pre_test_removal_20260622T163344Z`.
- Keep test execution times optimized (target under 10 seconds total).
- **Always update relevant documentation** (e.g., [ARCHITECTURE.md](file:///home/dev/hn-rewrite/ARCHITECTURE.md)) after making code or behavior changes.

## Common commands

- Install or refresh the environment: `uv sync`
- Run tests: `uv run pytest tests/`
- Run linting: `uv run ruff check .`
- Run persistent server: `systemctl --user {status|start|stop|restart} hn_rewrite.service` (or directly: `uv run python server.py`)
- Run one-shot generation: `uv run python generate.py`
- Migrate feedback from legacy JSON: `uv run python migrate_feedback.py`

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

## Comment Backfill

- `fetch_story` short-circuits on cached stories with stale/missing `top_comments`. Fixed: `comments_stale` check falls through to Algolia items API. Comment re-fetch capped at 100 per pipeline run (`fetch_stories_by_id` sorts stale IDs descending, takes top 100).
- Error paths in `fetch_story` (non-200, invalid item, exception) preserve existing cached rows — `if story is None` guard prevents `_empty_story(sid)` from overwriting real data on transient failures.
- Corrupted stories (`title=""` but `text_content` preserved from COALESCE) are detected and given priority in the re-fetch queue, ahead of the 100-slot stale-comment cap.
- `_empty_story` is destructive: zeros `title`, `time`, `score`, `self_text`, `top_comments`, `text_content`. Only `article_body` survives (COALESCE in `upsert_story`). A single transient 403 on the Algolia items API can permanently damage cached stories if error paths call `_empty_story`.
- `_row_to_story` recomposes `text_content` live from raw parts on every read. This hides corruption — zeroed `title`/`time` with preserved `article_body` produces non-empty `text_content`, so the story passes filtering but shows blank title and epoch timestamp ("20624d ago").
- `upsert_story` COALESCE only covers `article_body`. All other columns (title, score, time, self_text, top_comments, text_content) are overwritten unconditionally. This is an architectural vulnerability.
- 1,940 stories still need comment backfill. They'll be gradually re-fetched as they appear in future Algolia search windows (100 per pipeline run).
```

## Testing notes

- **Curl and spam users**: `curl -L` without a cookie jar (`-c/-b`) creates one user per redirect hop. Every `GET /u/<token>` returns a 302 to `../`, and without cookie persistence the redirect chain creates a new user on each hop. Always use `-c cookie.txt -b cookie.txt` when testing with curl.
- **Spam user cleanup**: 334 spam users created by curl redirect-loop testing were deleted on 2026-06-22. Users table currently has 3 real users: id=1 (token="default", 1789 feedback), id=78 (token="new", 32 feedback), id=79 (token="new2", 113 feedback).
- **Test story cleanup**: 756 stories with `time=0` were removed on 2026-06-22 (2 test stories + 754 empty _empty_story artifacts). Backup: `hn_rewrite.db.pre_test_removal_20260622T163344Z`. The 2 test stories (999, 99999998) were also removed from the legacy JSON at `/home/dev/hn_rerank/.cache/user_feedback/dashboard_feedback.json` via `jq` (backup: `dashboard_feedback.json.pre_test_removal_20260622T163344Z`). Re-running `migrate_feedback.py` will NOT re-insert them.
- **`_empty_story` forensic analysis**: The 754 empty stories spanned IDs 48219870–48590505 (avg spacing ~492), all from `source=hn` with empty title/text/score/comment_count. Pattern suggests two bursts: sparse (48.2M) and dense (48.4-48.6M). Root cause: Algolia API returned non-200 (likely 403/429) for uncached stories, triggering `if story is None: db.upsert_story(_empty_story(sid))` in three error paths of `fetch_story`. The existing `if story is None` guard prevents overwriting cached data but still creates empty rows for uncached stories. This was a single historical instability period — no new time=0 stories have been created since cleanup (0 in the 25+ minutes since deletion).

## Progress

### Done
1. **Title-embedding dedup removed** — `fast_rerank_for_user` reverted to simple gravity-sort + top-1000 pre-filter. `get_or_compute_title_embeddings`, title pre-caching, and `ModelConfig.title_similarity_*` fields deleted.
2. **Spam users cleaned up** — 334 spam users deleted from `curl -L` testing (no feedback). 3 real users remain.
3. **Tests passing** — 55 passed, 1 deselected (the dedup test that was removed). Lint: clean.
4. **Dual-gate SVM activation** — `min_up_for_svm=20`, `min_down_for_svm=20` in `pipeline.py`, `config.toml`, `ModelConfig`. Blend uses `min(n_up, n_down)` as basis with `blend_start = min(gates)`, `tier3_blend_window=60`. SVM only trains when both classes have ≥20 examples.
5. **Test story cleanup** — 756 time=0 stories removed (2 test stories + 754 _empty_story artifacts). User 1 feedback: 1791 → 1789. Backup: `hn_rewrite.db.pre_test_removal_20260622T163344Z`. Diagnostic script at `/tmp/diag_user79.py`. JSON edited via `jq` to remove the 2 test records from the legacy migration source.

### Known issues (not bugs, workload characteristics)
- **Cold render for user_id=1 (1789 feedback)**: first dashboard render after restart takes 3-5s and allocates ~1GB. The SVM is retrained live from cached embeddings on every request (no DB model cache). This is expected behavior for 1789 feedback points.
- **Memory doesn't shrink after request**: numpy/sklearn internals retain memory after training. Peak grows asymptotically to ~1GB. Systemd `Restart=on-failure` recovers if the OOM killer fires.
