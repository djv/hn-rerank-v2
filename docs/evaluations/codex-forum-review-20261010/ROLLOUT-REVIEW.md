# Review — Codex Forum rollout plan (target 794caa4)

**Verdict: NOT READY as written. The fixes are small plan edits, not code changes.** Pinning, the
worktree path, WIP preservation, the code-only rollback and the summary-key
invalidation are sound. One startup consequence is not addressed: side-embedding
coverage (H1). The backup time bounds and two smoke checks are not concrete
enough to execute or review (M1–M4). Deployment authorization remains out of scope.

Scope check: all 8 bundled sources match SOURCE-SHA256.txt (I re-hashed them). DIFF.patch
covers exactly those 8 files. I read the diff plus the support files (config,
side_embeddings, candidate_cache, ranking, server, database). I ran no tests.

## Findings

**H1 — Side-embedding coverage cliff is not planned for (plan defect).**
Plan §Intended behavior L40-41 and step 4 L79-81 say only that "derived vector hashes … may change".
- `story_embedding_text` now always applies the effective Codex view (ranking.py:647-661, diff L330-347).
  Every legacy Codex RSS description carried the generated footer, so essentially
  *every* stored Codex row, candidates and feedback alike (ranking.py:1255-1260),
  gets a new text hash.
- Side vectors are matched by that same hash (side_embeddings.py:39-41, 119-142).
  They are written only by the out-of-process `scripts/embed_side_vectors.py`
  (`hn-rewrite-side-embed.timer`, config.toml:44-46). That script is not in the bundle,
  and the plan never mentions the timer.
- `side_embedding_min_coverage = 0.98` (config.py:55): if Codex rows exceed about 2% of
  candidates plus feedback, `model_space` returns None (side_embeddings.py:171-178).
- Result: ranking for **all users** silently falls back to stored vectors only, a
  different production model, until the timer re-encodes. Rollback flips the hashes back
  and can cause a second drop.

Required:
- In preflight, record an aggregate count of Codex rows in the candidate window and in the
  feedback set against total pool size.
- Record the side-embed timer's schedule, its working directory/checkout and its last run.
  Decide before approval whether to accept the temporary fallback or run the timer once
  right after restart.
- In step 7, add `side_embeddings_low_coverage` and the `side_embeddings` trace label to
  the watched signals. Make persistent "off" a stop/report criterion.

**M1 — Backup "bounded timeouts" are unspecified (plan defect).** Step 2, L65-68.
- No method and no numbers are given. With a stepwise SQLite backup (`sqlite3.backup` with
  `pages>0`, or the CLI `.backup`), writes from the live service restart the copy. Under
  ordinary warming it may never finish without an overall deadline.
- Name the mechanism: Python `src.backup(dst, pages=-1)`, a single read transaction that is
  safe in WAL mode, or `VACUUM INTO`. Set concrete values, e.g. busy_timeout 30 s, overall
  15 min, and a `quick_check` cap.
- Feasibility is fine on the 16:30 figures: about 1.2 GB DB plus 2 GiB reserve against 23.7 GB
  free. Note that WAL cannot checkpoint past the backup's read snapshot, so the WAL grows
  while the backup runs.
- Free space is a live gate: it must be re-read immediately before the backup.

**M2 — Step 6 key calculation needs service env next to secrets (plan defect).** L93-95.
- `_tldr_cache_key` folds in `_llm_cache_identity()`, which is `LLM_PROVIDER`/`LLM_MODEL`
  from the process env (server.py:1251-1297). The cache GET response does not return
  the key (server.py:2959-2981).
- To compute the "deployed current key", step 6 must say how to obtain only those two
  non-secret values, e.g. `systemctl show -p Environment` or a key-filtered read, without
  opening or echoing API keys. Otherwise drop the exact-key comparison and rely on the
  pure-helper checks.

**M3 — "Repeated source-policy failure" is not observable (plan defect).** Step 7, L100.
- `pipeline/codex_forum.py` emits no logs, and no source-policy error string exists in
  the diff, so the criterion cannot fire.
- Replace it with concrete signals: tracebacks mentioning `codex_forum` or
  `effective_codex_story`, `tldr_detail … result=` events for the sampled Codex id, and
  the step-6 helper result.

**M4 — In-process primary re-embedding cost is not bounded (plan defect, live gate).**
- `get_or_compute_embeddings` (ranking.py:900-938) synchronously encodes every Codex
  candidate or feedback row whose hash changed. Settings: batch 1, 4096 max tokens,
  4-core VPS, `warm_pool_size=2`. This happens inside the first regen/warm and during
  feedback dedup (diff L96-112).
- Texts get shorter, since the article is dropped, but the plan gives no row count or time
  estimate for the 10-minute window. Use the H1 aggregate count to estimate it. A slow first
  regen should be reported as a live gap, not read as a regression.

**L1 — Target fetch and ref semantics (plan defect, minor).** Step 3, L70-73.
- Add `git cat-file -e TARGET^{commit}` (fetch if absent) and confirm which branch the
  `main` worktree has checked out, and that no other worktree holds that branch.
- Rollback detaches HEAD while the branch ref stays at TARGET (L110-112). Record that, so a
  later pull or another agent does not silently re-deploy.

**L2 — Rollback side effects are understated (plan defect, minor).** L114-116.
- Old code re-opens the Codex article lanes (enrichment.py `_NO_ARTICLE_SOURCES`,
  server.py:3175-3183). It can write new mixed `article_body` raw fields and
  mixed-input summaries.
- Legacy summary rows survive, because `tldr_cache` is keyed by (story_id, cache_key)
  (database.py:355-361), so old code can reuse them. Say this explicitly. Expect a side
  coverage drop again (H1).

**L3 — Smoke checks are weak but honestly labelled (acceptable).** Step 5.
- `min_version=0` always returns `ready:true` (server.py:3749-3763), so it is a liveness
  check only. Also re-poll with `min_version = current_version+1` after the first regen to
  show the deck advanced.
- Accepting a 204 from `/api/tldr-cache` (server.py:3905-3940) is correct, given the
  `codex-inputs-v1` miss (server.py:1313-1325) and the parked stale fallback
  (server.py:2989-3002). The plan correctly does not treat it as generation proof.

## Confirmed sound

- Pinning: exact SHA, no newer main, re-verify the 8-file manifest; config and lock unchanged.
- Paths are consistent: `/home/dev/hn-rewrite/main` worktree, and config `onnx_model_dir`
  under `/home/dev/hn-rewrite/shared`.
- The untracked archive does not collide, `--ff-only` protects tracked WIP, and there is
  no reset/clean anywhere.
- Cache invalidation matches the stated scope. The marker applies only to Codex
  (server.py:1313-1325), every key call site passes `source` (diff L418-531), and other
  sources hash as before. The stale/error fallback refuses Codex summaries whose key does
  not match, consistent with the parked decision, which I did not reopen.
- Provider use is bounded: stale prefetch adds 1 per run (config.py:235), is skipped
  during provider cooldown (server.py:1599-1607), and sits under the 240 uncached limits.
  Approval must acknowledge this, per step 4.
- Stop/acceptance criteria and the "do not claim full migration" wording are appropriate.
  No schema change; DB restore is kept out of rollback.

## Live gates to refresh at execution (not plan defects)

- Production HEAD (e1d3fe3 at 16:30) and the clean tracked tree/index.
- The untracked inventory.
- Unit WorkingDirectory/ExecStart and the current PID/start time.
- Disk, RAM, load, DB+WAL size.
- Config and uv.lock hashes.
- Presence of the primary model and the side-model artifacts.
- CI on the exact SHA.
- Side-embed timer state.
- Codex row counts (H1/M4).
- Provider cooldown state.
- Actual regen timing read back from config.

**To mark ready:** amend steps 1/4/7 for H1, give step 2 concrete backup mechanics and
deadlines (M1), resolve the step-6 env question (M2), and replace the M3 criterion.
