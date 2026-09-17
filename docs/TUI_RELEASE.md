# Terminal client release

## Build and verification

From `/home/d/hn-rerank`:

```sh
uv sync
HN_TEST_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model uv run pytest tests/ -n 4
uv run pytest clients/tui/tests
uv run ruff check .
uv run ty check
uv build --package hn-rerank
cd /tmp
uvx --from /home/d/hn-rerank/dist/hn_rerank-0.1.0-py3-none-any.whl hn-rerank
```

The ONNX override is for the existing real-embedding backend tests only. Test
DBs are temporary. The client itself needs no model, database or backend extras.
`.github/workflows/tui.yml` copies the package outside the workspace on Linux,
macOS and Windows, runs headless tests, builds it and starts the installed wheel.
A standalone copy can also use `uv build` directly in its package directory.

## Publish

The intended command is `uvx hn-rerank`. PyPI returned 404 for the name on
2026-09-13; this does not reserve it. Configure `UV_PUBLISH_TOKEN` locally or a
PyPI trusted publisher, then run `uv publish dist/hn_rerank-0.1.0*` only after
verification and VPS integration pass. Never store the token in this repository.
If PyPI rejects the name, change `[project].name` to `hn-rerank-tui`, rebuild and
use `uvx --from hn-rerank-tui hn-rerank`; the console entry point stays unchanged.
After uploading, test that exact index command from outside the checkout.

## Deployment

The actual VPS service is `hn_rewrite.service`, with working directory
`/home/dev/hn-rewrite/main`. Its code is newer than this laptop checkout and has
uncommitted work. Do not replace it with the laptop tree, run a broad sync or
reset its changes. Port only the feed route, shared rendering snapshot, model
module and the stale-HTML helper's preservation of `DashboardDocument`.
Production Explore is shuffled; preserve that behavior in its feed orders.
Inspect fresh source hashes and service state, keep backups, run tests, restart
only this service, then use a dedicated test profile for authenticated feed,
summary, feedback/undo and ranking readiness checks. Keep the existing network
access policy. Public package availability is independent of server access.

Current deployment evidence and rollback paths belong in [FINDINGS](../FINDINGS.md).

## Laptop rename

On 2026-09-13, `/home/d/hn-rerank-v2` moved to `/home/d/hn-rerank` after an
inventory found one worktree, no tracked WIP and no other process using the old
path. `.opencode/`, `.playwright-mcp/` and the existing `docs/` contents were
preserved. No operational old-path references were found in project files,
user systemd units or `~/bin`. Updated the project trust entry in
`/home/d/.codex/config.toml`; its private rollback copy is
`/home/d/.local/state/hn-rerank-rename/codex-config-before.toml`. Reinstalled the uv environment to repair absolute
console-script shebangs; `uv run pytest` resolves the new environment.
The GitHub repository name and VPS directory were not renamed. Historical records
retain their original paths. To roll back the laptop move, first check active
ownership, move the directory back under the project lock, and reinstall the uv
environment again; do not move a directory out from under an active process.
