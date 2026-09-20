# Terminal client release

## Build and verification

From `/home/d/hn-rerank`:

```sh
uv sync
HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model uv run pytest tests/ -n 4
uv run pytest clients/tui/tests
uv run ruff check .
uv run ty check
uv build --package hn-rerank
cd /tmp
uvx --from /home/d/hn-rerank/dist/hn_rerank-0.1.0-py3-none-any.whl hn-rerank
```

The ONNX override is needed because `config.toml` pins the VPS model path
(`HN_TEST_ONNX_MODEL_DIR` is not read; the variable is `HN_ONNX_MODEL_DIR`).
It applies to the real-embedding backend tests only. Test
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

`origin/main` is the single source of truth, and the VPS checkout at
`/home/dev/hn-rewrite/main` is a clean worktree of it. Tag the current commit
first, then deploy:

```sh
ssh hetzner 'cd /home/dev/hn-rewrite/main && git tag deploy-pre-<change> \
  $(git rev-parse --short HEAD) && git pull --ff-only origin main'
ssh hetzner 'cd /home/dev/hn-rewrite/main && /home/dev/.local/bin/uv sync'
ssh hetzner 'systemctl --user restart hn_rewrite.service'
```

Rollback is `git reset --hard deploy-pre-<change>` plus a restart. The service
is a systemd **user** unit (`systemctl --user ...`) running
`uv run python server.py` on 127.0.0.1:8766 behind Caddy; production Explore
stays shuffled and `_patch_current_version` preserves the attached
`DashboardDocument` feed.

After restart, smoke the dashboard plus cached/uncached `POST /api/tldr-detail`
with a dedicated test profile, and scan
`journalctl --user -u hn_rewrite.service --since '1 min ago'` for errors. Keep
the existing network access policy; public package availability is independent
of server access. Deployment evidence and rollback paths live in
[FINDINGS](../FINDINGS.md).

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
