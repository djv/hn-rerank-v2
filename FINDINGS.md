# HN Rerank findings

## Laptop project and package — 2026-09-13

- Agent-tested: checkout moved from `/home/d/hn-rerank-v2` to `/home/d/hn-rerank`;
  one worktree, no other cwd users at move time, original untracked `.opencode/`,
  `.playwright-mcp/`, `docs/MANUAL_TESTING.md` preserved. GitHub name remains
  `djv/hn-rerank-v2`. uv console scripts were reinstalled and verified at the new path.
- Agent-tested: `HN_TEST_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model uv run
  pytest tests/ -n 4`: 541 passed, 1 skipped. Real test model copied from the VPS;
  no production DB copied or used by development tests. Client: 16 headless tests
  passed, 1 Windows-only test skipped locally; Ruff and ty clean. A laptop PTY exercised navigation, feedback, undo,
  reading/help and clean exit at 120 columns; headless tests also cover 80 columns.
- Agent-tested: Hatchling wheel and sdist in `dist/`; installed wheel launched via
  `uvx --from` from `/tmp`. Isolated headless startup and config round trip passed;
  numpy, sklearn, ONNX Runtime and Flask absent. Default runtime dependencies are
  Textual, HTTPX, platformdirs and their transitive dependencies.
- Agent-tested: source revision `12f7b08` on `feat/terminal-client` passed Linux,
  macOS and Windows CI, including standalone build/install, headless startup and
  Windows owner-only credential ACL verification:
  https://github.com/djv/hn-rerank-v2/actions/runs/34738793586
- Configured and read back: updated only the project trust path in
  `/home/d/.codex/config.toml`, preserving all other TOML values. Private rollback
  copy: `/home/d/.local/state/hn-rerank-rename/codex-config-before.toml`.
  No active session restarted. Generated uv environment has no old-path matches.
- Publication pending: PyPI name lookup returned 404, which does not reserve the
  name. No local publishing token or OIDC identity configured; requested account
  setup through the question popup. No package uploaded and `uvx hn-rerank` from
  the public index is not yet verified. See [release instructions](docs/TUI_RELEASE.md).

## VPS feed deployment — 2026-09-13

- Agent-tested: live service `hn_rewrite.service` runs in `/home/dev/hn-rewrite/main`.
  Its baseline was `4ec6bc6`, ahead of its origin and with uncommitted database,
  ranking, enrichment, tests and documentation changes. Those changes were preserved.
- Agent-tested: ported only feed rendering/API/models and the new API test, using
  source-hash checks and a bounded remote flock. No other agent process had that cwd;
  the flock cannot arbitrate controllers that do not participate. Server policy,
  model configuration, deployment directory and production schema were untouched.
- The newer VPS Explore filter shuffles; the port preserves shuffled feed orders.
  Its stale HTML helper also updates latest-version metadata and now preserves the
  feed attachment. [Exact scoped patch](docs/TUI_VPS_PATCH.patch).
- Agent-tested: copied VPS source was tested on the laptop with temporary databases:
  220 API/server tests passed. VPS full suite before/after: 767/768 passed;
  Ruff and ty passed after deployment. Service restarted successfully.
- Agent-tested: live HTTPS `/hn/` integration created a dedicated profile, returned
  68 stories and a 3592-character summary, accepted an upvote, reached its ranking
  target with the story excluded, then cleared the vote and verified zero feedback.
  Importing the same profile preserved identity. The private test credential is in
  `/home/d/.local/state/hn-rerank-test/profile.json`; never commit its contents.
  A bounded post-restart journal scan found no ERROR/Traceback/Exception lines.
- Rollback: `/home/dev/hn-rewrite/shared/deploy-backups/20260913-tui-feed/` contains
  original `server.py`, `pipeline/render.py` and hash manifest. Recheck concurrent
  changes before restoring those two files, then restart only `hn_rewrite.service`.
  The new dependency-free model/test files may remain inert on rollback. No DB
  rollback is needed. Laptop rename rollback is in the release instructions.
