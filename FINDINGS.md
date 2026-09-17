# HN Rerank findings

## Editorial terminal client — 2026-09-17 handoff refresh

- Current state: warm editorial polish is implemented on `feat/terminal-client`; no additional code changes were needed for this handoff.
- Agent-tested: client suite 25 passed / 1 Windows-only skip; Ruff and ty clean; headless visual preview exited successfully.
- Remaining: native terminal visual confirmation and credential-dependent PyPI publication.

## Editorial terminal client — 2026-09-13

- Implemented: charcoal/ivory theme, restrained orange tabs/focus/selection,
  explicit `>` marker, domain/points/comments/age metadata, summary heading,
  Markdown spacing, thin pane divider and context-sensitive shortcut footer.
  Setup separates import and creation. Existing credential/API code is unchanged.
- Agent-tested: 25 client tests passed / 1 Windows-only skip; backend 541 passed /
  1 skipped. Ruff and ty clean. Rebuilt wheel version and headless setup startup
  passed outside the checkout. New tests cover 60/80/100/140 columns, filters,
  focus, preserved reading scroll and selection, setup validation and empty/errors.
- Visual evidence: offline SVG renders under /tmp/hn-editorial-*.svg inspected
  for populated/code, empty/error and setup states, including a long headline.
  NO_COLOR=1 is set in the agent shell; color preview explicitly unsets it.
  OptionList vertical component padding clips metadata in Textual 8, so options
  use horizontal padding and a literal marker instead of a decorative border.
- Preview: `env -u NO_COLOR TERM=xterm-256color COLORTERM=truecolor uv run python
  -m clients.tui.tests.preview --headless`; omit --headless for a synthetic terminal
  exercise. No production profile or server is used. Graphical windows launched,
  but a successful capture of the actual preview window was not obtained.
- Resolved: guard fix is included in the pushed branch; latest `tui.yml` run
  (2026-09-14, `34805118943`) passed. Native terminal visual confirmation and
  credential-dependent PyPI publication remain pending.
  Earlier CI below covers the old revision.
- Rollback: editorial work and the setup guard are committed on
  `origin/feat/terminal-client`; revert with `git revert` if needed.
  Existing unrelated untracked files were preserved.
  Wheel/sdist can be rebuilt from the previous source if needed.
- CI 2026-09-13: run `34739788455` failed all OS on ruff 0.16 I001 import order
  (local ruff was 0.15.17 and passed); fixed in `45ca54f`. Rerun `34739910637`:
  ubuntu/macos green, Windows red on
  `test_setup.py::test_import_validates_then_persists_and_relaunches` —
  `on_option_list_option_highlighted` queried `#headlines` while the Setup
  screen was active (straggler highlight during teardown). Guarded in `763cc20`
  (`setting_up`/existence early return); local client suite green.

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
