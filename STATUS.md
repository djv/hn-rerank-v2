# HN Rerank status

- Objective: rename the laptop project, implement and release the terminal reader
  with a backward-compatible VPS feed API.
- Owner: none; implementation handoff saved and resource locks released.
- Result: implemented, committed on `feat/terminal-client`, and pushed. Laptop is
  `/home/d/hn-rerank`. VPS API deployed and live integration passed. Wheel/sdist built.
- Verification: laptop backend 541 passed / 1 skipped; client 16 passed (1 Windows-only skip on Linux); VPS 768
  passed; Ruff/ty clean. Wheel isolation and laptop PTY verified. Final code revision
  `12f7b08` passed Linux/macOS/Windows CI:
  https://github.com/djv/hn-rerank-v2/actions/runs/34738793586
- Unresolved: PyPI credentials or trusted publishing are required; no public client
  package has been uploaded. The user question requesting publishing access is pending.
- Next action: publish with configured PyPI access, then verify `uvx hn-rerank` from
  outside the checkout. Until then, run the built wheel with `uvx --from
  /home/d/hn-rerank/dist/hn_rerank-0.1.0-py3-none-any.whl hn-rerank`. Evidence and rollback: [FINDINGS.md](FINDINGS.md).
