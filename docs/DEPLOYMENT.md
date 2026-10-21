# Deployment configuration

The server reads `config.toml` from its working directory and binds loopback.
The checked-in configuration retains the existing model/feed settings and port
8766. `config.example.toml` is a smaller fresh-clone starting point, without the
optional side encoder. The dataclass-only port default remains 8765.

## Portable paths and existing deployments

`pipeline/config.py` checks for `shared/` beside the checkout's parent, using
its own source path. For the existing `<vps-home>/hn-rewrite/main` layout,
`<vps-home>/hn-rewrite/shared` is still discovered, including both model
folders and `.env`. Other worktrees under the same repository parent also work.
No environment changes are required for that layout; this is source/test
compatibility evidence, not a new live VPS smoke test. An existing explicit
`onnx_model_dir` in the VPS's local `config.toml` still takes precedence.
The public `config.toml` omits that host-bound field; no feed/model change is needed.

For another layout, set these in the **process environment before startup**
(for systemd, use `Environment=` or a private `EnvironmentFile=`):

- `HN_ONNX_MODEL_DIR`: primary ONNX/tokenizer directory; fresh fallback `onnx_model`.
- `HN_SIDE_MODEL_DIR`: optional Gemma directory; fresh fallback `onnx_model_embeddinggemma`.
- `HN_ENV_FILE`: fallback secrets file; defaults to the sibling `shared/.env`.
- `HN_RERANK_SERVER`: TUI default server, including any proxy prefix; fresh fallback
  `http://127.0.0.1:8766/`. An explicit `--server` or saved profile remains usable.

A checkout-local `.env` takes precedence over the fallback secrets file. These
path constants are evaluated when Python imports the config module, before
`server.load_env()`, so putting them only in the server-parsed `.env` is too late.
`HN_ENV_FILE` also selects the secrets file copied by the backup script.
Use literal paths there; the simple parser does not expand `$HOME` or `~`.
See [.env.example](../.env.example). Keep populated files out of Git.

Deployment verification is separate from checkout checks: restart the owned
service after an approved rollout, request the dashboard and cached/uncached
summary endpoints with an existing profile, and inspect a bounded journal window.
This showcase worktree does not deploy or restart the user's service.

## Proxy and access

[Caddyfile](../Caddyfile) binds port 8000 to loopback and proxies `/hn/` and
`/api/*` to loopback 8766. It trusts forwarding headers only from loopback.
A separately configured TLS edge may expose it; that is an operator decision.
The Python process uses Flask's threaded development server, not a hardened
production WSGI deployment. Keep it on loopback and assess the edge before
exposing it to additional users.

Profile links and cookies are bearer credentials. Server routes enforce
same-origin mutation checks and request throttles. These controls do not replace
account lifecycle management. Outbound article fetches reject private/local
addresses at connection time, limit redirects and cap response size; see
[http_fetch.py](../http_fetch.py) and [privacy limits](PRIVACY.md).
