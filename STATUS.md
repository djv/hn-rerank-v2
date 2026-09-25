# HN Rerank status

## Objective
Deploy the rate-limit client-IP fix (`b4fb589` line, now `origin/main`
`f5cbb16`) on the VPS, including Caddy `trusted_proxies`.

## Verified result
- VPS `main` worktree at `f5cbb16` (clean); `uv sync`; VPS suite 860
  passed. GitHub CI `ci` green for `f5cbb16`.
- Caddy upgraded 2.6.2 (Ubuntu) → 2.11.4 (official apt repo); 2.6.2
  rejected `servers { trusted_proxies }`. Runs as system unit
  `hn-dashboard.service` from the tracked `Caddyfile`; packaged
  `caddy.service` stays disabled. Admin API shows `trusted_proxies`
  static `127.0.0.1/32`, `::1/128` live.
- Smoke test on the VPS and from the laptop via the Funnel URL: `/hn/`,
  `/api/feed`, `/hn/api/feed` 200; `tldr-detail` uncached then
  `cached=True`; journal clean.
- Loopback capture from the laptop: app receives
  `X-Forwarded-For: <laptop IPv6>, 127.0.0.1`, so `_flask_client_ip()`
  keys on the real client. A client-sent `X-Forwarded-For: 1.2.3.4` never
  arrives: Funnel overwrites it.

## Blocker / limits
- Smoke tests created throwaway users in the live DB (fresh cookie jars).
- Non-interactive SSH lacks `~/.local/bin` on PATH; export it before `uv`.
- 18 pre-existing environmental `test_pipeline` errors on the laptop.
- Another session has uncommitted ranking/eval WIP in the laptop checkout.

## Next step
None for the deploy. Relaunch the TUI to pick up the current Date view
(12 newest stories in the deck, `f5cbb16`).
