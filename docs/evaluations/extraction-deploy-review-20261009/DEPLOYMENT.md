# Authorized scoped deployment — 2026-10-09

At 17:58:08 UTC the VPS live worktree `/home/dev/hn-rewrite/main` fast-forwarded
from `be965fd25fa6390be0b0e4e46ff6ebf0231dbf36` to
`bf6e307c6c6d3200503ea00a56b4081fd759fd05`, then `hn_rewrite.service` restarted.
The tracked tree and index were checked clean immediately before deployment.
The commit range contains only two runtime lines in `server.py`, four fetch
regression cases, and handoff docs. Latest main was deliberately not deployed.
Untracked VPS research folders were preserved; no reset or data deletion.

- Rollback reference: `deploy-pre-empty-extraction-20261009` at the old HEAD.
- New systemd MainPID: 4033960; active/start timestamp read back at 17:58:08 UTC.
- `config.toml` SHA256 before and after:
  `99a2ad40f7f9973fd8e25946dd5d95e2eeac887e3ec3decc19323b091c67f57e`.
- Live profile151/arms/T0/final deadline unchanged by deployment. The restart
  naturally warms cached users and reranks; no research experiment was resumed.
- [Smoke result](DEPLOY-SMOKE.json): existing profile151 internally reused;
  dashboard/feed/readiness 200; read-only TLDR cache hit 200, misses 204;
  missing-story detail 404. No anonymous test profile or provider call made.
- A separate process importing deployed source verified empty/whitespace input
  returns None with all four extractors patched to raise. This is a simulated
  guard check, not a live fetched empty response.
- At 17:58:53 UTC the service remained active and the bounded journal since
  restart had zero matches for `ERROR|Traceback|quota_denied|empty HTML tree`.
  This short quiet interval does not prove the historical upstream issue fixed.
- Startup smoke local dashboard/feed latency was 1268/1505 ms and readiness
  417 ms, compared with the pre-restart quiet 6–16 ms probes. Normal warming
  was still running; these are individual probes, not distributions.

No uncached live TLDR generation was exercised, to avoid unnecessary quota.
The first missing-story detail probe used an incorrect JSON field and returned
400; the saved final smoke reran with `story_id` and verified 404. Existing
backend/TUI suite results were reused because no runtime code changed during
this deployment. Broad [three-model review](../general-review-20261009/SYNTHESIS.md)
found no blocker for this guard; proposed TUI changes remain unimplemented.
