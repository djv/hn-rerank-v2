# MUSE-ACK

- Model: opencode-go/muse-spark-1.3-contributor (Muse Spark via OpenCode).
- CWD: /home/dev/hn-rewrite/hn-codex-forum-fixes-20261011 (isolated worktree).
- Branch: agent/codex-forum-fixes-20261011.
- HEAD: 62af20e ("Plan Codex Forum fixes for Muse handoff"; base main 557c0ee).
- Assignment: docs/evaluations/codex-forum-review-20261010/MUSE-HANDOFF.md
  implementing FIX-PLAN.md steps 1-4 (step 5 = this worktree's gates).

Accepted scope (two confirmed source-content defects for
`rss_community_openai_com` only; source ID, canonical URLs/IDs retained):

1. Narrow no-article policy for the Codex source across RSS article
   prewarm, ranked/warm article selection, and the short-post tap lane.
2. Terminal Discourse count/"Read full topic" footer removal before
   plain-text conversion, preserving authored lookalikes and partial
   footers; unknown markup stays intact.
3. One shared, pure, idempotent effective-content normalization for
   stored Codex rows (omit article_body, strip confident terminal
   footer, recompose text_content) applied to candidate,
   feedback/training, embedding/hash, and summary reads; versioned
   Codex summary keys with legacy-key rejection in stale/error
   fallback. Raw stored fields, votes, identities preserved; no cache
   deletion, DB migration, or data cleanup.
4. Offline fixtures + behavioral tests (selectors and tap make zero
   article requests; legacy-row read/embedding/summary proofs;
   generic-RSS controls), network/providers/embedders stubbed,
   failures reproduced before fixing.

Out of scope (per handoff): reply/count adapter, other-feed cleanup,
model/ranking experiments, main merge/integration, service restart,
production requests/deployment, production DBs, shared .env,
credentials, private-assistant-context, agent-memory, real generation,
live model fits, seeding, migrations, cache cleanup, data deletion,
broad TUI sweep, and any additional reviewer/agent.

Constraints honored: only this worktree edited; temp fixture DBs only;
OMP/OPENBLAS/MKL threads capped at one; focused tests first, then
`pytest tests/ -n 4 -q`, `ruff check .`, `ruff format --check` on
changed Python, `ty check`, `git diff --check`; no main push without
the supervising agent's verification.
