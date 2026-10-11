# Codex Forum fixes and Muse handoff

Approved task: the user requested a fix plan and handoff to Muse on 2026-10-11.
Base: main 557c0ee. Evidence: [validated review](VALIDATION.md).

1. **Prevent mixed article input.** Add a narrow policy for
   rss_community_openai_com and apply it to RSS article prewarm, ranked/warm
   article selection, and short-post on-demand fetching. Codex topic HTML
   must never be fetched as an article; other sources retain their behavior.
2. **Remove generated RSS footer text.** Before plain-text conversion, remove
   only the terminal Discourse count paragraph and adjacent matching
   "Read full topic" anchor. Preserve opening-post HTML, authored small text,
   ordinary counts, and unrelated links. Unknown markup stays intact.
3. **Make legacy inputs safe without deleting data.** Use one source-specific,
   idempotent effective-content normalization for stored Codex rows: omit their
   article_body, remove only confidently recognized terminal footer text, and
   recompose text_content. Cover candidate and feedback/training reads plus
   embedding input/hash generation; changing only the parser would retain
   longer dirty DB text. Preserve raw stored fields, votes and identities.
   Apply the same normalized inputs to on-demand and prefetched summaries.
   Version Codex summary cache keys so prior mixed-input text is not reused;
   stale/error fallback must also reject incompatible keys. Other sources'
   cache/fallback behavior remains unchanged. No cache deletion or DB migration.
4. **Prove behavior.** Add offline fixtures for feed-footer removal and
   preservation; both proactive selectors and a short-post tap must make zero
   article requests. Seed a temporary legacy row with mixed body/footer and
   prove clean read/embedding/summary inputs, new hashes/keys, preserved raw
   data, no incompatible stale summary, and idempotence. Include generic-RSS
   controls. Stub network, providers and embedders; reproduce failures first.
5. **Validate and deliver.** Run affected tests, then backend tests/ -n 4,
   Ruff, touched-Python format, ty and diff checks. Use the VPS worktree below,
   numerical threads capped at one. Deliver the feature-branch diff and a
   CHECKS.md with exact results and unresolved limits; update worktree docs.
   The supervising agent verifies before any main merge or production rollout.

Muse: opencode-go/muse-spark-1.3-contributor, isolated VPS worktree
/home/dev/hn-rewrite/hn-codex-forum-fixes-20261011, branch
agent/codex-forum-fixes-20261011. Existing review already supplies the rationale;
this is implementation, not another review. No reply/count adapter, other-feed
cleanup, model/ranking experiment, production DB access, service restart or
deployment is part of this handoff. Existing data cleanup would be a separate
decision; the effective-input policy handles legacy contamination without it.
