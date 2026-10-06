# HN Rerank status

Saved 2026-10-05. Previous handoff (TUI one-row footer):
[before-tldr-section-bullets](docs/status-archive/before-tldr-section-bullets-20261005.md).

## Objective
Article summaries give each bullet a different section of the piece and
cover every section (user request 2026-10-05, after the Import AI 475 summary
spent all four bullets on its first item). Long articles get a slightly
larger budget (follow-up request, same day).

## Verified result
- Both article prompts ask for one bullet per section in order, and have
  shorter sections share a bullet when sections outnumber bullets. Article
  sections use no `####` headings. `TLDR_PROMPT_VERSION` is now detail-v15.
  Cause and evidence are in FINDINGS.md.
- Article budget tiers: base under 10k chars, +1 bullet / +30 words from
  10k, +2 / +60 from 20k (doubled for article-only). Discussions are
  unchanged. Tiers are in docs/tldr-pane-budget.md.
- Deployed `e10d492` (detail-v16; rollback tag `deploy-pre-tldr-feed-copy`):
  the prompt sends an RSS feed's copy of the article once, and closing stories
  count as sections. Live Import AI 475: 6 bullets, one per section in order,
  Tech Tales included; smoke test clean.

## Blocker / limits
- RSS `self_text` is still clipped at 8,000 chars at ingest; with a fetched
  body it is now dropped from the prompt when it is the feed's copy.
- Unrelated TUI test edit, mockups, TLDR inspect script and kernel log are
  preserved and uncommitted.

## Next step
- None for this objective. Watch other long multi-item posts; budgets and the
  feed-copy rule are in docs/tldr-pane-budget.md.
- The user still needs to check the footer and tint from 2026-10-03 in a real
  terminal.
