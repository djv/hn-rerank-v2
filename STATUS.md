# HN Rerank status

Saved 2026-10-05. Previous handoff (TUI one-row footer):
[before-tldr-section-bullets](docs/status-archive/before-tldr-section-bullets-20261005.md).

## Objective
Article summaries give each bullet a different section of the piece and
cover every section (user request 2026-10-05, after the Import AI 475 summary
spent all four bullets on its first item).

## Verified result
- Both article prompts ask for one bullet per section in order, and have
  shorter sections share a bullet when sections outnumber bullets. Article
  sections use no `####` headings. `TLDR_PROMPT_VERSION` is now detail-v15.
  Cause and evidence are in FINDINGS.md.
- Local tests only; the change has not been run against the live model.

## Blocker / limits
- Not deployed: SSH to the VPS needs the key's passphrase (`ssh-add`), and
  the agent's SSH agent had no key loaded.
- RSS `self_text` is clipped at 8,000 chars and probably duplicates the
  opening of the article body in the prompt. That is left unchanged and is
  recorded in FINDINGS.md.
- Unrelated TUI test edit, mockups, TLDR inspect script and kernel log are
  preserved and uncommitted.

## Next step
- Deploy (docs/TUI_RELEASE.md), regenerate story `-693920591`, and check that
  all six Import AI 475 items appear in the summary.
- The user still needs to check the footer and tint from 2026-10-03 in a real
  terminal.
