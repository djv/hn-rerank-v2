# TLDR pane budget

detail-v14 set summaries to a fixed 240-word reading pane, independent of
client size, zoom, fullscreen, and source length. No dimensions are sent to
the server. detail-v15 scales it slightly for long articles (below).

- Article-only: 6–8 bullets, aim for 240 words.
- Discussion-only: 6–8 bullets, aim for 240 words.
- Combined: 3–4 bullets and 120 words per section, 240 total.

These are generation targets, not guaranteed rendered heights. Thin sources
may use fewer words/bullets; prompts prohibit padding and invented details.
Output shaping caps bullets at eight total. Existing token ceilings remain.
The prompt-version bump invalidates exact-key caches; regeneration occurs on
normal demand/prefetch. Provider/quota fallback can still serve a stale summary.
No extraction, layout, zoom handlers, or client key bindings change in this deploy.

## Discussion emphasis (detail-v14)

The combined Discussion prompt now explicitly requires **bold** key terms in
every content bullet, matching the emphasis instruction in the other summary
paths. Budgets are unchanged. The cache-version bump makes existing exact-key
summaries regenerate on demand; provider/quota fallback can still serve stale text.

## Deployment verification

VPS deployed `daa8112`, exact-tree backend 820 passed; Ruff/format/ty clean.
Service restart active, dashboard and cached summary 200. Uncached Latent Space
bio-security summary regenerated successfully: 7 bullets / 414 whitespace-delimited
words versus the previous 4 bullets / 564 characters. The model overshot the
240-word target; this is a soft prompt budget, not a hard word cap. Actual TUI
screen fit remains unverified. Bounded post-restart journal showed no errors.

## Section coverage (detail-v15)

Both article prompts now give each bullet a different section of the piece,
in order, and cover every section, including late newsletter items and a
closing story. A section gets a second bullet only after every section has
one. When there are more sections than bullets, the shorter sections share
one bullet instead of being dropped. Article sections use no `####` headings;
each bullet starts with its section's topic in bold. Cause: Import AI 475
(2026-10-05) spent all four bullets on its first item; see FINDINGS.md.

## Long-article budget (detail-v15)

At the user's request (2026-10-05), long articles now get slightly more room.
The article's length is the longer of its self text and fetched body, not
their sum, because an RSS self text is usually the feed's copy of the body.
Discussions always keep the base budget.

| Article length | Combined article half | Article-only |
|---|---|---|
| under 10k chars | 3–4 bullets, 120 words | 6–8 bullets, 240 words |
| 10k–20k chars | 4–5 bullets, 150 words | 8–10 bullets, 300 words |
| 20k+ chars | 5–6 bullets, 180 words | 10–12 bullets, 360 words |

The bullet cap and the output-token ceiling grow with each tier, so the extra
bullets are kept and the longer output is not truncated. Summaries of long
articles can overflow the 240-word pane; Enter zooms the summary in the TUI.

## Feed copies sent once; closing stories (detail-v16)

When a story's self text is the RSS feed's copy of the fetched article (most
of ten sampled 8-word runs appear in the body), the prompt sends only the
article body. Sending both repeated the post's opening. A self text that
differs from the body, or that the body only partly contains, is still sent.
The article prompts also name a newsletter's later items and any closing
story, fiction or personal note as sections. Cause: the live detail-v15
Import AI 475 summary covered every news item but skipped Tech Tales.
