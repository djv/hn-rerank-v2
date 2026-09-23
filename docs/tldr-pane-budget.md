# Fixed TLDR pane budget (detail-v13)

Summaries target a fixed 240-word reading pane, independent of client size,
zoom, fullscreen, and source length. No dimensions are sent to the server.

- Article-only: 6–8 bullets, aim for 240 words.
- Discussion-only: 6–8 bullets, aim for 240 words.
- Combined: 3–4 bullets and 120 words per section, 240 total.

These are generation targets, not guaranteed rendered heights. Thin sources
may use fewer words/bullets; prompts prohibit padding and invented details.
Output shaping caps bullets at eight total. Existing token ceilings remain.
The prompt-version bump invalidates exact-key caches; regeneration occurs on
normal demand/prefetch. Provider/quota fallback can still serve a stale summary.
No extraction, layout, zoom handlers, or client key bindings change in this deploy.
