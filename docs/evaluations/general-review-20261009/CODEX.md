No finding blocks **only the two-line extraction guard deployment**, with the usual cached warm/restart procedure and no configuration change. Deployment was not attempted.

1. **P2 — Code-proven: temporary server errors hide stories.** [api.py:291](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/api.py:291), [app.py:1259](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1259). A summary request returning HTTP 500/503, with no previous summary, becomes a generic `APIError` → the selected story disappears for the session. Repeated failures can drain the visible list. **Minimal next step:** classify retryable server errors as transient and retain the story with retry feedback.

2. **P3 — Code-proven: the first counts poll can miss an update.** [app.py:1497](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1497). Hot counts change after feed loading but before the first poll → that poll records the new `counts_version` as its baseline without refetching. Displayed counts remain old until another change or refresh. The selective test explicitly expects baseline-only behavior. **Minimal next step:** synchronize counts when establishing the initial baseline.

3. **P3 — Plausible UX concern; behavior proven, impact unobserved.** [app.py:1460](/home/d/code/hn-rerank/clients/tui/src/hn_rerank/app.py:1460). Entering zoomed reading disables feed polling; `j/k` navigation keeps that mode active → prolonged reading can retain old decks/counts without an update notice. **Minimal next step:** physically check prolonged zoom navigation before deciding whether to add a pending-update indication.

The supplied **17:51 UTC `LIVE.json` snapshot**, not new measurements, records local feed response **10.68 ms** and rerank median/p95 **15.15/25.35 s**. These establish neither public-network nor physical TUI latency.

Freshness is layered: regeneration waits an hour between cycles; Reddit refresh starts are spaced at least two hours; hot-count checks wait ten minutes and ranking idle. Publication age, stored `fetched_at`, and deck version do not establish upstream freshness. No distinct new archive-delay or refit defect was established.

Scope: independent Codex review only; no other reviewers invoked, mutations, tests, providers, or runtime actions. Narrow/wide layout, focus, vote/undo and cancellation received limited source/test inspection; physical rendering remains unknown. The approximate line budget was exceeded; targeted read calls stayed below 16. ML trial state remains untouched.
