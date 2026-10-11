# Parked: clean Codex summary fallback

Decision: the user requested "put in a note and leave it" on 2026-10-11.
Leave the implementation unchanged. This is a low-priority availability
tradeoff, not a required fix or a blocker by itself.

Target: Muse's uncommitted branch `agent/codex-forum-fixes-20261011`, base
62af20e, `server.py:2988-3001` in that branch. Its Codex stale fallback
requires the stored summary key to equal the current normalized-input key.
That rejects legacy mixed-input summaries as intended, but also rejects a
previously clean summary after the model, prompt or stored post changes.

The availability loss requires **both** a changed key and regeneration that
fails or is unavailable. A normal provider outage/cooldown with unchanged
inputs still serves the exact-match cached summary. Model/provider selection
is configuration-driven; this is not an automatic model switch on each tap.
Changes to the generated RSS count footer alone are removed by normalization
and should not change the key.

Evidence: the supervising agent used temporary in-memory fixture databases,
cached a clean summary under the current Codex key, then stubbed a different
model identity. `_stale_tldr_fallback` returned None for Codex; a generic-source
control retained its stale summary. This proves the conditional code path,
not an observed production incident or an end-to-end outage. No live
occurrence or frequency was established. The original medium rating was
revised to low after discussing those conditions with the user.

One independent Claude cloud review supplied the finding:
https://claude.ai/code/session_01Y5WngjCSN5NmPQTejRg8tG
Review mailbox branch: `r/20261011-113150-hn-forum-result`.
Local full assessment and reproduction evidence were saved under
`/tmp/hn-forum-result-review-av8MqA/`.

Revisit only if this causes an observed reader problem or the user requests
it. A possible future design is a recognizable clean-input version in cache
keys, allowing compatible clean stale summaries while rejecting legacy keys;
that design is neither implemented nor validated. No cache deletion, DB
migration, runtime edit, integration, service restart or rollout accompanies
this note.

The separate LessWrong scope change and two test type errors remain pending;
this decision parks only the conditional clean-summary fallback finding.
