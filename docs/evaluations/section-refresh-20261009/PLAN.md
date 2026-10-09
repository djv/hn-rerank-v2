# Requested implementation plan

User choices, 2026-10-09: background refresh on s/h/l; r is only selected-summary
regeneration plus a background real points/comments check; update stats and ask
before regeneration when new discussion comments exceed the current summary's
known source snapshot. No deck-order popup. Independent Codex plan review
authorized before implementation. Implementation is complete and saved; final isolated gates and remaining boundaries are in
[IMPLEMENTATION-CHECKS.txt](IMPLEMENTATION-CHECKS.txt).

1. Sort switching stays immediate and uses the existing shared window feed.
   Schedule an existing-feed GET in the background on a genuine sort change,
   excluding initial mount and selector/tab echoes. One request in flight per
   window; repeated changes request at most one bounded follow-up. Keep existing
   stories/text/selection/scroll usable while awaiting it. Apply responses only
   for the same profile and valid window/request generation; handle restart
   version decreases separately from stale older requests. Avoid cancelling
   and restarting a network fetch on every keypress or an indefinite retry loop.
   Existing optimistic vote/undo state must survive a concurrent refresh.
2. r regenerates only the selected summary. It does not reload/reorder the feed,
   clear unrelated summaries, or restore hidden stories. Retain old text while
   awaiting the existing forced TLDR call. Check selected-story stats in the
   background, coordinating completion with the returned summary snapshot to
   avoid a redundant regeneration prompt during r. Automatic section refresh
   never sets force_refresh.
3. Add a small authenticated stats-only API using the existing bounded HN
   Firebase `_probe_live_items` helper. No article/comment hydration, LLM call,
   ranking, profile creation or feedback in this operation. Return typed points,
   comments and explicit live/stored/unavailable provenance. Unsupported RSS or
   archived Reddit IDs must not be sent to Firebase or called live. Keep error
   handling and server compatibility honest; preserve existing display on failure.
4. Carry known comment snapshot metadata alongside cached/generated summary
   text. The marker represents source descendants at hydration, not a claim that
   every comment was included in the LLM input. Current-key cache hits can expose
   a matching source snapshot; stale/provisional fallback and missing metadata
   cannot be treated as authoritative. Keep the protocol backward compatible.
5. Apply new points/comments to the displayed story without changing selection
   or scroll. When the still-selected story has a known older summary snapshot,
   offer a Textual modal naming the story and old/new comment counts, with
   Regenerate summary / Keep current. Escape/default keeps text; a positive
   action triggers only that story's existing forced TLDR call with fresh
   comments. That call currently generates the combined selected TLDR; this
   plan does not add a separate discussion-only generation pipeline. A decline
   suppresses the same count for that story; further growth can offer again.
   Revalidate profile/story when answering. A late answer/check cannot affect a
   different story or reopen a dismissed/profile-setup/teardown screen.
6. Production Reader/API/server regression tests: real repeated keypresses with
   network requests counted at start; held requests, bounded follow-up, newer
   feed/window/profile/teardown races and errors; r isolation; baseline validity;
   growth/decline/re-growth/accept, no-growth/unsupported cases and in-flight r;
   auth, missing stories, fake upstream failures and no provider activity.
   Focused tests, then full TUI and backend suites in isolated VPS copies of
   exact changed source. Ruff/ty/format clean. Preserve existing WIP/trial/DB.

No new dependency, generic job framework, live experiment or deployment is
included. Backend deployment and restart remain a separate action after concrete
code/validation and authorization. A result review is offered if implementation
diverges from the reviewed plan or reveals material risk, under the shared rules.

## Corrections from the authorized Codex review

All five findings were checked against the current source. Required corrections:

- Persist the generation's source-comment snapshot atomically with the TLDR
  cache record. An additive nullable field is acceptable; old records remain
  unknown and must not be backfilled from mutable stories. Preserve existing
  cache/string accessors and bind new metadata to the exact returned text.
- A requested fresh HN discussion must attempt hydration even if stored counts
  are zero. Hydration failure must not be labelled a fresh-comment success or
  write an authoritative new snapshot. Joining a non-forced flight does not
  satisfy a forced freshness request; coordinate one bounded fresh follow-up.
- Repeated r/modal acceptance joins the same forced operation per profile/story
  rather than cancelling it and spending again. Returned authoritative metadata
  invalidates outdated growth prompts.
- Strictly validate Firebase identity/type and nonnegative integer counts for
  the new stats endpoint; null/empty/invalid bodies are unavailable, not live
  zero. Existing helper behavior alone is insufficient proof of live data.
- Give polling and section refresh compatible request ownership/guards. Keep
  optimistic vote targets intact. If fresh deck membership removes the open
  story, retain its reader/scroll or defer order application until navigation;
  count updates and discussion prompts do not require an order popup.

See the actual [Codex review](CODEX-PLAN-REVIEW.md). These are plan corrections,
originally plan corrections. Implementation and final matching-source checks are
now recorded in IMPLEMENTATION-CHECKS.txt. Backend/schema testing used isolated
databases; live deployment remains separately authorized, with existing data
preserved. An independent result review remains pending approval.
