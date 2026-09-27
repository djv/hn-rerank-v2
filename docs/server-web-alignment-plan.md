# Plan: align the server and web dashboard with the TUI model

Status: approved 2026-09-26, in progress. Done: S1 (also removed
`/api/deck-cards`, planned for S3, and fixed a stale-deck gap the state
machine found; see WORKLOG), S4 (vote-triggered regen dropped), S2, S3. Server first, then web, then one
contract cleanup; one commit per stage, tests green at every commit. Nothing
is deployed until the end, and the VPS deploy waits for your OK.

The TUI model this converges on (see WORKLOG 2026-09-26 "TUI simplification"):
one summary request per story, cache-first speculation, one feed reload
path, one 60s version poller, optimistic ordered votes that revert on failure
and are never retried, 10s request / 150s generation timeouts.

## Part 1: server (no client-visible contract change)

### S1: one deck decision, monotonic versions, idempotent votes

Today `/`, `/api/feed` and `/api/ranking-ready` each decide "which deck, which
version, should a warm run" separately, and they disagree:

- `ranking-ready` warms and caches a personalized deck for a **0-vote** user
  (render serves the shared cold deck as current). Every 60s TUI poll from a
  new user adds them to `_decks`, and every regen re-ranks them.
- A user with votes and no deck gets `build_cold_deck(user_id=...)` (a full
  gravity rank of the pool) on **every** page load until the warm lands.
- `/api/feed` renders the whole Jinja page to get the feed, then recomputes
  `target_version`/`ready` on top.
- Versions restart at 1 when the process restarts, so a client can see the
  same number for a different deck, and `ready = cached >= min_version` is
  never true for a pre-restart `min_version`. (The TUI works around this
  with `!=`; the web poll just times out.)
- Repeating the same vote bumps the version, queues a warm, and restarts the
  global 300s regen timer.

Changes (all in `server.py`, plus one `database.py` return value):

1. `Handler._deck_for_user(user) -> DeckView(ranked, version, current)`, the
   only place that picks the deck and queues a warm. Rules: cached deck →
   `(deck, deck.version)`; no votes → `(shared cold deck, current)`; votes
   but no deck → `(shared cold deck minus voted ids, 0)` + warm. `/`,
   `/api/feed` and `/api/ranking-ready` all call it (`ranking-ready` via a
   cheap status-only variant that never builds a deck). A 0-vote user is
   never warmed or cached.
2. `/api/feed` builds `Feed` directly (`_build_dashboard_cards` +
   `prepare_feed`), no HTML. The page render uses the same `Feed`.
3. The feed never contains a story the user has voted on (filter by the
   user's voted ids at build time). Clients then only hide their own
   in-flight votes, and the web client can drop its localStorage voted-id
   list.
4. Boot-epoch versions: `_pool_generation` starts at boot time in
   milliseconds, so versions only ever increase, across restarts too. Still
   0 for the cold deck; tests keep pinning the generation to 1. JS numbers
   are exact up to 2^53, far above this.
5. `upsert_feedback` returns whether anything changed. A repeat vote with
   the same action changes nothing: no version bump, no warm, no regen
   timer, and it answers with the current `target_version`.

Tests:
- **Stateful Hypothesis model** (`RuleBasedStateMachine`, new
  `tests/test_deck_state_machine.py`): rules for vote / repeat vote / clear /
  regen (`_pool_changed`) / restart / warm finishes / read page / poll
  ready, with the warm scheduler run synchronously. Invariants: versions
  never decrease for a user (including across restart); a served deck's
  version is ≤ current; `ready` ⇔ served version ≥ current; page, feed and
  ready report the same version for the same state; a 0-vote user is never
  in `_decks`; a pending warm always lands at current; no voted story is
  ever served.
- Example tests for the repeat-vote no-op and the voter cold deck.

### S2: one summary generation per story

`_handle_flask_tldr_detail` (430 lines) has no in-flight dedup: two taps, a
web prefetch and the TUI on the same story each take a generation slot and
quota, hydrate the article again, and call the LLM again. The server's own
warm prefetch (`_prefetch_tldrs_for_ranked`) can race them too.

Changes:
1. Split the handler into gates (cache hit, cooldown, session, slot,
   quota) and one `_generate_tldr_reply(runtime, story, flags) -> TldrReply`
   (typed status + payload) covering hydration + LLM + caching.
2. A per-story flight table (`story_id -> Flight(Event, reply)`) around that
   function. A request that finds a flight waits for it (≤150s, the client
   timeout) and returns the same reply without taking a slot or quota; a
   flight that ends in a retryable error or no-content is shared once, never
   cached. Server prefetch skips stories that are in flight; a tap that joins
   a prefetch flight re-reads the cache when it lands.
3. `force_refresh` joins an in-flight generation instead of starting a second
   one (the in-flight one is fresh by definition).

Tests: N concurrent requests for one story (threads, stubbed hydration + LLM
that blocks on an event) make exactly one LLM call and all get the same
payload; slot and quota are taken once; an error reply is shared but a later
request starts a new flight; a prefetch never runs a story a tap is
generating. Existing TLDR tests stay as they are.

### S3: small fixes and property tests

1. One `_wants_background_tasks(config, ranked)` predicate: the spawn check in
   `_run_warm_attempt` ignores `tldr_prefetch_date_top_n`, which the task
   itself uses.
2. Remove `/api/deck-cards`, `_extract_cards_fragment` and the
   `<!--cards:start/end-->` markers (no caller since the web refill moved to
   `/api/feed`).
3. `prepare_feed` properties (`tests/test_render_properties.py`): every id in
   an order is a feed story; each order has no duplicates; Date orders are
   time-descending and identical for both ages; Recommended/Popular/Explore
   only contain stories whose memberships include `<age>_mixed`; Explore is a
   permutation of its member set; `Feed.parse(feed.to_dict())` round-trips.

### S4: does a vote need a global regen?

Every vote restarts a process-wide 300s timer; when it fires, a full regen
runs (ClickHouse fetch, cold deck rebuild, generation bump, every cached
user re-ranked). `fetch_candidates_only` reads all users' feedback ids/urls,
so find out what that changes in the pool (exclusion? archive seeding?) and
measure what a vote-triggered regen adds over the 4h cadence. Then either
drop the timer (a vote already triggers the user's own warm) or keep it with
the reason written down. Decision goes to you with the evidence.

## Part 2: web dashboard (`templates/index.html`, `pipeline/render.py`)

The web client has two card renderers (Jinja `story_card.html` for the first
page, JS `feedCard` for refills), re-sorts cards client-side instead of using
`feed.orders`, refetches the whole feed on every tab click, and has three
overlapping refresh mechanisms (refill loop, warm-poll loop with 0.4–2.5s
backoff for 30s, vote idle timer) plus a page-load special case.

### W1: render from the embedded feed

- The page embeds the feed: `<script type="application/json" id="feed">`
  with Jinja's `tojson` (escapes `<`, so a title containing `</script>`
  can't break out). Cards are built only by `feedCard`;
  `components/story_card.html` and the sentinels go.
- Client state is `feed` (+ `voted`, `history`, `restored` like the TUI).
  `render()` shows `feed.orders["<sort>:<age>"]` minus voted, capped at 12,
  keeping the active card. Explore shuffles client-side with kept positions
  (TUI `view_order`). Tab changes re-render locally, no fetch.

### W2: freshness and votes

- One `reload({manual})`: GET `/api/feed`, replace `feed`, re-render, keep
  the active card and its summary. `s` / refresh button = manual.
- One poller: every 60s and when the tab becomes visible, GET
  `ranking-ready` for `feed.ready ? feed.version : feed.target_version`;
  reload when the TUI rule says so. Removed: `queueRefill`/`runRefillLoop`,
  `runWarmPollLoop`/`waitForRankingReady`, `scheduleVoteRefresh` idle timer,
  `scheduleDeckRefresh`, the page-load version check, localStorage voted ids.
- Votes: hide + advance at once; one queue for all votes and undos (today
  it's one chain per story, so votes on different stories can reorder);
  success marks the feed stale (`target_version`); failure puts the story
  back with an error and is not retried. Fixes the card that stays faded
  when a vote fails within the 150ms removal window.
- Undo = last entry of `history`, same queue, restored story kept until the
  ready deck lands.

### W3: summaries

- `summaries: Map<id, text>`, `requests: Map<id, Promise>`: one request per
  story shared by the open card and prefetch.
- Upcoming cards: GET `/api/tldr-cache/<id>` only; generate for the active
  card and the next 3 (TUI default), at most 4 at once; 60s cooldown after a
  miss or failure; `force` (t / re-summarize) bypasses both.
- `fetch` with `AbortController`: 10s for feed / cache / vote / ready, 150s for
  generation.

Tests: rewrite `tests/test_client_js.py` for the new functions under Node
(vote order across stories, revert on failure, undo, poller reload decisions
over a table of (ready, current, version, target), shared summary request,
cache-only prefetch, explore order stability, view cap and backfill). Real
render + BeautifulSoup: the embedded feed parses and survives a hostile
title. Update the structural checks in `tests/test_server.py`.

### W4: keys match the TUI

j/k next/previous card, 1/2/3 vote up/neutral/down, u undo, o/c open
article/comments, y copy link, r reload (and regenerate the open summary,
like the TUI's r), s/l next sort, h previous sort, ? help, b badge legend.
Web-only keys that don't collide stay: a toggles age, f toggles the panel.
Tab labels, vote-bar hints and the first-time tip are updated to match.

## Part 3: contract cleanup (C1, after W4)

- `ranking-ready` returns `{ok, ready, current_version}`; takes
  `min_version`, ignores `target_version` (installed TUIs send it), drops the
  legacy `version=` alias and the echoed/cached fields.
- `feedback` returns `{ok, target_version}`.
- Docs: ARCHITECTURE, `docs/freshness-lifecycle.md`, WORKLOG, STATUS.

## Verification and deploy

Per stage: affected tests, then `uv run pytest tests/ -n 4` (keep it near
the 12s target), TUI tests `-n 8`, ruff, ruff format on touched files,
`ty check`. At the end: restart the local service, smoke `/`, `/api/feed`,
`ranking-ready`, cached and uncached `tldr-detail`, a vote + clear, the TUI
against it read-only, and a browser pass on the dashboard; journal scan.
Then ask before deploying to the VPS.

## Decisions (2026-09-26)

- Web polls every 60s like the TUI, plus when the tab becomes visible.
- Web keys match the TUI (W4 below).
- The vote-triggered global regen gets investigated as S4.
- Work on `main`, one commit per stage, pushed as each goes green.
