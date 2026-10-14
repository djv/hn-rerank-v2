# Reader reliability and normal-use check — 2026-10-14

User authorized: verify comments opening, check quit/save and summary retry in a real terminal, and assess several normal Queue sessions. Checkout inspected: 7a1f4e7. No production code or runtime changes; no live feedback or fabricated live interactions.

## Comments opening

- Live read-only DB check: profile 151 has 7 article_open events, but zero comments_open events. The user answered that comments opening has not been checked yet. Natural-use comments opening and live receipt remain unverified.
- Current action_open_url calls the opener for a valid discussion URL and sends comments_open with the selected card's origin context. In an actual tmux PTY, pressing c invoked the fixture opener with the expected discussion URL and emitted comments_open carrying original Recommended position 0/version 0, surface=queue, queue_position=0.
- The fixture opener records its URL instead of starting a browser. This proves the client control and request payload, not the user's browser or production delivery. No live profile/test profile was created or mutated.
- Current-backend interaction storage/parser checks passed on an isolated VPS checkout; source hashes matched current server.py and database.py. Still not a live comments-open action.
- Remaining next action: during ordinary reading, open one story's discussion with c, then read back live comments_open. Do not use test telemetry as a natural reading signal.

## Terminal control check

A detached, agent-owned tmux terminal ran the real Reader event loop against the existing fake API fixtures. Input went only to this fixture session, never the user's reader or another agent. Terminal captures inspected at 120x35, 80x24 and 41x33. This is actual PTY rendering, not a desktop screenshot or user-observed display check.

- Queue order: [1,3,2,5,4,6], origins Recommended, Explore, Recommended, Popular, Explore, Popular. No duplicate story slots.
- First summary deliberately returned HTTP 500: current card remained and rendered the error plus manual retry instructions. Pressing r fetched the successful summary and displayed Summary 1.
- Narrow layouts retained sort/window controls, selected headline, reading pane and status/help. Wide layout kept list/reading panes.
- Exhausting the six fixture cards rendered the Caught up summary state, with no selected story and no remaining IDs. State was recorded directly from the running app; the separately captured final terminal text raced ahead of the redraw, so it is not used as proof of the final screen.
- Each fixture vote took 1.5 seconds. Pressing q immediately after the last vote waited for the serialized save. All six votes reached the fixture; the last vote_saved preceded exit, quit_report was empty, and the temporary tmux session exited.
- No bug reproduced. Pending/failure/second-q variations are covered by existing focused tests; no production changes needed.

Verification on VPS (laptop was on battery):
- Current standalone client: test_dwell.py, test_medium_reliability.py, test_queue.py, 70 passed in 18.03s.
- Current backend: test_database.py and test_server_properties.py selected by interaction, 5 passed in 8.90s.
- An earlier backend run used an older isolated tree and is not counted as current verification. The current run used hash-matched source.
- Full suite/lint/types were not repeated for this documentation-only result.

## Normal-use assessment

Three completed October14 sessions (14:01–14:10, 16:28–17:05, 17:06–18:31 UTC) had 75 Queue impression events, 48 distinct stories and 17 source labels. Repeated impressions are not independent card ratings. Section counts: Recommended 45 events, Explore 21, Popular 9. Two sessions reached Queue positions 17 and 23 respectively; a third moved between views and windows. These are exposure observations, not causal comparisons or objective topic diversity.

All 45 Recommended impressions and all 9 Popular impressions matched positions in stored deck_scores. Explore had 15 raw-position differences: deck_scores records uncapped window views, whereas serve_window caps each badge group, filters age and re-sorts the served Explore list (pipeline/ranking.py:2425 onward; server.py:_record_deck_scores). Raw Explore position therefore is not a direct served-order oracle; this comparison does not establish a client rank bug. No historical telemetry was rewritten. Historical served Explore order cannot be fully reconstructed from this comparison alone.

User observations collected in this task:
- Queue is more varied and worth reading.
- Caught up was reached and felt like a useful stopping point.

That establishes the user's current subjective assessment. It does not establish long-term diversity, a trial winner, or an optimal section balance. Keep the existing mix; no tuning is justified by this small observational sample.
