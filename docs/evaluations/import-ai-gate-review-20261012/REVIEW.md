# Review: Import AI Recommended gate (fc36912)

Scope: `scripts/check_import_ai_recommended.py` (gate), `tests/test_import_ai_gate.py`, CLI.patch. I read the dependency code too: `database.py`, `pipeline/ranking.py`, `pipeline/enrichment.py`, `http_fetch.py`, `models.py` and `server.py`. This was a static review only. I ran nothing and did not contact any DB or service.
Verdict: **sound for its stated purpose; no confirmed false-PASS path through the gate's own logic.** One medium issue: window selection uses the publisher date, but the server uses the stored story time. Several smaller limitations follow.

## Confirmed defects

**M1 (medium): windows are chosen by the RSS pubDate, not the time the server ranks by.** Gate `check_issue` at gate:102 and gate:127-129.
- The gate decides which windows to check with `in_window(issue.published_at, …)`, where `published_at` comes from the publisher RSS.
- The server places the story into windows using `stories.time` from the DB.
- RSS ingestion skips any link that is already in `exclude_urls` (enrichment.py:889, :814). That set is most likely the URLs already ingested from HN; the caller isn't in the supplied files, so this is unverified. When an HN crosspost with the exact URL is ingested first, the only row is the HN story, and its `time` is the HN submit time, which is later than the pubDate.
- Example: pubDate is 14h old and the HN post is 3h old. The server serves the story in the 12h window, but the gate skips 12h. The issue could be absent from visible 12h Recommended and the gate would still PASS. This is a **false PASS for the "every age-eligible window" requirement.**
- The reverse case: pubDate is 8 days old and the HN crosspost is 1 day old. The gate returns NOT_APPLICABLE while the dashboard treats the story as fresh. This is defensible under a "fresh by publisher date" rule, but the rule should be written down.
- When the RSS row itself is ingested, `time = timegm(published_parsed)` (enrichment.py:892-898, :951), exactly matching the gate. So the common case is consistent.
- Fix: check the union of windows from `published_at` and every matched row's `stories.time`, or use min/max over the stored times. Keep the pubDate for the freshness test.
- Verify: add a test with only an HN-id row (`time = NOW-3h`, URL equal to the issue URL) and `published_at = NOW-14h`, with the issue left out of the 12h view. Today this PASSes.

**L1 (low): a future-dated entry is dropped, and an older issue is checked instead.** Gate `latest_issue` at gate:78.
- `if at <= now` excludes any entry with a pubDate ahead of local time, which happens with clock skew or a scheduled post. `max` then picks the previous issue, which is usually voted or older than 7 days, so the gate returns **exit 3 (false NOT_APPLICABLE)** instead of evaluating the new issue.
- `check_issue` already guards `0 <= age`, so the filter adds nothing beyond this side effect.
- Fix: select the max over all entries and treat a future date beyond a small tolerance (for example 10 minutes) as FAIL/error, or clamp it to `now`.
- Verify: unit test with `Import AI 501` dated `NOW+30` and `500` dated `NOW-60`.

## Limitations (by design or acceptable, but should be documented)

- **L2: a not-ready feed FAILs immediately, with no wait.** Gate gate:138.
  - Any regen or RSS refresh bumps `_pool_generation` (server.py docstring, `DeckView.ready`). That makes every voter's deck stale exactly when a new issue lands.
  - A manual run soon after ingestion therefore FAILs with "feed not ready" until the warm finishes.
  - FAIL is the safe direction, but it gives operators noisy signal. Suggest an optional bounded poll of `/api/ranking-ready`, or a distinct reason string or exit code for "not ready" versus "missing".
- **L3: the gate's DB is not cross-checked against the server's DB.** Gate gate:99-126.
  - Applicability (votes and events) is read from `--db`. The feed comes from whatever DB the server at `--base-url` uses.
  - With a stale copy or the wrong path, the gate can call an issue "unvoted" while the server already filters it out (`deck.without(voted)`), giving a false FAIL. The token can also authenticate as a different profile on the other DB.
  - Cheap guard: call `GET /api/user` first and require the returned `user_id` to equal `--user-id` (it is loopback-only, so the token is not exposed).
- **L4: `/api/feed` is not side-effect free on the server.** server.py `_deck_for_user`.
  - A stale deck queues `_trigger_warm` (CPU, and presumably the `deck_scores`/`interleave_decks` writes).
  - This is the same as a normal read, and profile history (feedback, interaction_events) is untouched. The test at test:113 confirms there are no event inserts. Still, "read-only" should mean "no history writes", not "no server work".
- **L5: URL identity is exact-string only.** Gate gate:107.
  - An HN crosspost of the same article that differs by `http`/`www`/trailing slash/query string gets a different row that the gate ignores. Votes or impressions on it don't make the issue NOT_APPLICABLE, and its position doesn't count.
  - Result: false FAIL or a missed NOT_APPLICABLE, never a false PASS. This matches the stated "exact publisher URL" contract.
- **L6: "visible 8" is `orders["recommended"][:8]` from the API (12 served).**
  - That is correct as long as the web/TUI clients render orders verbatim with no client-side drops (for example, an optimistic hide after a local vote). I had no client rendering code to confirm this.
  - Verify: one manual comparison of the UI's first 8 cards against the API's `orders[:8]`.
- **L7: exit 2 on argparse errors.** gate:166 `parser.error`.
  - This is a fourth code that the docstring (gate:3) doesn't mention. Wrappers must treat anything other than 0 or 3 as failure. Document it.
- **L8: the publisher fetch runs before the DB and user are validated.** gate:171-172. A bad `--user-id` or `--db` still makes one outbound RSS request. Harmless, but the checks could go first.
- **L9: an empty `checks` list would PASS** (`all(())`, gate:138). This can't happen today because any age in [0, 7d] is inside 1w and 1m. Adding `checks and …` makes it robust if `WINDOW_SECONDS` changes.

## Security and transport (no defects found)

- **Profile cookie stays on loopback.**
  - The host allow-list rejects userinfo, paths, queries and fragments (gate:157-168).
  - `trust_env=False`, so env proxies can't divert the request, and `follow_redirects=False`.
  - `raise_for_status` raises on 3xx, so a redirect FAILs rather than being followed.
  - The token never appears in the JSON output (`Issue` and `WindowCheck` only), and test:114 checks the repr.
- **Publisher fetch carries no credentials.** It goes through the shared `fetch_with_urllib_fallback` (httpx, then urllib on 403/503 or a transport error). Proxies and redirects are fine there because only a public User-Agent is sent.
- **Error handling.**
  - Non-200 → RuntimeError → exit 1 (gate:91).
  - A transport error from the urllib fallback propagates → exit 1.
  - An HTML challenge page → no entries → ValueError → exit 1.
  - Errors cannot become exit 3. Confirmed.
- **Read-only DB.** `Database(read_only=True)` opens with `mode=ro` (database.py:273-275) and skips schema creation. A missing DB file raises → exit 1. `execute()`'s `with conn:` commit is a no-op.
- **Applicability rules match the spec.** Any feedback row counts, including `neutral` (CHECK at database.py:690). Any `interaction_events` row counts as seen. Both are filtered by `user_id`. Every story id with the exact URL is checked.
- **No new third-party dependency.** `feedparser` and `httpx` were already in use.

## Tests (test realism)

- **Strong:**
  - They use the real Flask `/api/feed` over WSGI.
  - They cover the slot 8 vs 9 cutoff, not-ready, other-profile vote isolation, neutral vote, impression, older than 7 days, missing ingestion, `MockTransport` that forbids network for inapplicable cases, the origin rejection test, and the 503 path.
- **Gaps:**
  1. No test for `main()`'s exit mapping 0/1/3 or for the `except Exception → 1` path. The "distinct exit codes" claim is only exercised for exit 2. Add a test that monkeypatches `fetch_publisher` and `check_issue` and asserts each code.
  2. No duplicate-id (crosspost) test where the vote or the position is on the *second* id. That is the core of the "shared status" claim.
  3. No M1 or L1 cases.
  4. The fixture puts all 12 stories into all four windows at `NOW-60`, so window selection (`in_window` skipping) is never exercised. An issue 2 days old should produce 2 checks (1w, 1m), and the test always asserts 4.
  5. `monkeypatch.setattr("time.time", …)` patches the module globally; this is fine for the test but unrelated to `now`, which is passed explicitly.
  6. The "malformed publisher data" test only covers a missing date. Add an off-host link (for example `https://feeds.example/…`) to confirm it raises instead of being skipped.

## Suggested verification before relying on PASS

1. Fix M1, or at least document it, and add the HN-row-only test.
2. Add a `main()` exit-code test and a duplicate-id test.
3. On the VPS, run once with `--db` pointing at the live DB the server uses (L3). Compare `/api/user` against the expected id.
4. When the next issue lands and is still unseen, run the gate before the profile opens the dashboard, so the live PASS path is recorded. Only the exit 3 path has been observed live.
