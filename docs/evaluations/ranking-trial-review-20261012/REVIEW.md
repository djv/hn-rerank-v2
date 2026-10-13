# Independent review: HN ranking three-arm interleaving trial (interim, 2026-10-12)

Scope: source bundle at `main` ae98d0d, which is the files under `files/code/hn-rerank/`. Paths below are relative to `code/hn-rerank/`. I read `pipeline/interleave.py`, `scripts/interleave_report.py`, `scripts/simulate_interleaving.py` (`compare`), the deck and far-card code in `pipeline/ranking.py`, deck recording, versioning, vote and event ingestion in `server.py`, the `feedback`, `interleave_decks` and `interaction_events` schemas in `database.py`, both test files, `INTERLEAVING.md` and the saved `report.json`. I did not use runtime, databases or credentials. The runtime (b1897f4) is not verified here.

I recomputed the report's statistics from its own counts. The two-proportion p-values match the report: `joined_all` p = .191 and `joined_no_metadata` p = .355. The sign-test p of 1.0 is correct for 4 wins and 3 losses. The arm totals add up: 23 + 20 + 18 = 61 credited. The skip counts also add up: 61 + 123 + 13 + 1 = 198. The horizon arithmetic is correct: T0 + 2,419,200 s = 1793851642.75 = 2026-11-05 04:07:22 UTC = 2026-11-04 23:07:22 EST, since US DST ends on 2026-11-01. The calibration window is 2026-10-11 23:58:32 to 23:59:40 UTC. The report's `until` is 4.52 days after T0.

## Verdict

**Interim interpretation: supported after two wording changes.**
- "Trial ongoing; 61 credited votes across 16 decks; no winner established; continue until the fixed endpoint" is supported.
- "Challenger #2 looks promising" overstates the evidence. The observed up-rates are 10/20 = 50% against 7/23 = 30%. The difference is +19.6 pts, two-sided p = .19, with a 95% Wald interval of about −9 to +48 pts. That is fully consistent with no difference.
  - Suggested wording: "#2 is directionally ahead on a very small sample (20 vs 23 credited votes); no evidence either way yet."
- "Safety checks passed" is true under the written rule, with caveats.
  - #4's down-rate is +4.1 pts (7/18 against 8/23, one-sided p ≈ .39). #2's down-rate is lower than production's.
  - The median rank time is 16–20 s, under the 30 s limit.
  - The safety computation is ad hoc: the CLI has no down-rate test and no rank timing (F3).
  - At about 20 votes per arm, the rule "≥10 pts and p < .01" has almost no chance of triggering. "Passed" means "no alarm", not "shown safe".
  - Day 4.5 is not one of the preregistered weekly checks. That is fine for a descriptive look.

**Final-decision readiness: not ready yet. Everything that blocks it can be fixed in the script before 2026-11-05, without touching runtime, arms or dates.**
- The CLI cannot produce two quantities the preregistered decision needs:
  - the down-rate significance test;
  - the 95% interval of the up-rate difference.
- The CLI has no flag for the calibration exclusion.
- The CLI's window handling differs from the preregistration in two ways: it uses decks from before T0, and its end bound is inclusive.

Nothing found invalidates the data collected so far. The decisive risks are about producing and reproducing the final analysis, plus a sample that will be much smaller than planned (F2).

---

## Findings

Severity: **High** means it could change or invalidate the final decision. **Medium** means a preregistration or implementation deviation with probably small numeric impact that must be fixed or checked before the final run. **Low** covers small bugs, stale wording and robustness issues. **Optional** covers improvements. Each finding is also labelled as a bug, a methodological limitation, stale wording or an optional improvement.

### F1 — High (implementation gap): the CLI cannot apply the calibration exclusion, and `feedback` is mutable, so the final analysis is not reproducible
- **Where:** `scripts/interleave_report.py:158-166` (vote query), `:179-191` (no exclusion flag); `database.py:686-693`, `:1321-1340` (feedback upsert), `:1433-1444` (delete).
- **Evidence:**
  - `load()` selects every `feedback` row with `updated_at >= since`. There is no option to exclude a time range or a list of story IDs.
  - The 12 October report's `raw_votes`/`excluded_revotes` fields and its `rank_timing`/`reading_events` blocks are not produced by `main()`. The snapshot came from a wrapper that is not in the source.
  - Running the documented final command (INTERLEAVING.md:152) at the endpoint would therefore include the 29 calibration rows. `votes` would read 227+, not 198.
  - `feedback` has one row per (user, story). A changed vote overwrites both `action` and `updated_at` (`ON CONFLICT ... DO UPDATE`), and an undo deletes the row. No vote-history table exists; the schemas list none.
  - So the 29 calibration re-votes destroyed the original action and timestamp of those stories' earlier votes. Any later re-vote, undo or calibration also does this, including one made after the endpoint but before the final run. A vote changed after `--until` disappears from the window entirely.
- **Consequence:**
  - **Credited counts:**
    - The credited counts are unaffected only if none of the 29 stories had an interleaved Recommended impression that would have been its "last before vote" for an earlier in-trial vote. The claim "calibration rows had no trial impressions" covers the calibration votes themselves. I cannot verify it for the overwritten originals.
    - If any original was a credited trial vote, it is silently lost and cannot be recovered from `feedback`.
  - **Final result:** the final result depends on when the DB is read, not only on the preregistered window.
- **Smallest correction:**
  1. Add `--exclude-updated-between START END`, repeatable, to `load()`, and echo the raw and excluded counts in the report. Record the exact calibration interval [1791763112.57, 1791763180.12] in an addendum written before the endpoint.
  2. One read-only check now. For each of the 29 story IDs, query `interaction_events` for any `sort_mode='recommended'` impression whose `dashboard_version` is in `interleave_decks` with the story in that window, between T0 and 1791763112.57. If there are none, document that no credit can have been lost.
  3. At the endpoint, take a read-only copy of the DB, or at least export `feedback`, before any further voting or calibration. Run the final analysis on that frozen copy.
  4. Optional: confirm that the CLI with no exclusion still gives `credited = 61` on the October 12 data, so that the excluded rows are shown to be non-credited.

### F2 — High (methodological limitation; doesn't invalidate the trial but changes how to read it): accrual is far below the power assumption
- **Where:** INTERLEAVING.md:25-27 ("about 50 a day"), :49-54 (power table), :63-68; `report.json` (61 credited in 4.52 days).
- **Evidence:**
  - Credited accrual is 61 / 4.52 = 13.5 per day. The power replay assumed about 50 Recommended votes per day.
  - 123 of 198 eligible votes (62%) were made in Popular. Only 61 (31%) reached an interleaved Recommended view, against 438 of 1,153 (38%) in the October 7 snapshot. Volume is also lower.
  - Extrapolated, 28 days gives about 380 credited votes, about 125 per arm. That is roughly the replay's 7-day row: power about .25 for the #2 rate test and about .12 for #4. The replay is itself labelled optimistic (selected arms, frozen models, fully judged blocks).
- **Consequence:**
  - Outcome 2 ("no detectable difference at this power") is the most likely result regardless of truth. The decision rule already handles it, so this is not invalidating.
  - Because outcome 2 is likely, the up-rate interval (F3) becomes the main deliverable.
  - The observed +20 pt gap for #2 is about twice the replay's +9 pt. Such early leads usually shrink.
- **Smallest correction:**
  - No change to arms or dates; the horizon is fixed.
  - State the projected n and power in the interim note.
  - Asking the user to keep using Recommended normally is fine, since arms are blinded and the request does not favour any arm. Do not ask them to vote differently.

### F3 — High (gap between implementation and the documented decision): no down-rate test and no up-rate interval
- **Where:** `scripts/simulate_interleaving.py:160-193` (`compare` returns only point down-rates); `scripts/interleave_report.py:118-147`; INTERLEAVING.md:156-170.
- **Evidence:**
  - Outcome 1 requires "no significantly higher downvote rate".
  - The safety rule requires a down-rate ≥10 pts higher with p < .01.
  - Outcome 2 requires "the 95% interval of the upvote-rate difference".
  - The CLI outputs none of these. There is no down-rate p-value and no interval of any kind. The test for "significantly higher downvote rate" in outcome 1 is also not specified: its sidedness, alpha and Bonferroni treatment are missing.
  - As written, the final decision would have to be completed by hand after the data are seen, which invites analytic flexibility.
- **Consequence:** outcomes 1 and 2 cannot be produced mechanically from the preregistered command. Choosing the down-rate test after unblinding the final data would weaken the confirmatory claim.
- **Smallest correction:**
  - Before the endpoint, write a short addendum fixing the test. A suggested choice:
    - down-rate: a one-sided two-proportion z-test of challenger > production at .025, matching the per-test level;
    - safety: keep the existing rule (one-sided, p < .01, ≥10 pts);
    - up-rate interval: a 95% Newcombe or Wald interval, unadjusted, plus an optional 97.5% interval matching Bonferroni.
  - Add `down_rate_p`, `down_rate_rejects` and `up_rate_diff_ci` to the `summarize()` output.
  - Add a unit test with known counts. For reference, the interim values are: #2 up diff +.196 [−.093, +.484]; #4 +.140 [−.157, +.437]; down #2 −.248 (two-sided p .055); #4 +.041 (p .79).

### F4 — Medium (bug against the preregistration): decks from before T0 can still receive credit
- **Where:** `scripts/interleave_report.py:155` (`get_interleave_decks(user_id, since=0.0)`); INTERLEAVING.md:147-150.
- **Evidence:**
  - The preregistration excludes the first deployment's unbalanced decks (3/2/7 splits). The code filters only votes by time. It loads every deck ever stored.
  - Suppose a vote is cast at or after T0 and its last impression carries a first-deployment version. That can happen when a client is still showing the old deck after the restart; the client keeps logging impressions with the old `dashboard_version`. Such a vote is credited using the unbalanced deck.
  - Versions are boot-epoch-ms plus a counter (`server.py:1852-1855`, `:1884`), so old versions stay distinct and valid in `arm_of`.
- **Consequence:** this is a small, probably zero-to-few-vote contamination with exactly the decks the protocol says to exclude.
- **Smallest correction:**
  - Use `db.get_interleave_decks(user_id, since=since)`. T0 is defined as the first deck row of the fixed deployment, so `created_at >= T0` is exactly the intended set. Add a test with a pre-T0 deck.
  - Before the final run, count the credited votes whose `version` belongs to a deck with `created_at < T0`, to show the impact.

### F5 — Medium (bug, boundary): the end bound is inclusive, but the preregistered window is half-open
- **Where:** `scripts/interleave_report.py:165` (`float(at) <= end`); INTERLEAVING.md:151 ("[T0, T0 + 28 days)").
- **Evidence:** `--until T0+2419200` includes a vote stamped exactly at 1793851642.75. The impact is negligible with float timestamps, but the code does not match the stated rule.
- **Smallest correction:** use `< end`, document `--until` as exclusive, and add a boundary test.

### F6 — Medium (risk to attribution; needs a check, not a code change): impression times come from the client, vote times from the server, with 1 s slack
- **Where:** `scripts/interleave_report.py:37`, `:87-95`; `server.py:2798-2806` (client `occurred_at`); `database.py:1333` (`time.time()` for votes).
- **Evidence:**
  - `occurred_at` is client-supplied, while `feedback.updated_at` is server time. "Last impression before vote + 1 s" assumes the clocks agree to within 1 s.
  - If the client clock runs ahead, the impression of the voted card can sort after the vote. The vote then falls back to an older impression from another version or view, or to `no_impression`.
  - If the client clock runs behind, impressions logged after the vote can count. The impact is limited because voted stories are never re-served.
  - The interim data has only one `no_impression`, so large skew is unlikely, but this has not been measured.
- **Smallest correction:**
  - Run a read-only diagnostic: the distribution of `received_at − occurred_at` in `interaction_events`, which already stores server `received_at`, and of `vote.at − last_impression.at` for credited votes.
  - If the skew is above about 1 s, document it. Changing the crediting rule mid-trial is not recommended unless skew is shown to cause misattribution.

### F7 — Low (bug, robustness): a missing decks table becomes "no decks" without any error
- **Where:** `scripts/interleave_report.py:154-157`.
- **Evidence:** catching `sqlite3.OperationalError` sets `decks = []`. The result is that every vote is skipped as `deck_not_interleaved`, and the reports and p-values look valid. Pointing the tool at the wrong or an old DB at the final run would produce "no difference" instead of an error.
- **Smallest correction:** let the exception propagate, or exit non-zero when `decks` is empty and `since` falls inside the trial.

### F8 — Low (bug, preregistration drift): per-test alpha depends on `--arms`
- **Where:** `scripts/interleave_report.py:127`.
- **Evidence:** `level = alpha / (len(arms) - 1)`. If an arm is safety-stopped and the final run passes only the surviving arm, the per-test alpha becomes .05, not the preregistered .025.
- **Smallest correction:** fix the Bonferroni divisor at 2 (the number of preregistered challengers), or add an explicit `--bonferroni 2`.

### F9 — Low (methodological limitation): the z-test assumes independent votes
- **Where:** `scripts/simulate_interleaving.py:176-181`.
- **Evidence:**
  - Credited votes cluster within decks (about 3.8 per credited deck) and within vote bursts, and every arm refits after every vote.
  - The replay's calibration (.012–.024 against .025) used frozen models and fixed blocks, so it does not cover live adaptivity.
  - Mild anti-conservatism is possible.
- **Correction (optional, not decisive):** report one sensitivity analysis next to the primary test, for example a deck-cluster bootstrap of the up-rate difference. Do not change the primary test.

### F10 — Low (methodological limitation): the up-rate denominator is the number of votes, not exposure
- **Evidence:**
  - The primary metric is up-votes / credited votes. An arm whose cards are skipped more often (seen, never voted) is not penalised, and an arm that attracts more neutral "dismiss" votes is.
  - In a swipe deck where most visible cards are voted, this is close to per-impression. It has not been checked.
- **Correction (optional):** report, descriptively, credited impressions per arm and the vote rate per impression. If draft balance holds, impressions should be about 1/3 per arm.

### F11 — Low (stale wording): the INTERLEAVING.md first-deployment window contradicts T0
- **Where:** INTERLEAVING.md:149 ("03:56–04:3x UTC Oct 8") against :150 (T0 = 04:07:22 UTC, the first deck of the fixed deployment).
- **Evidence:** the first-deployment decks must end before 04:07:22. "04:3x" is a typo or stale. The heading "Live design (implemented, off by default, not deployed)" (:12) is also stale.
- **Correction:** fix the wording only. Don't change T0.

### F12 — Low (stale wording, plus an optional product point): the far card is at served slot 12 but clients show 8
- **Where:** `pipeline/ranking.py:320-323`, `:2373-2421` (far card at `VIEW_SIZE - 1` = index 11), `:2445-2452`; `pipeline/interleave.py:8-10`.
- **Evidence:**
  - With `VIEW_SIZE = 12` and 8 visible cards, the far card is visible only after the user has voted about 4 cards from that deck.
  - Crediting is correct: the far card is stored as `"far"`, skipped as `far_card`, and outside the draft.
  - The visible 8 are draft-order picks 1–8 (2 full rounds + 2), balanced in expectation, as `interleave.py`'s docstring says.
  - The docstring wording "the far card takes the last served slot" is accurate but no longer means "last visible slot". No impact on the trial.
- **Correction:** none for the trial. Clarify the comment.

### F13 — Optional (test gaps)
`tests/test_interleave_report.py` does not cover:
- the `far_card` skip;
- the slack boundary exactly at `vote.at + 1.0`;
- decks created before `since` (F4);
- the `until` boundary (F5);
- the empty-decks error path (F7).

`team_draft` has property tests covering balance and order, which is good. Impressions sharing a timestamp are ordered in DB order (`sorted` is stable over an un-ordered `SELECT`). Adding `ORDER BY occurred_at, received_at, event_id` at `:167-175` makes the run deterministic.

---

## Attribution: what was checked and found correct
- **Latest impression:** credit uses the last impression up to the vote time plus 1 s, from impressions back to `since − 1 day`. Earlier truncation cannot change which impression is last. View, deck version and window all come from that impression (`:87-114`). A story seen in an interleaved deck and later in Popular is correctly not credited. This applies to all arms equally.
- **Changed and undone votes:** undone votes disappear (row deleted). A changed vote counts with its final action, its final time, and the last impression before that final time. This is consistent with the documentation but leads to the mutability issue in F1.
- **Deck version and window:**
  - Versions only increase across restarts (boot-ms generation).
  - `_record_interleave` stores the deck before it can be served (`server.py:2135-2137`).
  - Decks that are production-only after a fit failure are not stored, so they are skipped as `deck_not_interleaved`. This handles the October 8 incident correctly.
  - The PK `(user_id, version)` together with `INSERT OR REPLACE` is safe, because a version cannot be reused.
- **Hidden and visible cards:** all 24 drafted picks plus the far card are stored per window. Credit requires an impression, so hidden picks earn credit only once they are actually shown.
- **Draft after dedup and the shared vote snapshot:** drafting runs on the post-drop pool (`ranking.py:2349-2370`), and every arm trains on one `training_feedback` snapshot (`:2507-2513`). Both match the documented changes.
- **Blinding:** in the source I read, the arm reaches only `deck_scores` and `interleave_decks`. I did not find it in any client serialization. Templates are not in the bundle, so this is not fully verified.

## Distinguishing categories
- **Implementation bugs or gaps to fix before the endpoint:** F1 (exclusion flag and frozen snapshot), F3 (down test and interval), F4 (pre-T0 decks), F5 (inclusive end), F7, F8.
- **Methodological limitations to state, not fix:** F2 (power), F6 (clock basis; check it), F9 (independence), F10 (denominator), plus the known limits: one user, Recommended only, a user who knows the trial is running, and arms selected on earlier labels.
- **Stale wording:** F11, F12, the INTERLEAVING.md "not deployed" heading, and the "looks promising" phrasing.
- **Optional:** F13, plus the sensitivity analyses in F9 and F10.

## Recommended interim statement
> Trial ongoing (day 4.5 of 28). 61 credited votes across 16 decks (production 23, #2 20, #4 18), after excluding 29 calibration re-votes. Credited up-rates: #2 50%, #4 44%, production 30%. Neither difference is significant (p = .19 and .36 against a per-test α of .025), and both intervals include zero. No winner, and none will be selected before the endpoint. Safety rule not triggered: no challenger's down-rate is ≥10 pts above production's, and median rank time is 16–20 s. Accrual is about 13.5 credited votes a day, so the final sample is likely about 125 per arm, with low power. Continue ordinary use until 2026-11-05 04:07:22 UTC. The report script needs F1, F3, F4 and F5 fixed and recorded in a pre-endpoint addendum before the final analysis.
