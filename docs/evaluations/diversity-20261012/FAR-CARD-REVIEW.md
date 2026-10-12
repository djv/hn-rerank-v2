# Review: Recommended "far card" (hn-rerank-v2 `far-card` @ 2a5fcfc)

Independent read-only review of `files/tmp/ccr-farcard/far-card.diff`, using
the full-file context in `files/code/hn-rerank-farcard/`. Line numbers refer
to the post-change files in the bundle.

**Verdict:** safe to deploy with respect to the trial constraints. I found no
high-severity defects. Two medium findings change what the user sees: the far
card can disappear while eligible far stories are still current, and profiles
without up-votes get a mislabelled far card. The rest are low-severity
correctness, reporting and test-coverage gaps.

---

## Findings

### M1 — If the chosen far card ages out between warm and serve, Recommended serves no far card, even when other eligible far stories are still current (medium)
`pipeline/ranking.py:2407-2411` (`place_far_card`) and `:2433-2439` (`serve_window`)

`place_far_card` chooses one far story at warm time and tags only that story
`FAR_ARM`. The other eligible tail and Discover candidates keep no record of
being eligible. In `serve_window`, if that one story has left the window, `far`
is `None` and the code falls back to `cur[:VIEW_SIZE]`, which is a plain
12-card Recommended.

- **Failure scenario:** 12h window, warm at T. The best far story X was posted
  at T−11h50m. A second far story Y, posted at T−1h, sits at tail position 14.
  The deck stays cached for more than 10 minutes, because no vote or regen
  happens in that time. At T+15m, X has aged out, and the user gets no far
  card even though Y qualifies. That breaks the objective: a far card should
  be missing only when no eligible story exists.
- **Most affected windows:** 12h and 1d. In those windows the best-scored far
  stories tend to be old: score correlates with points, and points correlate
  with age.
- **Fix:** tag every eligible candidate with a "far-eligible" flag (or keep a
  ranked list of them), and pick the first current one in `serve_window`. The
  alternative is to make `serve_window` re-run the selection over current tail
  and Discover cards. Whichever story is chosen gets `arm=FAR_ARM` at serve
  time. A non-chosen candidate must keep its drafted arm.
- **Trial impact if fixed at serve time:** `_record_interleave` stores the
  warm-time arms. A serve-time far card that was a drafted tail card would be
  recorded under a trial arm even though it was served as "far". So the
  warm-time approach is cleaner: store an ordered list of fallback far
  candidates, each already re-tagged `far` and removed from the drafted tail.
  The cost is that those candidates leave the draft's tail.

### M2 — Profiles with feedback but no up-votes: every story counts as "far", and the far card is just an ordinary card relabelled ✨ Discover (medium/low)
`pipeline/ranking.py:2753-2757` and `:1003-1005`. `place_far_card` is at `:2383-2397`.

With no up-votes, `cand_closest_up` is all zeros: `assemble_ranked_deck` sets
zeros explicitly, and `_knn_mean_and_max` returns zeros for an empty
reference. Every `closest_up` is therefore 0.0 < 0.60.

- **Failure scenario:** a user with only down or neutral votes. The far card is
  simply the best non-head story, which is production card 12. It gets the ✨
  Discover badge and the explainer text "unlike your up-votes", which says
  nothing for a user who has no up-votes. `card_attribution` also hides its
  match title because `is_discover=True`.
- **Interleaved profile with no up-votes:** this would pull a drafted card out
  of trial credit for no diversity gain. Profile 151 has up-votes, so this
  does not affect the live trial today.
- **Same issue elsewhere:** Explore's Discover pass already has this
  behaviour; this change copies it into Recommended.
- **Fix:** have `ExploreContext` or `RankedStory` mark "no up-votes"
  (`closest_up=None`) when the up-vote set is empty, and skip placement.

### L1 — The `deck_scores` reader silently coerces a non-string `arm` (low; conflicts with "no silent failures")
`database.py:1741, 1756-1758`

`arm=raw_arm if isinstance(raw_arm, str) else str(raw_arm)` turns
`"arm": null` into `"None"` and `"arm": 3` into `"3"`. Every other field in
this reader raises `ValueError` on a type mismatch through
`_req_*`/`_opt_*`.

- **Failure scenario:** a hand-edited or buggy row containing `"arm": null`
  loads with `arm == "None"`. A later analysis then counts it as a separate
  arm instead of failing loudly.
- **Fix:** accept a missing key as `""` (the legacy rows), and raise on
  anything else that is not a `str`.

### L2 — `interleave_report` top-level counts include far-card votes (low)
`scripts/interleave_report.py:102, 130, 204`

`credit_votes` stores far votes as `CreditedVote(arm="far")`. `summarize` and
`compare` drop them correctly, through `if c.arm in index` at line 122 and
`credited_by_arm` iterating only the named arms. But:

- The report's `"credited": len(credited)` counts far votes.
- `decks_with_credit` counts deck versions whose only credited vote was on the
  far card.

**Failure scenario:** the 2026-11-05 report shows `credited` greater than the
sum of `credited_by_arm`, and `decks_with_credit` overstates the trial's
effective n. Neither feeds the tests, but both appear in the headline numbers
the user reads.

**Fix:** in `credit_votes`, route `arm == FAR_ARM` to `skipped["far_card"]`.
Alternatively, compute both counts over named arms only.

### L3 — Trial exposure: no bias found; slightly more variance per deck (low, informational)
`pipeline/ranking.py:2380, 2439`; `pipeline/interleave.py:62-86`

**No bias in expectation:**
- With 3 arms, 12 served cards were exactly 4 complete rounds. 11 cards is 3
  rounds plus 2 picks of round 4.
- Round 4's arm order is a fresh `rng.permutation`, so each arm has a 1/3
  chance of being the one cut. That is symmetric, so credit stays unbiased in
  expectation. With 2 arms, the cut arm is a fair coin flip.
- The per-deck sign test gets slightly noisier, because one arm has 3 cards
  against the others' 4.
- The module docstring's guarantee ("any prefix that ends a round holds the
  same number of stories from each arm") no longer holds for the served
  prefix. Update the docstring so the 2026-11-05 analysis does not assume it.

**Second-order effect, negligible:**
- The far card is removed from the drafted tail (`rest`). Which tail card gets
  promoted when a head card ages out is therefore shifted by one position,
  and the removed card's arm correlates with arm behaviour, because it is the
  highest-production-score far story in the tail.
- This only matters when head cards age out within the cache lifetime, and
  aging already broke round balance before this change.

**Same deck version and window:** a far story cannot also be stored under a
trial arm. `head_ids` excludes it from candidates, `rest` removes it from the
tail, and the drafted list has unique ids, so `_record_interleave` stores it
exactly once, as `far`.

**Across windows:** the same story can be `production` in 1d and `far` in 1w.
`credit_votes` keys on `(version, window, story_id)` from the impression's
`age_filter`, so credit is correct.

### L4 — For the interleaved profile, "best-scored far story" can miss stories (low)
`pipeline/ranking.py:2293-2299, 2307, 2361-2363`

- With `arm_rankings`, the pre-draft Recommended is the union of every arm's
  top 24 stories, which can be about 40. `draft_recommended` keeps only 24.
- Discover excluded the whole union through `picked`. So a far story that was
  in the union but not drafted is in neither the tail nor Discover, and
  `place_far_card` never sees it.
- **Failure scenario:** a challenger-only story with a higher production score
  than every far story in the drafted tail and in Discover is never chosen.
- The `place_far_card` docstring claims "the Recommended tail plus Discover's
  picks suffice". That is true only for non-interleaved profiles.
- Low: the effect is small, and it ends with the trial.

### L5 — Explainer text is unconditional (low)
`templates/index.html:1395`; `clients/tui/src/hn_rerank/app.py:2693` (diff lines 10 and 219)

"Its last card is a ✨ Discover pick" shows even when no far card exists: cold
deck, no eligible story, M1's aged-out case, or a 1–3-card short window.

**Fix:** soften it to "may be", or render it only when a card with
`arm == "far"` is in the served view.

### L6 — Two smaller correctness points (low)
**`closest_up` after canonicalization** (`pipeline/__init__.py:271`):
`replace(r, story=out.story)` keeps the duplicate's `closest_up` while the
story id is rewritten to the canonical story. For HN dupes this is the same
article, so the similarity is effectively the same. No action needed beyond a
comment.

**`closest_up` survival through every step:**

| Step | Location | Keeps `closest_up`? |
|---|---|---|
| dedup | `__init__.py:1148` | Yes (filter only) |
| canonicalize | `__init__.py:271` | Yes (`replace(story=…)`) |
| draft | `ranking.py:2362` | Yes (`replace(arm=…)`) |
| attribution fill | `ranking.py:2702-2711` | Yes |
| `popular_card` | `ranking.py:2192` | Not a concern: runs before `with_rank` |

`compare=False` keeps it out of equality checks. That matches the
`svm_margin` and `pool_rank` fields, and it means the `deck == …` asserts in
the tests do not check it.

**Stale docstring** (`pipeline/ranking.py:2024`): the `WindowViews` docstring
says "Recommended is in model-score order". It no longer is: the far card can
score below the tail, and it already was not true for drafted decks. Update
it so later readers (and the Phase 0b script) do not assume score order.

### L7 — Evaluation confound for the ~2026-11-02 readout (low, plan-level)
PLAN "Far card", last bullets.

- `place_far_card` removes the chosen story from Explore, and that story is
  often Discover's top pick. Discover's live up-rate measured in Explore from
  now on is therefore the up-rate of picks 2 to 4, not 1 to 3.
- The planned "next to Discover" comparison will then flatter the far card
  against Discover.
- The far card's "cards 9–11" comparison uses `deck_scores.position`, which is
  the cached-deck position. After head cards age out, the served position can
  differ. Impressions carry the served `position`, so prefer that.

---

## Constraint checks: areas with nothing found

| Area | Verdict |
|---|---|
| Far card votes credited to no trial arm | **OK.** Stored as `(sid, "far")`. `summarize`/`compare` skip arms not in `--arms`; the default `--arms` list does not include "far". Only the L2 headline counts are affected. |
| Non-interleaved profiles get no `interleave_decks` rows | **OK.** `server.py:2280` now requires `PRODUCTION_ARM`. Arms are `{"", "far"}` for these profiles, and also for an interleaved profile whose challengers failed to fit, so no row is written. Tested (`test_record_interleave_far_only_writes_no_row`). |
| `"far"` in `InterleaveDeck.arms` | **OK.** Nothing in the bundle reads `.arms` except the test, and the report ignores it. |
| Legacy `deck_scores` rows (no `"arm"` key) | **OK.** `c.get("arm", "")`; tested. Only the non-string case (L1) is a problem. |
| Cold decks | **OK.** `fast_rerank_for_user` returns at `__init__.py:1048`, before `place_far_card`. The shared cold deck never calls it. `closest_up` is `None` without an `ExploreContext`, so the function would do nothing anyway. `serve_window` with no `far`-tagged card behaves exactly as before. |
| Duplicate story ids | **None found.** `head_ids` exclusion, candidate dedup by id, and removal from `rest` and `explore` keep Recommended and Explore unique per window. The property test checks this on served output. Caveat: `canonical_outcomes` is not in the bundle, so I could not verify that two inputs never canonicalize to the same id. That would be a pre-existing issue in any case. |
| Popular overlap and badge merge | **OK.** Popular is untouched. `WindowViews.stories()` visits Recommended first, so the merged card keeps `arm="far"` and `is_discover=True`, and Popular's 🔥/🏆/💬 badges are ORed in. Popular shows the same story with ✨, consistent with existing cross-view badge merging. |
| Short windows | **OK.** With fewer than 11 head cards, the far card is appended (Recommended grows by one; tested). With 0 tail cards and no Discover picks, nothing changes. |
| SQL | **OK.** No new SQL in production code; the test insert is parameterized. |
| Silent failures | Only L1. `_record_interleave` and `_record_deck_scores` keep their existing `logging.exception` paths. |

---

## Test adequacy (`tests/test_far_card.py`)

**Good coverage:**
- selection from the tail and from Discover;
- tie-break;
- head stories excluded from Explore candidates;
- no-candidate no-op;
- short windows;
- the `serve_window` age-out paths;
- a Hypothesis property test of served invariants;
- `closest_up` set or left `None` by `assemble_window_deck`;
- `_record_interleave` with and without production;
- report crediting;
- `deck_scores` round-trip and legacy load.

**Missing cases that matter:**
1. **M1 regression:** the far card aged out while another eligible current far
   story exists in the tail. The current test,
   `test_serve_window_far_aged_out_serves_plain_first_view`, asserts the
   current behaviour.
2. **No up-votes (M2):** an `ExploreContext` with all-zero `cand_closest_up`
   should place nothing, or the test should pin down the intended behaviour.
3. **End to end on the interleaved path:** run `draft_recommended` then
   `place_far_card` on a deck with `arm_rankings`, and assert:
   - the first 11 drafted (story, arm) pairs are unchanged;
   - the far card's id is not under any trial arm in the `_record_interleave`
     row for that window;
   - each arm's served count is 11 split as evenly as the round order allows.
4. **`closest_up` survives `finalize_ranked_deck`/`canonicalize_deck`**,
   including an id rewrite: the canonical story should be eligible and placed
   under its new id. No current test runs these stages, and `compare=False`
   hides a dropped field from the equality asserts.
5. **Malformed `arm`** in `deck_scores` (`null`, an int) should raise (L1).
6. **Report counts:** assert that `credited` and `decks_with_credit` exclude
   far votes once L2 is fixed. The current test asserts that the far vote is
   in `credited`.
7. **Multi-window decks:** the same story as far in one window and as a head
   card in another. Both `place_far_card` and `credit_votes` should keep them
   separate. All current tests use only `"1w"`.
8. **Hypothesis generator realism:** `_random_deck` sets `is_discover` at
   random on Recommended cards and allows Explore ids to overlap Recommended
   ids. That exercises dedup, which is useful, but it never builds a
   score-ordered drafted deck with arm tags. Item 3 covers that gap.

---

## Summary table

| ID | Severity | Location | Issue |
|---|---|---|---|
| M1 | medium | ranking.py:2407-2411, 2433-2439 | Chosen far card ages out, so no far card is served although other eligible stories are current |
| M2 | medium/low | ranking.py:2753-2757, 2383-2397 | No up-votes means every story is "far"; card 12 is relabelled Discover |
| L1 | low | database.py:1741-1758 | Non-string `arm` silently coerced with `str()` |
| L2 | low | interleave_report.py:102,130,204 | `credited`/`decks_with_credit` include far votes |
| L3 | low (info) | ranking.py:2380; interleave.py docstring | 11 served is not a round boundary: unbiased, more variance; docstring is stale |
| L4 | low | ranking.py:2293-2307, 2361 | Interleaved: undrafted union stories are never far candidates |
| L5 | low | index.html:1395; app.py:2693 | Explainer text claims a far card that may not exist |
| L6 | low | __init__.py:271; ranking.py:2024 | Pre-canonical `closest_up` kept (fine); WindowViews docstring is stale |
| L7 | low | PLAN.md "Far card" | Discover's readout is confounded; use served positions |
