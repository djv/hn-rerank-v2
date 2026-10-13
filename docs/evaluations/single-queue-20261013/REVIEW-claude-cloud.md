# Review: single-queue plan (hn-rerank, 2026-10-13)

Target: `docs/evaluations/single-queue-20261013/PLAN.md`. Read-only review.
Paths below are relative to `files/code/hn-rerank/`.

## Verdict: revise

A single voted-through stream fits how the user reads (FINDINGS "First
reading-signal review": votes on every card to move on), so the idea is sound
and worth doing after 2026-11-05. The plan as written has problems:

- Its stated build site (`assemble_window_deck`) cannot produce the queue it
  describes.
- Its Phase 0 gate is close to decided by arithmetic before any data exists:
  one half almost surely fails and the other almost surely passes.
- "pop-raw = unbiased labels" does not hold once the plan's own dedup rule
  is applied.
- It does least for the user's top goal. 60% of slots are still unmodified
  Recommended, and Popular votes, today's main source of out-of-bubble
  labels, drop sharply.

Fix those four, start from the much cheaper client-side version (see
"Simpler alternatives"), and keep Phase 0 small.

---

## High

### H1. The queue cannot be built in `assemble_window_deck` as described (PLAN "Design", L22–25, table row 1)

`assemble_window_deck` (`pipeline/ranking.py:2142–2342`) runs before:

- dedup and `canonicalize_deck` (ARCHITECTURE §3.4, "finalize_ranked_deck");
- `draft_recommended` (`ranking.py:2345`), which drafts after deduplication
  on purpose (FINDINGS "2026-10-08 Interleaving deploy: draft after
  deduplication");
- `place_far_card` (`ranking.py:2373`), which runs after the draft;
- `serve_window` (`ranking.py:2425`), which drops aged-out stories and caps
  views (VIEW_SIZE = 12, Explore at 3 per badge) at request time.

So a queue made there would contain:

- an undrafted Recommended, with no arm tags;
- no far card;
- stories that dedup or canonicalization later drop;
- fixed slot positions that fill with holes when cards age out, which is
  frequent in 12h and 1d.

The row "Recommended order (trial draft and far card as today)" is
impossible from that site.

There is also no `orders` dict on the server. `WindowViews` has three fixed
fields (`ranking.py:2031–2034`). `VIEWS`, `map_views` (positional
constructor, L2095–2107), `stories()`, `is_empty`, `deck_scores` and both
clients all assume exactly three views. `orders[sort]` exists only in the
web client's feed JSON (ARCHITECTURE "Web client model").

**Recommendation:** build the queue at serve time, from the output of
`serve_window`: served Recommended (drafted, with the far card), served
Popular, served Explore. Do it in `build_feed` or a new `serve_queue`, as a
deterministic function of those views. Then:

- no change to the cached deck, `WindowViews`, `map_views` or the
  `deck_scores` schema;
- the arm tags and far-card placement come along unchanged;
- aging out is already handled.

State this site in the plan, and fix the claim that "the server adds one
more order … in `assemble_window_deck`".

### H2. The Phase 0 gate is decided by arithmetic, not data (PLAN "Phase 0", L66–69)

**The "≥ 90% of Recommended's expected ups per 10 cards" half is close to
unachievable for every listed mix.** Rough check with the repo's own yields:

| Source | Up-rate | Where from |
|---|---|---|
| Recommended top-k | ~0.41 | Diversity Phase 0, B top-8 |
| Discover R1 | ~0.12 | Discover Phase 0 |
| Popular | ~0.08 | FINDINGS |
| Unsure | 1 of 6 recently | FINDINGS |

The 6/1/1/1/1 mix then gives about 6 × 0.42 + 4 × ~0.10 ≈ 2.9 ups per 10
cards, against ~3.8–4.1 for Recommended alone: about 70–75%. Even
7/1/1/1/0 lands near 80%, unless the τ filter makes filtered Popular nearly
as good as Recommended, which its 68% down-rate makes implausible.

**The "fewer expected downs than the current mix of views" half is close to
guaranteed.** Popular is ~43% of today's votes, at 68% down. Any mix with
2 or fewer Popular cards per 10 beats it.

**Topic coverage and pairwise diversity also rise mechanically.** Adding
any unpersonalized or far cards does this. FINDINGS "Sameness" shows
Recommended's topic entropy is already about equal to random, so "higher
topic coverage" says almost nothing.

**The novel-ups bar rests on single-digit counts.** In Diversity Phase 0, B
caught 4 novel ups in 35 groups, so "1.5×" means 6 versus 4. It is pure
noise with no interval requirement.

Net effect: the gate is fixed before the run, but its result is predictable
now (fail on ups retention), and the half that would pass measures nothing.
This is the same trap the earlier diversity gate hit, with a sharper edge.

**Recommendation:** pick one baseline that matches the decision.

- The real alternative to the queue is how the user reads today: visiting
  views, or perhaps Recommended plus a Popular glance. Measure that mix
  from impressions, per session.
- Frame the gate on a fixed vote budget, e.g. per 20 votes, the size of a
  calm session:
  - ups at least the current mix's (lower CI bound no worse than −X);
  - novel ups not fewer;
  - Popular "stay current" coverage: the window's top-N gravity stories
    still seen.
- Report Recommended alone only as context.
- Require bootstrap intervals over days, as the diversity plan did, not
  just "report counts and intervals".
- Write the arithmetic above into the plan, so nobody reads a predictable
  fail or pass as evidence.

### H3. "pop-raw = model-independent label stream" does not hold (PLAN L35, L46–48)

1. **The dedup rule makes pop-raw depend on the model.** "A story takes the
   first slot that claims it." Rec slots come first (positions 1, 2) and
   take 6 of every 10, so the stories pop-raw gets are "Popular minus what
   the model ranked high". That is a model-conditioned sample biased toward
   downs. The same applies to the `pop` slot's τ comparison.
2. **Popular covers HN only.** `is_hn_source` filter at `ranking.py:2305`.
   Even if raw, it is a gravity-ordered sample of HN, not of the candidate
   pool. Reddit and RSS stories, where much discovery happens (AINews 37%
   up), never get model-independent labels. It is fine as a "stay current"
   control but not an unbiased evaluation set for the ranker.
3. **The queue shrinks an existing model-independent stream.** Today
   Popular yields 797 votes since 2026-09-24, all model-independent. The
   queue cuts that to about 1 pop-raw per 10 cards.
4. **Votes are not a fixed truth.** They are mutable and context-dependent
   (FINDINGS "Trial stopping-rule review": feedback is mutable). A pop-raw
   card voted inside a Recommended-heavy stream may be judged differently
   from one voted in Popular.

**Recommendation:**

- If unbiased evaluation is the aim, make that slot a uniform random draw
  from the window's full scored pool, all sources. Draw it before the other
  slots claim stories, log its propensity (1 / pool size) and the draw
  seed, and keep it even when it duplicates a rec card: show it once,
  tagged with both kinds.
- Otherwise drop the "unbiased" claim and call it what it is: an HN
  front-page slot.
- Either way, the rule should be "the pop-raw / random slot claims first",
  not position order.

### H4. Weak on the worst failure, and may widen the bubble through training data (PLAN "Why", table)

STATUS ranks a same-y feed above duds, but:

- **The queue keeps the same-y part unchanged.** 6 of 10 cards are
  Recommended, which FINDINGS "Sameness" shows picks close neighbours
  within topics. Only one slot (Discover) is aimed at novelty, and the far
  card already exists.
- **Training labels will shift.** FINDINGS "First reading-signal review"
  notes Popular is "a large share of training downs". Moving from ~43% of
  votes in Popular to ~20% pop/pop-raw slots means many fewer labels on
  unpersonalized stories, so the model sees less of the space outside its
  current neighbourhood. That is the mechanism behind a filter bubble. The
  plan's Phase 1 guardrails do not watch training-label composition or
  model calibration.
- **Missing variants.** The mix list has no option with more discovery
  (e.g. 5/1/2/1/1). The 5/2/1/1/1 variant adds Popular, not discovery.

**Recommendation:**

- Add a discovery-heavier variant.
- Add as Phase 1 guardrails:
  - share of votes on non-Recommended slots;
  - weekly novel ups (already proposed in FINDINGS);
  - topic entropy of up-votes;
  - training-label composition before and after.
- Within the rec slots, order cards to avoid adjacent near-neighbours, e.g.
  reorder the served 12 so consecutive rec cards have cosine < some bar.
  This changes neither which stories are served nor the draft's credit, so
  it can be done even before Phase 0b. It directly reduces perceived
  sameness in a single stream, where adjacent sameness is most visible.

## Medium

### M1. Logging plan does not match where logging happens (PLAN L45–48)

- **`deck_scores` cannot hold the slot.** It is written at warm time from
  the cached views (ARCHITECTURE "Cached deck score ledger"), before
  `serve_window`. A serve-time queue (H1) has no row there, so
  `view="queue", slot=...` would need a new serve-time ledger.
- **Badges are the wrong field.** Putting `queue:rec` into `badges` puts
  slot kinds into a field that `badge_yield_report.py` reads as display
  badges, and its comma-joined column was not meant for this.
- **Recommendation:**
  - Add explicit `slot` and `queue_position` fields to impressions and the
    feed JSON, and log the served queue per (version, window).
  - Keep `deck_scores` as-is. A queue slot plus the card's existing
    `deck_scores` entry (joined on version and story_id) gives scores and
    slot together.
  - Log the random/pop-raw draw's propensity (H3).

### M2. Phase 0 cannot answer the main questions, and the plan should say so (PLAN L52–71)

- **Selection bias.** The pool holds only stories already shown, each with
  an outcome produced in its own view's context: Popular cards voted while
  "checking Popular", Explore shuffled and 42% unvoted (234 of 557).
- **Unvoted handling is undefined.** The plan says "outcome per
  shown-and-voted story", while the diversity method it reuses counts
  unvoted as not up. Explore's large unvoted share makes this choice move
  the Explore and Unsure yields a lot. Fix one rule.
- **Phase 0 cannot measure the actual claims:** less decision cost, a
  calmer habit, perceived sameness of the stream, or how votes change when
  kinds are interleaved.
- **Popular order replay.** Rescoring uses final points and comments
  (FINDINGS short-window review), so gravity order and P(down) for the τ
  sweep are optimistic.
- **Recommendation:**
  - Shrink Phase 0 to the τ sweep, sanity yields per kind with intervals,
    and the H2 arithmetic.
  - Answer the queue question live, with a design that has a control
    (M3).

### M3. Phase 1 pre/post baseline is confounded (PLAN L73–79)

Around 2026-11-05 several things change at once:

- the trial ends, and challenger arms leave Recommended;
- the far-card and Discover reads happen;
- a Phase 0b winner may ship onto the rec slots;
- news varies.

"Two weeks before" against "two weeks after" cannot separate these.

**Recommendation:**

- Order the changes in sequence: ship Phase 0b's outcome first, or hold
  it, then the queue.
- Alternate queue and views by day (or by session) for the two weeks, so
  the baseline runs at the same time.
- Use dwell only as a secondary measure: AUC 0.66 for up-vs-rest, and the
  idle cap changed recently.

### M4. Future team-draft "on rec slots only" needs the dedup rule fixed (PLAN L80–82)

If a pop or discover slot earlier in the queue can claim a story from the
drafted Recommended list, the served rec stream loses a drafted pick after
the draft. The arms' turns then become unbalanced, which is the exact bug
fixed on 2026-10-08.

**Recommendation:** rec slots have priority, and the other kinds exclude
the served Recommended. Explore already excludes Recommended
(`ranking.py:2311–2320`); Popular does not (`ranking.py:2304–2310`). The
only exception is the random slot, which may show a duplicate tagged with
both kinds (H3).

### M5. Queue length and end are undefined, which goes against the bounded habit (PLAN "Design")

- VIEW_SIZE = 12 served per view, with 8 shown plus 4 sliding in on the
  web (ARCHITECTURE §3.4), and Explore at 3 per badge. At 6 rec per 10,
  Recommended runs out after about 20 cards. "Empty kinds pass their slot
  to rec" then has nothing left to pass to.
- Unvoted cards reappear (FINDINGS: 92 of 397 Popular stories shown in
  more than one session).
- The far card sits at Recommended position 12. In the queue it lands near
  position 20, likely past where a quick-check session ends.

**Recommendation:**

- Define the queue's length per window (e.g. 20) and an explicit "done"
  state.
- Decide whether the far card keeps a fixed queue position (e.g. inside
  the first 10) instead of following rec order.

## Low

- **L1. Fixed slot positions confound kind with position (L37–38).** Per-kind
  yields in Phase 1 then mix in fatigue effects. This is acceptable for one
  user who votes on everything, but say so. A fixed per-day rotation of the
  pattern would separate the two cheaply.
- **L2. The Interest drop rests on n = 6 (2/2/2, L42–44).** "Phase 0 checks
  this", but no Interest variant appears in the mix list. Add one, or drop
  the claim.
- **L3. The Unsure slot serves the model, not the user's goals.** Recent
  yield is 1/0/5. Consider merging it with the random label slot (H3) into
  one "label" slot.
- **L4. Cold and new profiles are not covered.** With no SVM there is no
  `prob_down`, so the τ filter, Unsure and the far card are all absent
  (`ranking.py:2133`, `2318`). The queue degrades to rec plus pop. That is
  fine, but state it, since a new profile's first feed is a stated goal.
- **L5. The τ filter uses `prob_down`, which depends on the training
  snapshot.** The τ chosen offline on day-old models may not carry over
  (cf. the min-max `score` caveat in FINDINGS). Prefer a rank-based rule,
  e.g. skip the Popular cards in the window's top-quartile P(down), or
  re-check τ live.
- **L6. Reports.** The list (L92–94) is right but incomplete. Also check
  the far-card and Discover read scripts, and anything that joins
  impressions to `deck_scores` on view. TUI `position` semantics changed
  on 2026-10-13, and a queue adds a third meaning. Document which one each
  report uses.
- **L7. Web Explore shuffle.** The client shuffles Explore. The queue sort
  must not be shuffled, or slot positions and logging break.

## Simpler alternatives

1. **Client-only concatenation.** Ship this first, after 2026-11-05.
   - When the user finishes Recommended, the client flows straight into a
     short block of Explore (Discover first), then the top few Popular
     cards, then "done".
   - No server change and no new order. Impressions keep their real view,
     so every report and the trial machinery work unchanged.
   - It removes the "which view now?" decision, which is the stated "Why".
2. **Client-side interleave of the served views.** Same fixed pattern, with
   slot and queue position logged in impressions. Move it server-side only
   if it sticks.
3. **Reorder within Recommended for adjacency diversity (H4).** This is the
   cheapest direct fix for perceived sameness in a single stream.
4. **Skip the multi-mix Phase 0.** Run the τ sweep offline, then a 2–3 week
   day-alternating live comparison of the queue against views, judged on
   ups and novel ups per 20 votes, Popular top-N coverage, and downs.

## Summary of asks

| ID | Severity | Ask |
|---|---|---|
| H1 | High | Build the queue at serve time from served views. |
| H2 | High | Replace the gate with a vote-budget comparison against the real current reading mix, with CIs, and write down the expected arithmetic. |
| H3 | High | Make the label slot a logged-propensity random draw that claims first, or drop the "unbiased" claim. |
| H4 | High | Add a discovery-heavy variant, adjacency de-duplication in rec slots, and label-composition guardrails. |
| M1 | Medium | Log `slot` and `queue_position` as separate fields, not badges or `deck_scores`. |
| M3 | Medium | Use a concurrent day-alternating control instead of pre/post. |
| M4 | Medium | Rec slots take priority in dedup. |
| M5 | Medium | Define the queue's end. |
