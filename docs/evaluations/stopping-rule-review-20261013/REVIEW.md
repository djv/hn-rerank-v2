# Review: November 5 stopping design (hn-rerank interleaving trial)

## Recommendation and critical caveats

Mostly agree with the agent. **Run the preregistered Nov 5 analysis exactly as written**: freeze the snapshot and report it as an underpowered fixed-horizon result. **Then start a new, separately pre-specified Stage 2.** Do not quietly turn Nov 5 into a "checkpoint" of the same test. Nothing forces adopting a winner on Nov 5: outcome 2 (INTERLEAVING.md:166-169) already allows "no detectable difference".

Caveats the explanation needs:

- The z-test's nominal α already rests on independence of credited votes. That assumption is doubtful here. Votes cluster within decks and sessions. All arms refit after every vote, so past outcomes change future treatments. One story can recur across decks, and voting is selective. So the claim that a fixed date "supports" the test should read "supports the preregistered test, whose own error rate is approximate".
- "Checking p-values and stopping on a winner inflates false positives" is correct (Armitage, McPherson & Rowe 1969). But generic fixes don't apply automatically: Pocock or O'Brien–Fleming boundaries, Lan–DeMets spending and mixture-SPRT/always-valid p-values (Johari et al. 2017; Howard et al. 2021) assume i.i.d. or martingale increments. Adaptive retraining and clustering break that unless the deck becomes the unit.
- Accrual numbers are accurate. 60 corrected credits over 4.52 days (corrected-saved-window-report.json:353 and :343) is about 13.3/day across three arms, roughly 4.4 per arm per day, or about 125 per arm by Nov 5. The replay assumed 50/day (INTERLEAVING.md:43, :245-247). At p₀≈.30, α=.025 two-sided and 80% power, 125/arm detects only about 18 points. Detecting 10 points needs about 430/arm (~100 days), before any design effect.
- The interim effects (+.23 and +.17 up-rate; CIs at json:399-401 and :422-424) are noisy. They must not set the Stage 2 effect target.

## Q1–Q2: Changing the rule after seeing interim results

The protocol itself forbids extending the trial "to reach a desired result" (INTERLEAVING.md:248-249), and the Oct 12 counts have been seen. So:

- **Confirmatory:** the only clean confirmatory claim from [T0, Nov 5) is the preregistered one. A rule chosen now can be confirmatory only for **data not yet collected**, and only if it is written down and frozen before Stage 2 starts.
- **Defensible transition, without wasting data:**
  - Option (a), simplest: Stage 2 is a fresh test on post-Nov-5 data only. Stage 1 still informs the prior and product judgement.
  - Option (b): a pre-specified p-value combination, either inverse-normal with fixed weights (Lehmacher & Wassmer 1999) or Fisher's product (Bauer & Köhne 1994). These designs stay valid after unplanned interim looks, *provided* each stage's p-value is valid under H₀ given the earlier stages. That is plausible only with a deck-level randomization or cluster-robust stage test, not the per-vote pooled z (simulate_interleaving.py:179-181).
- **Exploratory/product:** pool everything, show estimates with intervals, and decide on product judgement. Label it as such.

## Q3: Recommended practical design

1. **Unit and estimand:** per served deck version, the difference in upvotes credited to challenger and to production, normalized by the number of slots each arm got. This is the standard team-draft outcome (Radlinski, Kurup & Joachims 2008; Chapelle et al. 2012). Use the per-vote rate only as a secondary measure.
2. **Stopping on sample, not calendar:**
   - Stop at a pre-set number of **decks with any credit**, for example 150 credited decks per comparison.
   - Add a calendar cap (for example Feb 1) and accept an interval-only result at the cap.
   - Allow at most 2 planned looks with O'Brien–Fleming-type spending, applied to a deck-level statistic.
3. **Precision target:** the user should state the smallest difference that matters (suggested: 10 points of up-rate, or 0.3 upvotes per deck). If that is unreachable within the cap, say "estimation only".
4. **Futility/tie:** if the 95% interval at the cap excludes the meaningful effect, call it a practical tie and pick on simplicity or latency, as #4 vs #2 already allows (INTERLEAVING.md:168-169). Optionally drop the weaker challenger at the first look, and pre-specify that drop.
5. **Safeguards:**
   - Freeze a snapshot at every stage boundary and every look.
   - Keep α=.025 per challenger and the existing safety stop.
   - Allow no model or config changes during a stage (the mid-test 12→8 view change, INTERLEAVING.md:34-38, is the counterexample).
6. **Telemetry:**
   - Already available: decks with per-window picks, impressions with version/window, and final votes, so slot exposure per arm can be computed.
   - Missing: full per-arm rankings, which a randomization test needs; an append-only vote log; and, as far as the reviewed files show, dwell or click data.

Missing user choices: the meaningful effect, the calendar cap, whether #4 vs #2 matters or only challenger vs production, and option (a) vs (b).

## Q4: Does credited up-rate answer the product question?

Only partly.

- It measures "given I voted on a card this arm drafted, did I upvote it". That is conditioned on a selective, post-treatment vote. An arm whose cards get skipped loses its denominator, not its score.
- It does not measure time saved, discovery, recency of important developments, or reduced sameness.
- **Necessary:** add per-deck credited ups per exposed slot (exposure-normalized) and report position-by-arm, because a round cut by the far card skews slot counts (interleave.py:7-10).
- **Optional rigor:**
  - topic/source diversity per arm (sameness);
  - age of credited stories (currency);
  - a weekly drift view by arm;
  - cluster-robust SEs.
- **Mutable history:** `feedback` keeps only the final state (INTERLEAVING.md:211-213). A Stage 2 re-vote moves `updated_at` and silently removes the vote from Stage 1 in the live DB.

## Q5: Verification and source defects

Verified:

- Arms are production, `joined_all` and `joined_no_metadata` (interleave.py:276-281; interleave_report.py:280).
- The window is exactly 28 days: T0=1791432442.75 (Oct 8 04:07:22 UTC) and end=1793851642.75 (Nov 5 04:07:22 UTC) (INTERLEAVING.md:150-151).
- Far-card votes are excluded (interleave_report.py:124-125).

| Sev | file:line | Issue |
|---|---|---|
| Medium | interleave_report.py:241-249 | Votes come from mutable `feedback`. Any later recomputation of Stage 1 from a post-Nov-5 DB changes its counts. Stage 1 must come only from the Nov 5 snapshot. |
| Medium | simulate_interleaving.py:179-181 (used by report:168) | The pooled per-vote z ignores deck clustering and adaptation. It can't serve as the stage statistic for a combination or sequential design without justification. |
| Low | interleave_report.py:236-237 | Decks are filtered by `since`. A Stage 2 run with `--since` set to Nov 5 drops votes on pre-boundary decks as `deck_not_interleaved`. This is acceptable, but should be pre-declared. |
| Low | interleave_report.py:40,153 | The CLI has a fixed α/2 and no concept of looks or stages. A sequential rule needs new, tested code. |
| Low | tests/test_interleave_report.py | No test covers the `far_card` skip path. The saved report shows no `far_card` or `story_not_in_window` counts, so storage of the far card can't be confirmed from these files. |
| Info | INTERLEAVING.md:151 | A sentence ends with a stray colon before the addendum reference. Cosmetic. |

## References

These are cited from memory. I did not verify them externally in this session.

- Armitage, McPherson & Rowe 1969 (JRSS A)
- O'Brien & Fleming 1979 (Biometrics)
- Lan & DeMets 1983 (Biometrika)
- Bauer & Köhne 1994 (Biometrics)
- Lehmacher & Wassmer 1999 (Biometrics)
- Johari et al. 2017 (KDD), "Peeking at A/B tests"
- Howard et al. 2021 (Ann. Stat.), confidence sequences
- Radlinski, Kurup & Joachims 2008 (CIKM)
- Chapelle et al. 2012 (TOIS)
