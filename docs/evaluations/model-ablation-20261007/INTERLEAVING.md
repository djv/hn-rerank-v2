# Interleaving and offline follow-ups, October 7

The user chose live interleaving of shortlist #2 (one joined logistic
classifier, all features) and #4 (the same classifier without metadata)
against production for profile 151 only. The user also chose four offline
follow-ups: a power replay, full-data fit timing, a metadata-scale middle
variant, and a short-history (cold-start) curve. The settings were recorded
in `followups/plan.json` before any run. All offline numbers below reuse
labels from the earlier model selection; they are exploratory, not
confirmation.

## Live design (implemented, off by default, not deployed)

- Arms: production (current three-model blend), `joined_all` (#2),
  `joined_no_metadata` (#4). Each window's Recommended view is a team-draft
  interleaving: every round visits the arms in a fresh random order, and
  each adds its best story not yet listed. Cards keep production's score,
  probabilities and explanation, and the arm is never sent to clients.
  Popular and Explore are unchanged.
- The warm stores each interleaved deck's arms per version in
  `interleave_decks`. `scripts/interleave_report.py` credits a vote to the
  arm that drafted the story, using the user's last impression before the
  vote (view, window and deck version from `interaction_events`). Undone
  votes drop out, and changed votes count with their final action.
- Measured on the 2026-10-07 snapshot (Sept 29 – Oct 7), 438 of profile 151's
  1,153 votes followed a Recommended impression: about 50 a day, about 13 of
  them upvotes. Only 2 votes had no prior impression.

## Power replay

`scripts/simulate_interleaving.py` replays the three arms' frozen
twelve-block scores. Each simulated day takes one judged block and 50 votes.
The arms are re-interleaved every 4 votes from the stories not yet voted
on, and each vote is credited to its arm. 1,000 replicates per setting.
Rejection rates are at alpha .05, Bonferroni over two challengers (.025 per
test). Credited upvote rates: #2 .483, #4 .468, production .394.

| Days (credited votes per arm) | #2 rate test | #2 deck sign test | #4 rate test | #4 deck sign test |
|---|---:|---:|---:|---:|
| 7 (~121) | .25 | .15 | .12 | .07 |
| 14 (~243) | .40 | .33 | .26 | .20 |
| 28 (~486) | .70 | .60 | .54 | .42 |
| 56 (~971) | .92 | .87 | .88 | .76 |

No replicate found a challenger significantly worse. With every arm
replaying production's scores, rejection rates were .012–.024 against .025
per test, so both tests are calibrated. Re-interleaving after every vote
lowers power (#2 rate test .66 at 28 days); re-interleaving every 12 votes
raises it (.73). Ranking each challenger halfway between its own and
production's percentile ranks barely shrinks the credited gap, because
team-draft credit comes from the stories where the arms disagree.

This replay is optimistic. Every story in a block was judged, the models
are frozen within a block, and #2 and #4 were selected on these labels.
Live differences are probably smaller, so a 28-day run should be read as
able to detect only a large difference, not as able to establish
equivalence.

## Metadata-scale middle variant

The same classifier as #2, with only the metadata block multiplied by 0.5
or 0.25, on the same twelve blocks (2,231 judged stories; production scores
identical to the earlier runs):

| Classifier | Mean AUC | NDCG@12 | Upvotes /144 | Downvotes /144 |
|---|---:|---:|---:|---:|
| Production blend | .8182 | .6656 | 94 | 2 |
| #2, metadata x1 | .8295 | .6944 | 97 | 6 |
| Metadata x0.5 | .8274 | .6997 | 97 | 6 |
| Metadata x0.25 | .8246 | .6996 | 98 | 7 |
| #4, no metadata | .8203 | .7094 | 98 | 5 |

Shrinking metadata moves AUC and NDCG between #2 and #4. Neither middle
setting beats both endpoints, so the live arms stay #2 and #4. AUC
against #2: x0.5 -.0021, block interval [-.0063, +.0012], sign-flip p .38;
x0.25 -.0049, [-.0158, +.0033], p .45. Against production: x0.5 +.0092
[.0000, .0180], p .083; x0.25 +.0064 [-.0023, +.0146], p .19
(`followups/summary.json`; 20,000 block draws).

## Short-history curve

Each block trains on only its N most recent prior votes (`--train-recent`).
Production's numbers come from the same truncated training sets. Twelve
blocks; mean AUC and top-12 upvotes/downvotes out of 144:

| Votes N | Production | #2 all features | #4 no metadata |
|---:|---|---|---|
| 100 | .6926, 65/32 | .6933, 66/31 | .6929, 66/31 |
| 200 | .7353, 75/21 | .7375, 77/20 | .7358, 79/19 |
| 400 | .7759, 82/13 | .7874, 85/12 | .7852, 86/11 |
| 800 | .7975, 79/11 | .8098, 85/9 | .8034, 86/8 |
| 1600 | .8134, 90/6 | .8263, 93/6 | .8146, 94/4 |

At 100 votes (about 22 upvotes) the classifier tier barely enters the tier
blend, so all arms nearly coincide. From 200 votes up, both challengers
match or beat production on every column. Paired AUC against production:
#2 at N = 400 / 800 / 1600 is +.0115 / +.0122 / +.0129, block intervals
excluding zero, sign-flip p .040 / .037 / .030 (unadjusted). #4's
intervals include zero at every N. The labels were reused, and
profile 151's recent votes stand in for a new user's first votes. So this
supports, but does not establish, extending the challengers to other users.

## Full-data fit timing

Measured on the production VPS (4 cores, one BLAS thread, nice 19) against
a copy of the live DB taken 2026-10-08 03:20 UTC: profile 151, 3,872 votes,
11,011 candidates. The production service was untouched.

| Warm, profile 151 | Production only | Production + both challengers |
|---|---:|---:|
| Models cached (no new vote) | 7.4 s | 13.4 s |
| After a vote, no warm starts | 9.3 s | 38.9 s |
| After one vote, warm starts | 10.3 s | 29.3 s |
| After one vote, warm starts + shared features (deployed) | ~10 s | 21.2 s |

The two after-a-vote rows come from different runs: the first is
`benchmark_rank_cold_cache.py` with every model cache cleared, the second
re-adds the newest vote to the copy and reranks. Without a warm start the
challengers' logistic fits took 12.9 s (#2) and 13.8 s (#4). Warm-starting
from the user's previous fit (af60164) cuts them to 4.1 s and 4.3 s, with
probabilities equal to lbfgs tolerance. The arms then share production's
side vectors, candidate and training features, cluster centres and the
candidates' word counts (`SharedFeatures`; scores identical with and
without sharing), leaving each challenger about 5.5 s: a 4 s fit and a
1 s scoring pass. Production's own
numbers match live `rank_perf` (9.8 s mean on a refit, 6.9 s cached, last 3
days). Live, after each vote burst the reranked deck arrives ~21 s later
instead of ~10 s; the user keeps voting on the previous deck, minus
voted stories, meanwhile.

## Decision rule (fixed before the live start)

The user chose a fixed 28-day horizon and to deploy current `main`, which
also ships the expanded Why-this-story panel.

- Start T0 is the first `interleave_decks` row after the deployment that
  drafts after deduplication (recorded in STATUS.md); decks from the first
  deployment (03:56–04:3x UTC Oct 8, unbalanced 3/2/7 splits after drops)
  are excluded. T0 = 1791432442.75 (2026-10-08 04:07:22 UTC, deploy
  8d85747); the window ends 1793851642.75 (2026-11-05 04:07:22 UTC); the one final analysis covers votes in [T0, T0 + 28 days):
  `uv run python scripts/interleave_report.py --db hn_rewrite.db --user-id
  151 --since T0 --until T0+2419200`.
- Primary test, per challenger: credited upvote rate against production's,
  two-proportion z-test, two-sided, alpha .025 (Bonferroni over two).
  Secondary: the per-deck upvote sign test and the credited downvote rates,
  both reported but not decisive.
- Weekly reports are descriptive. No arm is stopped for looking better.
  Safety stop for an arm: at a weekly check its downvote rate is at least
  10 points above production's with p < .01, or the warm's median rank
  time exceeds 30 s. Stopping means removing the arm from
  `interleave_arms`, with the user's approval.
- At 28 days:
  1. A challenger with a significantly higher upvote rate and no
     significantly higher downvote rate is proposed to replace production
     (a separate decision and deployment).
  2. No significant difference: no detectable difference at this power.
     Report the 95% interval of the upvote-rate difference; adopting the
     simpler model is the user's call, not a statistical conclusion.
  3. A challenger with a significantly lower upvote rate: keep production
     and drop that arm.
- Known limits: profile 151 only and Recommended views only. The user knows
  the test is running but not which arm drafted a card. All arms refit after
  every vote. Undo re-inserts a card by production's score.
