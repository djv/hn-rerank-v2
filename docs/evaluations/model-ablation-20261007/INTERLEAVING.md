# Interleaving and offline follow-ups, October 7

The user chose live interleaving of shortlist #2 (one joined logistic
classifier, all features) and #4 (the same classifier without metadata)
against production for profile 151 only. The user also chose four offline
follow-ups: a power replay, full-data fit timing, a metadata-scale middle
variant, and a short-history (cold-start) curve. The settings were recorded
in `followups/plan.json` before any run. All offline numbers below reuse
labels from the earlier model selection; they are exploratory, not
confirmation.

## Live design (deployed for profile 151; off by default for other profiles)

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
- Live incident, 2026-10-08 13:15:56 UTC (first warm after a restart): a
  vote landed between arms, so `joined_no_metadata` read 3,889 votes while
  the shared vote features held 3,888, failed to fit, and that warm's deck
  was production's alone (not stored in `interleave_decks`, so it credits
  no arm). 1 of 24 interleaved warms since T0. Fixed the same day: every
  arm of a rerank now trains on one vote snapshot.
- Design change mid-test, 2026-10-08 (user's choice): views show 8 stories
  instead of 12 (`VIEW_SIZE` 16 -> 12, so each arm drafts from its top 24
  instead of 32). Crediting is unchanged (per impression and vote); a
  view's first 8 cards split about 3/3/2 instead of 4/4/4. Effective from
  its VPS deploy on 2026-10-08 (WORKLOG).

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
  deployment (before 04:07:22 UTC Oct 8, unbalanced 3/2/7 splits after drops)
  are excluded. T0 = 1791432442.75 (2026-10-08 04:07:22 UTC, deploy
  8d85747); the window ends 1793851642.75 (2026-11-05 04:07:22 UTC); the one final analysis covers votes in [T0, T0 + 28 days):
  The final command and frozen-snapshot requirements are in the reporting
  addendum below. Analyze the snapshot, not a later mutable live DB.
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

## Reporting addendum — approved October 12, before the endpoint

This addendum follows the independent Claude cloud review and was selected
after seeing the interim counts. It completes previously unspecified reporting
choices; it is not a claim that these details were specified before T0.
The primary up-rate z-test, arms, fixed endpoint and runtime are unchanged.

- The family remains the two original challengers: alpha .05 / 2 = .025,
  even if a safety stop leaves only one challenger in `--arms`.
- Downvote harm is a **one-sided pooled two-proportion z-test** of
  challenger > production, at .025 for the final harm check. The safety
  rule uses the same p-value at .01 plus a difference of at least .10.
  These flags indicate an alarm, not proof of safety when absent.
- Report the up-rate difference with a descriptive, unadjusted **95%
  Newcombe interval combining Wilson score intervals** (no continuity
  correction). [Newcombe 1998](https://pubmed.ncbi.nlm.nih.gov/9595617/);
  [SciPy Wilson bounds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
  The existing primary p-value remains decisive. Clustering and adaptation
  remain limitations; the interval does not establish equivalence.
- Only decks created at or after T0 are eligible. Votes use the half-open
  interval [T0, endpoint). Missing tables or no eligible decks fail explicitly.
- Exclude the recorded calibration `updated_at` interval **inclusively**:
  [1791763112.57, 1791763180.12]. The CLI accepts repeatable exclusions
  and reports their bounds, raw counts and excluded counts. A read-only
  diagnostic found all 29 affected rows and zero Recommended impressions
  for those story IDs since T0, so no original or calibration trial credit
  was lost for them. A separate diagnostic found one production upvote
  attributed to a pre-T0 deck; the corrected loader excludes that deck.
- Impressions with identical timestamps are ordered deterministically by
  occurred_at, received_at and event_id. The existing one-second slack
  and latest-impression attribution rule are unchanged. Client/server clock
  skew still needs a descriptive check; the slack is not a clock guarantee.

### Freeze before the final analysis

`feedback` stores final mutable state, not a vote log. A later edit or undo
can overwrite a vote from the trial window. `--until` alone cannot recover
that overwritten state. Freeze a SQLite-consistent copy at the endpoint
**before further voting/calibration**, then run all final analyses on that
same file. No timer or production change is created by this addendum. If
the snapshot is taken late, disclose its capture time and any intervening
changes; do not label it exact endpoint state without evidence.

On the VPS, keep the snapshot outside Git in a private directory, using
SQLite's backup facility (copies consistently even when the live DB uses WAL):

```sh
install -d -m 700 /home/dev/hn-rewrite/analysis/trial-20261105
sqlite3 -readonly /home/dev/hn-rewrite/main/hn_rewrite.db \
  '.backup /home/dev/hn-rewrite/analysis/trial-20261105/endpoint.db'
chmod 600 /home/dev/hn-rewrite/analysis/trial-20261105/endpoint.db
```

Record actual capture time and the snapshot's SHA-256 alongside the result;
retain the snapshot unchanged. From a checkout containing these corrections:

```sh
uv run python scripts/interleave_report.py \
  --db /home/dev/hn-rewrite/analysis/trial-20261105/endpoint.db \
  --user-id 151 --since 1791432442.75 --until 1793851642.75 \
  --exclude-updated-between 1791763112.57 1791763180.12 \
  --output /home/dev/hn-rewrite/analysis/trial-20261105/report.json
```

The CLI is read-only on its input DB; `--output` writes the named report.
It does not copy the DB, schedule a capture, decide deployment, or measure
rank latency. Continue the separate descriptive `rank_perf` median check
against the 30-second safety bar. No additional review was authorized.

At the October 12 snapshot, 61 credits in 4.52 days imply about 13.5/day,
far below the replay's 50/day. Around 125/arm at 28 days would resemble the
optimistic replay's seven-day row (low power). State observed sample size
and uncertainty at the endpoint; do not extend the trial or change votes
to reach a desired result. The saved 61-credit snapshot includes the one
now-confirmed pre-T0 production credit and is historical, not corrected.

## Early endpoint — 2026-10-16 (user decision)

The user ended the trial on day 8.3 of 28 to switch production to the single
model (`joined_all`) for every profile. This is the "no significant
difference, adopting the simpler model is the user's call" outcome, taken
early; it is not a statistical win.

- Snapshot: SQLite `.backup` of the live DB, captured 2026-10-16
  12:15:44-12:15:49 UTC (`1792152944.25`), before the config switch, in VPS
  `/home/dev/hn-rewrite/analysis/trial-early-20261016/` (`endpoint.db`,
  SHA-256 in `endpoint.sha256`, `report.json`). Latest profile-151 vote in
  it: `1792120285.07`. The VPS rebooted once mid-trial (2026-10-16 02:04 UTC).
- Command: the addendum's, with `--until 1792152944.25` and the calibration
  exclusion.
- `joined_all`: 61 credits vs production 72; up rate 0.426 vs 0.361
  (difference +0.065, 95% Newcombe [-0.099, +0.226], p = 0.44); down rate
  0.082 vs 0.194 (one-sided harm p = 0.97); per-deck signs 9 wins, 11 losses
  (p = 0.82).
- `joined_no_metadata`: 63 credits; up 0.381 (+0.020, [-0.139, +0.180],
  p = 0.81); down 0.286 vs 0.194 (harm p = 0.11, no safety stop); decks 13/13.
- Deployed `f86d5de` (`classifier = "joined_logistic"`, blend off,
  interleaving off), restarted 12:18:45 UTC; rollback tag
  `deploy-pre-single-model-20261016` on the VPS.
