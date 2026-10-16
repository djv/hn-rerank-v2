# Feed exploratory round 2 — 2026-10-16

User: "do some exploratory work. use 10x muse workers as needed", then "and
5x codex". Four Go Muse Spark writers (pi, xhigh) each wrote one analysis
script from code only (`muse-common-prompt.md` + `eN-task.md`); Codex (Sol 6.1
high) wrote E7 and statically reviewed E3-E6 (`codex-reviews/`). The agent
reviewed every script against the source, checked or fixed each Codex
High finding (all four verdicts: RUN AFTER FIXES), and ran them read-only on the VPS
live DB (`mode=ro`, nice 19, one thread) from checkout `8371de7`. Profile 151.
Per-story outputs stay private in VPS `/home/dev/hnx-20261016/`; only
aggregates are saved here (E5's per-source table is omitted because it lists
the user's subscriptions). No code, runtime, trial or DB change.

## E3: is the served feed getting more same-y? (`sameness-trend.py`)
2,629 first-shown stories since impressions began (ISO weeks 39-42).
- Recommended is not narrowing: mean max-similarity to earlier ups went
  0.787, 0.741, 0.765, 0.756 by week, while the set of earlier ups only grows
  (which pushes max-similarity up). Share novel (< 0.60) 4-12%.
- Inside Recommended, up rate among voted is flat above 0.6 similarity
  (0.32 / 0.31 / 0.33 for 0.6-0.7 / 0.7-0.8 / >= 0.8) and lower only below
  0.6 (0.16-0.22, n = 80). In Explore and Popular up rate climbs steeply
  with similarity (Explore < 0.5: 0.02; >= 0.8: 0.37).
- The user's own new ups are widening: mean similarity to earlier ups
  0.81 (W39) to 0.74 (W42), novel ups 1% to 12.5% (rho -0.64 over 7 weeks).
- Only four weeks; selection confound (Recommended's far stories were chosen
  by a high model score). Codex's self-similarity leak finding affects 0 rows.

## E4: Combo vs novelty term (`combo-vs-novelty.py`, on E1's replay rows)
Per-day top 12, 19 days, day-block bootstrap; placebo = same number of
random swaps (percentile of the real delta among 200 placebos).
- Recommended: Combo (MMR 0.8 + far card) keeps ups (4.47 vs 4.47, 97th
  placebo percentile) and downs (-0.05, 0th), doubles novel ups (0.26 to
  0.53/day), adds spread (+0.044) and topics (+0.26). MMR alone loses ups
  (-0.16). Novelty term alone: +0.05 ups, +0.05 novel, little spread.
  Combo + novelty 0.10: ups +0.11, novel 0.53.
- All views: novelty 0.10-0.15 gives the most ups (+0.21/+0.26); Combo the
  most spread (0.663 vs 0.589).
- Inherits E1's limits: exposure-limited, current story text and the full
  candidate set feed each day's normalisation (Codex review item 2).
- Verdict: the planned Combo is the right first step; a novelty term on top
  adds little inside Recommended.

## E5: topics the feed under-shows (`exposure-gaps.py`)
First run (`e5-output.txt`) is superseded: Codex found that pre-2026-09-21
votes entered the universe and each reference up matched itself. The agent
fixed both and reran (`e5b-output.txt`): 9,233 stories since 2026-09-21 in
40 clusters; 2,656 shown, overall up rate 0.177 among shown with an outcome.
- Recommended exposure follows up rate: the most over-shown clusters in
  Recommended (AI model and tooling news, 4.2x and 3.3x) have up rates
  0.43/0.44.
- No under-shown hit. Three over-shown misses: AI-assisted coding opinion
  essays and AI industry/policy news (4-5% of the pool, 7-8% of shown, up
  rates 0.10/0.09, Recommended exposure 0.92/1.16, so mostly via Popular and
  Explore), and a small forum-discussion cluster (up rate 0.06, Recommended
  exposure 2.24).
- Limits: candidate eligibility is broader than production (disabled feeds
  and retired rows are included; Codex item 4, not fixed); the per-source
  table is omitted here.
- Verdict: no topic gap; the large misses enter through Popular.

## E6: label noise and a kNN reference (`label-noise.py`)
4,430 votes (951 up, 1,611 neutral, 1,868 down).
- Near-duplicates (cosine >= 0.90) are rare: 86 pairs; 36 conflict, but only
  one is up vs down. Neutral is the fuzzy class.
- Leave-one-out kNN up-share using all other up/down votes (including
  future ones) reaches AUC 0.885 (k = 15). This is retrospective agreement,
  not a prospective ceiling. The script's production comparison (0.878) is
  withdrawn: it scored each day's voted stories alone, but production's blend
  uses candidate-relative percentiles (Codex item 1). The E2b replay's
  production up-vs-down AUC (0.89, impressions pool) is the looser reference.
- Spec deviation: neutral votes were excluded as neighbours (Codex item 2).
- Drift: in the latest 30 days, neighbours from other periods predict worse
  than same-period ones (0.824 vs 0.858), unlike earlier periods. Recency
  decay was null three times, so this is a watch item, not a fix.

## E7: calibration and Unsure (`calibration.py`, Codex-written, agent-reviewed)
Day-forward production fits 2026-09-28 to 10-16; 1,764 voted stories.
The agent patched one crash (skip candidates without a current stored
vector, as the templates do; one day affected).
- Raw probabilities are badly calibrated: up/down ECE 0.142; the top up bins
  predict 0.70 but observe 0.36-0.46, and downs are under-predicted
  (pred 0.06, observed 0.16-0.42).
- Raw entropy hardly varies (1.05-1.15 bits) and does not mark mixed
  stories: the top entropy quintile is 53% down, 16% up.
- Day-forward Platt scaling on prob_up (fit only on earlier days) cuts up/down
  ECE to 0.070 and log loss 0.478 to 0.390. Its top entropy quintile is
  genuinely mixed (29% down, 39% neutral, 32% up). Unsure's daily top 10
  among first-shown stories changes (Jaccard 0.23) and its up rate among voted
  rises from 0.19 to 0.29.
- Verdict: calibrating before computing Unsure entropy is a cheap, likely
  improvement for Explore's Unsure (also a prerequisite for any Thompson-style
  exploration). Exposure-limited: only first-shown stories were checked.

## Review notes
- Output-path hazards (an OUT path could overwrite a file) were flagged in
  every review; every run wrote to a new private path on the VPS.
- E3/E4 High findings (self-similarity leak, None-maxsim far card) affect
  0 rows on this data (checked).
- Next: calibrated Unsure entropy and Combo are post-trial candidates;
  nothing here changes the running trial.
