## Verdict

The approach is sound in principle, but the evidence so far does not show that Muse finds any ML weakness. Test 1's headline numbers come from how its strata were sized. Test 2 measured nothing. A frozen prospective baseline is not needed: a chronological replay can stand in if the features are snapshotted.

## Key flaws

1. **Test 1's accuracies are set by the stratum quotas.** Production's 20/35 is 9 agree-right + 11 regressions by construction, and Opus's 21/35 is 9 + 12 (`targeted-muse-20261009/results.json`). Muse's 25/35 is mostly 11/12 on corrections, where it tracks Opus. On regressions it scored 5/11. Production's AUC (.83) beats Opus's (.75), so regressions probably outnumber corrections in the real population. Reweighted to real frequencies, Muse likely falls below production. The correct reading: Muse is roughly an Opus proxy, and a coin flip where production is right and Opus is wrong.
2. **Up vs rest discards most of the signal.** The window had 17 up, 42 neutral and 40 down votes (`REPORT.md:34`). Production is a 3-class down/neutral/up model (`pipeline/ranking.py:1271`), so the ordering up > neutral > down is the natively aligned target. With random pairing, about 66% of pairs would be informative, against about 29% under up vs rest. Up vs rest was also unlucky in test 2: both upvotes were paired together, and `evaluate_partial.py:68-71` discards every same-class pair.
3. **"No pre-vote ML scores" is fixable.** Refit production on votes before the window start, the same cutoff used for the frozen older training profile. The real leakage risk is mutable metadata: HN score and comment counts in the DB are current values, not the values at vote time. Use snapshotted values or drop those features for the replay.
4. **Exposure bias.** Every voted story was first served by a ranker (production, or the interleaved #2 and #4 arms since T0). Candidates are therefore already ML-favored, which compresses the score contrast. Pairs drawn only from voted stories will mostly be "both ML-high".
5. **The hypotheses were fished from the selected disagreements.** The overlap proxy was lexical and Muse-generated, and it failed (5/14). Each round spends quota on a new story, not on measuring a weakness.

## Selection without ties, leakage or easy wins

Stratify by the replayed ML's predicted class or score. Pair stories from adjacent bins within a topic and source, with an ML margin that is moderate rather than extreme. Score with ordinal labels, so neutral vs down counts too. Cap each story at one appearance. Do not use outcomes in selection.

## Three-step revised plan

1. **Retrospective mining (exploratory, no Muse).** Replay production chronologically on the 99 stories with snapshot-safe features. List its ordinal errors, then have Muse rank only pairs where ML is confident and the ordinal label disagrees: about 20 pairs, one batch. Cluster the errors Muse fixes and pick one candidate feature that already exists in the pipeline, such as recency or source prior. Lock the feature and its sign before step 2.
2. **Independent validation on untouched votes.** Use votes after 2026-10-09 (or after the cutoff, if that is earlier) that step 1 never touched. Compare the single change as a nested ablation: production vs production + feature, chronological folds, ordinal AUC with story-block bootstrap. Add a Muse ranking only if step 1 met its go gate, about 30 pairs, untouched stories, one appearance each.
3. **Interpretation.** Count pairs sharing a story as one cluster. Report results only by story-cluster bootstrap, never as a pair Wilson interval. Leave the production feature set unchanged until the interleaving test closes on 2026-11-05.

## Stop/go and quota

- **Go:** in step 1, Muse corrects at least 60% of the confident ML errors and agrees with ML on at least 70% of the control pairs where ML was right. In step 2, the ordinal AUC gain's lower bound must exceed 0.
- **Stop:** Muse fixes ML errors only about as often as it breaks correct ML choices (net ≤ 0), or two more hypotheses fail.
- **Quota cap:** at most 4 Muse batches in total (≤ 60 pairs, plus 10 reversed checks), and one Claude Code review per step.
