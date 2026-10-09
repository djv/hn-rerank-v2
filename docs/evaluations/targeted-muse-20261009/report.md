# Targeted Muse pairwise experiment (20261009) — report

## Objective
Test whether Muse pairwise preferences support a substantive vs
repetitive-question/social-chatter text feature, using same-source pos-neg
pairs from folds 3/4 where production and Opus disagree (corrections /
regressions) plus agreement controls.

## Method (quota-fixed scorer)
- Patched `score_muse.py` to REAL batches: at most 7 tasks per request, grouped
  by fold sharing ONE training-only profile; 6 serial batch requests
  (f3-fwd x5, f4-fwd x7/x7/x6, f3-rev x7, f4-rev x3) via
  `/home/dev/.opencode/bin/opencode run --pure --format json
  -m opencode-go/muse-spark-1.3-contributor`.
- Task keys visible to the scorer are opaque random neutral IDs (mapped locally
  only); labels, prod/Opus scores and group names never enter the prompt.
- Reversed re-scores are fresh calls in separate batches from forward pairs.
- 10 pre-existing valid cached answers retained without rescoring; 35 new tasks
  cached with prompt/model hashes. Responses validated as JSON arrays with exact
  opaque IDs, preferred story IDs, confidence and style tags (max 2 retries).
- For the 35 new batched answers, scorer NO TOOLS was enforced via child env
  `OPENCODE_CONFIG_CONTENT` permission deny for all tool types. The 10 retained
  earlier answers used a no-tools prompt instruction without this enforcement;
  this is a protocol difference. Global config was untouched.
- Fresh temp cwd per batch inside the experiment dir. Result: 45/45 parsed OK.

## Results (exploratory — selection-biased; folds 3/4 are dev, not evaluation)
- Overall forward (n=35): Muse 25/35 (71%), prod 20/35 (57%), Opus 21/35 (60%).
- Fold 3 (n=15): Muse 10/15; corr 4/5, regr 2/5, agrR 4/4, agrW 0/1.
- Fold 4 (n=20): Muse 15/20; corr 7/7, regr 3/6, agrR 5/5, agrW 0/2.
- Corrections (prod wrong / Opus right): Muse 11/12 — tracks Opus.
- Regressions (prod right / Opus wrong): Muse 5/11 — splits, leans Opus (6/11).
- Controls: agree_right 9/9, agree_wrong 0/3 (all three judges agree, rightly/wrongly).
- Reverse stability: 9/10 stable; only flip f4-regr-01 (fwd wrong, rev correct).
- Machine-readable: `results.json`, `muse_pair_scores.json`, `muse-cache.json`,
  `muse_batches.json` (opaque mapping local only).

## Feature decision: REJECT (rigorous negative result, no challenger built)
Muse's own style tags do not support a substantive-vs-chatter feature:
pos substantive 22/35 vs neg 23/35 (nearly equal); corrections pos_sub 6/12 vs
neg_sub 6/12; regressions neg MORE substantive (10/11) than pos (7/11).
Muse is correct on f4 corrections even when both stories are tagged
substantive/substantive or news-brief/news-brief, so preference does not track
tags. A training-only deterministic chatter heuristic + chronologically fitted
residual model would have no target and would reuse held-out signal if fitted
on tags; per the brief it is not built. No production feature implementation.

## Limits / next step
Pair selection used held-out labels + prod/Opus scores, so all accuracies above
are exploratory, not unbiased estimates. Folds 3/4 are now development data.
Any future claim requires untouched future votes. No DB writes, no deploy, no
`main/` edits; WIP preserved. Load checked before runs; analysis single-thread.
