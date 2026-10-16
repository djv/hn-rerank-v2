# Novelty term and side zero-fill — exploratory, 2026-10-16

User: "do some exploratory work, use muse workers as needed", after the
30x Muse ML review (`../ml-quality-review-20261016/REPORT.md`). Two follow-ups
from that review. Muse wrote both scripts from code only (prompts here); the
agent reviewed them and ran them read-only on the VPS live DB (`mode=ro`, nice 19,
one thread) from checkout `8371de7`. Per-story outputs stay private in VPS
`/home/dev/hnx-20261016/`. Only the aggregate outputs are saved here. No code,
runtime, trial or DB change.

## E2: side-vector zero-fill (`side-zero-fill.py`, `e2-output.txt`)
Profile 151, current pool of 10,743 candidates, 4,430 training votes.
- Each hidden story drops about 15 rank-percentile points (score -0.14).
  Renormalizing recovers 0.8 points; kNN-imputing the side vector recovers 2.4.
  Neither is a real fix.
- Exposure is small: the pool had no story younger than 1 h (25 at 1-6 h), and
  every story under 3 days old had a side vector. The 182 uncovered stories
  (1.7%) are all archive items older than 3 days (archive coverage 95.5%).
- Verdict: the review overrated this defect. Freshness is unaffected at this
  snapshot. If anything is done, the side timer should cover archive seeds.
  Second snapshot at 10:1x UTC (`e2-1010` on the VPS): pool 10,818, again no
  candidate under 1 h, every story under 3 days covered, the same 182 archive
  stories uncovered. The side timer logs "0 of ~17,800 need a side vector"
  while those archive stories lack one, so its selection skips them; it encodes
  up to 73 new stories per 30-min run (2,119 over 3 days).

## E1: novelty term (`novelty-replay.py`, `e1-output.txt`, `e1-robust*.txt`)
Day-forward replay 2026-09-28 to 10-16: each day's first-shown cards
(Recommended, Popular, Explore; 2,179 with outcomes, 19 days) are scored by
production trained on earlier votes, then blended within the day:
`(1-λ)·rankpct(score) + λ·rankpct(1 - max cos to earlier ups)`.
- Per-day top 12 of all views: at λ = 0.10-0.15, ups +0.21/+0.26, downs
  -0.05/+0.05, novel ups +0.16/+0.26 (day bootstrap CI excludes 0), pairwise
  diversity +0.02/+0.04 (CI excludes 0). λ 0.3 adds downs (+0.42); λ 0.5 loses
  hits (ups -1.1, downs +1.0).
- Placebo (novelty shuffled within day, 200 times): random perturbation at
  λ 0.15 gives ups -0.47..+0.26 and downs +0.05..+0.53. The real term sits at
  the good edge of both, so the novelty signal is more than noise.
- Recommended alone (where the term would live): up-vs-down AUC 0.769 → 0.783
  at λ 0.15. Top 12: ups +0.05, downs 0, novel +0.05/day (within placebo
  range). Top 6: ups -0.05 to -0.16 (within placebo range). The gain there is small.
- Limits: exposure-limited. Only shown cards have outcomes, and the shown
  Recommended pool was already chosen by the closeness-heavy model, so the
  replay cannot see far stories that were never shown. "Novel" uses the same
  similarity as the term, so novel-up AUC rises mechanically; ups/downs are the
  honest measures. Six λ values were tried on one sample.
- Verdict: a small novelty term (λ around 0.1) looks free on hits and adds a
  little spread. It is a candidate third interleaving arm or Combo companion
  after 2026-11-05, but this replay cannot prove its full-pool effect.

## E2b: side vectors on vs off (`side-ab-output.txt`)
User asked whether the side vectors could be removed (2026-10-16), then chose
to test removal offline. `scripts/eval_ranker_variants.py` on a frozen copy
of the live DB, profile 151, votes since gemma went live
(`--holdout-after` 2026-10-02 04:18 UTC, never used for tuning), 8 time blocks
each trained on all earlier votes; `production` vs
`prod[side_embedding_enabled=false]`.
- Impression pool (votes plus shown-unvoted cards): AUC up vs rest 0.791 vs
  0.776, up vs down 0.891 vs 0.880, NDCG@12 0.474 vs 0.405, P@12 0.44 vs 0.38;
  removal wins 2/8 blocks on the composite. Down share at 40 is 0.147 vs 0.128.
- Judged-only pool: AUC 0.791 vs 0.776, NDCG@12 0.488 vs 0.441; removal wins
  3/8.
- Verdict: the 2026-10-02 gain (AUC +0.011) holds out of sample (+0.015).
  Keep the side vectors.
