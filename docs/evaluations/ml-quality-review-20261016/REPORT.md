# ML model quality review — 2026-10-16

User `rv 30x muse` on "hn rerank ml model quality improvements". Read-only.
30 Go Muse Spark runs via pi (xhigh, read/grep/find/ls), each in its own
copy of the tracked tree at `ef7dc5d` (no DB, large eval JSON pruned):
six angles x 5 repeats (labels, features, eval, diversity, coldstart, code).
Prompt: `common.md` + `angles.txt`; raw outputs `runs/`. All 30 completed
on the first try. Counts are runs raising the item (out of 30); "verified"
means the main agent checked it against the source.

## Verified, worth doing after the trial (2026-11-05)
1. Side-vector zero-fill for uncovered stories (11/30; verified, still live,
   already known from the 2026-10-14 codebase review).
   `pipeline/side_embeddings.py:179-188` joins `zero` for a missing gemma vector,
   so the joined row has norm 1/sqrt(2) and every cosine similarity to it shrinks.
   The uncovered stories are the newest ones (between side-timer runs), which
   hurts "stay current". Cheap fix: use the stored vector in both halves, or
   renormalize. Validate: replay with coverage forced low on fresh stories.
2. Scorer itself rewards sameness (5/5 diversity runs; by design). Meta features
   [385-389] are all closeness to past ups (max/mean kNN, positive clusters).
   All anti-sameness is in post-passes. The planned Combo (MMR-0.8 + Far-1,
   Phase 0b 2026-11-05) is the runs' unanimous first step; the second is a
   novelty term or interest-coverage penalty, judged on novel up-votes and
   topic spread.
3. Evaluation protocol gaps (5/5 eval runs). Offline replay is judged-only
   (voted stories), features use current text (comments arriving after the
   vote leak in), and the main metric suite has no diversity/novelty/coverage
   gate. Proposal: make the impressions/served-deck (`deck_scores`) pool the
   primary gate and add topic-spread and novel-up metrics before any further
   model change. The trial's per-vote tests ignore clustering (already known).
4. Cold/new-profile feed (7/30; verified). Tier-1 gravity is `s.score / age^1.8`
   (`pipeline/ranking.py:1769-1777`), so score-0 sources (RSS) sink to the
   bottom for new users. Gravity fades out completely by 50 votes
   (`tier2_blend_window`), and the model has no freshness/importance feature.
   Tier-2 is a raw centroid difference. Relevant to the parked "good first feed" goal.

## Verified, minor
- `neutral_weight` is a dead knob (11/30): defined at `pipeline/config.py:36`,
  never read. Neutrals train as a full balanced third class. Delete or wire it.
- Positive-cluster similarity [389] has no train-time self-exclusion (7/30),
  unlike the kNN features. Each up is about 1/200 of its center for profile 151,
  so the leak is small. It matters more for small profiles.
- Dummy zero rows for an absent class enter the StandardScaler fit (5/30;
  `ranking.py:1575-1603`). This can only trigger for a profile with no neutrals,
  since the dual gate guarantees up and down. Tiny.
- Zero-vote `rank_stories` sorts by raw points (`ranking.py:1886`), while the
  served cold deck uses `cold_ranked_candidates`. This matters for offline evals only.
- ARCHITECTURE.md 3.3 still says C=0.1; live `svm_c = 4.0` with the linear blend
  since `26b9474` (5/30). Doc drift only.
- Unsure entropy uses softmaxed SVM margins (7/30). This is documented as
  uncalibrated. Calibrate before any Thompson-style exploration.

## Already tried; runs re-proposed despite instructions
- Training URL dedup (5/30): rejected in the 2026-09-22 confirmation
  (FINDINGS "Reserved training-dedup confirmation").
- Recency decay: null 2026-09-29, 10-02 and 10-11.
- Rank-percentile blend vs margin calibration (5/30): the percentile blend was
  the variant that won (linear_blend.py docstring). A calibrated-margin variant
  is untested but low priority.

## Not verified (single run or unchecked)
Unsure starving Discover (run18), interest picks taken against the pre-draft
top 12 (run30), candidate-pool embeddings stale within a generation
(runs 27-29), and implicit skips as weak negatives (runs 1/5; run02 says a
small test was null). Dwell weighting has eval plumbing
(`scripts/eval_ranker_variants.py:863`); the 2026-10-27 dwell check is the
existing gate.
