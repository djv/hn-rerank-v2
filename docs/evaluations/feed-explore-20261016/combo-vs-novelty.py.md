# revote/combo-vs-novelty.py — Combo (MMR-0.8 + far card) vs novelty term

Day-forward top-K replay on `novelty-replay.py` outputs (E1.json rows +
E1.npz L2-normalized embeddings; the DB is never opened): per UTC day in
two scopes (all views pooled, Recommended only), seven rules (base,
nov10, nov15, mmr80, far1, combo, combo+nov10) pick K; reports per-day
ups/downs/neutrals/novel ups, mean pairwise cosine distance, and distinct
k=40 topic clusters, with means, day-block bootstrap deltas vs base
(2000 resamples, seed 0), and 200-draw placebo percentiles per rule.

Assumptions: outcome/maxsim semantics match novelty-replay.py (prevotes
already dropped; None maxsim = no earlier ups, counts as novel with
median novelty rank); production `mmr_filter` is threshold-discard, so
mmr80 is greedy lambda-MMR per diversity-phase0b.py; the far card scans
the full day order (no window times/horizon in the replay); ties break by
story_id; placebo is maxsim-shuffle for nov rules, same-size random swap
for mmr/far/combo, percentile = share of placebo deltas <= real delta.

Limitations: exposure-limited (only shown cards have outcomes; deeper or
unshown picks are unscored); short days (n <= K) make all rules identical;
clusters (seed 0, fit once) are descriptive; NaN means/CIs serialize null.
