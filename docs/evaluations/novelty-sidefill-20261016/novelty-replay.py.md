# revote/novelty-replay.py — novelty term replay

Day-forward replay asking "does a novelty term keep hits while widening
topics?" Phase 1 re-scores each UTC day's first-shown cards with the production
ranker trained only on earlier votes (same FoldData/`_production_scores` shape
as short-window-scores.py) and records maxsim_up, the max cosine similarity to
earlier up-votes via stored L2-normalized embeddings. Phase 2 blends within-day
rank percentiles (`(1-l)*rankpct(score) + l*rankpct(1-maxsim_up)`) and reports
AUCs plus per-day top-12/24 hits, diversity, and 40-means topic spread with
day-block bootstrap (2000 resamples, seed 0) deltas vs lambda=0.

Assumptions: outcome rules match short-window-scores.py (prevotes dropped,
REVOTE-window votes excluded); "novel" is maxsim_up < 0.60; cards with no
earlier up-votes (null) count as novel with median novelty rank; DB is
read-only and per-story rows go only to OUT.json/OUT.npz.

Limitations: exposure-limited replay (outcomes only for shown stories); AUCs
pool within-day percentiles across days; tiny day slices report n/a; clusters
(k=40, seed 0, fit once) are descriptive, not causal.
