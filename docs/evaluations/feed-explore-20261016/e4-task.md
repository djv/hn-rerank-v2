TASK E4: revote/combo-vs-novelty.py — "the planned Combo (MMR 0.8 + far card) vs a novelty term, on the same day-forward replay". Analysis only, no model training.

Input: the outputs of revote/novelty-replay.py (OUT.json rows story_id, day, view, window, outcome, score, maxsim_up, and OUT.npz with the stored embeddings in story_id order; read novelty-replay.py to get the exact format and load it the same way). CLI: `uv run python revote/combo-vs-novelty.py E1.json E1.npz OUT.json [--k 12]`.
- For each UTC day (all views pooled, and Recommended only), select K stories from that day's stories by each rule:
  base: top K by score.
  nov10, nov15: blended (1-l)*rankpct(score) + l*rankpct(1-maxsim_up), l = 0.10, 0.15 (same as novelty-replay.py).
  mmr80: greedy MMR with lambda 0.8 over the day's stories. Reuse the production formula and score normalisation from pipeline/ranking.py `mmr_filter` (read it; if it needs objects you cannot build, reimplement its arithmetic exactly and cite the lines).
  far1: base top K-1 plus the "far card" chosen as production does (pipeline/ranking.py `place_far_card` and its helpers; diversity-phase0b.py's FAR rule is a template). If production's choice needs data you do not have (e.g. window horizon), state the approximation.
  combo: mmr80 top K-1 plus far card. combo+nov10: combo applied to the nov10 blended score.
- Per rule: per-day ups, downs, neutrals, novel ups (maxsim_up < 0.60), mean pairwise cosine distance of the selection, distinct clusters (MiniBatchKMeans k=40, random_state=0 fit on all embeddings); means over days; deltas vs base with day-block bootstrap 95% intervals (2000 resamples, seed 0).
- Placebo: for each rule that uses maxsim or embeddings, compare against 200 within-day shuffles of maxsim_up (for nov rules) or a random swap of the same number of cards out of base top K (for mmr/far/combo: replace as many base cards as the rule replaced, chosen at random from the rest of the day's stories); report where the real delta in ups and downs falls in the placebo distribution (percentile).
- Print a compact table and the caveat: exposure-limited (only shown cards have outcomes).
- Write per-day per-rule aggregates (no story IDs needed) to OUT.json.
