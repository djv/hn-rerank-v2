# revote/sameness-trend.py — sameness trend (no model training)

Answers "is the served feed getting more same-y, and do far stories get
upvoted less?" Pool = every story with an impression since SINCE (any
sort_mode), first showing per story; outcome rules match
short-window-scores.py (vote after first showing, prevotes and REVOTE-window
votes dropped). maxsim_up = max cosine to stories up-voted strictly before
that showing, via stored L2-normalized embeddings (most-common model_version).
Reports weekly x view stats (mean/median maxsim_up, share novel < 0.60,
pairwise distance sampled to 400 seed 0, MiniBatchKMeans k=40 clusters fit
once, outcome counts), bucket outcome rates with Wilson 95% CIs, Spearman
trends of weekly mean/novelty vs week index, and the same profile for the
user's own up-votes. Read-only DB; per-story rows plus slim aggregate
tables to OUT.json (stdout gets aggregates only, never titles or IDs).

Assumptions: earliest impression is the "first showing" even across views;
unknown sort_modes group as "other"; up rate among voted = up/(up+down+neutral);
distinct-clusters/n divides by n shown (stories without vectors contribute 0).

Limitations: similarity needs stored vectors (skipped stories counted, not
imputed); exposure-limited (outcomes only for shown stories); weekly means are
unweighted across weeks in trends; clusters (k=40, seed 0) are descriptive.
