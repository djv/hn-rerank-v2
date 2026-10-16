# revote/side-zero-fill.py — fresh-story demotion from zero-filled side vectors

Compares production scores at full side coverage (baseline) against three
re-scores with the side vectors of stories younger than F minutes hidden:
production zero-fill, renormalize (hidden joined rows scaled to unit norm),
and kNN imputation (hidden side vector = L2-normalized mean of the 5 nearest
covered stories by stored-embedding cosine).

Assumptions: pool = `load_production_candidate_stories(exclude_feedback=True)`;
training = all non-REVOTE votes; `--now` defaults to max(time.time(), newest
story time); hidden = age < F; windows use `WINDOW_SECONDS` (12h/1d) over
story age; rank percentile = 100·(N−rank)/(N−1) (100 = top).
Hiding/imputation wrap only `SideVectorCache.get` / `side_embeddings.model_space`
in memory; truly side-missing stories keep production zero-fill everywhere.

Known limitations: single fold (no temporal split, unlike short-window-scores);
one rescoring per F×variant (SVM fit is cache-shared, inference is exact);
kNN neighbors come from the candidate pool only; if hiding trips the 0.98
coverage gate the run aborts loudly (production would fall back to stored-only);
cold/low-vote profiles exercise the tier path, not the SVM side branch.
Per-story rows are private: OUT.json only, never the repo.
