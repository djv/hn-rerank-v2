TASK E5: revote/exposure-gaps.py — "which topics does the feed under-show relative to how often the user upvotes them?" No model training.

CLI: `uv run python revote/exposure-gaps.py USER SINCE_ISO OUT.json [--k 40]`.
- Universe: candidate-eligible stories that existed since SINCE (stories table, time >= SINCE, with a stored embedding; follow how ranking/eval builds the candidate pool, e.g. `_validated_embeddings`, and exclude what production would exclude if that is cheap to replicate; state approximations) plus every story the user voted on or was shown since SINCE.
- Clusters: MiniBatchKMeans(k, random_state=0, n_init=3) on the universe's stored embeddings (no scaling).
- Per cluster: pool share (share of universe), shown share (stories with an impression for USER, any view; also Recommended only), exposure ratio = shown share / pool share, voted counts up/down/neutral, up rate among shown-with-outcome (outcome rules as short-window-scores.py: vote after first showing; plus votes on never-shown stories counted separately as "found elsewhere"), Wilson 95% interval, mean maxsim to the user's ups before SINCE (how "far" the cluster is).
- Flag "under-shown hits": clusters with up-rate lower bound above the overall up rate and exposure ratio < 0.7. Flag "over-shown misses": up-rate upper bound below overall and exposure ratio > 1.3.
- Per source too (stories.source or however source is stored): pool share, shown share, up rate, Wilson interval.
- Stdout: cluster table sorted by exposure ratio, with cluster index, sizes and rates only. OUT.json: the same plus, per cluster, the 8 story titles nearest the centroid (private output, not printed), so the owner's agent can name clusters.
