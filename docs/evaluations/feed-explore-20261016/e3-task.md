TASK E3: revote/sameness-trend.py — "is the served feed getting more same-y over time, and do far stories get upvoted less?" No model training (fast).

CLI: `uv run python revote/sameness-trend.py USER SINCE_ISO OUT.json`.
- Pool: every story with an impression for USER since SINCE (interaction_events, event_type='impression'), any sort_mode. First showing per story (time, sort_mode, age_filter). Outcome per story with the same rules as short-window-scores.py (vote after first showing up/down/neutral; none; drop prevoted; drop REVOTE window).
- For each story: maxsim_up = max cosine to every story up-voted strictly before its first showing (vote time from feedback; stored embeddings). Skip stories without a stored embedding but count them.
- Per ISO week and per sort_mode (recommended, explore, popular, other): n shown, mean and median maxsim_up, share novel (maxsim_up < 0.60), mean pairwise cosine distance among that week's first-shown stories in that view (sample up to 400 if larger, seed 0), number of distinct clusters among them (MiniBatchKMeans k=40, random_state=0, fit once on all pool embeddings) divided by n, up/down/neutral/none counts, up rate among voted.
- Over the whole period and per view: outcome rates by maxsim_up bucket [<0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, >=0.8] with Wilson 95% intervals for up rate among voted and among all shown.
- Trend test: Spearman correlation of weekly mean maxsim_up (and weekly share novel) against week index, per view, with n weeks.
- Also the same weekly profile for the user's own up-votes: mean maxsim of each new up-vote to earlier ups (is what the user upvotes itself narrowing?).
- Write per-story rows (story_id, first_shown, view, window, outcome, maxsim_up, cluster) to OUT.json; print compact tables.
