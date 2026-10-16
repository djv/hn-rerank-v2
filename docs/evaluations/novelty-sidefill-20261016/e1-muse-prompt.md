You are writing one read-only offline analysis script for the hn-rerank repository (current directory, a scratch copy; you cannot run code, so write carefully and keep it simple). The owner's agent will review and run it on the VPS against the live SQLite DB opened READ-ONLY (`file:...?mode=ro` / `Database(DB, read_only=True)`); the script must never write to the DB. Per-story outputs are private: write them only to the OUT path given on the command line.

Read first: AGENTS.md "Working rules" (never standard-scale raw embeddings; never modify the DB), ARCHITECTURE.md 3.2-3.5, pipeline/ranking.py (`_score_and_rank`), pipeline/side_embeddings.py, scripts/eval_ranker_variants.py (`FoldData`, `_production_scores`, `_validated_embeddings`, `_embedding_text_hashes`, `_fold_database`), and the two existing analysis scripts in revote/ (short-window-scores.py is the closest template: it re-scores shown cards with the production ranker trained only on votes cast before each UTC day; sameness-metrics.py defines "novel" up-votes as max cosine < 0.60 to every earlier up-vote). Reuse their helpers and conventions (DB path constant, REVOTE exclusion window, outcome rules, `sys.path.insert(0, ".")`, run from the live checkout with `uv run python`).

Style: Python 3.12+, stdlib + numpy/sklearn already in the project, type hints, dataclasses for rows, no silent failures (raise on unexpected states), print progress with flush=True, keep under ~350 lines. Put the script at revote/OUTNAME (named below) and also write revote/OUTNAME.md: 10-20 lines on what it computes, assumptions, and known limitations. Do not edit any other file. Finish with a short summary of the design choices and anything the reviewer should double-check.

TASK E1: revote/novelty-replay.py — "does a novelty term in the ranking keep hits while widening topics?"

CLI: `uv run python revote/novelty-replay.py USER SINCE_ISO OUT.json [--analyze-only]`.

Phase 1 (scoring, slow; skipped with --analyze-only, which reloads OUT.json and OUT.npz):
- Pool: every story with an impression for USER in sort_mode recommended, popular or explore since SINCE (interaction_events, event_type='impression'). First showing per story across views: time, view, window (age_filter).
- Outcome per story, same rules as short-window-scores.py: current vote if cast after the first showing ("up"/"down"/"neutral"), "none" if no vote, drop "prevoted"; drop votes in the REVOTE window.
- Day-forward like short-window-scores.py: for each UTC day with first showings, train the production ranker (`_production_scores`) on votes before that day and score the pool minus trained stories; keep scores of stories first shown that day. Also compute for each such story maxsim_up = max cosine similarity (stored production embeddings, already L2-normalized) to the up-voted training stories before that day.
- Save rows (story_id, day, view, window, outcome, score, maxsim_up) to OUT.json and the stories' stored embeddings to OUT.npz (keyed by story_id order) for diversity metrics.

Phase 2 (analysis, fast):
- For lambda in [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]: within each day, blended = (1-lambda)*rankpct(score) + lambda*rankpct(1-maxsim_up) (rank percentiles within that day's stories).
- Report per lambda, overall and per view: AUC up vs down; AUC up vs rest; AUC novel-up vs down (novel = maxsim_up < 0.60).
- Per-day top-K selection (K=12 and K=24, or all if fewer) from that day's first-shown stories by blended score: count of ups, downs, novel ups, mean pairwise cosine distance among the selected (from OUT.npz), and number of distinct 40-means topic clusters (fit MiniBatchKMeans k=40, random_state=0, on all pool embeddings once).
- Day-block bootstrap (2000 resamples of days, seed 0) 95% intervals for the deltas vs lambda=0 in ups@K, downs@K, novel@K, diversity@K.
- Print a compact table; include per-day n and a caveat line: outcomes exist only for stories that were shown (exposure-limited replay).
