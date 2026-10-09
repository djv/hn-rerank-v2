# Claude read-only representation review, 2026-10-09

Actual CLI response below. Review was requested before the user selected the
upvote endpoint. NEXT-EXPERIMENT.md supersedes ordinal gates below. Reviewer
timing estimates were not measured; a VPS job does not require laptop AC.
For ordinal AUC the denominator is all cross-label pairs, not discordant pairs.

**Verdict: run it.** The design is sound and cheap: no LLM calls, one re-encode of about 3.4k texts, and 8 fits of a few seconds each. No earlier run answers this question (DATA-AUDIT.md:92-107). Expect a likely null, since earlier encoder swaps gave deltas near 0. It is safe only with the constraints below.

**What the agent checked**
- **Baseline:** `oof_baseline.py` uses the real `_score_and_rank`, with chronological vote-time quintiles and blocks 2..5 fit on earlier blocks. All 3,079 test rows were scored with no vector drift (`oof_metrics.json`).
- **Endpoint:** the ordinal score is `prob_up − prob_down`. `time.time()` only feeds the gravity `score`, so the endpoint is deterministic. Production serves `score`, so this metric does not measure served ranking. That is fine for exploratory work; record it as a limit.
- **Upsert hazard:** `upsert_story` merges an existing row by keeping the longer of each text field, then recomposes `text_content` (`database.py:639-696`). Writing a story row with no comments would quietly restore the comments.
- **Warm-start leak:** `linear_blend.latest` and `joined_classifier.latest` are held in memory per user (`ranking.py:1617,1638`). In one process, the challenger's block-2 fit would start from the baseline's block-5 fit, which was trained on block-2 labels. The SVM cache is safe because its key includes the training vectors (`ranking.py:1325-1349`). The disk warm store is off in scripts.

**Minimum constraints**
1. Give each arm its own copy of `/tmp/opencode/facet-residual-20261009/snapshot.db` and its own fresh process. Run folds in the same order (2→5), keep threads at 1, and assert `warm_store.enabled()` is false.
2. Challenger: only call `upsert_embedding` to replace the vector stored under the existing key (story, model version, hash of the stored text) with the comment-free vector. Never call `upsert_story`. Assert that the stories and side-vector tables hash the same in both arms.
3. Rows whose text has no comments need no re-encode; count them. Leave the 51 mismatched rows unchanged and flag them, but first count them in the 3,848 per block and per train/test split. As a sensitivity check (not a gate), report the delta with them excluded from scoring only.
4. Check the encoder before relying on it: re-encode about 20 full-text rows and require cosine ≥ 0.9999 against the stored vectors, or stop. Use 1 CPU, check that vectors are unit length, and run on AC power.
5. Reproduce the baseline first: the new harness's baseline arm must match the probabilities in `oof_scores.json` for each story to within 1e-6, or stop before encoding.
6. Write a manifest per story: hash of the source text, hash of the comment-free text, and vector hash. Also record the ONNX/tokenizer hashes, `max_tokens=4096`, and the script hashes. The manifest is the only record of the swap, because the replaced vector keeps its old key.
7. Pooled ordinal AUC is Σ concordant / Σ discordant pairs across blocks 2–5. The paired bootstrap resamples stories within each block, with the same draws for both arms, at least 2,000 reps and a frozen seed; fits are held fixed. Report UP-vs-rest and the 516 title-only rows separately; neither is a gate.
8. Commit the gates (pooled ≥ +.01, ≥ 3/4 blocks positive, bootstrap CI lower bound > 0) and the script hashes before encoding. Missing any gate ends representation rewriting on this evidence. Dev99 is excluded, and nothing is deployed.

Unverified: how the 51 mismatches split across blocks, and whether the VPS snapshot copy still matches the hash the audit used.
