# Ranker evaluation modes

Use `uv run python scripts/eval_ranker_variants.py --help`. Run serially,
with `nice -n 19` and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1` on a loaded machine. The evaluator snapshots SQLite
read-only, uses cached embeddings, and never updates production feedback.

## Distinct questions, distinct pools

- `--candidate-pool current` (default): can the current production pool recover
  historical held-out feedback? Sparse overlap is expected. Never interpret a
  Recent deck with no judged stories as either good or bad recommendation.
- `--candidate-pool heldout-feedback`: rank the deduplicated held-out feedback
  block itself, including up, neutral and down votes. This preserves older
  feedback regardless of whether it remains in the live candidate window.
  It measures retrospective discrimination among previously rated items,
  NOT retrieval/exposure quality. No unjudged distractors, no reconstructed
  historical feed; current stored text and engagement can postdate votes.
  Candidate caps and frozen embedding files are deliberately rejected here.

Both modes use strictly earlier timestamp groups for training and reserve
latest 20% of timestamp groups unless `--confirmation` is explicitly set.
They share production `dedup.normalize_url` article identity: training URL
cross-posts cannot become test cases or candidates; repeated held-out URL
articles count once (first timestamp, ID tie-break). Candidate duplicates
prefer the judged ID. Missing URLs use story ID. This does not resolve
semantic duplicates with different URLs, and does not deduplicate training
weights. The report records before/after group-isolation counts.

## Interpretation and safeguards

Report NDCG/recall at 10/12/40/100/200, MAP, eligible positive counts, judged
coverage, and per-fold results. Coverage warnings identify folds with fewer
than ten positives or twenty judged cards. These are minimum diagnostics,
not a statistical power guarantee. Undefined metrics remain null.

Do not compare raw metric levels between pools. Judged-only pools have high
positive prevalence and can produce NDCG above 0.8 without corresponding
live-feed quality. `--leak-check --leak-seeds 5` runs five independent label
permutations by default. Compare per-seed results with the random baseline
and prevalence-aware expected random NDCG. Elevated shuffled results require
investigation; neither one seed nor five seeds establish absence of leakage. Production and publication SVM softmax values
are not calibrated probabilities, so their Brier score stays null.

Example (local config must reference the intended DB and cached encoder):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  uv run python scripts/eval_ranker_variants.py --config eval.toml \
  --user-id 1 --variants publication_affinity --folds 3 \
  --candidate-pool heldout-feedback --leak-check --output /private/path/replay.json
```

Keep detailed reports (feedback IDs/configuration) outside Git; record only
aggregate results in FINDINGS.md. The source directory needs Git metadata
for report provenance; do not run a bare archive without initializing an
isolated source snapshot. Do not copy credentials into experiment folders.

## Hill-climbing and embedding comparisons (added 2026-09-25)

- Metrics now include `auc_up_vs_rest` / `auc_up_vs_down`: the share of
  (upvote, other) pairs ordered upvote-first over every judged card. 0.5 is
  random; it is far steadier than NDCG@k on ~350–700-card blocks.
- Variant names can carry `ModelConfig` overrides (`;`-separated, since
  `--variants` is comma-separated): `prod[svm_c=4.0;svm_gamma=0.05]` is
  production with those settings, `produd[...]` scores it
  softmax(up) − softmax(down), and `prodlr[...;lr_weight=0.6;knn_weight=0]`
  percentile-rank-blends it with `logreg_up_minus_down` (and optionally the
  kNN score).
- `--replay-embeddings FILE.npz` (heldout-feedback only) swaps in another
  model's vectors for every feedback story, text-hash checked. Repeat it to
  concatenate models (each part scaled by 1/√k). Make files with
  `scripts/encode_replay_embeddings.py` (`--repo`, `--pooling`,
  `--max-tokens`, `--prefix`, `--title-only`, or `--from-db` to export the
  stored production vectors). Non-384-d runs skip the dashboard-deck
  metrics, whose dedup is 384-d only; raw ranking metrics are unaffected.
- `scripts/summarize_eval_report.py REPORT.json ...` prints fold means, a
  composite (mean of AUC vs rest, MAP, NDCG@12, NDCG@40 and 1 − top-40
  downvote share) and how many folds beat production.
- `scripts/calibrate_rankings.py --db SNAPSHOT` asks you to order 5
  unvoted stories on which production and the challenger disagree near the
  top; every pair in a batch is one the rankers order differently, spread
  across topics. Scores are cached per snapshot/config, so restarts are
  quick. `--report` gives each ranker's pairwise agreement with you.
