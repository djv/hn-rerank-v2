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
