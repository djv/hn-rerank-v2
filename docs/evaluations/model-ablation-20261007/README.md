# Production blend ablation, 2026-10-07

Follow-up: [broader historical and exposure-pool comparison](BROADER.md)
finds small, period-dependent changes; the historical blocks favor keeping word.

Later work: [one preference classifier](SINGLE-MODEL.md), followed by the
[broader classifier, feature and encoder comparison](BROAD-ONE-CLASSIFIER.md).
The subsequent [twelve-block feature removals](FEATURE-REMOVALS.md) show
that removing words preserves counts but lowers NDCG, while removing
metadata trades wider ranking AUC for improved top-pick point estimates.
The user selected reporting tradeoffs only; production models are unchanged.

Question: can base, word, or content ranking be removed without losing metrics?
All computation ran on the VPS in the separate Why-this-story preview checkout.
Production configuration and service were not changed. Source SQLite was opened
with `mode=ro`, backed up once, then evaluated through the evaluator's own
read-only frozen snapshot. The preview's mutable DB was not used.

User 151: 3,848 feedback rows. Held out 698 votes since October 2 00:00 UTC
in four expanding chronological blocks; URL-group isolation left 695 candidates
(92 upvotes). Every block trained only on earlier votes. Current stored + Gemma
side vectors were used; the production scorer aborts on fit failure or insufficient
side-vector coverage, and neither occurred. This is retrospective judged-only
recovery using current stored content, not a causal test or full live feed replay.

Remaining weights were renormalized in their original proportions. No tuning.

| Variant | Base/content/word | Upvotes / 48 top slots | Downvotes / 48 | AUC | NDCG@12 |
|---|---|---:|---:|---:|---:|
| All three | .5/.2/.3 | 23 | 1 | .82037 | .52768 |
| Remove base | 0/.4/.6 | 22 | 3 | .80767 | .52017 |
| Remove word | .714286/.285714/0 | 24 | 1 | .82725 | .53672 |
| Remove content | .625/0/.375 | 20 | 3 | .81509 | .47031 |
| Base only | 1/0/0 | 22 | 3 | .82097 | .51878 |
| Content only | 0/1/0 | 24 | 2 | .83515 | .49918 |
| Word only | 0/0/1 | 19 | 5 | .77724 | .47989 |

No removal preserves the complete ordering. Remove-word top-12 overlap is
10/12, 10/12, 11/12, 8/12 across blocks. It wins/ties/loses P@12 in 1/2/1
blocks, and improves AUC in 2/4 blocks. It is the clearest removal candidate
on recent votes, but its apparent improvement is uncertain:

- Paired stratified story bootstrap, 2,000 draws: delta AUC +.00688,
  95% interval [-.00450, +.01782]; delta P@12 +.02083,
  interval [-.06250, +.12500].
- Whole-block bootstrap, 20,000 draws: delta AUC interval
  [-.00980, +.02763], P@12 [-.06250, +.12500].
- Four blocks are insufficient to establish equivalence or prove a gain.
  No practical equivalence margin was predeclared. Do not interpret a
  nonsignificant difference as unchanged performance.
- Base-only AUC is nearly identical, but its top 12 has one fewer upvote
  and two more downvotes. Content-only matches remove-word upvote count
  and has higher AUC, but has worse NDCG and more downvotes than the full blend.
- This experiment disables score contributions; current runtime still fits
  both linear models even with a zero weight. It does not measure the latency
  or memory savings of actually removing a model implementation.

Artifacts: `fresh.json` includes configuration, folds, provenance and all metrics;
`fresh-scores.json` contains paired scores; `summary.json` contains the paired
bootstrap results and ordering comparisons. Bootstrap methods reuse
`scripts/compare_eval_scores.py` with seed 20261007, judged labels and both
variants from each same-fold score dump. Raw lanes have no coverage warnings;
short/long slices and the historical recommended_1w lane have sparse positives,
so the conclusions above use the full judged pool only.

Frozen snapshot SHA-256:
`912bd080178158c1bee69a0ec863674667c66cae3f76dd5322c6895dc989f8c1`.
VPS artifacts and snapshot:
`/home/dev/hn-why-preview-20261007/.ablation-20261007/`.

Reproduction, using an isolated frozen DB path in the config:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.preview-batch /home/dev/.local/bin/uv run python scripts/eval_ranker_variants.py \
  --config .ablation-20261007/config.toml --user-id 151 \
  --candidate-pool heldout-feedback --holdout-after 1790899200 --holdout-blocks 4 \
  --variants 'production,prod[linear_blend_dense_weight=0.4;linear_blend_tfidf_weight=0.6],prod[linear_blend_dense_weight=0.2857142857142857;linear_blend_tfidf_weight=0],prod[linear_blend_dense_weight=0;linear_blend_tfidf_weight=0.375],prod[linear_blend_enabled=false],prod[linear_blend_dense_weight=1;linear_blend_tfidf_weight=0],prod[linear_blend_dense_weight=0;linear_blend_tfidf_weight=1]' \
  --output .ablation-20261007/fresh.json --dump-scores .ablation-20261007/fresh-scores.json
```
