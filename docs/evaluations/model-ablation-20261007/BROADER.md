# Broader word-model ablation

The user selected a broader evaluation after the four recent blocks gave a
small, uncertain gain from removing word. No production change was requested.

Compared the exact full blend (.5 base/.2 content/.3 word) against removing
word and renormalizing (.714286 base/.285714 content). Both use current stored
and Gemma side vectors, the same feedback snapshot SHA-256 as the first study,
chronological expanding training, production scoring and URL-group isolation.
No new weights were tuned. All evaluation and uncertainty computation ran on
the VPS under background priority, capped CPU/RAM and single library threads.

Added eight historical blocks from August 23 through September 30. Their
held-out story IDs are disjoint from the four recent October 2–7 blocks.
The older run reserves the latest 20% of timestamp groups, then evaluates
the last half of the remaining chronological feedback in eight blocks.
Reports record each cutoff, train/test IDs, configuration and dirty-code hash.
Current stored content is retrospective; historical folds are not new,
prospectively collected confirmation evidence. Semantic duplicates can remain.

| Pool | Full blend upvotes / top slots | Without word | Full AUC | Without word AUC |
|---|---:|---:|---:|---:|
| Eight historical judged blocks | 71/96 | 69/96 | .81713 | .81074 |
| Four recent judged blocks | 23/48 | 24/48 | .82037 | .82725 |
| All twelve judged blocks | 94/144 | 93/144 | .81821 | .81624 |
| Four recent blocks + shown/unvoted | 22/48 | 23/48 | .83048 | .83639 |

The impression-pool AUC is upvote versus **all** other cards; it is not
directly comparable to judged-only AUC. The impression pool adds 176
shown-but-unvoted candidates to the recent 695 judged ones, for 871 total.
Unvoted stories remain unknown, not inferred downvotes; counting them as
not-upvoted is an exposure-proxy sensitivity analysis. This reuses the recent
votes and must not be pooled as four additional independent blocks.

Historical downvotes in the top slots rise from 1/96 to 3/96 without word;
recent judged and impression pools both retain 1/48. Combined judged downvotes
rise from 2/144 to 4/144. Historical NDCG@12 falls .73461 to .71125;
recent judged NDCG@12 rises .52768 to .53672; impression NDCG@12 rises
.51254 to .52346. The effect changes by period and is small in aggregate.
There is no evidence here that removing word reliably improves metrics, or
that all three models are required. Exact metric preservation is false.

Paired uncertainty is saved in `broader-summary.json`: stratified within-fold
story bootstrap (4,000 draws), whole-block bootstrap (20,000 draws), exact
fold sign-flips and correlated-AUC DeLong, using
`scripts/compare_eval_scores.py`, seed 20261007. Within-fold bootstrap treats
the saved ranking models as fixed; block bootstrap is coarser and the folds
share expanding training data. These are exploratory historical comparisons,
not proof of equivalence or causal reading quality. No equivalence margin was
predeclared. The raw-pool scorer aborts on failed fits/side-vector coverage;
the completed runs did not fall back. Sparse slice warnings remain in reports.

The twelve judged blocks cover 2,231 disjoint held-out story IDs. Without-word
minus full-blend AUC is -.00197: story-bootstrap 95% interval
[-.00736, +.00349], whole-block interval [-.01008, +.00767]. P@12 delta
is -.00694, with story interval [-.05556, +.05556] and block interval
[-.04861, +.04167]. AUC improves in 5/12 blocks without word, worsens in
7/12; P@12 improves/ties/worsens in 3/4/5 blocks. Historical AUC's
story-bootstrap interval is negative, but its whole-block interval includes
zero, illustrating sensitivity to the inference unit. No stable equivalence
claim follows from these broad intervals.

Artifacts: `development.json`, `development-scores.json`, `impressions.json`,
`impressions-scores.json`, original `fresh*.json`, and `broader-summary.json`.
No production configuration, DB, service or preview ranking was changed.

Reproduce with the frozen config/DB and the background wrapper from README:

```sh
uv run python scripts/eval_ranker_variants.py \
  --config .ablation-20261007/config.toml --user-id 151 \
  --candidate-pool heldout-feedback --folds 8 \
  --variants 'production,prod[linear_blend_dense_weight=0.2857142857142857;linear_blend_tfidf_weight=0]' \
  --output .ablation-20261007/development.json \
  --dump-scores .ablation-20261007/development-scores.json

uv run python scripts/eval_ranker_variants.py \
  --config .ablation-20261007/config.toml --user-id 151 \
  --candidate-pool impressions --holdout-after 1790899200 --holdout-blocks 4 \
  --now 1791401987.9873054 \
  --variants 'production,prod[linear_blend_dense_weight=0.2857142857142857;linear_blend_tfidf_weight=0]' \
  --output .ablation-20261007/impressions.json \
  --dump-scores .ablation-20261007/impressions-scores.json
```
