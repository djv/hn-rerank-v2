# One preference model: fixed experiment plan

User requested testing a modified nonlinear model to replace the linear
model, then included embedding alternatives/settings and selected removing
**both** dense logistic regression and word TF-IDF. Target: one SVM classifier.
Production and the existing preview stay unchanged during research.

Before examining the screen's rankings:

- Main screen: four historical chronological folds on all 3,848 current
  feedback rows. Current stored + Gemma 1 embeddings, fixed snapshot and
  evaluation clock from the prior ablation. Baseline is the exact live blend.
  Single-model RBF candidates: C in {1, 4, 16}, gamma in {.003, .01, .03, .1}.
- Single-model hybrid candidates: one jointly trained SVC, kernel
  `(1-share)*RBF + share*(X @ Y.T)/mean_training_squared_norm`, shares
  {.1, .3, .6}, C in {1,4}, gamma=.03. Average linear training diagonal is
  normalized to one. This is one classifier, not a blend of trained models.
  Raw embeddings remain unchanged; only production metadata is standardized.
- Linear-SVM diagnostic: same one-classifier training, normalized linear
  kernel, C in {.1, 1,4}. This checks whether nonlinear complexity is needed.
- Embeddings: matched text-hash cohorts from cached vectors. Compare stored
  mxbai (4096 tokens), Gemma 1 (128 tokens), Gemma 2 classification/document
  prompts (128 tokens), and stored + those Gemma vectors. A separate smaller
  matched cohort compares mxbai's 512-token setting. Cohorts and dimensions
  are audited before runs; do not compare unpaired aggregate scores.
- Primary screening metric: mean raw-pool AUC. Flag losses in P@12,
  downvote fraction@12 and NDCG@12. A candidate counts as promising only if
  all those point estimates are no worse than the paired full-blend baseline.
  If none passes, report that and recheck the strongest contenders anyway.
- Pick at most two main-screen contenders for eight historical folds and
  four recent chronological blocks. Recheck the strongest embedding candidate
  on its matched cohort's reserved latest 20 percent. Historical/recent
  labels have already informed prior research: these are exploratory reusable
  checks, not unseen confirmation or a deployment decision.
- Report paired story/bootstrap and whole-block intervals; do not interpret
  nonsignificance as equivalence. Add a small label-shuffle sanity check for
  the finalist; compare against the candidate pool's positive prevalence.

Implementation: `scripts/eval_single_preference_model.py` wraps the canonical
evaluator; original blend baselines retain original RBF scoring. Experimental
kernels affect only variants with dense contribution disabled. Feedback cohort
filtering happens in memory over read-only snapshots. Kernel, cohort and
baseline-routing tests must pass; no new model dependencies or encoder loads
are needed for cached embedding comparisons.

Custom/precomputed SVM kernel support is documented by
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
