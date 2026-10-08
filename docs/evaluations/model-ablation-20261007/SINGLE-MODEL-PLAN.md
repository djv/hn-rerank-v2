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

## GPU extension fixed before its output was examined

After the user authorized laptop execution and requested GPU use, add one
setting: Gemma 1 classification embeddings at 256 rather than 128 tokens,
using OpenVINO on the laptop's Intel UHD GPU. Keep the same 3,234 text-hash
matched feedback cohort and frozen input stories. Run the same two RBF
single-model settings (C4/gamma .03 and C16/gamma .01), with stored mxbai
concatenated to the new vectors, on three development folds. Compare to
the incumbent full blend with stored + Gemma 1 at 128 tokens, using paired
identical stories/labels. If it meets the original screening criterion,
recheck the strongest setting on the reserved latest 20 percent. GPU
inference changes the vectors; SVM fitting and metrics remain the canonical
CPU implementation. No production encoder or ranking settings change.

## Claude extension fixed before its output was examined

Screen two specific review hypotheses on the same four development folds:
anchored `RBF + (linear_C / C) * raw_linear_kernel` with RBF C16/gamma .01
and effective linear C in {.03,.1,.3}; and continuous pairwise margin
scoring for RBF at C4/gamma .03 and C16/gamma .01. The continuous class
score sums its signed one-versus-one margins rather than using OVR vote
counts. Full-blend controls retain their original kernels and score form.
These follow the initial screen, so report them as additional exploratory
search, with no automatic deployment or equivalence claim. The two main
screen finalists and twelve-block checks remain the originally selected
linear C4 and hybrid share .6/C4. A TF-IDF kernel inside one classifier
remains a separate untested hypothesis.

Secondary finalist check: repeat the linear C4 finalist on the four recent
shown/voted impression pools, including their unvoted cards. Report it
separately; it reuses the recent labels and is not an independent sample or
an addition to the twelve-block bootstrap.

## Word-kernel extension fixed before its output was examined

User chose words inside one preference classifier. Screen normalized linear
C4 and RBF C16/gamma .01, each with effective word C in {.1,1,4,16}, on
four development folds. Add a TF-IDF dot-product kernel to the semantic
kernel before fitting one SVM; remove both logistic classifiers entirely.
Reuse production hashed title/text/domain/source 1-2 grams. Retained columns
(minimum training document frequency two) and IDF are fitted on training
rows only. Candidate words cannot influence either. Preserve production
sample weights, labels, metadata scaling and scoring form.

Select at most two settings by development AUC, flag other metric losses,
and recheck eight historical plus four recent blocks with paired controls.
If no candidate dominates all screening metrics, report that explicitly.
Existing labels have informed prior experiments, so these are exploratory
checks; no equivalence claim or deployment follows automatically.

During the running word-kernel screen, the user clarified that small measured
losses are acceptable for one classifier. Keep the fixed metric reporting
and AUC-based finalist selection; report all losses rather than calling a
failed all-metric screen equivalent. This preference changes the practical
recommendation, not the test population or statistical evidence threshold.

## Broader one-classifier request and BGE extension

User requested classifier families/settings, input features, embeddings and
embedding models, with a new Claude Opus 5.5 review. Before inspecting new
BGE outputs, add cached BAAI/bge-small-en-v1.5 CLS embeddings on the exact
3,234-story matched cohort: full production embedding text at 256 tokens
and title-only at 128 tokens, no retrieval-query prefix. Encode on Intel GPU
through OpenVINO, normalize, validate hashes/IDs and finite unit vectors.
Compare each alone and concatenated with stored mxbai; use the incumbent
full blend on stored + Gemma 1 as control. Further classifier/feature grid
will be fixed after receiving the requested review and before its outputs.

## Classifier and feature grid after Opus 5.5 review

Run on the laptop after user changed the execution host; encoders use the
Intel GPU with measured counters, scikit-learn fits use one background CPU.
Freeze the same snapshot/time/splits and preserve unchanged full-blend controls.
No new dependencies or production model changes.

- Existing linear+word SVM: word C4, semantic C4, embedding squared-norm
  weights 4 and 16 (uniform scalar sqrt(weight), never StandardScaler).
- Joined multinomial logistic: C4, dense block scale sqrt(.1/4), word scale1,
  embedding weights1/4/16, score P(up)-P(down). Also raw joined C.1/1/4
  scored by P(up), to separate penalty/scoring choices.
- LinearSVC: joined numeric+words, C.1/1/4, embedding weight4, OVR margin.
- Bounded nonlinear diagnostics: histogram gradient boosting, training-only
  embedding PCA64 + word SVD32 + metadata, leaves7/15, l2=10, 150 iterations;
  MLP32 with the same projected inputs, alpha=1/C for C.1/1, max300iterations.
  Opus excluded trees/MLPs on raw sparse high-dimensional inputs; these are
  deliberately bounded projected diagnostics, not the suggested raw setup.
- Feature sensitivity: strongest non-SVM adapter setting by development AUC
  versus numeric-only, word-only, embedding+words and metadata+words at the
  same model/settings. Report channels explicitly; word preprocessing already
  includes domain/source tokens.
- Embeddings: compare stored/Gemma1/Gemma2 cached alternatives and BGE's new
  full256/title128 GPU vectors on the same 3,234-story text-hash cohort, each
  alone and with stored mxbai. Add Granite Small English R2 at256tokens on
  that cohort if its ONNX export is compatible with the Intel GPU. Retain
  stored+Gemma1 full blend as incumbent, not a challenger encoder's blend.
- Select strongest SVM and strongest other family for twelve-block paired
  checks; report losses in top12 and downvotes as well as AUC, and add a
  small shuffled-label check. Use same-title-overlap exclusions as a paired
  sensitivity report; current comments remain a retrospective limitation.

Opus's proposed AUC -.005, upvotes -3/144 and downvotes +2/144 are useful
practical reference margins, not user-approved deployment criteria or proof
of equivalence. A joint linear model can represent the sum of two linear
logit functions, but a joint fit does not generally learn the sum of two
independent fits, particularly with different rows, weights or penalties.
The production percentile blend is nonlinear and depends on the candidate
pool, so it cannot generally be reproduced exactly by a joint LR.
Likewise predicted probabilities are not empirically calibrated
merely because the classifier is logistic.

User subsequently chose "Show tradeoffs only" when asked about extra
downvotes. Complete comparisons and finalist checks for evidence, but do
not recommend or select a replacement for production yet.
