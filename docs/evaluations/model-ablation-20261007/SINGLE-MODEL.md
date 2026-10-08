# One preference classifier: October 7 evaluation

The user selected replacing both dense logistic regression and the word
model with one preference classifier. The strongest initial kernel result
was a single linear-kernel SVM. It was close to the current blend, but these reused
retrospective labels do not establish unchanged or better live quality.
Production and the existing preview retain all three preference models.
The evaluation covers profile 151 with many votes, not cold-start behavior
or other profiles.

## Main comparison: 12 chronological blocks

Same frozen 3,848-feedback snapshot, current stored + Gemma side vectors,
production feature construction, training-only metadata scaling, expanding
training windows and normalized-URL isolation. Eight historical blocks and
four recent blocks contain 2,231 distinct judged test stories. Their test
IDs are disjoint; the recent shown/voted check is reported separately.

| Classifier | Mean AUC | Top-12 upvotes / 144 | Top-12 downvotes / 144 |
|---|---:|---:|---:|
| Current three-model blend | .8182 | 94 | 2 |
| Single linear SVM, C4 | .8200 | 95 | 3 |
| Single hybrid SVM, linear share .6/C4 | .8142 | 95 | 4 |

Historical linear result: 73/96 upvotes versus 71/96, AUC .8176 versus .8171.
Recent linear result: 22/48 versus 23/48, AUC .8249 versus .8204. Recent
NDCG@12 also falls (.485 versus .528). The single linear classifier therefore
has a period-dependent tradeoff, despite a slightly better pooled AUC.

Paired intervals use 3,000 within-block stratified story draws and 20,000
whole-block draws, seed 20261007. Differences are single model minus blend.

| Comparison | AUC difference | Story 95% interval | Block 95% interval |
|---|---:|---:|---:|
| Linear SVM | +.00178 | [-.00637, +.00947] | [-.00888, +.01329] |
| Hybrid SVM | -.00404 | [-.01104, +.00259] | [-.01315, +.00586] |

Linear top-12 precision difference is +.00694; story interval
[-.04861, +.07639], block interval [-.05556, +.06250]. Neither absence of
significance nor close point estimates proves equivalence. Selection used
overlapping historical labels, so these intervals do not account for the
full search or prove prospective non-inferiority.

## Kernel screens and Claude's suggestions

Four development folds screened 12 RBF settings, six normalized hybrid
settings and three linear settings. Full-blend score arrays were exactly
equal across all initial kernel families; hybrid scores differed from the
same-config RBF control. Routing tests also confirm that the full blend
retains its original classifier while single-model variants get the kernel.

The best initial RBF (C16/gamma .01) had AUC .8040 versus .8099 for the
blend. Linear C4 was strongest at .8094, matching 37/48 top-12 upvotes.
No initial native candidate met every predeclared point-metric criterion.

[Claude's original read-only review](CLAUDE-REVIEW.md) was received before
the final checks. Its conceptual point stands: a hybrid kernel can represent
linear functions, but it retains SVM hinge loss, lacks explicit word/domain
tokens, and does not reproduce the averaging of independently trained models.
Earlier screen numbers in that review are superseded by this report.

Two review hypotheses were screened before their outputs were examined:

- Anchored `RBF + (linear_C/C) * raw_linear_kernel`, with RBF C16/gamma .01
  and linear C .03/.1/.3: AUC .8045/.8054/.8092. The strongest AUC setting
  retrieves 36/48 upvotes and 2 downvotes versus the blend's 37 and 1.
- Continuous signed pairwise margins, rather than OVR vote-count tiers:
  best AUC .8071, 36/48 upvotes, 1 downvote. Rank-only comparison;
  softmax-transformed margins are not calibrated probabilities.

Neither screen beat every blend metric. The user subsequently selected a
TF-IDF kernel inside one classifier; the separately fixed extension follows
below.

## Embeddings and settings

Seven cached embedding combinations used the same 3,234 text-hash-matched
feedback cohort: mxbai alone, Gemma 1 alone, Gemma 2 classification/document
prompts alone, and stored mxbai concatenated with each Gemma alternative.
Use the full blend on stored + Gemma 1 as the cross-embedding incumbent;
full-blend results under a challenger encoder are not that incumbent.

On three development folds, one RBF on the existing stored + Gemma 1
vectors (C16/gamma .01) improves AUC .7686 → .7703 and top-12 upvotes
21/36 → 22/36, with zero downvotes in both. On the reserved latest 20%,
AUC .8678 → .8677 and upvotes 34/48 → 36/48, with one downvote in both.
This matched older cohort differs from the complete-profile main study;
the reserved votes overlap previous research and are not unseen evidence.

Gemma 2 alternatives did not beat the incumbent's AUC as a single model.
On the smaller 2,451-story token cohort, mxbai 512-token single-model AUC
.7045 trails the incumbent full blend's .7133 and the existing-embedding
single model's .7211. Its better top-12 count is a tradeoff, not dominance.

## GPU extension

Gemma 1 classification vectors at 256 tokens were computed on the laptop's
Intel UHD GPU through OpenVINO: 3,234 finite unit-normalized vectors in
867 seconds. Process-specific i915 counters measured **95.4% GPU render
utilization** over a three-second sample. Encoding ran on AC with background
resource caps; SVM fitting remains scikit-learn CPU work.

On the same three development folds, the stronger 256-token single RBF
(C16/gamma .01) has AUC **.7663** versus the incumbent full blend's **.7686**,
and 25/36 top-12 upvotes versus 21/36, with zero downvotes in both. The
128-token single model has AUC .7703 and 22/36 upvotes. Thus 256 tokens
trades AUC for more top picks and fails the all-metric screening criterion;
no reserved 256-token check was selected. GPU counter evidence, encoding
source/log and raw screen outputs are retained in `single-model/`.

Across the seven existing-embedding matched blocks, the 128-token single
model gains 3/84 upvotes (58 versus 55) and AUC +.00071. Story/block AUC
intervals are [-.00711,+.00853] / [-.00805,+.00957]; precision intervals
[-.03571,+.10714] / [-.02381,+.11905]. This remains exploratory evidence.

## Exact words inside one classifier

The eight-setting screen compared normalized linear C4 and RBF C16/gamma
.01, each with effective word C .1/1/4/16. The strongest development result
was linear + word C4: AUC .8139 versus .8099, 38/48 upvotes versus 37,
zero downvotes versus one. All production control arrays exactly matched
the prior native screen. The two highest-AUC settings were checked on the
same twelve blocks and 2,231 disjoint judged IDs.

| One classifier | Mean AUC | Upvotes /144 | Downvotes /144 |
|---|---:|---:|---:|
| Current blend | .8182 | 94 | 2 |
| Linear + word C4 | .8251 | 93 | 6 |
| Linear + word C1 | .8260 | 98 | 5 |

Word C1's recent result preserves 23/48 upvotes, increases downvotes
1→3, and improves AUC .8204→.8366; recent NDCG .5277→.5329. Word C4
has 22/48 upvotes and four downvotes despite AUC .8355. Exact words help
AUC, but the top-pick tradeoff changes by setting and period.

For word C1, AUC delta +.00783 has story interval [-.00051,+.01529]
and block interval [-.00320,+.02148]; precision delta +.02778 has both
intervals approximately [-.03472,+.09028]. Word C4 AUC delta +.00688
has story interval [-.00046,+.01405], block [-.00378,+.01929]. These are
exploratory comparisons with reused labels and selection uncertainty.

The user accepts small measured losses and subsequently requested a wider
one-classifier bakeoff and a new [Claude Opus 5.5 review](CLAUDE-OPUS55-REVIEW.md).
The broader classifier/feature/embedding sweep finished on the laptop;
its paired summaries are complete in
[BROAD-ONE-CLASSIFIER.md](BROAD-ONE-CLASSIFIER.md). Independent reviews are
complete; subsequent [feature-removal checks](FEATURE-REMOVALS.md) report
the twelve-block tradeoffs and verification boundary.
New encoding uses the Intel GPU; scikit-learn fits use one background CPU.
Feedback rows are frozen within each fold to avoid reliance on unordered
SQLite reads, and new reports record training block norms and title overlap.

## Sanity checks, implementation and limits

The single linear finalist's two label permutations yielded AUC .5048 and
.5147, with top-12 upvote fractions .0833/.2083 versus pool prevalences
.2259/.2311. No obvious leakage signal in this small sanity check; it does
not rule out all retrospective-content or duplicate effects.

On recent shown/voted pools, the linear finalist and blend both retrieve
22/48 upvotes and one downvote. Judged AUC is .8249 versus .8208; including
unvoted cards as rest gives .8339 versus .8305. NDCG falls .485 versus .513.
These pools reuse recent labels and are excluded from the twelve-block
bootstrap.

Judged-only replay deliberately has a higher positive density than a live
candidate pool. High NDCG/AUC here are not claims of beating Hacker News or
of predicting future reading quality. Current stored text is used even for
old votes; normalized URLs are isolated, but semantic duplicates can remain.
Text-hash cohort selection can also change the population being measured.

`scripts/eval_single_preference_model.py` is offline-only. It disables the
whole linear blend for the single-model variants, preserves the original
full-blend controls, filters cohorts in memory and rejects external snapshot
combinations that bypass that filtering. No embedding StandardScaler, new
dependencies, DB migration, production model change or preview change.

Earlier local validation of the broader drivers: **1,120 passed /18 skipped**.
After the review corrections and added tests, the latest full gate is
**1,128 passed /18 skipped**; Ruff and all21 modified/intended Python
format checks pass. `ty` reports only the
pre-existing unrelated TLDR-inspection diagnostic at line86. Earlier VPS
validation covered the earlier kernel driver, before the broader adapters.

The [fixed plan](SINGLE-MODEL-PLAN.md), [raw outputs and paired summaries](single-model/)
and [Claude opinion](CLAUDE-REVIEW.md) are retained. Snapshot SHA-256:
`912bd080178158c1bee69a0ec863674667c66cae3f76dd5322c6895dc989f8c1`.
Fixed evaluation time: `1791401987.9873054`; recent cutoff `1790899200`.
Runtime latency/memory savings are not established by this study. A future
comparison on votes not used for selection is needed before treating one
classifier as preserving quality.
