# Broader one-classifier results, October 7

The user requested tradeoffs only. All runs are offline; production and the
preview retain the original three classifiers. This is a retrospective
profile-151 study with reused selection labels, not prospective equivalence
or a claim of beating the Hacker News baseline.

## Classifier and feature screens

The fixed broader screen tried joined logistic regression, LinearSVC, a
32-unit MLP, projected histogram gradient boosting, and embedding-block
weights4/16 in the linear+word SVM. The strongest new development AUC was
joined logistic regression: C4, numeric scale sqrt(.1/4), word scale1,
embedding weight16, ranking by P(up)-P(down). That scalar weighting does
not standard-scale embedding dimensions. The same classifier/settings
were then used for the feature and encoder comparisons.

Four-fold feature screen, same candidates and production controls:

| Features in one logistic classifier | AUC | Upvotes /48 | Downvotes /48 |
|---|---:|---:|---:|
| Embeddings + metadata + words | .8191 | 37 | 2 |
| Embeddings + metadata, no words | .8200 | 36 | 2 |
| Words alone | .7982 | 36 | 1 |
| Embeddings + words, no metadata | .8214 | 40 | 1 |
| Metadata + words, no raw embedding block | .8076 | 37 | 1 |

The current blend has AUC .8099, 37 upvotes and one downvote on these
folds. Removing metadata looked promising in this development screen;
the subsequent [twelve-block feature check](FEATURE-REMOVALS.md) shows
an AUC/top-pick tradeoff. Dropping raw embeddings does
not remove semantic information from metadata: similarity/neighbor/cluster
features still derive from embeddings. Block norms (embedding1 versus
metadata approximately8.6 before weighting) describe geometric scale,
not learned predictive importance.

## Twelve chronological blocks

Eight historical plus four recent blocks contain 2,231 distinct judged
test stories. Controls match the earlier frozen production score arrays
exactly; candidates, labels and test IDs match between each comparison.

| Classifier | Mean AUC | Upvotes /144 | Downvotes /144 |
|---|---:|---:|---:|
| Current three-model blend | .8182 | 94 | 2 |
| Joined logistic, all features, embedding weight16 | .8295 | 97 | 6 |
| Linear+word SVM, word C4, embedding weight16 | .8262 | 94 | 6 |

The logistic model's historical counts are74/96 upvotes versus71, and
3 downvotes versus1. Recent counts are23/48 upvotes in both, but3
downvotes versus1; recent AUC .8366 versus .8204. Higher overall AUC
therefore does not establish unchanged top-pick quality.

Paired differences are challenger minus blend. With 3,000 within-block
stratified story draws and 20,000 whole-block draws (seed20261007), the
logistic AUC delta +.01125 has story95% interval [.00443,.01801] and
block interval [.00123,.02281]. Its exact block sign-flip p is .0654.
Top-12 precision delta +.02083 has story interval [-.03472,.09028]
and block interval [-.02778,.06944]. The different checks answer different
questions; none accounts for the full reused-label search.

The weighted SVM AUC delta +.00796 has story interval [-.00157,.01679]
and block interval [-.00401,.02077]. Its top-12 upvote count is unchanged
while downvotes rise2→6.

Title sensitivity excludes the ten test stories with a normalized title
matching any training title, then reranks both models on the remaining
candidate pools. Logistic AUC delta remains +.01125; upvotes are96
versus94, downvotes6 versus2. This rules out those exact title matches
as the sole explanation, not semantic duplicates or later-added content.
Two label-shuffle runs give logistic AUC .5109/.5304, versus the blend's
.5162/.5246. This small sanity check does not rule out all leakage.

## Matched encoder/input comparisons

All fourteen encoder combinations use the same3,234 text-hash-matched
feedback cohort and three chronological development folds. The incumbent
is the current blend on stored mxbai + Gemma1-128: AUC .7686,21/36
upvotes, zero downvotes. Each challenger encoder's own full-blend row is
not the cross-encoder incumbent.

| Encoder input to the same joined logistic model | AUC | Upvotes /36 | Downvotes /36 |
|---|---:|---:|---:|
| Stored + Gemma1-128 | .7811 | 25 | 1 |
| Stored + BGE title128 | .7807 | 23 | 0 |
| Stored + Gemma1-256 | .7787 | 26 | 1 |
| Stored + Gemma2 document128 | .7758 | 26 | 1 |
| Stored + Gemma2 classification128 | .7733 | 27 | 1 |
| Gemma1-128 | .7731 | 27 | 2 |
| Stored + BGE full256 | .7695 | 26 | 0 |
| Gemma2 document128 | .7688 | 26 | 1 |
| Stored + Granite full256 | .7687 | 23 | 1 |
| Stored mxbai | .7683 | 23 | 1 |
| BGE title128 | .7627 | 20 | 0 |
| Gemma2 classification128 | .7624 | 27 | 1 |
| BGE full256 | .7617 | 23 | 0 |
| Granite full256 | .7594 | 22 | 1 |

Stored+Gemma1 had the highest development AUC and received the exploratory
reserved latest20% check: AUC .8783 versus .8678,37/48 upvotes versus34,
one downvote in both. The seven matched blocks contain1,937 distinct
judged test IDs: AUC delta +.01139, story interval [.00315,.01998], block
[.00426,.01874]; upvotes62/84 versus55, downvotes2 versus1. Precision
delta +.08333 has story interval [-.02381,.14286] and block interval
[.01190,.15476]. These votes overlap prior research and are not unseen
prospective evidence. Do not pool these absolute metrics with the native
3,848-feedback study.

Gemma256, BGE full/title, and Granite full encodings ran on the laptop
Intel UHD GPU through OpenVINO; process-specific i915 counters confirmed
render-engine use. All vectors were finite and unit-normalized. Classifier
fits used one background CPU on AC. GPU encoding success does not imply
GPU classifier training or a measured serving speedup.

## Verification and artifacts

Local full suite:1,120 passed /18 skipped. Ruff and five touched Python
format checks pass. `ty` retains only the existing unrelated
`scripts/inspect_tldr_failures.py:86` diagnostic. The MLP reaches its
fixed300-iteration budget; its convergence warning limits conclusions
about that bounded configuration, not the entire model family.

[Independent read-only Claude Opus5.5 verification](CLAUDE-OPUS55-VERIFICATION.md)
found no leakage/scoring bug in the executed paths, but flagged incomplete
provenance and a statistics CLI period-label collision. The latter is fixed
with a regression test; new served-ranking, control-routing, deduplication,
feature-selection and missing-class checks pass (40 focused tests). All
four final-driver reruns reproduce every original challenger and control
score exactly, reconciling the mixed historic driver hashes. Source,
configuration, orchestration, frozen DB and replay hashes stayed unchanged.
[Claude's focused correction review](CLAUDE-OPUS55-CORRECTIONS.md)
found no implementation blocker. Numerical checks were run by the agent;
the independent reviews were static.

The user subsequently selected twelve-block checks of no-metadata and
no-word settings against the selected classifier and production blend.
Those followups were declared before running and remain exploratory because
their labels were used in previous work. They are complete with durable
logs:1,128 tests passed /18 skipped, Ruff and format pass (including all21
modified/intended Python files), ty retains only the prior unrelated
inspection-script diagnostic. Final selected-baseline parity reruns matched
every candidate ID, label and score across all twelve blocks, with stable
source/input hashes, resolving the review's remaining provenance concern. Raw
reports, score dumps, paired summary and driver source snapshots are in
[`single-model/`](single-model/). Frozen DB hash/time, initial kernel and
word studies, and study limitations are recorded in
[`SINGLE-MODEL.md`](SINGLE-MODEL.md). No runtime latency, memory saving,
calibration, live quality or equivalence claim is established.
