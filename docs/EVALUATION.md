# Ranker evaluation

The ranker learns one profile's up, neutral and down votes. Evaluation separates
retrospective preference recovery from live recommendation outcomes. The results
below concern one established profile; benefit for other profiles is unverified.
Private feedback and frozen database snapshots are not distributed.

## Offline protocol

Votes are grouped by timestamp and held out in chronological blocks. Each block
trains only on strictly earlier timestamp groups, with an expanding training
window. Equal timestamps stay together. The general evaluator reserves the latest
20% of timestamp groups unless confirmation mode is selected.[^protocol]

Normalized article URLs isolate training from test: a training cross-post cannot
reappear as a held-out article. Repeated held-out URLs count once, using the first
timestamp and then an ID tie-break. Missing URLs use story identity. Different
URLs for semantic duplicates can remain. Metadata scaling fits on training rows
only; raw normalized embedding dimensions are not standard-scaled.[^protocol]

The evaluator takes a consistent read-only SQLite snapshot and reuses cached
embeddings. Judged-only pools contain previously voted stories, including neutral
and down votes. Current-pool recovery and shown-but-unvoted sensitivity checks
answer different questions; their absolute metrics must not be pooled.
See [evaluation modes and commands](RANKER_EVALUATION.md) for implementation details.

## Metrics and interpretation

- **AUC, up versus rest:** fraction of up/non-up pairs ordered correctly, with
  half credit for ties. Up versus down is a separate endpoint. Ordinal AUC also
  compares neutral with down and must be labelled separately.
- **NDCG@k:** rewards placing held-out upvotes nearer the top; neutral/down have
  zero relevance. **MAP** summarizes precision at successive recovered upvotes.
- **Precision/recall@k:** upvote share of top slots and share of eligible upvotes
  recovered. Top-slot downvote counts provide a separate rejection guardrail.
- **Coverage and uncertainty:** report judged/positive counts and fold results.
  Paired story and whole-block resampling give different uncertainty estimates;
  neither accounts for an entire repeated model search.[^broad]

These are current-content retrospective tests. Stored text and engagement can
postdate a vote; feedback mutation time only approximates original chronology.
URL isolation and label-shuffle checks cannot rule out every leakage mechanism.
Judged-only pools lack unjudged distractors and historical exposure reconstruction.
High AUC or NDCG therefore does not establish a win over Hacker News.

## Model choice and headline results

Earlier serving combined an RBF SVM with dense and TF-IDF logistic models.
The simplification study tried linear/RBF/hybrid SVMs, word kernels, joined
logistic regression, LinearSVC, a small MLP and projected gradient boosting.
Joined logistic was the strongest new development-AUC candidate. It combines
embeddings, numeric metadata and words, ranking by P(up) minus P(down).[^broad]

On twelve chronological blocks with 2,231 distinct judged test stories, joined
logistic had mean up-versus-rest AUC **0.8295**, versus **0.8182** for the blend.
Its top-twelve slots contained **97 upvotes and 6 downvotes out of 144**, versus
**94 and 2**. Better overall discrimination came with more rejected top picks;
the reused selection labels make this exploratory evidence.[^broad]

A separate time-split primary-input study reported pooled within-block held-out
up-versus-rest AUC **0.779843** (about **0.78**) for its native baseline.
Removing comments from primary embeddings gave **0.778184**, with **29 upvotes
and 1 downvote in 48 top slots** for both. It failed its improvement gates.
This is a different cohort/protocol, not the joined model's headline score.[^input]

Feature removals were mixed: removing words from joined logistic changed mean
AUC **0.8295 → 0.8319** but NDCG@12 **0.6944 → 0.6697**. Removing metadata
gave **0.8203** AUC and **0.7094** NDCG@12. No clean dominance followed.[^features]
The all-feature model retains semantic and lexical signals without requiring
separate preference classifiers; its measured tradeoffs remain explicit.

Matched encoder development comparisons favored stored embeddings plus Gemma
side vectors: joined-model AUC **0.7811**, versus **0.7683** on stored embeddings
alone. Those cohorts overlap previous research and are not prospective proof.
The checked-in configuration selects joined logistic with all features and
optional side vectors; the fresh-clone example leaves side vectors off.[^broad]

## Live outcomes

Team-draft interleaving compared joined-all and a no-metadata challenger with
the older blend inside Recommended. Votes were credited using the last prior
impression, window and recorded deck version. This did **not** compare with the
HN front page. The trial ended early, after **8.3 of 28 planned days**.[^live]

Joined-all received **61** credited votes versus **72** for the blend: upvote
rates **42.6% versus 36.1%**, downvote rates **8.2% versus 19.4%**. The up-rate
difference was **+6.5 percentage points**, with a descriptive **95% Newcombe
interval [−9.9, +22.6] points**, **p = 0.44**. Adopting the simpler classifier
was an engineering tradeoff, not a statistically established win.[^live]

Separate served-view telemetry reported Recommended up shares of **24–42%**
depending on surface, versus **8–13%** for Popular with roughly **60% down**.
These are selected exposures, not randomized HN-front-page outcomes; the report
does not give a matching Recommended down-rate. A controlled ranked-feed versus
HN-front-page up/down-rate claim is unsupported.[^stats]

## Sources (not in the public copy)

Report filenames only; private snapshots and row-level outcomes are omitted.

[^protocol]: `SINGLE-MODEL.md` and `BROADER.md` (chronological/URL-isolated protocol).
[^broad]: `BROAD-ONE-CLASSIFIER.md` (classifier and matched encoder comparisons).
[^input]: `REPORT.md` (primary embedding input comparison, October 9, 2026).
[^features]: `FEATURE-REMOVALS.md` (fixed-classifier ablations).
[^live]: `INTERLEAVING.md` (early endpoint, October 16, 2026).
[^stats]: `README.md` (feed statistics, October 17, 2026).
