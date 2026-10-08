# One-classifier feature removals, October 7

After the independent Claude review, the user explicitly selected twelve
chronological blocks for both promising development ablations. The plan
was recorded before those runs, with the selected joined logistic settings
held fixed. The same 2,231 judged test IDs, labels and production control
scores match across comparisons. Eight historical and four recent blocks
are disjoint within the study, but their labels were used in earlier
research. These are exploratory comparisons, not prospective confirmation.

| Classifier / inputs | Mean AUC | NDCG@12 | Upvotes /144 | Downvotes /144 |
|---|---:|---:|---:|---:|
| Production three-model blend | .8182 | .6656 | 94 | 2 |
| One joined logistic, all features | .8295 | .6944 | 97 | 6 |
| Same classifier, no words | .8319 | .6697 | 97 | 6 |
| Same classifier, no metadata | .8203 | .7094 | 98 | 5 |

No words means embeddings + metadata; no metadata means embeddings +
words. Both remove the two independent logistic models and replace the
original SVM with the same single joined classifier. Removing an input
block is a separate question from removing its former classifier.

## Removing words

Against the selected all-feature classifier, AUC delta +.00243 has a
3,000-draw stratified story 95% interval [-.00202,+.00714] and a
20,000-draw whole-block interval [-.00163,+.00624], seed20261007.
Exact AUC sign-flip p=.2617, with seven blocks better and five worse.
Top-12 precision delta is zero; story interval [-.04861,+.05556], block
[-.03472,+.03472]. Matching pooled counts does not prove equivalence.
NDCG@12 falls .6944→.6697: those matching counts hide differences in
where the liked stories appear within the first twelve cards. Its point
estimate is descriptive; no NDCG uncertainty interval was computed here.

Historical upvotes rise74→75 /96, while recent upvotes fall23→22 /48.
Downvotes stay three in each period. Recent AUC rises .8366→.8395;
historical AUC rises .8259→.8281. The same aggregate top-pick counts
therefore hide period and story-order differences.

## Removing metadata

Against the selected classifier, AUC delta -.00915 has story interval
[-.02072,+.00174], block interval [-.02692,+.00459], exact sign-flip
p=.3545; four blocks improve and eight worsen. Top-12 precision delta
is +.00694, with story interval [-.05556,+.06962] and block interval
[-.04861,+.06944]. Higher point top-pick counts do not establish a reliable
improvement, and the AUC intervals are also wide.

Historical upvotes fall74→72 /96 and downvotes fall3→2; AUC rises
.8259→.8273. Recent upvotes rise23→26 /48 but downvotes stay three,
and AUC falls .8366→.8064. The initial four-fold development improvement
did not carry through to the wider ordering in the recent blocks.
Mean NDCG@12 rises .6944→.7094, another top-pick versus wider-ordering
tradeoff; this is a point estimate without an NDCG confidence interval.

Metadata's larger squared norm is a geometric scale observation, not
evidence that it is useless or overwhelms learned predictions. Under these
fixed settings the wider comparison shows a tradeoff. Similarity/neighbor/
cluster metadata still derives from embeddings, and changing the encoder
changes both those columns and the raw embedding block.

## Verification boundary

[Claude's original review](CLAUDE-OPUS55-VERIFICATION.md) found no scoring
or leakage bug in the executed paths, but flagged provenance and a
statistics CLI period-label collision. [Focused verification of the
corrections](CLAUDE-OPUS55-CORRECTIONS.md) found no implementation blocker:
the CLI fix, served ranking, routing, dedup, feature-subset and placeholder
tests passed, and four old-driver reruns reconciled every score exactly.
The reviews were static; numeric/hash checks were run by the agent.

Source, configuration, orchestration, frozen DB and replay SHA256 hashes
stayed unchanged during reconciliation. Exact all-feature selected-baseline
reruns reproduced every candidate ID, label and score across all twelve
blocks. Supplementary Holm adjustment of the four AUC sign-flip tests is
complete; adjusted p values range from .252 to .785. Reused-label
model-search uncertainty remains unadjusted.

The updated full suite passed:1,128 tests /18 skipped. All21 modified/
intended Python files passed formatting; Ruff passed. Type checking retains
only the pre-existing inspection-script diagnostic. Near duplicates, content
added after the vote, probability calibration, live quality, equivalence
and serving performance remain unverified. Production and preview retain
their existing models; no replacement is selected.

Raw plan/reports/scores, paired summaries, manifest, source snapshots and
durable gate logs are retained in [`single-model/`](single-model/).
