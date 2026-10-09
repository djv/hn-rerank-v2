**Endorse all five corrections, with these amendments.** The only thing that could still invalidate the experiment is a data-provenance issue, covered at the end.

**Sampling (frozen before any calls)**
- Cut the 3848 votes into 5 chronological blocks by vote time. Remove any of the 99 dev stories from the pool first.
- Use seed 20261009 to draw a uniform sample of 70 stories per block from blocks 2–5, with no labels involved (280 total). Then draw 20 repeats from the 379 the same way. Shuffle everything into 20 batches of about 20 stories.
- No redraws. If any training fold has fewer than 10 of a class, or a test block has fewer than 5, report "infeasible" and stop.

**Scores**
- Score block k with the production config fitted on blocks 1 to k−1.
- Convert to percentiles across all voted stories in that block, not just the 70 sampled. Do the same within the 99, using their replay scores.
- First confirm that the replay model's training cutoff falls before every one of the 99 votes.

**Splits and pooling**
- Each test block is used once:
  - test block 4, trained on blocks 2–3 (140 stories)
  - test block 5, trained on blocks 2–4 (210 stories)
  - test the 99, trained on all 280
- Ordinal AUC: among all pairs in the same test block with different labels, the share ordered correctly by P(up)−P(down), with ties counted as 0.5. Down < neutral < up.
- Pool blocks 4 and 5 by weighting each block by its pair count. Never compare pairs across blocks. Report the 99 separately.

**Arms**
- 0: native OOF score, no fitting
- 1: univariate 3-class multinomial
- 2: + proxy
- 3: + facets
- 4: + proxy + facets
- 5: the arm-3 model on permuted facets, 200 times. Each time, shuffle whole facet rows within source across train and test together. Keep scores and labels fixed, and use the same splits and C.
- Settings are frozen: C=1, balanced class weights.
- Drop a field if its repeat agreement is below 85%, decided from the repeats alone and before any evaluation.

**Uncertainty**
- Paired story bootstrap (2000 resamples, stratified by block only) for arm 3 − arm 1 and arm 4 − arm 2. Also report the delta for each block separately.
- The bootstrap ignores training-set variance, so the CI is too narrow, and that should be stated in the results.

**Gates**
- Keep the original gates and add arm 3 ≥ arm 0.
- If the thresholds pass but the CI includes 0, report "promising, inconclusive". GO only means a larger labeling round, with no production claim.
- Report the proxy arm's own gain. If the proxy adds nothing, the arm 4 − arm 2 gate is weak and should be flagged, not treated as a pass.

**Still at risk**
1. **Fundamental if the provenance check fails.** If the 99 overlap blocks 2–5 or the replay model saw them, the 99 test is contaminated. In that case, drop the 99 from the gates and keep it as a report only.
2. Up to 35 facet dimensions on 140–210 training rows with C=1 favour arm 1. The fixed C is conservative but leaves little power: the AUC standard error is about 0.05 per block, so a +0.02 gate is close to noise. The result can only justify more labeling.
3. If percentiles from the weaker early fits relate to labels differently from the replay model's, arms 1–4 are miscalibrated on the 99. Arm 0 is the check for this.
