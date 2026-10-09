**Recommendation:** have Muse label each story with a fixed set of user-independent "facets" (what kind of item it is, not what it's about). Then test whether those labels add anything on top of an out-of-fold production score, using a small temporal residual model. Unlike every earlier attempt, Muse never sees the user or any votes.

## Two strongest options

| | A. Closed-schema facet labels → residual model | B. LLM canonicalization ("what this item is", stripped of comments and boilerplate) → re-embed |
|---|---|---|
| Signal | Format, artifact type and framing: what the topic-heavy embedding blurs | A cleaner topic vector |
| Validity | Works on a labeled subset; the baseline stays fixed | The new vectors don't match the 3848 training vectors, so all of them would need re-encoding. That breaks the constraints |
| Quota | About 20 calls | 3848+ rewrites plus encoding |

Choose **A**. Contrastive interest profiles depend on the user, and they collapse back into the LLM-preference failure the audits already showed: Muse picks the more on-topic story even when it was down-voted.

**Why embeddings and TF-IDF miss this.** The production text is title + self text + 4000 article chars + up to 6000 comment chars, mean-pooled (`ranking.py:623`). The side vectors see 128 tokens. Both are dominated by topic. p-08 shows the gap: a withdrawal on GitHub and a withdrawal tweet are near-duplicates with opposite labels. TF-IDF partly covers this through `dom_*` and `src_*` tokens and "Ask/Show HN" (`linear_blend.py:45`). So a deterministic proxy is the key control.

## Design (`docs/evaluations/facet-residual-20261009/`)

- **`rubric.json`**: frozen and hashed before any calls. It sees title, domain, source and the first 1500 body chars, never comments or scores. Fields:
  - artifact {paper, repo/tool, product launch, news report, essay/opinion, personal narrative, how-to, question/discussion, social post, data/visual, media, other}
  - claim {new result, release, incident/event, argument, explainer, retrospective}
  - technical depth 0–3
  - first-hand vs aggregator/commentary
  - actionable 0–2
  - outrage/conflict framing 0–2
  - entity focus {company, person, government, OSS, science, consumer product, none}
  - That is about 35 one-hot dimensions. Output is strict JSON with opaque IDs and an "unclear" option per field (constrained abstention).
- **`oof_baseline.py`**: split the 3848 votes into 5 chronological blocks. Score each block 2–5 with the production config fitted only on earlier blocks (4 fits of about 14 s, 1 thread on the VPS). Convert scores to percentiles within each block, because fits on more data are calibrated differently. No story is scored by a fit that saw its own vote.
- **`sample.py`**: draw 280 stories from blocks 2–5, stratified by label × source × block. Using training labels for coverage is allowed. Add the 99 dev stories. Shuffle partitions and labels across batches.
- **`label_facets.py`**: Muse only. 20 calls × about 20 stories = 280 + 99 + 20 repeats of already-labeled stories, as a stability check. Mixed batches with the batch ID recorded.
- **`evaluate.py`**: ordinal multinomial logistic regression with C=1.0, balanced weights and frozen hyperparameters. Arms:
  1. OOF score only
  2. + deterministic proxy (domain class, title prefix, URL kind, `text_len`, source)
  3. + facets
  4. + proxy + facets
  5. + facets permuted within source, 200 times (null)
- **Tests:** forward chaining within the 280 (fit on blocks 2–3, test on 4–5; then fit on 2–4, test on 5). Second temporal test on the 99, using the existing replay scores. The facet arm is fitted on the 280 only.

## Risks

- **Representation vs artifact:** the proxy arm tests whether the LLM adds anything beyond cheap regexes. The permutation null tests the gain from extra dimensions alone. Fixed C rules out tuning.
- **Memorized popularity:** Muse may recognize famous stories. That isn't a vote leak, but the outrage facet may proxy HN score. Report it with and without that field.
- **Snapshot content:** labels use current content, which is the same footing for every arm.
- **Low power:** 280 stories give an AUC standard error of about 0.03, so this is a feasibility screen only. The 99 are dev data, so the whole result is exploratory. Reserve votes after 2026-10-09 for real validation.

## Go / no-go (frozen)

**GO to a larger labeling round (no production change)** only if all of these hold:
- Arm 3 beats arm 1 by ≥ +0.02 ordinal AUC, pooled over the forward test blocks.
- Arm 3 exceeds the 95th percentile of the permutation null.
- Arm 4 beats arm 2 by ≥ +0.01.
- The change on the 99 is ≥ 0.
- Facet stability is ≥ 85% on the repeats; fields below that are dropped before results, using the repeats alone.

**STOP the facet line** if arm 3 minus arm 1 is below +0.01, or arm 3 doesn't beat the proxy. No rubric revision or field picking after the results.

Code and results come back for the second review.
