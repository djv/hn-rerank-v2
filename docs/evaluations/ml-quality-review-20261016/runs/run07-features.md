## 1. Top findings/proposals, ranked

**P1. Zero-filled side vectors shrink norms for uncovered stories — defect.**
Evidence: `pipeline/side_embeddings.py:model_space` builds missing side parts as zeros then `side_by_side` scales by 1/√2; STATUS.md already flags "side-vector norm shrinkage." An uncovered story's joined vector has ~0.7× the norm of a covered one, distorting every RBF distance and k-NN sim for exactly the newest/long-tail stories discovery needs. Effect: saves time neutral-to-negative, discovery negative (tail penalized), sameness widens toward well-covered topics. Effort S: add missing-side indicator bit and/or fall back to stored-only scoring for uncovered rows instead of zeros. Validate: `scripts/eval_ranker_variants.py --candidate-pool heldout-feedback` with `--replay-embeddings`, metric NDCG@12 + novel-upvote count on `--holdout-after` fresh blocks; also slice by side-coverage flag. Confidence high.

**P2. Per-pool min-max/percentile applied three times washes out margin magnitude — defect.**
Evidence: `pipeline/ranking.py:1723` `_minmax01(raw_scores)`, tier blend `ranking.py:1791-1805` mixes that with max-normalized gravity and centroid scores, then `pipeline/linear_blend.py:187-189` re-ranks all three components to percentiles before the 0.5/0.2/0.3 blend. Outlier pools compress everyone; genuine strong likes become "rank 0.9 of a weak pool." `probs` via `_softmax_rows` over uncalibrated OVR margins (`ranking.py:1725`) then drives Unsure entropy — uncertainty sampling on uncalibrated numbers. Effect: less time saved (head slots noisier), sameness roughly neutral, calibration hurts Explore targeting. Effort S/M: keep one normalization at the end; pass raw margins into the blend (or z-score per user, not per pool); use joined-logistic probabilities for Unsure. Validate: same eval harness, metrics P@12 + down-rate in top-12 + Unsure hit-rate on fresh blocks. Confidence medium-high.

**P3. Single composed text dilutes title with 6k-char comment tail — untried improvement.**
Evidence: `ranking.py:compose_story_text` (title + self[:6000] + article[:4000] + comments[:6000]) into one 4096-token embed; `linear_blend.py:tfidf_text` uses title + first 5000 chars of `text_content`. Title signal drowns for discussion-heavy stories; prior work only tried equal-weight body+comments vectors (FINDINGS "Incremental hill-climb" step 6, rejected on P@12) and title-only TF-IDF (FINDINGS "TF-IDF sweep," worse — but that removed body entirely, not reweighted it). Effect: saves time positive, discovery positive if niche topics surface via titles, sameness neutral-to-positive. Effort M: title-upweighted embedding (e.g. 2× title vector mean, or title+body field pair) as an eval-only variant first; cache-safe since it needs a new `model_version`. Validate: `eval_ranker_variants.py` dev 8-fold + `--confirmation` newest-20%, metrics AUC + P@12 + non-HN/discovery upvotes. Confidence medium.

**P4. Redo stacking properly with out-of-fold content scores — untried (prior attempt flawed).**
Evidence: FINDINGS "Stacked ranker" rejected `stack[...]` but its own Codex review notes inner folds reused kNN features built from all outer-train labels, age used candidate vote time, and `comment_count_at_fetch` is always 0 on heldout rows — i.e. the negative result is not licensed. The live blend's fixed 0.5/0.2/0.3 weights double-count the source prior (dense LR trains on rows already containing source one-hots). Effect: time saved positive if top-12 precision rises; sameness depends on stacker inputs — constrain with diversity guardrails from diversity-20261012/PLAN.md gate. Effort M: fold-nested kNN/cluster features + tiny LR/GBM on (margins, TF-IDF score, source rates OOF). Validate: heldout-feedback 8-fold + fresh `--holdout-after` blocks, primary P@12/up-rate with paired bootstrap (`scripts/compare_eval_scores.py`), secondary down-share; require NDCG@12 not to fall (cf. FEATURE-REMOVALS.md words/metadata tradeoff). Confidence medium-low.

**P5. Down/neutral similarity block is noise-prone; no negative structure — untried improvement.**
Evidence: `ranking.py` trains 4 k-NN cols (mean-top10 + max per up/down) + up-only k-means k=4; down class is incoherent so `sim_to_downvoted`/`closest_downvoted` mostly measure "distance from a grab-bag." Neutrals get zero model weight (`neutral_weight=0.0`) yet occupy the middle of every pool; re-vote study (STATUS 2026-10-11) found neutral-as-down no gain. Effect: cutting dud-adjacent false positives saves time; sameness risk if down-sim currently suppresses whole topics — test novel-upvote count explicitly. Effort S: eval variants dropping down-sim cols, or adding down-cluster max / up-interest-membership (`interest_centers`, currently Explore-only) as model inputs. Validate: same harness, metrics P@12 + down-rate + novel upvotes (<0.60 sim). Confidence medium.

**P6. Cold-start/small-profile quality has no real measurement — evaluation gap.**
Evidence: dual gate + `alpha_3` ramp (`ranking.py:1786-1800`) matter only below ~80/80 votes; INTERLEAVING.md short-history curve proxies new users with profile 151's recent votes (labels reused, same user/topics). A new profile's first feed is an explicit Goal. Effect: directly serves the "first feed" aim; sameness unknown for cold users. Effort S: build a leave-users-out or early-history eval on small profiles (N=20–200 votes) measuring P@8 and topic entropy, before changing the ramp. Confidence high that the gap exists, medium on any fix.

**P7. Docs/config drift on `svm_c` — defect (docs).**
Evidence: ARCHITECTURE.md §3.3 claims `svm_c=0.1` pinned; `config.toml:37` live is `svm_c=4.0` (deployed per FINDINGS "Linear blend live"); `pipeline/config.py:25` default is 0.2. Stale tuning history invites a wrong "restore." Effort S: update §3.2–3.3. Confidence high. No ranking effect if fixed.

## 2. Checked / already tried — no new proposal

- Engagement metadata (score/comments) excluded: ARCH §3.2, hill-climb — keeps archive-inflation out; fine.
- Raw L2 embeddings never standard-scaled; scaler+clip only on meta cols (`ranking.py` ~1600s) — correct; leave alone.
- TF-IDF 1–2grams + domain tokens, C=4, w=0.3: FINDINGS TF-IDF sweep says keep; title-only worse.
- EmbeddingGemma-2 side model: FINDINGS 2026-10-07 no gain over Gemma-1; don't revisit.
- Joined logistic #2/#4: in live trial until 2026-11-05; outcome pending — no pre-emption.
- Pairwise/LambdaRank stacking at ~5k votes: surveyed as low value; P4 is a minimal retest, not a big ranker.
- Recency weighting, neutral-as-down: no gain on re-votes; dropped.
- MMR/topic caps/Far-1: diversity-20261012 Phase 0 gate failed except near-miss Far-1 (now live as far card); don't re-propose as ranker changes.
- Unevaluated field-level embedding mode (ARCH §3.2): noted; P3 subsumes it cheaply.
- Zero-row dummy classes + scaler fit and `publication_affinity` off: minor/uncertain, not ranked.

## 3. Open questions

- What fraction of live candidates currently hit the zero-side path, and do they skew non-HN/new? (Needs DB read; none here.)
- Which `svm_gamma` pairs with `svm_c=4.0` live (config shows only `svm_c`)? Confirm before any C/gamma work.
- Should Unsure switch to joined-logistic probabilities if #2/#4 wins, or stay SVM-based? Trial report should include Unsure hit-rate per arm.

