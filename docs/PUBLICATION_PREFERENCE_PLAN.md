# Publication preference: investigation and proposed experiment

## Implementation checkpoint

- Read-only audit implemented and live-tested. Host-based audit: 16 positive
  votes, 13 exact unique URLs, three duplicate issues; the earlier 17 was a
  title match, not a publication identity count.
- Experimental SVM features implemented behind an off-by-default flag,
  with strictly earlier training evidence, group exclusion and latest-vote
  deduplication. Current identity groups only normalized exact URLs, not all
  semantic cross-posts. Shared-host exclusions are conservative, not exhaustive.
- `publication_affinity` eval variant added; not yet benchmarked. Temporal
  folds still need duplicate-group overlap auditing before results can justify
  deployment. Diagnostic does not yet expose exact SVM margin attribution.
- No ranking changes deployed. Defaults and flag-off feature layout unchanged;
  model schema version changed to prevent stale fitted-model reuse.

## Objective

Learn repeated publication-level preferences without hardcoding Import AI,
forcing a top-10 slot, or replacing content relevance. Investigation only;
no ranking behavior changed.

## Verified current behavior

- Local and deployed `pipeline/ranking.py` have identical SHA-256 hashes.
- User-profile SELECT found 17 upvotes for titles matching Import AI, no
  neutral/downvotes. This is a title-based audit, not yet a deduplicated
  publication history; cross-posts and all jack-clark.net votes need auditing.
- Observed feed: Import AI 473 ranked 12th (0.978112); rank 10 was 0.978530.
  These are normalized ranking scores, not calibrated like probabilities.
- `Database.get_feedback_for_training` loads the user's saved up/neutral/down
  votes; no training-history time window in that query.
- `_score_and_rank` blends HN gravity, positive-minus-negative embedding
  centroids, and an RBF SVM. Defaults: centroid ramp over 50 votes; SVM gate
  at 20 up and 20 down; SVM blend reaches full weight at 80 of each. With
  this profile's thousands of votes, successful SVM execution should dominate;
  exact live trace should be checked before attributing individual scores.
- SVM inputs: unscaled normalized embeddings; metadata is log text length,
  mean top-k/nearest positive and negative similarities, positive-cluster
  similarity, and four source categories (HN live/archive/Reddit/RSS).
  Metadata alone is standardized and clipped. Training similarity features
  exclude self. Classes use balanced sample weights.
- There is no explicit publication/domain feature. RSS source IDs are reduced
  to a broad category. Title/content embeddings can implicitly encode series
  identity, so absence of an explicit feature does not prove the whole cause.
- SVM up-class margins are min-max normalized over the candidate pool.
  Softmax decision values used elsewhere are also not calibrated probabilities.
- Deck assembly determines membership (and optional diversity); the terminal
  feed then sorts Recommended by rank score (`pipeline/render.py`). Do not
  mistake deck membership or UI filtering for a publisher penalty.
- Embedding composition uses bounded leading text (6k self-text, 4k article,
  6k comments), not the new TLDR. Changing the summary does not directly fix
  ranking; multi-topic newsletters may have lead-biased representations.

## Proposed implementation, gated by evidence

### 1. Reproducible diagnostic

Add a read-only diagnostic script with typed output, using an explicit user
ID (never log profile tokens). Show training counts/tier, raw SVM margin,
normalized score, nearest positive/negative examples, source identity,
publication vote counts, and candidate/deck positions. Audit all votes for
jack-clark.net, cross-post duplication, and the actual embedding input.
Use cached embeddings; avoid production retraining loops and remote model
loads. Snapshot or bounded read-only queries; no production DB migration.

### 2. Canonical publication identity

Create a typed `PublicationKey` at the boundary. For the first experiment use
conservatively normalized article hostname (lowercase, IDNA, remove www,
ignore port), preserving subdomains. The same external URL host must match
across RSS and HN. Missing/invalid URLs get no publication signal.
Do not equate platform hosts (HN, Reddit, GitHub, shared blog hosts) with one
publisher; start with explicit handling or no signal for ambiguous hosts.
Do not blindly collapse to registrable domains or assume RSS source slug is
an author. Include a documented identity-version in cache fingerprints.

### 3. Smoothed preference features, not a forced ranking boost

For each user's publication, count up/neutral/down votes. Estimate a
three-class posterior `(count_class + strength * prior_class) / (n + strength)`
using that user's class proportions as prior. Include posterior contrasts
(relative to the user prior) and bounded support/confidence as SVM metadata.
No-history publications must have zero contrast and zero support. Neutral
is its own class, not a dislike. Learn influence jointly with content rather
than adding an arbitrary amount to candidate-normalized scores.

Tune smoothing strength on chronological training/validation only. Avoid
assuming 17 correlated newsletter issues are 17 independent quality proofs.
Audit duplicate/cross-post feedback; use existing canonical identities where
available, and report both raw and deduplicated support.

Training rows must never include their own vote in publication counts OR
user-prior counts. Prefer chronological/out-of-fold feature construction;
at minimum leave out the complete canonical story group, not just one row.
All held-out evaluation features use training history only. Fit scalers on
training metadata only; never standard-scale embeddings.

Keep old behavior behind a feature flag for ablation/rollback. Bump model
schema/cache signatures; include publication identity and feature settings so
an old fitted model/scaler cannot be reused with the new feature layout.

### 4. Evaluation before deployment

Extend the existing temporal ranker evaluation, not a separate optimistic
in-sample benchmark. Compare unchanged production versus publication features
on identical folds/candidate pools and multiple time windows. Include:

- held-out NDCG@10 and liked-item Recall@10, with counts/uncertainty;
- returning-publication vs unseen-publication performance;
- neutral/downvote ranking and publication concentration/diversity;
- Import AI as a case study, not a tuning target or required top-10 outcome;
- time-forward, group-safe feature tests (self-vote, future votes, cross-posts);
- missing URLs, ambiguous platforms, IDNA/subdomains and unseen publications;
- model-cache invalidation, metadata-only scaling, and feature-flag parity.

Observed votes are exposure-biased; offline improvement cannot establish a
causal benefit for unseen items. Treat unusually high metrics as leakage
warnings. If identity features do not improve held-out ranking, do not ship
just because this one article moves up. Investigate newsletter embedding
coverage separately rather than conflating it with publication affinity.

### 5. Rollout

Run checks serially/niced with two pytest workers on this host. Enable only
after acceptable held-out results, restart and smoke-test the requesting
profile. Log score components and source concentration without credentials.
Rollback by disabling the flag and invalidating model cache; no feedback
or story deletion required.

## Remaining uncertainty

The 17 positive votes establish a consistent observed series preference, not
why each of the eleven current competitors scored higher. Exact margin
attribution and counterfactual rank movement remain unmeasured. No guarantee
of top-10 placement is justified before the diagnostic and held-out tests.
