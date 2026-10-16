# Diversity / discovery / exploration review (read-only)

## 1. Top findings and proposals, ranked

### 1. Combo re-rank: MMR-0.8 head + Far-1 tail — test after 11-05
Category: untried improvement. Evidence: `docs/evaluations/diversity-20261012/PLAN.md` Phase 0b; `FINDINGS.md` "Diversity Phase 0" (Far 1: −1.1 pts up, +0.049 diversity, 8 vs 4 novel ups; MMR 0.8: −0.3 pts, +0.040, fewer downs); `pipeline/ranking.py:2296` (Recommended is pure score order, no serving MMR), `:2373` (`place_far_card`), `pipeline/config.py:37` (`enable_mmr=false`). The combination was suggested by the data but never tested.
Effect: keeps hits (both halves were ~free on up-rate) while attacking both sameness modes — MMR spreads within-topic neighbours, the far tail adds one genuinely new topic. Save-time neutral-to-positive; discovery up; current/calmer unchanged.
Effort: S (one ranking function + team-draft arm). Validation: Phase 0b protocol verbatim — `~/revote/diversity-phase0.py`-style replay on post-2026-10-13 shown stories, day-old production models, k=8, paired bootstrap over days; gate diversity ≥+0.05, up diff ≥−2 (CI ≥−5), novel ups ≥ B; then a 3-week interleaving arm with novel-ups/week and down-rate guardrails. Confidence: medium.

### 2. Far card drops out when its one pick ages out, though other far stories are current — defect
Category: defect. Evidence: `docs/evaluations/diversity-20261012/FAR-CARD-REVIEW.md` M1; `pipeline/ranking.py:2373-2425` picks a single `FAR_ARM` card at warm time, `:2447-2451` falls back to plain top-12 when it ages out. Most bites in 12h/1d, where the best far story is old. Related L4: interleaved undrafted union stories are never far candidates.
Effect: discovery silently missing exactly in the stay-current windows; save-time/calmer neutral; fix strictly widens served topics.
Effort: S (ordered fallback list or serve-time pick from current tail + Discover). Validation: regression test (far pick aged out, second eligible current → far still served) plus `deck_scores` × served-impression join readout (~11-02: far up/down-rate vs cards 9–11). Confidence: high.

### 3. No-upvote profiles: everything counts as "far" — defect, hits new users
Category: defect. Evidence: FAR-CARD-REVIEW M2; `pipeline/ranking.py:2128-2142`/`2241-2245` (Discover = score order among `closest_up < 0.60`); with empty up-votes the similarity array is all zeros so every story qualifies, and `place_far_card` relabels ordinary card 12 as ✨ Discover. A new profile's first feed is an explicit goal (`STATUS.md`).
Effect: first feed shows a misleading "unlike your up-votes" card with zero diversity logic; fix (require `has_upvotes`, else no far/Discover placement) is neutral on hits, positive on trust/calmer.
Effort: S. Validation: unit test (zero-upvote context → no placement) + cold/new-profile deck render check. Confidence: high.

### 4. Yes, the model itself drives sameness — nearest-upvote features plus word model
Category: defect (by design, not a bug). Evidence: `pipeline/ranking.py:987-1104` (top-k mean + max sim to ups/downs and 4-cluster up-centroid sim are trained features), `:1414-1428` (same computation for candidates); `pipeline/linear_blend.py` (`tfidf_text`, hashed 1–2-grams learn the vocabulary of ups) and `pipeline/joined_classifier.py` (same words joined to embeddings); `FINDINGS.md` "Sameness and novel up-votes" (Recommended diversity 0.15–0.21 below random availability at equal topic entropy; median nearest-up sim 0.77 in "Because you upvoted" study). Tier-2 centroid (`sim_up − sim_down`) adds a second pull toward the up-centroid. Dedup at 0.87 (`pipeline/config.py:80`) only removes near-dupes, so the within-topic neighbour-picking survives.
Effect: any dud-cutting on this score narrows topics unless paired with §1/§5. Proposal is not "remove the features" (they carry the hits) but keep them and diversify at selection, or add an explicit novelty/uncertainty bonus (see §5).
Effort: n/a (diagnosis). Validation: ablate `closest_up`/cluster columns in `scripts/eval_ranker_variants.py` (`margin3_plus_cluster`-style) and read diversity + novel-up recall alongside NDCG. Confidence: high for the mechanism, medium for magnitudes.

### 5. Do not add Thompson sampling on Unsure entropy before calibrating — evaluation gap
Category: evaluation gap. Evidence: `ARCHITECTURE.md` 3.3 (Unsure uses softmax over uncalibrated SVM margins; "not calibrated probabilities"); `FINDINGS.md` "Do Unsure votes teach the ranker more?" (`scripts/eval_unsure_votes.py`: entropy picks never beat random; live Unsure-like 1/78 up); `pipeline/ranking.py:2221-2232` (Unsure = top-entropy pass). Thompson sampling over uncalibrated entropy (or the R4 topic bandit, which passed Phase 0 only at Discover's 10.4% level vs R1's 11.8%) would spend the exploration budget on noise.
Effect: protects save-time/calmer; exploration budget stays with proven Discover R1 until uncertainty is real. Effort: M (reliability curves on time folds; compare SVM-softmax vs joined-logistic fitted probs already kept for challengers, `pipeline/ranking.py:212`). Validation: `eval_unsure_votes.py` + calibration plot; gate: entropy lift over random with CI clear of zero *and* calibrated probs before any bandit. Confidence: medium.

### 6. Offline evals cannot reward variety — measurement gap
Category: evaluation gap. Evidence: `FINDINGS.md` "MMR diversity on the top 12" ("judged-only pool, so the metric cannot reward variety"); Diversity/Discover Phase 0 limits (pool = already-shown stories; shown-unvoted = not liked; snapshot embeddings include later `top_comments`; ~110/12 ups — directional only); `ARCHITECTURE.md` 3.13 (canonical `eval_ranker_variants.py` reports NDCG/AUC recovery of held-out votes).
Effect: without deck metrics every re-rank looks like a loss. Add mean pairwise 1−cosine, topic entropy, and novel-up recall@k to the eval alongside NDCG, computed on time folds with impression pools (`scripts/narrowing_report.py`, `scripts/preview_explore.py`, `~/revote/sameness-metrics.py` pattern). Effort: S/M. Validation: re-run MMR/Far rows with new columns; gate stays the Phase 0 triple. Confidence: high that the gap exists.

## 2. Checked: fine or already tried — dropped

- Serving MMR: tried, costs up-votes at every threshold; keep off (FINDINGS 2026-09-29).
- Topic caps (1–2 per 40 k-means topics): tried; worst trade (−8.1 pts for Cap 1); drop (Diversity Phase 0).
- Novel = farthest-from-all-votes: matches random (3.9% vs 4.5%); Discover R1 (best score among far, τ=0.60) already shipped and beats it (+6.4 pts) — keep R1, don't revisit R0/R2/R3 (Discover Phase 0).
- Score bars / rank caps on short windows: cut novel ups disproportionately and don't transfer (`_minmax01` per-pool); correctly parked until serve-time scores accumulate (Short-window check + `deck_scores` logging live since 2026-10-12).
- Interest round-robin and Discover exclusion of voted-dupes: sound as built (`ranking.py:2250-2274`, `take_unmatched`); no change proposed.
- Raw-embedding scaling: correctly only metadata is standardized and clipped (`ranking.py:1191-1199` + meta clip); no proposal touches this.
- `svm_c` doc drift (ARCHITECTURE says `config.toml` pins 0.1; this tree's `config.toml` `[hn_rewrite.model]` says 4.0 with blend + side embeddings on): noted, not a diversity claim; verify before any retune.

## 3. Open questions

- Does the far card cannibalize Discover measurement (often Discover pick #1)? The readout plan notes it; confirm the combined far+Discover read ~11-02.
- Should the novelty threshold τ stay 0.60 for Recommended's tail as well as Discover, or differ by window (long windows lost more under strong rules)?
- After calibration (§5), is there any residual role for Unsure, or does Teach stay parked?
- `config.toml` `sweep` provenance: which commit moved `svm_c` 0.1 → 4.0, and do Phase 0 numbers (day-old models) still describe the live blend?

