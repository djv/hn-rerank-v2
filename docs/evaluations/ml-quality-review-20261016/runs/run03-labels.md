# Labels review: how votes become training rows

## 1. Top findings, ranked

**1. `updated_at` is not a vote timestamp — recency rests on a polluted column. Title: add `created_at`, stop using `updated_at` for decay.**
Category: defect. Evidence: `database.py:1338-1355` (`upsert_feedback` only bumps `updated_at` `WHERE action != excluded.action`, so same-action re-affirmation leaves time stale; flips move it); `database.py:1462-1529` (`get_feedback_for_training` returns `updated_at` as `vote_times`); `ROADMAP.md:299-311` (bulk touches put `updated_at` in last 60d; B1 "blocked, no true creation timestamp"); `pipeline/ranking.py:1258-1266` (production fetches `_vote_times` but never decays with it).
Effect on aims: time-decay done right saves time if taste drifts (re-votes: old votes flip 0.33 vs 0.14 new, `FINDINGS.md:842-877`) without narrowing topics if applied as a weight, not a filter. Effort: M (additive `created_at` + `vote_history` migration, backfill from first-seen `interaction_events`). Validation: `scripts/eval_ranker_variants.py` `half_life` variants on temporal splits (`_recency_decay`, `:787`; `_vote_weights`, `:1530`), metric AUC up-vs-rest + P@12 + novel up-votes; expect prior nulls to be re-tested only after clean times exist. Confidence: high the defect exists; medium that fixing enables any real recency test.

**2. `neutral_weight = 0.0` is dead — production weights neutrals equally. Title: wire or remove the knob.**
Category: defect (doc/code mismatch). Evidence: `pipeline/config.py:36` defines `neutral_weight = 0.0`; grep shows zero uses in `pipeline/`; actual weights are balanced `n/(classes*count)` in `pipeline/ranking.py:1594-1600` with 1e-6 dummies. `ARCHITECTURE.md:3.3` repeats `neutral_weight=0.0`.
Why it matters: a reader thinks neutrals are ignored; they are a full third class. Prior test says that is correct: neutral-as-down gives +0.001 held-out AUC but −0.040 on blind re-votes (`FINDINGS.md:842-877`, "Neutral as down"). So fix is documentation/code hygiene, not a ranking change. Topic spread: keeping neutral as its own class preserves the "maybe" middle that discovery needs. Effort: S. Validation: no eval needed; `ruff`/`ty` + config-attr test that the knob either scales class-1 weights or is deleted. Confidence: high.

**3. Flips erase history — one row per (user, story) discards disagreement. Title: log vote history, down-weight unstable labels.**
Category: untried improvement. Evidence: `database.py:702-704` PK `(user_id, story_id)`; re-vote overwrites; 46 blind re-votes agree only 59%, up/not-up flips 23% (`FINDINGS.md:842-877`); `STATUS.md:Next step` already quarantines the 29 calibration rows (`updated_at` 1791763112–1791763180) from fresh-vote evals because mutation breaks reproducibility.
Proposal (post-trial): additive `vote_history` table (every `upsert_feedback` appends), weight = balanced × (1 − flip-penalty) or drop second-guessed rows in a variant. Saves time (fewer noisy duds); sameness neutral if penalty is topic-blind. Effort: M. Validation: `eval_ranker_variants.py --candidate-pool heldout-feedback --split temporal`, compare `production` vs `prodlr[half_life…]`-style flip-weighted variant on AUC + down@12 + per-cluster coverage (`scripts/narrowing_report.py`). Confidence: low-medium; n=46 is thin and flips may be the signal (taste change), not noise.

**4. Cross-post double-counting is still on in production. Title: turn on URL-group training dedup.**
Category: untried (eval-tried, production-off). Evidence: `pipeline/config.py:30` `deduplicate_training_feedback = false`; `config.toml` sets no override; `pipeline/ranking.py:1274-1279` gates `pipeline/feedback.py:9` (`deduplicate_feedback`: latest vote per normalized URL). Eval: dedup NDCG@10 0.805→0.850 but MAP flat, gain in 1/3 folds; 5-seed follow-up mixed (`FINDINGS.md:3615-3666`).
Why rank it high: cheap (one flag), directly anti-same-y (same article via HN + Reddit RSS stops voting twice), reversible without DB rewrite. Effort: S. Validation: same temporal heldout-feedback replay with URL isolation on (already default in eval, `:3037`), metrics P@12/down@12 plus novel-up count; ship only if neutral on diversity. Confidence: medium-low; prior evidence is mixed, not rejected.

**5. Exposure bias is measured nowhere in the training loop. Title: position-aware offline eval before any IPS serving change.**
Category: evaluation gap. Evidence: training uses only voted rows (`database.py:1462`); impressions/opens/dwell live in `interaction_events` (`database.py:513-520`) with served `position`/`deck_version`, but no consumer; eval disclaimer "not historical exposure" (`scripts/eval_ranker_variants.py:3037`); `notes/improvement-brainstorm-2026-07-15.md:71-77` IPS/Boltzmann slots never built; `FINDINGS.md:2100-2155` impression-pool work adds shown-unvoted as not-up but without position/propensity.
Proposal: stratify offline metrics by served position/view from `deck_scores` + `interaction_events` (post-2026-10-12 `deck_scores` has per-card positions; pre-583d0bd TUI positions are list indexes — join on version, per `STATUS.md`). No serving change. Helps discovery explicitly (detects rankers that win by riding top slots). Effort: M. Validation: extend `_impression_pools` (`:1381`) + `ledger_report.py`; report P@12/AUC by position bucket and view. Confidence: medium the gap is real; low on near-term lift (single-user volume makes IPS high-variance).

**6. Long-dwell neutrals as soft ups — narrow retry of killed dwell idea. Title: selective dwell weight, not global dwell weight.**
Category: untried improvement (differs from rejected version). Evidence: global `margin3_dwell` killed (NDCG −0.005, worse than `margin3_up`; `ROADMAP.md:406-417`); `get_capped_dwell_by_story` (`database.py:1531`) exists for experiments only; new signal: long-dwell neutrals mean `prob_up` 0.49 vs 0.31, dwell AUC 0.82 not-down-vs-down but 0.66 up-vs-rest, n=28, view-confounded (`FINDINGS.md:233-287`).
Proposal: only bump `sample_weight` for neutrals with >15–30 s capped dwell, after the ~10-27 rerun (`STATUS.md:Next step`). Dwell is recorded while deciding so it can only weight, never be a feature. Sameness risk: Recommended cards get more dwell — must normalize by view/position or discovery narrows. Effort: S. Validation: `margin3_dwell`-style variant restricted to neutral-upweight on temporal splits; gate on down@12 AND novel-up rate, not AUC alone. Confidence: low.

## 2. Checked: fine or already tried

- Neutral-as-down tried, rejected — keep 3-class (`FINDINGS.md:842-877`).
- Global recency half-life 30/60/100000d tried, null CIs (`FINDINGS.md:842-877`, `:2100-2155`, `:3987`); blocked retroactively by finding 1.
- Shown-unvoted as down-30% / neutral tried, ±0.001 AUC, P@12 noise (`FINDINGS.md:2100-2155`); consistent with user voting nearly everything read (`FINDINGS.md:233-287`).
- Balanced class weights + 1e-6 dummy rows + meta-only StandardScaler with ±2.5 clip look correct (`ranking.py:1594-1610`); raw L2 embeddings untouched — constraint honored.
- Re-vote/upsert dedup to one row per story is intended; URL-group isolation in eval is sound (`eval_ranker_variants.py:3037`).
- `article_open`/`comments_open` too sparse (1–2 events) to train on — correctly unused; revisit after weeks of `deck_scores` + open telemetry.
- Missed-upvote audit (654/683 ups necessarily outside top slots) is a capacity description, not a label defect (`FINDINGS.md:1174-1206`).

## 3. Open questions

- Should `upsert_feedback` bump `updated_at` on same-action re-votes, or is `updated_at` "last flip" and a new `created_at` "first vote"? Decision needed before finding 1.
- Are old-vote flips taste drift (discount them) or label error (keep them)? The 46 re-votes cannot separate these; ~77 more calibration votes were estimated for ±0.07 (`STATUS.md:Next step`).
- Session fatigue (late-session grumpier votes) is hypothesized with dwell+impression timestamps available but unmeasured (`notes/improvement-brainstorm-2026-07-15.md:309-314`) — worth one measurement query, not a weight yet.

