# Training data & labels review (hn-rerank)

## 1. Top findings / proposals, ranked

**P1. Exposure bias: shown-but-skipped stories never train — the model's negatives are only explicit downs.**
Category: defect (bias) / untried improvement. Evidence: `database.py:1462` `get_feedback_for_training` reads only the `feedback` table; `pipeline/ranking.py:1264` trains solely on those rows. `interaction_events` (impressions, live since 2026-10-12) never enters training. With ~5,300 impressions vs ~4,000 votes (profile 151), most "shown, ignored" outcomes are discarded, so the model never learns what the user won't even vote on.
Effect on aims: cuts duds (save time), but naive skip-negatives risk punishing niche topics skipped when busy — must be low-weighted and paired with a novel-up-vote check, or it worsens sameness. Effort: M. Validation: `scripts/eval_ranker_variants.py`, temporal split, 30-day window; metrics NDCG/top-12 up-rate **plus** novel up-votes per week (Goals rule). Confidence: medium.

**P2. Dwell/opens as sample weights or soft labels — infra exists, unused.**
Category: untried improvement. Evidence: `database.py:1537` `get_capped_dwell_by_story` exists explicitly "for sample_weight experiments (B3)"; nothing in `pipeline/ranking.py` calls it. FINDINGS (2026-10-13): dwell separates not-down from down at AUC 0.82, but not up from neutral.
Effect: distinguishes "read then neutral" (soft up) from "glanced and skipped" without new labels; sameness-neutral to positive. Effort: S–M. Validation: first rerun the STATUS-scheduled ~10-27 dwell-vs-votes check (drop 90 s-capped segments); then eval with dwell-weighted neutrals/skips. Caveat: still sparse (235 dwell, 2 article_open, zero comments_open) — weight, don't threshold. Confidence: medium-low.

**P3. Same article can train twice with conflicting labels (training dedup off).**
Category: concrete defect (small). Evidence: `pipeline/config.py:30` `deduplicate_training_feedback=False` by default; `pipeline/feedback.py` keeps latest vote per normalized URL but is dormant. Cross-posted HN/Reddit/RSS dupes therefore train as independent rows, possibly up *and* down.
Effect: label noise, small dud cost; dedup slightly shrinks training set (no sameness harm expected). Effort: S. Validation: toggle the flag in `eval_ranker_variants.py` on/off, same temporal split. Confidence: medium.

**P4. Re-votes destroy label history — training set is a moving, unreproducible target.**
Category: defect (governance). Evidence: `database.py:1339` `upsert_feedback` overwrites action + `updated_at`; `delete_feedback` (undo/clear) removes the row outright. STATUS.md already flags mutable feedback as making retrospective reports non-reproducible; the 29 calibration re-vote rows (`updated_at` 1791763112–1791763180) must be manually excluded from every fresh-vote eval.
Proposal (no DB rewrite, per constraints): append-only vote-transition log for *future* votes + frozen label snapshots at eval/trial endpoints. Effort: S–M. Validation: reproducibility check (re-run a past eval from snapshot). Confidence: high (defect); value medium.

**P5. Evaluation gap: offline pools contain only voted stories, so P1/P2 cannot be validated.**
Category: evaluation gap. Evidence: eval configs use `"candidate_loader": "heldout-feedback"` (`docs/evaluations/model-ablation-20261007/followups/*.json`); `deck_scores` + `interaction_events` (positions, skips) postdate the eval harness. Any skip/dwell label has no offline metric today, and discovery (unvoted novelties) is unmeasurable in-pool.
Proposal: impression-joined offline eval (skips as weak negatives, position recorded) before any P1/P2 ships. Effort: M. Confidence: high that the gap exists.

**P6. `neutral_weight` is a dead knob — remove or wire it.**
Category: defect (hygiene, low value). Evidence: `pipeline/config.py:36` defines `neutral_weight = 0.0`, but no use found in `pipeline/ranking.py` (balanced weights at `ranking.py:~1590` weight all three classes equally). Neutrals-as-own-class is correct — neutral→down scored null/negative (FINDINGS 2026-10-11: held-out +0.001, re-votes −0.040) — so just delete the knob to avoid a future mis-set. Effort: S. Confidence: medium-high (uncertain only because grep, not execution, confirmed non-use).

**P7. Do NOT re-propose recency decay — already tried, repeatedly null.**
Already tried: FINDINGS 2026-10-11 (half-life 60/30 d: +0.004/+0.005, CI from 0), plus 2026-09-29 and 2026-10-02 nulls. Taste drift is real (re-votes: >95 d stories flip 0.33 vs 0.14 newer) but the headroom is unsettled (AUC ceiling 0.785 vs production 0.778, ±0.12). Differ-or-drop: only revisit after ~77 more calibration votes narrow the band to ±0.07 (STATUS next step); `_vote_times` is already plumbed (`ranking.py:1289`, currently only feeds publication features + dedup).

## 2. Checked — looks fine or already tried

- Re-vote = relabel, not duplicate row (`upsert_feedback` ON CONFLICT) — fine.
- Balanced class weights (ups minority: 835 vs 1597 down / 1416 neutral) — fine.
- LOOCV self-exclusion in training kNN features (`_loocv_knn_features`) — leakage guard fine.
- Zero-row padding (weight 1e-6) for absent classes — fine.
- 3-class labels kept; neutral-as-down rejected by data — already tried, correctly dropped.
- Dual-gate 20/20 + tier blend protects small/new profiles — fine for cold start.
- Small-data skepticism honored: no high-capacity label model proposed.

## 3. Open questions

- Does `publication_features` use `_vote_times` for recency in a way that interacts with P7?
- Are skips position-biased (lower cards skipped more)? `deck_scores` positions can answer before P1.
- Should undo restore the *prior* action rather than deleting the row (label-history loss)?
- TUI 1 s vs web immediate impression logging: which policy defines a "skip" for P1/P5?

