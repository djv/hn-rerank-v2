# Training data & labels review

## 1. Ranked findings

**1. Cross-post votes train as independent rows (dedup exists but is off).**
Category: defect. Evidence: `pipeline/config.py:30` (`deduplicate_training_feedback=False`), `config.toml [model]` never sets it; `pipeline/feedback.py:9` keeps latest vote per normalized URL but nothing live calls it (`pipeline/ranking.py:1274` gates on the flag). Same article via HN + Reddit = two rows, double weight; conflicting votes on two URLs both train. Expected effect: small time-saved gain, mildly pro-diversity (cross-posted topics currently over-weighted). Effort: S. Validation: `scripts/eval_ranker_variants.py` temporal folds, dedup on vs off, metrics P@12 + novel-up/block (not AUC alone). Confidence: medium.

**2. Re-votes destroy history (single row per story).**
Category: defect. Evidence: `database.py:702-709` PK `(user_id, story_id)`; `upsert_feedback:1342-1359` overwrites action + timestamp; `get_feedback_for_training:1462-1529` returns latest only. Calibration (FINDINGS 2026-10-11): 59% agreement, 25% of ups now down, old votes flip 0.33 vs 0.14 — yet the 17 flips were applied as overwrites. No flip-rate, per-story disagreement, or soft-label experiment is possible afterward. Expected effect: no direct ranking gain; unlocks noise-aware training later; zero sameness risk (additive table only, never rewrite votes). Effort: M. Validation: new append-only `feedback_history` table; offline, compare hard-label vs disagreement-downweighted fits on re-vote AUC. Confidence: medium.

**3. `neutral_weight` is a dead knob; neutral handling is actually fixed 3-class balancing.**
Category: defect (doc/config) + evaluation gap. Evidence: `neutral_weight` defined at `pipeline/config.py:36`, read nowhere (only ARCHITECTURE mentions). Real behavior: `pipeline/ranking.py:1595-1600` balanced weights give neutral 1/3 of total mass; eval labels ~1597 down / 1416 neutral / 835 up. Neutral-as-down already tried and rejected (FINDINGS 2026-10-11: held-out +0.001, re-vote −0.040 — neutral carries signal). But `lr_target=updown_only` (train on up/down only, `eval_ranker_variants.py:1594-1611`) exists with no recorded result I could find. Expected effect: clarifies tuning surface; a weight sweep aimed at P@12/novel-up (not AUC) could cut duds without touching topic spread. Effort: S. Validation: class-weight sweep in existing eval, judged on top-12 up-rate + novel ups. Confidence: medium.

**4. Dwell-confidence weights are built but apparently never run (B3).**
Category: untried improvement. Evidence: `database.py:1531 get_capped_dwell_by_story`, `eval_ranker_variants.py:844 _scores_margin3_dwell` (1 + gain·capped/cap, leak-cut at fold cutoff), `scripts/ledger_report.py` B3 gate. Dwell separates down-vs-rest (AUC 0.82) but not up-vs-neutral, and only 3/38 unvoted cards exceed 15 s dwell — the user votes nearly everything read (FINDINGS reading-signal; STATUS next step). Expected effect: small time-saved gain by down-weighting skimmed votes; sameness-neutral (reweights existing rows only). Effort: S. Validation: run B3 gate first (`ledger_report.py` dwell AUC ≥ threshold), then margin3_dwell vs control on temporal folds. Confidence: low-medium; data still sparse (profile 151: ~91–235 dwell events, ~1 article_open, 0 comments_open per STATUS).

**5. Exposure bias: shown-but-unvoted tried small, null — don't expand yet.**
Category: already tried. Evidence: `eval_ranker_variants.py` `skip_label`/`_with_skipped`; FINDINGS 2026-10-02: shown-unvoted as down (30%) 0.500/0.863, as neutral 0.521/0.869 vs live 0.500/0.865; pooled dAUC +0.001 — noise on 25–50 unvoted/block. Impression pools now exist (`--candidate-pool impressions`, `auc_up_vs_all`), so the only version worth revisiting is a larger skipped pool *after* the trial, not a new method. Expected effect: none demonstrated; caution — treating skips as negatives risks punishing discovery (unseen ≠ disliked) and narrowing the feed. Effort: M. Validation: rerun with bigger skipped pool, require novel-up/block non-inferiority as gate. Confidence: low.

**6. Vote timestamps are fetched and discarded; blanket recency decay tried twice, null.**
Category: already tried — park it. Evidence: `get_feedback_for_training` returns `vote_times`, production ignores them except the disabled publication path (`ranking.py:1289`, flag off). FINDINGS 2026-10-11 half-life 30/60 d: +0.004–0.006, CI from 0; 2026-10-02 half-life 30 d within noise. The 2026-10-11 note itself flags the limit (folds train pre-cutoff, can't show recent shifts). Only revisit as flip-aware downweighting (proposal 2), never blanket decay — blanket decay would erase stable long-term interests and risk sameness churn. Effort if revisited: S. Confidence: medium that blanket decay is worthless.

**7. Label noise is the ceiling argument for de-weighting AUC.**
Category: evaluation gap. Evidence: blind re-votes separate original ups at AUC 0.785 vs production 0.778; headroom +0.007 ±0.12 (FINDINGS 2026-10-11). Every proposal above should be judged on top-12 up-rate, down@12, and novel ups per block — the fresh-vote/impression-pool metrics from 2026-10-02 — not AUC. Effort: S (reporting convention). Confidence: high.

## 2. Checked, fine or already tried

- Balanced 3-class SVM weights + LOOCV self-exclusion (`ranking.py:1501-1560`): correct, no leakage.
- Neutral-as-down, half-life decay, shown-unvoted as down/neutral, up-minus-down scoring, source prior: all tried, all null (FINDINGS 2026-10-02, 2026-10-11).
- Joined-logistic challengers train on identical rows via shared snapshot (`ranking.py:1258`, interleave one-snapshot fix 623321b): sound; trial untouched.
- Raw embeddings unscaled, meta clipped ±2.5: correct per working rules.
- Missing-class zero rows at 1e-6 weight: harmless placeholder, fine.

## 3. Open questions

- Has `lr_target=updown_only` (drop neutrals in training) ever been scored? No result found.
- Has B3 `_scores_margin3_dwell` ever been executed, or only built? No result found.
- What fraction of profile 151's training rows are same-URL cross-posts (dedup impact size)?
- For new profiles: is any cross-user prior considered, or is sub-gate behavior purely tier-1/2? Out of this angle but label-relevant.
- Uncertain: exact current profile-151 class counts (1597/1416/835 read from a 2026-10-07 followup JSON; live counts have since grown past ~4,071 votes).

