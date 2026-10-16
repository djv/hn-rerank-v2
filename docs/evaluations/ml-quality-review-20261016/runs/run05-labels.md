## Training-data and labels review (read-only)

Verified against the live tree. No code run, no DB present.

### 1. Top findings / proposals (ranked)

**1. Conflicting cross-post votes train as independent rows — enable + evaluate URL-dedup of training rows.**
Category: defect (off-by-default hygiene). Evidence: `pipeline/ranking.py:1258-1278` fetches all votes and only dedups when `deduplicate_training_feedback` is set; `pipeline/config.py` defaults it `False`; the canonical `deduplicate_feedback` (`pipeline/feedback.py:1-30`, latest-vote-per-normalized-URL) is therefore dormant in production. Two rows with the same URL but different actions both enter KNN/LOOCV features and the fit, pulling similarity features both ways. Effect on aims: less label contradiction → fewer duds; sameness roughly neutral (dupes currently double-count one topic, so dedup weakly *widens* spread). Effort: S (flag + eval). Validation (offline first, post-trial): `scripts/eval_ranker_variants.py`, temporal splits, production vs `deduplicate_training_feedback=True`; report NDCG/AUC **plus** top-12 up-rate, topic entropy/diversity gain and novel up-votes per the diversity/discovery plans. Confidence: medium. Uncertain whether dedup was ever evaluated — I found no result; if it was, drop this.

**2. `neutral_weight` is a dead knob — remove or wire it.**
Category: defect (config/docs). Evidence: `pipeline/config.py:36` defines `neutral_weight = 0.0`; nothing in `pipeline/` reads it — `pipeline/ranking.py:~1592-1600` builds purely balanced weights `n/(classes·count)` + 1e-6 dummies, and `ARCHITECTURE.md:3.3` lists it as a tuned hyperparameter. Someone tuning "neutral weight" changes nothing. Effect: no ranking change; prevents a wasted/misleading experiment; sameness neutral. Effort: S. Validation: grep + suite. Confidence: high.

**3. Re-vote/undo timestamps make training history mutable — snapshot hygiene, and keep the calibration exclusion.**
Category: defect (minor) + evaluation gap. Evidence: `database.py:1349-1356` only bumps `updated_at` when the action *flips* (same-action re-vote is a no-op); undo *deletes* the row; the model signature (`pipeline/ranking.py` signature block) keys on `updated_at`, so flips invalidate but deletes silently drop negatives. The 29 calibration re-votes (`updated_at` 1791763112–1791763180) are old stories masquerading as fresh votes in training; `STATUS.md` "Next step" already requires excluding that window in *evals*. Extend the rule to any post-trial *training* comparison too, or tag calibration rows. Effect: cleaner train/test boundaries; no serving change; sameness neutral. Effort: S. Validation: row-count audit + rerun one eval with/without the window. Confidence: high on mechanism, low on magnitude.

**4. Adopt the 46 blind re-votes as the label-noise validation split for every label proposal.**
Category: evaluation gap. Evidence: `FINDINGS.md` "Blind re-vote calibration … 2026-10-11": 59% agreement, downs stable, ~half of neutrals now down, ups 45/30/25 up/neutral/down; older votes flip more (0.33 vs 0.14); headroom +0.007 with CI跨 zero. Every future weighting/relabelling idea should report original-held-out AUC **and** re-vote AUC, since they disagree (neutral-as-down: +0.001 vs −0.040). Effect: stops shipping noise-fitting "wins"; sameness neutral. Effort: S (protocol, no serving change). Validation: add the re-vote set as a fixed second test split in `eval_ranker_variants.py`. Confidence: medium. Note the planned ~77 more calibration votes would narrow this (±0.07).

**5. Narrow reading-signal test only: long-dwell neutrals as soft ups — after the 10-27 data cut, not now.**
Category: untried improvement (narrow variant of a tried-and-killed idea). Evidence: broad dwell-weighting was already tried and killed — `ROADMAP.md` "B3's consumer KILLED": `margin3_dwell` (`eval_ranker_variants.py:844-870`, weights via `database.py:1537 get_capped_dwell_by_story`) scored strictly worse than `margin3_up` on every headline metric despite dwell's 0.82 down-vs-rest AUC (`FINDINGS.md` "First reading-signal review"). Do **not** re-propose generic dwell weighting. The surviving hypothesis, already scheduled in `STATUS.md` ("rerun … long-dwell neutrals as soft ups; drop 90 s-cap segments"), is much narrower: dwell separates read-vs-skip, not up-vs-neutral (medians: up 22 s, neutral 24 s, down 8 s, unvoted 3.5 s), and only 3/38 unvoted cards had >15 s dwell — so coverage is tiny. Effect: at best converts a few neutrals; sameness risk is low but real (dwell concentrates on Recommended, so weight, don't relabel). Effort: M. Validation: temporal folds with only the relabel changed; require gains on both original and re-vote splits plus no drop in novel up-votes. Confidence: low-medium.

**6. Shown-but-unvoted impressions as weak negatives — offline experiment only, low priority.**
Category: untried improvement with a user constraint. Evidence: production trains exclusively on votes (`database.py:1462-1529 get_feedback_for_training`); `interaction_events` impressions (5,000+) never enter `_score_and_rank`. Eval scaffolding already exists and should be used first: `_load_skipped_pool` / `_impression_pools` / `_with_skipped` and `skip_label`/`skip_frac` (`eval_ranker_variants.py:1322-1470,1643-1662`), plus `--candidate-pool impressions`. But the user explicitly declined hide-seen and skip-as-weak-down on 2026-10-13 (`STATUS.md`/`FINDINGS.md` "First reading-signal review"). So: no serving change without a new explicit ask; if tested, prefer `skip_label=neutral` at small `skip_frac` over down-labeling, and gate on the sameness guardrail (this directly optimizes "fewer duds" and is the proposal most likely to narrow topics). Effort: M. Validation: impressions-pool eval + diversity/novel-ups readouts. Confidence: low.

**Dropped (already tried, do not resurrect):** general time-decay — `half_life` 30/60 gave +0.004–0.006 on re-votes, ≈0 on originals (`FINDINGS.md` recency check; eval support at `eval_ranker_variants.py:787-790`); production correctly ignores `vote_times` in weights today. Neutral-as-down — no gain, hurts re-votes. Broad dwell weighting — killed (above). LLM-derived labels — `RANKING-REASSESSMENT-20261009.md` verdict NONE.

### 2. Checked, fine or already tried (one line each)

- 3-class labels (down 0 / neutral 1 / up 2) with balanced sample weights: kept, neutral carries signal — fine.
- LOOCV self-exclusion in training KNN (`ranking.py:_loocv_knn_features`): correct leakage guard — fine.
- Raw embeddings never standard-scaled; scaler touches meta columns only, clipped ±2.5 — fine.
- Dual gate (20 up + 20 down) + tier ramp: reasonable for small profiles — fine.
- Same-action re-vote not bumping `updated_at`: correct (nothing changed) — fine.
- Impression-pool / skipped-pool eval paths exist; diversity/discovery/sameness offline plans exist — use them, don't rebuild.
- Trial freeze until 2026-11-05 respected: all above validate via `eval_ranker_variants.py` + existing diversity/discovery scripts first.

### 3. Open questions

- Was training-row URL-dedup ever evaluated? If yes, where is the result?
- Should calibration/re-vote rows be permanently tagged so future training snapshots exclude them by label, not by a magic timestamp window?
- For profiles with <20 downs (new/small users), is the tier-2 centroid the right fallback, or should weak negatives play a role there first?
- Dwell `position` history before the served-rank fix is list-index, not rank — does any planned dwell-weight eval accidentally join on it?
