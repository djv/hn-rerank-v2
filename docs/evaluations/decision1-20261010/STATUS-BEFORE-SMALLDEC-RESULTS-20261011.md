# Decision1 offline eval — Laya subset, 2026-10-11

## Objective
User dropped the Strands decider ("forget about strands, too heavy", 2026-10-11).
Evaluate Laya-multilingual (322M encoder) instead on the frozen held-out folds:
fixed-seed 100 stories per fold (subset-100.json), real and empty profiles,
1,000 jobs on the VPS CPU.

## Result
Pilot: 4.74 s/score at 3 threads (5.37 s at 2), batch==single exactly, no
truncation at max_len 3072, peak memory under 3 GB. Subset run
laya-subset100-20261011 launched 17:37 UTC (~80 min). Evidence and scripts:
laya-pilot-20261011/. Earlier Strands evidence is kept but no longer pursued.

Laya subset run stopped at user request ("stop laya") ~18:00 UTC with 608/1000
scores (500 empty, 108 real); partial AUC: empty 0.53, real 0.50 on 76 paired
stories vs production 0.79-0.84. No positive signal; no full report.
Next: Julia-1/Mira-v1 chain (smalldec-chain-20261011): pilot each, subset run only
if <= 2.5 s/score. Survey: SMALL-DECISION-MODELS-20261011.md.

## Blocker
/tmp/decision1-eval/snapshot.db was removed (not by this agent); snapshot
popularity controls are unavailable. folds.json hash still matches the receipt.

## Next step
Read /tmp/laya-pilot/full/report-subset100.json when scoring completes.
Strands artifacts (/tmp/strands-eval, cache_strands*) stay until the user
approves deletion.
