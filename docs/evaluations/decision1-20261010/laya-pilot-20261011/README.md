# Laya-multilingual VPS pilot — 2026-10-11

User asked for a small model/quantization that runs fast on the VPS, chose
Laya-multilingual (mmBERT-base, 322M; convaiinnovations/laya bundle commit
7b928d828b7b0e022f929d9bd2e44165aa270148, laya 0.4.2, torch 2.14.1+cpu), and
authorized an automatic full run only if the pilot was below 5 s/score.

Result: gate missed, no full run. 16 frozen fold-0 jobs (8 real/empty pairs,
spread over state length incl. the longest), max_len 3072 (no truncation; longest
state 2,640 tokens over all 4,132 jobs), 2 threads, Nice 15, CPUQuota 200%:
5.37 s/score single, 5.82 s/score batch2. Real ~9.6 s, empty ~0.6 s each.
Batch vs single scores identical (max diff 0.0). Peak memory 2.6-2.9 GB, no
throttle/OOM. Different model from Strands; zero scores published to Strands caches.

Earlier attempts: batch4 at max_len 4096 hit MemoryHigh 3G (75% stall), stopped by
the agent; three launches failed on script bugs (fold int, wrong revision repo,
property call) before inference.

Full-run tooling is ready: laya_full.py.txt (resumable JSONL, identity-bound,
per-chunk batch/single check) and laya_report.py.txt (eval_decision1 metrics;
dry run with random scores verified all 4,132 state hashes against folds.json).
Note: /tmp/decision1-eval/snapshot.db is gone (removed by someone else), so
snapshot popularity controls are unavailable; folds.json hash still matches.
Estimated full run at 2 threads: ~6 h (2,066 x 9.6 s + 2,066 x 0.6 s).
