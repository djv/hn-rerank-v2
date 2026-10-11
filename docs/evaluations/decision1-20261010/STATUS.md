# Decision1 offline eval — small models, fine-tune and tagging, 2026-10-11

## Objective
Find any cheap VPS-CPU model that adds ranking signal for user 1's HN votes on top
of production, on the frozen held-out folds. Strands dropped by the user ("too
heavy"); Laya stopped by the user ("stop laya"). User limits: VPS CPU only, no paid
GPU/API; a fine-tune must take under 30 min.

## Result
Evidence: SMALLDEC-TAGGING-RESULTS-20261011.md (scripts and aggregate reports in
smalldec-20261011/). AUC(up vs rest), blend = 0.8·production + 0.2·model, pooled.
- Zero-shot decision models with the profile (Laya partial, Julia-1, Mira-v1): no
  signal; every blend − production CI includes 0 or is negative.
- Ettin-17M fine-tune (25 min, 5 folds): model 0.676 vs TF-IDF 0.672; blend −
  production +0.0007 [−0.004, +0.006]. Real but redundant signal.
- Zero-shot tags: rules valid for show_hn and paywalled only; embedding prototypes
  mostly invalid (negation). No arm improves the blend.
- Julia-1 tagging stopped by the user after the silver set: no tag reaches the
  0.80 validity bar (several inverted); utility not measured.
- User decision 2026-10-11: stop the small-model line; production stays as is.

## Blocker
None for this question. /tmp/decision1-eval/snapshot.db was removed by someone
else, so snapshot popularity controls stay unavailable (folds.json hash matches).

## Next step
None on this line (user: "Stop the small-model line"). Strands artifacts
(/tmp/strands-eval, cache_strands*) and the /tmp model and score files
(/tmp/smalldec, /tmp/finetune, /tmp/tagging, /tmp/laya-pilot) stay until the user
approves deletion.
