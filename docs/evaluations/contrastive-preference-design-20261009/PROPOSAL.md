# Contrastive within-topic discriminator induction (2026-10-09, rev.1)

**UNTESTED HYPOTHESIS. No ranking improvement claimed or implied. No
ranking annotations, fits, embeddings, or dataset reads performed. Muse was
used for design and revision. Execution and any new
annotation budget are explicitly NOT authorized.**

## One hypothesis

Within a fixed topic, the user's UP-vs-DOWN split follows narrow,
evidence-quotable distinctions (e.g. measured firsthand build notes vs opinion
roundups *about the same tool*) that an LM can induce from same-topic UP+DOWN
contrasts and re-apply as abstaining discriminators — adding UP discovery
without the on-topic DOWN damage of broad-profile preference prediction.

## Difference — and limits

Prior tasks did blind pair ranking from a generic profile with labels hidden,
and showed topic-aligned DOWN picks:
high-confidence (68–85) DOWN picks, one neutral-over-true-UP at conf 86
([cc-replay §3](../cc-replay-20261009/REPORT.md)); 11/12 corrections track Opus
but 5/11 regressions lean Opus ([targeted-muse](../targeted-muse-20261009/report.md)).
Isolated facet tags lost (Δ **−.051**, 5/5 gates FAIL;
[facet-residual](../facet-residual-20261009/REPORT.md)). Novelty: supervised
induction with BOTH opposite TRAIN labels visible in-contrast. Limits: prior
pairs were already same-source
and token-matched, so topic-holding is approximate and can't guarantee against
topic bias; quotes need verified exact spans and don't prove a rule. No HN
deficit was observed ([source-diagnostic](../source-diagnostic-20261009/REPORT.md)) —
not proof none exists. The 19 replaceable picks were cohort-specific
([missed-upvotes](../missed-upvotes-20261009/REPORT.md)), not future headroom.

## Three partitions

**A** (older labeled) → rule induction. **B** (later train) → correction fit,
with rule generation excluding B labels. **C** (future) → pilot
test. Induction uses A outcomes; fitting uses B outcomes. Timestamps picked
at later freeze — unspecified here. Missing classes/topic support →
STOP before any paid induction.

## Schemas

Induction from A: `{rule ≤40 words, up_quotes, down_quotes}` or abstain; at
most 2 rules (1/group × 2 groups). Application per item: `{rule_id, verdict:
+1|−1|unknown, target_span}`; ± = UP- vs DOWN-associated side, unknown = 0.
Per-rule signed columns (≤2), no pooled counts, no confidence routing, no
assumed abstention floor — unsupported coverage is inconclusive with counts.

## Transfer (exact)

Pairwise logistic on B only:
`logit pref(i over j) = (native_i − native_j) + β·(f_i − f_j)`,
baseline coefficient fixed 1, β L2 penalty 10, no intercept, no search. Final
`native + clip(β·f, −.02, +.02)`; all-unknown items score exactly native.
Ranking scores, not probabilities; cap/penalty preregistered, not validated.
Paired UP-vs-nonUP comparisons within eligible/exposure windows only;
controls carry the identical fitting burden.

## Controls

Null: same-A induction with shuffled within-group labels; same texts/fit/cap.
Plus a zero-inference topic-match control. These can't prove
topic-free signal; assess treatment vs native, null, topic. Fixes/damage
separate on C; no model output is label truth.

## Allocation

2 induction (treat + shuffle; 2 groups × ≤2UP/2DOWN = 8 A items) + 4
application on B (16 items, batches of 8, treat/null separate) + 8 on C (32
items, batches of 8) + 2 shared retry reserve = **TOTAL 16 requests / 80k
input+output tokens**, all stages/retries/both arms included. Stop at cap, no
expansion. Atomic hashes on prompts/model/inputs/partitions. If support is
insufficient before calls, spend nothing. Falsification pilot, not powered
confirmation.

## Evaluation on C

C's 32 items acquired only after freeze via a fixed outcome-blind
eligible/exposed chronological rule; labels held by the evaluator, never in
prompts. Primary: directional UP-vs-rest AUC conditional on judged C items;
plus recovery/damage counts and a downvote-promotion guardrail. Unvoted =
unknown, never DOWN. No whole-pool top-12 counterfactual scoring where
outcomes are unshown; no CI/significance claims. Net damage, treatment ≤
null/topic, or stability/coverage problems → STOP/inconclusive; promising
does NOT authorize a feature. More UP discovery needs separate live
randomized/interleaved evaluation after the existing
window and explicit authorization.

## Next artifact (no ranking annotations; not built)

SPEC TEMPLATE only: schemas, partition-selection rule, quota driver, cache
policy, eval plan — no private roster or cutoffs picked from data here. No
automated next-permission flow.

## Risks

Tiny contrasts; poor groups; LM restating topic as "rule"; near-zero coverage
→ STOP. Transfer counts only if treatment beats native, null, and topic
controls with the DOWN guardrail intact.
