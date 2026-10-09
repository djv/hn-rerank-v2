# Ranking reassessment: reusable signal in Muse corrections/regressions (2026-10-09)

Scope: read-only inspection of saved artifacts only. No new judgments, fits,
rankings, or annotations. Raw rationales inspected on VPS; this report carries
pair IDs + aggregate categories only, no private fulltext/profile.

## Evidence inspected

- `targeted-muse-20261009`: 35 forward pairs (12 corrections, 11 regressions,
  9 agree-right, 3 agree-wrong) + 10 reversed; per-pair answers in VPS
  `llm-labels/targeted-muse-20261009/muse-cache.json` (keys = pair IDs,
  `parsed.{preferred,confidence,style_A,style_B,rationale}`); selection in
  `pairs.json`; aggregates local `docs/evaluations/targeted-muse-20261009/`.
- `cc-replay-20261009`: 5 ML-error + 5 control pairs (REPORT §3: fixes p-01,
  p-02; damage p-06, p-08, p-09, p-10; p-05 neutral-over-UP); answers in VPS
  `llm-labels/cc-replay-20261009/muse-cache.json` (14 entries incl. reverses).
- `muse-hypothesis-20261009/REPORT.md` + `results.json`.
- Ranker features: `pipeline/ranking.py:1045-1092` (meta cols 0-9),
  `pipeline/linear_blend.py:45-51` (TF-IDF text), `pipeline/ranking.py:359-387`
  (source onehots).

## Candidate table

| Candidate distinction | Fixes covered | Damage counterexamples | Already represented? |
|---|---|---|---|
| Substantive-vs-chatter style | none separable (corrections pos_sub 6/12 vs neg_sub 6/12) | regressions neg MORE substantive (10/11) than pos (7/11); overall pos 22/35 vs neg 23/35 | n/a — REJECTED, confirmed |
| Lexical familiarity (title overlap w/ 50 recent ups) | — | Muse picks higher-overlap 5/14 non-tied pairs, below 67% gate | partially (TF-IDF title tokens); NOT SUPPORTED, do not resurrect |
| Prior-UP precedent matching (saved explanations invoking past UP patterns) | All 11 successful targeted corrections invoke it; the remaining wrong correction does too | 10/11 regression-stratum rationales invoke it, including 5/6 damaged picks; f4-regr-06 instead cites a neutral precedent. High-confidence damage: f4-regr-03 (92), f3-regr-05 (85) | Related proxies already exist: `sim_to_upvoted`, `closest_upvoted`, `positive_cluster_similarity` (`ranking.py:1063-1067`) and fitted lexical features; not proof they implement the same reasoning |
| Confidence-gated deferral (trust Muse when confident) | targeted fixes 65-85; CC fixes 55 | targeted damage 64-92; CC damage 68-85; substantial overlap | n/a — no defensible gate demonstrated |
| Broad topic/source boost (e.g. AI/agents, HN) | — | same-source design (corrections 12/12, regressions 11/11 same-source) blocks source claims; AI-topic damage (f3-regr-01, f3-regr-05, f4-regr-03, p-10) blocks topic boost | source onehots already exist (`ranking.py:1066-1071`) |

## Verdict: NONE

No candidate is supported enough to implement. Prior-UP explanations occur
in both successful and damaging choices; similarity and lexical proxies
already exist, although their presence does not establish feature adequacy.
Saved rationales describe apparent justification, not verified causal
mechanisms or faithful explanations. Their cited precedents were not
independently checked against the original training labels in this review.
The agent verified all 23 targeted correction/regression forward choices,
orders, confidence values, and saved rationales. Muse fixed 11/12 errors
and damaged 6/11 correct picks in these selected strata; it did NOT get
all 11 regression pairs wrong. No reusable missing signal was established.
No claim of general advantage is licensed (25/35 vs 20/35 on
label/Opus-conditioned dev folds; 9/9 agree-right vs 0/3 agree-wrong shows
shared blind spots, not Muse superiority).

## Decision + smallest next step

- Build nothing. Spend nothing further on LLM pairwise annotation for
  feature discovery on this evidence. This is a cost-based stop decision,
  not proof that every useful LLM signal has been ruled out.
- Priority goes to the existing live ML interleaving path, which learns
  measures ranking outcomes from real user votes.
- No follow-up protocol, branches, or infrastructure. Artifacts stay
  private (VPS-only caches, snapshot DBs); aggregates + scripts only if
  mirrored.
