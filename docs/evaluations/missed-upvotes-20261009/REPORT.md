# Missed-upvotes exploratory analysis (2026-10-09)

Bounded, descriptive-only review of EXISTING held-out predictions
(`compare_baseline` / `compare_challenger`, 3079 rows, blocks 2–5) and
cached Muse facet annotations (`facet-residual-20261009`, 379 stories).
No new ranking calls, model fits, parameter sweeps, production changes,
or web browsing. All private rows stayed on the VPS; only aggregates,
code, and this report were copied locally. No titles, transcripts, or
vote-linked text appear below.

Artifacts: `analyze_missed_upvotes.py` (typed, artifact-only),
`aggregates.json` (counts only, `schema: missed-upvotes-v1`). Script
validation passed (denominator sums, per-block UP/DOWN and top-12 table sums, oracle
arithmetic, no private-text keys).

## Definitions (frozen before inspection)

- UP = `true_label == 2` (verified against snapshot feedback:
  `up→2`, `down→0`; spot-checked one story of each).
- Within-block rank = baseline native `production_score` desc,
  tie-break `story_id` asc. **Missed upvote** = UP row with rank > 12
  in its block (top-12 = dashboard first-screen width).
- Worst-ranked UP quintile = pooled UP rows sorted by **within-block
  rank** desc (worst first), first `ceil(683/5) = 137`. Rank-based
  only; absolute scores are never compared across blocks.
- Production scores are uncalibrated ranking scores, not confidences;
  a low score means a low rank, nothing more.
- Broad source: `hn|ch_seed|bq_seed→hn`; `rss_reddit_*→reddit`;
  `rss_lesswrong*→lesswrong`; else `other`.
- Content class from aggregate lengths only (no text read):
  `body` (self_text or article_body non-empty), `comments_only`,
  `title_only`.
- Facet scope: UP rows in blocks 2–5 carrying a cached annotation;
  repeats deduped (sample > dev > repeat). Subset-only denominators;
  **no population claims**.
- DOWN mirror scope: baseline top-12 placement of DOWN rows only;
  **not a test of Muse corrections**.

## Cohort and the oracle ceiling

3079 rows: blocks 2–5 (~770 each). Labels: 683 UP / 1065 neutral /
1331 DOWN. Story metadata matched 3079/3079 (read-only lengths-only
SELECTs). Facet annotations cover 379 stories total; 61 of the 683 UP
rows (8.9%).

48 top-12 slots (12 × 4 blocks) for 683 UPs set the arithmetic before
any modeling: an oracle fills at most 48 hits, leaving **635 misses
unavoidable**. Actual: 29 UP hits, 18 neutral + 1 DOWN in top-12.
So 654 − 635 = 19 further hits were reachable only by displacing the
**19 replaceable non-UP picks** — the entire headroom this cohort
offers a better ranker. The 95.8% miss rate is therefore structural
(slot scarcity), not proof of 95% ranking errors; the actionable
comparison is 29 hits against the 48-slot ceiling.

## 1. Miss rates by block (each block its own denominator)

| block | n_up | hit top12 | missed | miss rate | worst-quintile members (rate of block UP) |
|---|---|---|---|---|---|
| 2 | 196 | 7 | 189 | 96.4% | 52 (26.5%) |
| 3 | 236 | 6 | 230 | 97.5% | 63 (26.7%) |
| 4 | 151 | 11 | 140 | 92.7% | 11 (7.3%) |
| 5 | 100 | 5 | 95 | 95.0% | 11 (11.0%) |
| pooled | 683 | 29 | 654 | 95.8% | 137 |

Miss rates are uniform (92–98%); no block is an outlier. Blocks 2–3
contribute 115/137 of the worst tail, read against their own
denominators: 26–27% of b2/b3 UPs sit in the pooled worst tail vs
7–11% for b4/b5. Blocks with more UPs mechanically place more of them
outside 12 fixed slots (b3 cast 236 UPs vs b5's 100). This alone does
not explain their overrepresentation in the deeper tail. No
cross-block score explanation is invoked — ranks are within-block by
construction.

## 2. Broad source (denominators shown; enrichment vs all UP)

| source | n_up (share) | missed (share) | miss rate | enrichment |
|---|---|---|---|---|
| hn | 311 (45.5%) | 303 (46.3%) | 97.4% | 1.02 |
| reddit | 171 (25.0%) | 167 (25.5%) | 97.7% | 1.02 |
| lesswrong | 27 (4.0%) | 27 (4.1%) | 100% | 1.04 |
| other | 174 (25.5%) | 157 (24.0%) | 90.2% | 0.94 |

Miss rates show no consistent source signal (90–100%, enrichments
≈ 1.0; lesswrong n = 27 is small). Per-block miss rates by source
(`per_block_source` in aggregates) are likewise mixed: e.g. `other`
misses least in b2 (81%) and b4 (80%) but not in b3 (100%).

Worst-tail placement by source, per block (quintile rate of block UP
in that source vs block rate):

| block | hn | reddit | other | lesswrong | block rate |
|---|---|---|---|---|---|
| 2 (n_up 196) | 28/85 = 32.9% | 18/76 = 23.7% | 4/32 = 12.5% | 2/3 (small) | 26.5% |
| 3 (n_up 236) | 43/116 = 37.1% | 7/38 = 18.4% | 9/62 = 14.5% | 4/20 = 20.0% | 26.7% |
| 4 (n_up 151) | 7/71 = 9.9% | 3/38 = 7.9% | 1/41 = 2.4% | 0/1 (small) | 7.3% |
| 5 (n_up 100) | 9/39 = 23.1% | 0/19 = 0% | 2/39 = 5.1% | 0/3 (small) | 11.0% |

The hn direction is consistent across all four blocks (above the
block rate every time; `other` below it every time), but b4/b5
quintile cells are small (11 members each) and no mechanism is
asserted. It is a descriptive pattern to carry into any future
age-controlled test, not a finding about sources.

## 3. Content availability

| class | n_up (share) | missed | miss rate | enrichment |
|---|---|---|---|---|
| body | 617 (90.3%) | 589 | 95.5% | 1.00 |
| comments_only | 66 (9.7%) | 65 | 98.5% | 1.03 |
| title_only | 0 UP (10 DOWN) | — | — | — (small) |

Per-block content tables (`per_block_content`) show no consistent
direction: `comments_only` quintile rates vs block rate are 36%/26%
(b2), 21%/27% (b3), 9%/7% (b4), 80% on n = 5 (b5, small). Body text
was present for ~90% of UPs and of misses in every block; no UP row
is title-only at all. Absent evidence of a content-availability
mechanism is not evidence of absence — the ranker had text for
essentially every missed upvote, so coverage cannot separate hits
from misses here.

## 4. Facet subset (coverage 8.9%; subset-only, no population claims)

61 UP rows carry cached annotations; 58 missed, 14 in the worst
quintile. With only 3 annotated UP hits, within-subset miss rates are
saturated (~95%) for every facet value and cannot discriminate.
Enrichments across all seven fields (5 retained + 2 dropped) sit at
0.86–1.05 with per-value n mostly < 15 (flagged `small_n`). Largest
annotated UP cells: firsthand voice 35, conflict_framing=0 45,
actionable=0 34, artifact essay_opinion / question_discussion 12 each,
claim release 16 / argument 15 — misses track the subset's own
composition throughout. This sampled saturation yields no facet
direction; it does not rule any blind spot in or out.

DOWN mirror (baseline placement only): 131 facet-annotated DOWN rows,
**0 in any top-12**; population 1/1331 DOWNs in top-12 (block 3).
A placement tabulation, not a Muse-correction test — stated as a
scope limit.

## 5. Cached Muse evidence: some corrections, net negative on probes

Prior cached audits (no new calls; reused dev data, exploratory):

- CC replay: Muse corrected 2/5 confident ML errors (40%, gate ≥60%)
  with control agreement 1/5 (gate ≥70%); fixes−damage = 2−4 < 0
  (STOP). Damage included high-confidence (68–85) preferences for the
  DOWN-voted story and a neutral-over-true-UP pick at conf 86.
- Targeted pairs: Muse 11/12 on corrections (tracks Opus) but 5/11 on
  regressions (leans Opus); style tags do not support a
  substantive-vs-chatter feature (REJECT).
- Familiarity hypothesis: 5/14, below the 67% gate — unsupported.
- Facet residual, exact endpoints: primary ordinal-AUC Δ pooled
  arm3−arm1 = **−.051** (gate ≥+.02: FAIL); UP-vs-rest Δ (+.006,
  secondary, no gate, no conclusion drawn); all 5 frozen gates fail
  (STOP). The cached facet evidence adds no fitted ranking signal.

## 6. Private-example inspection (max 12, VPS-only; aggregates published)

Worst-ranked 12 missed UPs inspected privately on the VPS (metadata +
facets only; nothing copied out; no further reads). Aggregate
descriptions:

- 12/12 broad-source hn; **10/12 content class body, 2/12
  comments-only**; 3 article bodies at the 15000-char store cap and 7
  discussion texts at the 10000-char cap (remainder shorter but
  non-empty); comment counts 5–328.
- Native scores 0.03–0.20: low-ranked rows (scores uncalibrated —
  read as rank position only).
- 10/12 carry no facet annotation (coverage gap, not a content
  finding); the 2 annotated ones span unrelated categories.
- Topic mix is spread (news/politics items, OSS/tool releases,
  personal projects, tech-culture essays, one service-outage brief,
  one reader-app note) — no single topic dominates the 12.

Tentative hypotheses only (n = 12, not evidence): H1 — misses in the
tail tend to be content-rich, so coverage-style explanations do not
separate them; H2 — the hn tail enrichment (consistent direction,
§2) needs an age-controlled test before any source reading; H3 — the
facet/topic spread shows no single categorical concentration in this
sample.

## 7. Challenger consistency (descriptive overlap)

Same top-12 rule on challenger scores: 654 missed, miss rate 95.8%,
overlap with baseline misses 642/654; 12 baseline-only + 12
challenger-only. Per-block miss rates match within ±0.7pp. The two
arms miss nearly the same UPs — consistent with the arms differing
only on changed-input rows, not with either arm fixing misses.

## Conclusion

At least 635 UPs must remain outside 48 slots, regardless of which stories
are selected. The baseline fills 29 slots with UPs and 19 with non-UPs;
these are the historical top-12 improvement headroom, not fixed identities
that must remain missed. Deeper ranks and AUC can also improve.

HN overrepresentation in the deepest UP tail holds in all four blocks, a
descriptive pattern worth inspecting. Later blocks have small tail counts;
source, content and training history are confounded. Cached Muse evidence
has not established a reliable correction: prior probes added damage as
well as fixes, and facet coverage is only 61 of 683 UPs.

Next measurement: inspect native score components on matched HN UP/DOWN
controls to distinguish an appropriate source prior from a semantic blind
spot. Preserve native scores; any later Muse correction design must measure
both recovered UPs and damaged controls. Freeze that design before fresh
votes. This reused-outcome analysis is hypothesis generation, not confirmation.

## Limitations (binding)

- All outcomes are reused development data; everything above is
  descriptive. No causal, prospective, or deployment claim.
- No confidence intervals, p-values, or gates: rates here are
  structural (48 slots for 683 UPs), and any fished "significant"
  slice would be noise. Small cells (n < 10) flagged in
  `aggregates.json` (`small_n`).
- Confounds: block UP counts drift (236 → 100), mechanically loading
  more UPs outside fixed slots where UPs are numerous; source ×
  content availability are entangled; b4/b5 tail cells are small;
  facet subset is small and selection-shaped (prior design's
  sample+dev) — subset-only.
- Private-example inspection is hypothesis generation (n = 12), not
  evidence; titles/transcripts never leave the VPS.

## Files

- `analyze_missed_upvotes.py` — typed analysis script (run on VPS via
  project uv against the read-only snapshot; `--print-private-ids`
  prints VPS-only inspection IDs to stdout, never into aggregates).
- `aggregates.json` — full counts (`schema: missed-upvotes-v1`),
  incl. `per_block_source` / `per_block_content` and oracle fields.
- Root integration corrected per-category DOWN top-12 counts and added
  invariants; rerun on VPS with unchanged source data. Ruff/format/ty passed.
- No edits to root STATUS/FINDINGS/WORKLOG or prior trial docs (root
  handles those).
