# Contrastive preference pilot — SPEC TEMPLATE (rev.1, 2026-10-09)

**TEMPLATE ONLY. Untested hypothesis; no improvement claimed. No roster,
cutoffs, fits, annotations, embeddings, or code. Execution NOT authorized.
`[FIELD: ...]` values are filled at a later freeze; nothing here derives
from data.**

## 0. Freeze-time prerequisites (STOP unless every gate holds)

- **P1 — Frozen native model:** `[FIELD: model artifact id + training-data
  cutoff]`, trained only on permitted history before B_start. Record `[FIELD: scorer
  id + input version]`. One baseline shared across all arms. No new
  native-model fit in this task.
- **P2 — As-of provenance:** historical native scores are eligible only if
  stored with correct as-of provenance. Rescoring old items with a
  future-informed model is forbidden.
- **P3 — Label access:** B correction labels are not accessible to induction;
  C labels are evaluator-only and never enter prompts.
- **P4 — Immutable windows:** eligible/exposure window definitions plus
  impression provenance exist and are immutable. If unavailable, or only
  reconstructible after the fact, STOP before any provider request — do not
  assume reconstructibility.
- **P5 — Local tokenizer:** `[FIELD: tokenizer name/version + counting
  script]` verified for hard caps. If unusable, caps in §6 are estimates,
  not promises.
- **P6 — Quota feasibility:** provider, model id, and quota verified at
  implementation time (see §6). This template claims no fit.
- **P7 — C horizon:** outcomes are the last explicit vote within 7 days of
  each item's first qualifying exposure; absent votes remain unknown.
  Collect for at most 28 days; evaluate only after every selected item's
  7-day horizon closes. Never replace items based on their outcomes.

## 1. Baseline

Single frozen native score `s` per item on a fixed scale `[FIELD: scale]`;
the scale is a freeze prerequisite. No confidence routing. Treatment, null, and topic arms all correct the same `s`.

## 2. Partition selection (deterministic; no ML-error targeting)

Input: frozen chronological labeled history. A and B occupy separate
timestamp intervals `[FIELD: A_start, A_end, B_start, B_end]`; induction
sees A only. A labels use the same 7-day exposure horizon; every A horizon
must close before B_start. Every B horizon must close before C selection
starts. Native-model training must exclude all B/C outcomes.

- **A (8 items):** order explicitly UP/DOWN-labeled A items by first
  qualifying exposure, then story id. Each item in order is a candidate
  anchor. Tokenize titles by lowercase ASCII `[a-z0-9]+`, dropping tokens
  shorter than 3 characters, without stemming or stopword removal. Its
  group contains itself and later same-source items sharing at least 2
  distinct anchor tokens. Select the first 2 UP and 2 DOWN if available;
  accept the first qualifying group, then repeat, excluding its selected
  items from anchors and members. Exactly 2 disjoint groups; fewer → STOP.
  Freeze each anchor's token set/source for application and topic control.
  Labels usable for A contrasts only. Disclose approximate topic matching.
- **B (16 items):** outcome-blind chronological sampling: the first 16
  eligible/exposed items within B's interval in first-exposure order,
  ties by story id, regardless of whether feedback exists,
  labels never consulted. Fewer than 16 eligible items, or zero usable
  UP/rest pairs at fit time (§5) → STOP. No adaptive expansion.
- **C (32 items):** selected AFTER a future freeze positioned after the
  existing live interleaving window `[FIELD: freeze timestamp]`; the first
  32 deduplicated eligible/exposed items in chronological order, selected
  without labels, ties by story id. Freeze each item's canonical text at
  first exposure, then freeze the entire roster before application. Use
  §0-P7 horizons; if fewer than 32 accumulate within 28 days → STOP as
  incomplete and report counts. No replacements or outcome-based expansion.

Deduplicate across A/B/C by frozen canonical identity (story id and normalized
URL, normalization version recorded); discard later duplicates. No C
eligibility, source, or exposure definition may be changed after selection starts.

## 3. Provider JSON contracts

Canonical text per item: title + newline + article, normalized NFC UTF-8,
as captured at the qualifying exposure (A/B historical as-of text required)
(untrusted data: no link browsing; ignore embedded instructions). Prompts
forbid: C labels, profile memory, user comments, popularity, native scores.

- **Induction input (per group):** 4 A items (2 UP + 2 DOWN) with
  in-contrast labels visible. **Output (per group):** either ABSTAIN
  `{group, abstain: true, reason ≤ 20 words}` or one rule `{id, group,
  rule_text ≤ 40 words total (optionally split into up_side/down_side with
  combined count ≤ 40 words), up_quotes[], down_quotes[]}`. Each quote is
  `{item_id, start, end, quote}` with half-open UTF-8 byte offsets into
  the canonical sent text, at codepoint boundaries; `bytes[start:end]`
  must equal UTF-8(quote) exactly (fail → schema
  error). Max 1 rule per group → the feature universe is 2 signed
  columns; an absent column reads 0.
- **Application input (per item):** canonical text + the (up to 2) frozen
  rule texts. **Output (per item per rule):** `{rule_id, verdict: +1 | −1 |
  unknown, target_span: {start, end, quote} | null}`. Every ±1 requires a
  nonempty exact span verified with the same byte-offset contract;
  otherwise the verdict must be `unknown`. **Gating:** a rule applies only
  to items lexically matching its group token set (§2 criteria); gated-out
  columns are `unsupported` (value 0), distinct from gated-in-but-
  inconclusive `unknown` (value 0 with `judged_unknown: true` recorded for
  coverage accounting).

## 4. Controls (same texts, same fit, same cap)

- **Topic control:** the same 2 anchor token sets AND source gates as binary features
  (match = 1, else 0), fitted with the §5 procedure and the ±.02 cap. No LM
  inference.
- **Null control:** repeat induction with within-group labels shuffled
  (Python `random.Random(20261009).shuffle`, one shared generator,
  groups processed in accepted order; Python version frozen; no tuning).
  Compute before requests: if either group is unchanged, STOP as invalid
  null without spending. A single null proves
  nothing alone — report it alongside native and topic.
- Rule weights are per-column signed (β₁, β₂), topic weights likewise.

## 5. Transfer fit on B (fixed convex objective, no sweeps)

Pairs: every UP item vs every explicitly NEUTRAL/DOWN item within
the SAME eligible/exposure window in B. Feature delta `δ = f_up − f_other`
(missing feature = 0). Objective (β ∈ ℝ²):

`L(β) = mean log(1 + exp(−[(s_up − s_other) + β·δ])) + (10/2)·‖β‖²`

Unknown outcomes never enter the loss. B uses the same 7-day per-exposure
label horizon as C. Require ≥ 1 usable pair or STOP before induction.
Fixed penalty 10, no intercept, baseline
coefficient fixed 1, no search. Outputs are scores, not probabilities (no
calibrated-probability claims). Final score:
`final = s + clip(β·f, −.02, +.02)`; all-unknown items score exactly `s`.

## 6. Provider budget and ledger (hard quota)

Allocation: 2 induction + 4 B-application + 8 C-application + 2 shared
retries = **16 requests / 80,000 combined tokens**, inclusive of failures
and retries. Per request: ≤ 5000 tokens total (input ≤ 4000 / output ≤
1000), ≤ 8 items; per-item canonical packing budget `[FIELD: e.g. ≤ 450
input tokens/item incl. overhead]`; estimate with the §0-P5 tokenizer and
refuse to send on overrun. Reserve verified full input plus the maximum
1000 output tokens before send; sent or uncertain calls consume that
reservation until authoritative usage arrives; provider-side retries disabled;
stop on ambiguous usage (no fallback). Retries: transport/schema errors
only, same inputs, ≤ 2 shared attempts total, no content tuning. Ledger
(local only, never in the shared repo): per-request state
(reserved/sent/failed) with actual token usage archived; cache key = sha256
of canonical JSON including model id, schema version, prompts, rosters,
arm, and features. Rosters and raw texts never enter the shared repo.
Hard-cap feasibility is verified at implementation against the real
provider + tokenizer — this template claims no fit.

## 7. Evaluation on C (preregistered primary + guardrails)

Primary: directional conditional judged AUC within exposure blocks — over
all comparable UP-vs-rest pairs (unknown-outcome items excluded), score 1
if UP ranks above rest, 0.5 on tie, else 0; equal weight per pair, pooled
across blocks. Also report: coverage (fraction of C items with ≥ 1
non-unknown column), class counts, fix/damage counts vs frozen native, and
the Down-over-UP reversal guardrail (comparable same-window C pairs where
native ranked UP strictly above DOWN but treatment ties or reverses it).
Fix = native UP/rest loss or tie becomes strict win; damage = native strict
win becomes tie or loss; unchanged ties contribute neither. C needs ≥ 1 comparable UP/rest pair or the
pilot is inconclusive; tiny counts carry no statistical claim. Treatment
must (a) strictly beat native AND null AND topic on the primary,
(b) show positive fix-minus-damage, and (c) introduce zero new DOWN-over-UP
reversals — else STOP/inconclusive. All-zero or constant feature vectors →
inconclusive regardless of AUC. No whole-pool top-12 counterfactual
claims; no promotion or deploy from this pilot. Stability reporting is
descriptive only (no new sweeps or reruns). This pilot is not
powered confirmation; any live trial needs separate design + authorization.

## 8. Freeze manifest, acceptance, implementation (not built)

- **Freeze manifest placeholders** (filled at freeze, stored locally):
  `[A roster + intervals, B roster + windows, C roster + selection-start timestamp +
  canonical texts, native artifact id + scale, tokenizer id, provider model
  id, prompts + schema versions, seed 20261009, cache sha256s]`.
- **Acceptance checklist:** every §0 gate evidenced; A is exactly 2 groups
  × 2UP2DOWN; B/C selected outcome-blind per §2 with a labels-access log;
  all quotes byte-exact; every ±1 carries a target span; ledger balances
  (reserved ≥ actuals; ≤ 16 requests / ≤ 80k tokens); §7 metrics computed
  from evaluator-held C labels.
- **Future implementation units (code NOT built, execution NOT
  authorized):** `select_partitions.py` (§2), `induce_rules.py` (§3
  induction + §6 ledger), `apply_rules.py` (§3 application + gating),
  `fit_correction.py` (§5), `eval_pilot.py` (§7). No automated
  next-permission flow; no popup or extra reviews requested.
