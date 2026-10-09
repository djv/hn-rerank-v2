# Facet-residual 20261009 — COMPLETE report (annotation complete, gates assessed)

Export note: committed scripts have formatting, unused-code cleanup, and
explicit type annotations/contracts for repository checks. Original run
copies remain on the VPS; results were not recomputed during export. These
are reusable copies, not byte-exact evidence of the original run.

Status: **COMPLETE — 399/399 parsed, 20/20 repeat pairs, full evaluation run.
Verdict: STOP this bounded facet design (all 5 frozen gates fail).** No production claim;
no further labeling on this line without a new design. Code + aggregates
go to Claude for final review. No private bodies, caches, or profiles below.

## Protocol history (quota accounted)

- Budget: **20 annotation requests max, incl. interrupted/failed**
  (`facet_attempts.json` = 20/20, reserved atomically BEFORE each call).
- Body-input repair (amendment frozen BEFORE results,
  `body_fix_amendment.json`): `label_facets.bodies()` fell back to
  `text_content` (= comments) for 38/379 stories. **4 cached slots
  invalidated** (archived), **56 preserved**; rule since: self_text OR
  article_body OR EMPTY, `text_content`/`top_comments` never to Muse.
- **Deviation documented:** frozen plan was MAX_RETRIES=0 / STOP on failed
  batch. A sibling run retried the one failed 25-item batch under the same
  prompt hash and completed it (attempts 15→17/20 at the time, final 20/20
  within budget, inputs/hashes unchanged). Inputs stayed fixed; no extra
  quota consumed beyond the 20 cap.
- Repeats: 20 seeded repeats as separate opaque items in separate batches
  (separate calls by construction). No redraws, no rubric/sample/cache
  changes during evaluation.

## Stability gate (repeats ONLY, before any fit)

20/20 pairs complete. Agreement: artifact .85, claim .95,
conflict_framing .90, entity_focus .95, voice .85 → **retained**;
technical_depth .50, actionable .75 → **dropped**. Written to
`facet_fields.json` before any model score. No field picking after metrics.

## Code repairs in evaluate.py (frozen impl `31e892f4d805d5da` at
2026-10-09T13:20:38Z, BEFORE first fit; proxy rules hash unchanged
`ce196174ff706176`)

1. `vec()`: was `X=[cols]` (1×n) with mismatched proxy/facet branches.
   Now one row per story: `[pct] + proxy-onehot + facet-onehot` (n×p).
2. Proxy columns: were 5 always-1 membership flags; arm4 overwrote proxy
   with facets. Now true one-hot per frozen train-only vocab key (zeros
   for unseen test values); **arm4 concatenates BOTH** (width = 1+P+F).
3. Null mapping: was `zip(tr, flattened grouped perms)` (rows to wrong
   story IDs). Now filled **group-by-group within (raw source, block)**,
   train/test separately, independent RNG stream per (rep, split, part);
   asserts: same raw source AND block, unchanged facet-row multiset per
   stratum, no train/test transfer, class-free (no labels in mapping).
4. Proxy body length: was `len(self_text + article_body)` (mean 5582).
   Now `len(clean(self_text OR article_body OR EMPTY)[:1500])` — the exact
   annotator-visible input (mean 989, median capped 1500, 38 EMPTY; 224/379
   changed). Raw sources normalize rss_reddit_*→reddit (104),
   rss_lesswrong*→lesswrong (15), hn/bq_seed/ch_seed→hn (214), else other
   (46). Coarse proxy field types unchanged.
- sklearn: `LogisticRegression(C=1, balanced, lbfgs)` — no `multi_class`
  param (removed in current sklearn; 3-class multinomial preserved).
- Focused pure invariants (pass; no full suite): n×p shapes, per-key
  one-hot incl. unseen-zero negatives, arm4 width, retention-before-outcome
  assert, null stratum asserts. `ruff check` clean.

## Evaluation (fixed; threads=1, niced, load ~0.2, RAM free)

- Splits: train b2-3 (140) → test b4 (70); train b2-4 (210) → test b5;
  train 280 → test dev99. Arms: 0 native OOF pct, 1 pct-only fit,
  2 +proxy, 3 +facets, 4 +both, 5 = arm3 on 200 within-stratum
  whole-row perms. C=1 balanced, score P(up)−P(down), ordinal AUC pooled
  b4+b5 by pair count, dev99 separate, 2000 conditional paired-story
  bootstraps (stratified by block). Up-vs-rest AUC added as secondary
  metric only (same scores, no learner change, no gates).

## Results (ordinal AUC)

| test | arm0 nat | arm1 | arm2 proxy | arm3 facet | arm4 both |
|---|---|---|---|---|---|
| b4 | .817 | .817 | .781 | .760 | .734 |
| b5 | .752 | .752 | .712 | .706 | .689 |
| dev99 | .752 | .752 | .765 | .714 | .720 |
| pooled b4+b5 | .786 | .786 | .747 | .734 | .712 |

- Δ pooled arm3−arm1 = **−.051** (gate ≥+.02: FAIL).
- Null (200): mean −.047, p95 −.007; observed −.051 > p95? **FAIL**.
- Δ pooled arm4−arm2 = **−.035** (gate ≥+.01: FAIL).
- Δ dev99 arm3−arm1 = **−.038** (gate ≥0: FAIL).
- arm3 ≥ arm0 pooled? .734 < .786: **FAIL**.
- Bootstrap 95% CI (conditional on fits): d31 (−.105, +.003),
  d42 (−.093, +.020) — both include 0.
- Secondary up-vs-rest pooled: arm1 .7823, arm3 .7882, arm4 .7658
  (reported only; no gate, no conclusion drawn).
- Consistency: arm0 == arm1 exactly (univariate fit of pct is monotone →
  same ranking), as expected.
- **Verdict per frozen rule (arm3−arm1 < +.01): STOP this bounded facet design.**

## Standing caveats (unchanged)

- Current snapshot text scores historical votes (same footing all arms).
- All 280+99 are development data; exploratory only — GO would have meant
  larger labeling, never production. No external validation claimed.
- Transductive full-block percentiles; bootstrap CI ignores training
  variance (too narrow, stated).
- Conflict-excluded sensitivity omitted (declared allowed, not required).
- Outcome-blind sampling/feasibility (all train classes ≥10, test ≥5)
  held per frozen `sample.json`.

## Hygiene

- No annotation-model calls, no cache/sample/rubric edits in this step
  (mtimes verified); writes: `evaluate.py`, `facet_fields.json`
  (same 20-pair retention), `eval_report.json`, this report.
- No live DB / main / shared-cache / production changes.

## Final Claude review and corrections

See CLAUDE-REVIEW.md for the actual read-only response. The negative result is
credible for this bounded design, not a rejection of all LLM-derived features.
Proxy and shuffled features also lose quality, consistent with a small-data
regularization/feature-count penalty; causation is not established.

- Snapshot check: all 72,944 story rows identical on id, title, url, source,
  self_text, article_body between annotation and evaluation snapshots.
- Secondary pooling corrected after review: weights now UP x non-UP pairs
  (b4 741, b5 600), rather than three-class pair counts. Recomputed from the
  existing per-block scores; no fits, new annotation, or primary gate changes.
- Frozen rubric.json still lists text_content. Preserved as historical evidence;
  body_fix_amendment.json and actual repaired annotator excluded it before results.
- Raw-source null strata include singleton groups that cannot shuffle; the null
  is partly contaminated by actual features. No stronger inference drawn.
- C=1 penalizes baseline percentile along with facet coefficients. A protected
  baseline offset would be a new design requiring an independent test, not a
  retune on these spent outcomes.
