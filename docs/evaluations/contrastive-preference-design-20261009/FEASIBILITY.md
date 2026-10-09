# Contrastive preference pilot — feasibility (2026-10-09, read-only)

Archive note: initial Muse source-scoped assessment, retained after the user
set this branch aside. No complete inventory of historical backups/artifacts
was performed. Its absolute absence/recoverability claims below exceed the
inspected scope; they establish missing support in inspected source, not
proof that every retrospective alternative is impossible. Current decision
is in STATUS.md; no follow-on capture or implementation is planned.

**Verdict: BLOCKED.** The literal SPEC cannot be met retrospectively from
existing artifacts. No ranking calls, fits, embeddings, tests, or mutations
were performed for this review.

## Gates

- **P1 frozen native model — BLOCKED.** No frozen ranker artifact store
  exists. Production scores are recomputed live from current DB state
  (`scripts/eval_ranker_variants.py:253-275` `_production_scores` refits via
  `_score_and_rank` per fold). Warm-start cache explicitly "cannot score"
  (`pipeline/joined_classifier.py:142-146`; `pipeline/warm_store.py:78-98`
  saves warm-start arrays only). `pipeline/model_manifest.py:18-30` versions
  only embedding ONNX/tokenizer bytes, not ranker weights or training cutoff.
- **P2 as-of score provenance — BLOCKED.** No stored-scores table exists:
  the 16 `CREATE TABLE` sites in `database.py` (`256,298,329,341,355,373,
  385,409,420,461,479,490,502,528,595` + migrator) include no
  native-scores archive. Rescoring old items would use a future-informed
  model — forbidden by SPEC §0-P2.
- **P4 exposure provenance / immutable windows — PARTIAL schema, BLOCKED
  literally.** Capability: `interaction_events` stores `impression` rows with
  `dashboard_version/position/sort_mode/occurred_at`
  (`database.py:420-440`); `interleave_decks` stores per-version arms/windows
  (`database.py:409-417`); vote-crediting joins last impression to deck
  (`scripts/interleave_report.py:65-112`). But window definitions are
  reconstructed after the fact (`MIN(occurred_at)` pattern in
  `scripts/eval_ranker_variants.py:1333-1336`), not immutable frozen rosters.
  Actual archive availability: UNKNOWN (no row reads or count sweeps per
  review budget; schema ≠ populated archive).
- **First-exposure canonical text — BLOCKED.** `stories` is a mutable
  upsert: longest-wins text merge + metadata overwrite
  (`database.py:636-724`, `ON CONFLICT(id) DO UPDATE` at `706-724`). No
  `story_snapshots`/as-of text table exists among the 16 tables above. Only
  current text is recoverable.
- **P7 vote-event history (last vote ≤7d of first exposure) — BLOCKED
  retrospectively.** `feedback` is keyed `(user_id, story_id)` with
  `action/updated_at` overwritten on conflict (`database.py:595-603`,
  `1200-1220`). No vote-history table exists. First-impression time is in
  principle derivable, but the in-window last vote is unrecoverable after
  overwrite; existing evals note "`updated_at` approximates chronology."
- **Dedup identity — PARTIAL.** `normalize_url` is idempotent and shared
  (`dedup.py:67-136`; used as `_url_group` in
  `scripts/eval_ranker_variants.py:1406-1407`). No recorded normalization
  version or frozen canonical-identity log; HN-dupe resolution is
  best-effort with retry state (`database.py:461-470`,
  `pipeline/hn_dupes.py:214-295`). Capability yes, immutability no.
- **P5 local tokenizer — BLOCKED.** Only the embedding tokenizer exists
  (`pipeline/ranking.py:724`, `pipeline/config.py:126-127`). The LLM path
  states "a character-based estimate is not a tokenizer"
  (`llm_limiter.py:72-73`). No provider-LM tokenizer + counting script.
- **P6 quota (Muse via opencode) — UNKNOWN.** Accounting capability exists
  (`opencode stats` lists usage; `opencode models` lists `opencode/*`
  Muse models; `llm_usage_daily` in `database.py:373-382`). Automatic
  retries, output-limit enforcement, and tokenizer identity were NOT
  verifiable from `opencode --help` / `run --help` / `debug --help` alone
  (bundled binary, no installed-source inspection surface); no provider
  request was made per instructions. Hard-cap fit is unverified, as the
  SPEC itself states.
- **P3 label access — PASS (procedural).** Per-user `feedback` rows plus a
  local evaluator-held C set can satisfy the access rule; requires a
  labels-access log at freeze (not yet built).

## Inspected scope

`SPEC.md`, `PROPOSAL.md`, `STATUS.md` (this folder); `AGENTS.md` project
rules; targeted `rg` + reads only: `database.py` schema/upsert/feedback,
`dedup.py:67-136`, `pipeline/model_manifest.py`, `pipeline/ranking.py:697-744`,
`pipeline/joined_classifier.py:130-209`, `llm_limiter.py`,
`scripts/eval_ranker_variants.py:253-348,1330-1421`,
`scripts/interleave_report.py:60-112`, `opencode --help/models/stats/run/debug
--help`. No DB row reads, no count sweeps, no SSH, no provider calls, no
private values printed.

## Uncertainty

Schema capability ≠ archive availability for impressions/decks/votes; live-DB
inventory (row windows, survivor bias pre-2026-07-15 noted in
`scripts/narrowing_report.py:15`) was out of scope. Provider retry/output-limit
semantics remain unknown pending installed-source or authoritative-doc check.

## Smallest next step (no spec change made here)

Prospective capture only: freeze native artifact id + cutoff, log as-of
scores, snapshot canonical text at first exposure, append-only vote events,
and record deck/window + dedup-normalization versions — then select C
prospectively under §0-P7. Any retrospective relaxation (e.g. current-text
rescoring, `updated_at`-as-chronology) needs a justified protocol alternative
and explicit authorization; not implemented by this review. No inference quota
used beyond this review.
