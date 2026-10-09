# Data audit: stored text vs full composer on the frozen vote cohort

## Primary review (2026-10-09)

The audit counts below are verified aggregates. The protocol later in this
Muse report predates the user's questionnaire choice and is superseded by
NEXT-EXPERIMENT.md's user-selected endpoint section: find more upvotes is the
priority, UP-versus-rest AUC is primary, and ordinal AUC is descriptive only.
For mismatched rows the selected rule is **retain the original vector**, not
rebuild the input while flagging it. Derive old-cohort counts before encoding.
The replay hashes are source identity plus recorded transformation settings;
no corrupted cache or rejection of body NPZs was demonstrated by this audit.
The bounded search found no completed body-only comparison in the searched
directories; it does not establish absence elsewhere. The source labels for
the three RSS mismatch rows are rss_reddit_digitalnomad,
rss_reddit_localllama and rss_reddit_maps.

Claude's read-only review supports proceeding with these conditions: separate
fresh processes and task-owned DB copies per arm; no story upserts; identical
story/side inputs; disk warm store disabled; cached baseline probabilities
reproduced before new encoding; approximately 20 full-text encoder checks
against stored vectors (cosine >=.9999) before body encoding; source/input/vector
hash manifest and fixed encoder settings. Warm-start state held in memory per
user must not carry a later baseline fold into an earlier challenger fold.
The new chosen score endpoint needs a fixed scoring clock in both arms;
historical baseline probability reproduction alone does not reproduce served
ranking scores. No new encoding, fitting or metric result is claimed here.

Status: read-only audit executed on a task-owned VPS snapshot. No embedding
inference, no fits, no ranking annotations, no external paid calls, no live
DB/service/shared-cache touches. This document reports aggregates only; no
private titles, bodies, or comments below.

Primary scope correction (REPRESENTATION-AUDIT.md §primary review) overrides
its §4 body: the next two-arm comparison changes primary embedding inputs
only and keeps stored story rows, lexical features, metadata, and side
vectors identical between arms. Vector-derived similarity/cluster features
follow the arm's vectors. No new annotation, encoding, or fit has run here.

## Exact inputs

- VPS project: `/home/dev/hn-rewrite/main` (git root; AGENTS.md + STATUS.md
  read before scripts; STATUS 2026-10-08 interleaving + Arctic Shift live).
- Task snapshot: `/tmp/opencode/cc-replay/snapshot.db` (1.2G) + `-shm` 32K +
  `-wal` 0 bytes. WAL confirmed empty before and inside the script.
- Frozen cohort map: `/home/dev/hn-rewrite/llm-labels/facet-residual-20261009/sample.json`
  (`seed 20261009`, `user_id 151`, `n_old 3848`, `n_dev 99`,
  `block_sizes {769,770,769,770,770}`, 4 cut times).
- Audit script (typed): `docs/evaluations/facet-residual-20261009/audit_inputs.py`
  (`script_sha16 4669c130dca648bb`). VPS copy:
  `/home/dev/hn-rewrite/main/docs/evaluations/facet-residual-20261009/audit_inputs.py`.
- Aggregate output: `docs/evaluations/facet-residual-20261009/data_audit.json`
  (local copy of `/tmp/opencode/cc-replay/work/data_audit.json`).

## Exact commands (VPS via `ssh hetzner`)

```bash
ls -lh /tmp/opencode/cc-replay/snapshot.db*
stat -c '%n %s' /tmp/opencode/cc-replay/snapshot.db-wal  # 0
scp docs/evaluations/facet-residual-20261009/audit_inputs.py \
  hetzner:/home/dev/hn-rewrite/main/docs/evaluations/facet-residual-20261009/audit_inputs.py
/home/dev/.local/bin/uv run --project /home/dev/hn-rewrite/main \
  python /home/dev/hn-rewrite/main/docs/evaluations/facet-residual-20261009/audit_inputs.py \
  --snapshot /tmp/opencode/cc-replay/snapshot.db \
  --sample /home/dev/hn-rewrite/llm-labels/facet-residual-20261009/sample.json \
  --user-id 151 --out /tmp/opencode/cc-replay/work/data_audit.json
rg --files --hidden --no-ignore \
  /home/dev/hn-rewrite/main/docs/evaluations/model-ablation-20261007/single-model | head
rg --files --hidden --no-ignore /tmp/opencode | head
find /home/dev/hn-rewrite/main/docs/evaluations/model-ablation-20261007/single-model \
  -maxdepth 2 -name '*.npz'
find /tmp/opencode -maxdepth 4 \( -name '*.npz' -o -name '*body*' \)
```

Read mode: `sqlite3 'file:/tmp/opencode/cc-replay/snapshot.db?mode=ro&immutable=1'`
and the script's `file:...?mode=ro&immutable=1`. Single
`feedback JOIN stories WHERE user_id=151` query only; no whole-DB scan.
Local checks: `ruff check` + `ruff format --check` + `ty check` on the new
script (clean). No test suite run claimed here.

## Results (user 151, snapshot now)

- `voted_rows 3947` = frozen `blocks1..5 3848` + `unmapped 99`.
  Blocks: `769/770/769/770/770`; unmapped labels `down 40 / neutral 42 / up 17`
  exactly match `sample.json` dev99 coverage, i.e. post-freeze votes.
- Labels overall: `down 1637 / neutral 1458 / up 852`.
  Per frozen block in `data_audit.json` (e.g. b1 `266/351/152`, b5 `442/228/100`).
- `stored == compose_story_text(title,self,top_comments,article)`: `3896/3947`
  equal, `51` mismatch (1.3%). `empty_stored_text 0`.
- Mismatch by source (equal vs total): `hn 2372/2408` (36), `ch_seed 144/156`
  (12), `rss_digitalnomad 106/107`, `rss_localllama 88/89`, `rss_maps 1/2`.
  All other sources 100% equal.
- No self/article content: raw-empty `512`, cleaned-empty `516`.
  `T_nocom == story_section_text(body)` on all `3947` rows; `T_nocom`
  trivial (title-only) on the same `516` cleaned-empty rows.
- Source spread is wide (`hn 2408`, `ch_seed 156`, reddit/LW/RSS long tail);
  full table in `data_audit.json`. No private text emitted.

## Existing body-section artifacts (bounded, metadata only)

- `rg --files` over `.../model-ablation-20261007/single-model` lists only
  `.json/.log/.md` screen/broad/check/feature-removal files; over
  `/tmp/opencode` lists only the two `snapshot.db*` copies, uv cache,
  `facet_check_invariants.py`, and rsync-test files.
- `find ... -name '*.npz'` in both scopes: no files. No `*body*` NPZ/results
  in `/tmp/opencode`. No heavy model loaded; home not scanned.
- `rg -l '"section"|section.*body'` over single-model: no hits. The
  `encode_replay_embeddings.py --section body` path exists
  (`story_section_text(story,"body")`) but no saved body vectors or scores
  were located in the two bounded dirs.
- Nearest prior cohort metadata found:
  `single-model/cohort-audit.json`: `u151-stored.npz` 3239 vectors / 3234
  matched feedback, 384-d stored mxbai; Gemma NPZs same 3234-row cohort.
  `embedding-summary.json` reports matched-embedding AUC/AP/P@12 deltas ~0
  with wide intervals — a different question (stored vs alternate encoders
  on a 3234-row matched cohort), not the comments-in/out contrast.

## Do previous section tests answer the two-arm question?

No. Settings/cohort differences:

- Cohort: single-model matched-cohort `3234` rows vs frozen full-vote
  `3848` (+ dev `99`); different time window and vote composition.
- Inputs: prior tests compare encoders/prompts/token budgets (e.g. stored
  384-d mxbai 4096 vs Gemma variants, 512-token checks); none isolates
  `T_full` vs `T_nocom` with the production `mxbai-embed-xsmall-v1|mean|norm|4096`
  encoder, truncation, and cache discipline frozen per arm.
- Ranking config: single-model wraps one-SVM capacity screens, not the
  production blend with identical lexical/meta/side inputs plus arm-following
  similarity features.
- Hash discipline: `encode_replay_embeddings --section body` writes
  production text hashes (REPRESENTATION-AUDIT §5.3 hazard); no located
  artifact records separate source + transformed-input hashes for a body arm.

## Feasibility decision

The comments-only baseline as a pure primary-input swap is feasible, with
one explicit qualification — not a silent redesign:

- `98.7%` of voted rows have byte-identical stored vs recomposed text, and
  `T_nocom == body-section` holds `100%`. The challenger definition
  `compose(title,self,"",article)` is therefore well-defined and reusable.
- The `51` mismatched rows (mostly `hn`/`ch_seed` staleness) mean stored
  `text_content` is not exactly the full composer everywhere. Do not call
  the contrast "comments-only" without handling: freeze one rule before any
  encode — e.g. baseline arm uses stored bytes verbatim, challenger uses
  `T_nocom`, and the 51 rows are retained with their mismatch flag reported
  (or a pre-registered exclusion sensitivity). Changing TF-IDF/text-length
  or side inputs to "fix" this would answer a broader question and is not
  adopted per the primary correction.

## Frozen next two-arm protocol (no runs authorized here)

- Cohort/splits: existing chronological full-vote `blocks1..5` (`3848`);
  expanding-window fits on earlier blocks, tests on `block2..block5`
  (b2, b3, b4, b5); dev `99` reported separately, never a gate.
- Arms: `T_full` = `story_embedding_text` (stored verbatim) vs `T_nocom` =
  `compose(title,self,"",article)`; production encoder/settings frozen;
  isolated embedding caches keyed on transformed-input hash plus recorded
  source hash; one CPU, capped threads.
- All else identical: same story rows, lexical/TF-IDF, metadata
  (incl. comment-bearing `text_length`), side vectors; only primary vectors
  and the similarity features derived from them follow the arm.
- Metrics: primary within-block ordinal AUC (`Down<Neutral<Up`) pooled by
  ordinal-pair count; secondary `UP`-vs-rest AUC pooled by UP/non-UP pairs,
  no override. Gates: pooled ordinal delta `>= +.01`, positive in `>=3/4`
  blocks, paired story-bootstrap lower bound `>0` (conditional on fits);
  per NEXT-EXPERIMENT.md. Fail stops representation rewriting on this evidence.
- Prospective Muse canonicalizer only on a pass, with a new freeze on
  post-design votes; existing 99-story outcomes cannot validate it.

## Unknowns / not verified

- Byte content of the 51 mismatches (only counts recorded; no text read out).
- Whether the live DB has drifted past the snapshot's `3947` votes.
- NPZ/results outside the two bounded dirs were not searched (home-wide
  scans explicitly avoided); absence is scoped to those dirs.
- No encode, fit, metric, bootstrap, or LLM call executed in this audit.
