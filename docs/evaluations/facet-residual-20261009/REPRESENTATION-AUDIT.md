# Representation audit: does the proposed comment-free comparison isolate comments?

## Primary review and scope correction (2026-10-09)

This report is a Muse source audit, not an executed experiment. The initial
question concerns the **input to the primary embedding**, not removal of every
comment-dependent ranking feature. For that comparison, keep stored story rows,
lexical features, metadata and side vectors identical between arms; change only
primary vectors and the similarity features derived from them. Section 4's
proposed TF-IDF/text-length changes would answer a broader question and are not
adopted. Keeping side vectors fixed tests the primary-input change conditional
on the existing side model; it does not test a wholly comment-free ranker.

Production prefers stored text_content; recomposition can change other content.
First compare stored text byte-for-byte with the full composer on an isolated
snapshot, and count mismatches before calling the contrast comments-only.
Existing body-section code is reusable, but earlier section experiment results
must be located before repeating any encoding.

Replay NPZ text_hashes identify the source story input; its model/settings field
identifies the section transform. Source-hash validation does not itself reject
body vectors saved by the existing encoder. Sections 5.3 and the stdout claim
overstate the need for new hash plumbing. A new experiment should explicitly
record both source and transformed-input hashes, plus settings, to make that
distinction reviewable. No cache corruption was demonstrated.

Next step: inspect task-owned VPS snapshots and existing section artifacts,
read-only, to establish coverage, source/composer agreement and reusable vectors.
Freeze cohort, split, config and cache identities before any encoding or fits.

Scope: bounded read-only source audit (no DB access, no runs, no counts verified).
Ground truth is the live tree; line numbers are the current checkout.

## 1. Exact current production embedding input

- Selector: `story_embedding_text` (`pipeline/ranking.py:647-656`) returns
  `story.text_content` verbatim whenever it is non-empty; only empty-text
  stories fall back to `compose_story_text(title, self_text, top_comments,
  article_body)`.
- Composer (`pipeline/ranking.py:623-644`): `clean_text` each part, then
  char-caps `self[:6000]`, `comments[:6000]`, `article[:4000]`, joined in
  order **title, self, article, comments**.
- Cleaner (`pipeline/ranking.py:413-439`): HTML-strip + unescape, strip
  box/line unicode runs, collapse whitespace, drop text with
  `alnum/len < 0.5` or `len <= min_len`.
- `text_content` is comment-bearing by construction at every ingestion
  point: CH live rows start title+self (`pipeline/enrichment.py:64-86`,
  `top_comments=""`), then comment prewarm recomposes title+self+comments
  +article (`pipeline/enrichment.py:307-312`); Algolia refresh does the
  same while preserving the stored article (`pipeline/enrichment.py:168-173`);
  the non-HN merge helper does the same (`pipeline/enrichment.py:434-436`);
  `Database.upsert_story` keeps the longest self/comments/body and
  recomposes via `compose_story_text(title, final_self, final_comments,
  final_body)` (`database.py:636-696`).
- Comment supply: `_select_top_comments` (`pipeline/ranking.py:518-596`,
  knobs `pipeline/ranking.py:303-311`: limit 40, 4 core threads, max 6 per
  thread) joined whole-comment-or-skip under 24k chars on `\n\n---\n\n`
  (`pipeline/ranking.py:599-620`, `HN_COMMENTS_CACHE_CHAR_LIMIT` at
  `pipeline/ranking.py:316`).

## 2. Truncation / aggregation / cache settings (all must be frozen)

- Tokenizer truncation: `Embedder.encode` uses `truncation=True,
  max_length=self.max_tokens` (`pipeline/ranking.py:817-825`); production
  `max_tokens=4096`, model `mxbai-embed-xsmall-v1|mean|norm|4096`
  (`config.toml:3-5`, defaults `pipeline/config.py:89-93`).
- Aggregation: attention-mask mean-pool + L2 normalize
  (`pipeline/ranking.py:837-850`); 384-d (`pipeline/ranking.py:786-800`).
- Cache identity: `get_or_compute_embeddings`
  (`pipeline/ranking.py:890-928`) keys stored vectors on
  `sha256(story_embedding_text) + model_version`; any challenger text
  needs its own isolated cache/table or hash domain — reusing the
  production cache rows would silently score production vectors.
- Ranking pool shortcut: `candidate_cache._ranking_copy`
  (`pipeline/candidate_cache.py:69-72`) empties `self_text/top_comments/
  article_body` because ranking reads only `text_content` **length**
  (`len(s.text_content)` at `pipeline/ranking.py:1405-1407` train-equivalent
  `1508-1510`).

## 3. How article_body / self_text / comments are used elsewhere in ranking

Comment text leaks into ranking through three non-embedding channels; a
comment-free arm that only swaps vectors does **not** isolate comments:

1. `text_length` meta column = `len(text_content)`, comment-bearing
   (`pipeline/ranking.py:1045-1092` layout, filled at `1405-1407`,
   `1508-1510`; scaled with the other meta columns at `1574-1580` —
   StandardScaler touches meta only, never embeddings).
2. TF-IDF words = `tfidf_text` = domain + source + title +
   `text_content[:5000]` (`pipeline/linear_blend.py:45-51`), used by the
   live linear blend (`config.toml:38` sets `linear_blend_enabled=true`)
   and by `joined_logistic` via `count_rows`
   (`pipeline/joined_classifier.py:75-107`).
3. Similarity/cluster meta features are vector-derived
   (`pipeline/ranking.py:1385-1404`) and follow the arm's vectors
   automatically — fine. Source one-hot (`1408-1414`) is comment-free.
- Open definition gap: live config also sets `side_embedding_enabled=true`
  and `svm_c=4.0` (`config.toml:37-43`). NEXT-EXPERIMENT step 2 says
  "the production encoder" (singular). If the arms run with the frozen
  production config, the Gemma side vectors (a second model over the same
  comment-bearing text) also need a comment-free re-encode, or both arms
  must pin `side_embedding_enabled=false` — which is then not production
  config. Freeze one option explicitly.

## 4. Revised frozen definitions (comment-free, nothing else moves)

- Baseline text `T_full(s)` = `story_embedding_text(s)` exactly
  (i.e. stored `text_content`, fallback compose) — byte-identical to
  production, including its staleness (whatever comments were in the row).
- Challenger text `T_nocom(s)` = `compose_story_text(s.title,
  s.self_text, "", s.article_body)` — identical cleaner, identical
  6000/4000 char caps, identical title-first ordering; the empty-comments
  branch contributes nothing, so ordering matches production. This is
  exactly the existing `"body"` section
  (`pipeline/embedding_sections.py:63-83`, esp. `75-76`); do not invent a
  second definition. (Note: `section_texts` at
  `pipeline/embedding_sections.py:27` with `article or self or
  text_content` priority is a different, chunked-coverage helper — not
  the challenger.)
- Encoder/truncation/aggregation frozen: production ONNX dir + tokenizer,
  `max_tokens=4096`, batch path of `Embedder.encode`, mean-pool, L2 norm.
- Ranker frozen: same snapshot config (including the blend/sideDecision
  from §3), same `min_up_for_svm/min_down_for_svm` gates, same tier blend
  and MMR/dedup downstream.
- Leakage closure: challenger arm must also feed `T_nocom`-derived
  `text_length` (`len` of the challenger text, not stored `text_content`)
  and `T_nocom`-derived TF-IDF/word rows; else the "comment-free" delta
  is attenuated by comment-bearing metadata. No hidden fallback: stories
  with empty title+self+article on both arms embed title-or-empty per
  `story_section_text` convention — count and report them, do not
  substitute comments.
- Hash/cache discipline: challenger vectors keyed on `sha256(T_nocom)`,
  stored separately from production rows; never mix.

## 5. Can the existing harness reproduce production ranking?

- `scripts/eval_ranker_variants.py` — **yes, with prerequisites**.
  `_production_scores` (`:253-321`) replays the real `_score_and_rank`
  on a per-fold in-memory DB (`:174-234`) with `_FrozenEmbedder`
  (`:164-170`), which raises on any uncaptured encode — so the snapshot
  must contain a valid stored vector for **every** candidate and train
  story under the arm's exact text hash. Snapshot trust comes from
  `frozen_database` + `_snapshot_context` (`:90-136`, requires
  `config_json`, `evaluation_time`, `database_snapshot`+sha,
  `user_id`), external-vector validation (`:447-494`, unit-norm check
  `:139-161`), chronological expanding-window splits
  (`_temporal_splits` `:2203-2233`, holdout variant `:2236-2251`,
  URL-group dedup `:2124-2154`), and per-fold train cutoff discipline
  (tier1 `now` = max train vote time, `:921-963`).
- Gaps for this experiment:
  1. **No ordinal (Down<Neutral<Up) AUC metric.** The harness scores
     `auc_up_vs_rest/down/all`, NDCG@k, recall (`:1925-1983`); the only
     up-minus-down scorers are variant scores (`:1076-1083`,
     `:1539-1602`), not metrics. NEXT-EXPERIMENT's primary (within-block
     ordinal AUC pooled by ordinal-pair count) needs a new metric
     function; the facet harness's `ordinal_auc`
     (`docs/evaluations/facet-residual-20261009/evaluate.py:149-177`)
     is the reference implementation but lives in a harness that fits
     proxy/facet logregs on OOF percentiles, not production ranking.
  2. **"Reuse the chronological blocks" is ambiguous.** The facet blocks
     b2–b5/dev99 (`evaluate.py:275-320`) are OOF-percentile cohorts, not
     the harness's vote-time folds. Freeze which split the gates apply
     to before encoding.
  3. **Replay plumbing rejects challenger vectors by design.**
     `_load_replay_embeddings` (`:1754-1777`) requires exact production
     text-hash match (else "stale"); `encode_replay_embeddings.py --section body`
     (`scripts/encode_replay_embeddings.py:131-135,244-254`) encodes body
     text but `_save` (`:55-63`) writes **production** text hashes — an
     identity conflation hazard. A two-arm run needs a snapshot whose
     stored rows (or a parallel vector domain) carry challenger hashes,
     plus per-arm `FoldData` (separate `cand_emb`/`train_emb`, cf.
     `_make_fold` `:2104-2200`); this is new harness work, not a flag
     flip.
  4. `evaluate.py` (facet harness) **cannot** reproduce production
     ranking at all — univariate/multinomial logregs on OOF percentile +
     proxy/facet one-hots. Do not reuse it for this experiment.
- Unverified from this audit (no DB access): the "3,848 voted stories"
  count, the "how many items have no article/selftext" count, and any
  block membership — none are marked verified here.

## 6. Smallest next executable step + blockers

- Next step (frozen, no training): on a task-owned snapshot, dump the
  frozen story rows + `T_full`/`T_nocom` texts and their sha256 hashes
  for the vote cohort, and confirm byte-level that `T_nocom ==
  story_section_text(s, "body")` for all rows; report empty-body counts.
  No encode, no fit, no live writes.
- Blockers before any encode/fit: (a) freeze §4 definitions incl. the
  side-embedding decision and TF-IDF/text-length closure; (b) freeze
  which chronological split the gates apply to (harness folds vs facet
  blocks); (c) specify the challenger-vector snapshot/harness plumbing
  (§5.3); (d) add the ordinal-AUC metric to the canonical harness
  (§5.1). Only then authorize the isolated re-encode.
