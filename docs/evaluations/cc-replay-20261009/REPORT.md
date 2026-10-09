# CC-review replay: retrospective ML baseline + blinded Muse audit (2026-10-09)

Export note: committed scripts have formatting and unused-code cleanup for
repository checks. Original run copies remain on the VPS; results were not
recomputed during export. These scripts are reusable copies, not byte-exact
evidence of the original run.

Decision: **The exploratory gate failed; no challenger or production change.**
Muse corrected 2/5 confident ML errors (40%, gate ≥60%) and agreed with ML on
1/5 correct controls (20%, gate ≥70%). Fixes−damage = 2−4 < 0: the CC stop
rule. Stability 4/4 does not rescue accuracy. All development-only: the 99 stories were
already inspected, so nothing here is independent validation.

## 1. Retrospective replay (honest, not exact replication)

- Training cutoff = original eval now `1791401987.9873054` (3848 user-151 votes);
  99 window votes scored, never in training (feedback PK is `(user, story)`).
- Production config from `main/config.toml` (RBF SVM C=4, linear blend on, side
  embeddings on, engagement features off), one bounded fit (14.3 s, threads=1,
  `XDG_CACHE_HOME` redirected, warm-start writes kept out of shared cache).
- Live DB never opened (mode would be ro anyway); all reads from task-owned
  snapshot `/tmp/opencode/cc-replay/snapshot.db`; pipeline cache writes landed
  there. Cached embeddings reused with text-hash validation: **0 drift drops**,
  99/99 scored. No local model inference.
- Mutable-content limits (stories table has no history; values are
  snapshot-current): HN score/comments do **not** enter replay scores (tier-1
  gravity weight ~0 at 3848 votes; engagement features off). `text_len`/TF-IDF
  use current text — reported, identical footing for any future challenger.
  Matching cached text hashes establish consistency with this snapshot, not
  proof that content was unchanged since the historical training cutoff.
- Endpoints: ordinal primary `P(up)−P(down)` (native up>neutral>down order);
  UP-vs-rest `P(up)` reported too.
- Replay: ordinal AUC **0.752** (3074 pairs, cluster95 [0.675, 0.823]); UP-vs-rest
  **0.741** (1394 pairs, [0.597, 0.864]); pred-class accuracy 0.465
  (labels 17 up / 42 neutral / 40 down). Exposure bias note: every voted story
  was first served by a ranker, compressing score contrast.

## 2. Blinded audit (frozen before ranking)

- Pool: same-source informative pairs, hn needs shared title token (topic proxy),
  ML margin [0.10, 0.90): 17 pairs → **5 errors + 5 controls, 20 distinct
  stories, cap-1 reuse**; 9/10 pairs down-vs-neutral (limitation).
- Muse (`opencode-go/muse-spark-1.3-contributor` only, 3 real batches 7+3+4rev,
  one frozen training-only profile, no-tools child env, opaque IDs, isolated
  cwds, validated JSON, all first-attempt OK; invocation model verified, no
  Luna text, clean stderr). The supervising agent subsequently checked relay
  request logs: all requests in the run used Muse Go, with no fallback events.
- Results: corrections **2/5**, control agreement **1/5**, stability **4/4** →
  gate fails on both accuracy arms.

## 3. Fixes-vs-damage audit (why no feature)

- Fixes: p-01 (Japan over onebag-Q, conf 55), p-02 (museum over sim, conf 55).
- Damage (all high-confidence 68–85): p-06 thorium over space-weapons, p-08
  withdrawal-github over withdrawal-tweet (near-duplicates, opposite labels),
  p-09 demoscene over Ask-HN-games, p-10 Docker over Meta-dossier. Muse
  re-picks the more on-profile-topic story — the DOWN-voted one. p-05: Muse
  even picked neutral over a true UP (conf 86).
- The one natural deterministic feature this suggests (title-novelty vs prior
  ups) is exactly the familiarity direction the prior experiment already tested
  and was **unsupported** (5/14). This small sample did not justify a new feature;
  it cannot rule out useful features in general.
  General source onehots already exist and were not reinvented.

## 4. Validation: pending, nothing to validate

- Post-window user-151 votes (`> 1791511537.886615`): **0**. No untouched data,
  and no candidate (rejected). Method for any future candidate is frozen in
  `validation_readiness.json`. No deploy/restart/interleaving change (closes
  2026-11-05; untouched).

## Quota & files

- Supervising readback verified disjoint training/candidate IDs, 20 distinct
  audit stories, 14 valid answers, reported counts and unchanged frozen hashes.
  The preregistration's manually entered `12:30:00Z` timestamp is inaccurate;
  filesystem modification time is `12:06:47.290877Z`. Preserve the original
  artifact and use its hashes and run ordering as the audit record.

- Fresh Muse batches: **3/4** (14/24 comparisons). No further ranking.
- This dir: `replay.py`, `select_pairs.py`, `score_muse.py`, `evaluate.py`
  (safe reproducible scripts); `replay_scores/metrics/uncertainty.json`,
  `selection/preregistration/settings*.json`, `muse_pair_scores/batches.json`,
  `results.json`, `validation_readiness.json`, `REPORT.md` (aggregates).
- Private, local, untracked (never to mirror): `profile.txt`, `muse-cache.json`
  (raw prompts/answers), snapshot DB in `/tmp/opencode/cc-replay/`.
- Mirror to laptop later: aggregates + scripts only.
