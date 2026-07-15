# Improvement brainstorm & Codex findings review, 2026-07-15

Two inputs reviewed together: a Fable 5 brainstorm (read-only agent, full repo
read + read-only DB SELECTs) and a separately-generated Codex brainstorm. Both
were cross-checked against the live tree and DB per the "trust the live tree"
rule in AGENTS.md. One finding (interaction-ledger data loss, §Codex 4.1 below)
was confirmed as a live bug and fixed same-day — see WORKLOG.md 2026-07-15
("fix: interaction ledger silently dropped whole batches") and commit
`3a5a77c`. Everything else here is unimplemented backlog, not yet planned.

Ideas already tracked in ROADMAP.md (F1-F3, B1-B3, REF-1-3, PERF-4) and in
`plans/recommender-features-research.md` were deliberately excluded from the
brainstorm prompt.

**Stale-doc finding:** the brainstorm prompt's constraint said "SVM over
MiniLM embeddings," but production flipped to **mxbai-embed-xsmall-v1 at
4096 tokens** (commit 8613c67, config.toml lines 3-5). ARCHITECTURE.md's
embedding-model table and the AGENTS.md "384-d MiniLM" scaling note may
still be stale in places — worth a pass.

---

## Fable 5 brainstorm

Provenance: read-only agent that read AGENTS.md, ARCHITECTURE.md,
WORKLOG.md, ROADMAP.md, `plans/recommender-features-research.md`,
`pipeline/`, `ch_client.py`, `database.py`, `server.py`, `config.toml`,
and ran read-only SELECTs against the live DB (user 1: 1,235 up / 1,158
down / 1,179 neutral; 677 impressions in the ledger; 25k ch_seed rows).

### 1. Ranking / ML

**1.1 — Fix the recall stage: personalized candidate generation.** All the
ranking effort so far tunes a reranker over HN's winners —
`query_live_window(min_score=5, limit=5000)` ordered by score means the
personalization ceiling is HN's own popularity filter, and stories you'd
love that flopped on HN can never surface. Run a second, wider CH query
(score ≥ 1, or ORDER BY time), embed titles cheaply, and admit only the
top-k by similarity to your positive cluster centers into the pool — a
classic two-stage recsys shape where the recall stage is currently pure
popularity. Risk: low-score stories are often low-score for a reason
(spam, dupes, dead links); embedding cost at 4096-token mxbai is
nontrivial, so it needs a title-only fast path and a strict per-cycle
budget.

**1.2 — The precomputed kernel just became a free model-design surface;
use it.** PERF-3 means you already materialize kernels by hand via
`rbf_kernel` (pipeline/ranking.py:99,110) — so composite kernels are now
~20 lines: separate gammas for the embedding block vs. the scaled-metadata
block, a cosine kernel on embeddings summed with an RBF on metadata, or a
recency-decay multiplier folded into the training kernel. This is a
structurally new model class that stays "an SVM over embeddings +
metadata" and costs nothing at inference. Risk: the hyperparameter surface
grows just as your eval harness shows fold std of ±0.07-0.12, so without
paired significance testing (see 1.7) you'll chase noise.

**1.3 — Distill an "HN hivemind prior" from ClickHouse.** You have free
SQL access to the entire scored HN corpus; train an offline regressor
(embedding → log score) on a few hundred thousand stories, then use its
prediction and the residual (user upvote relative to crowd-predicted
quality) as features. It cleanly separates "generically good" from
"specifically me," which the current features conflate — and the same
model doubles as the admission filter for 1.1. Risk: one large offline
embedding run (hours; use titles or truncated text), score is confounded
by posting-time luck, and it's exactly the kind of feature that can
produce suspiciously high offline metrics (the NDCG > 0.40 skepticism
rule applies).

**1.4 — Randomized exploration slots with logged propensities →
counterfactual eval.** Every badge pass is deterministic, so training
labels and eval are both conditioned on what past models chose to show —
you can never measure what the ranker suppresses. Reserve 1-2 deck slots
sampled with a known probability (Boltzmann over scores), log the
propensity into `interaction_events` (it already carries ranker_arm and
position), and use IPS/SNIPS estimators to score future ranking changes
on live traffic instead of NDCG over biased historical labels. Risk: at
~n=1 vote volume the estimators are high-variance and need weeks of data
per comparison; a couple of genuinely random cards per session is the
visible UX price.

**1.5 — The WORKLOG's SVM-vs-DL verdict is stale on four axes.** Every DL
number (attention MLP, blend_score_75 at +0.055 NDCG@40 over SVM)
predates the temporal split (2026-07-01, which cut headline NDCG@40
roughly in half), the production-legs candidate pool, the 4-binary source
features, and the mxbai embedding switch — and the blend was rejected
partly on a 3s cold-render cost that the PERF-2/3 cadence work has since
made irrelevant. One `--group dl-experiment` re-run under the current
harness would either retire the question properly or resurrect a shelved
+0.05. Risk: torch group is a 700MB install and the answer may still be
"no," but right now the conclusion in the docs isn't supported by the
current methodology.

**1.6 — Disambiguate the neutral class.** 1,179 skips (31% of labels!)
train as a full third class with balanced weight, but the label conflates
"meh," "already read it elsewhere," and "fine but not now" — three
different decision-boundary semantics in one label. Cheap experiments in
the existing variant harness: neutral-as-soft-negative sample weights,
ordinal targets (down < neutral < up), or excluding neutral from fit
while keeping it for dedup exclusion. Risk: binary_margin_no_neutral was
roughly at parity historically, so gains may be small; a two-flavor skip
gesture adds UX friction for a data-quality win you can't validate
retroactively.

**1.7 — Eval gaps: paired tests, slice reports, and deck-level replay as
first-class.** Promotion decisions currently read mean-of-5-folds tables
with no significance test, and the one end-to-end "final queue" eval
(badges + dedup + combos, 2026-06-28) was a one-off rather than a harness
mode — yet the deck, not pool-NDCG@40, is what you actually experience.
Add per-fold paired permutation tests, per-source/per-cluster win/loss
slices, and make full-pipeline deck replay a standing eval mode. Risk:
essentially none, except discovering that some past promote/reject calls
were coin flips — which is the point.

**1.8 — Session structure in the labels.** 2,738 votes landed in June
alone; bursty sessions mean correlated labels (topic runs, mood,
end-of-session fatigue) that inflate effective sample size and leak
across nearby folds. Derive session IDs from vote-time gaps, use
grouped/temporal CV, and consider per-session weight normalization so a
100-swipe binge doesn't outvote ten deliberate sessions. Risk: down-
weighting your highest-volume data; the temporal split may already absorb
most of the benefit.

**1.9 — A learned metric instead of a learned classifier.** Fit a
low-rank linear adapter W (384×64, numpy-only, triplet loss on up/down
pairs) on top of the frozen encoder and feed transformed vectors to the
same SVM — this personalizes the geometry that every similarity feature,
cluster feature, dedup threshold, and badge inherits, which no classifier
swap can do. Risk: ~1.2k positives is thin for metric learning; it
invalidates the meaning of every cosine-threshold knob downstream and
needs its own schema-version style versioning.

**1.10 — Conformal "Unsure" instead of entropy over an uncalibrated
softmax.** The Unsure badge currently ranks by Shannon entropy of
softmaxed margins that the docs themselves say aren't probabilities.
Conformal prediction over held-out/LOOCV margins gives distribution-free
"prediction set = {up, down}" flags in ~50 lines, and doubles as a
principled active-learning selector — the deck is, after all, a labeling
interface. Risk: small felt difference if entropy already correlates
well; needs margins retained on the cache-hit path.

### 2. New features

**2.1 — Emit a personal RSS/Atom feed of the ranked deck.** The system
ingests ~80 feeds and emits none; a token-guarded /feed.xml of the
current top-ranked stories (with cached TLDRs as descriptions) makes the
reranker composable with every reader app, watch, and e-ink device you
own — the most local-first feature possible. Risk: feed reads bypass the
vote loop entirely (no labels), and a token-in-URL feed on the public
funnel is a mild leak surface.

**2.2 — Taste map: named upvote clusters with mute/boost dials.** You
already fit KMeans over positive embeddings every render; lift it to a
UI — cluster your upvotes (k≈12), name each cluster with one cached LLM
call, show volume/recency/hit-rate, and let you mute or boost clusters
(a per-user bias vector folded into scoring). Voting steers per-story;
this steers per-topic, and it makes the opaque SVM legible for the first
time. Risk: cluster identities drift as feedback grows (names must be
re-derived), and mute is blunt enough to cancel the exploration the Novel
badge exists to protect.

**2.3 — Session-local instant adaptation of the queued deck.** With the
10-vote/3s cadence, downvoting three consecutive crypto stories still
shows you the fourth queued crypto card — the client only has
data-score. Ship a coarse cluster-id per card and demote same-cluster
queued cards client-side after ≥2 consecutive downvotes, giving instant
"I get it, you're not into this today" behavior between server reranks.
Risk: a second, simplified ranking policy living in JS will drift from
the server's; too-aggressive thresholds feel twitchy.

**2.4 — "What you missed" recap using the impression ledger.** Now that
impressions are logged, you can compute the set of stories that scored
high but aged out of the 30-day window with zero impressions — a weekly
recap deck that quantifies coverage and rescues buried gems. This is also
the first consumer for `interaction_events`, which currently has no
reader at all. Risk: without a "was in top-X of its cohort" predicate it
degenerates into re-serving the mid-ranked stuff the primary pass
correctly demoted.

**2.5 — Bootstrap import of your real HN account upvotes.** A one-time
scrape of news.ycombinator.com/upvoted with your session cookie could
triple your positive labels with data that predates the deck's own
selection bias — the only way to get exposure-unbiased positives without
waiting for 1.4. Risk: credentialed scraping is brittle and ToS-gray;
imported positives arrive with no matched negatives (class balance
shifts) and may encode stale taste, so they probably need their own
sample-weight class.

**2.6 — Ask-your-corpus (semantic RAG over voted/saved/TLDR'd stories).**
Embeddings for 42k stories already sit in SQLite; brute-force ANN at that
scale is milliseconds, so a /ask endpoint that retrieves top-k personal
stories and synthesizes an answer through the existing LLM plumbing turns
the archive into a memory ("what was that Register piece about sidebar
extraction?"). Risk: puts an LLM dependency in a core retrieval feature
where ROADMAP F1's FTS5 works fully offline — build it on top of F1, not
instead.

### 3. Refactoring / architecture

**3.1 — Materialize ranked decks as data, not HTML bytes.** The warm
path's terminal artifact is rendered HTML in an in-process dict, with
refills byte-slicing between `<!--cards:start-->` sentinels; persisting
(user, version, story_id, rank, score, badges, arm) rows and rendering
cards at read time unlocks deck diffing, the /feed.xml feature, B2
interleaving, and — critically — historical reconstruction of "what was
shown" for any counterfactual analysis. Risk: it reopens the
carefully-debugged version/ready-gating semantics (see the 2026-06-28→30
WORKLOG saga), so it must be done behind the existing contract tests.

**3.2 — Append-only vote journal.** `feedback` is an upsert keyed
(user_id, story_id) and clear deletes the row — vote→undo→revote history
is unrecoverable, and the new interaction ledger deliberately excludes
votes, so the exact join every future analysis needs (shown at position p
under arm a → voted x) doesn't exist. A tiny `feedback_events` insert
alongside every write preserves label provenance and lets you re-derive
labels under different semantics later (e.g., "ever-upvoted" vs
"currently-upvoted"). Risk: dual-write consistency; storage is trivial.

**3.3 — Split "model refresh" from "pool scoring" with a persistent score
cache.** Feedback changes require a re-fit, but regen only changes a few
hundred candidate rows — yet both trigger the full 8k-row SQL +
embedding-lookup + feature-prep + decision pipeline (still ~5-6s warm,
now dominated by everything except the SVM). Persist per-story decision
values keyed on (feedback_signature, text_hash) and re-score only deltas;
incremental scoring through the precomputed kernel is milliseconds,
making most warms nearly free. Risk: minmax01 normalization is
pool-relative and must move to read time, and a two-key invalidation
surface is exactly the kind of cache-coherence trap that has bitten this
codebase before.

**3.4 — One RankingPolicy object executed by both server and eval.**
eval_ranker_variants.py re-implements feature assembly and classifier
config next to the production path; the candidate legs were unified in
July, but the rest is hand-synced (and eval.py's legacy_features drift
already happened once). Extract the policy so offline numbers are
by-construction the shipped code, and B2 interleaving becomes
"instantiate two policies." Risk: eval legitimately needs to vary
internals, so the seams must be chosen well or people will route around
the abstraction.

**3.5 — Embedding registry keyed (story_id, model_version).** The
`embeddings` table has story_id INTEGER PRIMARY KEY — one vector per
story, ever — which made the mxbai switch a big-bang cache swap, makes
production A/B between embedding spaces impossible, and makes rollback an
hours-long recompute at 3.3 stories/s. An additive composite-key
migration lets two spaces coexist during any future transition (the
bakeoff snapshot machinery already proved you need this). Risk: transient
double storage (~60MB per space, fine) and threading model_version
through every lookup site.

### 4. Things you're probably not thinking about

**4.1 — The precomputed SVM has a quadratic memory cliff about a year
out.** PrecomputedRbfSVC.fit materializes the full n×n float64 training
kernel: 3.3k feedback ≈ 90MB today, but at ~1k votes/month that's ~512MB
at 8k rows and ~1.2GB at 12k — on a host that had 2.6GB free and already
peaks at 834MB RSS. Chunk the training kernel the way inference already
is, add a feedback-row tripwire to rank_perf, and plan feedback coresets
(per-class k-center selection, or folding old votes into compact prior
features) before the cliff, not after the OOM. Risk: coresets change
ranking semantics and need the eval harness; doing nothing has a known
failure date.

**4.2 — The loop is now closed and nothing measures the narrowing.**
Labels come only from what past models chose to show, the positive-
cluster feature explicitly rewards proximity to existing interests, and
the exploration budget is ~4 Novel/Unsure cards in a ~40-card deck — the
textbook rich-get-richer setup. Before more model power, add a coverage
report: fraction of candidate-pool embedding clusters ever impressed, and
drift of shown-centroid vs pool-centroid over months, straight from
interaction_events. Risk: none; the current risk is that the narrowing is
invisible by construction.

**4.3 — The HN-only fortnight poisoned the source prior.** Between
2026-07-10 and 2026-07-12 the dashboard was hardcoded HN-only, so zero
non-HN labels accumulated while non-HN stories kept flowing into the DB —
after restoration, the 4-binary source features partly learn
"reddit/rss = never upvoted" because those stories were never shown.
Votes carry no record of the active policy; log a policy epoch with each
vote (via 3.2) and exclude or reweight source-feature training across
policy discontinuities. This generalizes: every UI/cadence/source change
silently shifts the label distribution, and you've made several per
month.

**4.4 — ClickHouse Playground is a single point of failure with
invisible staleness.** The "local-first" system's only live story source
is a free public endpoint with no SLA; on failure, fetch_candidates logs
"live source empty" and the deck silently ages — no staleness surfaced in
the UI, and the Algolia fallback only covers single-story fetches, not
the live window. Add a last-successful-regen indicator to the dashboard
and a degraded-mode live-window fallback (Algolia search loop still
exists in scripts/_archive). Risk: rarely-exercised fallback code rots;
it needs a periodic forced drill.

**4.5 — Label-text drift: the vector you train on isn't the text you
judged.** The self-healing text_hash invalidation means a feedback story
whose comments/article hydrate after your vote gets re-embedded, so the
training pair (embedding, label) drifts from the artifact you actually
evaluated. First measure it (one SQL join: feedback rows whose embedding
hash changed after updated_at); if material, freeze the embedding text at
vote time via the vote journal. Risk: frozen text also forgoes enrichment
that genuinely improves similarity — this is an empirical question, not
an obvious fix.

**4.6 — Anonymous-user proliferation quietly pollutes the "multi-user"
story.** Every drive-by public-demo visit creates a users row (423 users,
10 with any feedback); tables grow unboundedly and any future cross-user
idea (global priors, co-vote signals) would ingest drive-by noise as if
it were users. Idle-user pruning (0 feedback, 0 interactions, >N days) is
simple but touches the DB-safety rule, so it needs explicit sign-off and
a backup-gated script. Risk: deleting a row for someone who bookmarked
their /u/<token> link and returns in month three.

**4.7 — Session fatigue is a label-quality confounder you can now see.**
Late-session votes in a forced-choice swipe deck are systematically
grumpier and faster; with dwell + impression timestamps you can measure
vote-quality decay within sessions and, if real, down-weight
tail-of-session labels. Risk: it's speculative until measured, and it
interacts with 1.8 — do the measurement before either.

### Where the constraints bite

Local-first with n=1 makes every counterfactual estimator (1.4)
data-starved — that's the real cost of no cross-user data, not the
SQLite or SVM constraints, which none of these ideas strain. The
"SVM + embeddings" constraint is actually generous now that the kernel
is hand-materialized (1.2); the place it genuinely limits you is sequence
modeling of sessions, which nothing above requires.

### Critical files

- `pipeline/ranking.py` — kernel surface, features, badges, model cache
  (1.2, 1.9, 1.10, 4.1)
- `pipeline/__init__.py` — candidate legs, fast_rerank_for_user, cold deck
  (1.1, 3.3, 3.4)
- `database.py` — schema for vote journal, embedding registry, deck
  materialization (3.1, 3.2, 3.5)
- `server.py` — warm lifecycle, ledger ingestion, feed/ask endpoints
  (2.1, 2.4, 4.4)
- `scripts/eval_ranker_variants.py` — temporal harness for every
  promotion gate (1.5, 1.7)

---

## Review of Codex findings (verified against live tree + DB, 2026-07-15)

### Confirmed — Codex 4.1 was a live data-loss bug (fixed same-day)

- `server.py:1843` rejected `story_id <= 0`; any single bad event raised
  ValueError → the whole request 400'd (`server.py:1919`) and nothing
  from the batch was inserted.
- Every non-HN source uses negative synthetic IDs: `enrichment.py:656`
  `synthetic_id = -(val % (2**31))`. That's all ~5,920 non-HN stories
  (every rss_*, reddit, tildes, slashdot, lesswrong, digg,
  github_trending row).
- Client queued events for every card with no sign filter
  (`index.html:961`).
- **sendBeacon path** (`index.html:986-989`): batch was spliced out as
  soon as the beacon was *accepted by the browser* — the server's 400
  was never observed. One RSS event silently destroyed every event in
  its batch, including valid HN ones.
- **fetch fallback** (`index.html:997-1004`): on !resp.ok the batch was
  never spliced and `scheduleInteractionFlush` refired at setTimeout(0)
  → poison-pill infinite tight retry loop when sendBeacon was
  unavailable.
- DB evidence: 458 events logged under source_filter='mixed', yet
  **zero** events from any non-HN story existed in the ledger before the
  fix. Mixed decks were browsed; poisoned batches provably died.

Fixed in commit `3a5a77c` (2026-07-15): per-event accept/reject
semantics server-side, `story_id == 0` (not `<= 0`) is the only invalid
ID, client drops (doesn't retry) permanently-rejected batches. See
WORKLOG.md for full detail. **Data-quality note:** the pre-2026-07-15
`mixed`-source ledger is HN-survivor-biased — do not compare analyses
naively across the fix date.

### Other verified claims

- All 1,384 ledger events (pre-fix) had ranker_arm='baseline';
  impressions 325/143/47 at positions 0/1/2 — Codex's intro stats check
  out.
- **4.7 longest-text-wins**: confirmed `database.py:483-505` —
  self_text, top_comments, article_body each keep whichever version is
  longer, forever. Monotonic poisoning; no fix implemented yet.
- **4.9 RSS ID collisions**: mechanism confirmed, 31-bit space; birthday
  50% around ~55-65k rows (5,920 today). Slow-burning; a collision
  silently merges feedback across unrelated stories via the PK.
- **4.3 updated_at**: structurally confirmed — feedback is one row per
  (user, story), revotes move updated_at into the future.
- **4.8 neutral dwell**: directionally confirmed (avg dwell neutral
  73.4s vs up 69.3s vs down 68.7s; n=255/124/214) but the deltas are ~6%
  with no significance test and the join is against *current* vote
  state. Hypothesis, not evidence — see "Weaker items" below.
- Embeddings schema: `story_id INTEGER PRIMARY KEY` (model_version/
  text_hash are invalidation columns, not key parts) — Fable 3.5 and
  Codex arch-2 both stand.

### Codex ∩ Fable consensus (independent convergence — likely the real priorities)

- Append-only feedback ledger (Codex 3.1 ≈ Fable 3.2)
- Ranking/deck snapshots as data, not HTML (Codex 3.3 ≈ Fable 3.1)
- Content/embedding versioning; freeze what was judged (Codex 3.2, 4.4 ≈
  Fable 4.5, 3.5)
- Propensity-logged exploration → off-policy eval (Codex 1.1, 4.2 ≈
  Fable 1.4)
- Slate-level eval with one pre-registered primary metric (Codex 1.7 ≈
  Fable 1.7)
- Retrieval-stage bias dominates the SVM (Codex 4.6 ≈ Fable 1.1)
- Neutral-class semantics (Codex 2.5/2.6/4.8 ≈ Fable 1.6)

### Novel to Codex (not in Fable's list)

- Choice-set pairwise learning from exposed slates (1.6)
- Split article-vs-discussion embeddings with agreement features (1.5)
- Graph propagation over kNN embedding graph as challenger arm (1.4)
- Taste modes / mixture-of-scorers (1.3); temporary lenses (2.3);
  reading missions (2.1)
- Evolving-story threads (2.7)
- Typed CandidateBatch retrieval budgets (3.4); SQLite work scheduler
  (3.5); dependency-driven regen (3.7)

### Weaker items

- SVM committee (1.2): 5× fit cost points straight into the Fable-4.1
  kernel memory cliff; conformal prediction (Fable 1.10) gets similar
  value cheaper.
- 4.8 dwell claim: see above — measure properly (significance test)
  before acting on it.

---

## Status

Nothing in this note beyond the interaction-ledger fix has been
implemented or planned. Starting any item here requires explicit user
direction — this is a backlog reference, not a roadmap commitment.
