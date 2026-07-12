# hn-rewrite: improvement roadmap

## Context

This is a personal, single-user, local-first Hacker News reranking dashboard.
`GET /` is SWR-cached (`Handler._dashboard_cache`, server.py:920) — the
user-facing latency that matters is **vote → warm completion → ready-gated
refill**, not page-render time. Deployment shape is SQLite + one systemd
service; do not reach for Postgres, Redis, a message broker, FAISS, or a deep
model to fix these problems — the bottlenecks are the current warm-path work
and its lifecycle, not the storage/serving layer.

This file merges two earlier advisory reviews — `fable_plan.md` (rich per-item
technical detail, file:line references) and `codex_ultra_plan.md` (a later
reconciliation against the live tree, with corrected priorities and two items
fable lacked) — into one canonical roadmap, deduped and re-numbered. Merged
2026-07-10; both source files are deleted.

---

## Status ledger

- ✅ **Backups repaired + restore drill** — 2026-07-10. Fixed
  `hn-rewrite-backup.service`'s stale `WorkingDirectory`/`ExecStart` (pointed
  at an old checkout), ran the corrected unit, and completed a non-destructive
  restore drill (checksum + `PRAGMA integrity_check` + row-count comparison,
  all clean). See WORKLOG.md 2026-07-10.
- ✅ **`rank_perf` metrics table + `scripts/perf_report.py`** — `470f787`,
  2026-07-09. Every warm persists a typed `RankPerfSample` row
  (`database.py`'s `insert_rank_perf`, wired into `server.py`'s
  `_run_warm_attempt`); `perf_report.py` prints p50/p95/max per stage, split
  by `model_cache`. This is the before/after instrument for every `PERF-*`
  item below.
- ⛔ **`/healthz`** — closed without code, 2026-07-10. Redundant for this
  deployment shape: DB reachability, last regen, warm failures, and the
  Reddit circuit are all already in journalctl or directly queryable in the
  DB; a live status endpoint only pays off with an external poller, and none
  exists here. See WORKLOG.md 2026-07-10.

---

## 1. Performance (target: the multi-second warm)

The `rank_perf` trace (now persisted, see status ledger) already partitions
stage times — pull a window via `perf_report.py` before touching anything.
Recent live warm traces: ~8.6k candidates, ~3.2k feedback rows, runs of
10-29s. One 28.95s run: 14.9s candidate embedding, 5.0s SVM inference, 2.9s
candidate SQL, 3.0s HN duplicate resolution. KNN feature prep was ~0.17s on
cache hits — leave KNN/FAISS alone, it's already cheap.

### PERF-1. Resolve HN duplicate canonicalization off the warm path (M, 1-2 days)

**Completed 2026-07-10.** Canonicalization is now a SQLite-only warm-path
lookup. A coalescing daemon submitted directly after HN candidate fetches
persists bounded Firebase resolutions (including retry/negative TTLs), so an
unknown or failed lookup leaves the original card visible without delaying a
deck.

**First actual warm-path fix.** `canonicalize_hn_dupes` (pipeline/__init__.py,
called under `trace.stage("hn_dupes")`) does Firebase HTTP lookups
synchronously after ranking — bounded (8 workers, 1.5s timeout, TTL cache) but
network-dependent, and `hn_dupes_ms` has been observed from ~7ms to 12.3s.
Resolve canonical IDs and negative results during regen or a low-priority
worker, persist them with TTL/checked metadata, and make the warm path a
cache-only dict lookup. On unknown/error, retain the original card; an
optional bounded best-effort fallback must never block the full warm.

Test: warm cache hits make no network calls, plus TTL expiry, error fallback,
replacement/suppression behavior, and p50/p95 `hn_dupes_ms` before/after via
`perf_report.py`.

### PERF-2. Stop paying a full warm per vote (S-M)

**Completed 2026-07-11 (`748d3c8`).** Feedback now invalidates the dashboard
immediately and refills from the stale cached deck while the client suppresses
voted story IDs. Per-user warm work is coalesced until either the configurable
vote threshold is reached (default 10) or the configurable idle timer fires
(default 3 seconds). The implementation preserves version/ready-gating
semantics and is covered by threshold, idle, stale-refill, and warm-coalescing
tests.

**Biggest felt win, smallest diff.** Before this change, a 30-swipe session
triggered ~30 full reranks (1s debounce, server.py `_WARM_DEBOUNCE_S`). One
vote among thousands of feedback rows barely moves the ranking, and the client
already suppresses `votedStoryIds` locally. Policy change in
`_handle_flask_feedback` / `_trigger_warm`: serve refills immediately from the
*stale* ranking minus voted cards, and only ready-gate a real rerank every Nth
vote (e.g. N=10) or after T seconds idle. Effect: vote→refill goes from several
seconds to ~instant for most votes, with zero model work.

Keep it configurable (a knob, not a hard-coded "every ten votes") and test
freshness/version semantics rather than assuming a fixed cadence. Extend the
existing `test_ready_gated_refill_*` / warm-coalescing tests
(test_server.py) — the harness is already built for this. Live smoke: swipe
10 cards, confirm instant refills + one real warm.

### PERF-3. Precomputed-kernel SVM inference (M-L, 2-4 days)

**Completed 2026-07-12.** Production now uses the exact precomputed RBF model
with candidate inference chunked at 512 rows and the original libsvm path
retained behind a config fallback. The live production-shaped cold benchmark
reduced decision inference from 5.83s to 0.58s and total ranking from 12.86s to
6.58s; warm runs measured 0.56-0.58s decision and about 5.1s total. Peak RSS
rose from 730MiB to 834MiB on a host with 2.6GiB available. Exact top-40 parity
means temporal ranking metrics are unchanged by construction.

**Read-only benchmark completed 2026-07-12.** On the live user-1 shape (7,910
candidates, 3,347 training rows), current libsvm inference took 5.78s versus
445ms to construct the candidate kernel plus 301ms for precomputed inference.
The top-40 ordering was identical and maximum absolute decision drift was
`3.12e-7`. Explicit kernel arrays occupied about 101MiB (candidate) and 43MiB
(training), so the production design must bound peak memory. Reproduce with
`scripts/benchmark_precomputed_svm.py`; it opens SQLite read-only and refuses
to run if required embeddings are absent.

**Stable bottleneck, but the highest-effort/least-certain item — do after
PERF-1/PERF-2, not before.** `SVC.decision_function` is consistently ~5s:
libsvm's single-threaded per-SV loop, likely >1500 SVs at C=0.1 on ~2500
training rows. Switch to `SVC(kernel="precomputed")`:

- fit on `K_train = rbf(X_fb, X_fb)` (2500² — trivial),
- score with `decision_function(rbf(X_cand, X_fb))`, where the kernel matrix
  is one BLAS matmul via `‖x−y‖² = ‖x‖²+‖y‖²−2x·y`.
- The `X_cand @ X_fb.T` matmul is the same product the KNN pass already
  computes on the embedding block (`_knn_mean_and_max`) — share it. **Caveat**
  (do not oversell this): the SVM also has scaled metadata columns beyond the
  embedding, so the shared matmul only covers the embedding portion of the
  kernel — it is not a fully free shared matrix.
- Identical model, verifiable with `np.allclose` against the current path on
  the real DB (read-only script under `scripts/`), plus NDCG parity via
  `scripts/eval_ranker_variants.py`. Bump `_MODEL_SCHEMA_VERSION` since cached
  model objects change shape.
- Keep the existing libsvm path behind a config fallback until decision/order
  parity is proven.
- Fallback if more speed is needed later: `Nystroem` + `LinearSVC`
  (approximate, O(candidates × components)) — only reach for it if the exact
  precomputed-kernel path isn't enough; it needs eval-harness sign-off, the
  exact rewrite doesn't.

Promotion gates: near-bitwise decision comparison, exact top-40
ordering/tie behavior, temporal-evaluation parity, full-suite verification,
peak-memory sampling. Do not materialize a full candidate-by-training kernel
while the service is memory constrained.

### PERF-4. Cache the candidate matrix per regen cycle (S, demoted)

**Demote until instrumented** — a production-shaped read found only two
embedding misses, so the occasional 15s `candidate_embedding` tail is likely
concurrency/content-change related, not "recompute from scratch every warm."
Add finer substage tracing first (cache hit/miss count, SQLite read, hashing,
ONNX compute, competing regen/background work) before retaining full
candidate matrices in memory. Every warm currently re-runs
`load_production_candidate_stories` and `get_or_compute_embeddings` against
the DB; candidates only change 4-hourly, so holding `(stories,
embedding_matrix)` in-process keyed by regen generation is plausible
(~8000×384 float32 ≈ 12MB, not a memory concern), but confirm the tail is
actually recompute-bound before adding the cache.

---

## 2. Refactoring

### REF-1. Make enrichment pipeline-owned (M, 2-3 days)

**Highest refactor payoff.** Create `pipeline/article_fetch.py` for article
extraction, HTTP retry and fallback, `ArticleFetchResult`, article-failure
backoff, persistence, and the one embedding-refresh path. Broader and safer
than moving only `_extract_article_body`: `pipeline/enrichment.py` currently
reverse-imports `server.py` via late imports (`_fetch_reddit_rss_context`,
`_extract_lesswrong_post_id`/`_fetch_lesswrong_context`,
`ARTICLE_BODY_CHAR_LIMIT`/`_fetch_article_body_with_result`), and article
persistence/backoff is duplicated between batch enrichment and TLDR-on-demand.
Put Reddit and LessWrong source-context adapters in sibling pipeline modules.

Pure extraction targets: `_extract_article_body` + its 4 extractors,
`_normalize_article_text`/`_looks_bad_extraction`, `ARTICLE_BODY_CHAR_LIMIT`
— zero Flask/Handler deps. Known import sites to update:
`scripts/fetch_articles_for_source.py`, `tests/test_server.py`.

Safe sequence: (1) move pure extraction helpers with temporary server
re-exports; (2) move fetch/result handling and batch callers; (3) replace the
on-demand duplicate persistence path with the shared helper; (4) migrate
script/test imports, then remove compatibility aliases.

Acceptance tests: extractor ordering, content-type guards, 429/503 retry,
urllib fallback, permanent/backoff policy, exactly one embedding update per
changed story.

### REF-2. Replace class-global `Handler` state with an injected runtime (M, 2-3 days)

**Highest correctness/testability payoff.** All `Handler` state is
class-level (server.py:915-934), shared process-wide — why conftest needs
autouse singleton-reset fixtures and why parallel test isolation is fragile.
Introduce an `AppRuntime`/instance (`config`, `db`, `embedder`, regen event,
public-demo limiter) and a `DashboardService` owning versions, render locks,
warm timers, cold deck, and HTML cache. Pass it to `create_app(runtime)` (it
already closes over `runtime`); keep a module-level instance during
transition so `test_server.py`'s tests migrate incrementally.

Use a compatibility adapter first; migrate `main()` and test fixtures next;
then change classmethods to instance methods without changing debounce or
cache semantics. Add explicit runtime shutdown/drain support for tests.
Payoff: real test isolation, and it unlocks running two app instances in one
process later (see B2).

Acceptance tests: cache hit/stale hit, rapid-vote coalescing, newer-version
during warm, timer cleanup, no cross-test state leakage.

### REF-3. Extract the TLDR service, then the inline deck script (M + S-M)

**High priority; do the service extraction before the cosmetic JS move.**
First: a framework-free `TldrDetailService` returning typed outcomes. Flask
stays responsible only for JSON parsing, origin/session checks, and response
mapping — move cache-key construction, source hydration, LLM generation,
fallback behavior, and persistence behind the service. Don't start with
Blueprints; they'd just relocate the same tangled handler.

Then, as a fourth refactor once that boundary exists: extract the ~945-line
inline `<script>` in `index.html` (52% of the only template) to
`static/deck.js`. Cuts every dashboard render's payload, ends the
template-string-test brittleness for JS internals, opens the door to real
JS/browser unit tests, and gives `tests/test_server.py`'s template assertions
a one-time repointing (do as its own commit, no behavior change).

Acceptance tests (service extraction): preserve cache-hit-before-quota
ordering, active-HN refresh, stale-TLDR fallback, and the rule that
`no_content` is retryable rather than persisted.

---

## 3. New features / UX

### F1. Personal archive: true save/read-later + SQLite FTS5 search (M, 1-2 days)

**Highest daily-driver feature.** Right now upvote conflates "good signal"
with "want to keep." Add a `saved_items` table independent of `feedback`, a
save button + keyboard shortcut, and a `/library` view backed by SQLite FTS5
over title, self_text, article_body, and cached TLDRs. A save must **never**
alter the SVM training label — that's the ranking-isolation invariant to
test. The killer daily-driver query is "I saw something about X three weeks
ago"; the corpus already exists, it just isn't searchable. Stdlib, local-first,
no new deps: one FTS5 virtual table synced by trigger from `stories` +
`tldr_cache`.

Test: migration, FTS sync after story/TLDR writes, result quality on a
temporary DB, and ranking isolation.

### F2. "Because you upvoted …" attribution on cards (S-M, 0.5-1 day)

**High trust/debug value.** The rank pass already computes each candidate's
nearest upvoted feedback story (`cand_closest_up`, reused via
`RankScoreContext`). Populate `RankedStory.best_match_title` from that
existing computation and render a compact, collapsible "Because you upvoted
…" line plus existing provenance badges. Do **not** present a fabricated
calibrated probability. Cheap — the data exists at rank time, no new
full-pool similarity pass. Builds trust and makes bad recommendations
diagnosable: when the deck goes weird, you'll see exactly which old upvote is
dragging it. Natural follow-on: a "less like this" action that downweights
that neighbor.

Test: cold users, deleted feedback stories, HTML escaping, no new similarity
pass added.

### F3. Explore/exploit dial (S, ~1 day)

**High priority.** 7 discovery passes with badges already exist
(`rerank_candidates`). Expose one user-facing control — `focused` / `balanced`
/ `adventurous` — persisted per profile, that changes server-side discovery-
slot allocation: focused favors primary relevance, balanced preserves current
behavior, adventurous reserves more novel/uncertain/similar cards. Distinct
from the existing client-side Recommended/Popular/Explore filters. Server-side
it's a per-user field; the deck already rebuilds per version bump.

Test: default-policy parity, per-combo/source quotas, no duplicate cards,
profile persistence through `/u/<token>`.

---

## 4. Infrastructure / ops (remaining)

### OPS-1. Decouple Reddit from core regeneration (M, 1-2 days)

**Completed 2026-07-12.** Core ClickHouse/HN and ordinary RSS regeneration
now publishes without waiting for Reddit. A coalescing Reddit worker refreshes
topfeeds and comments asynchronously, persists per-feed snapshots/retry state
and the global circuit cooldown in SQLite, and invalidates the deck once per
changed batch. Recent rows from all currently configured non-HN feeds are
again eligible for the mixed deck; removed and historical feed rows are not.

Test with a fake queue/clock: HN regeneration succeeds on schedule when
Reddit is circuit-open, stale Reddit data remains visible with a clear
freshness state, partial hydration cannot stall core publication.

(The other original ops items — backup repair, `/healthz`, `rank_perf`
persistence — are resolved; see the status ledger above. Local DB-snapshot
retention/hygiene in the repo dir is a separate small housekeeping item if
disk pressure ever becomes real — deletion of any DB copy needs explicit
sign-off per the DB-safety rule.)

---

## 5. Blind spot: the measurement and learning loop

### B1. Test preference drift explicitly

The ranker receives feedback timestamps but doesn't use them — a 2024-vintage
upvote counts the same as yesterday's. Add a configuration-gated exponential
time-decay factor to existing sample weights (half-life ~6 months, one
config knob) in `_score_and_rank`. Promote only if time-split evaluation
(`eval_ranker_variants.py` is already time-split, so this is a one-flag
ablation) improves without erasing stable long-term interests. Highest-
leverage *model* change available that isn't deep learning.

Test class gates, cache invalidation, source-level time-split metrics.

### B2. Add online comparison before adopting model changes

NDCG on historical splits ≠ what you actually upvote. Use deterministic
team-draft interleaving for a baseline vs. experimental deck (built for n=1):
interleave decks from ranker A and B, record which variant's cards win votes,
read the sign after a few hundred swipes. Store arm, position, impression,
and eventual vote; prevent duplicate exposure and attribute undo correctly.
~100 lines given REF-2 (two ranker configs, one deck). Turns every future
ranking tweak (PERF-3 parity check, B1 decay, F3 dial) into a measured
decision instead of vibes.

### B3. Build an impression ledger; don't discard implicit signals

Dwell time per card, TLDR expansions, discussion-link clicks, saves all
happen client-side and vanish today. Record deck version, rank, source,
ranker arm, card exposure, article/comment open, save, dwell, and eventual
vote in a local event table (one `events` table, one beacon endpoint, ~50
lines) even with no consumer yet. Do **not** use automatic TLDR-open as a
signal — it's not deliberate intent in the current client. This makes
position bias visible and gives future ranking changes a trustworthy
denominator; in six months you'll have a graded-relevance corpus for free
(sample_weight modifiers or eval labels), no DL required. The cost of not
logging is unrecoverable data.

Test event idempotency and session/card association.

---

## Suggested order

1. ~~O2 (metrics table)~~ — done, baseline available via `perf_report.py`.
2. ~~O1 (backup repair + drill)~~ — done.
3. ~~**PERF-1** — make HN duplicate resolution local-only during warm
   ranking~~ — done.
4. ~~**PERF-2** — rerank cadence / stale-deck refill policy~~ — done.
5. ~~**PERF-3** — precomputed kernel SVM~~ — done. **PERF-4** remains
   conditional on finer candidate-embedding tracing.
6. ~~**OPS-1** — isolate Reddit from core regeneration~~ — done.
7. **REF-1 → REF-2 → REF-3**, then **F1-F3** and **B1-B3** by appetite.

## Verification (applies to whichever items proceed)

```bash
uv run pytest tests/ -n 4
uv run ruff check .
uv run ty check
```

- PERF-3: `np.allclose` old-vs-new decision values on the live DB (read-only
  script under `scripts/`), NDCG parity via `eval_ranker_variants.py`, then
  `rank_perf`/`perf_report.py` before/after.
- PERF-2: extend `test_ready_gated_refill_*` and warm-coalescing tests in
  `test_server.py`; live smoke: swipe 10 cards, confirm instant refills + one
  real warm.
- REF-1/REF-2/REF-3: behavior-preserving — full suite green per commit, no
  test-semantics changes except import paths / fixture wiring.
- Service changes: restart `hn_rewrite.service`, exercise relevant endpoints
  with a persistent session, scan bounded recent journalctl.
