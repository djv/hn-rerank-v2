# hn-rewrite: prioritized improvement recommendations

## Context

Advisory review of the reranking dashboard, grounded in the current tree (including the uncommitted `hn_dupes` WIP). Top-3 per category, ranked by payoff-to-churn. Effort: **S** < ½ day, **M** 1–2 days, **L** 3+ days.

## Two corrections to your priors first

1. **`decision_function` is not on the GET / path.** `GET /` serves cached HTML bytes (SWR, `Handler._dashboard_cache`, server.py:941–1015). The 4–5s SVM cost lives in the background warm (`_run_warm_attempt` → `fast_rerank_for_user`, server.py:1113/1146). But every vote invalidates the cache (server.py:1563) and the client's ready-gated refill *waits* on that warm (`runWarmPollLoop`, index.html:1290). So the real user-facing metric is **vote → refill latency**, and *policy* (how often you rerank) matters as much as raw compute. This reframes perf item #3 below.
2. **Pain point #6 is stale.** `/api/deck-cards`, warm-refill, and undo all have coverage: `test_deck_cards_*` (test_server.py:4008, 4017), `test_ready_gated_refill_*` (:4248–4287), warm coalescing (:1344–1510), undo (:4120, :4214). The genuinely uncovered surfaces are the CORS/OPTIONS handlers (server.py:2045–2052) and the nonexistent health endpoint. Drop #6 from the roadmap.

---

## 1. Performance (target: the ~6.5s warm)

The `rank_perf` trace (server.py:1193) already partitions stage times — pull a week of journalctl before touching anything, but the structural wins are clear:

### P1. Precomputed-kernel SVM scoring — exact, ~10× on the dominant stage (M)
`SVC.decision_function` (ranking.py:1049/1051) is libsvm's single-threaded per-SV loop. With C=0.1 on ~2500 rows, the SV count is likely >1500, so scoring is ~8000 × 1500+ kernel evals in a C loop. Switch to `SVC(kernel="precomputed")`:
- fit on `K_train = rbf(X_fb, X_fb)` (2500² — trivial),
- score with `decision_function(rbf(X_cand, X_fb))` where the kernel matrix is one BLAS matmul via `‖x−y‖² = ‖x‖²+‖y‖²−2x·y`.
- **Bonus**: the `X_cand @ X_fb.T` matmul is the *same* product the KNN pass already computes (`_knn_mean_and_max`, ranking.py:616) on the embedding block — share it and pain point #2 (the 24M dot products) is amortized into the kernel computation instead of being a second pass.
- Identical model, verifiable with `np.allclose` against the current path on the real DB, plus NDCG parity in `scripts/eval_ranker_variants.py`. Bump `_MODEL_SCHEMA_VERSION` (ranking.py:58) since cached model objects change shape.
- Fallback if you want more: `Nystroem(gamma=0.03, n_components≈300)` + `LinearSVC` — approximate but O(candidates × 300). Only reach for it if precomputed-kernel isn't enough; it needs eval-harness sign-off, the exact rewrite doesn't.

### P2. Rerank cadence: stop paying full warm per vote (S–M)
A 30-swipe session currently triggers ~30 full reranks (1s debounce, server.py:933). One vote among 2500 feedback rows barely moves the ranking. Policy change in `_handle_flask_feedback` / `_trigger_warm`:
- Serve refills immediately from the *stale* ranking minus voted cards (the client already suppresses `votedStoryIds`, index.html:1413) and only ready-gate every Nth vote (e.g. N=10) or after T seconds idle.
- Effect: vote→refill goes from ~6.5s to ~instant for 90% of votes, with zero model work. Biggest UX win per line of code in this list. Testing: extend the existing `test_ready_gated_refill_*` / warm-coalescing tests — the harness is already built for this.

### P3. Cache the candidate matrix per regen cycle (S)
Every warm re-runs `load_production_candidate_stories` (pipeline/__init__.py:470) and `get_or_compute_embeddings` (:482) against the 541MB DB — pure waste between regens, since candidates only change 4-hourly. Hold `(stories, embedding_matrix)` in-process keyed by regen generation; invalidate in `regen_loop`. Likely saves 1–2s per warm. Risk: memory (~8000×384 float32 ≈ 12MB — nothing).

**Also (flagging, not top-3):** the uncommitted `hn_dupes` stage puts Firebase HTTP calls on the warm path (pipeline/__init__.py:506–512). It's bounded (8 workers, 1.5s timeout, TTL cache) but network-dependent — consider resolving at regen/prewarm into a persisted canonical map, keeping warm-time as a dict lookup.

---

## 2. Refactoring

### R1. Evict non-server concerns from server.py (M)
server.py is three modules in a trenchcoat. Two mechanical extractions, each safe standalone:
- **Article extraction → `pipeline/enrichment.py`**: `_extract_article_body` (server.py:286) + its 4 extractors + `_normalize_article_text`/`_looks_bad_extraction` + `ARTICLE_BODY_CHAR_LIMIT` are pure (HTML → text), zero Flask/Handler deps. Update 3 import sites: `scripts/fetch_articles_for_source.py:18,92`, `tests/test_server.py:4391,4408`, and the internal caller. Fixes your known layering violation; enrichment.py already owns `fetch_and_cache_article_bodies`.
- **TLDR/LLM machinery → `pipeline/tldr.py`** (or top-level `tldr.py`): `TldrResult`, `_tldr_cache_key` (:635), `_prefetch_tldrs_for_ranked` (:806), the guts of `_handle_flask_tldr_detail` (:1606). ~600 lines out of server.py.
Intermediate state after each move: re-export from server.py so nothing breaks, then update imports, then drop the re-export. server.py lands around 1000 lines of actual HTTP concern.

### R2. Handler classmethod-state → instance injected into `create_app` (M)
All Handler state is class-level (server.py:915–934), shared process-wide — which is why conftest needs autouse singleton-reset fixtures and why parallel test isolation is fragile. Convert to a plain instance (`runtime = Handler(config, db, embedder, regen_event)`), pass it to `create_app(runtime)` (it already closes over `runtime`), keep a module-level instance during transition so `test_server.py`'s 139 tests migrate incrementally. Payoff: real test isolation, and it unlocks running two app instances in one process (A/B eval, see blind spot B2). Churn is wide but shallow — mostly `cls.` → `self.`.

### R3. Extract the ~945-line inline `<script>` from index.html to `static/deck.js` (S–M)
52% of the only template is JS (index.html:864–1809). Serving it as a static file with cache headers cuts every dashboard render's payload, ends the template-string-test brittleness for JS internals, and opens the door to JS unit tests later. Keep the small bootstrap inline. Testing: `tests/test_server.py` template assertions need a one-time repointing — do it as its own commit, no behavior change.

(ranking.py at 1543 lines could split into features/model/blend, but it's cohesive and hot — I'd leave it until after P1 lands.)

---

## 3. New features / UX

### F1. Personal archive with FTS5 search + explicit "save" action (M)
Right now upvote conflates "good signal" with "want to keep". Add a third action (save/bookmark), a reading-list view, and SQLite **FTS5** over title + self_text + article_body + cached TLDRs. Stdlib, local-first, no deps. The killer daily-driver query is "I saw something about X three weeks ago" — you already store the corpus; you just can't search it. Schema: one FTS5 virtual table synced by trigger from `stories` + `tldr_cache`. Testing straightforward (DB-level).

### F2. "Because you upvoted …" attribution on cards (S–M)
The rank pass already computes each candidate's nearest upvoted feedback story (`cand_closest_up`, ranking.py:868–880, reused via `RankScoreContext`). Surface the argmax neighbor's title as a one-line card annotation. Cheap (the data exists at rank time; thread the story-id through `DashboardCardView`), builds trust in the ranker, and makes bad recommendations *diagnosable* — when the deck goes weird you'll see exactly which old upvote is dragging it. Natural follow-on: a "less like this" action that downweights that neighbor.

### F3. Explore/exploit dial (S)
You already have 7 discovery passes with badges (`rerank_candidates`, ranking.py:1219+). Expose one user-facing control (3 positions: focused / balanced / adventurous) that scales the discovery-pass quotas and the novelty weight. Serendipity is the thing personalized rankers kill slowly; you built the machinery, it's just not steerable. Server-side it's a per-user int in `users`; the deck already rebuilds per version bump.

---

## 4. Infrastructure / ops

### O1. `/healthz` + systemd hardening (S)
No health endpoint exists. Add one returning `{last_regen_age_s, cache_entries, db_ok}` (cheap `SELECT 1` through the pool), then in the unit: `Restart=on-failure`, plus a `curl --fail`-based watchdog timer or `systemd`'s `WatchdogSec` with sdnotify. Also covers the currently-untested OPTIONS/CORS surface while you're in there. On a $5 VPS with a daemon-thread regen loop, silent regen-thread death is your most likely unnoticed failure mode.

### O2. Persist `rank_perf` traces to a metrics table (S)
You have excellent per-stage timing (`trace.format_log_fields()`) that evaporates into journalctl. Append each warm's stage breakdown to a `rank_perf` SQLite table (timestamp, user, stage, ms) and add `scripts/perf_report.py` (p50/p95 per stage, weekly trend). This is the before/after instrument for every P-item above — land it *first*.

### O3. Backup verification + local snapshot hygiene (S)
`backup_hn_db.sh` verifies the remote checksum but never runs `PRAGMA integrity_check` on the snapshot or exercises restore. Add integrity_check to the script, and do one manual restore drill from Drive. Separately: the repo dir holds 5+ timestamped 460–520MB DB snapshots (~2.5GB+) — on a $5 VPS that's disk pressure with no retention rule. Define one (keep newest 2 local; Drive holds 30) — **deletion needs your explicit sign-off per the DB-safety rule**, so this plan only proposes it.

---

## 5. Blind spots

### B1. Temporal drift: your 2500 feedback rows are weighted as if interests never change
Every feedback row from 2024-vintage you counts the same as yesterday's. Add exponential time-decay to `sample_weight` in `_score_and_rank` (ranking.py:996–1003) — half-life ~6 months, one config knob. Measurable today: `eval_ranker_variants.py` is already time-split, so decay is a one-flag ablation. This is the highest-leverage *model* change available that isn't deep learning.

### B2. You have offline eval but zero online eval
NDCG on historical splits ≠ what you actually upvote. **Team-draft interleaving** is built for n=1: interleave decks from ranker A and B, record which variant's cards win your votes, read the sign after a few hundred swipes. It's ~100 lines given R2 (two ranker configs, one deck), turns every future ranking tweak (P1 parity check, B1 decay, F3 dial) into a measured decision instead of vibes, and dovetails with your existing "be skeptical of high metrics" rule.

### B3. You discard implicit signals you already generate
Dwell time per card, TLDR expansions, discussion-link clicks all happen client-side and vanish. Log them now (one `events` table, one beacon endpoint, ~50 lines) even with no consumer: in six months you'll have a graded-relevance corpus for free — as `sample_weight` modifiers or eval labels, no DL required. The cost of *not* logging is unrecoverable data.

---

## Suggested order (if implementing)

1. **O2** (metrics table) → baseline numbers
2. **P2** (rerank cadence) → biggest felt win, smallest diff
3. **P1** (precomputed kernel) + **P3** (candidate matrix cache) → warm under ~1.5s
4. **R1** → **R2** (server decomposition, then Handler instance)
5. **O1**, **O3**, then features **F1–F3** / **B1–B3** by appetite

## Verification (applies to whichever items proceed)

- Standard protocol: `uv run pytest tests/ -n 4`, `ruff check .`, `ty check`.
- P1: `np.allclose` old-vs-new decision values on the live DB (read-only script under scripts/), NDCG parity via `eval_ranker_variants.py`, then `rank_perf` before/after from O2's table.
- P2: extend `test_ready_gated_refill_*` and warm-coalescing tests in test_server.py; live smoke: swipe 10 cards, confirm instant refills + one warm.
- R1/R2/R3: behavior-preserving — full suite green per commit, no test semantics changes except import paths / fixture wiring.
- Service changes: restart `hn_rewrite.service`, journalctl scan per the repo protocol.
