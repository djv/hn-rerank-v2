# HN Rerank status

Saved 2026-10-02 15:25 UTC. Previous status:
[docs/status-archive/before-gemma-handoff-20261002.md](docs/status-archive/before-gemma-handoff-20261002.md).

## Objective
Improve what the dashboard shows the live profile 151 (user 1 stopped
2026-09-24): ranking quality and which sources feed it. This round: ranking
evals on fresh votes and impressions, then gemma side-by-side ranking, now
live. Open user decisions: personalized Popular order; reader mockup concepts.

## Verified result
- Gemma side by side live on the VPS (`5b4b939` code, `2639f8f` flag on,
  docs `16ba800`; restart 2026-10-02 13:32:34 UTC). Backfill 17,053 stories at
  0.218 s/story niced; `hn-rewrite-side-embed.timer` every 30 min (runs 13:35,
  14:05, 14:35, 15:05 OK; units in system-setup `f31d149`). Dashboard/feed 200,
  no journal errors through 15:21 UTC. One-off rerank of 151 on a live-DB
  copy: `side_embeddings=on`, 6.9 s refit / 5.3 s cache hit (before: 5.4-8.7 s
  / 3.8 s); Recommended top 12 changed 1/12 (1d) and 5/12 (1w). CI green on
  `5b4b939` and `2639f8f`. Tests 1,061 passed; Ruff/format clean; ty only the
  untracked `scripts/inspect_tldr_failures.py`.
- Offline (snapshot 2026-10-02, 394 unseen votes since the blend): live blend
  beats legacy (AUC 0.795 -> 0.865, P@12 0.375 -> 0.500). Knob tuning has
  plateaued (12 blocks, all within ±0.01 AUC). Gemma at 128 tokens: AUC +0.011,
  P@12 +0.056 (about +1 liked / -1 disliked card per 1-2 screens offline;
  likely smaller live). Popular (unpersonalized) is 2.4% upvoted live; the
  model's order would cut its top-12 downvotes ~54% -> 31% (70/30 with gravity:
  40%). Evidence: FINDINGS.md "Fresh-vote, impression-pool and live-yield evals".
- Earlier, still current: TLDR diagnosis saved, fixes parked (FINDINGS.md
  "TLDR follow-up diagnosis"); review fixes deployed `4d1ff85`; reader mockup
  ready (`hn-reader-mockup.service` active, http://127.0.0.1:8766/bloomberg-reader.html).

## Blocker / limits
- No live 151 rerank seen since the gemma restart (user idle through 15:21
  UTC), so the live label/latency check is pending; real-user gain unmeasured.
- Offline gains rest on re-ranking already-voted stories (57 fresh upvotes);
  Popular's upvote rates rest on 8 upvotes (downvote result on 95).
- Uncommitted WIP from earlier sessions, kept out of these commits:
  `clients/tui/tests/test_client.py` (status-line assertion), `docs/mockups/`,
  `scripts/inspect_tldr_failures.py`, `kernel.errors.txt` (OpenVINO dump).
- TLDR discussion call still fails intermittently (article-only salvage).
- LLM: Go limit resets 2026-10-06 16:28 UTC; then set `LLM_PROVIDER=gospark`
  (now `mistral`, $10 cap) in VPS `shared/.env` and restart.

## Next step
- Gemma: confirm `side_embeddings=on` and ~5-9 s `rank_total_ms` in a live
  `rank_perf ... user_id=151` line; after ~200 shown stories re-read yields
  with `scripts/badge_yield_report.py --db <snapshot> --user-id 151` and rerun
  `~/.local/state/hn-rerank-eval/run_eval.sh` on a fresh snapshot. Rollback:
  `side_embedding_enabled = false` in config.toml, deploy, restart.
- Ask the user about Popular: re-order its gravity candidates 70/30
  model/gravity (or model-only), then live-check its up/down rates.
- Explore badges: around 2026-10-14 decide Unsure/Novel/Interest from
  `badge_yield_report.py` (so far Unsure 0/19, Novel 0/17, Interest 2/23).
- Reader mockup: get the user's concept picks; stop the mockup service when
  review ends.
- Parked (resume only on request): TLDR fixes (bullet retry, raw PDF guard,
  archive dupes); title-based "Because you upvoted"; embedding-model
  hill-climb (memory pressure); AINews/tweet review items.
