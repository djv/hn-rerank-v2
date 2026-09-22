# HN Rerank status

## Objective and scope

Improve eval controls and training deduplication only. User also authorized
committing/deploying the earlier TLDR coverage fix and TUI impression logging.
Full-body embedding work remains deferred; no publication tuning or automatic
ranking rollout. Previous status archived in
`docs/status-archive/2026-09-22-before-ranking-deploy.md`.

## Verified result

- Code commit `f03c34e` pushed to origin/main and deployed to VPS
  `/home/dev/hn-rewrite/main`; service restarted successfully. VPS code tree
  clean. Prior identical TLDR patch preserved in named stash
  `pre-f03c34e-deploy-identical-detail-v12` (not reapplied or dropped).
- Dashboard HTTP 200. Import AI 473 served its cached seven-bullet v12 summary;
  another uncached story generated successfully via Muse Spark in ~13 seconds.
  Neither response was stale/retryable. Bounded journal scans found no errors.
- TUI restarted in `work:1.1` using the same default profile (profile SHA-256
  unchanged). An actual selected-story impression reached the VPS ledger:
  `tui_observed`, Recommended/Recent, position 0. No votes changed by deployment.
  Restart reset the view from Explore to Recommended.
- `publication_affinity_enabled=false` and
  `deduplicate_training_feedback=false` verified from deployed config.
- Exact staged snapshot: 795 backend tests passed with two niced workers.
  Client: 63 passed / 1 Windows-only skip. Ruff/format/ty clean. Initial
  snapshot test attempt lacked Git provenance; after initializing an isolated
  source-only Git snapshot, full backend suite passed.

## Evaluation and remaining uncertainty

- Three development folds, five shuffled-label seeds, isolated single-thread
  VPS evaluation completed. Expected shuffled NDCG@10 0.3402; observed means:
  production 0.3352, publication 0.3545, training dedup 0.3211.
- Dedup real NDCG@10 0.8502 versus production 0.8054; MAP slightly worse
  (0.4664 vs 0.4682). Gain concentrated in one fold. Judged-only replay scores
  are not live-feed quality; this is not sufficient evidence to enable it.
- Detailed evidence and private artifact paths: FINDINGS.md. Latest 20%
  confirmation period remains untested. No eval jobs left running.

## Preserved uncommitted work (excluded from deployment)

- `pipeline/embedding_sections.py`
- `scripts/bakeoff_embedding_models.py`
- `tests/test_embedding_sections.py`

## Next step / blockers

If ranking work resumes, predeclare a production-vs-dedup confirmation test;
do not tune against it or enable ranking flags without reviewing results.
No deployment blocker. Separate older project blocker: PyPI publishing
credentials still needed. No further work scheduled in this session.
