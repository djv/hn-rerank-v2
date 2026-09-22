# HN Rerank status

## Objective and scope

Improve eval controls and training deduplication only. User also authorized
committing/deploying the earlier TLDR coverage fix and TUI impression logging.
Full-body embedding work remains deferred; no publication tuning or automatic
ranking rollout. Previous status archived in
`docs/status-archive/2026-09-22-before-ranking-deploy.md`.

## Verified result

- Latest data maintenance: user authorized removing three older duplicate
  Import AI votes (issues 458/459/460). Exactly three feedback rows removed;
  newer votes and all story rows preserved. 13 Jack Clark upvotes remain.
  Full private backup/row manifest retained (path in FINDINGS.md). Service
  restarted, dashboard 200, bounded logs clean. Experiment flags unchanged.

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
- Reserved confirmation completed with unchanged config/boundary: 995 rated
  stories, three time-forward folds. Production/dedup NDCG@10 0.7603/0.7138,
  NDCG@40 0.5658/0.5443, MAP 0.4992/0.4956. Top-10 worse in two folds,
  tied in the third. Keep dedup disabled; development gain did not confirm.
  Confirmation is now consumed, not an untouched tuning target. Detailed
  evidence/private report paths in FINDINGS.md. No eval jobs left running.

## Preserved uncommitted work (excluded from deployment)

- `pipeline/embedding_sections.py`
- `scripts/bakeoff_embedding_models.py`
- `tests/test_embedding_sections.py`

## Next step / blockers

No ranking rollout: confirmation favored unchanged production. Keep both
experiment flags off. Further experiments require a new evidence plan/future
feedback, not tuning against the now-consumed confirmation period.
No deployment blocker. Separate older project blocker: PyPI publishing
credentials still needed. No further work scheduled in this session.
