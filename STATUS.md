# HN Rerank status

## Objective

Scoped ranking investigation and authorized Import AI duplicate-vote cleanup
are complete. Keep production ranking unchanged; no further work scheduled.
Full-body embedding work remains deferred.

## Verified result

- Pushed/deployed: code `f03c34e`, documentation `b0c73cb`. TLDR detail-v12
  preserves scaled article/discussion budgets; Muse Spark serves summaries.
  Import AI 473 coverage verified, including machine hermeneutics.
- TUI selected-story impression logging deployed and verified in the VPS
  ledger. Default profile preserved. Dashboard/cached/uncached summary checks
  passed; service active and bounded journal scans clean at last verification.
- Commit snapshot checks: 795 backend tests passed; client 63 passed / one
  Windows-only skip; Ruff/format/ty clean.
- Confirmation evaluation: 995 judged items, three chronological folds.
  Production/dedup NDCG@10 **0.7603/0.7138**, MAP **0.4992/0.4956**.
  Development gain did not confirm. Both `publication_affinity_enabled` and
  `deduplicate_training_feedback` remain false. The confirmation period is
  consumed; do not tune against it or claim these replay metrics are live
  recommendation quality.
- User-authorized cleanup removed exactly three older duplicate Import AI
  upvotes (458/459/460). Newer votes and all six article rows preserved;
  13 Jack Clark upvotes remain. Service restarted and dashboard verified 200.
  Full SQLite backup and exact affected-row manifest retained privately at
  `/home/dev/hn-rewrite/shared/feedback-cleanup-backups/20260922T201813Z/`.
  Earlier eval artifacts describe the pre-cleanup feedback snapshot.

## Workspace / evidence

- Local uncommitted documentation records confirmation and cleanup:
  `FINDINGS.md`, `WORKLOG.md`, `STATUS.md`, and this save's status archive.
- Deferred uncommitted embedding WIP, not deployed:
  `pipeline/embedding_sections.py`, `scripts/bakeoff_embedding_models.py`,
  `tests/test_embedding_sections.py`. Preserve; do not accidentally stage.
- Detailed evidence/report paths: `FINDINGS.md`. Previous status archived in
  `docs/status-archive/2026-09-22-before-final-save.md`.
- Remote pre-deploy TLDR patch remains safely preserved in named stash
  `pre-f03c34e-deploy-identical-detail-v12`. No eval jobs left running.

## Blocker / next step

No operational blocker. Stop here unless asked to continue. Any new ranking
experiment needs a fresh evidence plan or future feedback, not repeated
confirmation tuning. Documentation changes above are not yet committed.
Separate existing project blocker: PyPI publication needs credentials.
