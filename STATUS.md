# HN Rerank status

Updated 2026-10-08. Previous handoff (one-classifier research, Why-this-story
preview, pending Reddit/Gemma items):
[before interleaving](docs/status-archive/before-interleaving-20261007.md).

## Objective
Compare shortlist #2 (one joined logistic classifier, all features) and #4
(same, no metadata) live against production through team-draft
interleaving on profile 151's Recommended views for 28 days, plus four
offline follow-ups the user chose (power replay, fit timing,
metadata-scale middle variant, short-history curve).

## Result
- Implemented (off by default): `model.classifier = "joined_logistic"`,
  `interleave_user_ids`, `interleave_decks` table,
  `scripts/interleave_report.py`; challengers warm-start and share
  production's feature build. Design, power, timing and the fixed
  28-day decision rule:
  [INTERLEAVING.md](docs/evaluations/model-ablation-20261007/INTERLEAVING.md).
- Power replay (optimistic): 28 days gives ~70% (#2) / ~54% (#4) power for
  the upvote-rate test; null false-positive rate 1-2%.
- VPS timing (copy of live DB): post-vote rerank 21 s with interleaving vs
  ~10 s production alone.
- Gates: VPS copy 1,168 passed (3 git-dependent eval tests fail there only;
  they pass locally), Ruff clean, ty only the existing inspection-script
  diagnostic.
- User asked to merge everything into main and deploy on the VPS
  (includes the Why-this-story expansion); profile 151 enabled in
  config.toml.

## Next step
- Deploy main to the VPS, smoke-test, record T0 (first interleave_decks
  row) here; final analysis at T0 + 28 days with the command in
  INTERLEAVING.md; weekly descriptive reports and safety checks.
- Offline follow-ups still running on the laptop
  (`/tmp/hn-single-followup-20261007/launch.py`, clean worktree
  ../hn-rerank-offline-20261007 at b5eabdc); copy results into
  docs/evaluations/model-ablation-20261007/followups/ and fill the pending
  INTERLEAVING.md sections.
- The separate Why preview (VPS hn-why-preview-20261007.service, port
  8767) becomes redundant after the deploy; stop it when the user agrees.
- Carried over: Reddit replacement before November 13, 1m Popular age-mix
  check, real-terminal tint/footer check, October 21 Gemma 2 future-vote
  recheck (/home/d/TASKS.md).
