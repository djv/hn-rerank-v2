# HN Rerank status

## Objective
Improve what the dashboard shows the user (live profile 151; user 1 is the
old profile, stopped 2026-09-24): ranking quality and which sources feed it.

## Verified result
- Live on the VPS at `f49ff0f` (2026-09-29 13:06 UTC): time-window selector
  (12h/1d/1w/1m/Archive, `d` cycles; web and TUI reopen on the last window
  picked), AINews per-topic source (`rss_ainews`), 2026-09-26/27 source
  changes, server/web aligned with the TUI. Live ranker unchanged:
  production (SVM C=0.1 on stored embeddings).
- TLDRs (live `edb1316`, 2026-09-29 17:02 UTC): OpenCode Go plan limit
  spent until ~2026-10-06, so `LLM_PROVIDER=gofree` (free
  `longcat-2.5-preview-free` on the Go gateway, `reasoning_effort=low`).
  Smoke: 5 of 6 TLDRs complete, 15-45s each. WORKLOG.md 2026-09-29.
- TUI `a` (uncommitted, another session's work; tests pass): Claude Code in a tmux pane split beside
  the reader with the article/comments links and a dig-deeper prompt.
- Profile merge (live, 2026-09-29 14:01 UTC, user chose "July onward"):
  user 1's 2,332 votes since 2026-07-01 on stories 151 had not voted on
  copied to 151 (467 -> 2,799 votes). Backup
  `main/hn_rewrite.db.pre_merge_1_to_151_20260929T140130Z` on the VPS.
  Offline, merging raised AUC on 151's later votes ~0.84 -> 0.88-0.91 with
  the top 12 unchanged. FINDINGS.md "Merging user 1's votes...".
- Offline ranking (FINDINGS.md 2026-09-29 sections): best is stored +
  embeddinggemma side by side, `prodlr[svm_c=4.0;lr_weight=0.2;
  tfidf_weight=0.3]` (SVM C=4 rank-blended with logreg 0.2 and a word
  TF-IDF logreg 0.3). TF-IDF alone added AUC 0.791 -> 0.801 (7/8 folds) and
  was better or equal on all five unseen-vote checks (user 1's reserved
  newest 20%, 151's votes after 09-26/09-27, alone and merged).
- Linear blend built and pushed (`f01dbba`, `21f870c`), off by default and
  not deployed: `linear_blend_enabled` in `pipeline/linear_blend.py`.
  Against the actual live ranker (`svm_c=0.1`, stored embeddings only) the
  full eval gives AUC 0.726 -> 0.790 (p=0.017), P@12 0.625 -> 0.719; the
  newest-20% run is flat (AUC 0.739 -> 0.751, P@12 0.667 both). TF-IDF alone
  on top of svm_c=4 + LR: +0.016 AUC. Codex review: no leakage, but the
  "unseen votes" checks overlap and informed tuning, so the size of the gain
  is unconfirmed. FINDINGS.md "Proposal vs the actual live ranker".
- TF-IDF sweep (full eval, FINDINGS.md "TF-IDF tuning sweep"): nothing
  beats weight 0.3 beyond noise. Title-only input, source prior 0.1 and the
  joint model are worse at the top 12; char n-grams, TF-IDF C and half-life
  make no difference.
- Rejected on the full eval: MMR (costs upvotes at every threshold), one
  SVM per embedding (AUC up, top 12 down), skipped stories as weak
  negatives, binary up-vs-rest logreg, stacking as implemented (Codex found
  leaks; not a verdict on stacking). Embedding models: gemma best;
  harrier-0.6b ties it only without the instruct prefix.
- Eval fixes: tie-aware AUC and average-rank percentiles; URL-group
  isolation for synthetic training rows. Codex review of the evals:
  `/tmp/hn-eval-local/codex-ml-review.md` (how to ask: `~/AGENT-ACCESS.md`).
- Feed check (2026-09-29): Reddit 429s ~55-63/day (was ~150), every
  configured subreddit fresh except r/ocaml (7 failures); 531 RSS rows
  without article text; r/transit and r/MachineLearning kept (too few
  shown). FINDINGS.md "Feed yield check — 2026-09-29".

## Blocker / limits
- Enabling the linear blend changes the live ranker: needs the user's OK.
  Gemma side by side is not live and would need gemma on the VPS.
- Go limit resets ~2026-10-06: then set `LLM_PROVIDER=gospark` in the
  VPS `shared/.env` and restart. Luna via OpenAI needs an API key (none;
  the ChatGPT plan only covers Codex).
- The merge's live effect needs ~100 new votes from 151 to read.
- Offline gains over live are large on the full eval but flat on the newest 20%.
- `/tmp` is wiped at boot: the eval dir is mirrored to
  `~/.local/state/hn-rerank-eval/hn-eval-local` (last synced 2026-09-29 12:15);
  snapshots and merged copies live in `~/.local/state/hn-rerank-eval/`.
- Empty `~/hn-rewrite/hn_rewrite.db` on the VPS (stray, harmless): delete
  only with the user's OK.

## Next step
- Built, off by default (`linear_blend_enabled`, `pipeline/linear_blend.py`):
  0.5 production + 0.2 dense LR + 0.3 TF-IDF LR. Vs live (`svm_c=0.1`, stored
  embeddings) AUC 0.726 -> 0.790 on the full eval, flat on the newest 20%.
  Awaiting the user's OK to enable: `svm_c = 4.0` and
  `linear_blend_enabled = true` in `config.toml`, deploy, restart. Then judge
  on new votes from 151.
- Commit the TUI `a` key (`clients/tui/`, WORKLOG entry already written)
  when the user asks; restart the reader to use it.
- After ~100 new votes: compare 151's up rate on shown stories before and
  after the merge (`scripts/source_yield_report.py --user-id 151`).
- Optional, low value: embedding queue (jina-v5-nano, mdbr-leaf-mt,
  Qwen3-0.6B, KaLM-mini), all without instruct prefixes.
