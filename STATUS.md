# HN Rerank status

## Objective
Improve what the dashboard shows the user (live profile 151; user 1 is the
old profile, stopped 2026-09-24): ranking quality and which sources feed it.

## Verified result
- Linear blend live on the VPS since `26b9474` (2026-09-29 17:17 UTC), now
  at `7d9c231` (17:27 UTC, hashed TF-IDF rows cached per story):
  `svm_c = 4.0`, `linear_blend_enabled = true` (0.5 production + 0.2 dense
  LR + 0.3 TF-IDF LR). No errors after restart; feed reads 200 in ~0.5 s via
  Tailscale for every window. First rerank cost 21-29 s (fit 14 s,
  scoring 8-9 s); with the row cache warm a rerank is 7.7 s (fit 3.6 s,
  scoring 0.5 s; live before the blend: 4-13 s). The fit repeats after every
  vote. A
  "refresh failing" report matched the ~100 s restart gap, not a bug.
  FINDINGS.md "Linear blend live: rank latency".
- Also live (from `f49ff0f`): time-window selector (12h/1d/1w/1m/Archive,
  `d` cycles; web and TUI reopen on the last window picked), AINews
  per-topic source (`rss_ainews`), 2026-09-26/27 source changes.
- TLDRs (live `ea1403b`, 2026-09-29 20:44 UTC): `LLM_PROVIDER=mistral`
  (`mistral-small-latest`, paid key with a $10 cap), 2-4 s per TLDR; the
  free `gofree` model took p50 45 s and left 60 of 132 failed or half-only.
  Same stories: Mistral is correct but more generic than longcat. Fixes:
  one retry when a reply has no bullets (lost halves), pointer threads
  ("Comments moved to item?id=N") follow the link, tweet URLs summarized via
  fxtwitter plus the page they link. Live checks: 32148318, 30230620.
  FINDINGS.md "TLDR providers and quality".
- ClickHouse source (live `7b1d70a`, 20:57 UTC): live-window query retried
  3 times; comments nested in HN order up to 30 levels (was a flat list,
  so thread-aware selection saw depth 0); live HN caps 5000 -> 10,000.
  First regen: 9,187 candidates, no errors, dashboard 0.23 s; first rerank
  over 10,111 candidates took 20 s. FINDINGS.md "ClickHouse source review".
- TUI `a` (committed `48e460f`, tests pass): Claude Code in a tmux pane split beside
  the reader with the article/comments links and a dig-deeper prompt.
- TUI status line (committed `48e460f`, 2026-09-29): always one row; long messages
  end in `…` instead of wrapping to 3 rows (hints still stack below when
  both don't fit). TUI tests 152 pass, ruff clean.
- TUI headline dividers (2026-09-29): tried and reverted at the user's call;
  list unchanged. Rule rows cost a row per story; a meta-line underline
  looked link-like. FINDINGS.md "TUI headline dividers".
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
- Linear blend offline case (`f01dbba`, `21f870c`, `pipeline/linear_blend.py`).
  Against the previous live ranker (`svm_c=0.1`, stored embeddings only) the
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
- Gemma side by side is not live and would need gemma on the VPS.
- Blend gain is unconfirmed (flat on the newest 20%).
- Go limit resets 2026-10-06 16:28 UTC: then set `LLM_PROVIDER=gospark`
  (now `mistral`) in the VPS `shared/.env` and restart. Mistral stops at its
  $10 cap (balance not checked). OpenRouter, Zen and Cerebras have no
  credit; Gemini free is 20 requests/day.
- The merge's live effect needs ~100 new votes from 151 to read.
- Offline gains over live are large on the full eval but flat on the newest 20%.
- `/tmp` is wiped at boot: the eval dir is mirrored to
  `~/.local/state/hn-rerank-eval/hn-eval-local` (last synced 2026-09-29 12:15);
  snapshots and merged copies live in `~/.local/state/hn-rerank-eval/`.

## Next step
- Judge the live blend on new votes from 151 (up rate on shown stories
  before/after 17:17 UTC 2026-09-29). Rollback: `linear_blend_enabled =
  false`, `svm_c = 0.1` in `config.toml`, deploy, restart.
- Measure a warm rerank with 10,111 candidates (first was 20 s vs 7.7 s
  before); if it stays slow, lower `recent_candidate_hn_limit`.
- Open TLDR gaps: a raw PDF stored as article text (46108780 fails);
  archive dupe cards are not swapped by the dupe resolver (live `hn` only).
- Restart the TUI reader to pick up the `a` key and one-row status line.
- After ~100 new votes: compare 151's up rate on shown stories before and
  after the merge (`scripts/source_yield_report.py --user-id 151`).
- Optional, low value: embedding queue (jina-v5-nano, mdbr-leaf-mt,
  Qwen3-0.6B, KaLM-mini), all without instruct prefixes.
