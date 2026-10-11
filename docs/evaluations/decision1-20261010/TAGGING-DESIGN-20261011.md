# Zero-shot story tagging: experiment design (draft, 2026-10-11)

## Question
Do zero-shot tags from a small decision model add ranking signal for user 1's HN votes
beyond production, and are the tags accurate enough to show or filter on?
Two separate claims, tested separately: **(A) tag validity**, **(B) ranking utility**.

## Data (frozen)
- Evaluator: `scripts/eval_ranker_variants.py`, frozen report
  `/tmp/decision1-eval/work_heldout/evaluator_report.json`.
- Votes: user 1 has 4,151 valid votes (1,327 down, 1,431 neutral, 1,393 up).
  There are 5,189 candidates in total.
- Folds: 5 time-ordered expanding-window folds, 415 test rows each, with
  2,076–3,736 training rows.
- Production per-story features come from mxbai and embeddinggemma embeddings,
  plus 10 meta columns (length, similarities, source flags). TF-IDF with
  `dom_`/`src_` tokens is blended in.
- Production has no feature for paywall, clickbait, genre or format.

## Tag set (pre-registered; frozen before any scoring)
Each tag has a one-sentence yes/no criterion, written into `tags.json` and hashed.

| # | Tag | Criterion (abridged) | Heuristic truth? |
|---|-----|----------------------|------------------|
| 1 | show_hn | the author presents their own project | title prefix |
| 2 | ask_hn | a question or discussion with no external article | title prefix |
| 3 | research | a paper or preprint, or an article mainly reporting one | arxiv/doi/acm/ieee domains |
| 4 | tutorial | teaches how to do something step by step | — |
| 5 | opinion | an argued personal essay or opinion piece | — |
| 6 | news_event | reports a recent event: release, outage, acquisition, lawsuit | — |
| 7 | business | about funding, earnings, layoffs or company strategy | — |
| 8 | politics_policy | about government, law, regulation or elections | — |
| 9 | ai_ml | mainly about AI or ML | — |
| 10 | product_pr | a company announcing its own product or feature | — |
| 11 | sensational | the title overstates or baits | — |
| 12 | paywalled | the source is usually paywalled | domain list |

## Arms
| Arm | What | Cost |
|-----|------|------|
| T0 heuristics | regex/domain rules for tags 1, 2, 3 and 12 only | free |
| T1 embed-proto | cosine of the stored mxbai story embedding to an embedded tag criterion | seconds; no new model |
| T2 Julia-1 | the noul P(yes) per (story, tag). The state holds the story only (title, domain, source, 500-char excerpt; under 300 tokens) and the question holds the criterion | pilot first |
| N noise | 12 columns of N(0,1), seed 0 | control |

Mira is dropped: its yes/no protocol is not documented, and it was slower (3.3 s/score).

## (A) Validity
- **Silver set:** 240 stories, stratified at random by fold (seed 1) and
  pre-registered. Two independent labelers tag them blind to model scores:
  - the agent, and Muse via OpenCode (on the user's plan, no API credits);
  - only stories where both agree form the silver labels, and the agreement rate
    is reported per tag.
- **Heuristic tags (1, 2, 3, 12):** the rule is ground truth on all 5.2k stories.
- **Metrics per tag:** AUC, F1 at the threshold chosen on half of the silver set,
  and prevalence. Report T1 vs T2.
- **Validity bar:** AUC ≥ 0.80 on silver. Tags below the bar are reported as invalid
  but kept in (B), since (B) needs no valid labels.

## (B) Ranking utility (primary)
- **Tag model:** per fold, fit a logistic regression on the 12 tag scores (logit)
  using that fold's training votes, target up vs rest. The regularization λ is
  chosen on the last 15% of training votes by time. Score the held-out fold.
  This is the same protocol as the Ettin/TF-IDF fine-tune.
- **Primary metric:** pooled within-fold percentile AUC of
  `0.8·pct(prod) + 0.2·pct(tag LR)` minus production, using a story bootstrap
  (2,000 resamples), the same as the earlier runs.
- **Pre-registered decision rule:**
  - T2 "adds signal" only if the lower bound of its 95% CI is above 0,
  - and its gain is above the gain of N noise,
  - and its gain is at least the gain of T1.
  - If T1 matches T2, the cheap prototype tagger wins.
- **Secondary (exploratory):**
  - tag-LR AUC alone;
  - per-tag ablation (drop one tag, change in gain);
  - per-tag up-rate lift (e.g. P(up | tag) − P(up)) with a bootstrap CI;
  - `margin3_plus_tags` vs `margin3_plus` through the existing
    `_ablation_extra_columns` hook, as a within-SVC check;
  - other blend weights.
- **Leakage:** tags are label-free and computed per story, so they are not fitted
  on votes. The tag text is frozen before scoring. Fold, label and production
  inputs are the frozen folds.json and report, checked against their SHA-256
  receipt.

## Compute (VPS CPU, after the Ettin run ends)
- A T2 pilot on 50 stories × 12 tags measures s/pair and tests a batch vs single
  call. The run continues only if the full job finishes in under 2.5 h:
  - 5,189 stories × 12 tags is about 62k pairs, so the gate is about 0.15 s/pair;
  - if over the gate, T2 runs on judged test-fold stories plus a 1,500-story
    training sample, and the tag LR is trained on that sample.
- Runs as a systemd user unit at Nice=10 with MemoryHigh 3000M, 4 threads,
  resumable JSONL, model identity and weight SHA-256 in every row.
  The `hn_rewrite.service` is not touched.

## Expected result (stated before running)
Production already encodes topic through two embeddings and TF-IDF, so topic tags
(ai_ml, politics) probably add nothing. Any gain is most likely from tags that
embeddings encode poorly: format (tutorial, opinion, show_hn), sensational and
paywalled. A realistic upper bound is +0.005 to +0.015 AUC. A null result is
still useful: tags would then serve only as user-visible filters and explanations,
which part (A) covers.

## Outputs
`/tmp/tagging/` for scores and reports; the summary goes into this directory's
`FINDINGS.md` and `STATUS.md`.
