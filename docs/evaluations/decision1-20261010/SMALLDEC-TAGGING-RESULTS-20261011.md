# Small decision models, Ettin fine-tune and zero-shot tagging — results 2026-10-11

All numbers are AUC(up vs rest) on judged held-out stories of the frozen folds
(`/tmp/decision1-eval/work_heldout/folds.json`, sha256 `804d28ae…f0db`), VPS CPU only.
"Blend" is `0.8·pct(production) + 0.2·pct(model)` within each fold; w = 0.2 is the
only pre-registered weight (other weights in the raw outputs are exploratory).
Pooled numbers use within-fold percentiles and a within-fold story bootstrap.
Per-story scores and labels stay on the VPS under `/tmp` (public repo; not committed).

## 1. Zero-shot decision models with the reader profile (Julia-1, Mira-v1)

Subset: fixed-seed 100 candidates per fold (`subset-100.json`, seed 0), real and
empty profile states.

| Run | Model AUC | Production | Blend − production [95% CI] |
|---|---|---|---|
| Laya-multilingual, partial (608/1000, stopped by user) | empty 0.53, real 0.50 (76 paired) | 0.79–0.84 | not computed |
| Julia-1, long profile (~2.5K tokens), 5 folds | real 0.47 (mean), empty 0.56 | 0.79 | real −0.015 [−0.030, −0.001]; empty −0.003 [−0.016, +0.011] |
| Julia-1, variant q, folds 1–4 pooled | 0.507 [0.44, 0.58] | 0.801 | −0.013 [−0.027, +0.001] |
| Mira-v1 on variant q (logit gap), folds 1–4 | 0.540 [0.47, 0.61] | 0.801 | −0.003 [−0.018, +0.011] |

Variant q was picked over "short" on fold 0 (0.693 vs 0.630); folds 1–4 are the
confirmation. Real-minus-empty for the long Julia profile was −0.10 [−0.18, −0.02]:
the profile made scores worse, consistent with a 2.5K-token state being out of
distribution for a model benchmarked at 1,024 tokens.

**Reading:** no zero-shot decision model showed taste signal; none is worth a
production slot.

Independent review (cloud `rv`, findings verified by the agent):
- Fixed before the q runs: Mira probability saturation (the score is now the raw
  logit gap).
- Kept as caveats: the Laya partial run was sorted by state length, so its partial
  numbers cover the shortest stories only; only w = 0.2 was pre-registered; report
  fields say "laya_"/"julia" for every backend (`report-mira-q.json` fields named
  "julia" are Mira); model weights are not hashed into the score identity (revision
  pin only); `ft_fold.py` did not check the folds receipt.
- Rejected: "unjudged candidates counted as negatives" — every held-out candidate in
  folds.json is judged (labels 0/1/2), so the AUC set is already judged-only.

## 2. Ettin-encoder-17M fine-tune (user constraint: under 30 min per fine-tune)

`ft_tiny.py`: jhu-clsp/ettin-encoder-17m (revision `98760745…`), 2 epochs, lr 1e-4,
160 tokens (title, domain, source, excerpt), target up vs rest; epoch picked on the
last 15% of training votes by time. Baseline: TF-IDF logistic regression on the
same text.

| Fold | Ettin | TF-IDF | Production | Blend Ettin | Blend TF-IDF | Epoch | Minutes |
|---|---|---|---|---|---|---|---|
| 0 | 0.665 | 0.677 | 0.749 | 0.758 | 0.759 | 2 | 3.0 |
| 1 | 0.689 | 0.668 | 0.805 | 0.801 | 0.801 | 2 | 3.8 |
| 2 | 0.678 | 0.669 | 0.876 | 0.864 | 0.861 | 1 | 7.5 |
| 3 | 0.738 | 0.725 | 0.840 | 0.841 | 0.840 | 2 | 5.2 |
| 4 | 0.662 | 0.664 | 0.723 | 0.729 | 0.723 | 1 | 5.7 |

Pooled over 5 folds (2,066 judged held-out stories, production 0.781):

| Arm | Model AUC [95% CI] | Blend − production [95% CI] |
|---|---|---|
| Ettin-17M | 0.676 [0.650, 0.702] | +0.0007 [−0.0041, +0.0056] |
| TF-IDF LR | 0.672 [0.645, 0.699] | −0.0010 [−0.0059, +0.0038] |

**Reading:** the fine-tuned encoder learns real taste signal (0.68, far above
chance) but no more than TF-IDF, and adds nothing on top of production, which already
uses two embeddings plus TF-IDF. Whole run 25 min on 4 VPS cores (each fold under
8 min), within the user's 30-minute limit. One seed only; no seed averaging.

## 3. Zero-shot story tagging (design: TAGGING-DESIGN-20261011.md)

12 pre-registered tags, frozen `tags.json` (sha256 in `freeze-receipt.json`).
Arms: T0 rules (show_hn, ask_hn, research, paywalled), T1 embedding prototypes
(mxbai cosine to the "true" minus the "false" criterion), T2 Julia-1 noul logit
gap, N seeded noise.

### (A) Validity on the silver set
240 pre-registered stories (seed 1, 40 per stratum), labelled blind by the agent
(Sonnet subagent) and Muse via OpenCode; only per-tag agreements are kept.
Agreement: show 0.98, ask 1.00, research 0.93, tutorial 0.99, opinion 0.97,
news 0.88, business 0.93, politics 0.97, ai 0.96, product_pr 0.95, sensational
0.94, paywalled 0.97. Agreed positives are scarce for tutorial (5) and
sensational (5), so those AUCs are very noisy.

AUC / F1 on the silver set (F1 on the held-out half; "—" = no rule for that tag):

| Tag | Silver n (pos) | T0 rule | T1 proto | T2 Julia |
|---|---|---|---|---|
| show_hn | 235 (22) | 0.93 / 0.93 | 0.79 / 0.29 | 0.42 / 0.13 |
| ask_hn | 240 (27) | 0.63 / 0.22 | 0.51 / 0.21 | 0.51 / 0.13 |
| research | 224 (12) | 0.62 / 0.00 | 0.48 / 0.00 | 0.65 / 0.14 |
| tutorial | 237 (5) | — | 0.82 / 0.22 | 0.51 / 0.00 |
| opinion | 233 (34) | — | 0.61 / 0.18 | 0.54 / 0.23 |
| news_event | 212 (82) | — | 0.45 / 0.52 | 0.32 / 0.53 |
| business | 224 (19) | — | 0.45 / 0.12 | 0.56 / 0.18 |
| politics_policy | 234 (31) | — | 0.27 / 0.21 | 0.49 / 0.28 |
| ai_ml | 231 (87) | — | 0.59 / 0.47 | 0.35 / 0.54 |
| product_pr | 227 (14) | — | 0.83 / 0.33 | 0.29 / 0.13 |
| sensational | 225 (5) | — | 0.43 / 0.00 | 0.59 / 0.00 |
| paywalled | 233 (10) | 1.00 / 0.91 | 0.43 / 0.00 | 0.39 / 0.00 |

Validity bar (AUC ≥ 0.80) met by: T0 show_hn and paywalled; T1 tutorial and
product_pr (AUC only; F1 ≤ 0.33, tutorial has 5 positives); T2 none. Several T2 tags
are inverted (product_pr 0.29, news_event 0.32, ai_ml 0.35); even sign-flipped, no T2
tag would exceed 0.71, so the failure does not depend on the option-order convention.

F1 uses a threshold chosen on a seeded, score-blind half of the silver ids
(`random.Random(1)`), scored on the other half (added to the report after the
cheap-arm AUCs were seen, before any Julia score existed).

Caveats:
- T1 is weak or inverted on most topic tags (politics 0.27). Likely cause: the
  "false" criterion repeats the topic words ("Politics, law and policy are not its
  main subject") and embeddings ignore negation, so a design flaw of T1, not a
  property of the tags.
- The ask_hn rule matches only the "Ask HN" title prefix; both labellers also
  tagged Reddit/Tildes question posts and "AskHN:"/"Tell HN" titles. The research
  rule misses nature.com/thelancet.com papers and news reports of studies. Rule
  "truth" is therefore exact only for show_hn and paywalled.

### (B) Ranking utility (primary)
Per fold: L2 logistic regression on standardized tag scores, λ from
{1e-4, 1e-3, 1e-2, 1e-1} on the last 15% of training votes by time; held-out
blend at w = 0.2; 2,000 bootstrap draws, paired across arms.

| Arm | Tag-LR AUC [95% CI] | Blend − production [95% CI] |
|---|---|---|
| N noise | 0.501 [0.474, 0.529] | −0.0074 [−0.0135, −0.0013] |
| T0 rules | 0.480 [0.453, 0.509] | −0.0017 [−0.0054, +0.0020] |
| T1 proto | 0.632 [0.605, 0.658] | −0.0059 [−0.0111, −0.0006] |
| T2 Julia | not run (stopped) | — |

Production alone: 0.781 pooled (2,066 held-out stories, 2,000 bootstrap draws).

T2 compute: the 50-story pilot ran at 2.32 s/story (12 tags per call, 4 threads,
batch vs single max diff 1e-6), over the 2.1 s/story gate (full universe ≈ 2.7 h > 2.5 h).
Per the design fallback, T2 scores the 240 silver stories, all 2,066 held-out
stories and the first 1,500 training-only stories of the seeded shuffle (3,606 in
all, ≈ 2.3 h; `tagging.py julia --train-sample 1500`). The T2 tag LR therefore
trains on fewer stories than T0/T1/N (fold 0: only sampled training-only stories).

Pre-registered rule: Julia "adds signal" only if its blend − production CI lower
bound is above 0, its gain beats noise, and its gain is at least T1's.

**T2 verdict:** the user stopped the run ("Stop Julia, commit now") after 347 of
3,606 stories (all 240 silver + 107 others), so the pre-registered utility rule could
not be applied. The silver validity result above already shows that Julia's
zero-shot tags are not usable. `report.json` also has a matched-set
sensitivity (all arms trained on Julia-scored stories only); it was tested with
proto scores substituted for Julia (Julia − proto CI exactly [0, 0]) but never run on
real Julia scores.

Note: even the noise arm costs −0.007 at w = 0.2, so the blend test penalizes any
weak column; T1's tag model alone reaches 0.63 yet still lowers the blend.

Not run (exploratory items in the design): per-tag ablation, per-tag up-rate lift,
`margin3_plus_tags` through `_ablation_extra_columns`, other blend weights.

## Artifacts (VPS)
- Scripts: `smalldec-20261011/` in this directory (copies of `/tmp/smalldec/scripts`,
  `/tmp/finetune/ft_tiny.py`, `/tmp/tagging/scripts`).
- Aggregate reports: `smalldec-20261011/*.json` (no per-story rows).
- Per-story scores, silver labels and model files: `/tmp/smalldec`, `/tmp/finetune`,
  `/tmp/tagging` on the VPS only.
