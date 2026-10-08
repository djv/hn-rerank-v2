**Verdict:** A hybrid SVC can, in principle, absorb the kind of function dense LR learns. This screen doesn't test that well yet, and nothing in it absorbs word evidence. The best single RBF loses about .006 AUC. Earlier studies found removing dense LR costs about .003 (worse in 11/12 blocks, `FINDINGS.md:769`) and removing words about .002. So the gap is roughly what removing both would be expected to cost.

**1. Conceptual (confirmed from source)**
- `(1−s)·RBF + s·linear` is one SVM over "RBF part + linear part". Dense LR trains on the same scaled rows (`pipeline/ranking.py:1506-1508`), so the hybrid can represent the same linear functions.
- What it can't absorb:
  - **Loss:** LR uses log loss over every point; the SVM's hinge loss is driven only by points near the margin.
  - **Score:** LR gives a smooth P(up)−P(down). The SVM's "up" score is a vote count (0–2) plus a confidence term bounded to ±1/3 (sklearn `multiclass.py:557-599`, used at `ranking.py:1556`).
  - **Ensemble effect:** the blend averages the ranks of independently fitted models, so their errors partly cancel. A joint fit splits one solution between its parts instead.
  - **Words:** domain and word tokens (`pipeline/linear_blend.py:43-49`) never reach the SVM.

**2. Wrapper findings**
- **Medium – the share grid mixes three things.** The linear part's effective C is about C·s/L. L is the mean squared row norm: about 1 from the embedding plus about 10 standardized metadata columns (`ranking.py:1025,1476-1481`). At C4, s=.1 that is about .04, against LR's `dense_c=.1` (`pipeline/config.py:40`). The (1−s) factor also shrinks the RBF part's C, and the hybrids ran at γ.03 while the best RBF used γ.01. The share-.1 result may have tested an almost inactive linear term (hypothesis: hinge C and log-loss C don't map exactly).
- **Medium – no routing test.** The plan claims one (`SINGLE-MODEL-PLAN.md:42-43`), but the tests cover only the kernel and the cohort. Identical baseline scores show the baseline was untouched, not that the no-dense variants got the new kernel. Check that hybrid score dumps differ from same-config RBF dumps.
- **Low-medium:** `--feedback-cohort` only patches `get_feedback_for_training` (`scripts/eval_single_preference_model.py:153-162`). With `--embeddings-file`, test feedback comes from `_snapshot_feedback` (`scripts/eval_ranker_variants.py:2566`), so training would be filtered and the test set would not. The heldout-feedback runs don't hit this; the wrapper should reject the combination.
- **Low:** the cohort's identity is only printed (`:180-183`) and never written to the report JSON.
- **Low:** a hybrid run gives the new kernel to every no-dense variant, so there is no RBF control in the same process (`:138-151`).
- **No leakage found.** Linear scale, StandardScaler, kNN/cluster features and TF-IDF vocabulary/IDF are all fitted on training rows only. `user_id=None` disables the model caches (`eval_ranker_variants.py:267`, `ranking.py:1222`). URL-group isolation is intact. Replay concatenation scales each part by 1/√k, matching production (`pipeline/side_embeddings.py:54`).
- **Fairness:**
  - The baseline's weights and C values were tuned on overlapping historical labels, while the winner here is best of 12; which way the net bias goes is unknown.
  - The reserved latest 20% overlaps the Oct 2–7 blocks already examined in the README, so it is not fresh evidence.
  - P@12 moves in steps of 1/48: .7708 vs .75 is one card. Requiring all four point estimates to be no worse mostly filters on noise.
  - Embedding cohorts are judged-only replays with side vectors off, so they can't be compared with main-screen numbers. Picking stories by matching text hash may also skew the cohort.

**3. Highest-value experiments**
1. **Anchored hybrid:** RBF(C16, γ.01) + λ·linear, without shrinking the RBF part, with λ set so the linear part's effective C is in {.03, .1, .3}. Log how much of each decision comes from the linear vs the RBF part, to show the linear term is active.
2. **Continuous margin (untested):** take the same SVC with `decision_function_shape="ovo"` and score up-vs-down margin + up-vs-neutral margin. Alternatives are a binary up-vs-rest model or two ordinal thresholds. C/gamma/kernel changes can't remove the vote-count steps; this can. Prior `produd` (`FINDINGS.md:745`) was a softmax over the vote-count scores, not this. The effect may be small, because the confidence term already orders cards within the top vote tier.
3. **Words inside the one classifier:** add a TF-IDF cosine kernel (vocabulary/IDF fitted on training only), or a cheaper domain-prior column. `publication_affinity_enabled` exists, but its only test (2026-09-22) used a different, sparse pool and settles nothing. This keeps one fit but also keeps the word pipeline; the user decides whether that meets the goal. Otherwise, accept losing words explicitly.

**4. Evidence needed to remove both**
- Fix a non-inferiority margin before running, for example ΔAUC ≥ −.005 and down@12 at most one card worse pooled.
- Pick one finalist on development folds only.
- Evaluate it on all 12 blocks and, decisively, on votes collected after 2026-10-07 that no selection has touched.
- Require the lower bounds of both the paired story bootstrap and the whole-block interval to clear the margin, and run a label-shuffle check against the pool's upvote rate.
- Measure the real latency and memory saving. Currently the runtime still fits both linear models even at zero weight.

The current −.006 would fail any reasonable margin, but it is not yet proven to be a real loss.
