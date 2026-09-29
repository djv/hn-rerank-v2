# HN Rerank findings

## Incremental ranker hill-climb — 2026-09-28

Setup: `eval_ranker_variants.py --candidate-pool heldout-feedback`, user 1,
snapshot 2026-09-25 (same development folds as the 2026-09-25 study; the
newest 20% stays reserved). Each step changes one thing from the current
best: first a small test (300 votes/class, 3 folds, ~1 min), then the full
eval (all 5,189 votes, 8 folds, ~5 min) only if the small test is no
worse on the top-12 upvote rate and better on AUC. Primary metric: upvote
rate in the top 12 (the clients' 12-card view); also AUC(up vs rest),
downvote/neutral share of the top 12, discovery upvotes (top-12 upvotes
below the fold's median similarity to training upvotes) and non-HN
upvotes. Reports: `/tmp/hn-eval-local/{small,full}-*.json` (not kept).

| Step | Change | Small test | Full eval (vs previous best) | Kept |
|---|---|---|---|---|
| 1 | up-minus-down margin scoring | worse | – | no |
| 1-2 | SVM C 0.1 → 2 | P@12 0.306→0.389, AUC 0.577→0.657, 3/3 folds | P@12 0.625→0.646 (p=0.52), AUC 0.726→0.759 (p=0.046), discovery 0.25→1.12 | yes |
| 3 | mxbai-xsmall text cut at 512 tokens | 2/3 folds better | P@12 0.646→0.615 (worse 7/8), AUC +0.009 | no |
| 4 | γ 0.015 / 0.05 at C=2 | P@12 lower | – | no |
| 5 | + logreg rank blend, weight 0.3 (0.6 tested) | P@12 tie, AUC 0.657→0.674 | P@12 0.646→0.688 (p=0.10), AUC 0.759→0.769 (p=0.07) | yes |
| 6 | body + comments as two 512-token vectors (mxbai-xsmall) | P@12 0.389→0.500 (1/3 folds), AUC 0.674→0.700 | P@12 0.688→0.635 (n.s.), AUC 0.769→0.788 (p=0.025, 7/8), discovery 1.25→0.12 | no (top 12 is the target) |
| 7 | embeddinggemma-300m (512 tokens, classification prefix) instead of stored | P@12 0.389→0.611 (3/3), AUC 0.674→0.710 | P@12 0.688→0.594 (worse 6/8), AUC 0.769→0.794 (p=0.011, 7/8) | no |
| 8 | stored + gemma side by side (50/50) | P@12 →0.500, AUC →0.706 (3/3, p=0.003) | P@12 0.688→0.667 (n.s.), AUC →0.785 (8/8, p=0.002), downvotes 0.031→0.010 | close |
| 9 | stored + gemma, 30% gemma (70%: small AUC 0.716) | AUC 0.683 | P@12 0.656, AUC 0.782 | no |
| 10 | step 8 retuned: C=4 (γ 0.015/0.06, blend 0.5 tested) | C=4 and blend 0.5 ≥ step 8 | **P@12 0.688→0.708 (4/8 better, n.s.), AUC 0.769→0.791 (8/8, p=0.002)**, downvotes 0.031→0.010 | yes |

Step 10 attribution: C=4 on stored alone gives P@12 0.698, AUC 0.774
(+0.005); adding gemma at C=4 adds AUC +0.016 (8/8, p=0.004). On the
fresh votes (below) step 10 gets AUC 0.885 ± 0.07 vs 0.859 and 9 vs 10
upvotes in the top 12 (one card; noise). Shipping it needs gemma
embeddings for every candidate on the VPS (not measured there; 0.45-0.6
s/story on the laptop iGPU). The small test predicts AUC direction but
not the top 12 (step 7: +0.22 small, -0.09 full).

Best on stored embeddings (no re-encoding), `prodlr[svm_c=2.0;lr_weight=0.3]`, vs production: top-12
upvotes 7.50→8.25 of 12 (p=0.20), AUC 0.726→0.769 (p=0.034), downvotes
0.62→0.38 and neutral 3.9→3.4 per top 12, discovery upvotes 0.25→1.25,
non-HN upvotes 1.25→2.75. Caveats: the small test misled once (step 3:
~1 story per fold at top 12); all steps share the development folds; the
2026-09-25 C=4 + blend challenger failed on the newest votes and hand
orderings, so votes after 2026-09-25 (a fresh snapshot) are the real test.

Fresh votes: user 1 stopped on 2026-09-24; the user's new default profile
is user 151 (363 votes, 2026-09-24 to 2026-09-28; 135 stories voted by
both). Snapshot `snapshot-20260928.db` (read-only VPS copy);
`scripts/merge_profiles_snapshot.py` builds a copy with user 900001 = 151's
votes plus user 1's on other stories. `--holdout-after 1790380800` trains on
votes before 2026-09-26 and tests on 151's 161 later votes (44 up):

| Trained on | Ranker | Upvotes in top 12 | AUC ±95% |
|---|---|---|---|
| 151 only (live today) | production | 8 | 0.844 ± 0.08 |
| 151 only | C=2 + blend 0.3 | 9 | 0.862 |
| merged 1+151 | production | 7 | 0.837 |
| merged 1+151 | C=2 / C=2 + blend 0.3 | 10 / 10 | 0.852 / 0.859 |

Direction holds on unseen votes, within noise (one 12-card block). AUC is
higher than on the development folds (0.73-0.77); not a text leak: text
length, article presence and comment count alone give AUC 0.36-0.53.
The live profile trains on 363 votes, not 5,500: merging needs the user's OK.

Taste drift check (2026-09-28 snapshot): old votes stay useful. On the
135 stories both profiles voted on, the same person agreed with their
earlier vote 70% of the time (up to down: 2; down to up: 0). A logistic
model on the stored embeddings, trained on 600 of user 1's votes from
one month, predicts user 151's votes on the 228 unseen stories (48 up)
about equally well:

| Month voted | AUC |
|---|---|
| June | 0.83 |
| July | 0.83 |
| August | 0.75 |
| September | 0.81 |

All 5,189 votes give AUC 0.82 (95% CI 0.75-0.89); August and September
alone give 0.79. Every vote was cast between June and September 2026, so
this covers four months of taste; old stories are not the same as old
votes.

The earlier embedding screen (700 votes/class, 8 folds, same day) found
body + comments as two 512-token vectors best at top 12 (0.656 vs 0.594
full text at 512), within noise; bge-base, arctic-m-v2 and single-vector
variants (averaged, weighted, split budget, chunked comments) did not
beat it. On the laptop iGPU (OpenVINO f16, `--device gpu`) encoding is
~4x faster than CPU (bge-base 0.21 s/story, mxbai-large 1.05 s/story at
512 tokens); `scripts/bench_embed_gpu.py` times a model.

## Untuned embedding probe and 2026 models — 2026-09-28

The hill-climb eval compares new embeddings with a ranker (C, gamma, blend)
tuned on the stored ones, and against the tuned stored+gemma combo, so a
new model can lose on fit alone. `scripts/probe_embeddings.py` scores each
embedding file the same way, hyperparameters picked by inner CV per
embedding (logreg, RBF SVM, cosine kNN), plus a taste-free check: source
classification (macro-F1) and k-means V-measure on non-HN stories. `meta`
= source one-hot + log points/comments/length, no text. 900-story sample
(300/class), 5 folds, score P(up)-P(down):

| Embedding | best AUC up vs rest | up vs down | up vs neutral | up in top 12 |
|---|---|---|---|---|
| meta (no text) | 0.661 (svm) | 0.696 | 0.626 | 0.50 |
| stored mxbai-xsmall | 0.756 (svm) | 0.866 | 0.645 | 0.63 |
| embeddinggemma-300m | 0.767 (svm) | 0.859 | 0.674 | 0.67 |
| harrier-oss-v1-270m, no prefix | 0.752 (logreg/svm) | 0.850 | 0.654 | 0.63 |
| harrier-270m, classify prefix | 0.729 (logreg) | 0.838 | 0.621 | 0.58 |

Pairs (stored+X) and the remaining rows: `probe-s300*.log` in the saved
eval dir (below). Up vs down is easy (0.85+); up vs neutral is where every
embedding is weak (0.63-0.67), and metadata alone reaches 0.66 overall,
signal the SVM-on-embeddings ranker barely uses.

harrier-oss-v1 (Microsoft, 2026-03-30; MMTEB v2 66.5 / 69.0) is Gemma 3
(270m) and Qwen3 (0.6b). 270m on the iGPU gives NaN at f16:
`scripts/debug_fp16_overflow.py` traced RMSNorm-input Adds up to 2.7e7
(f16 max 65504). OpenVINO's `ACTIVATIONS_SCALE_FACTOR` of 8+ on GPU fixes
it (cos 1.0000 vs CPU f32); not yet wired into the encoder (it ran at f32,
0.65 s/story). 0.6b: f32 ran out of GPU resources; f16 passed the check,
then failed at 600/900 stories (batch 8); resumed with batch 2. Small
tuned-eval (3 folds) vs stored+gemma: 270m ties at best (AUC 0.708-0.726
vs 0.720, n.s.). Research shortlist (classification-heavy MTEB, fits the
UHD 620): jina-v5-text-nano (classification variant), Qwen3-Embedding-0.6B,
KaLM-mini-v2.5, mdbr-leaf-mt (cheap VPS option).

## TUI simplification — 2026-09-26

- Review (code-review, high) of `clients/tui/src`: a passive refresh could
  strand "Loading summary…" (it cancelled the prefetch the selection had
  joined); a failed `r` left a forced regeneration armed; reverse sort
  logged reversed list positions as impressions; 45s client timeout vs
  ~2 min server generation; post-vote loop refetched the feed every 1s for
  up to 30s. All fixed; reverse sort (`v`) removed.
- Refactor in 4 commits (2592706, 4274de9, ff85910, d1b4bea). State removed:
  prefetch queue/worker/`prefetching`/`prefetch_requests`/cache-miss set,
  `target`, `force_summary_id`, `pending`, `can_read` field + 3 timers + 1s
  interval. `app.py` 1680 -> 1625 lines (docstrings ate most line savings).
- Test speed: 54-60s -> ~39s at `-n 4`, ~25s at `-n 8`. Floor is Textual,
  ~0.5s CPU per app lifecycle (1.0s with the 5-section markdown fixture,
  0.86s at 3 sections). Fixed sleeps became `tests/_settle.py::settle`.
- Two undo-during-stale-deck tests had passed without reaching the
  `restored` path once votes stopped refetching; they now reload the stale
  deck and fail if that path is removed. Each new regression test was
  checked to fail without its fix.
- Live (read-only, `--prefetch-generate 0`, no votes): feed, prefetched
  summaries on j, sort switch, zoom and quit all worked against the VPS.
- Diagrams (today vs proposed): `/tmp/hn-tui-diagrams/` (not in git).

## Source yield review — 2026-09-26

Per source, all users (stories fetched in 30d / shown / % of shown upvoted).
HN baseline: 6031 / 1696 / 21%.

- Above HN: latent.space 43/69/59%, Slashdot 251/69/52%, r/digitalnomad
  160/69/47%, r/Bogleheads 159/67/37%.
- Dropped (0-13%): Tildes 515/74/6% (36 downvotes), r/awardtravel,
  r/programming, r/ProgrammingLanguages, r/Urbanism, r/selfhosted,
  r/dataengineering, r/CreditCards.
- Added (similar to the winners): understandingai, dwarkesh,
  pragmaticengineer, oneusefulthing, thezvi, The Register, r/expats,
  r/eupersonalfinance. Recheck their yield after a few weeks.
- Fetched but rarely shown: text is thin, so the ranker has little to go on.
  lobste.rs: 75/120 URLs are also HN stories and collapse into them.
  LWN (41 article 403s, subscriber pages) and Aeon (38 article 429s)
  keep only RSS snippets (~500-1000 chars vs 4-9k for Slashdot and
  latent.space). r/Compilers has text but no upvotes from anyone.
- Silent blogs are healthy feeds that post rarely (aphyr, jvns, eugeneyan,
  pudding, earlyretirementnow); huyenchip last posted 2025-01.
- Fetch errors 2026-09-23..25 were a VPS DNS outage (157 errors, all
  feeds, 4 empty HN live windows; LessWrong's 18 "timeouts" are all in it).
  Cause: all VPS egress, DNS included, goes through the Proton WireGuard
  tunnel (resolver 10.2.0.1, fallbacks 1.1.1.1/9.9.9.9 also via the tunnel),
  so fallback resolvers cannot help. Errors stopped after proton.conf's
  routing rules were rewritten on 2026-09-26 (system-setup, not this repo).
  An empty live window only skips that regen's HN score refresh; ranking
  reads stories from the DB, so the deck keeps its HN rows.
- Follow-ups (2026-09-27): LWN, lobste.rs, r/Compilers, huyenchip dropped.
  Regen now fetches article text for up to 30 RSS snippet rows per run
  (1167 of 1669 recent rows had none); the first run got 30/30. Per-source
  report: `scripts/source_yield_report.py --since YYYY-MM-DD`.
- Backfill 2026-09-27 (`scripts/backfill_rss_articles.py`, configured feeds
  only): article text for 464/478 RSS rows (plus 185 before the
  configured-feeds filter, mostly Tildes).
- LessWrong: upvote rate is flat across karma (<50: 15%, 50-99: 24%,
  100-199: 14%, 200+: 18%; n=16-39), so the karmaThreshold=20 feed stays.
  thezvi.substack.com shares no titles with LessWrong.
- Legacy source labels: 272 pre-June rows (tildes, digg, bare `rss`,
  reddit_*, ...) had no source category, or RSS instead of Reddit, in the
  ranker's one-hot. Relabelled with `scripts/relabel_legacy_sources.py`
  after an 8-fold heldout-feedback eval on a DB copy: composite 0.669 ->
  0.675 (MAP 0.484 -> 0.491, NDCG@40 0.527 -> 0.548, top-40 downvotes
  5.9% -> 5.3%; 5/8 folds up), i.e. no drop. Rollback list (id, old source):
  VPS `~/hn-rewrite/relabel_rollback_20260927.tsv`; reports
  `~/.local/state/hn-rerank-eval/relabel-*-20260927.json` on the VPS.
- Feed picks by user 1's past votes per URL domain (HN stories; overall
  up rate 32%): AI lab announcements lead (blog.google 26/33 up,
  anthropic.com 29/46, openai.com 29/49, mistral.ai 5/8). Their feeds were
  added, then removed at the user's request: no AI provider blogs. The Register, added the day before, had 4 up / 14 down of
  28 and was dropped; arstechnica.com (6/22) was not added. Query: feedback
  joined to stories, grouped by URL host.
- Same scoring over every configured feed (its own stories, plus HN
  stories from its host): dropped r/ChatGPTCoding (11 votes, 18% up, 36%
  down), blog.cloudflare.com (0 up of 3 own + 4 HN), github.blog (0/4 HN),
  pluralistic.net (0/5 own, 3 down), danluu.com (0/5 HN). Borderline, kept:
  r/transit (27% up / 36% down), r/MachineLearning (34% / 38%).
- No AI provider blogs (user preference, 2026-09-27): deepmind.google,
  research.google and huggingface.co feeds dropped too.
- Independent sites user 1 upvotes on HN, added 2026-09-27: dynomight.net
  (3/5 up), borretti.me (3/4), martinalderson.com (3/4), vickiboykis.com
  (3/4), e360.yale.edu (3/5), sciencedaily.com (5/9). Small samples; judge
  with `source_yield_report.py` after a few weeks. Skipped: newyorker
  (paywall), macrumors (3 down of 7), seangoedecke (2 down of 6).
- Reddit 429s ~150/day: mostly one retry, circuit opened once in 3 days,
  no feed failed. Cause: each regen refreshes every subreddit, and votes
  trigger a regen about every 20 minutes; only ~4 of 21 weekly-top feeds
  change per refresh.

Query: stories x interaction_events (impression) x feedback, grouped by
`stories.source`.

## Ranking-quality study — 2026-09-25

Setup: `eval_ranker_variants.py --candidate-pool heldout-feedback`, user 1
(5,189 votes: 1,707 up / 1,813 neutral / 1,669 down), temporal expanding
folds over the development 80%; newest 20% reserved. Composite = mean of
AUC(up vs rest), MAP, NDCG@12, NDCG@40, 1 − downvote share of top 40.
Reports: `~/.local/state/hn-rerank-eval/*-20260925.json` (private).

- Baselines (3 folds): production AUC 0.697 / MAP 0.428; up−down centroid
  0.715 / 0.424; random 0.52 / 0.27; HN gravity 0.39 / 0.22 (worse than
  random: upvotes lean away from high-points stories). Shuffled-label
  controls sit at AUC ≈ 0.50 for every scorer (5 seeds), incl. new variants.
- Plain logistic regression beat production on dev (AUC 0.78 vs 0.70) but
  the pre-declared confirmation on the newest 20% tied/lost (AUC 0.739 vs
  0.737, NDCG@12 0.645 vs 0.720, top-40 downvotes 12.5% vs 2.5%). Not adopted.
- Hill climb (8 folds, composite; production 0.669): SVM C 0.1→4 with
  γ 0.05 → 0.705 (8/8 folds); + percentile-rank blend with
  logreg P(up)−P(down) → 0.714. Flat/negative: knn_k, positive_cluster_k,
  training dedup, P(up)−P(down) for the SVM, kNN in the blend, HN
  points/comments features (`engagement_features_enabled`, opt-in, 0.635 at
  C=0.1). Confirmation second look (already used once, so contaminated):
  blend 0.717 vs production 0.715 — within noise.
- Embeddings (replay-only `--replay-embeddings`, 512 tokens, blend
  composite): mxbai-xsmall@4096 0.714, mxbai-xsmall@512 0.732, bge-base-
  en-v1.5@512 0.743 (C=4, γ=0.025, logreg weight 0.6; neighbours
  0.733–0.741). Shorter context does not hurt; a stronger model helps.
  Concatenated bge-base+mxbai@512 (each /√2): 0.752–0.754, 8/8 folds, best
  on every metric (AUC 0.804, MAP 0.58, NDCG@12 0.77, top-40 downvotes 2.2%);
  arctic-embed-m-v1.5@512 alone 0.747, arctic+bge+mxbai 0.746, nomic-embed-
  text-v1.5@512 ("classification: " prefix) 0.744; stored@4096+bge 0.723. Stronger
  embeddings cluster at 0.745–0.754 (differences within fold noise).
- Short context / titles (blend, dev): mxbai@128 0.699, @256 0.719, @512
  0.732, @4096 0.714; bge title-only 0.681, title+body concat 0.713 < body
  0.743. 512 tokens of body is the sweet spot; titles alone lose signal.
- Confirmation (newest 20%, third look, so contaminated; 3 blocks, ~100
  upvotes each): production 0.715 composite; blend on stored embeddings
  0.717; bge+mxbai blend 0.678 and arctic blend 0.662 despite AUC
  0.766/0.765 vs 0.737. Top-of-list metrics swing 0.18–0.82 per block
  (NDCG@12) and top-40 downvote gaps are 1–3 cards, so the newest block
  cannot separate candidates at the top; whole-list AUC favours the new
  embeddings in the two later blocks.
- Hand orderings (`calibrate_rankings.py`, stored mxbai embeddings,
  challenger = C=4/γ=0.05 + logreg weight 0.4; 8 batches of 5 stories on
  which the two rankers order every pair differently): production agrees
  with the user on 43/80 pairs (54%), challenger 37/80; batches 5–3. First
  3 batches 20–10 for production, next 5 batches 23–27: noise. Every
  ordered story sat in both rankers' top 5%, so they differ only in fine
  order at the top. Decision: keep production; the offline gain does not
  show up in the user's own judgement. Clean holdout: votes after
  2026-09-25.
- Score correction from the orderings (`scripts/fit_score_adjustment.py`:
  production logit + weighted points/comments/age/HN-source/challenger,
  pairwise logistic, leave-one-batch-out, C 0.1–1): production alone 43/80;
  +points 43/80; every other set worse (23–38/80, +all 29/80). Fitted
  weights are tiny and flip between folds: 8 batches carry no usable
  correction. Revisit only with several dozen batches.
- Tooling: `scripts/summarize_eval_report.py` (composite + fold wins),
  `scripts/encode_replay_embeddings.py`, `scripts/calibrate_rankings.py`
  (hand-order 5 disputed stories; pairwise agreement per ranker).

## TUI review and narrow-pane evidence (2026-09-24)

- `--server` bug: the argparse default made `explicit_server` always set,
  contradicting the WORKLOG intent ("mismatch guard only for explicit
  `--server`"). No effect on the user's own profile, which already uses
  `DEFAULT_SERVER`.
- Live 51-column pane (`work:3.2`): the hints `Static` (`width: auto`)
  consumed the whole footer, leaving `#status` (`1fr`) at 0 cells. Status and
  error messages were never visible in narrow mode.
- Half-box frame: `#reading-pane.has-story:focus-within` (id + class +
  pseudo) outranks `.narrow #reading-pane` (id + class), so its left border
  leaked into narrow mode.
- Test time is spread across Textual pilot tests with real `pilot.pause`
  calls; no single slow test (max 3.5 s). xdist `-n 4`: 90 s → 25 s.
- Screenshot rig: `/tmp/tui-shots/shots.py` (EditorialServer, no profile
  access). Pilot `save_screenshot` produces SVG; `google-chrome --headless=new
  --screenshot --window-size=<viewBox>` produces PNG.
- `hn` is an alias to the editable `clients/tui/.venv/bin/hn-rerank`, so the
  checkout is live without a reinstall.
- The earlier one-off TUI flake did not recur across roughly 8 full-suite
  runs today.

## TUI focus-visibility evidence (2026-09-24)

- Screenshot rig: `clients/tui` pilot harness (`export_screenshot` → SVG) +
  `google-chrome --headless --screenshot` → PNG; outputs in `/tmp/tui_*.png`
  (narrow/wide × headlines/sort/summary focus). Rig script removed after
  use; recreate from WORKLOG recipe if needed.
- Key results: orange pane frames only on focus; amber `#5A3A12` selected
  row (full `#FF914D` washed out metadata, `#3D2A17` too faint);
  `Select:focus` never fires (focus lands on inner `SelectCurrent`) — fixed
  with `:focus-within`; `outline` spills on 1-row widgets — permanent
  invisible border recolored on focus instead; `SelectCurrent` needs
  explicit `height: 3` once bordered (border-box squeezes text otherwise).
- `wmctrl -x -a Navigator.firefox` verified live for focus pull.
- Flaky TUI test observed once (fail → pass on rerun, same code); test name
  not captured — rerun full TUI suite before committing the UI batch.

## Client-side heavy work (2026-09-23)

- Compute is not the bottleneck anywhere: local ONNX embed of 68 stories =
  0.18s (3ms/story) on 8 CPU cores; server (4 cores) likewise trivial.
  Ranking/SVM cost is rounding error.
- Real server "heavy" = LLM TLDR API calls ($/rate-limit bound, Groq
  default; ~155 tldr journal lines/hr) + network I/O (CH/LW/Reddit).
  Feedback DB must stay server-side (multi-device + backup).
- Client: 8 cores/11GB, no GPU, ollama binary present but no daemon/models.
  A 3-8B q4 local model would fit RAM at ~10-30 tok/s (5-15s/summary),
  quality below Groq-70B/Mistral.
- Recommendation: (1) client-side TLDR via BYO Groq key with server
  fallback — removes all server quota pressure, trivial HTTP reuse;
  (2) local TUI disk cache for summaries; (3) ollama only as offline
  fallback experiment. Do NOT move ranking/embeddings client-side:
  no measurable gain, splits model/feedback source of truth.

## LessWrong score attribution — snapshot counterfactual

The primary profile's feed had 10 LessWrong stories at Recent Recommended
positions 1–10. An offline SQLite backup (private on VPS) contained 114 eligible
LessWrong candidates. `scripts/diagnose_source_scores.py` held training rows and
candidate embeddings fixed while altering only candidate source/length metadata:

| Candidate metadata | LW median score | Median paired change |
|---|---:|---:|
| Actual | 0.9587 | 0 |
| Source category changed to HN | 0.2892 | -0.4544 |
| Text length changed to HN median | 0.9704 | +0.0115 |
| Both changed | 0.9360 | -0.0193 |

HN median text length was 3,520 characters; LW median was 9,827.5. This exposes
strong nonlinear source/length interactions, not a client sorting bug or simply
an additive LessWrong boost. Source features group RSS together, not LessWrong
alone. Single-feature ablations can be out-of-distribution; these are model
sensitivities, not causal evidence of user preference or quality probabilities.
Do not tune against the consumed confirmation set. No ranker changes made.

## LessWrong concentration — model-driver attribution (snapshot-only)

`scripts/diagnose_rank_drivers.py` reuses the production-trained SVM/scaler
via the model cache (reconstruction fidelity: top-100 overlap 100/100, rank
correlation 1.0) on a disposable read-only snapshot for user 1.

- Tier math: with 5003 feedback (1671 up / 1569 down), alpha_2 = alpha_3 = 1,
  so final scores are 100% SVM decision. HN gravity and centroid similarity
  have exactly zero weight for this profile.
- Raw SVM margins: LW median +2.15 vs HN/other-RSS median -0.21.
- Neutralizing all 10 meta features collapses the LW-vs-HN gap (+2.36 → +0.01)
  and sinks mean LW rank 1758 → 4688. Zeroing all 384 embedding dims barely
  moves it (gap 2.35, rank 1859). The advantage lives in meta, not raw text.
- No single meta column explains it (per-column drops shift the gap only
  ~0.03–0.16): the RBF SVM learned a joint pattern, not one dominant feature.
- Training revealed preference matches: RSS up-rate 43.8% (highest), HN-live
  30.6% (lowest). Within RSS, upvoted items are ~3.6x longer (median 6033 vs
  1676 chars); HN shows no length split. Decided LW votes are 71% up (25/35).
- LW candidates combine all three learned signals: is_rss=1, very long
  (median 9827 chars), high similarity to upvoted content (0.70 vs 0.48–0.55).

Conclusion: the concentration is faithful personalization of recorded votes,
not a client bug or an additive source boost (the SVM cannot even distinguish
LW from other RSS — same one-hot bucket). Concentration rotates among RSS
sources as candidate sets change. Any diversity intervention is a product
decision requiring explicit authorization and an evidence plan; the
confirmation set is consumed and must not be tuned against.

## Authorized duplicate Import AI vote cleanup — 2026-09-22

- User explicitly chose cleanup of duplicate feedback records. Scope limited
  to the three previously identified Import AI 458/459/460 pairs for user 1.
- Deleted only older feedback rows for story IDs -9281630763986,
  -78711871060061, -129256259946967. Retained newer upvotes on
  -623350225, -851489639, -1662778741. Verified exact matching URL/action,
  newer timestamp, and no concurrent changes since backup before deletion.
- Full SQLite/WAL-consistent backup (quick_check ok), plus exact affected-row
  manifest, retained privately on VPS under
  `/home/dev/hn-rewrite/shared/feedback-cleanup-backups/20260922T201813Z/`.
  Files: `before.sqlite`, `affected-votes.json`. Restore individual rows if
  required, not the whole database over later feedback.
- Transaction verification: total feedback decreased by exactly 3; story
  count unchanged; all six article rows and all retained votes intact.
  Read-only post-check: jack-clark.net has 13 upvotes, no neutral/downvotes.
- Service restarted to clear cached profile/model state; active, dashboard
  200, bounded journal scan clean. Training-dedup experiment remains off.
  Earlier evaluation artifacts describe the pre-cleanup feedback snapshot.

## Reserved training-dedup confirmation — 2026-09-22

- Ran only production vs deduplicated, no tuning, isolated nice-19 single
  CPU-thread job. `--candidate-pool heldout-feedback --confirmation --folds 3
  --variants deduplicated`. Verified frozen configuration and confirmation
  boundary match the preceding development run. 332/332/331 held-out items,
  all retained after normalized-URL group isolation, 83/98/120 positives.
- Production/dedup means: NDCG@10 0.76029/0.71383; NDCG@40
  0.56584/0.54434; MAP 0.49923/0.49562. Per-fold NDCG@10:
  0.6620/0.5837, 0.8390/0.7779, 0.7799/0.7799. The development
  top-10 gain did not replicate. Keep dedup disabled; no production changes.
- This remains retrospective judged-only discrimination, not live retrieval
  quality. Three folds are not a significance guarantee, but these results
  do not justify rollout. The reserved period has now been used and must
  not be described as untouched or repeatedly tuned against.
- Report: `/home/d/.local/state/hn-rerank-eval/dedup-confirmation.json`;
  VPS `/tmp/hn-publication-eval/confirmation.json`. No eval job left running.

## Commit and deployment verification — 2026-09-22

- `f03c34e` pushed and deployed to VPS. Only pre-deploy remote change was
  server.py, byte-identical to the committed v12 fix (SHA-256
  `91fa1931ee4963ded153317f25ff8111e425d6564513d10b6c12a57449c0e510`).
  Preserved via named stash `pre-f03c34e-deploy-identical-detail-v12`; clean
  fast-forward then service restart. No stash dropped, no feedback modified.
- Dashboard HTTP 200; Import AI 473 returned cached v12 output (1030 chars);
  uncached story -2028198065 generated 505 chars via Muse Spark in ~13s.
  Both ok, neither stale/retryable. Bounded journal scans free of
  ERROR/Traceback/Exception. Deployed experiment flags both false.
- Restarted existing TUI pane work:1.1, default profile hash unchanged. SQL
  read-only verification found new ledger event: impression, story 49803863,
  position 0, recommended/recent, ranker_arm tui_observed. This verifies the
  real client-to-server insertion path, not just a mocked endpoint.
- Exact staged tree passed 795 backend tests; full client suite 63 passed,
  one Windows-only skip. Deferred embedding code and tests were excluded
  from staging/deployment and preserved locally. No live ranking experiment
  enabled. Documentation-only wrap-up follows the code commit.

## Five-seed controls and training deduplication — 2026-09-22

- User narrowed scope to eval controls + training deduplication only. No
  deployment. Deferred section-embedding and TUI-impression WIP preserved.
- Completed isolated single-thread/nice-19 VPS replay: 3 development folds,
  five independent shuffled-label seeds. Four already-started variants ran;
  no further publication tuning. Expected shuffled NDCG@10 averaged 0.3402.
  Production shuffled mean 0.3352 (seed means 0.2766–0.4024); publication
  0.3545 (0.2613–0.4498); dedup 0.3211 (0.2829–0.3863); combined 0.3725
  (0.2604–0.4822). Prior one-seed elevation is not consistently reproduced;
  five seeds do not prove absence of leakage. These are descriptive seed
  means, not independent-fold confidence intervals.
- Real production/dedup: NDCG@10 0.80537/0.85016, NDCG@40
  0.61111/0.61219, MAP 0.46820/0.46637. Dedup per-fold NDCG@10:
  0.7530/0.7975/1.0000 versus 0.8205/0.7949/0.8007. Gain concentrated in
  fold 3; mixed outcomes do not justify claiming a general improvement.
- Dedup uses latest vote per production-normalized URL (ID tie-break), only
  in training memory; original feedback rows remain intact. Disabled by
  default. Publication canonical URLs now also use production normalization.
- Artifact: `/home/d/.local/state/hn-rerank-eval/publication-multiseed.json`
  (VPS `/tmp/hn-publication-eval/multiseed.json`, report schema 4). Confirmation
  period untouched. Next scoped step: predeclared production-vs-dedup
  confirmation evaluation, no hyperparameter search.

## Group-isolated judged-feedback replay — 2026-09-22

- Added `--candidate-pool heldout-feedback` to retain past rated stories,
  with all three labels; current-pool evaluation remains a separate mode.
  Shared production URL normalization excludes training cross-posts and
  deduplicates test/candidates. Effective test counts: 657/661/662 from
  664/663/663 (ten overlapping/repeated test rows removed). All replay
  candidates judged; average 175 eligible positives per fold.
- Low-priority, single-thread isolated VPS run, 3 development folds plus
  shuffled-label control. No production changes. Replay report:
  `/home/d/.local/state/hn-rerank-eval/publication-replay.json` locally and
  `/tmp/hn-publication-eval/replay.json` on VPS (schema 2 artifact immediately
  before the new report contract was versioned to 3).
- Production vs affinity: NDCG@10 0.80537 / 0.80493;
  NDCG@40 0.61111 / 0.61074; MAP 0.46820 / 0.46231.
  No demonstrated benefit from enabling affinity. High levels reflect the
  judged-only, exposure-selected pool and cannot be compared to live-pool
  NDCG (~0.03). Stored content is current, not historical.
- Shuffled control: NDCG@10 0.27660 / 0.43605;
  NDCG@40 0.32250 / 0.41762; MAP 0.32604 / 0.35010.
  Affinity control is elevated: one seed/three folds does NOT establish
  leakage absence. Investigate with multiple permutations and null baselines
  before treating small differences as meaningful. Latest confirmation data
  remains untested. Semantic duplicates beyond normalized URL remain possible.
- New metrics include @10, coverage warnings, group-isolation counts; SVM
  Brier reporting disabled. See docs/RANKER_EVALUATION.md for mode semantics.

## Publication-affinity initial evaluation — 2026-09-22

- Ran production vs `publication_affinity` on a read-only SQLite backup in
  `/tmp/hn-publication-eval` on VPS; production source/config unchanged. One
  thread, nice 19, cached embeddings only. Command: `uv run python
  scripts/eval_ranker_variants.py --config eval.toml --user-id 1 --variants
  publication_affinity --folds 3 --output result.json` (via production venv,
  isolated source). 12,671 candidates, 3,981 development votes; latest 20%
  timestamp groups reserved, no confirmation or label-shuffle run.
- Mean raw NDCG@12: 0.01826 → 0; NDCG@40: 0.03135 → 0;
  MAP: 0.02133 → 0.01321. No evidence for enabling this implementation.
- Major limits: only 85/20/2 eligible positive judgments across folds;
  current candidate pool contains only 314/1664 saved positive stories.
  Recommended Recent mixed had ZERO judged cards across development folds,
  so this does not measure current newsletter recommendation quality.
- Separate exact normalized URL overlap audit found 3/2/0 test stories with
  same-article training history. Not yet a duplicate-group-clean evaluation.
  Identity excludes ambiguous shared hosts, so this is a lower-bound audit.
  Snapshot retrieval is retrospective, not historical exposure evaluation.
- Initial scoring finished but report writing failed because archived source
  had no Git metadata. Initialized an isolated source-only Git snapshot and
  reran successfully. Report: VPS `/tmp/hn-publication-eval/result.json`;
  local `/home/d/.local/state/hn-rerank-eval/publication-first.json`.
  Private report stays outside Git. Treat affinity Brier output as invalid:
  SVM decision softmax is uncalibrated, despite evaluator's generic label.
- Decision: keep feature disabled. Improve group isolation and held-out
  candidate coverage before tuning; retain the current production ranker.

## Import AI 473 TLDR drops the tail of a long newsletter — 2026-09-22

- Story `Import AI 473` (jack-clark.net, VPS `stories.id = -1607225291`) headlined
  three topics: US superintelligence strategy, brain chimeras, machine
  hermeneutics. The cached TLDR (824 chars, 4 bullets) covers the RAND strategy
  plus one chimera bullet; machine hermeneutics is absent entirely.
- Not a fetch bug for this story: stored `article_body` is 24,067 chars (under
  the 30k `ARTICLE_BODY_CHAR_LIMIT`) and contains all three topics (RAND @ch15,
  chimera @ch8150, hermeneutics @ch21382). The full body goes into the prompt.
- Structural cause is prompt/budget: `_section_budget()` caps output at
  "3-4 bullets, max 90 words" for ANY input over 5k chars, and
  `prompts/article_v4.txt` says "Summarize the article" with no
  cover-each-section instruction — so the model writes a lead-biased summary
  and the section starting 88% into the text is dropped.
- Caveats: jack-clark.net now serves a JS browser-check to curl, so the source
  page could not be re-verified; the stored text ends with a personal-dream
  coda that reads like a natural ending. The TLDR's freeform "### Summary"
  heading was model-chosen (freeform headings allowed since prompt v7).
- Fix needs a product call: `afafefc` deliberately targets one-screen summaries,
  which conflicts with covering every section of a 24k-char newsletter. Any
  prompt change bumps `TLDR_PROMPT_VERSION` and invalidates the whole
  `tldr_cache` — plan around the 120/hr uncached limit and the Mistral $10 cap.
- Read-only investigation: VPS SELECTs plus one curl; no code or data changed.

## Terminal reader — read mode gated on overflow — 2026-09-20

- Article + Discussion fit one screen for most stories, so the read mode is
  offered only where it is needed: Enter is enabled — and the footer hint
  shown — only while the summary overflows its pane. Enter expands the summary
  (full height below 100 columns; focus plus `j/k` scrolling above) and Escape
  leaves it. Fitting summaries keep the plain layout with Enter disabled and
  no hint. Read mode is frozen while active so the expanded pane cannot flip
  the state, and the overflow check re-runs after every render and resize.
- Verified: client 40 passed / 1 Windows-only skip; standalone copy with fresh
  deps (ruff 0.16.8, ty 0.0.82) clean; in-tree ruff/format/ty clean; narrow
  80×30 render inspected.

## Terminal reader / origin/main merge — 2026-09-20

- Merged the 98-commit backend line into the terminal-reader branch (`354acd6`).
  Resolutions: backend files and tests take `origin/main`; `clients/tui/**`
  keeps the branch side (the newer feed models the server imports); WORKLOG
  keeps origin/main's full history plus the TUI entries and drops a stray
  committed `>>>>>>> theirs` marker; ROADMAP restores §6; ARCHITECTURE adds the
  client package section; AGENTS.md stays origin/main's long form.
- `uv.lock` then regenerated to include the `clients/tui` workspace member
  (`6a0dc99`); the origin/main lock predated the member.
- Verified: backend 769 passed with `HN_ONNX_MODEL_DIR`; client 38 passed /
  1 Windows-only skip; Ruff and ty clean.
- Open follow-ups (config path pinning, stale env-var docs, superseded WORKLOG
  vacuum and AGENTS slimming, stale remote branch, VPS pull + restart) are
  listed in [STATUS.md](STATUS.md).
- Rollback: `backup/tui-pre-merge` marks the pre-merge tip; the merge commit's
  first parent is the terminal-reader line and the second is `origin/main`
  `97e1e25`.

## Editorial terminal client — 2026-09-20 review

- Live review against the VPS deployment (throwaway profile) found and fixed
  four defects: selector focus could strand every keybinding below 100 columns;
  Escape out of help left the Shortcuts text in the reading pane; a feed
  refresh with an unchanged selection left the heading on stale points and
  comments; and `text-align` never overrode Textual's centered H1 content
  alignment. The footer now labels the vote keys (`1 up · 2 neutral · 3 down`)
  and Enter toggled back to headlines (read mode is now gated on overflow).
- Article + Discussion now fit one screen. Companion server work landed on
  `origin/main` as `detail-v9`: section budgets target 45/70/90 words and a
  deterministic cap keeps at most four bullets and one `####` heading per
  section. The reader renders Markdown blocks without per-block margins; the
  Textual pilot measured 8/8 sampled recommended stories inside the 27-line
  reading pane at 145×38 (4/8 before).
- Verified: client suite 38 passed / 1 Windows-only skip; Ruff and ty clean;
  live terminal sessions at 145×38 and 80×30 exercised filters, reading,
  voting/undo/neutral, help, empty and failure states. New tests cover the
  help exit, refreshed-heading sync, footer vote labels, Enter toggle,
  heading alignment and narrow-layout selector escape.

## Editorial terminal client — 2026-09-17 handoff refresh

- Current state: full visual polish is implemented on `feat/terminal-client` in
  the working tree, uncommitted, on top of the reading-focus cue: selection
  contrast, themed scrollbars, docked footer with per-filter counts and `✗`
  errors, age guard, headed empty/error notices (`show_failure()`/`last_error`
  with recovery), setup card, 100-col reading cap with headline hairline, and
  vote toast. ROADMAP §6 polish is fully done.
- Agent-tested: client suite 32 passed / 1 Windows-only skip; Ruff and ty clean;
  offline SVG renders inspected for populated/empty/error/setup states.
- Remaining: native terminal visual confirmation and credential-dependent PyPI
  publication.

## Editorial terminal client — 2026-09-13

- Implemented: charcoal/ivory theme, restrained orange tabs/focus/selection,
  explicit `>` marker, domain/points/comments/age metadata, summary heading,
  Markdown spacing, thin pane divider and context-sensitive shortcut footer.
  Setup separates import and creation. Existing credential/API code is unchanged.
- Agent-tested: 25 client tests passed / 1 Windows-only skip; backend 541 passed /
  1 skipped. Ruff and ty clean. Rebuilt wheel version and headless setup startup
  passed outside the checkout. New tests cover 60/80/100/140 columns, filters,
  focus, preserved reading scroll and selection, setup validation and empty/errors.
- Visual evidence: offline SVG renders under /tmp/hn-editorial-*.svg inspected
  for populated/code, empty/error and setup states, including a long headline.
  NO_COLOR=1 is set in the agent shell; color preview explicitly unsets it.
  OptionList vertical component padding clips metadata in Textual 8, so options
  use horizontal padding and a literal marker instead of a decorative border.
- Preview: `env -u NO_COLOR TERM=xterm-256color COLORTERM=truecolor uv run python
  -m clients.tui.tests.preview --headless`; omit --headless for a synthetic terminal
  exercise. No production profile or server is used. Graphical windows launched,
  but a successful capture of the actual preview window was not obtained.
- Resolved: guard fix is included in the pushed branch; latest `tui.yml` run
  (2026-09-14, `34805118943`) passed. Native terminal visual confirmation and
  credential-dependent PyPI publication remain pending.
  Earlier CI below covers the old revision.
- Rollback: editorial work and the setup guard are committed on
  `origin/feat/terminal-client`; revert with `git revert` if needed.
  Existing unrelated untracked files were preserved.
  Wheel/sdist can be rebuilt from the previous source if needed.
- CI 2026-09-13: run `34739788455` failed all OS on ruff 0.16 I001 import order
  (local ruff was 0.15.17 and passed); fixed in `45ca54f`. Rerun `34739910637`:
  ubuntu/macos green, Windows red on
  `test_setup.py::test_import_validates_then_persists_and_relaunches` —
  `on_option_list_option_highlighted` queried `#headlines` while the Setup
  screen was active (straggler highlight during teardown). Guarded in `763cc20`
  (`setting_up`/existence early return); local client suite green.

## Laptop project and package — 2026-09-13

- Agent-tested: checkout moved from `/home/d/hn-rerank-v2` to `/home/d/hn-rerank`;
  one worktree, no other cwd users at move time, original untracked `.opencode/`,
  `.playwright-mcp/`, `docs/MANUAL_TESTING.md` preserved. GitHub name remains
  `djv/hn-rerank-v2`. uv console scripts were reinstalled and verified at the new path.
- Agent-tested: `HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model uv run
  pytest tests/ -n 4`: 541 passed, 1 skipped. Real test model copied from the VPS;
  no production DB copied or used by development tests. Client: 16 headless tests
  passed, 1 Windows-only test skipped locally; Ruff and ty clean. A laptop PTY exercised navigation, feedback, undo,
  reading/help and clean exit at 120 columns; headless tests also cover 80 columns.
- Agent-tested: Hatchling wheel and sdist in `dist/`; installed wheel launched via
  `uvx --from` from `/tmp`. Isolated headless startup and config round trip passed;
  numpy, sklearn, ONNX Runtime and Flask absent. Default runtime dependencies are
  Textual, HTTPX, platformdirs and their transitive dependencies.
- Agent-tested: source revision `12f7b08` on `feat/terminal-client` passed Linux,
  macOS and Windows CI, including standalone build/install, headless startup and
  Windows owner-only credential ACL verification:
  https://github.com/djv/hn-rerank-v2/actions/runs/34738793586
- Configured and read back: updated only the project trust path in
  `/home/d/.codex/config.toml`, preserving all other TOML values. Private rollback
  copy: `/home/d/.local/state/hn-rerank-rename/codex-config-before.toml`.
  No active session restarted. Generated uv environment has no old-path matches.
- Publication pending: PyPI name lookup returned 404, which does not reserve the
  name. No local publishing token or OIDC identity configured; requested account
  setup through the question popup. No package uploaded and `uvx hn-rerank` from
  the public index is not yet verified. See [release instructions](docs/TUI_RELEASE.md).

## VPS feed deployment — 2026-09-13

- Agent-tested: live service `hn_rewrite.service` runs in `/home/dev/hn-rewrite/main`.
  Its baseline was `4ec6bc6`, ahead of its origin and with uncommitted database,
  ranking, enrichment, tests and documentation changes. Those changes were preserved.
- Agent-tested: ported only feed rendering/API/models and the new API test, using
  source-hash checks and a bounded remote flock. No other agent process had that cwd;
  the flock cannot arbitrate controllers that do not participate. Server policy,
  model configuration, deployment directory and production schema were untouched.
- The newer VPS Explore filter shuffles; the port preserves shuffled feed orders.
  Its stale HTML helper also updates latest-version metadata and now preserves the
  feed attachment. [Exact scoped patch](docs/TUI_VPS_PATCH.patch).
- Agent-tested: copied VPS source was tested on the laptop with temporary databases:
  220 API/server tests passed. VPS full suite before/after: 767/768 passed;
  Ruff and ty passed after deployment. Service restarted successfully.
- Agent-tested: live HTTPS `/hn/` integration created a dedicated profile, returned
  68 stories and a 3592-character summary, accepted an upvote, reached its ranking
  target with the story excluded, then cleared the vote and verified zero feedback.
  Importing the same profile preserved identity. The private test credential is in
  `/home/d/.local/state/hn-rerank-test/profile.json`; never commit its contents.
  A bounded post-restart journal scan found no ERROR/Traceback/Exception lines.
- Rollback: `/home/dev/hn-rewrite/shared/deploy-backups/20260913-tui-feed/` contains
  original `server.py`, `pipeline/render.py` and hash manifest. Recheck concurrent
  changes before restoring those two files, then restart only `hn_rewrite.service`.
  The new dependency-free model/test files may remain inert on rollback. No DB
  rollback is needed. Laptop rename rollback is in the release instructions.

## 2026-09-24 reader-plan coverage inspection (read-only, VPS DB)

Replicated docs/source-review.md 30-day content check: asterisk 0/4,
Hugging Face 0/23, OCaml.org 3/27, Lobsters 16/123 stories with >=300 chars
self_text/article_body or any top_comments. All-time fully-empty counts are
small (HF 12/104, OCaml 6/103, Lobsters/Asterisk 0). article_fetch_failures
holds exactly one row across all four sources (a transient Lobsters
ConnectTimeout) — the fetcher is not failing on these rows, it is not
attempting them (or attempts predate the table).

Nuance for the doc's starvation theory: is_summarizable() accepts ANY
nonempty text, so HF rows with 36–62 char RSS snippets pass the filter and
reach ranking; only fully-empty rows are filtered pre-rank. Sampled HF rows
have short self_text, zero article_body, zero comments, fetchable public blog
URLs, no failure rows. Whether they miss the 50-slot article-fetch budget on
rank or another gate is unproven — no scheduling change made.

Summary spot-check (mixed set): discussion-only HN thread 49555155 renders
6 bolded bullets and explicitly notes thin/author-only signal (~190 words,
appropriately under target, no padding). Combined Reddit case verified
previously (bold in 4/4 Discussion bullets). Latent Space article-only
overshoot (414 words vs 240 target) already recorded in STATUS.md. No new
prompt tuning demonstrated; prompts untouched.

## 2026-09-24 TLDR zoom

The existing read action was gated on summary overflow, and only the narrow
CSS hid headlines. Zoom now accepts any selected story and hides headlines
at every width. The follow-up reading-width refinement centers the zoomed
pane and retains the 100-column cap; narrow terminals use their full width.
Both Enter and Escape restore list focus. Tests verify dimensions, focus, selection and
footer controls for short/long content at 51, 80 and 140 columns.

Validation: TUI 116 passed, 1 skipped; backend 821 passed using the local
`HN_ONNX_MODEL_DIR=/home/d/.cache/hn-rerank/onnx_model` override. The initial
backend run had 18 setup errors because its default model path points at
the VPS. Ruff/format/ty clean. User independently reported fullscreen works.
No backend deployment or database changes were needed.

## 2026-09-24 Vote-to-advance footer

Adding `→ next story` exposed a cramped wide footer at 100 columns: status
shrunk to seven cells. Stacking status above shortcuts at all widths keeps
counts readable. At 51 columns the shortcuts wrap onto two lines.

## 2026-09-24 Navigation-aware prefetch

Previous default: ten cache-only lookups in sequence after foreground completion.
Missing server summaries were not generated until selection. New rolling window:
20 forward cache targets, previous three, other-sort first three; generation for
next three plus previous/other-sort neighbors, four background requests maximum.
The selected story reuses an in-flight request. Existing server cache/generation
endpoints suffice. Tests verify navigation readiness and one request per generated
selection; live provider latency can still exceed rapid navigation.

Live verification after relaunch: 44 stories in `work:3.1`. A bounded VPS
journal read showed concurrent cache hits/misses and successful generation
(2026-09-24 20:57:05 UTC, story 49525378, HTTP 200, 9280 ms total). This
confirms live cache access and generation, not guaranteed readiness for every
possible rapid navigation path. No server mutation/restart was performed.

## 2026-09-24 Pre-commit review

Reproduced and fixed one issue: at 51 columns in zoom mode, a three-line
error plus the two-line legend overran the footer by one row. A new layout
test failed with `hints.bottom=38`, `footer.bottom=37`; removing the overall
footer height cap makes all rows visible. Status remains limited to three
lines. Reviewed navigation prefetch bounds, cache-only opt-out, cancellation,
rate-limit handling, request reuse and documentation; no other blocker found.

## 2026-09-24 Post-push CI status

Head `71de3b4c3049d36d22a15b3745bfd1dfac1e3b81` is pushed to `origin/main`.
Backend CI succeeded: https://github.com/djv/hn-rerank-v2/actions/runs/36059751527
Terminal client failed: https://github.com/djv/hn-rerank-v2/actions/runs/36059751541

All terminal-client test jobs passed before lint: Windows 122 passed; Linux
and macOS 121 passed/1 skipped. Each Ruff step reports eight errors, including
I001 imports and SIM102 nested conditions (`tests/test_client.py:566`).
`.github/workflows/tui.yml` copies the package outside the backend workspace
before running `uv run ruff check src tests`; reproduce that isolation for
the next fix. No code changes or CI reruns were performed for this status save.

## 2026-09-24 Quota doubling evidence (240/hr, deployed)

Pre-restart journal showed `tldr_detail ... result=stale_fallback
reason=quota_denied` under the old 120/hr limits. After pulling `247f0c1`
to the VPS `main` worktree and restarting `hn_rewrite.service` (22:04:31
UTC, new PID 597572), host `config.toml` confirms 240/240 and zero
`quota_denied` lines appear post-restart; dashboard returns 200.

VPS layout for future deploys: `/home/dev/hn-rewrite` repo with worktrees
`main` (live, `WorkingDirectory` of `hn_rewrite.service`, runs
`uv run python server.py`) and `hn-rewrite-explore` (separate branch,
left alone). Deploy = push, `git pull --ff-only` (must be clean),
`systemctl --user restart hn_rewrite.service`, smoke test dashboard +
`api/ranking-ready`, scan journal for `quota_denied`/tracebacks.

Backend suite note: 803 passed with 18 `test_pipeline.py` errors that are
pre-existing/environmental (HF repo-id validation); verified identical
with the quota change stashed, so they do not gate config-only changes.
