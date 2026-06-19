# Recommender System Features Research

Research conducted in support of the hn-rewrite feature ablation study. Goal: identify concrete features used by production recommender systems that could improve the current 395-dim SVM ranking pipeline.

## 1. Production Systems Surveyed

| System | Core approach | Key feature families |
|---|---|---|
| **Twitter/X Heavy Ranker** | Weighted sum of 10 P(engagement); MaskNet parallel architecture | Per-user/author/topic aggregates over 30min–50d windows; EWMA, days_since_last; text micro-features |
| **Reddit** (open source `_sorts.pyx`) | `hot`: log10(score)+age/45000s; `confidence`: Wilson lower bound; `controversy`: magnitude^balance | Log score, linear time drift, Wilson score, vote balance |
| **Hacker News** | `score/(age_hours+2)^1.8` (power-law gravity≈1.8) | Power-law age decay; +2 smoothing |
| **YouTube** (Covington 2016) | Two-tower DNN (candidate gen + ranking) | User watch history embedding, "example age" (essential), video embedding, source IDs |
| **Netflix Prize** (BellKor 2009) | SVD++ + temporal dynamics + neighbor models + implicit feedback | Latent factors, user bias over time, item-item kNN, implicit feedback |
| **Google News** (Das 2007) | MinHash/LSH + story clustering + CF | Click history → MinHash → collaborative; TF-IDF story clustering; popularity; freshness |
| **Spotify** | Audio features (13 bounded descriptors) + CF + session-based | acousticness, energy, danceability, valence, tempo + co-listening |

## 2. Universal Patterns (every system uses these)

### 2.1 Age/time decay is non-negotiable
- YouTube (Covington 2016): "example age" called out as **essential** — without it the model favors old evergreen content
- HN: power-law `score/(age_hours+2)^1.8`
- Reddit `hot`: linear `age/45000` seconds (≈12.5h per unit)
- Twitter: decayed counts at 30min/1d/3d/50d windows
- Evan Miller's "fixed" Reddit: `ln(score) − λt` (Poisson-visitor model)

### 2.2 Two-vector user model (long-term + short-term)
- YouTube: watch history embedding (long-term) + recent sequence (short-term)
- Twitter: `user_aggregate` 50-day (long) + `tweet_aggregate` 30-min (real-time)
- Netflix: `p_u(t)` user-factor drift over time
- News-rec literature: global interest + local subgraphs

### 2.3 Negative signal weighted heavily
Twitter engagement weights (Apr 5 2023, from `the-algorithm-ml` README):
```
reply_engaged_by_author: +75.0
good_profile_click:      +12.0
reply:                   +13.5
good_click:              +11.0
good_click_v2 (dwell≥2m):+10.0
retweet:                 +1.0
fav:                     +0.5
video_playback50:        +0.005
negative_feedback:       -74.0
report_tweet:            -369.0
```
The negative-feedback and report signals carry **massive** negative weight — far larger than positive engagement. Author-reciprocal engagement (reply engaged by author) is the most valued positive signal.

### 2.4 Multi-window aggregates
Twitter taxonomy: 30-min (real-time), 1-day, 2-day, 3-day, 50-day (long-term).
Aggregation operations: `count, sum, sumsq, ewma, days_since_last, elapsed_days, non_zero_days, sparse_top1/top2/mean/sum/nonzero`.

### 2.5 Metadata as first-class (not afterthought)
arXiv:2112.14370 ("On the Overlooked Significance of Underutilized Contextual Features"): a **purposefully simple** contextual module (CTR, popularity, freshness) **beats** sophisticated deep content-only models by a large margin. Over-engineering contextual features *hurts*.

## 3. Concrete Features Computable from HN Data (no new collection)

### 3.1 Tier 1 — free, no schema change, ship first

| Feature | Formula | Precedent |
|---|---|---|
| HN power-law decay | `score/(age_h+2)^1.8` | HN baseline (current system uses linear `score/(h+1)`) |
| Exponential decay | `score·exp(−λ·age_h)` | Evan Miller; Twitter decayed counts |
| Log score | `log10(max(score,1))` | Reddit `hot` |
| Comment/score ratio | `comment_count/max(score,1)` | Reddit `controversy` proxy (no public downvotes) |
| Time-decayed user profile | `EWMA(upvoted_embs, τ=30d)` using `vote_times` | Twitter realgraph ewma; Netflix `p_u(t)` |
| Wilson score on user feedback | `wilson(up, up+down, 0.8)` per cluster | Reddit `confidence` |
| Domain category | `urlparse(url).netloc` → github/arxiv/news/blog | YouTube source IDs; Twitter has_link/has_news |
| Title lexical | `has_question`, `title_len`, `caps_ratio`, `is_self_post` | Twitter tweetsource features |
| Hour/day vote context | P(up\|hour), P(up\|dow) from `vote_times` | Twitter user_request_context |
| Negative profile vector | `mean(downvoted_embs)` + `sim_to_negatives` | Twitter negative_feedback (−74 weight) |
| Bayesian-shrunk score | `(score + C·m)/(1 + C)` | Evan Miller; prevents low-vote flukes |
| Velocity | `score / max(age_h, 0.1)` | Twitter tweet_aggregate momentum |
| Comment velocity | `comment_count / max(age_h, 0.1)` | Twitter conversational_count |

### 3.2 Tier 2 — additive schema (one Algolia field + columns)

| Feature | Precedent |
|---|---|
| Persist `author` from `item["author"]` → author karma, per-author history, author productivity | Twitter `author_aggregate` (largest feature block), `user_author_aggregate` |
| Comment-quality aggregates (currently computed then discarded): `max_comment_points`, `mean_comment_points`, `max_depth`, `num_top_level_comments`, `comment_points_top1_share`, `mean_comment_len` | Reddit qa length bonus; PG "stupid comments are short" |
| `score_at_fetch` column → real vote-velocity derivative | Twitter tweet_aggregate real-time |
| Author karma (from `hn.algolia.com/api/v1/users/{author}`) | Twitter `user_rep`, HN karma as long-run reputation |

### 3.3 Tier 3 — offline computation

| Feature | Precedent |
|---|---|
| Title-only + comment-only embeddings + agreement cosines | Twitter `text_score`/`blender_score`; linkbait detection |
| Topic clustering (k-means on embeddings) → `cluster_id`, `topic_up_frac`, `topic_burst_score` | Google News story clustering; Twitter topic_aggregate |
| Candidate-aware attention: `softmax(cos(candidate, history_embs))` weighted profile | News-rec literature (Qi et al.) |
| MinHash collaborative filtering over voted-story sets | Google News (Das 2007) |
| Author-topic affinity | Twitter `author_topic_aggregate` |

## 4. Twitter Heavy Ranker — Detailed Feature Manifest

From `twitter/the-algorithm-ml/projects/home/recap/FEATURES.md` (60KB feature manifest):

| Feature family | Description | HN-relevance |
|---|---|---|
| `user_aggregate` | Rolling counts/sums over user engagements, 50-day + 30-min windows. Includes `is_favorited`, `is_replied`, `is_retweeted`, `is_clicked`, `is_dwelled`, dwell-time buckets (8/15/25/30s) | HIGH — user's up/down history over time windows |
| `author_aggregate` | Same but keyed by tweet author; 30-min "real-time" track of author engagement | LOW (no author data) |
| `user_author_aggregate` | Per (user, author) pair engagement counts, 50-day | LOW (no author) |
| `author-topic_aggregate` | Per (author, topic) 50-day engagement counts | LOW |
| `user_topic_aggregate` / `user_inferred_topic_aggregate` | Per (user, topic) engagement counts; `sparse_top1/top2/mean/nonzero` summaries | HIGH — topic≈embedding cluster |
| `topic_aggregate` | Per-topic real-time (30min/24h/3day) engagement counts incl. `is_not_interested_in_topic`, `is_see_fewer`, `is_unfollow_topic` | MED — cluster-level freshness/negativity |
| `user_request_context_aggregate` | Aggregates keyed by **day-of-week** and **hour-of-day** | HIGH — compute from `vote_times` |
| `user_engager_aggregate` | Counts of other users who engaged with the same tweets the user engaged with; `sparse_top1/top2` | MED — co-voters on HN stories |
| `realgraph` | User↔author edge weights: `num_favorites, num_retweets, num_direct_messages, num_follow, num_mutual_follow, num_profile_views, total_dwell_time, num_mutes, num_blocks`, each with `ewma / mean / days_since_last / non_zero_days / variance` | MED — ewma/days_since_last apply to user-story feedback |
| `tweet_aggregate (real_time)` | Per-tweet rolling 30-min counts of every engagement type + dwell buckets | HIGH — story-level vote velocity |
| `tweetfeature` | `fav_count, reply_count, retweet_count, conversational_count, has_link/has_image/has_video/has_news, num_hashtags, is_reply, is_retweet, is_sensitive, user_rep, from_mutual_follow, from_verified_account, is_author_bot/new/spam, language` | HIGH — score, comment_count, url, source |
| `tweetsource` | `text.length, text.length_type, text.has_question, text.num_caps, text.num_newlines, media.aspect_ratio, video_duration, num_tags` | MED — title length, has_question, caps ratio |
| `user_state` | `is_user_heavy_tweeter / medium_tweeter / light / new / heavy_non_tweeter` | MED — classify voter as heavy/light/new |
| `dwell-time buckets` | `is_tweet_detail_dwelled_8/15/25/30_sec`, `is_profile_dwelled_*`, `is_fullscreen_video_dwelled_5/10/20/30_sec` | MED — map to "expanded comments / read time" |
| `negative_feedback_union` | Union of `dont_like, block, mute, report, see_fewer, not_interested_in_topic` | HIGH — user's downvotes direct analog |

## 5. Reddit Sorting Algorithms (source: `_sorts.pyx`)

### `hot` (front-page ranking)
```
s       = ups - downs
order   = log10(max(abs(s), 1))
sign    = +1 if s>0 else -1 if s<0 else 0
seconds = epoch_seconds(date) - 1134028003     # epoch offset
hot     = sign * order + seconds / 45000
```
Vote contribution is **log10** (diminishing returns); time contribution is **linear in seconds ÷ 45000** (≈12.5h per unit). Newer items get a continuous boost; high vote counts saturate.

### `confidence` ("best" — Wilson score lower bound)
```
n     = ups + downs;  if n==0: return 0
z     = 1.281551565545      # z for 80% confidence
p     = ups / n
left  = p + z²/(2n)
right = z * sqrt(p(1-p)/n + z²/(4n²))
under = 1 + z²/n
conf  = (left - right) / under
```
Lower bound of the Wilson score interval at 80% confidence. Penalizes small-sample high-ratio items. **The canonical "how not to sort by average rating" solution.**

### `controversy`
```
if downs<=0 or ups<=0: return 0
magnitude = ups + downs
balance   = min(downs/ups, ups/downs)     # ∈ (0,1]
controversy = magnitude ** balance
```
High when both up & down are large AND near-balanced.

### `qa` (Q&A threads)
```
question_score = confidence(q_ups, q_downs)
best_answer    = max confidence over OP child answers
length_mod     = log10(question_length + answer_length)
qa = (question_score + answer_score) + length_mod/5
```
Wilson score + log length bonus.

## 6. Feature Ablation Findings (in-repo)

5-fold stratified CV, 1711 feedback, 4048 candidates, MMR@0.50, embeddings preserved raw (StandardScaler only on metadata columns):

| Feature set | dims | NDCG@40 | hit@40 | MAP | med_rank | Brier |
|---|---|---|---|---|---|---|
| full (384+8) | 392 | 0.9787 | 0.1151 | 0.2563 | **87.7** | **0.1491** |
| emb_only (384) | 384 | 0.4021 | 0.0544 | 0.0391 | 483.1 | 0.2332 |
| emb+pers (384+4) | 388 | 0.5911 | 0.0789 | 0.0776 | 282.4 | 0.2241 |
| meta_only (8) | 8 | 0.9787 | 0.1151 | 0.2563 | 88.8 | 0.1502 |
| pers_only (4) | 4 | 0.5759 | 0.0760 | 0.0764 | 324.1 | 0.2280 |
| meta_no_pers (4) | 4 | 0.9931 | 0.1169 | 0.2627 | 134.0 | 0.1671 |

**Key insight**: embeddings contribute to median rank (87.7 vs 134.0, 35% improvement) even though they're tied or slightly worse on NDCG@40. The full model's embeddings earn their place by pulling upvotable stories toward the top of the full ranking, which matters for the badge surfacing passes that pull from positions 40–200.

## 7. Highest-Leverage Recommendations (synthesis)

1. **Treat metadata as first-class** — `score`, `comment_count`, `age_hours` should be explicit features, not just ranking priors. (AGENTS.md caution re: leakage still applies — watch for target leakage in train/test splits.)
2. **Two-vector user model** — long-term profile (mean of all upvoted embs) + short-term (EWMA / attention over recent votes).
3. **Age is non-negotiable** — keep `age_decay` even inside a learned model.
4. **Negative signal is high-value** — build a `negative_vec` and `sim_to_negatives`; weight user downvotes strongly.
5. **Wilson score for small-sample confidence** — use Reddit's `confidence` on per-user-per-cluster up/down counts.
6. **MinHash for cheap collaborative filtering** — Google News approach fits local-first constraints.
7. **Source/domain as embedded categorical** — YouTube embeds topic/source IDs; embed HN source domains.
8. **Window taxonomy** — adopt Twitter's 30min / 1d / 3d / 50d windows for user aggregates; compute `days_since_last` and `ewma` per cluster.
9. **Candidate-aware attention** — news-rec literature shows weighting user history by similarity-to-candidate beats a static profile.
10. **Diversity guard** — coverage-attentive / debiasing work warns of topic monoculture.

## 8. Features NOT available without new data collection

- Author quality / author-aggregates (Twitter `author_aggregate`, `realgraph`) — requires storing `by` field
- Dwell time / read time (Twitter dwell buckets, YouTube) — requires client-side timing
- Co-engager social graph / two-hop (Twitter) — requires shared vote visibility
- Per-comment structure (only top-24 baked into text currently)

## Caveat (per AGENTS.md)

Several news-rec systems report AUC ~0.57–0.67 on MIND; "naive contextual beats deep" (arXiv:2112.14370) is a healthy reminder that beating the HN baseline by a large margin is unlikely and high measured gains should be treated with suspicion (leakage/saturation). NDCG@40 > 0.40 in this project's eval is likely an artifact of the eval methodology (test feedback stories in candidate pool), not a true generalization metric.
