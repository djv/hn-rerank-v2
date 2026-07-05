# Code Review, Invariants, and Test Plan

**Scope:** `server.py`, `pipeline.py`, `database.py`, `templates/index.html`,
and the existing test suite (`tests/test_pipeline.py`, `test_server.py`,
`test_database.py`, `test_fetch.py`, `test_eval.py`).

**Baseline:** 84 passed, 1 failed (a regression), 1 deselected.

## Critical Issues to Fix

### 1. Failing test regression

`tests/test_pipeline.py:446` — `test_dashboard_primary_limit_reduces_ranked_slice_without_counting_uncertainty`

The test asserts `_dashboard_primary_limit(40)[0] == 32` (the old
`round(40 * 0.80)` formula using the removed `PRIMARY_RANKED_FRACTION`
constant), but the current implementation caps at
`DASHBOARD_QUEUE_SIZE = 12`.

**History:** `pipeline.py:139` previously computed
`primary_limit = max(1, int(round(config_count * PRIMARY_RANKED_FRACTION)))`
with `PRIMARY_RANKED_FRACTION = 0.80` (commit `eb51714`). The constant
was removed and the cap was changed to `min(config_count, DASHBOARD_QUEUE_SIZE)`,
but this test was not updated.

**Fix:**
```python
def test_dashboard_primary_limit_reduces_ranked_slice_without_counting_uncertainty():
    primary_limit, uncertain_slots = _dashboard_primary_limit(40)
    # Capped at DASHBOARD_QUEUE_SIZE=12; uncertain slots gated by config_count>=10
    assert primary_limit == 12
    assert uncertain_slots == 5


def test_dashboard_primary_limit_uncertain_threshold_at_10():
    assert _dashboard_primary_limit(9) == (9, 0)
    assert _dashboard_primary_limit(10) == (10, 5)
    assert _dashboard_primary_limit(1) == (1, 0)
```

**Decision required:** Is the new 12-card cap the target, or do you want
`primary_limit = round(40 * 0.80) = 32`? (The other test
`test_dashboard_primary_limit_is_capped_to_queue_size` at line 36 already
asserts the cap-at-12 behavior, so 12 is the target.)

### 2. Stale plan residue — `sort-toggle` CSS without HTML/JS

`templates/index.html:70-90` defines the `.sort-toggle` class but the
template renders **no element with that class** and the JS has **no
`applySort()` function**. The test at `test_server.py:98` even asserts
`assert 'id="sort-toggle"' not in resp.text` — confirming the toggle is
absent.

**Decision required:** Implement the 2-state Rank ⇄ Date toggle, or
remove the dead CSS.

### 3. `generate_detailed_tldr` return annotation mismatch

`server.py:281` — annotated `-> str | None` but every code path returns
a `str` (error cases return formatted error strings). The caller at
`server.py:793` does `if tldr:` which is dead code.

**Fix:** Change annotation to `-> str`.

## Bugs and Invariant Violations

### B1. `asyncio.run()` in synchronous `do_POST` handler

`server.py:683, 698, 743, 777, 837` call `asyncio.run(...)` to run
async fetchers. Functionally correct (each call creates a fresh event
loop in a fresh thread via `ThreadingHTTPServer`), but:

- No test covers concurrent TLDR requests on the same story.
- `self.db.upsert_story(...)` writes after each fetch could race, but
  the longest-prefix merge in `database.py:262-274` preserves the
  longer version.

**Test gap:** threaded concurrent-fetch test.

### B2. `_fetch_article_body` retry path is asymmetric

`server.py:98-104` retries 503/429 once with 1s sleep, but does **not**
retry on `httpx.ConnectError` or `httpx.TimeoutException`. The bare
`except Exception: return None` catches everything and silently
returns None.

**Suggestion:** add a test for the transient-timeout case.

### B3. `_tldr_cache_key` truncates inputs to fixed lengths

`server.py:268-270` truncates `self_text`, `top_comments`, `article_body`
to fixed limits. This is intentional (it normalizes the cache key for
LLM prompt variation) but **not tested**. Two inputs that differ only
after the limit must produce the same key.

**Test gap:** truncation invariant test (see P10).

### B4. `is_uncertain` entropy threshold is a weak filter

`pipeline.py:1701-1710` — `uncertain_entropy_threshold = get_entropy(uncertain_candidates[-1])`
is the entropy of the **5th-most-uncertain** in `remaining`. A primary
story with higher entropy gets `is_uncertain=True`, but this is
uncommon since primary stories usually have confident predictions.

**Test gap:** verify primary story with high entropy gets `is_uncertain=True`
(see U24).

### B5. `rerank_candidates` does not filter voted stories

`pipeline.py:1584-1837` — the function does not exclude stories that
the user has voted on. The caller is responsible (`fast_rerank_for_user`
at `pipeline.py:2042` excludes them, and `run_pipeline` at
`pipeline.py:1851` does the same). This is a **hidden coupling**.

**Test gap:** verify the contract (U19), or add an internal filter.

### B6. `rerank_candidates` total card count has no upper bound

`pipeline.py:1584-1837` — with `config.count=40`, primary is 12 cards.
Each of the 6 discovery passes can add up to `DISCOVERY_SLOT_LIMIT=5`
or `POPULARITY_DISCOVERY_SLOT_LIMIT=8` items. **Maximum total: 12 + 5
uncertain + 5 novel + 5 similar + 5 discussion + 8 engagement + 8 hot
= 48 cards.** There is no cap on the total dashboard size.

**Test gap:** verify the bound (U22), or add a hard cap.

### B7. `is_similar` is computed from upvote vectors only

`pipeline.py:1631-1640` — a "Similar" story must be close to an
**upvoted** story. A story similar only to downvoted stories gets
`cand_closest_up = 0` and can never be flagged Similar. This is
intentional (positive signal) but not tested.

**Test gap:** verify the contract (U6 invariant).

## Logical Invariants to Encode as Tests

| #   | Invariant                                                                    | Source                                       | Approach                |
| --- | ---------------------------------------------------------------------------- | -------------------------------------------- | ----------------------- |
| I1  | `_dashboard_primary_limit(c)` always returns `1 <= primary <= c`             | pipeline.py:139                              | Property: c ∈ [1, 100]  |
| I2  | `_dashboard_primary_limit(c)` returns `num_uncertain=0` for `c < 10` else 5  | pipeline.py:140                              | Property                |
| I3  | `rerank_candidates` final count bounded by `primary_limit + 36`             | pipeline.py:1584-1837                        | Property: random input  |
| I4  | `is_similar=True` is mutually exclusive with primary attribution            | pipeline.py:1704-1725                        | Unit (exists: 1129)     |
| I5  | No primary story ever has `is_similar=True`                                  | pipeline.py:1696-1700                        | Unit (strengthen)       |
| I6  | A story similar only to downvoted stories never gets `is_similar=True`       | pipeline.py:1631-1640                        | Unit                    |
| I7  | A primary story with entropy > threshold gets `is_uncertain=True`           | pipeline.py:1701-1710                        | Unit                    |
| I8  | `proactive fetch + db.upsert_story` merge keeps the longest `article_body`   | pipeline.py:1909-1946, database.py:262-274   | Unit                    |
| I9  | `_tldr_cache_key` for strings differing only after limit produces same key  | server.py:258-273                             | Unit                    |
| I10 | `_fetch_article_body` returns None on non-200 (no retry, no raise)          | server.py:89-125                              | Unit (exists)           |
| I11 | TLDR handler returns 404 for missing story, never crashes                   | server.py:663-668                            | Unit                    |
| I12 | Feedback `clear` then re-insert works (no double-insert error)              | server.py:631-638, database.py:470-489       | Unit                    |
| I13 | Concurrent TLDR requests on same story return identical text                | server.py:777-792                            | Threaded                |
| I14 | `upsert_story` is idempotent for identical inputs                            | database.py:250-335                          | Unit                    |
| I15 | `get_tldr_cache` returns last `upsert_tldr_cache` for (sid, key)            | database.py:449-468                          | Property (exists: 122)  |
| I16 | `rerank_candidates` output is a permutation of input story IDs              | pipeline.py:1584-1837                        | Property                |
| I17 | `rerank_candidates` is deterministic given identical inputs                 | pipeline.py:1584-1837                        | Property                |
| I18 | `_is_refetch_eligible` returns True iff 30%+, ≤24h, no fb, baseline>0, growth>0 | pipeline.py:498                            | Property                |
| I19 | `prune_stories` never removes a story with feedback                          | database.py:383-392                          | Property (exists: 223)  |
| I20 | `feedback_should_refresh` honors threshold and explicit override            | server.py:604-614                            | Property                |
| I21 | Dashboard cache version monotonically increases on invalidate                | server.py:563-572                            | Property (exists: 302)  |
| I22 | Stale render never overwrites fresh cache entry                              | server.py:482-546                            | Property (exists: 263)  |

## Suggested Improvements (no code change required to add tests)

- **S1.** Test that `applyGradient` (client-side) is rank-percentile, not
  linear (`templates/index.html:752-765`).
- **S2.** Test that `_fetch_article_body` strips the chrome tags
  (`<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`).
- **S3.** Test that `_fetch_reddit_rss_context` honors
  `REDDIT_COMMENT_LIMIT=40` and `REDDIT_COMMENTS_CACHE_CHAR_LIMIT=10000`.
- **S4.** Test that `is_low_signal_reddit_comment` filters out
  `withoutreason1729`, `i am a bot...`, `[deleted]`, `[removed]`,
  "your post is getting popular", and short comments.
- **S5.** Test the merge in `upsert_story` when `article_body` from a
  longer call is preserved over a shorter one (I8).

## New Unit Tests (priority order)

| ID  | Name                                                                          | File                |
| --- | ----------------------------------------------------------------------------- | ------------------- |
| U1  | `test_tldr_handler_returns_404_for_missing_story`                             | test_server.py      |
| U2  | `test_tldr_handler_with_no_url_or_self_text_returns_useful_summary`          | test_server.py      |
| U3  | `test_tldr_handler_includes_article_body_from_db`                             | test_server.py      |
| U4  | `test_tldr_handler_skips_fetch_when_article_body_present`                    | test_server.py      |
| U5  | `test_tldr_handler_does_not_fetch_for_reddit_source`                          | test_server.py      |
| U6  | `test_feedback_clear_then_revote_creates_new_record`                          | test_server.py      |
| U7  | `test_dashboard_route_with_no_user_creates_token`                             | test_server.py      |
| U8  | `test_tldr_handler_concurrent_requests_have_single_llm_call`                  | test_server.py      |
| U9  | `test_dashboard_render_after_feedback_invalidates_cache`                      | test_server.py      |
| U10 | `test_upsert_story_preserves_longer_article_body_across_fetches`              | test_database.py    |
| U11 | `test_upsert_story_recomputes_text_content_on_merge`                          | test_database.py    |
| U12 | `test_fetch_article_body_403_returns_none`                                    | test_fetch.py       |
| U13 | `test_fetch_article_body_strips_all_chrome_tags`                              | test_fetch.py       |
| U14 | `test_reddit_rss_context_filters_bot_comments`                                | test_server.py      |
| U15 | `test_reddit_rss_context_caps_at_40_comments`                                 | test_server.py      |
| U16 | `test_proactive_fetch_failure_preserves_article_body`                         | test_pipeline.py    |
| U17 | `test_proactive_fetch_uses_shorter_text_for_embedding`                        | test_pipeline.py    |
| U18 | `test_run_pipeline_swallows_proactive_fetch_exceptions`                       | test_pipeline.py    |
| U19 | `test_rerank_candidates_does_not_filter_voted_stories`                        | test_pipeline.py    |
| U20 | `test_tldr_cache_key_truncates_long_inputs`                                   | test_server.py      |
| U21 | `test_dashboard_primary_limit_uncertain_threshold_at_10`                      | test_pipeline.py    |
| U22 | `test_rerank_candidates_total_card_count_cap`                                 | test_pipeline.py    |
| U23 | `test_rerank_candidates_is_similar_excluded_for_primary`                      | test_pipeline.py    |
| U24 | `test_rerank_candidates_is_uncertain_for_high_entropy_primary`                | test_pipeline.py    |
| U25 | `test_rerank_candidates_is_uncertain_false_for_unanimous_primary`             | test_pipeline.py    |
| U26 | `test_rerank_candidates_final_sorted_by_score`                                | test_pipeline.py    |
| U27 | `test_dashboard_route_no_user_creates_token_and_redirects`                    | test_server.py      |

## New Property Tests (priority order)

| ID  | Name                                                                       | Strategy                |
| --- | -------------------------------------------------------------------------- | ----------------------- |
| P1  | `test_clean_text_idempotent`                                               | x = clean(clean(x))     |
| P2  | `test_rank_percentiles_in_unit_interval_and_monotonic`                     | numpy arrays            |
| P3  | `test_softmax_rows_sums_to_one_and_in_unit_interval`                       | numpy arrays            |
| P4  | `test_augment_features_shape_and_preserves_embeddings`                     | random n, emb_dim=384   |
| P5  | `test_minmax01_maps_min_to_zero_max_to_one`                                | arrays                  |
| P6  | `test_get_or_compute_embeddings_is_deterministic_across_calls`             | repeated calls          |
| P7  | `test_compose_story_text_contains_all_inputs_and_idempotent`               | (title, s, c, b) tuples |
| P8  | `test_mmr_filter_output_is_subset_and_preserves_order`                     | (exists: 389)           |
| P9  | `test_fetch_article_body_retry_count_is_bounded`                           | 503/429/timeout         |
| P10 | `test_tldr_cache_key_truncation_invariant`                                 | long inputs             |
| P11 | `test_normalize_tldr_markdown_idempotent_and_no_carriage_return`            | random strings          |
| P12 | `test_source_label_filter_idempotent`                                      | random strings          |
| P13 | `test_rss_source_name_is_deterministic_and_lowercase`                      | random URLs             |
| P14 | `test_time_ago_filter_returns_nonempty_string_for_valid_timestamp`         | timestamps              |
| P15 | `test_dashboard_cache_version_monotonic_under_random_invalidate_render`    | (exists: 310)           |
| P16 | `test_rerank_candidates_output_is_permutation_of_input`                    | random stories          |
| P17 | `test_upsert_story_longest_prefix_merge`                                   | two upsert calls        |
| P18 | `test_is_refetch_eligible_threshold_at_30pct_exact`                        | boundary cases          |
| P19 | `test_select_refetch_ids_respects_max_per_regen`                           | (exists: 829)           |
| P20 | `test_prune_stories_invariant_with_random_feedback`                        | (exists: 223)           |
| P21 | `test_feedback_should_refresh_threshold_logic`                             | payloads                |
| P22 | `test_rerank_candidates_score_in_unit_interval`                            | random stories          |

## Summary of Changes

### Code (small, targeted)

1. Fix the failing test regression at `tests/test_pipeline.py:446` (assert
   `primary_limit == 12`).
2. Change `generate_detailed_tldr` annotation from `-> str | None` to
   `-> str` (`server.py:281`).
3. Decide on `sort-toggle` CSS: implement the 2-state Rank ⇄ Date
   button, or remove the dead CSS.

### Tests (new)

- 27 new unit tests (U1–U27 above).
- 22 new property tests (P1–P22 above).
- A test_server.py fixture for concurrent TLDR requests (U8).

### Documentation

- `ARCHITECTURE.md`: document `_dashboard_primary_limit` formula and
  the cap at `DASHBOARD_QUEUE_SIZE`.
- `ARCHITECTURE.md`: document the `upsert_story` longest-prefix merge
  contract.

## Open Questions

1. **`sort-toggle`**: implement the 2-state Rank ⇄ Date button (with
   the test), or remove the dead CSS?
2. **Cap at 12 primary cards**: is `DASHBOARD_QUEUE_SIZE=12` the target,
   or do you want `primary_limit = round(40 * 0.80) = 32`?
3. **Should `rerank_candidates` filter voted stories internally**
   (relieving the caller of the contract), or keep the current
   caller-must-filter contract?
4. **TLDR cache truncation invariant** (I9/P10): is the current
   truncation-before-hashing intentional? If so, add the test; if not,
   this is a bug to fix.
