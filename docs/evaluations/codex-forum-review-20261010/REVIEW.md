# Review: Codex Forum source (eb643bb, snapshot 6db1077)

## Verdict

The two-line config and label change is mechanically correct. The feed parses, gets the source ID `rss_community_openai_com`, uses topic URLs as story IDs, and the label maps as intended. **One introduced correctness/representation issue (Medium):** the new ARCHITECTURE.md says the cards carry "the opening post" only. That is wrong. The existing generic RSS article fetch pulls the Discourse crawler HTML for each topic. That HTML contains the first page of up to 20 posts, replies included. The replies are stored once, never refreshed, and passed to the summarizer as `Article body:`, so the model can attribute replies to the author. Two Low items follow. No crash, identity, or upsert defect was found.

## Introduced defects (ranked)

### M1: Replies leak into `article_body` without a label, as a frozen snapshot, contrary to the documented "opening post only"

- **Where:** the source is not excluded at `pipeline/enrichment.py:1180` (`_NO_ARTICLE_SOURCES`). Prewarm selection is at `pipeline/enrichment.py:1199-1230` and has no `self_text` length gate. The fetch and write happen at `pipeline/enrichment.py:1365-1374`. The tap path is `server.py:3135-3142` (only when `self_text` < 500 chars). Prompt framing is `server.py:1384-1388`. The doc claim is in ARCHITECTURE.md, in the diff hunk at line 630.
- **Evidence (fetched 2026-10-11):** `GET https://community.openai.com/t/has-anyone-experienced-frequent-restrictions-on-daybreak-blue-lately-for-the-last-two-days/1404351` with the Chrome UA from `server.py:477` returned 200 `text/html` with 21 `.crawler-post` nodes. I took the 20 posts from `/t/1404351.json`; all 20 of their authors appear in BeautifulSoup body text after `script/style/nav/footer/header/aside` removal, which mirrors `_extract_raw_body` at `server.py:404`. The page has no `<article>` or `<main>`, so `_extract_with_bs_semantic` falls through. The body text also holds a "related topics" list and a site banner ("Suspicious activity detected…"). jusText and trafilatura are not installed here, so I could not reproduce the exact tier output. Any tier that returns text will return reply text as well.
- **Impact:** (a) Detailed summaries can present replies as the author's article. That is the "accurately represented" failure the prompt asks about. (b) `text_content` and the embedding (`enrichment.py:1368-1384`) now include a reply snapshot, so ranking input differs from what the doc describes. (c) Because of `if not story.url or story.article_body: return False` (`enrichment.py:1186`), the snapshot is taken once, near creation, and never refreshed. It is stale by construction. (d) Boilerplate such as related topics and the banner pollutes the text.
- **Minimal remedy:** add `rss_community_openai_com` to `_NO_ARTICLE_SOURCES`. That one line makes the cards opening-post-only, which is what the doc already claims. Add a test asserting that `select_rss_article_prewarm` and `_article_fetch_eligible` skip the source. Then check the tap path: `server.py:3135` checks `AINEWS_SOURCE` directly rather than `_NO_ARTICLE_SOURCES`, so short posts would still fetch on tap. Either gate the tap path on the same set or accept that behaviour and document it. If any Codex rows already have an `article_body`, decide whether to clear it. The existing "Haskell Discourse" source (`render.py:82`) has the same exposure (pre-existing).

### L1: Stale count text in `self_text`

- Each RSS `description` ends with `<p><small>N posts - M participants</small></p>` and a "Read full topic" link (verified in `https://community.openai.com/c/codex/37.rss`). `clean_text` keeps this text in `self_text` and `text_content` (`enrichment.py:928-931`). The upsert only takes the new value when it is at least as long (`database.py:~663`). The count updates while the topic is still in the 25-item feed, which covers roughly 2 days of creation-ordered topics, then freezes. Summaries may then say things like "1 post, no replies".
- **Remedy:** strip the trailing `<small>` count paragraph and the "Read full topic" anchor for Discourse feeds before `clean_text`, and add one fixture test.

### L2: The host-wide label is a latent mislabel

- `render.py:83` maps every `community.openai.com` feed to "Codex Forum", because `_rss_source_name` (`enrichment.py:812-823`) keys on host only. Adding any other category later (API, Prompting, and so on) would be mislabeled, and both categories would merge into one source ID. **Remedy:** none needed now. Leave a comment at the mapping, or make it category-aware when a second category is added.

## Verified OK (Scope A)

- RSS returns 200 `application/rss+xml`, with 25 items, a canonical `link` of the form `/t/<slug>/<id>`, a guid of the form `community.openai.com-topic-<id>`, and the `pubDate` set to topic creation (so the cutoff logic is correct).
- The feed includes subcategories (Feature Requests, Codex CLI). Pinned flags are exposed as `discourse:topicPinned`, and the RSS items I sampled had it set to "No".
- Identity: `rss_story_id(link)` is stable unless a topic is renamed. A slug change produces a new URL and a new ID, so a renamed topic would show up as a duplicate. This is the same for all RSS sources and is pre-existing.
- Label: the double `rss_` strip and the known-map lookup give "Codex Forum". There is no collision with `openai_com` ("OpenAI").
- `_web_url` admits https links. Reddit/LessWrong-specific branches are not triggered.

## Pre-existing (not regressions)

- Score and comment count are always 0 for RSS. There is no reply adapter (acknowledged).
- Slug renames re-key the story (generic RSS behaviour).
- The Haskell Discourse source has the same M1 and L1 exposure.

## Scope B: is explicit Discourse reply/count support the right next step?

**Recommendation:** first ship M1's one-line exclusion plus L1's stripping. They are cheap, they make the documentation true, and they remove the mislabelled replies. Build a small Discourse adapter only if Codex Forum cards actually earn taps or summaries. It is feasible, but it adds a fetch lane, a cache, and failure modes.

**Verified public access (no auth, 2026-10-11):**

- `https://community.openai.com/c/codex/37.json` returned 200. It gives a 30-topic `topic_list` with `posts_count`, `reply_count`, `like_count`, `views`, `bumped_at`, `pinned`, and `more_topics_url=/c/codex/37?page=1`. This one call can supply counts for every listed topic.
- `https://community.openai.com/t/1404351.json` returned 200. It gave `posts_count=76`, `highest_post_number=79`, `chunk_size=20`, 20 posts in `post_stream.posts`, and 76 IDs in `stream`. Post #3 is missing, which shows that deleted or hidden posts are dropped from the public stream. Each post carries `post_number`, `post_type`, `hidden`, `deleted_at`, `user_deleted`, `action_code`, `reply_to_post_number`, and `cooked` (HTML).
- `https://community.openai.com/t/1404351/posts.json?post_ids[]=…` (20 IDs from `stream[20:40]`) returned 200 with 20 posts, so pagination works.
- `https://community.openai.com/t/1404351.rss` returned 404. Do not rely on per-topic RSS without the slug.
- Responses send `cache-control: no-cache, no-store`. I saw no rate-limit headers. Discourse's documented anonymous limits are an **assumption**; I did not verify them.

**Smallest viable design:**

1. Allowlist the hosts explicitly (`{"community.openai.com"}`; optionally `discourse.haskell.org`). Parse the topic ID from `/t/<slug>/<id>` with a regex anchored on the path. Never follow arbitrary hosts.
2. Counts: once per regen, read the category `.json`, which costs one request. Set `comment_count = posts_count - 1`; the opening post is not a reply. Note that `reply_count` in Discourse counts direct replies only, which is a different semantics. Leave `comment_count_at_fetch` at its existing meaning.
3. Replies: fetch on tap or summary only, never during prewarm. Fetch `/t/<id>.json` and use only the first chunk (≤20 posts), or ask for at most one more `posts.json` page for a bounded sample. Skip `post_number==1`, `post_type!=1`, `hidden`, `deleted_at`, `user_deleted`, and `action_code`. Sanitize `cooked` through the same HTML-to-text path as Reddit/LW. Strip quotes and onebox blocks. Format replies as `top_comments` so the prompt labels them as discussion.
4. Refresh on growth: reuse the existing rule (`posts_count - 1 > comment_count_at_fetch`). A forced refresh bypasses the cache. On failure, keep the old snapshot and never report it as fresh, mirroring the HN rule at `server.py:3123-3127`.
5. Network behaviour: shared `httpx` client, 10–15 s timeout, semaphore of 2–4, one retry on 429/503 that honours `Retry-After`, and `guarded_get`.
6. Provenance: store a fetch timestamp and `posts_count` at fetch time, so the card can show "first 20 of 76 posts as of T".
7. Fallback: unknown host, non-200, or a JSON shape mismatch falls back to opening-post-only. It must not fall back to an HTML article fetch.

**Essential tests:** URL to topic-ID parsing (slug change, trailing `/N` post anchor, foreign host rejected); category-JSON count mapping (`posts_count-1`, pinned "About" topic excluded); post filtering (deleted, hidden, small-action, OP excluded); partial stream (fewer posts than `posts_count`, and the summary says it is a sample); growth-triggered and forced refresh plus stale-on-failure; a 429 retry bounded to one attempt; `cooked` sanitization (script, quote, onebox); and a regression test that the source never enters article prewarm.

**Proposal risks:** the forum is a support forum, so replies are often "+1, same here", which is low signal for the extra fetches. API shape drift is possible. The JSON exposes more PII-adjacent fields (usernames, avatars) than the dashboard needs, so store only the text.

## Coverage and limits

I read `pipeline/render.py`, the RSS, upsert-adjacent, prewarm, and article paths of `pipeline/enrichment.py`, the article fetch, extraction, and prompt code in `server.py`, and the diff. I skimmed `database.py` upsert merging. I did not open `clean_text` or the bundled tests, and no test in the bundle references the new label. I ran no tests: jusText and trafilatura are absent, and the bundle is partial. The extraction claim rests on the raw-body path plus the presence of crawler posts. I made no production access and used no credentials. The only network calls were the five public GETs listed above, plus a header-only re-fetch of `/t/1404351.json`.
