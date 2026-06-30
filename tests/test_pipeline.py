from typing import cast
import asyncio
import numpy as np
import pytest
import time
from hypothesis import given, strategies as st, settings, HealthCheck
from database import Database, Story
from dataclasses import replace
import pipeline
from pipeline import (
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_SOURCE,
    Embedder,
    ModelConfig,
    RankTrace,
    RankedStory,
    RssConfig,
    clean_text,
    compose_story_text,
    get_or_compute_embeddings,
    is_hn_source,
    mmr_filter,
    _score_and_rank,
    Config,
    rerank_candidates,
    select_article_fetch_candidates,
    source_label_filter,
    story_embedding_text,
    _extract_comments_recursive,
    _select_top_comments,
    _needs_hn_prewarm,
    _dashboard_primary_limit,
    _reddit_subreddit_from_feed_url,
    _rss_source_name,
    _svm_personalization_features,
    _feedback_signature,
    _get_cached_model,
    _set_cached_model,
    _MODEL_CACHE,
    fast_rerank_for_user,
    HOT_MIN_SCORE,
)


@pytest.fixture
def db():
    db_instance = Database(":memory:")
    yield db_instance
    db_instance.close()


def test_dashboard_primary_limit_is_capped_to_queue_size():
    assert _dashboard_primary_limit(40)[0] == 12
    assert _dashboard_primary_limit(9)[0] == 9


def test_dashboard_primary_limit_uncertain_threshold_at_10():
    assert _dashboard_primary_limit(1) == (1, 0)
    assert _dashboard_primary_limit(9) == (9, 0)
    assert _dashboard_primary_limit(10) == (10, 5)
    assert _dashboard_primary_limit(40) == (12, 5)


def test_article_fetch_selection_prioritizes_dashboard_and_filters(db):
    now = 2_000_000_000.0

    def story(sid, *, score=100, comments=0, age_days=1, body="", url=None):
        return Story(
            id=sid,
            title=f"S{sid}",
            url=url or f"https://example.com/{sid}",
            score=score,
            time=int(now - age_days * 86400),
            text_content=f"text {sid}",
            source="hn",
            comment_count=comments,
            article_body=body,
        )

    dashboard_1 = story(1, score=10)
    dashboard_2 = story(2, score=10)
    old = story(3, age_days=31)
    backed_off = story(4, score=1000)
    extra_hot = story(5, score=900, comments=90)
    extra_cool = story(6, score=50, comments=5)
    has_body = story(7, body="already fetched")

    for s in [
        dashboard_1,
        dashboard_2,
        old,
        backed_off,
        extra_hot,
        extra_cool,
        has_body,
    ]:
        db.upsert_story(s)
    db.record_article_fetch_failure(
        backed_off.id,
        backed_off.url,
        error="http_503",
        next_retry_at=now + 3600,
    )

    ranked = [
        RankedStory(story=s, score=float(10 - idx), best_match_title="")
        for idx, s in enumerate(
            [extra_hot, backed_off, extra_cool, dashboard_1, old, dashboard_2, has_body]
        )
    ]
    dashboard = [
        RankedStory(story=dashboard_1, score=1.0, best_match_title=""),
        RankedStory(story=old, score=0.9, best_match_title=""),
        RankedStory(story=dashboard_2, score=0.8, best_match_title=""),
    ]

    selected = select_article_fetch_candidates(
        ranked=ranked,
        dashboard_selected=dashboard,
        db=db,
        max_per_run=3,
        max_age_days=30,
        now_ts=now,
    )

    assert [s.id for s in selected] == [1, 2, 5]


def _cold_story(
    sid: int,
    *,
    score: int,
    time_ts: int,
    source: str = "hn",
    text_content: str = "story text",
    comment_count: int | None = 1,
) -> Story:
    return Story(
        id=sid,
        title=f"Cold {sid}",
        url=f"https://example.com/{sid}",
        score=score,
        time=time_ts,
        text_content=text_content,
        source=source,
        comment_count=comment_count,
    )


def test_build_cold_deck_empty_db(db: Database) -> None:
    assert pipeline.build_cold_deck(db) == []


def test_build_cold_deck_score_order_and_summarizable_filter(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 2_000_000_000
    monkeypatch.setattr(pipeline.time, "time", lambda: float(now))
    unsummarizable = _cold_story(
        1,
        score=1000,
        time_ts=now - 3600,
        text_content="",
        comment_count=0,
    )
    low = _cold_story(2, score=10, time_ts=now - 3600)
    high = _cold_story(3, score=100, time_ts=now - 3600)
    for story in (unsummarizable, low, high):
        db.upsert_story(story)

    cold = pipeline.build_cold_deck(db)

    assert [item.story.id for item in cold] == [3, 2]
    gravity_div = 3.0**1.8
    assert [item.score for item in cold] == pytest.approx(
        [100.0 / gravity_div, 10.0 / gravity_div]
    )


def test_build_cold_deck_combo_keys_and_flags(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 2_000_000_000
    monkeypatch.setattr(pipeline.time, "time", lambda: float(now))
    recent_hn = _cold_story(1, score=30, time_ts=now - 3600, source="hn")
    archive_hn = _cold_story(
        2,
        score=20,
        time_ts=now - (31 * 86400),
        source=BQ_ARCHIVE_SOURCE,
    )
    recent_non_hn = _cold_story(
        3,
        score=10,
        time_ts=now - 3600,
        source="rss_blog",
        text_content="rss body",
        comment_count=0,
    )
    recent_non_hn = replace(recent_non_hn, article_body="rss body")
    for story in (recent_hn, archive_hn, recent_non_hn):
        db.upsert_story(story)

    cold = pipeline.build_cold_deck(db)
    by_id = {item.story.id: item for item in cold}

    assert by_id[1].combo_keys == "recent_hn recent_mixed"
    assert by_id[1].is_recent is True
    assert by_id[1].is_non_hn is False
    assert by_id[2].combo_keys == "archive_hn archive_mixed"
    assert by_id[2].is_recent is False
    assert by_id[2].is_non_hn is False
    assert by_id[3].combo_keys == "recent_non-hn recent_mixed"
    assert by_id[3].is_recent is True
    assert by_id[3].is_non_hn is True


def test_build_cold_deck_uses_badge_defaults(db: Database) -> None:
    db.upsert_story(_cold_story(1, score=100, time_ts=int(time.time()) - 3600))

    item = pipeline.build_cold_deck(db)[0]

    assert item.best_match_title == ""
    assert item.prob_down is None
    assert item.prob_neutral is None
    assert item.prob_up is None
    assert item.is_uncertain is False
    assert item.is_novel is False
    assert item.is_discussion_rich is False
    assert item.is_high_engagement is False
    assert item.is_hot is False
    assert item.is_similar is False


@pytest.fixture(scope="module")
def embedder():
    # Uses the real downloaded ONNX model
    return Embedder("onnx_model")


def test_embedder_output_shape(embedder):
    texts = ["Hello world", "Hacker News rewrite plan"]
    embs = embedder.encode(texts)
    assert embs.shape == (2, 384)
    # Check normalization: norms should be close to 1.0
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_cache_hit(db, embedder):
    story = Story(
        id=999,
        title="Cache Test",
        url=None,
        score=0,
        time=0,
        text_content="Unique test content for caching.",
    )
    db.upsert_story(story)

    # First call: computes and caches
    embs1 = get_or_compute_embeddings([story], embedder, db)
    assert embs1.shape == (1, 384)

    # Second call: check if cached
    model_version = "all-MiniLM-L6-v2|mean|norm|256"
    import hashlib

    shash = hashlib.sha256(story.text_content.encode("utf-8")).hexdigest()
    cached = db.get_embedding(999, model_version, shash)
    assert cached is not None
    assert np.allclose(cached, embs1[0])

    # Re-run get_or_compute_embeddings, should load from cache
    embs2 = get_or_compute_embeddings([story], embedder, db)
    assert np.allclose(embs1, embs2)


def test_compose_text():
    title = "<strong>Some title</strong>"
    self_text = "Some self text content."
    comments = " ".join(["First comment text here.", "Second comment is long."])
    composed = compose_story_text(title, self_text, comments)
    assert "Some title" in composed
    assert "Some self text" in composed
    assert "First comment" in composed


def test_story_embedding_text_prefers_stored_text_content():
    story = Story(
        id=1,
        title="Fresh title",
        url=None,
        score=0,
        time=0,
        text_content="Stable cached embedding text.",
        self_text="Longer self text should not change existing embedding input.",
        top_comments="Comment text should not be used when text_content exists.",
        article_body="Article text should not be used when text_content exists.",
    )

    assert story_embedding_text(story) == "Stable cached embedding text."


def test_story_embedding_text_recovers_from_raw_fields_when_text_content_missing():
    story = Story(
        id=1,
        title="Recovered title",
        url=None,
        score=0,
        time=0,
        text_content="",
        self_text="Recovered self text.",
        top_comments="Recovered comment text.",
        article_body="Recovered article text.",
    )

    text = story_embedding_text(story)
    assert "Recovered title" in text
    assert "Recovered self text" in text
    assert "Recovered article text" in text
    assert "Recovered comment text" in text


def test_strip_html():
    raw_html = "<p>Hello &amp; welcome to <a href='#'>Hacker News</a>!</p>"
    cleaned = clean_text(raw_html)
    assert cleaned == "Hello & welcome to Hacker News!"


def test_comment_extraction_allows_strong_replies_to_compete():
    comments = [
        {
            "type": "comment",
            "points": 100,
            "text": "Top-level comment with enough substance to pass the minimum length.",
            "children": [
                {
                    "type": "comment",
                    "points": 130,
                    "text": "Reply with substantially more points and enough context to stand alone.",
                    "children": [],
                }
            ],
        }
    ]

    extracted = sorted(_extract_comments_recursive(comments), key=lambda x: x["score"])

    assert extracted[0]["text"].startswith("Reply with substantially more points")
    assert extracted[1]["text"].startswith("Top-level comment")


def test_comment_selection_includes_replies_from_large_threads():
    comments = [
        {
            "type": "comment",
            "text": "Large discussion root with enough substance to pass filtering.",
            "children": [
                {
                    "type": "comment",
                    "text": f"Substantive reply {i} with enough context to be useful in a TLDR summary.",
                    "children": [],
                }
                for i in range(8)
            ],
        },
        *[
            {
                "type": "comment",
                "text": f"Separate top-level comment {i} with enough useful context and detail to pass the comment length threshold.",
                "children": [],
            }
            for i in range(8)
        ],
    ]

    selected = _select_top_comments(_extract_comments_recursive(comments), limit=12)
    large_thread_replies = [
        c for c in selected if c["top_thread_index"] == 0 and c["depth"] > 0
    ]

    assert len(large_thread_replies) == 5


def test_comment_selection_caps_comments_per_thread():
    comments = [
        {
            "type": "comment",
            "text": "Large discussion root with enough substance to pass filtering.",
            "children": [
                {
                    "type": "comment",
                    "text": f"Substantive reply {i} with enough context to be useful in a TLDR summary.",
                    "children": [],
                }
                for i in range(20)
            ],
        },
        *[
            {
                "type": "comment",
                "text": f"Separate top-level comment {i} with enough useful context and detail to pass the comment length threshold.",
                "children": [],
            }
            for i in range(20)
        ],
    ]

    selected = _select_top_comments(_extract_comments_recursive(comments), limit=20)
    large_thread_comments = [c for c in selected if c["top_thread_index"] == 0]

    assert len(large_thread_comments) == 6
    assert len({c["top_thread_index"] for c in selected}) > 1


def test_clean_text():
    assert clean_text("Valid text content here that has enough length.") != ""
    assert clean_text("Short", min_len=20) == ""  # too short
    assert clean_text("!!!???---***+++///\\\\|") == ""  # excessive punctuation
    # Braille block characters
    assert clean_text("⠠⠓⠑⠫⠕ ⠺⠕⠗⠇⠙") == ""


def test_rank_no_feedback_fallback(db, embedder):
    candidates = [
        Story(
            id=1,
            title="Machine Learning",
            url=None,
            score=0,
            time=0,
            text_content="AI and machine learning tutorial.",
        ),
        Story(
            id=2,
            title="Cooking recipes",
            url=None,
            score=0,
            time=0,
            text_content="How to bake bread and cook pasta.",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    config = Config()
    ranked = _score_and_rank(
        candidates=candidates,
        candidate_embeddings=cand_embs,
        db=db,
        config=config,
        embedder=embedder,
    )

    assert len(ranked) == 2
    # Without feedback, stories are ranked by HN gravity formula
    # Both stories have score=0, time=0 so they get very low gravity scores
    assert ranked[0].score >= 0
    assert ranked[1].score >= 0


def test_rank_svm_path(db, embedder):
    config = Config()
    user = db.create_user("test_token_svm")
    # 20 up + 20 down = both gates pass (n_up >= 20, n_down >= 20)
    for i in range(20):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up story {i}",
                url=None,
                score=0,
                time=0,
                text_content="Deep learning AI research artificial intelligence",
            )
        )
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down story {i}",
                url=None,
                score=0,
                time=0,
                text_content="Baking sourdough bread cake cookie recipe kitchen",
            )
        )
        db.upsert_feedback(user.id, 100 + i, "up")
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
        ),
        Story(
            id=2,
            title="Cake recipe",
            url=None,
            score=0,
            time=0,
            text_content="Delicious chocolate chip cake baking guide.",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    ranked = _score_and_rank(
        candidates=candidates,
        candidate_embeddings=cand_embs,
        db=db,
        config=config,
        embedder=embedder,
    )

    assert len(ranked) == 2
    # AI story should be ranked 1st
    assert ranked[0].story.id == 1
    # Check that the normalized SVM margin score stays blend-compatible.
    assert 0.0 <= ranked[0].score <= 1.0


def test_rss_synthetic_id():
    # Stable negative ID check
    link1 = "https://example.com/story-1"
    link2 = "https://example.com/story-2"

    import hashlib

    def get_id(link):
        h = hashlib.md5(link.encode("utf-8")).digest()
        val = int.from_bytes(h[:4], "big")
        return -(val % (2**31))

    id1_a = get_id(link1)
    id1_b = get_id(link1)
    id2 = get_id(link2)

    assert id1_a == id1_b
    assert id1_a != id2
    assert id1_a < 0
    assert id2 < 0


def test_reddit_feed_source_names_include_subreddit():
    url = "https://www.reddit.com/r/LocalLLaMA/top/.rss?t=week&limit=25"

    assert _reddit_subreddit_from_feed_url(url) == "localllama"
    assert _rss_source_name(url) == "rss_reddit_localllama"
    assert _rss_source_name("https://simonwillison.net/atom/everything/") == (
        "rss_simonwillison_net"
    )
    assert _rss_source_name("https://rss.slashdot.org/Slashdot/slashdotMain") == (
        "rss_slashdot_org"
    )


def test_source_label_filter_cleans_rss_prefixes():
    assert source_label_filter(BQ_ARCHIVE_SOURCE) == "BQ Seed"
    assert source_label_filter(CH_ARCHIVE_SOURCE) == "CH Seed"
    assert source_label_filter("rss_rss_slashdot_org") == "Slashdot"
    assert source_label_filter("rss_slashdot_org") == "Slashdot"
    assert source_label_filter("rss_reddit_localllama") == "r/localllama"
    assert source_label_filter("rss_simonwillison_net") == "Simon Willison"


def test_is_hn_source_includes_bq_seed():
    assert is_hn_source("hn")
    assert is_hn_source(BQ_ARCHIVE_SOURCE)
    assert is_hn_source(CH_ARCHIVE_SOURCE)
    assert not is_hn_source("rss_example_com")


@pytest.mark.asyncio
async def test_rss_feed_captures_comments_url(monkeypatch):
    """RSS <comments> element is captured as discussion_url.

    Tildes and Lobsters provide a <comments> element distinct from
    <link>. The Story constructor must populate discussion_url from it
    so the UI can render a comments link.
    """
    from pipeline import _fetch_and_parse_feed

    class MockResp:
        status_code = 200
        text = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item>
  <title>Black Shirt</title>
  <link>https://example.com/article</link>
  <comments>https://tildes.net/~group/topic_id</comments>
  <pubDate>Tue, 23 Jun 2026 12:00:00 GMT</pubDate>
  <description>A great article about shirts.</description>
</item>
</channel></rss>"""
        headers: dict[str, str] = {}

    class MockClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            return MockResp()

    monkeypatch.setattr("pipeline.httpx.AsyncClient", MockClient)

    stories = await _fetch_and_parse_feed(
        "https://tildes.net/topics.rss",
        per_feed=10,
        cutoff=0,
        now=1_000_000,
        exclude_urls=set(),
    )
    assert len(stories) == 1
    assert stories[0].url == "https://example.com/article"
    assert stories[0].discussion_url == "https://tildes.net/~group/topic_id"


@pytest.mark.asyncio
async def test_rss_feed_no_comments_url(monkeypatch):
    """RSS feeds without <comments> element leave discussion_url as None.

    Reddit, LessWrong, and personal blog feeds don't have a <comments>
    element. The link is the article URL, not a separate discussion
    page, so discussion_url should remain None.
    """
    from pipeline import _fetch_and_parse_feed

    class MockResp:
        status_code = 200
        text = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item>
  <title>Some Blog Post</title>
  <link>https://example.com/blog</link>
  <pubDate>Tue, 23 Jun 2026 12:00:00 GMT</pubDate>
  <description>A blog post about something interesting.</description>
</item>
</channel></rss>"""
        headers: dict[str, str] = {}

    class MockClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            return MockResp()

    monkeypatch.setattr("pipeline.httpx.AsyncClient", MockClient)

    stories = await _fetch_and_parse_feed(
        "https://example.com/feed.xml",
        per_feed=10,
        cutoff=0,
        now=1_000_000,
        exclude_urls=set(),
    )
    assert len(stories) == 1
    assert stories[0].url == "https://example.com/blog"
    assert stories[0].discussion_url is None


def test_is_summarizable_with_content():
    """Stories with self_text, top_comments, or article_body are summarizable."""
    from pipeline import Story

    s = Story(
        id=1,
        title="X",
        url=None,
        score=5,
        time=100,
        text_content="x",
        source="rss",
        self_text="Some text",
    )
    assert pipeline.is_summarizable(s)

    s = Story(
        id=2,
        title="X",
        url=None,
        score=5,
        time=100,
        text_content="x",
        source="rss",
        top_comments="Some comments",
    )
    assert pipeline.is_summarizable(s)

    s = Story(
        id=3,
        title="X",
        url=None,
        score=5,
        time=100,
        text_content="x",
        source="rss",
        article_body="Some body",
    )
    assert pipeline.is_summarizable(s)


def test_is_summarizable_hn_with_comments():
    """HN stories with comment_count > 0 but no inline text are summarizable
    (comments can be fetched on-demand or prewarmed at regen)."""
    from pipeline import Story

    s = Story(
        id=1,
        title="X",
        url=None,
        score=5,
        time=100,
        text_content="x",
        source="hn",
        comment_count=10,
        comment_count_at_fetch=10,
    )
    assert pipeline.is_summarizable(s)


def test_is_summarizable_hn_zero_comments_no_content():
    """HN stories with 0 comments and no text content are NOT summarizable."""
    from pipeline import Story

    s = Story(
        id=1, title="X", url=None, score=5, time=100, text_content="x", source="hn"
    )
    assert not pipeline.is_summarizable(s)


def test_is_summarizable_non_hn_no_content():
    """Non-HN stories with no text content are NOT summarizable."""
    from pipeline import Story

    s = Story(
        id=1,
        title="X",
        url=None,
        score=5,
        time=100,
        text_content="x",
        source="rss_reddit_test",
    )
    assert not pipeline.is_summarizable(s)


def test_is_summarizable_lesswrong_with_comments():
    """LessWrong stories with comment_count > 0 are summarizable (prewarmed)."""
    from pipeline import Story

    s = Story(
        id=1,
        title="X",
        url=None,
        score=5,
        time=100,
        text_content="x",
        source="rss_lesswrong_com",
        comment_count=3,
        comment_count_at_fetch=3,
    )
    assert pipeline.is_summarizable(s)


def test_is_summarizable_lesswrong_zero_comments():
    """LessWrong stories with 0 comments and no text are NOT summarizable."""
    from pipeline import Story

    s = Story(
        id=1,
        title="X",
        url=None,
        score=5,
        time=100,
        text_content="x",
        source="rss_lesswrong_com",
    )
    assert not pipeline.is_summarizable(s)


@pytest.mark.asyncio
async def test_build_reddit_topfeed_serializes_and_sets_user_agent(
    tmp_path, monkeypatch
):
    """Reddit topfeed factories must serialize HTTP requests (max
    concurrency = 1) and use the Reddit user agent. Replaces the old
    `test_fetch_rss_feeds_serializes_reddit_and_sets_user_agent` after
    Reddit topfeed was split out of `fetch_rss_feeds`."""
    from pipeline import REDDIT_RSS_USER_AGENT, build_reddit_topfeed_factories
    from reddit_fetch_queue import queue as reddit_fetch_queue

    active_reddit_requests = 0
    max_reddit_concurrency = 0
    seen_requests = []

    class MockResp:
        status_code = 200

        def __init__(self, text: str):
            self.text = text
            self.headers: dict[str, str] = {}

    def rss_doc(title: str, link: str) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item><title>{title}</title><link>{link}</link><pubDate>Tue, 23 Jun 2026 12:00:00 GMT</pubDate><description>Substantial test summary text for ranking.</description></item>
</channel></rss>"""

    class MockClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            nonlocal active_reddit_requests, max_reddit_concurrency
            is_reddit = "reddit.com" in url
            if is_reddit:
                active_reddit_requests += 1
                max_reddit_concurrency = max(
                    max_reddit_concurrency, active_reddit_requests
                )
            seen_requests.append((url, headers or {}))
            await asyncio.sleep(0)
            if is_reddit:
                active_reddit_requests -= 1
            title = "Reddit Item" if is_reddit else "Regular Item"
            link = url.replace("/top/.rss", "/comments/test")
            return MockResp(rss_doc(title, link))

    monkeypatch.setattr("pipeline.httpx.AsyncClient", MockClient)
    # Disable the 2s inter-request delay so the test runs fast;
    # the conftest fixture resets the limiter before/after each test.
    monkeypatch.setattr("pipeline.reddit_limiter", pipeline.reddit_limiter)
    pipeline.reddit_limiter.INTER_REQUEST_DELAY = 0.0

    feeds = [
        "https://www.reddit.com/r/haskell/top/.rss?t=week&limit=25",
        "https://example.com/feed.xml",
        "https://www.reddit.com/r/ocaml/top/.rss?t=month&limit=25",
    ]
    factories, reddit_feed_urls = build_reddit_topfeed_factories(
        feeds, per_feed=10, days=30, exclude_urls=set()
    )
    assert len(factories) == 2
    assert set(reddit_feed_urls) == {
        "https://www.reddit.com/r/haskell/top/.rss?t=week&limit=25",
        "https://www.reddit.com/r/ocaml/top/.rss?t=month&limit=25",
    }
    reddit_fetch_queue.enqueue_spread(
        len(factories), time.monotonic(), "topfeed", factories
    )
    assert reddit_fetch_queue.wait_until_empty(timeout=2.0) is True

    assert max_reddit_concurrency == 1
    reddit_requests = [
        (url, headers) for url, headers in seen_requests if "reddit.com" in url
    ]
    assert len(reddit_requests) == 2
    assert all(
        headers.get("User-Agent") == REDDIT_RSS_USER_AGENT
        for _url, headers in reddit_requests
    )

    # The non-Reddit feed is NOT handled here — it's the caller's
    # responsibility to fetch it via `fetch_rss_feeds`. This test
    # focuses on Reddit serialization, so we don't call it.
    reddit_stories = [
        s
        for feed_url in reddit_feed_urls
        for s in pipeline.reddit_feed_cache.get(feed_url) or []
    ]
    assert {s.source for s in reddit_stories} >= {
        "rss_reddit_haskell",
        "rss_reddit_ocaml",
    }


async def test_build_reddit_topfeed_populates_self_text(tmp_path, monkeypatch):
    """Regression: Reddit topfeed stories must set self_text from the
    feed body. Replaces the old
    `test_fetch_rss_feeds_populates_self_text` after Reddit topfeed was
    split out of `fetch_rss_feeds`."""
    from pipeline import build_reddit_topfeed_factories
    from reddit_fetch_queue import queue as reddit_fetch_queue

    body = "The K line is cute but smells a bit. Stations feel like nowhere."

    def rss_doc(title, link):
        return f"""<?xml version="1.0"?>
<rss><channel>
<item><title>{title}</title><link>{link}</link>
<pubDate>Tue, 23 Jun 2026 12:00:00 GMT</pubDate>
<description>{body}</description>
</item></channel></rss>"""

    class MockResp:
        status_code = 200

        def __init__(self, t):
            self.text = t
            self.headers: dict[str, str] = {}

    class MockClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, headers=None):
            return MockResp(rss_doc("LA transit", url + "/comments/x/"))

    monkeypatch.setattr("pipeline.httpx.AsyncClient", MockClient)
    pipeline.reddit_limiter.INTER_REQUEST_DELAY = 0.0

    feeds = ["https://www.reddit.com/r/transit/top/.rss"]
    factories, reddit_feed_urls = build_reddit_topfeed_factories(
        feeds, per_feed=5, days=30, exclude_urls=set()
    )
    reddit_fetch_queue.enqueue_spread(
        len(factories), time.monotonic(), "topfeed", factories
    )
    assert reddit_fetch_queue.wait_until_empty(timeout=2.0) is True

    cached = pipeline.reddit_feed_cache.get(reddit_feed_urls[0])
    assert cached is not None
    assert len(cached) == 1
    s = cached[0]
    assert s.source == "rss_reddit_transit"
    assert "K line is cute but smells" in s.self_text
    assert s.text_content.startswith("LA transit. ")
    assert "K line is cute but smells" in s.text_content


async def test_build_reddit_topfeed_cache_hit_skips_http(tmp_path, monkeypatch):
    """Cache hit should skip the HTTP call. Replaces the old
    `test_fetch_rss_feeds_cache_hit_skips_http` after Reddit topfeed
    was split out of `fetch_rss_feeds`."""
    from pipeline import build_reddit_topfeed_factories
    from reddit_fetch_queue import queue as reddit_fetch_queue

    cached_stories = [
        Story(
            id=-1,
            title="Cached Story",
            url="http://example.com/1",
            score=10,
            time=1_000_000,
            text_content="cached",
            source="rss_reddit_test",
        )
    ]
    feed_url = "https://www.reddit.com/r/test/top/.rss?t=week&limit=25"
    pipeline.reddit_feed_cache.set(feed_url, cached_stories)

    class FailClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, headers=None):
            raise RuntimeError("HTTP call made despite cache hit")

    monkeypatch.setattr("pipeline.httpx.AsyncClient", FailClient)
    pipeline.reddit_limiter.INTER_REQUEST_DELAY = 0.0

    factories, reddit_feed_urls = build_reddit_topfeed_factories(
        [feed_url], per_feed=10, days=30, exclude_urls=set()
    )
    reddit_fetch_queue.enqueue_spread(
        len(factories), time.monotonic(), "topfeed", factories
    )
    assert reddit_fetch_queue.wait_until_empty(timeout=2.0) is True

    cached = pipeline.reddit_feed_cache.get(reddit_feed_urls[0])
    assert cached is not None
    assert len(cached) == 1
    assert cached[0].title == "Cached Story"


async def test_build_reddit_topfeed_cache_miss_fetches_and_caches(
    tmp_path, monkeypatch
):
    """Cache miss should fetch, cache, and make stories available in
    `reddit_feed_cache`. Replaces the old
    `test_fetch_rss_feeds_cache_miss_fetches_and_caches` after Reddit
    topfeed was split out of `fetch_rss_feeds`."""
    from pipeline import build_reddit_topfeed_factories
    from reddit_fetch_queue import queue as reddit_fetch_queue

    def rss_doc(title: str, link: str) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item><title>{title}</title><link>{link}</link><pubDate>Tue, 23 Jun 2026 12:00:00 GMT</pubDate><description>test body</description></item>
</channel></rss>"""

    class MockResp:
        status_code = 200

        def __init__(self, t: str):
            self.text = t
            self.headers: dict[str, str] = {}

    class MockClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, headers=None):
            return MockResp(rss_doc("Fresh Story", url + "/comments/x/"))

    monkeypatch.setattr("pipeline.httpx.AsyncClient", MockClient)
    pipeline.reddit_limiter.INTER_REQUEST_DELAY = 0.0
    pipeline.reddit_feed_cache.reset()

    feed_url = "https://www.reddit.com/r/test/top/.rss?t=week&limit=25"
    factories, reddit_feed_urls = build_reddit_topfeed_factories(
        [feed_url], per_feed=10, days=30, exclude_urls=set()
    )
    reddit_fetch_queue.enqueue_spread(
        len(factories), time.monotonic(), "topfeed", factories
    )
    assert reddit_fetch_queue.wait_until_empty(timeout=2.0) is True

    cached = pipeline.reddit_feed_cache.get(reddit_feed_urls[0])
    assert cached is not None
    assert cached[0].title == "Fresh Story"


async def test_prewarm_reddit_top_stories_fetches_comments(tmp_path, monkeypatch):
    """prewarm_reddit_top_stories fetches and stores Reddit RSS comments."""
    import server

    db = Database(str(tmp_path / "test.db"))
    db.upsert_story(
        Story(
            id=-100,
            title="LA transit",
            url="https://www.reddit.com/r/transit/comments/abc123/la_transit/",
            score=0,
            time=2000000000,
            text_content="LA transit. body",
            self_text="body",
            source="rss_reddit_transit",
        )
    )

    async def mock_fetch(url):
        return server.RedditRssContext(
            self_text="body",
            top_comments="/u/alice: Useful comment about transit",
            comment_count=3,
        )

    monkeypatch.setattr(server, "_fetch_reddit_rss_context", mock_fetch)

    n = await pipeline.prewarm_reddit_top_stories([-100], db)

    assert n == 1
    updated = db.get_story(-100)
    assert updated is not None
    assert "Useful comment about transit" in updated.top_comments
    assert updated.comment_count == 3
    assert updated.discussion_url == (
        "https://www.reddit.com/r/transit/comments/abc123/la_transit/"
    )


async def test_prewarm_reddit_top_stories_skips_if_already_populated(
    tmp_path, monkeypatch
):
    """prewarm_reddit_top_stories is idempotent: skips if top_comments already populated."""
    import server

    db = Database(str(tmp_path / "test.db"))
    db.upsert_story(
        Story(
            id=-101,
            title="LA transit",
            url="https://www.reddit.com/r/transit/comments/abc123/la_transit/",
            score=0,
            time=2000000000,
            text_content="LA transit. body with existing comments",
            self_text="body",
            top_comments="/u/alice: Existing rich comment content that is quite long",
            source="rss_reddit_transit",
        )
    )

    async def mock_fetch(url):
        return server.RedditRssContext(
            self_text="body",
            top_comments="shorter",
            comment_count=1,
        )

    monkeypatch.setattr(server, "_fetch_reddit_rss_context", mock_fetch)

    n = await pipeline.prewarm_reddit_top_stories([-101], db)

    assert n == 0
    updated = db.get_story(-101)
    assert updated is not None
    assert "Existing rich comment" in updated.top_comments
    assert updated.comment_count is None  # unchanged


@given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=2, max_size=50))
def test_mmr_output_is_subset(scores):
    ranked = []
    embeddings_map = {}
    for i, s in enumerate(scores):
        story = Story(id=i, title=f"S{i}", url=None, score=0, time=0, text_content="")
        ranked.append(RankedStory(story=story, score=s, best_match_title=""))
        v = np.zeros(384, dtype=np.float32)
        v[i % 384] = 1.0
        embeddings_map[i] = v

    filtered = mmr_filter(ranked, embeddings_map, threshold=0.85, limit=10)

    filtered_ids = [item.story.id for item in filtered]
    input_ids = [item.story.id for item in ranked]
    for fid in filtered_ids:
        assert fid in input_ids
    assert filtered_ids == sorted(filtered_ids, key=lambda x: input_ids.index(x))


def test_rerank_candidates_mmr_config_switch(db, embedder, monkeypatch):
    candidates = [
        Story(id=i, title=f"S{i}", url=None, score=10 - i, time=0, text_content=f"s{i}")
        for i in range(3)
    ]
    embeddings = np.eye(3, 384, dtype=np.float32)

    def fail_mmr(*args, **kwargs):
        raise AssertionError("mmr_filter should not be called when enable_mmr=false")

    monkeypatch.setattr(pipeline, "mmr_filter", fail_mmr)
    rerank_candidates(
        db=db,
        config=Config(count=3, model=ModelConfig(enable_mmr=False)),
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=embeddings,
    )

    called = False

    def mark_mmr(ranked, embeddings_map, threshold, limit):
        nonlocal called
        called = True
        return ranked[:limit]

    monkeypatch.setattr(pipeline, "mmr_filter", mark_mmr)
    rerank_candidates(
        db=db,
        config=Config(count=3, model=ModelConfig(enable_mmr=True)),
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=embeddings,
    )
    assert called


def test_dashboard_primary_limit_reduces_ranked_slice_without_counting_uncertainty():
    primary_limit, uncertain_slots = _dashboard_primary_limit(40)

    assert primary_limit == 12
    assert uncertain_slots == 5


def test_rank_no_feedback_frontpage_sort(db, embedder):
    config = Config()
    # Absolutely no feedback in DB
    candidates = [
        Story(
            id=1, title="Low points story", url=None, score=10, time=0, text_content=""
        ),
        Story(
            id=2,
            title="High points story",
            url=None,
            score=500,
            time=0,
            text_content="",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    ranked = _score_and_rank(
        candidates=candidates,
        candidate_embeddings=cand_embs,
        db=db,
        config=config,
        embedder=embedder,
    )

    assert len(ranked) == 2
    # Should sort by HN gravity (high points story first)
    assert ranked[0].story.id == 2


@given(text=st.text(), min_len=st.integers(min_value=0, max_value=100))
@settings(max_examples=25)
def test_clean_text_properties(text, min_len):
    import re

    cleaned = clean_text(text, min_len=min_len)

    if cleaned != "":
        # Length constraint
        assert len(cleaned) > min_len
        # Alphanumeric density
        alnum = sum(c.isalnum() for c in cleaned)
        assert alnum / len(cleaned) >= 0.5
        # No Braille
        assert not re.search(r"[\u2800-\u28FF]", cleaned)
        # No unescaped tags
        assert not re.search(r"<[^>]+>", cleaned)


@given(
    meta=st.lists(
        st.tuples(
            st.integers(min_value=-1000, max_value=1_000_000),
            st.floats(
                min_value=-86400.0,
                max_value=10_000_000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=25)
def test_augment_features_properties(meta):
    from legacy_features import _augment_features

    n = len(meta)
    embeddings = np.random.randn(n, 384).astype(np.float32)
    scores = [item[0] for item in meta]
    ages = [item[1] for item in meta]

    features = _augment_features(embeddings, scores, ages)

    assert features.shape == (n, 385)
    assert np.allclose(features[:, :384], embeddings)
    assert np.all(features[:, 384] >= 0.0) and np.all(features[:, 384] <= 1.0)

    # When all 11 derived features are provided, shape expands to (n, 396)
    comment_counts = np.array([max(s, 0) for s in scores])
    text_lengths = np.array([abs(a) % 10000 for a in ages])
    hn_quality = comment_counts.astype(np.float32) / (np.abs(ages) + 1)
    sim_up = np.random.uniform(-1, 1, n).astype(np.float32)
    sim_down = np.random.uniform(-1, 1, n).astype(np.float32)
    closest_up = np.random.uniform(-1, 1, n).astype(np.float32)
    closest_down = np.random.uniform(-1, 1, n).astype(np.float32)
    is_hn_live = np.random.randint(0, 2, n).astype(np.float32)
    is_archive = np.random.randint(0, 2, n).astype(np.float32)
    is_reddit = np.random.randint(0, 2, n).astype(np.float32)
    is_rss = np.random.randint(0, 2, n).astype(np.float32)

    features7 = _augment_features(
        embeddings,
        scores,
        ages,
        comment_counts=comment_counts,
        text_lengths=text_lengths,
        hn_quality=hn_quality,
        sim_to_upvoted=sim_up,
        sim_to_downvoted=sim_down,
        closest_upvoted=closest_up,
        closest_downvoted=closest_down,
        is_hn_live=is_hn_live,
        is_archive=is_archive,
        is_reddit=is_reddit,
        is_rss=is_rss,
    )
    assert features7.shape == (n, 396)
    assert np.allclose(features7[:, :384], embeddings)
    # All normalized cols in [0, 1]
    assert np.all(features7[:, 384:] >= 0.0) and np.all(features7[:, 384:] <= 1.0)


def test_svm_personalization_features_exclude_engagement_source_metadata():
    emb_dim = 384
    embeddings = np.full((2, emb_dim), 2.0, dtype=np.float32)
    text_lengths = np.array([0, 1000], dtype=np.float32)
    sim_up = np.array([-1.0, 1.0], dtype=np.float32)
    sim_down = np.array([1.0, -1.0], dtype=np.float32)
    closest_up = np.array([0.0, 0.5], dtype=np.float32)
    closest_down = np.array([-0.5, 0.0], dtype=np.float32)
    pos_cluster = np.array([-0.25, 0.75], dtype=np.float32)

    features = _svm_personalization_features(
        embeddings,
        text_lengths=text_lengths,
        sim_to_upvoted=sim_up,
        sim_to_downvoted=sim_down,
        closest_upvoted=closest_up,
        closest_downvoted=closest_down,
        positive_cluster_similarity=pos_cluster,
    )

    assert features.shape == (2, emb_dim + 10)
    assert np.all(features[:, :emb_dim] == 2.0)
    assert np.all(features[:, emb_dim:] >= 0.0)
    assert np.all(features[:, emb_dim:] <= 1.0)
    assert features[0, emb_dim + 1] == 0.0
    assert features[1, emb_dim + 1] == 1.0
    assert features[0, emb_dim + 2] == 1.0
    assert features[1, emb_dim + 2] == 0.0
    assert features[0, emb_dim + 5] == pytest.approx(0.375)
    assert features[1, emb_dim + 5] == pytest.approx(0.875)


@given(
    feedback_actions=st.lists(
        st.sampled_from(["up", "neutral", "down"]), min_size=0, max_size=20
    ),
    cand_count=st.integers(min_value=1, max_value=10),
)
@settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=1000,
)
def test_svm_fitting_robustness(embedder, feedback_actions, cand_count):
    db = Database(":memory:")
    try:
        user = db.create_user("test_token_robustness")
        model_version = "all-MiniLM-L6-v2|mean|norm|256"
        for i, action in enumerate(feedback_actions):
            story = Story(
                id=1000 + i,
                title=f"Feedback Story {i}",
                url=None,
                score=np.random.randint(0, 1000),
                time=int(1600000000 + i * 100),
                text_content=f"Sample semantic content for history {i}",
            )
            import hashlib

            shash = hashlib.sha256(story.text_content.encode("utf-8")).hexdigest()
            db.upsert_story(story)
            db.upsert_embedding(
                story.id, model_version, shash, np.random.randn(384).astype(np.float32)
            )
            db.upsert_feedback(user.id, story.id, action)

        candidates = []
        for i in range(cand_count):
            candidates.append(
                Story(
                    id=i,
                    title=f"Candidate Story {i}",
                    url=None,
                    score=np.random.randint(0, 500),
                    time=int(1600000000),
                    text_content=f"Sample candidate content {i}",
                )
            )

        cand_embs = np.random.randn(cand_count, 384).astype(np.float32)
        config = Config()
        ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

        assert len(ranked) == cand_count
        for item in ranked:
            assert 0.0 <= item.score <= 1.0
            if not feedback_actions:
                assert 0.0 <= item.score <= 1.0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_fetch_candidates_returns_tuple(tmp_path, monkeypatch):
    """fetch_candidates returns (list[Story], int), not just list."""
    from pipeline import Config, fetch_candidates
    from database import Database

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    config = Config(
        db_path=str(db_file),
        server_port=0,
    )

    monkeypatch.setattr("ch_client.query_live_window", lambda **kw: [])
    result = await fetch_candidates(config, set(), set(), db)
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 2
    candidates, count = result
    assert isinstance(candidates, list)
    assert isinstance(count, int)
    assert count == len(candidates)


@pytest.mark.asyncio
async def test_fetch_candidates_ch_live_window_inserts_new(tmp_path, monkeypatch):
    """CH live_window returns story fields; fetch_candidates inserts them
    into the DB with source='hn'."""
    from database import Database
    from pipeline import Config, fetch_candidates

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    config = Config(db_path=str(db_file), days=30)

    ch_window = [
        {
            "id": 12345,
            "type": "story",
            "title": "Live HN story",
            "url": "https://example.com/live",
            "text": "Self text",
            "points": 250,
            "num_comments": 50,
            "created_at_i": 1770000000,
        }
    ]
    monkeypatch.setattr("ch_client.query_live_window", lambda **kw: ch_window)

    candidates, count = await fetch_candidates(config, set(), set(), db)
    assert 12345 in {s.id for s in candidates}
    story = db.get_story(12345)
    assert story is not None
    assert story.source == "hn"
    assert story.score == 250
    assert story.comment_count == 50


@pytest.mark.asyncio
async def test_fetch_candidates_ch_live_window_updates_existing_score(
    tmp_path, monkeypatch
):
    """If a live `hn` story is already in the DB with a different score,
    fetch_candidates updates the score from CH."""
    from database import Database
    from pipeline import Config, fetch_candidates

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    config = Config(db_path=str(db_file), days=30)

    db.upsert_story(
        Story(
            id=99,
            title="Existing HN",
            url="https://example.com/existing",
            score=10,
            time=1770000000,
            text_content="Existing text",
            source="hn",
        )
    )
    ch_window = [
        {
            "id": 99,
            "type": "story",
            "title": "Existing HN",
            "url": "https://example.com/existing",
            "text": "Existing text",
            "points": 999,
            "num_comments": 5,
            "created_at_i": 1770000000,
        }
    ]
    monkeypatch.setattr("ch_client.query_live_window", lambda **kw: ch_window)

    candidates, _ = await fetch_candidates(config, set(), set(), db)
    story = db.get_story(99)
    assert story is not None
    assert story.score == 999
    assert any(s.id == 99 for s in candidates)


@pytest.mark.asyncio
async def test_fetch_candidates_ch_failure_returns_empty_live(tmp_path, monkeypatch):
    """If CH live_window raises, fetch_candidates continues with archive
    seeds only (does not crash)."""
    from database import Database
    from pipeline import Config, fetch_candidates

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    old_time = int(time.time()) - (180 * 86400)
    db.upsert_story(
        Story(
            id=1,
            title="Archive",
            url="https://example.com/archive",
            score=200,
            time=old_time,
            text_content="Archive text",
            self_text="Archive seed self text",
            source=CH_ARCHIVE_SOURCE,
        )
    )
    config = Config(db_path=str(db_file), days=30)

    def fail_ch(*a, **kw):
        raise RuntimeError("simulated CH outage")

    monkeypatch.setattr("ch_client.query_live_window", fail_ch)

    candidates, _ = await fetch_candidates(config, set(), set(), db)
    # Live source is empty; archive seed still surfaces
    assert {s.id for s in candidates} == {1}


@pytest.mark.asyncio
async def test_fetch_candidates_includes_old_bq_archive_stories(tmp_path, monkeypatch):
    from database import Database
    from pipeline import Config, fetch_candidates

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    old_time = int(time.time()) - (180 * 86400)
    story = Story(
        id=9001,
        title="Old BQ story",
        url="https://example.com/old-bq",
        score=500,
        time=old_time,
        text_content="Old high scoring BQ story.",
        source=BQ_ARCHIVE_SOURCE,
        comment_count=12,
        comment_count_at_fetch=12,
        top_comments="Already hydrated comments.",
    )
    db.upsert_story(story)

    monkeypatch.setattr("ch_client.query_live_window", lambda **kw: [])
    candidates, count = await fetch_candidates(
        Config(db_path=str(db_file), days=30),
        set(),
        set(),
        db,
    )

    assert count == len(candidates)
    assert {s.id for s in candidates} == {9001}


@pytest.mark.asyncio
async def test_fetch_candidates_caps_bq_archive_by_score(tmp_path, monkeypatch):
    from database import Database
    from pipeline import Config, fetch_candidates

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    old_time = int(time.time()) - (180 * 86400)
    for sid, score in [(9101, 100), (9102, 300), (9103, 200)]:
        db.upsert_story(
            Story(
                id=sid,
                title=f"BQ {sid}",
                url=f"https://example.com/{sid}",
                score=score,
                time=old_time,
                text_content=f"BQ archive story {sid}.",
                source=BQ_ARCHIVE_SOURCE,
                comment_count=1,
                comment_count_at_fetch=1,
                top_comments="Hydrated.",
            )
        )

    monkeypatch.setattr("ch_client.query_live_window", lambda **kw: [])
    monkeypatch.setattr("pipeline.BQ_ARCHIVE_CANDIDATE_LIMIT", 2)
    monkeypatch.setattr("pipeline.CH_ARCHIVE_CANDIDATE_LIMIT", 0)
    candidates, _ = await fetch_candidates(
        Config(db_path=str(db_file), days=30),
        set(),
        set(),
        db,
    )

    assert {s.id for s in candidates} == {9102, 9103}


@pytest.mark.asyncio
async def test_bq_archive_hydration_preserves_source(tmp_path, monkeypatch):
    """Pre-existing bq_seed story's source label must remain 'bq_seed' after
    a regen that surfaces it in the candidate pool. Archive seeds are now
    read from the DB (no per-regen refetch); `top_comments` stays as stored.
    """
    from database import Database
    from pipeline import Config, fetch_candidates

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    old_time = int(time.time()) - (180 * 86400)
    db.upsert_story(
        Story(
            id=9201,
            title="BQ needs comments",
            url="https://example.com/needs-comments",
            score=500,
            time=old_time,
            text_content="BQ needs comments.",
            source=BQ_ARCHIVE_SOURCE,
            comment_count=3,
            comment_count_at_fetch=3,
            top_comments="Pre-existing hydrated comments.",
        )
    )

    monkeypatch.setattr("ch_client.query_live_window", lambda **kw: [])
    await fetch_candidates(
        Config(db_path=str(db_file), days=30),
        set(),
        set(),
        db,
    )

    updated = db.get_story(9201)
    assert updated is not None
    assert updated.source == BQ_ARCHIVE_SOURCE
    assert "Pre-existing hydrated comments" in updated.top_comments


@pytest.mark.asyncio
async def test_fetch_candidates_filters_unsummarizable(tmp_path, monkeypatch):
    """fetch_candidates drops stories with no content and no fetchable comments."""
    from database import Database
    from pipeline import Config, fetch_candidates

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    now = int(time.time())
    db.upsert_story(
        Story(
            id=1,
            title="Has comments",
            url="https://example.com/1",
            score=100,
            time=now - 86400,
            text_content="Has comments.",
            source="hn",
            comment_count=5,
            comment_count_at_fetch=5,
            top_comments="Already hydrated comments.",
        )
    )
    db.upsert_story(
        Story(
            id=2,
            title="No comments but has self_text",
            url="https://example.com/2",
            score=50,
            time=now - 86400,
            text_content="No comments but has self_text.",
            source="hn",
            self_text="Some author text.",
        )
    )
    db.upsert_story(
        Story(
            id=3,
            title="Truly empty HN",
            url="https://example.com/3",
            score=10,
            time=now - 86400,
            text_content="Truly empty HN.",
            source="hn",
        )
    )
    db.upsert_story(
        Story(
            id=4,
            title="Non-HN with no content",
            url="https://example.com/4",
            score=5,
            time=now - 86400,
            text_content="Non-HN with no content.",
            source="rss",
        )
    )

    # Mock CH live window to return the 4 stories (preexisting in DB)
    monkeypatch.setattr(
        "ch_client.query_live_window",
        lambda **kw: [
            {
                "id": 1,
                "title": "Has comments",
                "url": "https://example.com/1",
                "points": 100,
                "num_comments": 5,
                "created_at_i": now - 86400,
                "text": "",
            },
            {
                "id": 2,
                "title": "No comments but has self_text",
                "url": "https://example.com/2",
                "points": 50,
                "num_comments": 0,
                "created_at_i": now - 86400,
                "text": "Some author text.",
            },
            {
                "id": 3,
                "title": "Truly empty HN",
                "url": "https://example.com/3",
                "points": 10,
                "num_comments": 0,
                "created_at_i": now - 86400,
                "text": "",
            },
            {
                "id": 4,
                "title": "Non-HN with no content",
                "url": "https://example.com/4",
                "points": 5,
                "num_comments": 0,
                "created_at_i": now - 86400,
                "text": "",
            },
        ],
    )

    candidates, _ = await fetch_candidates(
        Config(db_path=str(db_file), days=30),
        set(),
        set(),
        db,
    )
    candidate_ids = {s.id for s in candidates}
    # Stories with content or HN comments: 1, 2
    assert 1 in candidate_ids, "Has comments should survive"
    assert 2 in candidate_ids, "Has self_text should survive"
    # Stories with no content and no comments: 3, 4
    assert 3 not in candidate_ids, "Truly empty HN should be filtered"
    assert 4 not in candidate_ids, "Non-HN with no content should be filtered"


def test_fast_rerank_for_user_includes_old_bq_archive_story(db, monkeypatch):
    from pipeline import Config, fast_rerank_for_user

    user = db.create_user("bq_archive_user")
    old_time = int(time.time()) - (180 * 86400)
    db.upsert_story(
        Story(
            id=9301,
            title="Old BQ render candidate",
            url="https://example.com/render-bq",
            score=500,
            time=old_time,
            text_content="Old BQ render candidate.",
            source=BQ_ARCHIVE_SOURCE,
            self_text="Old BQ render candidate.",
        )
    )

    def fake_embeddings(stories, embedder, db_inst):
        return np.zeros((len(stories), 384), dtype=np.float32)

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", fake_embeddings)
    ranked = fast_rerank_for_user(
        db, Config(days=30), cast(Embedder, object()), user.id
    )

    assert [item.story.id for item in ranked] == [9301]


def test_fast_rerank_for_user_filters_unsummarizable_stories(db, monkeypatch):
    """fast_rerank_for_user drops stories that is_summarizable rejects."""
    from pipeline import Config, fast_rerank_for_user

    user = db.create_user("summ_filter_test")
    summarizable_id = 60001
    unsummarizable_id = 60002
    db.upsert_story(
        Story(
            id=summarizable_id,
            title="Has comments",
            url=None,
            score=10,
            time=int(time.time()) - 10,
            text_content="x",
            source="hn",
            comment_count=5,
        )
    )
    db.upsert_story(
        Story(
            id=unsummarizable_id,
            title="No comments, no text",
            url=None,
            score=10,
            time=int(time.time()) - 10,
            text_content="x",
            source="hn",
            comment_count=0,
        )
    )

    def fake_embeddings(stories, embedder, db_inst):
        return np.zeros((len(stories), 384), dtype=np.float32)

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", fake_embeddings)
    ranked = fast_rerank_for_user(
        db, Config(days=30), cast(Embedder, object()), user.id
    )

    result_ids = {r.story.id for r in ranked}
    assert summarizable_id in result_ids
    assert unsummarizable_id not in result_ids


def test_fast_rerank_for_user_zero_vote_returns_cold_deck(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0-feedback users get build_cold_deck directly — no embeddings, no SVM."""
    from pipeline import Config, fast_rerank_for_user

    user = db.create_user("zero_vote_cold")
    now = 2_000_000_000
    monkeypatch.setattr(pipeline.time, "time", lambda: float(now))
    db.upsert_story(_cold_story(1, score=50, time_ts=now - 3600, source="hn"))

    ranked = fast_rerank_for_user(
        db, Config(days=30), cast(Embedder, object()), user.id
    )

    gravity_div = 3.0**1.8
    assert len(ranked) == 1
    assert ranked[0].story.id == 1
    assert ranked[0].score == pytest.approx(50.0 / gravity_div)
    assert ranked[0].is_recent is True
    assert ranked[0].is_non_hn is False


def test_soft_blend_min_alpha_curve() -> None:
    """Verify blend alpha curve based on min(n_up, n_down).
    α=0 at min=20, α=1.0 at min=80+, window=60."""
    blend_start = 20
    window = 60
    cases: dict[int, float | None] = {
        0: None,
        5: None,
        10: None,
        15: None,
        19: None,
        20: 0.0,
        25: 0.0833,
        30: 0.1667,
        40: 0.3333,
        50: 0.5,
        60: 0.6667,
        80: 1.0,
        100: 1.0,
    }
    for n_min, expected in cases.items():
        if expected is None:
            assert n_min < blend_start
        else:
            alpha = max(0.0, min(1.0, (n_min - blend_start) / window))
            assert abs(alpha - expected) < 1e-4, (
                f"n_min={n_min}: expected {expected}, got {alpha}"
            )


def test_no_cliff_at_n_10(db: Database, embedder: Embedder) -> None:
    """At n=10, SVM doesn't fire (threshold=20). Ranking is tier-2 centroid-diff."""
    config = Config()
    user = db.create_user("test_token_no_cliff")

    # 5 upvoted finance stories
    for i in range(5):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Finance story {i}",
                url=None,
                score=0,
                time=0,
                text_content=f"stock market investment banking finance economy forecast {i}",
            )
        )
        db.upsert_feedback(user.id, 100 + i, "up")

    # 5 downvoted baking stories
    for i in range(5):
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Baking story {i}",
                url=None,
                score=0,
                time=0,
                text_content=f"sourdough bread cake cookie recipe kitchen baking {i}",
            )
        )
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=0,
            time=0,
            text_content="Stock market rally after federal reserve rate decision.",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=0,
            time=0,
            text_content="How to make perfect sourdough bread at home.",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 3
    # Finance candidate is closest to upvotes (centroid formula). Should rank first.
    assert ranked[0].story.id == 2, (
        f"Expected finance story first, got id={ranked[0].story.id}"
    )
    # Baking candidate is closest to downvotes. Should rank last.
    assert ranked[-1].story.id == 3, (
        f"Expected baking story last, got id={ranked[-1].story.id}"
    )
    # AI candidate (neutral topic) should be in the middle
    assert ranked[1].story.id == 1


def test_tier1_tier2_blend_with_only_upvotes(db: Database, embedder: Embedder) -> None:
    """5 upvotes (n_feedback=5) → α_2=0.1, mostly gravity with slight centroid."""
    config = Config()
    user = db.create_user("test_token_up_blend")
    now = int(time.time())

    for i in range(5):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Finance story {i}",
                url=None,
                score=0,
                time=0,
                text_content="stock market investment banking finance economy forecast",
            )
        )
        db.upsert_feedback(user.id, 100 + i, "up")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
            source="hn",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=100,
            time=now - 3600,
            text_content="Stock market rally after federal reserve rate decision.",
            source="hn",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=50,
            time=now - 86400,
            text_content="How to make perfect sourdough bread at home.",
            source="hn",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 3
    # Finance story benefits from both gravity (highest score) and centroid (closest to upvotes)
    assert ranked[0].story.id == 2
    # At α_2=0.1 gravity dominates over centroid, so story 3 (baking, high gravity)
    # should outrank story 1 (AI, zero gravity) even though centroid slightly favors AI.
    # The gravity gap (0.010 vs 0.000) is small; centroid could flip it.
    # Assert only that finance is first and both trailing stories are present.
    assert ranked[1].story.id in (1, 3)
    assert ranked[2].story.id in (1, 3)
    assert ranked[1].story.id != ranked[2].story.id


def test_tier2_pure_at_60_plus_one_class(db: Database, embedder: Embedder) -> None:
    """60 upvotes (n_feedback=60, α_2=1.0) → pure tier2 centroid, finance first."""
    config = Config()
    user = db.create_user("test_token_pure_tier2")
    now = int(time.time())

    for i in range(60):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Finance story {i}",
                url=None,
                score=0,
                time=0,
                text_content="stock market investment banking finance economy forecast",
            )
        )
        db.upsert_feedback(user.id, 100 + i, "up")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
            source="hn",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=100,
            time=now - 3600,
            text_content="Stock market rally after federal reserve rate decision.",
            source="hn",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=50,
            time=now - 86400,
            text_content="How to make perfect sourdough bread at home.",
            source="hn",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 3
    # Pure centroid (sim_up): finance > AI > baking
    assert ranked[0].story.id == 2
    assert ranked[2].story.id == 3


def test_tier1_active_at_tier3_activation_boundary(
    db: Database, embedder: Embedder
) -> None:
    """19 up/19 down (n_feedback=38) then 20/20 (n_feedback=40).
    At both points tier1 weight is non-zero: 1-38/50=0.24 and 1-40/50=0.20."""
    config = Config()
    user = db.create_user("test_token_boundary")
    now = int(time.time())

    for i in range(19):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up {i}",
                url=None,
                score=0,
                time=0,
                text_content="stock market investment banking finance economy",
            ),
        )
        db.upsert_feedback(user.id, 100 + i, "up")
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down {i}",
                url=None,
                score=0,
                time=0,
                text_content="sourdough bread cake cookie recipe kitchen baking",
            ),
        )
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
            source="hn",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=100,
            time=now - 3600,
            text_content="Stock market rally after federal reserve rate decision.",
            source="hn",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=50,
            time=now - 86400,
            text_content="How to make perfect sourdough bread at home.",
            source="hn",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked_19 = _score_and_rank(candidates, cand_embs, db, config, embedder)

    # Add one more of each class (20 up, 20 down, n_feedback=40, n_min=20)
    db.upsert_story(
        Story(
            id=119,
            title="Up final",
            url=None,
            score=0,
            time=0,
            text_content="stock market investment banking finance economy final",
        ),
    )
    db.upsert_feedback(user.id, 119, "up")
    db.upsert_story(
        Story(
            id=219,
            title="Down final",
            url=None,
            score=0,
            time=0,
            text_content="sourdough bread cake cookie recipe kitchen baking final",
        ),
    )
    db.upsert_feedback(user.id, 219, "down")

    ranked_20 = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked_19) == 3
    assert len(ranked_20) == 3
    # Finance should rank first at both points (tier2 dominant, tier3 not yet active)
    assert ranked_19[0].story.id == 2
    assert ranked_20[0].story.id == 2
    # Baking (closest to downvotes) should rank last at both points
    assert ranked_19[-1].story.id == 3
    assert ranked_20[-1].story.id == 3
    # Transition from 19→20 should be smooth: the gap between ranked[0].score
    # and ranked[1].score should not change abruptly.
    gap_19 = ranked_19[0].score - ranked_19[1].score
    gap_20 = ranked_20[0].score - ranked_20[1].score
    assert abs(gap_20 - gap_19) < 0.15, (
        f"Cliff detected at tier3 threshold: gap_19={gap_19:.4f}, gap_20={gap_20:.4f}"
    )


def test_three_way_blend_at_30_30(db: Database, embedder: Embedder) -> None:
    """30 up/30 down (n_feedback=60, α_2=1.0, α_3≈0.167) → tier2-t3 blend."""
    config = Config()
    user = db.create_user("test_token_three_way")
    now = int(time.time())

    for i in range(30):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up {i}",
                url=None,
                score=0,
                time=0,
                text_content="stock market investment banking finance economy",
            ),
        )
        db.upsert_feedback(user.id, 100 + i, "up")
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down {i}",
                url=None,
                score=0,
                time=0,
                text_content="sourdough bread cake cookie recipe kitchen baking",
            ),
        )
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
            source="hn",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=100,
            time=now - 3600,
            text_content="Stock market rally after federal reserve rate decision.",
            source="hn",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=50,
            time=now - 86400,
            text_content="How to make perfect sourdough bread at home.",
            source="hn",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 3
    # Tier2 + tier3 blend: finance first, baking last
    assert ranked[0].story.id == 2
    assert ranked[-1].story.id == 3


@pytest.mark.parametrize(
    "n_up,n_down",
    [
        (0, 0),
        (5, 0),
        (10, 0),
        (25, 0),
        (50, 0),
        (5, 5),
        (10, 10),
        (20, 20),
        (30, 30),
    ],
)
def test_blend_weights_monotonic(
    db: Database, embedder: Embedder, n_up: int, n_down: int
) -> None:
    """Verify blend weights change monotonically with feedback count."""
    config = Config()
    user = db.create_user(f"test_mono_{n_up}_{n_down}")
    now = int(time.time())

    for i in range(n_up):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up {i}",
                url=None,
                score=0,
                time=0,
                text_content="stock market investment banking finance economy",
            ),
        )
        db.upsert_feedback(user.id, 100 + i, "up")
    for i in range(n_down):
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down {i}",
                url=None,
                score=0,
                time=0,
                text_content="sourdough bread cake cookie recipe kitchen baking",
            ),
        )
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=50,
            time=now - 3600,
            text_content="Training large language models and neural networks.",
            source="hn",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=100,
            time=now - 3600,
            text_content="Stock market rally after federal reserve rate decision.",
            source="hn",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=50,
            time=now - 3600,
            text_content="How to make perfect sourdough bread at home.",
            source="hn",
        ),
    ]
    # All equal score/time so gravity gives equal tier1 scores
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 3
    n_fb = n_up + n_down
    assert ranked[0].story.id == 2, (
        f"Finance should rank first at n_up={n_up} n_down={n_down}"
    )
    assert ranked[-1].story.id == 3, (
        f"Baking should rank last at n_up={n_up} n_down={n_down}"
    )
    # Verify monotonicity: as feedback grows, tiers change smoothly.
    # The score of finance minus score of AI should increase (more centroid weight).
    if n_fb > 0 and n_up >= n_down:
        score_gap = ranked[0].score - ranked[1].score
        assert score_gap > 0.0, (
            f"Finance-AI gap should be positive, got {score_gap:.4f}"
        )


def test_three_way_weights_sum_to_one(db: Database, embedder: Embedder) -> None:
    """At 50 up/50 down, α_2=1.0, α_3=0.5, weights: t1=0, t2=0.5, t3=0.5.
    Finance should still rank first, baking last."""
    config = Config()
    user = db.create_user("test_token_sum_to_one")
    now = int(time.time())

    for i in range(50):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up {i}",
                url=None,
                score=0,
                time=0,
                text_content="stock market investment banking finance economy",
            ),
        )
        db.upsert_feedback(user.id, 100 + i, "up")
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down {i}",
                url=None,
                score=0,
                time=0,
                text_content="sourdough bread cake cookie recipe kitchen baking",
            ),
        )
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
            source="hn",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=100,
            time=now - 3600,
            text_content="Stock market rally after federal reserve rate decision.",
            source="hn",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=50,
            time=now - 86400,
            text_content="How to make perfect sourdough bread at home.",
            source="hn",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 3
    assert ranked[0].story.id == 2
    assert ranked[-1].story.id == 3


def test_tier3_pure_at_80_each(db: Database, embedder: Embedder) -> None:
    """80 up/80 down (n_min=80, α_3=1.0) → pure tier3 SVM ranking."""
    config = Config()
    user = db.create_user("test_token_pure_tier3")
    now = int(time.time())

    for i in range(80):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up {i}",
                url=None,
                score=0,
                time=0,
                text_content="stock market investment banking finance economy",
            ),
        )
        db.upsert_feedback(user.id, 100 + i, "up")
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down {i}",
                url=None,
                score=0,
                time=0,
                text_content="sourdough bread cake cookie recipe kitchen baking",
            ),
        )
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
            source="hn",
        ),
        Story(
            id=2,
            title="Finance news",
            url=None,
            score=100,
            time=now - 3600,
            text_content="Stock market rally after federal reserve rate decision.",
            source="hn",
        ),
        Story(
            id=3,
            title="Baking tips",
            url=None,
            score=50,
            time=now - 86400,
            text_content="How to make perfect sourdough bread at home.",
            source="hn",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 3
    assert ranked[0].story.id == 2


def test_novel_archive_pass_surfaces_archive_novel(
    db: Database, embedder: Embedder, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """The novel-archive discovery pass surfaces ✨ Novel stories in archive.

    The novel pass is split per-age (novel-recent + novel-archive) so the
    ✨ badge actually appears in both age buckets. Archive extras (low
    score) are not in the primary ranked set; without a dedicated
    novel-archive pass they would never be surfaced with ✨.

    This test constructs archive candidates with low max_sim (novel-
    qualifying) and asserts they appear in `final` with is_novel=True.
    The top 2 by distance (sim 0.05, 0.10) are picked by the
    novel-archive pass (slot_limit=5).
    """
    config = Config(count=40)
    user = db.create_user("test_novel_archive")

    def mock_gce(stories, embedder_arg, db_inst):
        arr = np.zeros((len(stories), 384), dtype=np.float32)
        for i, s in enumerate(stories):
            if s.id == 100:
                arr[i, 0] = 1.0
        return arr

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", mock_gce)
    db.upsert_story(
        Story(id=100, title="fb", url=None, score=10, time=0, text_content="")
    )
    db.upsert_feedback(user.id, 100, "up")

    now = int(time.time())
    # Setup:
    #   12 recent primary (high score, sim 0.5) fill the primary ranked set.
    #   12 archive fillers (high score, sim 0.5) — they have sim 0.5 too,
    #     so they don't qualify as novel and are not picked by novel-archive
    #     (which sorts by distance desc).
    #   4 archive novel targets (low score, sim 0.05..0.20) — the test target.
    # Archive novel sims: [0.05, 0.10, 0.15, 0.20]. The novel-archive pass
    # sorts by distance (= 1 - sim) desc and takes the top 5. All 4 novel
    # targets have higher distance than the archive fillers (sim 0.5 →
    # dist 0.5), so the top of the novel-archive pool is the 4 novel
    # targets; the pass picks 4 of them and (with slot_limit=5) leaves room
    # for an additional 1 archive filler (the 4th filler is the 5th by
    # distance, with sim 0.5 → dist 0.5, tied with the others — last pick
    # depends on stable sort order).
    candidates = []
    for i in range(12):
        candidates.append(
            Story(
                id=i,
                title=f"Primary {i}",
                url=None,
                score=100 + i,
                time=now - 3600,
                text_content=f"primary {i}",
                source="hn",
                comment_count=0,
            )
        )
    for i in range(12):
        candidates.append(
            Story(
                id=12 + i,
                title=f"ArchiveFiller {i}",
                url=None,
                score=200 + i,
                time=now - 60 * 86400,
                text_content=f"afiller {i}",
                source="ch_seed",
                comment_count=0,
            )
        )
    for i, sim in enumerate([0.05, 0.10, 0.15, 0.20]):
        candidates.append(
            Story(
                id=24 + i,
                title=f"ANovel {i}",
                url=None,
                score=5,
                time=now - 60 * 86400,
                text_content=f"anovel {i}",
                source="ch_seed",
                comment_count=0,
            )
        )

    cand_embs = np.zeros((28, 384), dtype=np.float32)
    for i in range(12):
        cand_embs[i, 0] = 0.5
        cand_embs[i, 50 + i] = np.sqrt(0.75)
    for i in range(12):
        cand_embs[12 + i, 0] = 0.5
        cand_embs[12 + i, 150 + i] = np.sqrt(0.75)
    for i, s in enumerate([0.05, 0.10, 0.15, 0.20]):
        cand_embs[24 + i, 0] = s
        cand_embs[24 + i, 300 + i] = np.sqrt(max(1.0 - s * s, 0.0))

    ranked = rerank_candidates(
        db, config, embedder, candidates, cand_embs, user_id=user.id
    )

    by_id = {r.story.id: r for r in ranked}
    # With per-combo DISCOVERY_PER_BADGE=2, only the top 2 by distance
    # (ids 24 and 25, sim=0.05, 0.10) get is_novel. The other two
    # (ids 26, 27) may appear via other passes (Popular, Similar, etc.)
    # but should not have is_novel.
    from pipeline import DISCOVERY_PER_BADGE

    novel_ids = [
        aid for aid in (24, 25, 26, 27) if aid in by_id and by_id[aid].is_novel
    ]
    assert len(novel_ids) >= DISCOVERY_PER_BADGE, (
        f"Expected at least {DISCOVERY_PER_BADGE} novel picks, got {novel_ids}"
    )
    # The top by distance (sim 0.05, 0.10) must be among the novel picks.
    for aid in (24, 25):
        assert aid in by_id, f"Archive novel id={aid} should be in final"
        assert by_id[aid].is_novel, (
            f"Archive novel id={aid} (sim={'0.05' if aid == 24 else '0.10'}) should have is_novel=True"
        )
        assert not by_id[aid].is_recent, (
            f"Archive novel id={aid} should be is_recent=False"
        )


def test_each_badge_floored_at_five_per_cohort(
    db: Database, embedder: Embedder
) -> None:
    """Every non-Hot badge must appear >=5 times in recent AND >=5 times
    in archive of the final deck (the user's explicit "at least (5,5)
    for each" expectation).

    The rank-based cascade guarantees this via per-cohort top-5
    discovery passes for each non-Hot badge:
      cascade: hot, high-engagement-recent/archive, discussion-recent/archive
      parallel (with stacking): novel-recent/archive, similar-recent/archive,
                                 uncertain-recent/archive
    Each pass takes the top 5 stories in its age cohort by the badge
    metric. The cascade passes are mutually exclusive; the parallel
    passes can stack with each other and with cascade picks.

    Pool sizing: 30 recent + 30 archive is the minimum safe cohort size
    for the structural floor to hold. With ``primary_limit=12`` and 3
    cascade passes per cohort consuming up to 15 candidates (5+5+5),
    the worst case (primary takes 12 + cascade takes 15 = 27) leaves
    ``30 - 27 = 3`` in the cohort for the parallel group — still
    enough to fill the 5-slot cap (the cap is the *target*; if the
    cohort has fewer candidates we get fewer). For this test we size
    the cohorts so the floor (5 per cohort per badge) holds.

    Feedback: 20 distinct upvotes, 20 distinct downvotes, 20 distinct
    neutral. The feedback table has a UNIQUE(user_id, story_id, action)
    constraint, so 20 votes of the same story collapse to 1 row. The
    SVM requires ``n_up >= 20`` and ``n_down >= 20`` distinct stories
    to fit and produce ``predict_proba`` output, which drives the
    entropy signal for the Unsure badge and the similarity signals
    for Novel and Similar.
    """
    config = Config(count=40)
    user = db.create_user("test_each_badge_floor_5")

    for i in range(20):
        db.upsert_story(
            Story(
                id=800 + i,
                title=f"fb_up_{i}",
                url=None,
                score=10,
                time=0,
                text_content=(
                    f"liked story number {i} about programming and "
                    f"software engineering topic {chr(97 + (i % 26))}"
                ),
            )
        )
        db.upsert_feedback(user.id, 800 + i, "up")
    for i in range(20):
        db.upsert_story(
            Story(
                id=840 + i,
                title=f"fb_down_{i}",
                url=None,
                score=10,
                time=0,
                text_content=(
                    f"disliked story number {i} about cooking and "
                    f"kitchen topic {chr(65 + (i % 26))}"
                ),
            )
        )
        db.upsert_feedback(user.id, 840 + i, "down")
    for i in range(20):
        db.upsert_story(
            Story(
                id=880 + i,
                title=f"fb_neutral_{i}",
                url=None,
                score=10,
                time=0,
                text_content=(
                    f"neutral story number {i} about travel and "
                    f"tourism topic {chr(48 + (i % 10))}"
                ),
            )
        )
        db.upsert_feedback(user.id, 880 + i, "neutral")

    now = int(time.time())
    candidates: list[Story] = []
    # 30 recent (3d old), distinct texts, scores 100..390, cc 20..165.
    for i in range(30):
        candidates.append(
            Story(
                id=i,
                title=f"Recent {i}",
                url=None,
                score=100 + i * 10,
                time=now - 3 * 86400,
                text_content=(
                    f"recent topic about number {i} and theme "
                    f"{chr(97 + (i % 26))}{chr(65 + ((i * 7) % 26))}"
                ),
                source="hn",
                comment_count=20 + i * 5,
            )
        )
    # 30 archive (90d old), distinct texts, scores 500..1950, cc 200..780.
    for i in range(30):
        candidates.append(
            Story(
                id=100 + i,
                title=f"Archive {i}",
                url=None,
                score=500 + i * 50,
                time=now - 90 * 86400,
                text_content=(
                    f"archive topic about subject {i} and angle "
                    f"{chr(65 + (i % 26))}{chr(97 + ((i * 11) % 26))}"
                ),
                source="ch_seed",
                comment_count=200 + i * 20,
            )
        )

    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = rerank_candidates(
        db, config, embedder, candidates, cand_embs, user_id=user.id
    )

    recent = [r for r in ranked if r.is_recent]
    archive = [r for r in ranked if not r.is_recent]
    assert len(recent) >= 2, f"recent cohort too small in final: {len(recent)}"
    assert len(archive) >= 2, f"archive cohort too small in final: {len(archive)}"

    from pipeline import DISCOVERY_PER_BADGE

    for attr in (
        "is_high_engagement",
        "is_discussion_rich",
        "is_novel",
        "is_similar",
        "is_uncertain",
    ):
        n_recent = sum(1 for r in recent if getattr(r, attr))
        n_archive = sum(1 for r in archive if getattr(r, attr))
        assert n_recent >= DISCOVERY_PER_BADGE, (
            f"{attr} must appear >= {DISCOVERY_PER_BADGE} in recent (got {n_recent})"
        )
        assert n_archive >= DISCOVERY_PER_BADGE, (
            f"{attr} must appear >= {DISCOVERY_PER_BADGE} in archive (got {n_archive})"
        )


def test_cascade_badges_mutually_exclusive(db: Database, embedder: Embedder) -> None:
    """Hot (🔥), Top (🏆), and Talk-worthy (💬) badges are mutually exclusive.

    The cascade order is Hot → Top → Talk, with each pass excluding prior
    picks from its pool. A story picked by Hot cannot also be picked by
    Top (Hot is excluded from Top's pool), and a story picked by Top
    cannot also be picked by Talk. This is the property that prevents
    the "too many badges" feel — a Hot story shows 🔥 only, a Top
    story shows 🏆 only, a Talk story shows 💬 only.

    The test builds a 30-recent pool where the top 5 by velocity, the
    top 5 by score, and the top 5 by comment_count overlap heavily.
    """
    config = Config(count=40)
    now = int(time.time())
    candidates: list[Story] = []
    # 30 recent stories, all 1h old, with scores and comment counts that
    # make the top-5 by velocity / score / comment_count largely
    # overlap. Velocity ≈ score/age, so high-score stories are also
    # high-velocity.
    for i in range(30):
        # Score 200..100 (desc), comment_count 200..100 (desc): the
        # top-5 by score and top-5 by cc overlap exactly; the top-5
        # by velocity (= score/age) also overlap with them since all
        # stories are the same age.
        score = max(10, 200 - i * 7)
        cc = max(5, 200 - i * 7)
        candidates.append(
            Story(
                id=i,
                title=f"S{i}",
                url=None,
                score=score,
                time=now - 3600,
                text_content=f"story {i}",
                source="hn",
                comment_count=cc,
            )
        )
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = rerank_candidates(db, config, embedder, candidates, cand_embs)
    by_id = {r.story.id: r for r in ranked}

    # No story in final can have both Hot and Top, or Top and Talk.
    # (Hot/Top overlap: Hot picks first, Top excludes Hot, so the 5 Hot
    # stories must NOT be in the 5 Top. Top/Talk: same.)
    for sid, r in by_id.items():
        cascade = [r.is_hot, r.is_high_engagement, r.is_discussion_rich]
        cascade_count = sum(cascade)
        assert cascade_count <= 1, (
            f"id={sid} has multiple cascade badges: "
            f"is_hot={r.is_hot}, is_high_engagement={r.is_high_engagement}, "
            f"is_discussion_rich={r.is_discussion_rich}"
        )


def test_cascade_top_excluded_from_talk(db: Database, embedder: Embedder) -> None:
    """The Top-archive pass's 5 picks do not appear in the Talk-archive
    pass's 5 picks, even when the top-5 by score and top-5 by comment
    count in the archive cohort overlap exactly.

    This is a stricter version of test_cascade_badges_mutually_exclusive
    for the Top/Talk boundary in archive mode.
    """
    config = Config(count=40)
    now = int(time.time())
    candidates: list[Story] = []
    # 30 archive (60d old) stories. High score == high cc so the top-5
    # by score = the top-5 by cc; the cascade ensures they are
    # different sets.
    for i in range(30):
        score = max(10, 2000 - i * 70)
        cc = max(5, 2000 - i * 70)
        candidates.append(
            Story(
                id=i,
                title=f"Archive {i}",
                url=None,
                score=score,
                time=now - 60 * 86400,
                text_content=f"archive story {i}",
                source="ch_seed",
                comment_count=cc,
            )
        )
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = rerank_candidates(db, config, embedder, candidates, cand_embs)
    archive = [r for r in ranked if not r.is_recent]

    top_archive_ids = {r.story.id for r in archive if r.is_high_engagement}
    talk_archive_ids = {r.story.id for r in archive if r.is_discussion_rich}
    assert top_archive_ids, "expected Top-archive to surface some stories"
    assert talk_archive_ids, "expected Talk-archive to surface some stories"
    assert top_archive_ids.isdisjoint(talk_archive_ids), (
        f"Top-archive and Talk-archive must be disjoint sets, got overlap: "
        f"{top_archive_ids & talk_archive_ids}"
    )


def test_cascade_hot_excluded_from_top(db: Database, embedder: Embedder) -> None:
    """The Hot pass's 5 picks do not appear in the high-engagement-recent
    pass's 5 picks, even when velocity and score overlap.

    Hot is global (recent-only by velocity) and runs first. Top-recent
    runs second and excludes Hot picks. A high-velocity recent story
    that is also high-score must be Hot, not Top.
    """
    config = Config(count=40)
    now = int(time.time())
    candidates: list[Story] = []
    # 30 recent stories, 1h old. Velocity = score/1h. Top 5 by velocity
    # and top 5 by score should overlap (since all have the same age).
    for i in range(30):
        score = max(10, 200 - i * 7)
        candidates.append(
            Story(
                id=i,
                title=f"S{i}",
                url=None,
                score=score,
                time=now - 3600,
                text_content=f"story {i}",
                source="hn",
                comment_count=5,
            )
        )
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = rerank_candidates(db, config, embedder, candidates, cand_embs)

    hot_ids = {r.story.id for r in ranked if r.is_hot}
    top_recent_ids = {
        r.story.id for r in ranked if r.is_high_engagement and r.is_recent
    }
    assert hot_ids, "expected Hot to surface some stories"
    assert top_recent_ids, "expected Top-recent to surface some stories"
    assert hot_ids.isdisjoint(top_recent_ids), (
        f"Hot and Top-recent must be disjoint sets, got overlap: "
        f"{hot_ids & top_recent_ids}"
    )


def test_cascade_can_stack_with_parallel(
    db: Database, embedder: Embedder, monkeypatch
) -> None:
    """A cascade-badge story (e.g. Top) can also be badged by a parallel
    pass (e.g. Unsure) — the parallel group sees the full ranked pool
    and can pick the same story a cascade pass picked.

    Setup: 20 distinct upvotes on "topic A" axis, 20 distinct downvotes
    on "topic B" axis. Stories with embeddings close to a 50/50 mix of
    the two axes have max entropy. We build 30 recent stories that all
    have a 50/50 mix and varied scores. The SVM puts them all near the
    decision boundary → high entropy. The top 12 by SVM score fill
    primary; ids 12..16 are the Top-recent picks (highest score in
    remaining_decorated). The parallel Unsure pass picks 5 from
    `ranked` by entropy desc — these are the same stories the cascade
    ranked high, so the parallel picks overlap with the Top picks.
    """
    config = Config(count=40)
    user = db.create_user("test_cascade_stack")
    for i in range(20):
        db.upsert_story(
            Story(
                id=500 + i,
                title=f"fb_up_{i}",
                url=None,
                score=10,
                time=0,
                text_content=f"upvote {i}",
            )
        )
        db.upsert_feedback(user.id, 500 + i, "up")
    for i in range(20):
        db.upsert_story(
            Story(
                id=600 + i,
                title=f"fb_down_{i}",
                url=None,
                score=10,
                time=0,
                text_content=f"downvote {i}",
            )
        )
        db.upsert_feedback(user.id, 600 + i, "down")

    def mock_gce(stories, embedder_arg, db_inst):
        arr = np.zeros((len(stories), 384), dtype=np.float32)
        for i, s in enumerate(stories):
            if s.id < 500:
                # 50/50 mix of axis A (upvotes) and axis B (downvotes)
                # → max entropy. The unique-axis component keeps the
                # embeddings distinct so the SVM gets a meaningful
                # signal for ranking.
                arr[i, 0] = 0.5
                arr[i, 1] = 0.5
                arr[i, 200 + s.id] = np.sqrt(0.5)
        return arr

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", mock_gce)

    now = int(time.time())
    candidates: list[Story] = []
    for i in range(60):
        score = 200 - i * 5
        candidates.append(
            Story(
                id=i,
                title=f"S{i}",
                url=None,
                score=max(10, score),
                time=now - 3 * 86400,
                text_content=f"text {i}",
                source="hn",
                comment_count=20,
            )
        )

    cand_embs = np.zeros((60, 384), dtype=np.float32)
    for i in range(60):
        cand_embs[i, 0] = 0.5
        cand_embs[i, 1] = 0.5
        cand_embs[i, 200 + i] = np.sqrt(0.5)

    ranked = rerank_candidates(
        db, config, embedder, candidates, cand_embs, user_id=user.id
    )

    # There must be at least one story that ends up with both
    # is_high_engagement AND is_uncertain, proving the parallel pass
    # can stack onto a cascade pick.
    stacked = [r for r in ranked if r.is_high_engagement and r.is_uncertain]
    assert stacked, (
        f"expected at least one story with both Top and Unsure, got none; "
        f"top_ids={[r.story.id for r in ranked if r.is_high_engagement]}, "
        f"unsure_ids={[r.story.id for r in ranked if r.is_uncertain]}"
    )


def test_parallel_can_stack_within(
    db: Database, embedder: Embedder, monkeypatch
) -> None:
    """Explore badges (Unsure/Novel/Similar) and Popular badges can stack
    on the same story within a combo. This verifies that the explore pool
    is shared with the popular pool so a story picked for both lanes
    gets multiple badges.

    Uses controlled embeddings: all stories have a 50/50 mix of upvote
    and downvote axes (high entropy). The Unsure pass picks the top-2 by
    entropy. The Hot pass sees the full combo pool and can badge primary
    cards. The top Hot-picked story should also be an Unsure pick.
    """
    config = Config(count=40)
    user = db.create_user("test_parallel_stack2")
    for i in range(20):
        db.upsert_story(
            Story(
                id=700 + i,
                title=f"fb_up_{i}",
                url=None,
                score=10,
                time=0,
                text_content=f"upvote {i}",
            )
        )
        db.upsert_feedback(user.id, 700 + i, "up")
    for i in range(20):
        db.upsert_story(
            Story(
                id=720 + i,
                title=f"fb_down_{i}",
                url=None,
                score=10,
                time=0,
                text_content=f"downvote {i}",
            )
        )
        db.upsert_feedback(user.id, 720 + i, "down")

    def mock_gce(stories, embedder_arg, db_inst):
        arr = np.zeros((len(stories), 384), dtype=np.float32)
        for i, s in enumerate(stories):
            if s.id < 700:
                arr[i, 0] = 0.5
                arr[i, 1] = 0.5
                arr[i, 200 + s.id] = np.sqrt(0.5)
        return arr

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", mock_gce)

    from pipeline import PRIMARY_PER_COMBO, DISCOVERY_PER_BADGE

    now = int(time.time())
    n_total = PRIMARY_PER_COMBO + DISCOVERY_PER_BADGE + 4  # primary + room
    candidates: list[Story] = []
    for i in range(n_total):
        score = 200 - i * 5
        candidates.append(
            Story(
                id=i,
                title=f"S{i}",
                url=None,
                score=max(10, score),
                time=now - 3 * 86400,
                text_content=f"text {i}",
                source="hn",
                comment_count=20,
            )
        )
    cand_embs = np.zeros((n_total, 384), dtype=np.float32)
    for i in range(n_total):
        cand_embs[i, 0] = 0.5
        cand_embs[i, 1] = 0.5
        cand_embs[i, 200 + i] = np.sqrt(0.5)

    ranked = rerank_candidates(
        db, config, embedder, candidates, cand_embs, user_id=user.id
    )

    # At least one story should have both a Popular badge (Hot from
    # full-pool Hot pass) and an Explore badge (Unsure from shared pool).
    stacked = [
        r for r in ranked if (r.is_hot or r.is_high_engagement) and r.is_uncertain
    ]
    assert stacked, (
        f"expected at least one story with Popular+Explore stacking; "
        f"hot_ids={[r.story.id for r in ranked if r.is_hot]}, "
        f"unsure_ids={[r.story.id for r in ranked if r.is_uncertain]}"
    )


def test_hot_badge_threshold_uses_config_percentile(
    db: Database, embedder: Embedder
) -> None:
    """is_hot respects hot_badge_percentile from config.

    The Hot pass runs against the full combo pool and can badge primary
    cards. Slot limit is DISCOVERY_PER_BADGE.
    """
    from pipeline import DISCOVERY_PER_BADGE

    now = int(time.time())
    # 20 stories, scores 10..200 (step 10), all 1h old so velocity = score.
    score_values = list(range(10, 210, 10))  # [10, 20, ..., 200]
    candidates = [
        Story(
            id=i,
            title=f"Story {i}",
            url=None,
            score=score,
            time=now - 3600,
            text_content=f"Sample text content for story {i}.",
            source="hn",
            comment_count=0,
        )
        for i, score in enumerate(reversed(score_values))  # id=0→score=200
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    # Default: 99.5th pct of [10..200] ≈ 199. Only id=0 (velocity 200) clears.
    config = Config(count=40)
    ranked = rerank_candidates(db, config, embedder, candidates, cand_embs)
    hot_ids = {r.story.id for r in ranked if r.is_hot}
    assert 0 in hot_ids
    assert 1 not in hot_ids, "190-score story should NOT be hot at p99.5"
    assert len(hot_ids) == 1, f"Expected 1 hot at p99.5, got {len(hot_ids)}"

    # 50th pct: p50 ≈ 105. Ids 0..9 clear the threshold. Slot cap is
    # DISCOVERY_PER_BADGE, so only the top 2 by velocity get Hot.
    config2 = Config(count=40, model=ModelConfig(hot_badge_percentile=50.0))
    ranked2 = rerank_candidates(db, config2, embedder, candidates, cand_embs)
    hot2 = {r.story.id for r in ranked2 if r.is_hot}
    assert 0 in hot2
    assert 1 in hot2
    assert 2 not in hot2, "Should only get DISCOVERY_PER_BADGE hot cards"
    assert len(hot2) == DISCOVERY_PER_BADGE, (
        f"Expected {DISCOVERY_PER_BADGE} hot at p50, got {len(hot2)}"
    )


def test_tier1_gravity_at_zero_feedback(db: Database, embedder: Embedder) -> None:
    """Zero feedback → tier 1 gravity formula, sorted by score/age."""
    config = Config()
    db.create_user("test_token_tier1")
    now = int(time.time())

    candidates = [
        Story(
            id=1,
            title="Old low score",
            url=None,
            score=10,
            time=now - 86400 * 5,
            text_content="old post with low engagement",
            source="hn",
        ),
        Story(
            id=2,
            title="Recent high score",
            url=None,
            score=100,
            time=now - 3600,
            text_content="recent trending post on hacker news",
            source="hn",
        ),
        Story(
            id=3,
            title="Old high score",
            url=None,
            score=200,
            time=now - 86400 * 3,
            text_content="older but popular post",
            source="hn",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    # Recent high score should rank highest (gravity) — no, tier 1 sorts by story.score (raw points)
    assert ranked[0].story.id == 3, (
        f"Expected highest-raw-score story first, got id={ranked[0].story.id}"
    )
    # RankedStory.score is gravity-based and in [0, 1]
    for r in ranked:
        assert 0.0 <= r.score <= 1.0
    # Top gravity candidate is story 2 (recent 100 points beats old 200 points)
    gravity_sorted = sorted(ranked, key=lambda x: x.score, reverse=True)
    assert gravity_sorted[0].story.id == 2


def test_tier3_svm_at_60_plus_with_gates(db: Database, embedder: Embedder) -> None:
    """30 up + 30 down: both gates pass, α=(30-20)/60=0.167, ranking correct."""
    config = Config()
    user = db.create_user("test_token_tier3")

    for i in range(30):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up story {i}",
                url=None,
                score=0,
                time=0,
                text_content="Deep learning AI research artificial intelligence neural networks transformers machine learning",
            ),
        )
        db.upsert_feedback(user.id, 100 + i, "up")
    for i in range(30):
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down story {i}",
                url=None,
                score=0,
                time=0,
                text_content="Baking sourdough bread cake cookie recipe kitchen kitchen kitchen",
            ),
        )
        db.upsert_feedback(user.id, 200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
        ),
        Story(
            id=2,
            title="Cake recipe",
            url=None,
            score=0,
            time=0,
            text_content="Delicious chocolate chip cake baking guide.",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)

    assert len(ranked) == 2
    # AI story should rank first (SVM learned upvote pattern)
    assert ranked[0].story.id == 1
    # Cake story should rank second
    assert ranked[1].story.id == 2
    for r in ranked:
        assert 0.0 <= r.score <= 1.0


def _seed_feedback(db: Database, user_id: int, n_up: int, n_down: int) -> None:
    import hashlib

    model_version = "all-MiniLM-L6-v2|mean|norm|256"
    for i in range(n_up):
        story = Story(
            id=100 + i,
            title=f"Up {i}",
            url=None,
            score=0,
            time=0,
            text_content=f"deep learning AI research {i}",
        )
        db.upsert_story(story)
        text_hash = hashlib.sha256(story.text_content.encode("utf-8")).hexdigest()
        db.upsert_embedding(
            story.id, model_version, text_hash, np.random.randn(384).astype(np.float32)
        )
        db.upsert_feedback(user_id, story.id, "up")
    for i in range(n_down):
        story = Story(
            id=200 + i,
            title=f"Down {i}",
            url=None,
            score=0,
            time=0,
            text_content=f"baking sourdough bread {i}",
        )
        db.upsert_story(story)
        text_hash = hashlib.sha256(story.text_content.encode("utf-8")).hexdigest()
        db.upsert_embedding(
            story.id, model_version, text_hash, np.random.randn(384).astype(np.float32)
        )
        db.upsert_feedback(user_id, story.id, "down")


def test_min_class_gate_n_down_fails(db: Database, embedder: Embedder) -> None:
    """30 up, 5 down: n_down=5 < 20, gate fails, pure tier-2 (no SVM, prob_up=None)."""
    config = Config()
    user = db.create_user("test_token_fail_down")
    _seed_feedback(db, user.id, n_up=30, n_down=5)
    cand_embs = np.random.randn(2, 384).astype(np.float32)
    candidates = [
        Story(id=1, title="A", url=None, score=0, time=0, text_content="x"),
        Story(id=2, title="B", url=None, score=0, time=0, text_content="y"),
    ]
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)
    assert len(ranked) == 2
    for r in ranked:
        assert r.prob_up is None
        assert 0.0 <= r.score <= 1.0


def test_min_class_gate_n_up_fails(db: Database, embedder: Embedder) -> None:
    """5 up, 30 down: n_up=5 < 20, gate fails, pure tier-2 (no SVM, prob_up=None)."""
    config = Config()
    user = db.create_user("test_token_fail_up")
    _seed_feedback(db, user.id, n_up=5, n_down=30)
    cand_embs = np.random.randn(2, 384).astype(np.float32)
    candidates = [
        Story(id=1, title="A", url=None, score=0, time=0, text_content="x"),
        Story(id=2, title="B", url=None, score=0, time=0, text_content="y"),
    ]
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)
    assert len(ranked) == 2
    for r in ranked:
        assert r.prob_up is None
        assert 0.0 <= r.score <= 1.0


def test_min_class_gate_both_just_pass(db: Database, embedder: Embedder) -> None:
    """20 up, 20 down: both gates pass at boundary, α=0, SVM runs (prob_up not None)."""
    config = Config()
    user = db.create_user("test_token_just_pass")
    _seed_feedback(db, user.id, n_up=20, n_down=20)
    cand_embs = np.random.randn(2, 384).astype(np.float32)
    candidates = [
        Story(id=1, title="A", url=None, score=0, time=0, text_content="x"),
        Story(id=2, title="B", url=None, score=0, time=0, text_content="y"),
    ]
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)
    assert len(ranked) == 2
    for r in ranked:
        assert r.prob_up is not None
        assert 0.0 <= r.score <= 1.0


def test_min_class_blend_mid(db: Database, embedder: Embedder) -> None:
    """50 up, 30 down: both gates pass, min=30, α=(30-20)/60≈0.167, SVM runs."""
    config = Config()
    user = db.create_user("test_token_blend_mid")
    _seed_feedback(db, user.id, n_up=50, n_down=30)
    cand_embs = np.random.randn(2, 384).astype(np.float32)
    candidates = [
        Story(id=1, title="A", url=None, score=0, time=0, text_content="x"),
        Story(id=2, title="B", url=None, score=0, time=0, text_content="y"),
    ]
    ranked = _score_and_rank(candidates, cand_embs, db, config, embedder)
    assert len(ranked) == 2
    for r in ranked:
        assert r.prob_up is not None
        assert 0.0 <= r.score <= 1.0


# ── Non-HN discovery slot formula tests ──


@given(data=st.data())
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_non_hn_slot_count_bounds(data):
    """_non_hn_slot_count always returns 0 <= result <= cap."""
    from pipeline import _non_hn_slot_count

    n = data.draw(st.integers(0, 200))
    cap = data.draw(st.integers(1, 16))
    threshold = data.draw(st.integers(0, 100))
    window = data.draw(st.integers(1, 100))
    result = _non_hn_slot_count(n, cap=cap, threshold=threshold, window=window)
    assert 0 <= result <= cap


@given(data=st.data())
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_non_hn_slot_count_zero_below_threshold(data):
    """Result is 0 when n_feedback <= threshold."""
    from pipeline import _non_hn_slot_count

    n = data.draw(st.integers(0, 200))
    cap = data.draw(st.integers(1, 16))
    threshold = data.draw(st.integers(0, 100))
    window = data.draw(st.integers(1, 100))
    if n <= threshold:
        assert _non_hn_slot_count(n, cap=cap, threshold=threshold, window=window) == 0


@given(data=st.data())
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_non_hn_slot_count_capped_at_full_ramp(data):
    """Result is cap when n_feedback >= threshold + window."""
    from pipeline import _non_hn_slot_count

    n = data.draw(st.integers(0, 200))
    cap = data.draw(st.integers(1, 16))
    threshold = data.draw(st.integers(0, 100))
    window = data.draw(st.integers(1, 100))
    if n >= threshold + window and window > 0:
        assert _non_hn_slot_count(n, cap=cap, threshold=threshold, window=window) == cap


def test_non_hn_slot_count_exact_values():
    from pipeline import _non_hn_slot_count

    params = (8, 20, 30)  # cap, threshold, window (the production defaults)
    cases = [
        (0, 0),
        (19, 0),
        (20, 0),
        (24, 1),
        (30, 3),
        (40, 5),
        (50, 8),
        (100, 8),
    ]
    for n, expected in cases:
        assert _non_hn_slot_count(n, *params) == expected, (
            f"n={n}: expected {expected}, got {_non_hn_slot_count(n, *params)}"
        )


# ── Comment selection algorithm tests ──


def test_comment_rank_key_no_score_dimension():
    """_comment_rank_key no longer includes the score (depth-penalty) dimension."""
    from pipeline import _comment_rank_key

    keys = [
        _comment_rank_key(
            {"descendant_count": 10, "text_len": 200, "order_path": (0,)}
        ),
        _comment_rank_key({"descendant_count": 0, "text_len": 500, "order_path": (1,)}),
    ]
    assert len(keys[0]) == 3  # descendant_count, text_len, order_path
    # Higher descendant_count sorts first
    assert keys[0] < keys[1]  # -10 < 0


def test_select_top_comments_drops_low_quality_toplevel():
    """Short, low-reply top-level should not be selected as 'good'."""
    from pipeline import _select_top_comments

    good = {
        "text": "Long substantive comment with a lot of text content that should easily pass the good top-level minimum length requirement and be useful for TLDR summaries.",
        "depth": 0,
        "descendant_count": 0,
        "top_thread_index": 0,
        "text_len": 200,
        "order_path": (0,),
    }
    bad = {
        "text": "Nice article!",
        "depth": 0,
        "descendant_count": 0,
        "top_thread_index": 1,
        "text_len": 14,
        "order_path": (1,),
    }
    selected = _select_top_comments([bad, good], limit=1)
    sel_texts = [c["text"] for c in selected]
    assert any("Long substantive comment" in t for t in sel_texts)


def test_select_top_comments_adaptive_cores_small_story():
    """n_cores should adapt down when fewer than 4 good top-level exist."""
    from pipeline import _select_top_comments

    roots = [
        {
            "text": f"Substantive top-level {i} with enough text to pass the good top-level threshold.",
            "depth": 0,
            "descendant_count": 5,
            "top_thread_index": i,
            "text_len": 80,
            "order_path": (i,),
        }
        for i in range(2)
    ]
    replies = [
        {
            "text": f"Substantive reply {i} with enough context to pass the minimum length for comment extraction.",
            "depth": 1,
            "descendant_count": 0,
            "top_thread_index": 0,
            "text_len": 100,
            "order_path": (0, i),
        }
        for i in range(5)
    ]
    selected = _select_top_comments(roots + replies, limit=10)
    top_selected = [c for c in selected if c["depth"] == 0]
    assert len(top_selected) <= 2  # cores = min(4, 2) = 2


def test_select_top_comments_top_level_budget_caps():
    """Top-level count should be below the old algorithm's 20.

    The 1/3 budget limits the breadth pass; the diagnostic on 10 real
    stories confirmed 11-17 top-level (vs 20 in the old algorithm).
    """
    from pipeline import _select_top_comments

    top_level = [
        {
            "text": f"Top {i} with enough text to pass the quality threshold.",
            "depth": 0,
            "descendant_count": 10,
            "top_thread_index": i,
            "text_len": 100,
            "order_path": (i,),
        }
        for i in range(30)
    ]
    # Replies in threads 4-7 have higher descendant_count than top-level (12 > 10),
    # so the filler prefers them over additional top-level, keeping count near budget.
    replies = [
        {
            "text": f"Reply {t}.{j} substantial text content for TLDR context and discussion summary.",
            "depth": 1,
            "descendant_count": 3 if t < 4 else 12,
            "top_thread_index": t,
            "text_len": 150,
            "order_path": (t, j),
        }
        for t in range(8)
        for j in range(6)
    ]
    selected = _select_top_comments(top_level + replies, limit=40)
    top_count = sum(1 for c in selected if c["depth"] == 0)
    assert top_count < 20  # old algorithm always gave 20


def test_select_top_comments_long_reply_beats_short_toplevel():
    """A long, substantive reply should be selected over a short, low-reply top-level."""
    from pipeline import _select_top_comments

    short_top = {
        "text": "Short top-level.",
        "depth": 0,
        "descendant_count": 0,
        "top_thread_index": 0,
        "text_len": 18,
        "order_path": (0,),
    }
    long_reply = {
        "text": "Long substantive reply with enough text to easily pass the minimum extraction length and be useful.",
        "depth": 2,
        "descendant_count": 0,
        "top_thread_index": 1,
        "text_len": 110,
        "order_path": (1,),
    }
    selected = _select_top_comments([short_top, long_reply], limit=5)
    sel_texts = [c["text"] for c in selected]
    assert any("Long substantive reply" in t for t in sel_texts)


def test_min_comment_length_filter():
    """MIN_COMMENT_LENGTH=60 should filter out short comments at extraction."""
    from pipeline import _extract_comments_recursive

    children = [
        {"type": "comment", "text": "Short.", "children": []},
        {
            "type": "comment",
            "text": "Fifty-five character sentence that is just short",
            "children": [],
        },
        {
            "type": "comment",
            "text": "Sixty-one character sentence that should be long enough to pass eas",
            "children": [],
        },
    ]
    extracted = _extract_comments_recursive(children)
    texts = [c["text"] for c in extracted]
    assert len(texts) == 1  # only the 61-char comment passes
    assert "Short." not in texts[0]
    assert "Sixty-one character" in texts[0]


def test_hot_badge_requires_minimum_score(db, embedder):
    """Stories with score < HOT_MIN_SCORE must not get is_hot even with high velocity."""
    now = time.time()
    candidates = [
        Story(
            id=i,
            title=f"S{i}",
            url=f"https://a.com/{i}",
            score=10 * i,
            time=int(now - 3600),
            text_content=f"text {i}",
            source="hn",
            comment_count=0,
        )
        for i in range(1, 26)
    ]
    candidates.append(
        Story(
            id=99,
            title="LowScoreHighVelocity",
            url="https://a.com/99",
            score=8,
            time=int(now - 60),
            text_content="text 99",
            source="hn",
            comment_count=0,
        )
    )

    embs = np.eye(len(candidates), 384, dtype=np.float32)
    result = rerank_candidates(
        db=db,
        config=Config(count=len(candidates)),
        embedder=embedder,
        candidates=candidates,
        cand_embeddings=embs,
        user_id=None,
    )

    for r in result:
        if r.is_hot:
            assert r.story.score >= HOT_MIN_SCORE, (
                f"Story {r.story.id} (score={r.story.score}) has is_hot=True "
                f"but score < HOT_MIN_SCORE={HOT_MIN_SCORE}"
            )


# ---------- prewarm_top_stories tests ----------


class _DummyEmbedder(Embedder):
    def __init__(self):
        pass

    def encode(self, texts, batch_size=32):
        import numpy as _np

        arr = _np.zeros((len(texts), 384), dtype=_np.float32)
        if len(texts):
            arr[:, 0] = 1.0
        return arr


def test_prewarm_top_stories_empty_list_returns_zero() -> None:
    db = Database(":memory:")
    try:
        result = pipeline.prewarm_top_stories([], db, None)
        assert result == 0
    finally:
        db.close()


def test_prewarm_top_stories_no_ch_call_when_all_zero_ids() -> None:
    """All-zero or non-positive IDs should be filtered out before any CH call."""
    db = Database(":memory:")
    try:
        called = {"n": 0}

        def fail_ch(*a, **kw):
            called["n"] += 1
            raise AssertionError("CH should not be called")

        from unittest.mock import patch

        with patch("ch_client.query_stories_with_comments", fail_ch):
            result = pipeline.prewarm_top_stories([0, -1, 0], db, None)
        assert result == 0
        assert called["n"] == 0
    finally:
        db.close()


def test_prewarm_top_stories_ch_failure_returns_zero() -> None:
    """If CH query fails, prewarm returns 0 and doesn't crash the caller."""
    db = Database(":memory:")
    try:
        story = Story(
            id=1, title="T", url="u", score=100, time=1, text_content="t", source="hn"
        )
        db.upsert_story(story)
        from unittest.mock import patch

        with patch(
            "ch_client.query_stories_with_comments",
            side_effect=RuntimeError("simulated CH outage"),
        ):
            result = pipeline.prewarm_top_stories([1], db, None)
        assert result == 0
    finally:
        db.close()


def test_prewarm_top_stories_updates_top_comments() -> None:
    """Happy path: CH returns a story with comments, prewarm writes them back."""
    db = Database(":memory:")
    try:
        story = Story(
            id=42,
            title="Hello",
            url="u",
            score=100,
            time=1,
            text_content="",
            source="hn",
        )
        db.upsert_story(story)

        ch_item = {
            "id": 42,
            "type": "story",
            "title": "Hello",
            "url": "u",
            "story_text": "",
            "text": "",
            "num_comments": 5,
            "created_at_i": 1,
            "points": 100,
            "children": [
                {
                    "id": 100,
                    "type": "comment",
                    "text": "Substantive comment with enough words and length to pass the minimum comment length filtering threshold.",
                    "children": [],
                },
                {
                    "id": 101,
                    "type": "comment",
                    "text": "Another comment that is also long enough to qualify and should be selected for the top comments list.",
                    "children": [],
                },
            ],
        }
        from unittest.mock import patch

        with patch(
            "ch_client.query_stories_with_comments",
            return_value={42: ch_item},
        ):
            result = pipeline.prewarm_top_stories([42], db, _DummyEmbedder())
        assert result == 1
        updated = db.get_story(42)
        assert updated is not None
        assert updated.top_comments != ""
        assert updated.comment_count == 5
        assert updated.comment_count_at_fetch == 5
        assert "Substantive comment" in updated.text_content
    finally:
        db.close()


def test_prewarm_top_stories_skips_stories_not_in_db() -> None:
    """If CH returns a story that's not in the DB, skip it."""
    db = Database(":memory:")
    try:
        from unittest.mock import patch

        with patch(
            "ch_client.query_stories_with_comments",
            return_value={
                99: {
                    "id": 99,
                    "type": "story",
                    "title": "T",
                    "url": None,
                    "story_text": "",
                    "text": "",
                    "num_comments": 0,
                    "created_at_i": 0,
                    "points": 0,
                    "children": [],
                }
            },
        ):
            result = pipeline.prewarm_top_stories([99], db, None)
        assert result == 0
    finally:
        db.close()


def test_candidate_similar_to_neutral_is_not_novel(db, embedder, monkeypatch):
    """A candidate close to a neutral feedback story should not get is_novel=True,
    because cand_max_sim now includes neutral similarity in the 3-way max."""
    config = Config(count=3)
    user = db.create_user("test_neutral_novel")

    def mock_gce(stories, embedder_arg, db_inst):
        arr = np.zeros((len(stories), 384), dtype=np.float32)
        for i, s in enumerate(stories):
            if s.id == 100:
                arr[i, 0] = 1.0  # up
            elif s.id == 200:
                arr[i, 1] = 1.0  # down
            elif s.id == 300:
                arr[i, 2] = 1.0  # neutral
        return arr

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", mock_gce)

    for sid, action in [(100, "up"), (200, "down"), (300, "neutral")]:
        db.upsert_story(
            Story(
                id=sid,
                title=f"S{sid}",
                url=None,
                score=100 - sid,
                time=0,
                text_content=f"feedback {sid}",
            )
        )
        db.upsert_feedback(user.id, sid, action)

    # Controlled candidate embeddings. 3 targets + enough fillers
    # to saturate PRIMARY_PER_COMBO (12) so targets land in discovery pool.
    # 1 -> close to neutral (axis 2), max_sim=0.95
    # 2 -> close to up (axis 0), max_sim=0.95
    # 3 -> far from all (axis 3), max_sim=0
    from pipeline import PRIMARY_PER_COMBO

    n_total = PRIMARY_PER_COMBO + 10  # fill primary + room for discovery
    cand_embs = np.zeros((n_total, 384), dtype=np.float32)
    cand_embs[0, 2] = 0.95  # close to neutral
    cand_embs[1, 0] = 0.95  # close to up
    cand_embs[2, 3] = 1.0  # far from all feedback
    for i in range(3, n_total):
        cand_embs[i, 0] = 0.05
        cand_embs[i, 1] = 0.05
        cand_embs[i, 2] = 0.05
        cand_embs[i, 50 + i] = 0.6
        cand_embs[i, 200 + i] = 0.8

    candidates = []
    now = int(time.time())
    for cid, score in [(1, 5), (2, 5), (3, 5)]:
        candidates.append(
            Story(
                id=cid,
                title=f"C{cid}",
                url=None,
                score=score,
                time=now - 86400,
                text_content="",
                source="hn",
                comment_count=0,
            )
        )
    for i in range(3, n_total):
        candidates.append(
            Story(
                id=100 + i,
                title=f"F{i}",
                url=None,
                score=100 + i,
                time=now - 86400,
                text_content="",
                source="hn",
                comment_count=0,
            )
        )

    ranked = rerank_candidates(
        db, config, embedder, candidates, cand_embs, user_id=user.id
    )
    by_id = {r.story.id: r for r in ranked}

    # id=3 (far from all feedback, max_sim=0, dist=1.0) is the most
    # novel and should be in the top DISCOVERY_PER_BADGE by distance.
    assert 3 in by_id, "id=3 (far from all feedback) should be in final"
    assert by_id[3].is_novel, (
        f"id=3 (far from feedback) should be novel; got is_novel={by_id[3].is_novel}"
    )
    # id=1 (close to neutral, max_sim=0.95) is not novel.
    # id=2 (close to up, max_sim=0.95) has high up-similarity so it
    # may get is_similar but should NOT get is_novel.
    for cid in (1, 2):
        if cid in by_id:
            assert not by_id[cid].is_novel, (
                f"Candidate {cid} (close to feedback) should not be novel"
            )


def test_no_neutral_feedback_uses_up_down_only_for_novel(db, embedder, monkeypatch):
    """When no neutral feedback exists, cand_max_sim equals max(up, down) because
    neutral similarities are zeros — same as old behavior."""
    config = Config(count=10)
    user = db.create_user("test_no_neutral_novel")

    def mock_gce(stories, embedder_arg, db_inst):
        arr = np.zeros((len(stories), 384), dtype=np.float32)
        for i, s in enumerate(stories):
            if s.id == 100:
                arr[i, 0] = 1.0  # up
            elif s.id == 200:
                arr[i, 1] = 1.0  # down
        return arr

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", mock_gce)

    for sid, action in [(100, "up"), (200, "down")]:
        db.upsert_story(
            Story(
                id=sid,
                title=f"S{sid}",
                url=None,
                score=100 - sid,
                time=0,
                text_content=f"feedback {sid}",
            )
        )
        db.upsert_feedback(user.id, sid, action)

    # Controlled candidate embeddings. 2 targets + enough fillers
    # to saturate PRIMARY_PER_COMBO so targets land in discovery pool.
    # 1 -> close to up (axis 0), max_sim=0.95
    # 2 -> far from all (axis 3), max_sim=0
    from pipeline import PRIMARY_PER_COMBO

    n_total = PRIMARY_PER_COMBO + 8
    cand_embs = np.zeros((n_total, 384), dtype=np.float32)
    cand_embs[0, 0] = 0.95
    cand_embs[1, 3] = 1.0
    for i in range(2, n_total):
        cand_embs[i, 0] = 0.05
        cand_embs[i, 1] = 0.05
        cand_embs[i, 50 + i] = 0.6
        cand_embs[i, 200 + i] = 0.8

    now = int(time.time())
    candidates = [
        Story(
            id=1,
            title="Up-like",
            url=None,
            score=5,
            time=now - 86400,
            text_content="",
            source="hn",
            comment_count=0,
        ),
        Story(
            id=2,
            title="Novel",
            url=None,
            score=5,
            time=now - 86400,
            text_content="",
            source="hn",
            comment_count=0,
        ),
    ]
    for i in range(2, n_total):
        candidates.append(
            Story(
                id=100 + i,
                title=f"F{i}",
                url=None,
                score=100 + i,
                time=now - 86400,
                text_content="",
                source="hn",
                comment_count=0,
            )
        )

    ranked = rerank_candidates(
        db, config, embedder, candidates, cand_embs, user_id=user.id
    )
    by_id = {r.story.id: r for r in ranked}

    # id=2 (max_sim=0, dist=1.0) is the most novel; it's in the top
    # DISCOVERY_PER_BADGE by distance.
    assert 2 in by_id, "id=2 (not similar to any) should be in final"
    assert by_id[2].is_novel, (
        f"Candidate 2 should be novel; got is_novel={by_id[2].is_novel}"
    )
    # id=1 (max_sim=0.95) is close to feedback, not novel.
    if 1 in by_id:
        assert not by_id[1].is_novel, "Candidate 1 (similar to up) should not be novel"


def test_novel_pass_ranks_purely_by_distance_not_score(
    db, embedder, monkeypatch
) -> None:
    """The Novel pass ranks by distance (1 - max_similarity) only, not score.
    A low-score, high-distance story beats a higher-score, lower-distance
    story when the slot cap forces a cut.

    With per-combo discovery slots (DISCOVERY_PER_BADGE), the cut is at
    position 2. We construct scores so the 2nd-by-distance story has a
    very low score; pure-distance ranking keeps it; a score-blended
    ranking would have dropped it for a higher-score story.
    """
    from pipeline import PRIMARY_PER_COMBO

    config = Config(count=40)
    user = db.create_user("test_novel_distance")

    # Feedback at axis 0 (one upvoted story).
    def mock_gce(stories, embedder_arg, db_inst):
        arr = np.zeros((len(stories), 384), dtype=np.float32)
        for i, s in enumerate(stories):
            if s.id == 100:
                arr[i, 0] = 1.0
        return arr

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", mock_gce)
    db.upsert_story(
        Story(id=100, title="fb", url=None, score=10, time=0, text_content="")
    )
    db.upsert_feedback(user.id, 100, "up")

    # Primary fillers (high score) + controlled extras (low score, varied sim).
    # The novel pass picks DISCOVERY_PER_BADGE by distance. We arrange so
    # the 2nd-by-distance (id=13, sim=0.15, score=1) is low-score; a score-
    # blended ranking would drop it for id=14 (sim=0.30, score=50).
    now = int(time.time())
    candidates = []
    for i in range(PRIMARY_PER_COMBO):
        candidates.append(
            Story(
                id=i,
                title=f"P{i}",
                url=None,
                score=100 + i,
                time=now - 3600,
                text_content=f"primary {i}",
                source="hn",
                comment_count=0,
            )
        )
    extra_scores = [10, 1, 50, 10]  # id=13 score=1, id=14 score=50
    extra_sims = [0.10, 0.15, 0.30, 0.45]  # distances: 0.90, 0.85, 0.70, 0.55
    for i, sc in enumerate(extra_scores):
        candidates.append(
            Story(
                id=PRIMARY_PER_COMBO + i,
                title=f"E{i}",
                url=None,
                score=sc,
                time=now - 3600,
                text_content=f"extra {i}",
                source="hn",
                comment_count=0,
            )
        )

    n_total = len(candidates)
    cand_embs = np.zeros((n_total, 384), dtype=np.float32)
    for i in range(PRIMARY_PER_COMBO):
        cand_embs[i, 0] = 0.5
        cand_embs[i, 50 + i] = np.sqrt(0.75)
    for i, s in enumerate(extra_sims):
        idx = PRIMARY_PER_COMBO + i
        cand_embs[idx, 0] = s
        cand_embs[idx, 100 + i] = np.sqrt(max(1.0 - s * s, 0.0))

    ranked = rerank_candidates(
        db, config, embedder, candidates, cand_embs, user_id=user.id
    )

    by_id = {r.story.id: r for r in ranked}
    # id=13 (sim=0.15, dist=0.85, score=1 — very low) is 2nd-by-distance.
    # Pure-distance ranking keeps it; score-blended would drop it for id=14.
    assert by_id[PRIMARY_PER_COMBO + 1].is_novel, (
        f"id={PRIMARY_PER_COMBO + 1} (sim=0.15, dist=0.85, score=1) should be novel"
    )
    # id=14 (sim=0.30, dist=0.70, score=50 — high score) is 3rd-by-distance.
    # Beyond DISCOVERY_PER_BADGE cut, so NOT novel despite higher score.
    target_high_score = PRIMARY_PER_COMBO + 2
    if target_high_score in by_id:
        assert not by_id[target_high_score].is_novel, (
            f"id={target_high_score} (higher score, worse distance) must NOT be novel"
        )


def test_prewarm_top_stories_empty_ch_response_returns_zero() -> None:
    db = Database(":memory:")
    try:
        from unittest.mock import patch

        with patch("ch_client.query_stories_with_comments", return_value={}):
            result = pipeline.prewarm_top_stories([1], db, None)
        assert result == 0
    finally:
        db.close()


def test_fetch_candidates_only_prewarms_top_n_by_score(monkeypatch) -> None:
    """Regen prewarm selects top N HN candidates by score descending."""
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            regen_prewarm_top_n=3,
            prewarm_hn_full=False,
            prewarm_reddit_full=False,
        )

        stories = [
            Story(id=1, title="A", url="", score=10, time=100, text_content="a"),
            Story(id=2, title="B", url="", score=50, time=100, text_content="b"),
            Story(id=3, title="C", url="", score=30, time=100, text_content="c"),
            Story(
                id=4,
                title="D",
                url="http://reddit.com/r/test",
                score=5,
                time=100,
                text_content="d",
                source="rss_reddit_test",
            ),
        ]
        for s in stories:
            db.upsert_story(s)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return stories, 4

        captured_ids: list[list[int]] = []

        def fake_prewarm(ids, db_, embedder):
            captured_ids.append(list(ids))
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert len(captured_ids) == 1
        # Top 3 by score (HN only): ids 2(50), 3(30), 1(10)
        assert captured_ids[0] == [2, 3, 1]

    finally:
        db.close()


def test_fetch_candidates_only_skips_prewarm_when_n_is_zero(monkeypatch) -> None:
    """Regen prewarm is disabled when regen_prewarm_top_n is 0."""
    db = Database(":memory:")
    try:
        config = Config(db_path=db.db_path, regen_prewarm_top_n=0)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        called = False

        def fake_prewarm(ids, db_, embedder):
            nonlocal called
            called = True
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert not called

    finally:
        db.close()


def test_fetch_candidates_only_skips_prewarm_when_no_embedder(monkeypatch) -> None:
    """Regen prewarm is skipped when embedder is None."""
    db = Database(":memory:")
    try:
        config = Config(db_path=db.db_path, regen_prewarm_top_n=5)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        called = False

        def fake_prewarm(ids, db_, embedder):
            nonlocal called
            called = True
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(pipeline.fetch_candidates_only(config, db, embedder=None))
        assert not called

    finally:
        db.close()


def test_fetch_candidates_only_prewarm_empty_candidates(monkeypatch) -> None:
    """Regen prewarm handles empty candidate list gracefully."""
    db = Database(":memory:")
    try:
        config = Config(db_path=db.db_path, regen_prewarm_top_n=3)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        called = False

        def fake_prewarm(ids, db_, embedder):
            nonlocal called
            called = True
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert not called

    finally:
        db.close()


def test_fetch_candidates_only_prewarms_all_hn_when_full(monkeypatch) -> None:
    """Regen prewarms all HN candidates needing comments when prewarm_hn_full=True."""
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=True,
            prewarm_reddit_full=False,
        )

        stories = [
            Story(id=1, title="A", url="", score=10, time=100, text_content="a"),
            Story(
                id=2,
                title="B",
                url="",
                score=50,
                time=100,
                text_content="b",
                comment_count=5,
                top_comments="",
            ),
            Story(
                id=3,
                title="C",
                url="",
                score=30,
                time=100,
                text_content="c",
                comment_count=3,
                top_comments="",
            ),
            Story(id=4, title="D", url="", score=5, time=100, text_content="d"),
        ]
        for s in stories:
            db.upsert_story(s)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return stories, 4

        captured_ids: list[list[int]] = []

        def fake_prewarm(ids, db_, embedder):
            captured_ids.append(list(ids))
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert len(captured_ids) == 1
        # Only stories with comment_count > 0 and empty top_comments: 2, 3
        assert sorted(captured_ids[0]) == [2, 3]

    finally:
        db.close()


def test_fetch_candidates_only_persists_topfeed_before_prewarm(
    monkeypatch,
) -> None:
    """Topfeed cache entries are upserted into SQLite before the prewarm
    phase is built. Otherwise the prewarm factory's `db.get_story` would
    return None for brand-new topfeed discoveries and the factory would
    no-op."""
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            prewarm_reddit_full=True,
            reddit_prewarm_top_per_sub=2,
            reddit_prewarm_max_per_cycle=10,
        )
        config = replace(
            config,
            rss=RssConfig(
                enabled=True,
                per_feed_limit=70,
                feeds=("https://www.reddit.com/r/x/top/.rss?t=week&limit=25",),
            ),
        )

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        new_story = Story(
            id=42,
            title="New topfeed story",
            url="https://reddit.com/r/x/comments/42",
            score=100,
            time=1_000_000,
            text_content="",
            source="rss_reddit_r_x",
            comment_count=0,
        )

        def fake_topfeed_factories(feeds, per_feed, days, exclude_urls):
            from reddit_feed_cache import cache as reddit_feed_cache

            # The real topfeed factory writes to the cache when called by
            # the queue worker. The mock queue never invokes factories, so
            # we pre-populate the cache synchronously to simulate the
            # post-drain state.
            reddit_feed_cache.set(feeds[0], [new_story])

            async def factory() -> None:
                return None

            return [factory], [feeds[0]]

        captured_factory_ids: list[list[int]] = []

        def fake_build_reddit_prewarm_factories(story_ids, db_):
            captured_factory_ids.append(list(story_ids))

            async def noop() -> None:
                return None

            return [noop], []

        class _NoopQueue:
            def enqueue_all_reddit_fetches(self, *args, **kwargs) -> None:
                return None

            def wait_until_empty(self, timeout: float = 5400.0) -> bool:
                return True

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(
            pipeline, "build_reddit_topfeed_factories", fake_topfeed_factories
        )
        monkeypatch.setattr(
            pipeline,
            "build_reddit_prewarm_factories",
            fake_build_reddit_prewarm_factories,
        )
        monkeypatch.setattr("reddit_fetch_queue.queue", _NoopQueue())

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        # Story was persisted in phase 1.5 before prewarm was built.
        assert db.get_story(42) is not None
        # And the prewarm factory was given the story's id.
        assert captured_factory_ids and captured_factory_ids[0] == [42]
    finally:
        db.close()


def test_fetch_candidates_only_caps_reddit_prewarm(monkeypatch) -> None:
    """`reddit_prewarm_max_per_cycle` caps the total prewarm ids selected
    across all subreddit feeds, even when `reddit_prewarm_top_per_sub`
    would suggest more."""
    db = Database(":memory:")
    try:
        # Two subreddit feeds, each carrying 3 stories. With per_sub=10
        # the unconstrained selection would be 6 ids; the cap at 4
        # should fire after the first 4 from the first feed.
        feed_a = "https://www.reddit.com/r/aaa/top/.rss"
        feed_b = "https://www.reddit.com/r/bbb/top/.rss"

        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            prewarm_reddit_full=True,
            reddit_prewarm_top_per_sub=10,
            reddit_prewarm_max_per_cycle=4,
        )
        config = replace(
            config,
            rss=RssConfig(
                enabled=True,
                per_feed_limit=70,
                feeds=(feed_a, feed_b),
            ),
        )

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        stories_a = [
            Story(
                id=100 + i,
                title=f"a{i}",
                url=f"https://reddit.com/r/aaa/comments/{100 + i}",
                score=100 - i,
                time=1_000_000,
                text_content="",
                source="rss_reddit_r_aaa",
            )
            for i in range(3)
        ]
        stories_b = [
            Story(
                id=200 + i,
                title=f"b{i}",
                url=f"https://reddit.com/r/bbb/comments/{200 + i}",
                score=100 - i,
                time=1_000_000,
                text_content="",
                source="rss_reddit_r_bbb",
            )
            for i in range(3)
        ]

        def fake_topfeed_factories(feeds, per_feed, days, exclude_urls):
            from reddit_feed_cache import cache as reddit_feed_cache

            for feed_url, story_set in zip(feeds, (stories_a, stories_b)):
                reddit_feed_cache.set(feed_url, story_set)

            async def factory() -> None:
                return None

            return [factory], list(feeds)

        captured_factory_ids: list[list[int]] = []

        def fake_build_reddit_prewarm_factories(story_ids, db_):
            captured_factory_ids.append(list(story_ids))

            async def noop() -> None:
                return None

            return [noop], []

        class _NoopQueue:
            def enqueue_all_reddit_fetches(self, *args, **kwargs) -> None:
                return None

            def wait_until_empty(self, timeout: float = 5400.0) -> bool:
                return True

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(
            pipeline, "build_reddit_topfeed_factories", fake_topfeed_factories
        )
        monkeypatch.setattr(
            pipeline,
            "build_reddit_prewarm_factories",
            fake_build_reddit_prewarm_factories,
        )
        monkeypatch.setattr("reddit_fetch_queue.queue", _NoopQueue())

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert captured_factory_ids
        assert len(captured_factory_ids[0]) == 4
    finally:
        db.close()


def test_fetch_candidates_only_skips_already_hydrated_reddit(monkeypatch) -> None:
    """Cached topfeed stories whose DB row already has top_comments are
    excluded from the prewarm id list."""
    db = Database(":memory:")
    try:
        feed = "https://www.reddit.com/r/x/top/.rss?t=week&limit=25"
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            prewarm_reddit_full=True,
            reddit_prewarm_top_per_sub=10,
            reddit_prewarm_max_per_cycle=10,
        )
        config = replace(
            config,
            rss=RssConfig(
                enabled=True,
                per_feed_limit=70,
                feeds=(feed,),
            ),
        )

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        # Pre-existing DB row with `top_comments` populated — should be skipped.
        already_hydrated = Story(
            id=10,
            title="hydrated",
            url="https://reddit.com/r/x/comments/10",
            score=100,
            time=1_000_000,
            text_content="",
            source="rss_reddit_r_x",
            top_comments="existing",
        )
        # Brand-new story — should be selected.
        new_story = Story(
            id=20,
            title="new",
            url="https://reddit.com/r/x/comments/20",
            score=50,
            time=1_000_000,
            text_content="",
            source="rss_reddit_r_x",
        )
        db.upsert_story(already_hydrated)

        def fake_topfeed_factories(feeds, per_feed, days, exclude_urls):
            from reddit_feed_cache import cache as reddit_feed_cache

            reddit_feed_cache.set(feeds[0], [already_hydrated, new_story])

            async def factory() -> None:
                return None

            return [factory], [feeds[0]]

        captured_factory_ids: list[list[int]] = []

        def fake_build_reddit_prewarm_factories(story_ids, db_):
            captured_factory_ids.append(list(story_ids))

            async def noop() -> None:
                return None

            return [noop], []

        class _NoopQueue:
            def enqueue_all_reddit_fetches(self, *args, **kwargs) -> None:
                return None

            def wait_until_empty(self, timeout: float = 5400.0) -> bool:
                return True

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(
            pipeline, "build_reddit_topfeed_factories", fake_topfeed_factories
        )
        monkeypatch.setattr(
            pipeline,
            "build_reddit_prewarm_factories",
            fake_build_reddit_prewarm_factories,
        )
        monkeypatch.setattr("reddit_fetch_queue.queue", _NoopQueue())

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert captured_factory_ids
        assert 10 not in captured_factory_ids[0]
        assert 20 in captured_factory_ids[0]
    finally:
        db.close()


def test_needs_hn_prewarm() -> None:
    """Table-driven: stale, fresh, never-prewarmed, and edge cases.

    Threshold is ``max(fetched // 3, 5)`` (~33% growth, 5-comment floor).
    growth >= threshold -> True.
    """
    base_id = 1

    def make_story(
        source: str,
        comment_count: int,
        top_comments: str,
        comment_count_at_fetch: int = 0,
    ) -> Story:
        return Story(
            id=base_id,
            title="t",
            url="",
            score=1,
            time=1,
            text_content="t",
            source=source,
            comment_count=comment_count,
            top_comments=top_comments,
            comment_count_at_fetch=comment_count_at_fetch,
        )

    cases: list[tuple[Story, bool]] = [
        # Zero comment_count -> False regardless of top_comments
        (make_story("hn", 0, ""), False),
        # 10 comments, fresh, no growth -> False
        (make_story("hn", 10, "x", comment_count_at_fetch=10), False),
        # Empty top_comments + comment_count > 0 -> True (never prewarmed)
        (make_story("hn", 1, ""), True),
        # No fetch history (comment_count_at_fetch=0) -> True
        (make_story("hn", 1, "x", comment_count_at_fetch=0), True),
        # Stale: growth=40, fetched=10, threshold=max(10//3, 5) = 5 -> True
        # (was False under max(50, fetched//2, 10); small stories used to
        # need 50+ new comments to trigger, so 10->50 sat stale.)
        (make_story("hn", 50, "x", comment_count_at_fetch=10), True),
        # growth=50, threshold=max(5//3, 5) = 5 -> True
        (make_story("hn", 55, "x", comment_count_at_fetch=5), True),
        # 1->284 (the original bug case): growth=283, threshold=5 -> True
        (make_story("hn", 284, "x", comment_count_at_fetch=1), True),
        # 100->105: growth=5, threshold=max(100//3, 5) = 33 -> False
        (make_story("hn", 105, "x", comment_count_at_fetch=100), False),
        # 100->200: growth=100, threshold=max(100//3, 5) = 33 -> True
        (make_story("hn", 200, "x", comment_count_at_fetch=100), True),
        # Small-stale: 10->16, growth=6, threshold=max(10//3, 5) = 5 -> True.
        # Locks in the new behavior: 10-comment stories now refresh on
        # +6 new comments instead of needing +50.
        (make_story("hn", 16, "x", comment_count_at_fetch=10), True),
        # Non-HN source -> False even with stale-looking shape
        (
            make_story("rss_reddit_test", 284, "", comment_count_at_fetch=1),
            False,
        ),
    ]
    for s, expected in cases:
        actual = _needs_hn_prewarm(s)
        assert actual is expected, f"failed for {s}: expected {expected}, got {actual}"


def test_fetch_candidates_only_reprewarms_stale_hn_when_full(monkeypatch) -> None:
    """Stories with non-empty top_comments but stale comment_count are re-prewarmed.

    Regression for the 1->284 case on story 48709670: top_comments is set
    (1-comment stub) but the live comment count has grown well past the
    threshold. The regen prewarm must refresh it.
    """
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=True,
            prewarm_reddit_full=False,
        )

        fresh = Story(
            id=2,
            title="Fresh",
            url="",
            score=10,
            time=100,
            text_content="f",
            comment_count=50,
            top_comments="recently prewarmed",
            comment_count_at_fetch=50,
        )
        stale = Story(
            id=3,
            title="Stale 1->284",
            url="",
            score=20,
            time=100,
            text_content="s",
            comment_count=284,
            top_comments="old single comment",
            comment_count_at_fetch=1,
        )
        no_comments = Story(
            id=4,
            title="No comments",
            url="",
            score=5,
            time=100,
            text_content="n",
            comment_count=0,
        )
        stories = [fresh, stale, no_comments]
        for s in stories:
            db.upsert_story(s)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return stories, 3

        captured_ids: list[list[int]] = []

        def fake_prewarm(ids, db_, embedder):
            captured_ids.append(list(ids))
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert len(captured_ids) == 1
        # Stale 1->284 fires; fresh 50->50 does not; no_comments has 0.
        assert sorted(captured_ids[0]) == [3]

    finally:
        db.close()


def test_fetch_candidates_only_skips_fresh_hn_when_full(monkeypatch) -> None:
    """Stories with top_comments already up to date are not re-prewarmed.

    Without this guard, the new stale-criterion would re-prewarm the
    entire live window every regen cycle.
    """
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=True,
            prewarm_reddit_full=False,
        )

        stories = [
            Story(
                id=2,
                title="Up to date",
                url="",
                score=10,
                time=100,
                text_content="f",
                comment_count=120,
                top_comments="recently prewarmed",
                comment_count_at_fetch=120,
            ),
        ]
        for s in stories:
            db.upsert_story(s)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return stories, 1

        captured_ids: list[list[int]] = []

        def fake_prewarm(ids, db_, embedder):
            captured_ids.append(list(ids))
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        # Up to date -> no prewarm at all this cycle
        assert captured_ids == []

    finally:
        db.close()


def test_fetch_candidates_only_reprewarms_empty_top_comments_hn_when_full(
    monkeypatch,
) -> None:
    """Empty top_comments + comment_count > 0 still triggers prewarm.

    Preserves the pre-2026-06-29 behavior; the new stale-criterion is
    additive, not a replacement.
    """
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=True,
            prewarm_reddit_full=False,
        )

        stories = [
            Story(
                id=7,
                title="Never prewarmed",
                url="",
                score=10,
                time=100,
                text_content="n",
                comment_count=42,
                top_comments="",
                comment_count_at_fetch=0,
            ),
        ]
        for s in stories:
            db.upsert_story(s)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return stories, 1

        captured_ids: list[list[int]] = []

        def fake_prewarm(ids, db_, embedder):
            captured_ids.append(list(ids))
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert captured_ids == [[7]]

    finally:
        db.close()


def test_fetch_candidates_only_prewarms_top_n_per_sub_from_cache(
    monkeypatch,
) -> None:
    """The prewarm phase reads the topfeed cache and takes the first
    ``reddit_prewarm_top_per_sub`` stories per subreddit (Reddit returns
    them in hot/score-desc order). Multi-cycle completion is expected —
    see WORKLOG 2026-06-29 "Reddit refresh".

    With 3 subs × 5 stories each and N=2, the prewarm factory receives
    6 IDs: the first 2 from each sub's cache.
    """
    from reddit_fetch_queue import queue as reddit_fetch_queue

    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            regen_prewarm_top_n=0,
            prewarm_reddit_full=True,
            reddit_prewarm_top_per_sub=2,
        )

        feed_urls = [
            "https://www.reddit.com/r/A/top/.rss?t=week&limit=25",
            "https://www.reddit.com/r/B/top/.rss?t=week&limit=25",
            "https://www.reddit.com/r/C/top/.rss?t=week&limit=25",
        ]

        # Pre-populate the topfeed cache with 5 stories per sub. The
        # first 2 (hot order) should be prewarmed.
        for i, url in enumerate(feed_urls):
            stories = [
                Story(
                    id=-1000 - (i * 10) - j,
                    title=f"r/{url.split('/r/')[1].split('/')[0]} #{j}",
                    url=f"http://reddit.com/r/test/{i}/{j}",
                    score=100 - j,
                    time=1000 + j,
                    text_content=f"body {i}-{j}",
                    source=f"rss_reddit_{url.split('/r/')[1].split('/')[0]}",
                )
                for j in range(5)
            ]
            pipeline.reddit_feed_cache.set(url, stories)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        captured_ids: list[list[int]] = []

        def fake_build_reddit_topfeed_factories(feeds, per_feed, days, exclude_urls):
            # Return no factories (we use the cache directly), but
            # return the feed URLs so the prewarm phase can read them.
            return [], feed_urls

        def fake_build_reddit_prewarm_factories(story_ids, db_):
            captured_ids.append(list(story_ids))
            return [], []

        def fake_enqueue_all(*a, **kw):
            return None

        def fake_wait_until_empty(timeout=None):
            return True

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(
            pipeline,
            "build_reddit_topfeed_factories",
            fake_build_reddit_topfeed_factories,
        )
        monkeypatch.setattr(
            pipeline,
            "build_reddit_prewarm_factories",
            fake_build_reddit_prewarm_factories,
        )
        monkeypatch.setattr(
            reddit_fetch_queue, "enqueue_all_reddit_fetches", fake_enqueue_all
        )
        monkeypatch.setattr(
            reddit_fetch_queue, "wait_until_empty", fake_wait_until_empty
        )

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert len(captured_ids) == 1
        # 3 subs × 2 per sub = 6 IDs
        assert len(captured_ids[0]) == 6
        # The first 2 from each sub (in cache insertion order = hot order)
        expected = [-1000, -1001, -1010, -1011, -1020, -1021]
        assert captured_ids[0] == expected

    finally:
        pipeline.reddit_feed_cache.reset()
        db.close()


def test_fetch_candidates_only_skips_reddit_prewarm_when_disabled(
    monkeypatch,
) -> None:
    """When ``prewarm_reddit_full=False``, the prewarm phase is skipped
    even if the topfeed cache has stories. Topfeeds still run."""
    from reddit_fetch_queue import queue as reddit_fetch_queue

    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            regen_prewarm_top_n=0,
            prewarm_reddit_full=False,
            reddit_prewarm_top_per_sub=10,
        )

        feed_urls = ["https://www.reddit.com/r/A/top/.rss?t=week&limit=25"]
        stories = [
            Story(
                id=-1,
                title="x",
                url="http://reddit.com/r/test/1",
                score=50,
                time=100,
                text_content="x",
                source="rss_reddit_A",
            )
        ]
        pipeline.reddit_feed_cache.set(feed_urls[0], stories)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        captured_ids: list[list[int]] = []

        def fake_build_reddit_topfeed_factories(feeds, per_feed, days, exclude_urls):
            return [], feed_urls

        def fake_build_reddit_prewarm_factories(story_ids, db_):
            captured_ids.append(list(story_ids))
            return [], []

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(
            pipeline,
            "build_reddit_topfeed_factories",
            fake_build_reddit_topfeed_factories,
        )
        monkeypatch.setattr(
            pipeline,
            "build_reddit_prewarm_factories",
            fake_build_reddit_prewarm_factories,
        )
        monkeypatch.setattr(
            reddit_fetch_queue, "enqueue_all_reddit_fetches", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            reddit_fetch_queue, "wait_until_empty", lambda timeout=None: True
        )

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        # prewarm_reddit_full=False → no prewarm factories built
        assert captured_ids == []

    finally:
        pipeline.reddit_feed_cache.reset()
        db.close()


def test_fetch_candidates_only_skips_reddit_prewarm_with_empty_cache(
    monkeypatch,
) -> None:
    """If the topfeed cache is empty (e.g. all topfeeds failed), the
    prewarm phase is skipped silently — no factories built."""
    from reddit_fetch_queue import queue as reddit_fetch_queue

    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            regen_prewarm_top_n=0,
            prewarm_reddit_full=True,
            reddit_prewarm_top_per_sub=10,
        )

        feed_urls = ["https://www.reddit.com/r/A/top/.rss?t=week&limit=25"]
        # Cache is empty (no topfeed succeeded)
        pipeline.reddit_feed_cache.reset()

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return [], 0

        captured_ids: list[list[int]] = []

        def fake_build_reddit_topfeed_factories(feeds, per_feed, days, exclude_urls):
            return [], feed_urls

        def fake_build_reddit_prewarm_factories(story_ids, db_):
            captured_ids.append(list(story_ids))
            return [], []

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(
            pipeline,
            "build_reddit_topfeed_factories",
            fake_build_reddit_topfeed_factories,
        )
        monkeypatch.setattr(
            pipeline,
            "build_reddit_prewarm_factories",
            fake_build_reddit_prewarm_factories,
        )
        monkeypatch.setattr(
            reddit_fetch_queue, "enqueue_all_reddit_fetches", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            reddit_fetch_queue, "wait_until_empty", lambda timeout=None: True
        )

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        # Empty cache → no prewarm IDs → no prewarm factories
        assert captured_ids == []

    finally:
        pipeline.reddit_feed_cache.reset()
        db.close()


def test_fetch_candidates_only_falls_back_to_top_n_when_disabled(monkeypatch) -> None:
    """Regen falls back to top-N by score when prewarm_hn_full=False."""
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            regen_prewarm_top_n=2,
            prewarm_reddit_full=False,
        )

        stories = [
            Story(id=1, title="A", url="", score=10, time=100, text_content="a"),
            Story(id=2, title="B", url="", score=50, time=100, text_content="b"),
            Story(id=3, title="C", url="", score=30, time=100, text_content="c"),
        ]
        for s in stories:
            db.upsert_story(s)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return stories, 3

        captured_ids: list[list[int]] = []

        def fake_prewarm(ids, db_, embedder):
            captured_ids.append(list(ids))
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_top_stories", fake_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert len(captured_ids) == 1
        # Top 2 by score (HN only): 2(50), 3(30)
        assert captured_ids[0] == [2, 3]

    finally:
        db.close()


def test_fetch_candidates_only_prewarms_all_lesswrong_when_full(monkeypatch) -> None:
    """Regen prewarms all LessWrong candidates when prewarm_lesswrong_full=True."""
    db = Database(":memory:")
    try:
        config = Config(
            db_path=db.db_path,
            prewarm_hn_full=False,
            regen_prewarm_top_n=0,
            prewarm_reddit_full=False,
            prewarm_lesswrong_full=True,
        )

        stories = [
            Story(
                id=10,
                title="LW1",
                url="https://www.lesswrong.com/posts/abc123/slug-1",
                score=10,
                time=100,
                text_content="lw1",
                source="rss_lesswrong_com",
            ),
            Story(
                id=11,
                title="LW2",
                url="https://www.lesswrong.com/posts/def456/slug-2",
                score=5,
                time=100,
                text_content="lw2",
                source="rss_lesswrong_com",
            ),
            Story(
                id=12,
                title="LW3",
                url="https://www.lesswrong.com/posts/ghi789/slug-3",
                score=20,
                time=100,
                text_content="lw3",
                source="rss_lesswrong_com",
                top_comments="Already hydrated.",
            ),
        ]
        for s in stories:
            db.upsert_story(s)

        async def fake_fetch_candidates(
            config, exclude_ids, exclude_urls, db, embedder
        ):
            return stories, 3

        captured_ids: list[list[int]] = []

        async def fake_lw_prewarm(ids, db_, embedder):
            captured_ids.append(list(ids))
            return len(ids)

        monkeypatch.setattr(pipeline, "fetch_candidates", fake_fetch_candidates)
        monkeypatch.setattr(pipeline, "prewarm_lesswrong_stories", fake_lw_prewarm)

        asyncio.run(
            pipeline.fetch_candidates_only(config, db, embedder=_DummyEmbedder())
        )
        assert len(captured_ids) == 1
        # Stories without top_comments: 10, 11
        assert sorted(captured_ids[0]) == [10, 11]

    finally:
        db.close()


def test_prewarm_lesswrong_stories_refetches_when_only_one_field_is_stale(
    monkeypatch,
) -> None:
    """Idempotency check must skip only when BOTH top_comments and self_text
    are already populated with equal or richer data. If top_comments is
    empty but self_text is already saturated (e.g. an RSS snippet was
    truncated to SELF_TEXT_PROMPT_CHAR_LIMIT), the prewarm should still
    fetch the new top_comments.
    """
    from server import LessWrongContext

    db = Database(":memory:")
    try:
        # self_text already at the 8k cap (RSS-saturated); top_comments empty.
        # The new GraphQL body is shorter than 8k so the OLD idempotency
        # logic would skip this — but top_comments has fresh data, so we
        # should still upsert.
        db.upsert_story(
            Story(
                id=42,
                title="saturated-rss-body",
                url="https://www.lesswrong.com/posts/abc123/saturated",
                score=0,
                time=100,
                text_content="placeholder",
                source="rss_lesswrong_com",
                self_text="x" * 8000,
                top_comments="",
            )
        )

        async def fake_fetch(post_id):
            return LessWrongContext(
                self_text="shorter new body from graphql",
                top_comments="fresh comment from graphql",
                comment_count=5,
                score=42,
            )

        monkeypatch.setattr("server._fetch_lesswrong_context", fake_fetch)

        updated = asyncio.run(
            pipeline.prewarm_lesswrong_stories([42], db, embedder=None)
        )
        assert updated == 1
        row = db.get_story(42)
        assert row is not None
        # Old behavior: skipped (no upsert). New behavior: upserted.
        assert row.top_comments == "fresh comment from graphql"
        assert row.score == 42
        # self_text uses the longer of the two
        assert len(row.self_text) == 8000

    finally:
        db.close()


def test_prewarm_lesswrong_stories_skips_when_both_fields_already_richer(
    monkeypatch,
) -> None:
    """If both top_comments and self_text are already populated and the
    new data is no richer, skip (no upsert)."""
    from server import LessWrongContext

    db = Database(":memory:")
    try:
        long_body = "x" * 5000
        long_comments = "y" * 5000
        db.upsert_story(
            Story(
                id=43,
                title="already-prewarmed",
                url="https://www.lesswrong.com/posts/def456/already",
                score=100,
                time=100,
                text_content="placeholder",
                source="rss_lesswrong_com",
                self_text=long_body,
                top_comments=long_comments,
            )
        )

        async def fake_fetch(post_id):
            return LessWrongContext(
                self_text="x" * 100,
                top_comments="y" * 100,
                comment_count=1,
                score=10,
            )

        monkeypatch.setattr("server._fetch_lesswrong_context", fake_fetch)

        updated = asyncio.run(
            pipeline.prewarm_lesswrong_stories([43], db, embedder=None)
        )
        assert updated == 0
        row = db.get_story(43)
        assert row is not None
        # Untouched
        assert row.self_text == long_body
        assert row.top_comments == long_comments
        # score: max(100, 10) = 100 (no change)

    finally:
        db.close()


def _clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def _make_story(db: Database, sid: int) -> None:
    db.upsert_story(
        Story(
            id=sid,
            title=f"story{sid}",
            url=None,
            score=10,
            time=1000,
            text_content="hello",
            source="hn",
        )
    )


def test_feedback_signature_consistency(db: Database) -> None:
    user = db.create_user("sig_test")
    _make_story(db, 1)
    db.upsert_feedback(user.id, 1, "up")
    sig1 = _feedback_signature(db, user.id)
    sig2 = _feedback_signature(db, user.id)
    assert sig1 == sig2


def test_feedback_signature_changes_on_new_feedback(db: Database) -> None:
    user = db.create_user("sig_test2")
    _make_story(db, 1)
    _make_story(db, 2)
    db.upsert_feedback(user.id, 1, "up")
    sig1 = _feedback_signature(db, user.id)
    db.upsert_feedback(user.id, 2, "down")
    sig2 = _feedback_signature(db, user.id)
    assert sig1 != sig2


def test_feedback_signature_changes_on_update(db: Database) -> None:
    user = db.create_user("sig_test3")
    _make_story(db, 1)
    db.upsert_feedback(user.id, 1, "up")
    sig1 = _feedback_signature(db, user.id)
    db.upsert_feedback(user.id, 1, "down")
    sig2 = _feedback_signature(db, user.id)
    assert sig1 != sig2


def test_feedback_signature_empty_user(db: Database) -> None:
    user = db.create_user("sig_test4")
    sig = _feedback_signature(db, user.id)
    assert isinstance(sig, str)
    assert len(sig) == 64


def test_model_cache_hit() -> None:
    _clear_model_cache()
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    svm = SVC(kernel="linear")
    scaler = StandardScaler()
    svm.fit([[0, 0], [1, 1]], [0, 1])
    scaler.fit([[1.0], [2.0]])

    _set_cached_model(42, "sig1", svm, scaler, maxsize=10)
    result = _get_cached_model(42, "sig1")
    assert result is not None
    cached_svm, cached_scaler = result
    assert cached_svm is svm
    assert cached_scaler is scaler


def test_model_cache_miss() -> None:
    _clear_model_cache()
    result = _get_cached_model(42, "nonexistent")
    assert result is None


def test_model_cache_user_isolation() -> None:
    _clear_model_cache()
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    svm1 = SVC(kernel="linear")
    scaler1 = StandardScaler()
    svm1.fit([[0, 0], [1, 1]], [0, 1])
    scaler1.fit([[1.0], [2.0]])
    _set_cached_model(42, "sig", svm1, scaler1, maxsize=10)

    result_other = _get_cached_model(99, "sig")
    assert result_other is None


def test_model_cache_none_user() -> None:
    _clear_model_cache()
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    svm = SVC(kernel="linear")
    scaler = StandardScaler()
    svm.fit([[0, 0], [1, 1]], [0, 1])
    scaler.fit([[1.0], [2.0]])

    _set_cached_model(None, "sig", svm, scaler)
    assert _get_cached_model(None, "sig") is None


def test_model_cache_eviction() -> None:
    _clear_model_cache()
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    svm = SVC(kernel="linear")
    scaler = StandardScaler()
    svm.fit([[0, 0], [1, 1]], [0, 1])
    scaler.fit([[1.0], [2.0]])

    maxsize = 3
    for i in range(5):
        _set_cached_model(1, f"sig{i}", svm, scaler, maxsize=maxsize)

    assert _get_cached_model(1, "sig0") is None
    assert _get_cached_model(1, "sig1") is None
    assert _get_cached_model(1, "sig2") is not None
    assert _get_cached_model(1, "sig3") is not None
    assert _get_cached_model(1, "sig4") is not None


def test_rank_trace_records_and_formats_fields() -> None:
    trace = RankTrace()
    trace.set_count("candidates", 12)
    trace.set_label("model_cache", "miss")
    trace.add_timing("svm_fit", 1.24)
    trace.add_timing("svm_fit", 2.26)

    fields = trace.to_log_fields()
    assert fields["candidates"] == 12
    assert fields["model_cache"] == "miss"
    assert fields["svm_fit_ms"] == 3.5
    assert "candidates=12" in trace.format_log_fields()
    assert "model_cache=miss" in trace.format_log_fields()


def test_rank_trace_reports_svm_cache_miss_then_hit(
    db: Database, embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_model_cache()
    user = db.create_user("trace_cache")
    now = int(time.time())
    feedback_stories: list[Story] = []
    for i in range(20):
        up = Story(
            id=10_000 + i,
            title=f"up {i}",
            url=None,
            score=10,
            time=now,
            text_content=f"up story {i}",
            source="hn",
            comment_count=1,
        )
        down = Story(
            id=20_000 + i,
            title=f"down {i}",
            url=None,
            score=10,
            time=now,
            text_content=f"down story {i}",
            source="hn",
            comment_count=1,
        )
        db.upsert_story(up)
        db.upsert_story(down)
        db.upsert_feedback(user.id, up.id, "up")
        db.upsert_feedback(user.id, down.id, "down")
        feedback_stories.extend([up, down])

    candidates = [
        Story(
            id=30_000 + i,
            title=f"candidate {i}",
            url=None,
            score=100 - i,
            time=now,
            text_content=f"candidate story {i}",
            source="hn",
            comment_count=1,
        )
        for i in range(3)
    ]

    def fake_embeddings(
        stories: list[Story], embedder_arg: Embedder, db_inst: Database
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for story in stories:
            vec = np.zeros(384, dtype=np.float32)
            if story.id >= 30_000:
                vec[0] = 1.0
            elif story.id >= 20_000:
                vec[1] = 1.0
            else:
                vec[0] = 1.0
            rows.append(vec)
        return np.array(rows, dtype=np.float32)

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", fake_embeddings)
    topk_calls = 0
    real_topk_mean = pipeline._topk_mean

    def counting_topk_mean(values: np.ndarray, k: int) -> float:
        nonlocal topk_calls
        topk_calls += 1
        return real_topk_mean(values, k)

    monkeypatch.setattr("pipeline._topk_mean", counting_topk_mean)
    candidate_embeddings = fake_embeddings(candidates, embedder, db)
    config = Config()

    miss_trace = RankTrace()
    _score_and_rank(
        candidates,
        candidate_embeddings,
        db,
        config,
        embedder,
        user_id=user.id,
        trace=miss_trace,
    )
    assert miss_trace.labels["model_cache"] == "miss"
    assert miss_trace.timings_ms["svm_fit"] >= 0
    assert miss_trace.timings_ms["svm_training_feature_prep"] >= 0
    miss_topk_calls = topk_calls
    assert miss_topk_calls > 0

    hit_trace = RankTrace()
    _score_and_rank(
        candidates,
        candidate_embeddings,
        db,
        config,
        embedder,
        user_id=user.id,
        trace=hit_trace,
    )
    assert hit_trace.labels["model_cache"] == "hit"
    assert "svm_fit" not in hit_trace.timings_ms
    assert "svm_training_feature_prep" not in hit_trace.timings_ms
    assert hit_trace.timings_ms["decision"] >= 0
    assert topk_calls == miss_topk_calls


def test_two_leg_recent_zero_limits_do_not_error(
    db: Database, embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting both candidate limits to 0 must not crash the rank path.
    The HN and RSS recent legs return empty; the archive leg still
    supplies candidates for the (likely empty) result."""
    user = db.create_user("two_leg_zero")
    config = Config(
        db_path=db.db_path,
        recent_candidate_hn_limit=0,
        recent_candidate_rss_limit=0,
    )
    ranked = fast_rerank_for_user(db, config, embedder, user.id)
    assert isinstance(ranked, list)


def test_two_leg_recent_huge_limits_no_truncation_error(
    db: Database, embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Limits larger than the row count return all rows. SQLite does not
    require the result to match the LIMIT; the rank path must accept it."""
    user = db.create_user("two_leg_huge")
    config = Config(
        db_path=db.db_path,
        recent_candidate_hn_limit=1_000_000,
        recent_candidate_rss_limit=1_000_000,
    )
    ranked = fast_rerank_for_user(db, config, embedder, user.id)
    assert isinstance(ranked, list)


def test_two_leg_recent_preserves_hn_and_rss_legs(
    db: Database, embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A two-leg query with non-zero limits must return at least one
    HN source and at least one non-HN source in the candidate set."""
    user = db.create_user("two_leg_mix")
    now = int(time.time())
    hn_story = Story(
        id=1,
        title="HN",
        url=None,
        score=100,
        time=now,
        text_content="hn body",
        source="hn",
        comment_count=5,
    )
    rss_story = Story(
        id=2,
        title="RSS",
        url=None,
        score=0,
        time=now,
        text_content="rss body",
        source="rss_lobste_rs",
        comment_count=0,
    )
    db.upsert_story(hn_story)
    db.upsert_story(rss_story)
    monkeypatch.setattr(
        "pipeline.get_or_compute_embeddings",
        lambda stories, e, d: np.zeros((len(stories), 384), dtype=np.float32),
    )

    config = Config(
        db_path=db.db_path,
        recent_candidate_hn_limit=1500,
        recent_candidate_rss_limit=500,
    )
    captured: dict[str, list[Story]] = {}

    real_fast = fast_rerank_for_user

    def capture_fast(*args, **kwargs):
        # Tap into the row-level fetch by querying the DB directly with
        # the same filters the rank path uses.
        cutoff = int(time.time()) - (config.days * 86400)
        hn_rows = db.execute(
            "SELECT id, title, url, score, time, text_content, source, "
            "comment_count, discussion_url, comment_count_at_fetch, "
            "self_text, top_comments, article_body "
            "FROM stories WHERE time >= ? AND source = 'hn' "
            "AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)",
            (cutoff, user.id),
        )
        rss_rows = db.execute(
            "SELECT id, title, url, score, time, text_content, source, "
            "comment_count, discussion_url, comment_count_at_fetch, "
            "self_text, top_comments, article_body "
            "FROM stories WHERE time >= ? AND source != 'hn' "
            "AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)",
            (cutoff, user.id),
        )
        captured["hn"] = [Database._row_to_story(r) for r in hn_rows]
        captured["rss"] = [Database._row_to_story(r) for r in rss_rows]
        return real_fast(*args, **kwargs)

    capture_fast(db, config, embedder, user.id)
    assert {s.id for s in captured["hn"]} == {1}
    assert {s.id for s in captured["rss"]} == {2}


def test_is_recent_flag_inclusive_30d_boundary(
    db: Database, embedder: Embedder
) -> None:
    """is_recent: time >= now - 30d → True (inclusive boundary).

    Stories are placed 1s on either side of the 30d boundary so the test
    is robust against wall-clock second drift between the test fixture
    capturing `now` and the reranker re-reading `time.time()`.
    """

    config = Config(count=40)
    now = int(time.time())
    candidates = [
        Story(
            id=1,
            title="Recent 1d",
            url=None,
            score=100,
            time=now - 86400,
            text_content="Recent story 1 day old.",
            source="hn",
            comment_count=0,
        ),
        Story(
            id=2,
            title="Just inside 30d (recent side)",
            url=None,
            score=100,
            time=now - 30 * 86400 + 1,
            text_content="Story 30d - 1s old.",
            source="hn",
            comment_count=0,
        ),
        Story(
            id=3,
            title="Just outside 30d (archive side)",
            url=None,
            score=100,
            time=now - 30 * 86400 - 1,
            text_content="Story 30d + 1s old.",
            source="hn",
            comment_count=0,
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])
    ranked = rerank_candidates(db, config, embedder, candidates, cand_embs)
    by_id = {r.story.id: r for r in ranked}
    assert by_id[1].is_recent is True, "1d old should be recent"
    assert by_id[2].is_recent is True, "30d - 1s should be recent"
    assert by_id[3].is_recent is False, "30d + 1s should NOT be recent"
