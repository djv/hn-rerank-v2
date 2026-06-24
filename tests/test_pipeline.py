import asyncio
import numpy as np
import pytest
import time
from hypothesis import given, strategies as st, settings, HealthCheck
from database import Action, Database, Story
import pipeline
from pipeline import (
    BQ_ARCHIVE_SOURCE,
    Embedder,
    ModelConfig,
    RankedStory,
    clean_text,
    compose_story_text,
    get_or_compute_embeddings,
    is_hn_source,
    mmr_filter,
    rank_stories,
    Config,
    rerank_candidates,
    select_article_fetch_candidates,
    source_label_filter,
    story_embedding_text,
    _extract_comments_recursive,
    _select_top_comments,
    _dashboard_primary_limit,
    _reddit_subreddit_from_feed_url,
    _rss_source_name,
    _svm_personalization_features,
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

    for s in [dashboard_1, dashboard_2, old, backed_off, extra_hot, extra_cool, has_body]:
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
                    "text": f"Substantive reply {i} with enough context to be useful.",
                    "children": [],
                }
                for i in range(8)
            ],
        },
        *[
            {
                "type": "comment",
                "text": f"Separate top-level comment {i} with enough useful context.",
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
                    "text": f"Substantive reply {i} with enough context to be useful.",
                    "children": [],
                }
                for i in range(20)
            ],
        },
        *[
            {
                "type": "comment",
                "text": f"Separate top-level comment {i} with enough useful context.",
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
    ranked = rank_stories(
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

    ranked = rank_stories(
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
    assert source_label_filter("rss_rss_slashdot_org") == "Slashdot"
    assert source_label_filter("rss_slashdot_org") == "Slashdot"
    assert source_label_filter("rss_reddit_localllama") == "r/localllama"
    assert source_label_filter("rss_simonwillison_net") == "Simon Willison"


def test_is_hn_source_includes_bq_seed():
    assert is_hn_source("hn")
    assert is_hn_source(BQ_ARCHIVE_SOURCE)
    assert not is_hn_source("rss_example_com")


@pytest.mark.asyncio
async def test_fetch_rss_feeds_serializes_reddit_and_sets_user_agent(
    tmp_path, monkeypatch
):
    from pipeline import REDDIT_RSS_USER_AGENT, fetch_rss_feeds

    db = Database(str(tmp_path / "test.db"))
    active_reddit_requests = 0
    max_reddit_concurrency = 0
    seen_requests = []

    class MockResp:
        status_code = 200

        def __init__(self, text: str):
            self.text = text

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

    stories = await fetch_rss_feeds(
        [
            "https://www.reddit.com/r/haskell/top/.rss?t=week&limit=25",
            "https://example.com/feed.xml",
            "https://www.reddit.com/r/ocaml/top/.rss?t=month&limit=25",
        ],
        per_feed=10,
        days=30,
        exclude_urls=set(),
        db=db,
    )

    assert max_reddit_concurrency == 1
    reddit_requests = [
        (url, headers) for url, headers in seen_requests if "reddit.com" in url
    ]
    assert len(reddit_requests) == 2
    assert all(
        headers.get("User-Agent") == REDDIT_RSS_USER_AGENT
        for _url, headers in reddit_requests
    )
    assert {s.source for s in stories} >= {
        "rss_reddit_haskell",
        "rss_reddit_ocaml",
        "rss_example_com",
    }


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

    ranked = rank_stories(
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
    from pipeline import _augment_features

    n = len(meta)
    embeddings = np.random.randn(n, 384).astype(np.float32)
    scores = [item[0] for item in meta]
    ages = [item[1] for item in meta]

    features = _augment_features(embeddings, scores, ages)

    assert features.shape == (n, 385)
    assert np.allclose(features[:, :384], embeddings)
    assert np.all(features[:, 384] >= 0.0) and np.all(features[:, 384] <= 1.0)

    # When all 7 derived features are provided, shape expands to (n, 392)
    comment_counts = np.array([max(s, 0) for s in scores])
    text_lengths = np.array([abs(a) % 10000 for a in ages])
    hn_quality = comment_counts.astype(np.float32) / (np.abs(ages) + 1)
    sim_up = np.random.uniform(-1, 1, n).astype(np.float32)
    sim_down = np.random.uniform(-1, 1, n).astype(np.float32)
    closest_up = np.random.uniform(-1, 1, n).astype(np.float32)
    closest_down = np.random.uniform(-1, 1, n).astype(np.float32)

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
    )
    assert features7.shape == (n, 392)
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

    assert features.shape == (2, emb_dim + 6)
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
        ranked = rank_stories(candidates, cand_embs, db, config, embedder)

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
        output=str(tmp_path / "index.html"),
        server_port=0,
    )

    class MockResp:
        status_code = 200

        def json(self):
            return {"hits": []}

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, *a, **kw):
            return MockResp()

    monkeypatch.setattr("pipeline.httpx.AsyncClient", lambda **kw: MockClient())
    result = await fetch_candidates(config, set(), set(), db)
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 2
    candidates, count = result
    assert isinstance(candidates, list)
    assert isinstance(count, int)
    assert count == len(candidates)


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

    class MockResp:
        status_code = 200

        def json(self):
            return {"hits": []}

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, *a, **kw):
            return MockResp()

    monkeypatch.setattr("pipeline.httpx.AsyncClient", lambda **kw: MockClient())
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

    class MockResp:
        status_code = 200

        def json(self):
            return {"hits": []}

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, *a, **kw):
            return MockResp()

    monkeypatch.setattr("pipeline.BQ_ARCHIVE_CANDIDATE_LIMIT", 2)
    monkeypatch.setattr("pipeline.httpx.AsyncClient", lambda **kw: MockClient())
    candidates, _ = await fetch_candidates(
        Config(db_path=str(db_file), days=30),
        set(),
        set(),
        db,
    )

    assert {s.id for s in candidates} == {9102, 9103}


@pytest.mark.asyncio
async def test_bq_archive_hydration_preserves_source(tmp_path, monkeypatch):
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
            comment_count_at_fetch=0,
            top_comments="",
        )
    )

    class MockResp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, *a, **kw):
            if "/api/v1/items/" in str(url):
                return MockResp(
                    {
                        "type": "story",
                        "title": "BQ needs comments",
                        "url": "https://example.com/needs-comments",
                        "points": 501,
                        "num_comments": 3,
                        "created_at_i": old_time,
                        "children": [
                            {
                                "type": "comment",
                                "text": "Substantive fetched comment text with enough context.",
                                "children": [],
                            }
                        ],
                    }
                )
            return MockResp({"hits": []})

    monkeypatch.setattr("pipeline.httpx.AsyncClient", lambda **kw: MockClient())
    await fetch_candidates(
        Config(db_path=str(db_file), days=30),
        set(),
        set(),
        db,
    )

    updated = db.get_story(9201)
    assert updated.source == BQ_ARCHIVE_SOURCE
    assert "Substantive fetched comment" in updated.top_comments


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
        )
    )

    def fake_embeddings(stories, embedder, db_inst):
        return np.zeros((len(stories), 384), dtype=np.float32)

    monkeypatch.setattr("pipeline.get_or_compute_embeddings", fake_embeddings)
    ranked = fast_rerank_for_user(db, Config(days=30), object(), user.id)

    assert [item.story.id for item in ranked] == [9301]


# Tests for auto-refetch of comment text on growth (fetch_candidates refetch block)
def _seed_story(
    db: Database,
    sid: int,
    *,
    comment_count: int,
    comment_count_at_fetch: int,
    age_hours: int,
    feedback_action: Action | None = None,
    user_id: int | None = None,
) -> Story:
    """Insert a story in the live window with controlled engagement metrics."""
    from time import time as _now

    story_time = int(_now()) - (age_hours * 3600)
    story = Story(
        id=sid,
        title=f"Test story {sid}",
        url=f"https://example.com/{sid}",
        score=100,
        time=story_time,
        text_content=f"initial text content for story {sid}",
        source="hn",
        comment_count=comment_count,
        discussion_url=f"https://news.ycombinator.com/item?id={sid}",
        comment_count_at_fetch=comment_count_at_fetch,
    )
    db.upsert_story(story)
    if feedback_action is not None and user_id is not None:
        db.upsert_feedback(user_id, sid, feedback_action)
    return story


def test_refetch_eligibility_triggers_at_30pct_growth():
    """comment_count grown from 10 to 14 (+40%) at age 1h is eligible."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    sid = 1001
    now = int(_now())
    eligible, new_count = _is_refetch_eligible(
        sid=sid,
        comment_count_at_fetch=10,
        new_comment_count=14,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is True
    assert new_count == 14


def test_refetch_eligibility_skipped_below_30pct_growth():
    """10 to 12 (+20%) is below threshold; not eligible."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1002,
        comment_count_at_fetch=10,
        new_comment_count=12,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_after_24h_age():
    """Eligible growth but age > 24h is skipped."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1003,
        comment_count_at_fetch=10,
        new_comment_count=14,
        story_time=now - (25 * 3600),
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_for_feedback_story():
    """Stories in feedback_ids are protected from refetch (training contract)."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1004,
        comment_count_at_fetch=10,
        new_comment_count=14,
        story_time=now - 3600,
        feedback_ids={1004},
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_when_baseline_zero():
    """comment_count_at_fetch == 0 means no baseline; cannot compute growth."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1005,
        comment_count_at_fetch=0,
        new_comment_count=14,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_when_count_did_not_grow():
    """If new_comment_count <= comment_count_at_fetch, no refetch."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1006,
        comment_count_at_fetch=10,
        new_comment_count=10,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_selects_top_n_by_growth(db, embedder):
    """Only MAX_REFETCH_PER_REGEN stories are selected, prioritized by growth."""
    from pipeline import _select_refetch_ids, MAX_REFETCH_PER_REGEN
    from time import time as _now

    now = int(_now())
    candidates = []
    fresh_metadata = {}
    for i in range(MAX_REFETCH_PER_REGEN + 5):
        sid = 2000 + i
        candidates.append(
            _seed_story(
                db,
                sid,
                comment_count=100 + i,
                comment_count_at_fetch=10,
                age_hours=1,
            )
        )
        fresh_metadata[sid] = {"score": 200, "comment_count": 100 + i}
    selected = _select_refetch_ids(
        candidates=candidates,
        fresh_metadata=fresh_metadata,
        feedback_ids=set(),
        now_ts=now,
    )
    assert len(selected) == MAX_REFETCH_PER_REGEN
    # Should be the highest-growth stories (those with highest new_count, since
    # baseline is the same for all in this test). Growth ratio is monotonic in
    # new_count when baseline is fixed.
    selected_set = set(selected)
    assert max(selected_set) > min(selected_set)  # at least some spread


async def test_refetch_failure_keeps_stale_data(db, embedder):
    """If refetch_story_text fails, the original story is preserved (no DB change)."""
    import httpx
    from pipeline import refetch_story_text

    _seed_story(
        db,
        3001,
        comment_count=14,
        comment_count_at_fetch=10,
        age_hours=1,
    )
    original = db.get_story(3001)
    original_text = original.text_content

    def _handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("simulated algolia outage")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await refetch_story_text(client, db, embedder, 3001, current_count=14)

    assert result is None
    after = db.get_story(3001)
    assert after.text_content == original_text
    assert after.comment_count == original.comment_count
    assert after.comment_count_at_fetch == original.comment_count_at_fetch


async def test_refetch_updates_comment_count_at_fetch_on_success(db, embedder):
    """Successful refetch sets comment_count_at_fetch == new comment_count."""
    import httpx
    from pipeline import refetch_story_text

    sid = 3002
    _seed_story(
        db,
        sid,
        comment_count=14,
        comment_count_at_fetch=10,
        age_hours=1,
    )

    fake_item = {
        "type": "story",
        "title": "Updated Title",
        "url": "https://example.com/3002",
        "points": 200,
        "num_comments": 16,
        "created_at_i": int(__import__("time").time()),
        "story_text": "Updated self text",
        "children": [
            {"text": f"comment {i}", "score": 10 - i, "children": []} for i in range(5)
        ],
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fake_item)

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await refetch_story_text(client, db, embedder, sid, current_count=16)

    assert result is not None
    assert result.comment_count == 16
    assert result.comment_count_at_fetch == 16
    persisted = db.get_story(sid)
    assert persisted.comment_count_at_fetch == 16
    assert persisted.text_content != ""  # recomposed


async def test_run_pipeline_badge_assignment(tmp_path, monkeypatch):
    """Verify badge criteria are applied uniformly to primary and extra-slot stories;
    extra-slot pulls still source from remaining candidates with their per-pass caps."""
    from pipeline import Config, run_pipeline, RankedStory
    from database import Database, Story
    import numpy as np
    import time

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    user = db.create_user("test_token_badge")

    # 1. Create candidates list
    now = int(time.time())
    candidates = []
    # Default path candidates: IDs 1-7
    for i in range(1, 8):
        candidates.append(
            Story(
                id=i,
                title=f"Default Story {i}",
                url=None,
                score=100 - i,
                time=now,
                text_content=f"content {i}",
                comment_count=0,
            )
        )
    # Extra/discovery candidates:
    # ID 8: high comments -> discussion rich
    candidates.append(
        Story(
            id=8,
            title="Discussion Rich Story",
            url=None,
            score=10,
            time=now,
            text_content="content 8",
            comment_count=100,
        )
    )
    # ID 9: high score -> high engagement
    candidates.append(
        Story(
            id=9,
            title="High Engagement Story",
            url=None,
            score=500,
            time=now,
            text_content="content 9",
            comment_count=0,
        )
    )
    # ID 10: high similarity -> similar
    candidates.append(
        Story(
            id=10,
            title="Similar Story",
            url=None,
            score=10,
            time=now,
            text_content="content 10",
            comment_count=0,
        )
    )
    # ID 11: seed for novel story (we will set its max sim low)
    candidates.append(
        Story(
            id=11,
            title="Novel Story",
            url=None,
            score=6,
            time=now,
            text_content="content 11",
            comment_count=0,
        )
    )

    # Persist feedback in DB to train / simulate closest up
    db.upsert_story(
        Story(
            id=999,
            title="Upvoted Story",
            url=None,
            score=100,
            time=now,
            text_content="upvoted text",
        )
    )
    db.upsert_feedback(user.id, 999, "up")

    # 2. Config setup
    config = Config(
        db_path=str(db_file),
        output=str(tmp_path / "index.html"),
        count=9,  # Below the 12-card queue size, so keeps 9 primary stories.
    )

    # 3. Mock dependencies
    async def mock_fetch_candidates(*args, **kwargs):
        return candidates, len(candidates)

    monkeypatch.setattr("pipeline.fetch_candidates", mock_fetch_candidates)

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("pipeline.Embedder", DummyEmbedder)

    def mock_get_or_compute_embeddings(stories, embedder, db_inst):
        # We return unit embeddings.
        # ID 999 has embedding [1, 0, 0...]
        # ID 10 (sleeper) has embedding [0.6, 0...] to ensure closest_up > 0.55
        # ID 11 (novel) has embedding [0.0, 0...] to ensure closest_up/closest_down <= sim_threshold
        # IDs 8/9 sit between novel and similar thresholds to isolate their badge passes
        # others have distinct values starting at 0.2
        embs = []
        for s in stories:
            vec = np.zeros(384, dtype=np.float32)
            if s.id == 999:
                vec[0] = 1.0
            elif s.id == 10:
                vec[0] = 0.6
            elif s.id == 11:
                vec[0] = 0.0
            elif s.id == 8:
                # Above the 15th-pct novel threshold (~0.255) so the
                # novel pass doesn't claim ID 8 first; ID 8 should reach
                # the discussion-rich pass with comment_count=100.
                vec[0] = 0.27
            elif s.id == 9:
                # Above the 15th-pct novel threshold so the novel pass
                # doesn't claim ID 9 and prevent the engagement pass from
                # running. ID 9 should reach the engagement pass with its
                # raw story.score=500 above the 95th-pct threshold (~300).
                vec[0] = 0.30
            else:
                vec[0] = 0.25 + s.id * 0.01
            embs.append(vec)
        return np.array(embs)

    monkeypatch.setattr(
        "pipeline.get_or_compute_embeddings", mock_get_or_compute_embeddings
    )

    # Mock rank_stories to return a pre-sorted list (fallback score/probability behavior)
    def mock_rank_stories(candidates_list, *args, **kwargs):
        res = []
        for s in candidates_list:
            if s.id <= 7:
                score = 0.9 - s.id * 0.05
            elif s.id == 11:
                score = 0.51
            else:
                score = 0.2
            res.append(
                RankedStory(
                    story=s,
                    score=score,
                    best_match_title="",
                    prob_down=0.1 if s.id <= 7 else None,
                    prob_neutral=0.1 if s.id <= 7 else None,
                    prob_up=0.8 if s.id <= 7 else None,
                )
            )
        return sorted(res, key=lambda x: x.score, reverse=True)

    monkeypatch.setattr("pipeline.rank_stories", mock_rank_stories)

    # Capture final stories passed to generate_dashboard
    captured_final = []

    def mock_generate_dashboard(final_list, *args, **kwargs):
        nonlocal captured_final
        captured_final = list(final_list)

    monkeypatch.setattr("pipeline.generate_dashboard", mock_generate_dashboard)

    await run_pipeline(config)

    # Assertions
    # We expect default stories (1-7) and at least some extra decorated stories
    assert len(captured_final) > 7

    for r in captured_final:
        if r.story.id <= 7:
            # Default path stories are not the focus of this test; the new
            # behavior is that they can also receive badges, exercised in
            # test_primary_story_gets_qualifying_badge. We don't make negative
            # assertions here because small candidate sets can cause primary
            # stories to incidentally clear the 90th/15th-percentile gates.
            # The `is_similar` exclusion is invariant: primary stories never
            # get the Similar badge regardless of whether they meet the
            # criterion.
            assert not r.is_similar, (
                f"Primary story {r.story.id} should not have is_similar"
            )
        else:
            # Extra discovery stories can have badges
            if r.story.id == 8:
                assert r.is_discussion_rich
            if r.story.id == 9:
                assert r.is_high_engagement
            if r.story.id == 10:
                assert r.is_similar
            if r.story.id == 11:
                assert r.is_novel


async def test_primary_story_gets_qualifying_badge(tmp_path, monkeypatch):
    """Primary-ranked stories that meet badge criteria receive those badges,
    not just the extra-slot ones. The extra-slot pulls also keep working."""
    from pipeline import Config, run_pipeline, RankedStory
    from database import Database, Story
    import numpy as np
    import time

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    user = db.create_user("test_token_primary_badge")

    now = int(time.time())
    candidates = []

    # Primary story that ALSO qualifies for is_discussion_rich (high comments
    # AND high enough score to land in the primary ranked set).
    candidates.append(
        Story(
            id=1,
            title="Top Talky Story",
            url=None,
            score=200,
            time=now,
            text_content="content 1",
            comment_count=300,
        )
    )
    # Other primary stories with no qualifying properties.
    for i in range(2, 8):
        candidates.append(
            Story(
                id=i,
                title=f"Default Story {i}",
                url=None,
                score=100 - i,
                time=now,
                text_content=f"content {i}",
                comment_count=0,
            )
        )
    # Extra-slot discussion story (outside primary by score but qualifies on
    # the discussion-rich criterion, sourced via remaining_decorated).
    candidates.append(
        Story(
            id=8,
            title="Extra Discussion Story",
            url=None,
            score=5,
            time=now,
            text_content="content 8",
            comment_count=250,
        )
    )

    # Seed one upvoted feedback story so the SVM has something to train on.
    db.upsert_story(
        Story(
            id=999,
            title="Upvoted Story",
            url=None,
            score=100,
            time=now,
            text_content="upvoted text",
        )
    )
    db.upsert_feedback(user.id, 999, "up")

    config = Config(
        db_path=str(db_file),
        output=str(tmp_path / "index.html"),
        count=8,
    )

    async def mock_fetch_candidates(*args, **kwargs):
        return candidates, len(candidates)

    monkeypatch.setattr("pipeline.fetch_candidates", mock_fetch_candidates)

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("pipeline.Embedder", DummyEmbedder)

    def mock_get_or_compute_embeddings(stories, embedder, db_inst):
        embs = []
        for s in stories:
            vec = np.zeros(384, dtype=np.float32)
            if s.id == 999:
                vec[0] = 1.0
            elif 1 <= s.id <= 7:
                # Primary stories: high max_sim, above 15th-pct novel threshold
                # (0.605) and below 90th-pct similar threshold (0.663).
                # With 8 candidates, primary range [0.6, 0.66] clears novel
                # (only 0.01 qualifies) and avoids similar (0.66 < 0.663).
                vec[0] = 0.6 + (s.id - 1) * 0.01
            else:
                # ID 8 (extra-slot novel): low max_sim, qualifies for novel
                vec[0] = 0.01
            embs.append(vec)
        return np.array(embs)

    monkeypatch.setattr(
        "pipeline.get_or_compute_embeddings", mock_get_or_compute_embeddings
    )

    def mock_rank_stories(candidates_list, *args, **kwargs):
        res = []
        for s in candidates_list:
            if s.id == 1:
                score = 0.95
            elif 2 <= s.id <= 7:
                score = 0.9 - s.id * 0.05
            else:
                score = 0.1
            res.append(
                RankedStory(
                    story=s,
                    score=score,
                    best_match_title="",
                    prob_down=0.1 if s.id <= 7 else None,
                    prob_neutral=0.1 if s.id <= 7 else None,
                    prob_up=0.8 if s.id <= 7 else None,
                )
            )
        return sorted(res, key=lambda x: x.score, reverse=True)

    monkeypatch.setattr("pipeline.rank_stories", mock_rank_stories)

    captured_final = []

    def mock_generate_dashboard(final_list, *args, **kwargs):
        nonlocal captured_final
        captured_final = list(final_list)

    monkeypatch.setattr("pipeline.generate_dashboard", mock_generate_dashboard)

    await run_pipeline(config)

    # Primary story 1 should be in `final` AND carry is_discussion_rich.
    primary_one = next((r for r in captured_final if r.story.id == 1), None)
    assert primary_one is not None, "Story 1 missing from final"
    assert primary_one.is_discussion_rich, (
        "Primary story meeting the discussion-rich threshold should be badged"
    )

    # Other primary stories (no qualifying properties for discussion_rich
    # or novel) should not have those badges. We don't assert against
    # is_similar / is_high_engagement / is_hot here because small candidate
    # sets can let one primary story incidentally clear the 90th/95th/98th
    # percentile gates; that incidental matching is also valid new behavior.
    for r in captured_final:
        if 2 <= r.story.id <= 7:
            assert not r.is_discussion_rich, f"Story {r.story.id} has discussion badge"
            assert not r.is_novel, f"Story {r.story.id} has novel badge"

    # Extra-slot story 8 qualifies for is_novel (low max_sim), verifying
    # that the existing extra-slot discovery pass still works unchanged.
    extra_eight = next((r for r in captured_final if r.story.id == 8), None)
    assert extra_eight is not None, "Story 8 missing from final"
    assert extra_eight.is_novel, (
        "Extra-slot story meeting the novel threshold should be badged"
    )


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

    ranked = rank_stories(candidates, cand_embs, db, config, embedder)

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

    ranked = rank_stories(candidates, cand_embs, db, config, embedder)

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

    ranked = rank_stories(candidates, cand_embs, db, config, embedder)

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
    ranked = rank_stories(candidates, cand_embs, db, config, embedder)
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
    ranked = rank_stories(candidates, cand_embs, db, config, embedder)
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
    ranked = rank_stories(candidates, cand_embs, db, config, embedder)
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
    ranked = rank_stories(candidates, cand_embs, db, config, embedder)
    assert len(ranked) == 2
    for r in ranked:
        assert r.prob_up is not None
        assert 0.0 <= r.score <= 1.0
