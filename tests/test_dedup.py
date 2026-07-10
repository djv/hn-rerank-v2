from __future__ import annotations

import time
from dataclasses import replace
from typing import Literal, cast

import numpy as np
import pytest
from database import Database, FeedbackRecord, Story
from dedup import (
    DedupConfig,
    dedup_ranked,
    normalize_url,
)
from hypothesis import HealthCheck, given, settings, strategies as st
from pipeline import Config, Embedder, fast_rerank_for_user


@pytest.fixture
def db():
    db_instance = Database(":memory:")
    yield db_instance
    db_instance.close()


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------


def test_normalize_url_handles_basic_variants() -> None:
    assert normalize_url("https://example.com/path") == normalize_url(
        "http://example.com/path"
    )
    assert normalize_url("https://www.example.com/path") == normalize_url(
        "https://example.com/path"
    )
    assert normalize_url("https://example.com/path/") == normalize_url(
        "https://example.com/path"
    )
    assert normalize_url("https://EXAMPLE.com/Path") == normalize_url(
        "https://example.com/Path"
    )


def test_normalize_url_strips_trackers() -> None:
    bare = normalize_url("https://news.ycombinator.com/item?id=48680194")
    noisy = normalize_url(
        "https://news.ycombinator.com/item?id=48680194&utm_source=hn"
        "&fbclid=abc&ref_src=foo"
    )
    assert bare == noisy
    # Non-tracking query params are preserved (and sorted).
    kept = normalize_url("https://example.com/x?b=2&a=1")
    assert kept == "example.com/x?a=1&b=2"


def test_normalize_url_drops_fragment() -> None:
    assert normalize_url("https://example.com/a#section") == normalize_url(
        "https://example.com/a"
    )


def test_normalize_url_idempotent() -> None:
    samples = [
        "https://www.theverge.com/ai-artificial-intelligence/957372/"
        "openai-will-delay-gpt-5-6-after-trump-administration-request",
        "https://old.reddit.com/r/MachineLearning/comments/abc/x/",
        "https://example.com",
    ]
    for raw in samples:
        once = normalize_url(raw)
        twice = normalize_url(once)
        assert once == twice, f"idempotency failed for {raw!r}"


def test_normalize_url_unparseable() -> None:
    assert normalize_url(None) is None
    assert normalize_url("") is None
    assert normalize_url("   ") is None
    assert normalize_url("not a url") is None  # no netloc


def test_normalize_url_real_world_hn() -> None:
    url1 = normalize_url(
        "https://www.theverge.com/ai-artificial-intelligence/957372/"
        "openai-will-delay-gpt-5-6-after-trump-administration-request"
    )
    url2 = normalize_url(
        "http://theverge.com/ai-artificial-intelligence/957372/"
        "openai-will-delay-gpt-5-6-after-trump-administration-request?utm_source=hn"
    )
    assert url1 == url2


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.sampled_from(
        [
            "https://www.example.com/path",
            "https://Example.com/Path",
            "https://example.com/path/",
            "https://example.com/path?utm_source=hn",
        ]
    )
)
def test_normalize_url_property_idempotent_on_variants(raw: str) -> None:
    once = normalize_url(raw)
    twice = normalize_url(once)
    assert once == twice


# ---------------------------------------------------------------------------
# dedup_ranked — URL dedup
# ---------------------------------------------------------------------------


def _story(sid: int, *, url: str | None, source: str, score: int = 100) -> Story:
    return Story(
        id=sid,
        title=f"Story {sid}",
        url=url,
        score=score,
        time=int(time.time()),
        text_content=f"text {sid}",
        source=source,
    )


def _with_title(story: Story, title: str) -> Story:
    return replace(story, title=title)


def _with_text(story: Story, text_content: str) -> Story:
    return replace(story, text_content=text_content)


def test_dedup_ranked_disabled_returns_input_unchanged() -> None:
    stories = [
        _story(1, url="https://example.com/a", source="hn"),
        _story(2, url="https://example.com/a", source="rss_reddit_x"),
    ]
    cfg = DedupConfig(render_enabled=False)
    assert dedup_ranked(stories, [], cfg) == stories


def test_dedup_ranked_drops_duplicate_urls_normalized() -> None:
    """Same article across HN and Reddit, varying URL forms → keep one."""
    hn_story = _story(1, url="https://www.theverge.com/x", source="hn", score=50)
    reddit_story = _story(
        2, url="http://theverge.com/x?utm_source=hn", source="rss_reddit_x", score=10
    )
    out = dedup_ranked([hn_story, reddit_story], [], DedupConfig())
    assert len(out) == 1
    assert out[0].id == 1  # HN wins on source preference


def test_dedup_ranked_prefers_higher_score_within_same_source() -> None:
    a = _story(1, url="https://example.com/a", source="hn", score=10)
    b = _story(2, url="https://example.com/a", source="hn", score=500)
    out = dedup_ranked([a, b], [], DedupConfig())
    assert [s.id for s in out] == [2]


def test_dedup_ranked_keeps_null_url_stories_through() -> None:
    """Self-posts (url=None) are not deduped by URL."""
    a = _story(1, url=None, source="hn")
    b = _story(2, url=None, source="hn")
    out = dedup_ranked([a, b], [], DedupConfig())
    assert [s.id for s in out] == [1, 2]


def test_dedup_ranked_preserves_input_order_across_buckets() -> None:
    a = _story(1, url="https://example.com/a", source="hn", score=500)
    b = _story(2, url="https://example.com/b", source="hn", score=400)
    c = _story(3, url="https://example.com/a", source="rss_reddit_x", score=10)
    out = dedup_ranked([a, b, c], [], DedupConfig())
    # The dedup happens bucket by bucket. The order between surviving
    # stories is "first occurrence in input", so a, then b.
    assert [s.id for s in out] == [1, 2]


# ---------------------------------------------------------------------------
# dedup_ranked — feedback URL exclusion
# ---------------------------------------------------------------------------


def _fb(
    story_id: int,
    action: Literal["up", "neutral", "down"],
    url: str = "",
    title: str = "",
) -> FeedbackRecord:
    return FeedbackRecord(
        story_id=story_id,
        action=action,
        title=title,
        url=url or None,
        text_content="",
        source="hn",
        updated_at=time.time(),
    )


def test_dedup_ranked_excludes_upvoted_url() -> None:
    """Voting the HN version of an article drops the Reddit version too."""
    candidate = _story(2, url="https://www.theverge.com/x", source="rss_reddit_x")
    feedback = [_fb(1, "up", url="https://www.theverge.com/x", title="...")]
    out = dedup_ranked([candidate], feedback, DedupConfig())
    assert out == []


def test_dedup_ranked_excludes_neutral_voted_url() -> None:
    candidate = _story(2, url="https://www.theverge.com/x", source="rss_reddit_x")
    feedback = [_fb(1, "neutral", url="https://www.theverge.com/x", title="...")]
    out = dedup_ranked([candidate], feedback, DedupConfig())
    assert out == []


def test_dedup_ranked_does_not_exclude_downvoted_url() -> None:
    """Downvotes do not suppress alternate versions of the same article."""
    candidate = _story(2, url="https://www.theverge.com/x", source="rss_reddit_x")
    feedback = [_fb(1, "down", url="https://www.theverge.com/x", title="...")]
    out = dedup_ranked([candidate], feedback, DedupConfig())
    assert [s.id for s in out] == [2]


def test_dedup_ranked_url_exclusion_uses_normalization() -> None:
    candidate = _story(
        2, url="https://www.theverge.com/x?utm_source=hn", source="rss_reddit_x"
    )
    feedback = [_fb(1, "up", url="http://theverge.com/x", title="...")]
    out = dedup_ranked([candidate], feedback, DedupConfig())
    assert out == []


# ---------------------------------------------------------------------------
# dedup_ranked — embedding cosine dedup
# ---------------------------------------------------------------------------


def _unit_vec(bits: int) -> np.ndarray:
    """Return a unit-norm vector encoding *bits* (or zeros if bits==0)."""
    if bits == 0:
        return np.zeros(384, dtype=np.float32)
    vec = np.array([float(bits & (1 << i) and 1) for i in range(384)], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


def _emb_map(stories: list[Story], vec_fn) -> dict[int, np.ndarray]:
    """Build an id→embedding map from stories using *vec_fn(story.id)*."""
    return {s.id: vec_fn(s.id) for s in stories}


def test_embedding_cosine_merges_near_identical() -> None:
    """Stories with identical-content embeddings are deduped."""
    # Same vector = cosine 1.0, well above 0.87 threshold
    a = _story(1, url="https://a.com/x", source="hn", score=50)
    b = _story(2, url="https://b.com/x", source="rss_slashdot_org", score=10)
    v = _unit_vec(1)
    embeddings = {1: v, 2: v}
    out = dedup_ranked([a, b], [], DedupConfig(), embeddings=embeddings)
    assert len(out) == 1
    assert out[0].id == 1  # HN wins source preference


def test_embedding_cosine_keeps_different_content() -> None:
    """Stories with different embeddings (low cosine sim) are not merged."""
    a = _story(1, url="https://a.com/x", source="hn")
    b = _story(2, url="https://b.com/y", source="rss_slashdot_org")
    # Orthogonal vectors → cos ~ 0.0, below 0.87
    embeddings = {1: _unit_vec(1), 2: _unit_vec(2)}
    out = dedup_ranked([a, b], [], DedupConfig(), embeddings=embeddings)
    assert [s.id for s in out] == [1, 2]


def test_embedding_cosine_below_threshold_not_merged() -> None:
    """Stories whose cosine sim is below the threshold are kept separate."""
    a = _story(1, url="https://a.com/x", source="hn")
    b = _story(2, url="https://b.com/y", source="rss_slashdot_org")
    # Build two vectors with controlled cosine sim ~0.80 (< 0.87)
    # v1 = [1, 0, 0, ...], v2 = [0.8, 0.6, 0, ...] → dot = 0.8
    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(384, dtype=np.float32)
    v2[0] = 0.8
    v2[1] = 0.6
    cos_sim = float(np.dot(v1, v2))
    assert 0.75 < cos_sim < 0.87, f"cosine sim {cos_sim:.4f} should be in (0.75, 0.87)"
    embeddings = {1: v1, 2: v2}
    out = dedup_ranked([a, b], [], DedupConfig(), embeddings=embeddings)
    assert [s.id for s in out] == [1, 2]


def test_embedding_cosine_disabled() -> None:
    """When embedding cosine is disabled, different-url stories pass through."""
    a = _story(1, url="https://a.com/x", source="hn")
    b = _story(2, url="https://b.com/x", source="rss_slashdot_org")
    v = _unit_vec(1)
    embeddings = {1: v, 2: v}
    cfg = DedupConfig(embedding_cosine_enabled=False)
    out = dedup_ranked([a, b], [], cfg, embeddings=embeddings)
    assert [s.id for s in out] == [1, 2]


def test_embedding_cosine_source_preference_tiebreak() -> None:
    """When a slashdot story appears first, the later HN story wins the swap."""
    a = _story(1, url="https://slashdot.org/story", source="rss_slashdot_org", score=50)
    b = _story(2, url="https://arstechnica.com/article", source="hn", score=100)
    v = _unit_vec(1)
    embeddings = {1: v, 2: v}
    out = dedup_ranked([a, b], [], DedupConfig(), embeddings=embeddings)
    assert len(out) == 1
    assert out[0].id == 2  # HN wins even though slashdot was first


def test_embedding_cosine_no_embedding_for_story() -> None:
    """Stories missing from the embedding map are kept (not deduped)."""
    a = _story(1, url="https://a.com/x", source="hn")
    b = _story(2, url="https://b.com/x", source="rss_slashdot_org")
    embeddings = {1: _unit_vec(1)}  # b has no embedding
    out = dedup_ranked([a, b], [], DedupConfig(), embeddings=embeddings)
    assert [s.id for s in out] == [1, 2]


# ---------------------------------------------------------------------------
# Integration: fast_rerank_for_user wires dedup
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> Embedder:
    return Embedder()


def test_fast_rerank_for_user_dedups_duplicate_urls(db, monkeypatch) -> None:
    """End-to-end: two stories with the same normalized URL → one card."""
    import numpy as np

    user = db.create_user("dedup_e2e")
    now = int(time.time())
    hn_id = 77001
    reddit_id = -77002
    hn = Story(
        id=hn_id,
        title="OpenAI delays GPT-5.6",
        url="https://www.theverge.com/x",
        score=200,
        time=now - 3600,
        text_content="hn text",
        source="hn",
        comment_count=5,
    )
    reddit = Story(
        id=reddit_id,
        title="OpenAI delays GPT-5.6",
        url="http://theverge.com/x?utm_source=hn",
        score=10,
        time=now - 3600,
        text_content="reddit text",
        source="rss_reddit_x",
        comment_count=2,
    )
    db.upsert_story(hn)
    db.upsert_story(reddit)

    monkeypatch.setattr(
        "pipeline.get_or_compute_embeddings",
        lambda stories, embedder, db_inst: np.zeros(
            (len(stories), 384), dtype=np.float32
        ),
    )
    config = Config(days=30)
    ranked = fast_rerank_for_user(db, config, cast(Embedder, object()), user.id)
    survivor_ids = {r.story.id for r in ranked}
    assert hn_id in survivor_ids
    assert reddit_id not in survivor_ids


def test_fast_rerank_for_user_excludes_upvoted_duplicate(db, monkeypatch) -> None:
    """Voting the HN version removes the Reddit version from future decks."""
    import numpy as np

    user = db.create_user("dedup_fb_e2e")
    now = int(time.time())
    hn_id = 77010
    reddit_id = -77011
    hn = Story(
        id=hn_id,
        title="A duplicate article",
        url="https://example.com/duplicate",
        score=200,
        time=now - 3600,
        text_content="hn text",
        source="hn",
        comment_count=5,
    )
    reddit = Story(
        id=reddit_id,
        title="A duplicate article (reddit mirror)",
        url="https://example.com/duplicate",
        score=10,
        time=now - 3600,
        text_content="reddit text",
        source="rss_reddit_x",
        comment_count=2,
    )
    db.upsert_story(hn)
    db.upsert_story(reddit)
    db.upsert_feedback(user.id, hn_id, "up")

    monkeypatch.setattr(
        "pipeline.get_or_compute_embeddings",
        lambda stories, embedder, db_inst: np.zeros(
            (len(stories), 384), dtype=np.float32
        ),
    )
    config = Config(days=30)
    ranked = fast_rerank_for_user(db, config, cast(Embedder, object()), user.id)
    survivor_ids = {r.story.id for r in ranked}
    # The HN story is excluded (already in feedback), and the Reddit story
    # is excluded by URL-match against the upvoted feedback.
    assert hn_id not in survivor_ids
    assert reddit_id not in survivor_ids


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _make_story(story_id: int, source: str, url: str, title: str) -> Story:
    now = int(time.time())
    return Story(
        id=story_id,
        title=title,
        url=url,
        score=100,
        time=now,
        text_content="",
        source=source,
    )


def test_dedup_ranked_emits_info_summary(caplog) -> None:
    """Every dedup pass emits a single INFO summary with key=value fields."""
    stories = [
        _make_story(1, "hn", "https://example.com/a", "Story A"),
        _make_story(2, "hn", "https://example.com/a", "Story A duplicate"),
        _make_story(3, "ch_seed", "https://example.com/b", "Story B"),
    ]
    with caplog.at_level("INFO", logger="dedup"):
        dedup_ranked(stories, [], DedupConfig(), user_id=42)

    summaries = [r for r in caplog.records if r.message.startswith("dedup user_id=")]
    assert len(summaries) == 1
    msg = summaries[0].message
    assert "user_id=42" in msg
    assert "in=3" in msg
    assert "out=2" in msg
    assert "suppressed=1" in msg
    assert "url_dups=1" in msg
    assert "fb_url=0" in msg
    assert "embedding=off" in msg


def test_dedup_ranked_emits_debug_per_suppression(caplog) -> None:
    """Each suppressed story produces a DEBUG line with reason and ids."""
    hn = _make_story(1, "hn", "https://example.com/x", "X")
    reddit = _make_story(2, "rss_reddit_", "https://example.com/x", "X on Reddit")
    with caplog.at_level("DEBUG", logger="dedup"):
        dedup_ranked([hn, reddit], [], DedupConfig(), user_id=7)

    suppression_lines = [
        r for r in caplog.records if r.message.startswith("dedup-suppress ")
    ]
    assert len(suppression_lines) == 1
    msg = suppression_lines[0].message
    assert "reason=url_dup" in msg
    assert "dropped_id=2" in msg  # reddit lost
    assert "dropped_source=rss_reddit_" in msg
    assert "kept_id=1" in msg
    assert "kept_source=hn" in msg


def test_dedup_ranked_logs_fb_url_suppression(caplog) -> None:
    """Feedback-URL exclusion is logged with reason=fb_url and fb_url= field."""
    story = _make_story(1, "ch_seed", "https://example.com/feedbacked", "X")
    feedback = [
        FeedbackRecord(
            story_id=99,
            action="up",
            title="X",
            url="https://example.com/feedbacked",
            text_content="",
            source="hn",
            updated_at=time.time(),
        )
    ]
    with caplog.at_level("DEBUG", logger="dedup"):
        dedup_ranked([story], feedback, DedupConfig(), user_id=11)

    lines = [r for r in caplog.records if r.message.startswith("dedup-suppress ")]
    assert any("reason=fb_url" in r.message for r in lines)
    summary = [r for r in caplog.records if r.message.startswith("dedup user_id=")][0]
    assert "fb_url=1" in summary.message


def test_dedup_ranked_logs_embedding_cosine_when_enabled(caplog) -> None:
    """Embedding-cosine suppressions get reason=embedding_cosine DEBUG lines."""
    a = _make_story(1, "hn", "https://a.com/p", "South Korea chips")
    b = _make_story(
        2, "rss_slashdot_org", "https://slashdot.org/story", "South Korea To Spend $1T"
    )
    # Same embedding → cosine 1.0, above 0.87 threshold
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)  # unit norm
    embeddings = {1: v, 2: v}
    with caplog.at_level("DEBUG", logger="dedup"):
        dedup_ranked(
            [a, b],
            [],
            DedupConfig(),
            user_id=3,
            embeddings=embeddings,
        )
    summary = [r for r in caplog.records if r.message.startswith("dedup user_id=")][0]
    assert "embedding=on" in summary.message
    assert "embedding_dups=1" in summary.message
    cosine_lines = [
        r
        for r in caplog.records
        if r.message.startswith("dedup-suppress ")
        and "reason=embedding_cosine" in r.message
    ]
    assert len(cosine_lines) == 1
    assert "dropped_source=rss_slashdot_org" in cosine_lines[0].message
    assert "kept_source=hn" in cosine_lines[0].message
    assert "cosine_sim=" in cosine_lines[0].message


def test_dedup_ranked_no_suppression_still_logs_summary(caplog) -> None:
    """A clean pass still emits the INFO summary line so we know dedup ran."""
    stories = [
        _make_story(1, "hn", "https://example.com/a", "A"),
        _make_story(2, "ch_seed", "https://example.com/b", "B"),
    ]
    with caplog.at_level("INFO", logger="dedup"):
        dedup_ranked(stories, [], DedupConfig(), user_id=1)
    summary = [r for r in caplog.records if r.message.startswith("dedup user_id=")][0]
    assert "suppressed=0" in summary.message
    assert "url_dups=0" in summary.message


def test_dedup_ranked_summary_includes_user_id_when_provided(caplog) -> None:
    """The user_id kwarg is rendered in the summary line for multi-user logs."""
    stories = [_make_story(1, "hn", "https://example.com/a", "A")]
    with caplog.at_level("INFO", logger="dedup"):
        dedup_ranked(stories, [], DedupConfig(), user_id=123)
    summary = [r for r in caplog.records if r.message.startswith("dedup user_id=")][0]
    assert "user_id=123" in summary.message


def test_dedup_ranked_summary_uses_placeholder_when_user_id_omitted(caplog) -> None:
    """If user_id is None (test/script callers), summary shows user_id=?"""
    stories = [_make_story(1, "hn", "https://example.com/a", "A")]
    with caplog.at_level("INFO", logger="dedup"):
        dedup_ranked(stories, [], DedupConfig())
    summary = [r for r in caplog.records if r.message.startswith("dedup user_id=")][0]
    assert "user_id=?" in summary.message
