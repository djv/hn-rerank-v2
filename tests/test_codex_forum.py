"""Codex Forum source-content fixes: footer removal, article suppression,
legacy effective-view normalization, versioned summary keys.

Offline fixtures only; network, providers, and embedders are stubbed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import server
from database import Database, Story
from pipeline.codex_forum import (
    CODEX_FORUM_SOURCE,
    CODEX_SUMMARY_INPUT_VERSION,
    effective_codex_story,
    is_codex_source,
    strip_discourse_footer_html,
    strip_discourse_footer_text,
)
from pipeline.ranking import Embedder

NOW = 2_000_000_000.0
CODEX_URL = "https://community.openai.com/t/speed-up-codex-runs/1"
GENERIC_SOURCE = "rss_aeon_co"


@pytest.fixture
def db() -> Iterator[Database]:
    db_instance = Database(":memory:")
    yield db_instance
    db_instance.close()


def _codex_story(sid: int = 101, **overrides: Any) -> Story:
    fields: dict[str, Any] = {
        "id": sid,
        "title": "Speed up codex runs",
        "url": CODEX_URL,
        "score": 0,
        "time": int(NOW - 86400),
        "text_content": "Speed up codex runs. How do I speed up my codex runs?",
        "source": CODEX_FORUM_SOURCE,
        "self_text": "How do I speed up my codex runs?",
    }
    fields.update(overrides)
    return Story(**fields)  # type: ignore[arg-type]


def _legacy_codex_story(sid: int = 102) -> Story:
    """A stored row from before the policy: mixed body, flattened footer."""
    return _codex_story(
        sid,
        self_text="Opening post text 1 post - 1 participant Read full topic",
        top_comments="A real comment.",
        article_body="REPLY TEXT FROM NON-OPENING POSTS plus boilerplate",
        text_content=(
            "Speed up codex runs. Opening post text 1 post - 1 participant "
            "Read full topic REPLY TEXT FROM NON-OPENING POSTS plus boilerplate "
            "A real comment."
        ),
    )


# Feed-footer removal (raw HTML, before plain-text conversion).

SPLIT_FOOTER = (
    "<p>How do I speed up my codex runs?</p>"
    "<p><small>3 posts - 2 participants</small></p>"
    '<p><a href="https://community.openai.com/t/x/1">Read full topic</a></p>'
)
COMBINED_FOOTER = (
    "<p>How do I speed up my codex runs?</p>"
    '<p><small>12 posts - 7 participants</small> <a href="https://community.openai.com/t/x/1">Read full topic</a></p>'
)


def test_html_split_footer_removed_opening_post_kept() -> None:
    assert strip_discourse_footer_html(SPLIT_FOOTER) == (
        "<p>How do I speed up my codex runs?</p>"
    )


def test_html_combined_footer_removed() -> None:
    assert strip_discourse_footer_html(COMBINED_FOOTER) == (
        "<p>How do I speed up my codex runs?</p>"
    )


@pytest.mark.parametrize(
    "raw",
    [
        # Bare count without the link: authored lookalike, preserved.
        "<p>Real post</p><p><small>3 posts - 2 participants</small></p>",
        # Bare link without the count: preserved.
        '<p>Real post</p><p><a href="https://example.com/t/1">Read full topic</a></p>',
        # Complete footer mid-post followed by real content: preserved.
        SPLIT_FOOTER + "<p>More real content after.</p>",
        # Unrelated trailing link: preserved.
        '<p>Real post</p><p><a href="https://example.com">Continue reading</a></p>',
        # Authored small text that is not a count: preserved.
        "<p>Real post <small>edited for clarity</small></p>",
        "plain text without any markup",
        "",
    ],
)
def test_html_footer_lookalikes_preserved(raw: str) -> None:
    assert strip_discourse_footer_html(raw) == raw


def test_html_footer_strip_idempotent() -> None:
    once = strip_discourse_footer_html(SPLIT_FOOTER)
    assert strip_discourse_footer_html(once) == once


# Flattened-footer removal (legacy plain text).


def test_text_footer_singular_and_plural_removed() -> None:
    assert (
        strip_discourse_footer_text(
            "Opening post 1 post - 1 participant Read full topic"
        )
        == "Opening post"
    )
    assert (
        strip_discourse_footer_text(
            "Opening post 12 posts - 7 participants Read full topic"
        )
        == "Opening post"
    )


@pytest.mark.parametrize(
    "plain",
    [
        "Opening post 3 posts - 2 participants",
        "Opening post Read full topic",
        "3 posts - 2 participants Read full topic, plus real trailing text",
        "A post about 3 posts - 2 giraffes Read full topic".replace(
            "giraffes", "people"
        ),
        "",
    ],
)
def test_text_footer_lookalikes_preserved(plain: str) -> None:
    assert strip_discourse_footer_text(plain) == plain


def test_text_footer_strip_idempotent() -> None:
    once = strip_discourse_footer_text(
        "Opening post 3 posts - 2 participants Read full topic"
    )
    assert strip_discourse_footer_text(once) == once


# Effective-view normalization for stored rows.


def test_effective_view_cleans_legacy_row_and_preserves_raw() -> None:
    from pipeline import compose_story_text

    legacy = _legacy_codex_story()
    effective = effective_codex_story(legacy)
    assert effective.source == CODEX_FORUM_SOURCE
    assert effective.title == legacy.title
    assert effective.self_text == "Opening post text"
    assert effective.article_body == ""
    assert effective.top_comments == "A real comment."
    assert "REPLY TEXT" not in effective.text_content
    assert "participant" not in effective.text_content
    assert effective.text_content == compose_story_text(
        legacy.title, "Opening post text", "A real comment.", ""
    )
    # Raw stored object untouched: votes, identity, and fields preserved.
    assert legacy.article_body.startswith("REPLY TEXT")
    assert "participant" in legacy.self_text
    assert legacy.id == effective.id and legacy.url == effective.url


def test_effective_view_idempotent() -> None:
    effective = effective_codex_story(_legacy_codex_story())
    assert effective_codex_story(effective) == effective


def test_effective_view_clean_row_unchanged() -> None:
    story = _codex_story()
    assert effective_codex_story(story) is story


def test_effective_view_non_codex_passthrough() -> None:
    story = _codex_story(103, source=GENERIC_SOURCE)
    assert effective_codex_story(story) is story
    assert not is_codex_source(GENERIC_SOURCE)
    assert is_codex_source(CODEX_FORUM_SOURCE)


def test_upsert_longest_merge_keeps_raw_but_effective_read_is_clean(
    db: Database,
) -> None:
    """The longest-text merge preserves raw dirty fields; reads normalize."""
    db.upsert_story(_legacy_codex_story())
    clean_short = _codex_story(102)
    db.upsert_story(clean_short)
    stored = db.get_story(102)
    assert stored is not None
    # Raw preservation: the longer dirty body survived the merge.
    assert stored.article_body.startswith("REPLY TEXT")
    effective = effective_codex_story(stored)
    assert effective.article_body == ""
    assert "REPLY TEXT" not in effective.text_content
    assert "participant" not in effective.text_content


# Article-lane suppression (all three selectors agree).


def test_article_gates_exclude_codex_but_keep_generic_control(
    db: Database,
) -> None:
    from pipeline import RankedStory
    from pipeline.enrichment import (
        _NO_ARTICLE_SOURCES,
        _article_fetch_eligible,
        select_article_fetch_candidates,
        select_rss_article_prewarm,
    )

    assert CODEX_FORUM_SOURCE in _NO_ARTICLE_SOURCES
    # Fetchable URLs isolate the source policy from the paywall/host block.
    codex = _codex_story(url="https://example.com/codex-topic")
    control = _codex_story(103, source=GENERIC_SOURCE, url="https://example.com/essay")
    for row in (codex, control):
        db.upsert_story(row)
    min_time = NOW - 30 * 86400
    assert not _article_fetch_eligible(codex, db, min_time, NOW)
    assert _article_fetch_eligible(control, db, min_time, NOW)
    assert select_rss_article_prewarm(
        [codex, control], db, max_per_run=10, now_ts=NOW
    ) == [
        s for s in select_rss_article_prewarm([control], db, max_per_run=10, now_ts=NOW)
    ]
    assert [
        s.id
        for s in select_rss_article_prewarm([codex], db, max_per_run=10, now_ts=NOW)
    ] == []
    ranked = [
        RankedStory(story=codex, score=50.0, best_match_title=""),
        RankedStory(story=control, score=1.0, best_match_title=""),
    ]
    picked = select_article_fetch_candidates(
        ranked=ranked, dashboard_selected=ranked, db=db, max_per_run=10
    )
    assert [s.id for s in picked] == [103]


# RSS parse path: footer stripped for Codex only.


def _feed_xml(description_html: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item>
  <title>Speed up codex runs</title>
  <link>{CODEX_URL}</link>
  <pubDate>Tue, 23 Jun 2026 12:00:00 GMT</pubDate>
  <description><![CDATA[{description_html}]]></description>
</item>
</channel></rss>"""


async def _parse_feed(
    monkeypatch: pytest.MonkeyPatch, feed_url: str, xml: str
) -> list[Story]:
    from pipeline import _fetch_and_parse_feed

    class MockResp:
        status_code = 200
        text = xml
        headers: dict[str, str] = {}

    class MockClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> MockClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def get(self, url: str, headers: Any = None) -> MockResp:
            return MockResp()

    monkeypatch.setattr("pipeline.enrichment.httpx.AsyncClient", MockClient)
    return await _fetch_and_parse_feed(
        feed_url, per_feed=10, cutoff=0, now=1_000_000, exclude_urls=set()
    )


@pytest.mark.asyncio
async def test_parse_strips_codex_footer(monkeypatch: pytest.MonkeyPatch) -> None:
    stories = await _parse_feed(
        monkeypatch,
        "https://community.openai.com/c/codex/37.rss",
        _feed_xml(SPLIT_FOOTER),
    )
    assert len(stories) == 1
    assert stories[0].source == CODEX_FORUM_SOURCE
    assert "participants" not in stories[0].self_text
    assert "Read full topic" not in stories[0].self_text
    assert "How do I speed up my codex runs?" in stories[0].self_text


@pytest.mark.asyncio
async def test_parse_keeps_generic_feed_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stories = await _parse_feed(
        monkeypatch, "https://example.com/feed.xml", _feed_xml(SPLIT_FOOTER)
    )
    assert len(stories) == 1
    assert stories[0].source == "rss_example_com"
    # Out of scope: other feeds keep their existing behavior.
    assert "participants" in stories[0].self_text


# Embedding inputs and hashes follow the effective view.


class _FakeEmbedder(Embedder):
    model_version = "fake-v1"
    model_onnx_sha = "fake-sha"
    embedding_dim = 4
    captured: list[str]

    def __init__(self) -> None:
        # Skip the real tokenizer/ONNX constructor; tests capture text only.
        self.captured = []

    def encode(
        self, texts: list[str], batch_size: int | None = None
    ) -> NDArray[np.float32]:
        self.captured.extend(texts)
        return np.array([[float(len(t))] * 4 for t in texts], dtype=np.float32)


def test_embedding_text_uses_effective_view() -> None:
    from pipeline import compose_story_text, story_embedding_text

    legacy = _legacy_codex_story()
    assert "REPLY TEXT" in legacy.text_content
    assert story_embedding_text(legacy) == compose_story_text(
        legacy.title, "Opening post text", "A real comment.", ""
    )
    assert story_embedding_text(_codex_story()) == _codex_story().text_content


def test_embedding_hash_changes_with_clean_inputs(db: Database) -> None:
    from pipeline import get_or_compute_embeddings, story_embedding_text

    legacy = _legacy_codex_story()
    db.upsert_story(legacy)
    dirty_hash = hashlib.sha256(legacy.text_content.encode("utf-8")).hexdigest()
    embedder = _FakeEmbedder()
    vecs = get_or_compute_embeddings([legacy], embedder, db)
    assert len(vecs) == 1
    assert len(embedder.captured) == 1
    assert "REPLY TEXT" not in embedder.captured[0]
    assert "participant" not in embedder.captured[0]
    clean_hash = hashlib.sha256(embedder.captured[0].encode("utf-8")).hexdigest()
    assert clean_hash != dirty_hash
    assert (
        clean_hash
        == hashlib.sha256(story_embedding_text(legacy).encode("utf-8")).hexdigest()
    )


# Summary keys: versioned for Codex, unchanged otherwise.


def test_codex_key_rejects_legacy_mixed_inputs() -> None:
    legacy = _legacy_codex_story()
    legacy_key = server._tldr_cache_key(
        title=legacy.title,
        self_text=legacy.self_text,
        top_comments=legacy.top_comments,
        article_body=legacy.article_body,
    )
    effective = effective_codex_story(legacy)
    current_key = server._tldr_cache_key(
        title=effective.title,
        self_text=effective.self_text or "",
        top_comments=effective.top_comments or "",
        article_body=effective.article_body or "",
        source=effective.source,
    )
    assert current_key != legacy_key
    assert CODEX_SUMMARY_INPUT_VERSION  # marker wired into the key path
    again = server._tldr_cache_key(
        title=effective.title,
        self_text=effective.self_text or "",
        top_comments=effective.top_comments or "",
        article_body=effective.article_body or "",
        source=effective.source,
    )
    assert again == current_key


def test_non_codex_key_unchanged() -> None:
    kwargs: dict[str, str] = {
        "title": "T",
        "self_text": "S",
        "top_comments": "C",
        "article_body": "A",
    }
    assert server._tldr_cache_key(**kwargs) == server._tldr_cache_key(
        **kwargs, source=GENERIC_SOURCE
    )


def test_stale_fallback_rejects_legacy_codex_key(db: Database) -> None:
    legacy = _legacy_codex_story()
    db.upsert_story(legacy)
    legacy_key = server._tldr_cache_key(
        title=legacy.title,
        self_text=legacy.self_text,
        top_comments=legacy.top_comments,
        article_body=legacy.article_body,
    )
    db.upsert_tldr_cache(legacy.id, legacy_key, "legacy mixed summary")
    assert server._stale_tldr_fallback(db, legacy.id, "test") is None
    # A summary cached under the current normalized inputs still serves.
    effective = effective_codex_story(legacy)
    current_key = server._tldr_cache_key(
        title=effective.title,
        self_text=effective.self_text or "",
        top_comments=effective.top_comments or "",
        article_body=effective.article_body or "",
        source=effective.source,
    )
    db.upsert_tldr_cache(legacy.id, current_key, "clean summary")
    fallback = server._stale_tldr_fallback(db, legacy.id, "test")
    assert fallback is not None


def test_stale_fallback_unchanged_for_generic_source(db: Database) -> None:
    story = _codex_story(103, source=GENERIC_SOURCE, article_body="Cached body.")
    db.upsert_story(story)
    key = server._tldr_cache_key(
        title=story.title,
        self_text=story.self_text,
        top_comments="",
        article_body=story.article_body,
    )
    db.upsert_tldr_cache(story.id, key, "generic summary")
    assert server._stale_tldr_fallback(db, story.id, "test") is not None


# Candidate reads serve the effective view.


def test_candidate_pool_read_is_effective(db: Database) -> None:
    from pipeline import Config, load_production_candidate_stories
    from pipeline.config import RssConfig

    db.upsert_story(_legacy_codex_story())
    config = Config(
        days=30,
        rss=RssConfig(feeds=("https://community.openai.com/c/codex/37.rss",)),
    )
    stories = load_production_candidate_stories(
        db, config, user_id=None, exclude_feedback=False, now_ts=int(NOW)
    )
    by_id = {s.id: s for s in stories}
    assert 102 in by_id
    assert by_id[102].article_body == ""
    assert "REPLY TEXT" not in by_id[102].text_content
    assert "participant" not in by_id[102].text_content
    # Raw row preserved in the DB.
    stored = db.get_story(102)
    assert stored is not None and stored.article_body.startswith("REPLY TEXT")


def test_feedback_reads_are_effective(db: Database) -> None:
    from pipeline import get_or_compute_embeddings, story_embedding_text

    db.upsert_story(_legacy_codex_story())
    user = db.create_user("codex-test")
    db.upsert_feedback(user.id, 102, "up")
    fb_stories, labels, _ = db.get_feedback_for_training(user_id=user.id)
    assert labels == [2]
    effective = [effective_codex_story(s) for s in fb_stories]
    assert effective[0].article_body == ""
    assert "REPLY TEXT" not in story_embedding_text(effective[0])
    dedup_stories = db.get_feedback_stories(user.id, ("up",))
    assert (
        dedup_stories
        and dedup_stories[0].self_text
        == "Opening post text 1 post - 1 participant Read full topic"
    )
    embedder = _FakeEmbedder()
    get_or_compute_embeddings(
        [effective_codex_story(s) for s in dedup_stories],
        embedder,
        db,
    )
    assert embedder.captured and "REPLY TEXT" not in embedder.captured[0]


def test_idempotent_double_application(db: Database) -> None:
    db.upsert_story(_legacy_codex_story())
    stored = db.get_story(102)
    assert stored is not None
    once = effective_codex_story(stored)
    assert effective_codex_story(replace(once)) == once
