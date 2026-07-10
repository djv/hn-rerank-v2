from __future__ import annotations

from collections.abc import Mapping

from database import Story
from pipeline.hn_dupes import (
    HnDupeResolver,
    extract_hn_dupe_target_id,
    extract_hn_story_link_ids,
    story_from_firebase_item,
)


FirebaseItem = Mapping[str, object]


def test_extract_hn_story_link_ids_plain_link() -> None:
    assert (
        extract_hn_story_link_ids(
            "Earlier: https://news.ycombinator.com/item?id=123",
            source_id=99,
        )
        == [123]
    )


def test_extract_hn_story_link_ids_multiple_links() -> None:
    assert extract_hn_story_link_ids(
        "Older https://news.ycombinator.com/item?id=456 and "
        "newer https://news.ycombinator.com/item?id=789",
        source_id=99,
    ) == [456, 789]


def test_extract_hn_story_link_ids_html_escaped_comment_text() -> None:
    text = (
        "&lt;a href=&quot;https://news.ycombinator.com/item?id=789&quot;&gt;"
        "older discussion&lt;/a&gt;"
    )
    assert extract_hn_story_link_ids(text, source_id=99) == [789]


def test_extract_hn_story_link_ids_relative_href_and_self_links() -> None:
    assert extract_hn_story_link_ids(
        '<a href="/item?id=123">thread</a> <a href="/item?id=99">self</a>',
        source_id=99,
    ) == [123]


def test_extract_hn_dupe_target_compatibility_wrapper() -> None:
    assert (
        extract_hn_dupe_target_id(
            "Discussion: https://news.ycombinator.com/item?id=456",
            source_id=99,
        )
        == 456
    )


def test_resolver_finds_direct_child_hn_link_and_validates_target() -> None:
    calls: list[int] = []
    items: dict[int, FirebaseItem] = {
        10: {
            "id": 10,
            "type": "story",
            "title": "Claude Fable extended to July 12",
            "score": 10,
            "descendants": 2,
            "kids": [11, 12],
        },
        11: {"id": 11, "type": "comment", "text": "ordinary"},
        12: {
            "id": 12,
            "type": "comment",
            "text": "Earlier: https://news.ycombinator.com/item?id=20",
        },
        20: {
            "id": 20,
            "type": "story",
            "title": "We're extending access to Fable 5 on all paid plans through July 12",
            "score": 50,
            "time": 100,
            "descendants": 3,
        },
    }

    def fetch_item(story_id: int) -> FirebaseItem | None:
        calls.append(story_id)
        return items.get(story_id)

    resolver = HnDupeResolver(fetch_item=fetch_item)

    assert resolver.find_canonical_story_id(10) == 20
    assert calls == [10, 11, 12, 20]


def test_resolver_ignores_deleted_comments_and_targets() -> None:
    items: dict[int, FirebaseItem] = {
        10: {
            "id": 10,
            "type": "story",
            "title": "Similar Story",
            "score": 10,
            "descendants": 2,
            "kids": [11, 12],
        },
        11: {
            "id": 11,
            "type": "comment",
            "deleted": True,
            "text": "https://news.ycombinator.com/item?id=20",
        },
        12: {
            "id": 12,
            "type": "comment",
            "text": "https://news.ycombinator.com/item?id=21",
        },
        21: {"id": 21, "type": "story", "deleted": True, "title": "Gone"},
    }
    resolver = HnDupeResolver(fetch_item=items.get)

    assert resolver.find_canonical_story_id(10) is None


def test_resolver_honors_bounded_kid_fetches_and_cache_hits() -> None:
    calls: list[int] = []
    items: dict[int, FirebaseItem] = {
        10: {
            "id": 10,
            "type": "story",
            "title": "Canonical Story",
            "score": 10,
            "descendants": 3,
            "kids": [11, 12, 13],
        },
        11: {"id": 11, "type": "comment", "text": "ordinary"},
        12: {"id": 12, "type": "comment", "text": "ordinary"},
        13: {
            "id": 13,
            "type": "comment",
            "text": "https://news.ycombinator.com/item?id=20",
        },
        20: {
            "id": 20,
            "type": "story",
            "title": "Canonical Story",
            "score": 50,
            "descendants": 10,
        },
    }

    def fetch_item(story_id: int) -> FirebaseItem | None:
        calls.append(story_id)
        return items.get(story_id)

    resolver = HnDupeResolver(fetch_item=fetch_item, max_kids=2)

    assert resolver.find_canonical_story_id(10) is None
    assert resolver.find_canonical_story_id(10) is None
    assert calls == [10, 11, 12]


def test_resolver_rejects_dissimilar_title() -> None:
    items: dict[int, FirebaseItem] = {
        10: {
            "id": 10,
            "type": "story",
            "title": "Claude Fable extended to July 12",
            "score": 10,
            "descendants": 1,
            "kids": [11],
        },
        11: {
            "id": 11,
            "type": "comment",
            "text": "Related: https://news.ycombinator.com/item?id=20",
        },
        20: {
            "id": 20,
            "type": "story",
            "title": "Completely unrelated database benchmark",
            "score": 100,
            "descendants": 20,
        },
    }
    resolver = HnDupeResolver(fetch_item=items.get)

    assert resolver.find_canonical_story_id(10) is None


def test_resolver_rejects_weaker_target() -> None:
    items: dict[int, FirebaseItem] = {
        10: {
            "id": 10,
            "type": "story",
            "title": "Rewriting Bun in Rust",
            "score": 100,
            "descendants": 5,
            "kids": [11],
        },
        11: {
            "id": 11,
            "type": "comment",
            "text": "Earlier: https://news.ycombinator.com/item?id=20",
        },
        20: {
            "id": 20,
            "type": "story",
            "title": "Rewriting Bun in Rust",
            "score": 10,
            "descendants": 1,
        },
    }
    resolver = HnDupeResolver(fetch_item=items.get)

    assert resolver.find_canonical_story_id(10) is None


def test_resolver_network_failure_returns_no_target() -> None:
    def fetch_item(_story_id: int) -> FirebaseItem | None:
        raise TimeoutError("boom")

    resolver = HnDupeResolver(fetch_item=fetch_item)

    assert resolver.find_canonical_story_id(10) is None


def test_story_from_firebase_item_normalizes_valid_story() -> None:
    story = story_from_firebase_item(
        {
            "id": 20,
            "type": "story",
            "title": "Canonical &amp; Good",
            "url": "https://example.com",
            "score": 100,
            "time": 1234,
            "descendants": 5,
            "text": "Self &lt;b&gt;text&lt;/b&gt;",
        }
    )

    assert isinstance(story, Story)
    assert story.id == 20
    assert story.title == "Canonical & Good"
    assert story.self_text == "Self text"
    assert story.comment_count == 5
