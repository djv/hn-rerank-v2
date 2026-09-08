from __future__ import annotations

from collections.abc import Mapping
from difflib import SequenceMatcher

from hypothesis import HealthCheck, given, settings, strategies as st

from database import Database, Story
from pipeline.hn_dupes import (
    HnDupeResolver,
    _informative_title_tokens,
    _load_feedback_context,
    _matches_feedback,
    _normalize_title,
    _ratio_length_plausible,
    _story_titles_are_similar,
    extract_hn_dupe_target_id,
    extract_hn_story_link_ids,
    story_from_firebase_item,
)


FirebaseItem = Mapping[str, object]


def test_extract_hn_story_link_ids_plain_link() -> None:
    assert extract_hn_story_link_ids(
        "Earlier: https://news.ycombinator.com/item?id=123",
        source_id=99,
    ) == [123]


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


def _naive_titles_are_similar(source_title: str, target_title: str) -> bool:
    """Pre-optimization reference: normalize, full difflib ratio, token gate."""
    from pipeline.hn_dupes import (
        MIN_SHARED_TITLE_TOKENS,
        TITLE_RATIO_THRESHOLD,
        TOKEN_JACCARD_THRESHOLD,
    )

    source_norm = _normalize_title(source_title)
    target_norm = _normalize_title(target_title)
    if not source_norm or not target_norm:
        return False
    if SequenceMatcher(None, source_norm, target_norm).ratio() >= (
        TITLE_RATIO_THRESHOLD
    ):
        return True
    source_tokens = _informative_title_tokens(source_norm)
    target_tokens = _informative_title_tokens(target_norm)
    if not source_tokens or not target_tokens:
        return False
    shared = source_tokens & target_tokens
    if len(shared) < MIN_SHARED_TITLE_TOKENS:
        return False
    return len(shared) / len(source_tokens | target_tokens) >= (TOKEN_JACCARD_THRESHOLD)


_TITLE_WORDS = st.sampled_from(
    [
        "show",
        "hn",
        "launch",
        "linux",
        "kernel",
        "gpu",
        "apple",
        "silicon",
        "reverse",
        "engineering",
        "open",
        "source",
        "compiler",
        "rust",
        "python",
        "database",
        "postgres",
        "sqlite",
        "vector",
        "embedding",
        "transformer",
        "diffusion",
        "quantization",
        "inference",
        "agent",
        "browser",
        "terminal",
        "keyboard",
        "monitor",
        "battery",
        "solar",
        "a",
        "the",
        "of",
        "and",
        "for",
        "with",
        "supercalifragilistic",
    ]
)


def _hnish_title(words: list[str], prefix: str) -> str:
    return prefix + " ".join(words)


_TITLE_LIST = st.lists(_TITLE_WORDS, min_size=2, max_size=14)
_TITLE_PREFIX = st.sampled_from(["", "", "", "Show HN: ", "Launch HN: ", "Ask HN: "])


@given(st.data())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_precomputed_similarity_matches_naive_reference(data: st.DataObject) -> None:
    """The length-gated + precomputed path must agree with the naive
    normalize-then-full-difflib reference on every pair, including
    length-mismatched pairs where the gate skips difflib entirely."""
    a = _hnish_title(data.draw(_TITLE_LIST), data.draw(_TITLE_PREFIX))
    b = _hnish_title(data.draw(_TITLE_LIST), data.draw(_TITLE_PREFIX))
    assert _story_titles_are_similar(a, b) == _naive_titles_are_similar(a, b)


def test_ratio_length_bound_boundaries() -> None:
    # Equal lengths always plausible; 3*min < max never is (threshold 0.5).
    assert _ratio_length_plausible(10, 10)
    assert _ratio_length_plausible(10, 30)
    assert not _ratio_length_plausible(9, 30)
    assert not _ratio_length_plausible(0, 5)
    assert _ratio_length_plausible(0, 0)


def test_matches_feedback_uses_precomputed_keys() -> None:
    """_load_feedback_context must keep hn_title_keys parallel to hn_stories,
    and _matches_feedback must consult them (URL + id fast paths first)."""
    db = Database(":memory:")
    try:
        db.upsert_story(
            Story(
                id=7,
                title="Asahi Linux on M3",
                url="https://example.com/asahi",
                score=10,
                time=1600000000,
                text_content="",
                source="hn",
            )
        )
        db.upsert_feedback(1, story_id=7, action="up")
        ctx = _load_feedback_context(db, user_id=1, actions=("up",))
        assert len(ctx.hn_title_keys) == len(ctx.hn_stories) == 1
        norm, tokens = ctx.hn_title_keys[0]
        assert norm == _normalize_title("Asahi Linux on M3")
        assert tokens == frozenset(_informative_title_tokens(norm))

        same_id = Story(
            id=7,
            title="Different",
            url=None,
            score=0,
            time=0,
            text_content="",
            source="hn",
        )
        assert _matches_feedback(same_id, ctx)
        near_dupe = Story(
            id=8,
            title="Asahi Linux on M3 Pro",
            url=None,
            score=0,
            time=0,
            text_content="",
            source="hn",
            comment_count=3,
        )
        assert _matches_feedback(near_dupe, ctx)
        unrelated = Story(
            id=9,
            title="Completely different baking recipe sourdough",
            url=None,
            score=0,
            time=0,
            text_content="",
            source="hn",
            comment_count=3,
        )
        assert not _matches_feedback(unrelated, ctx)
    finally:
        db.close()


def test_ratio_verdict_cache_reuses_pairs() -> None:
    """The pair cache must serve repeat title pairs without recomputation:
    consecutive warms evaluate mostly identical pairs, so the second pass
    over the same pairs must be cache hits (the ~8s/warm difflib lever)."""
    from pipeline.hn_dupes import _cached_ratio_ge

    _cached_ratio_ge.cache_clear()
    try:
        assert _cached_ratio_ge("asahi linux on m3", "asahi linux on m3 pro")
        info = _cached_ratio_ge.cache_info()
        assert info.misses == 1 and info.hits == 0
        assert _cached_ratio_ge("asahi linux on m3", "asahi linux on m3 pro")
        info = _cached_ratio_ge.cache_info()
        assert info.hits == 1
    finally:
        _cached_ratio_ge.cache_clear()
