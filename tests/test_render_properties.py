"""Card links from untrusted feeds: only http(s) may reach href/window.open.
Feed orders follow the view rules both clients rely on."""

from __future__ import annotations

import json

from hypothesis import given, settings, strategies as st

from clients.tui.src.hn_rerank.models import Feed
from database import Story
from pipeline import Config, RankedStory
from pipeline.render import RECOMMENDED_LIMIT, _web_url, build_feed

_SCHEMES = st.sampled_from(
    ["http", "https", "javascript", "data", "vbscript", "file", "ftp", "blob"]
)


def _browser_scheme(url: str) -> str | None:
    """Spec (WHATWG URL parsing, the parts that matter for a scheme): strip
    leading/trailing C0 controls and spaces, drop every tab/CR/LF, then the
    scheme is an ASCII letter followed by letters, digits, + - . up to ':'."""
    url = url.strip("".join(map(chr, range(0x21))))
    url = url.replace("\t", "").replace("\n", "").replace("\r", "")
    head, sep, _ = url.partition(":")
    if not sep or not head or not head[0].isascii() or not head[0].isalpha():
        return None
    if not all(c.isascii() and (c.isalnum() or c in "+-.") for c in head):
        return None
    return head.lower()


@st.composite
def _hostile_urls(draw: st.DrawFn) -> str:
    scheme = draw(_SCHEMES)
    # Mixed case and embedded tab/newline: browsers still read the scheme.
    scheme = "".join(
        draw(st.sampled_from([c.lower(), c.upper()]))
        + draw(st.sampled_from(["", "", "\t", "\n"]))
        for c in scheme
    )
    lead = draw(st.text(alphabet=" \t\n\x00\x01\x1f", max_size=3))
    rest = draw(st.sampled_from(["//example.com/a", "alert(1)", "text/html,<b>"]))
    return lead + scheme + ":" + rest


@given(st.one_of(_hostile_urls(), st.text(max_size=30)))
def test_card_links_are_http_or_empty(url: str) -> None:
    out = _web_url(url)
    assert out in ("", url)  # never rewritten, only dropped
    if out:
        assert _browser_scheme(out) in ("http", "https")


@given(_hostile_urls())
def test_real_web_links_survive(url: str) -> None:
    if _browser_scheme(url) in ("http", "https"):
        assert _web_url(url) == url


_FLAGS = (
    "is_hot",
    "is_high_engagement",
    "is_discussion_rich",
    "is_uncertain",
    "is_similar",
    "is_novel",
)


@st.composite
def _decks(draw: st.DrawFn) -> list[RankedStory]:
    """Ranked decks as the ranker emits them: one age and source per story,
    usually in its age's mixed deck, with any mix of badges."""
    ids = draw(st.lists(st.integers(1, 10**6), unique=True, max_size=60))
    deck = []
    for story_id in ids:
        age = draw(st.sampled_from(["recent", "archive"]))
        source = draw(st.sampled_from(["hn", "non-hn"]))
        keys = [f"{age}_{source}"]
        if draw(st.booleans()) or draw(st.booleans()):
            keys.append(f"{age}_mixed")
        story = Story(
            id=story_id,
            title=draw(
                st.text(
                    st.characters(codec="utf-8", exclude_categories=["Cc", "Cs"]),
                    max_size=20,
                )
            ),
            url=draw(st.none() | st.just(f"https://example.com/{story_id}")),
            score=draw(st.integers(0, 500)),
            time=draw(st.integers(1_600_000_000, 1_700_000_000)),
            text_content="",
            source="hn" if source == "hn" else "rss_example_com",
        )
        deck.append(
            RankedStory(
                story=story,
                # Coarse scores so ties are common: orders must still agree.
                score=draw(st.integers(-3, 3)) / 2,
                best_match_title="",
                combo_keys=" ".join(keys),
                is_recent=age == "recent",
                **{flag: draw(st.booleans()) for flag in _FLAGS},
            )
        )
    return deck


@settings(max_examples=80, deadline=None)
@given(
    deck=_decks(),
    version=st.integers(0, 10**13),
    target=st.integers(0, 10**13),
)
def test_feed_orders_follow_the_view_rules(
    deck: list[RankedStory], version: int, target: int
) -> None:
    feed = build_feed(
        deck, Config(), {"up": 1, "neutral": 0, "down": 2}, version, target
    )
    ids = [story.id for story in feed.stories]
    by_id = {story.id: story for story in feed.stories}
    assert len(ids) == len(set(ids)) and set(ids) <= {r.story.id for r in deck}
    assert (feed.version, feed.target_version, feed.ready) == (
        version,
        target,
        version >= target,
    )
    for key, order in feed.orders.items():
        assert len(order) == len(set(order)) and set(order) <= set(ids), key

    for age in ("recent", "archive"):
        in_age = [r for r in deck if f"{age}_mixed" in r.combo_keys.split()]
        by_score = sorted(in_age, key=lambda r: r.score, reverse=True)
        assert feed.orders[f"recommended:{age}"] == [
            r.story.id for r in by_score[:RECOMMENDED_LIMIT]
        ]
        assert feed.orders[f"popular:{age}"] == [
            r.story.id
            for r in by_score
            if r.is_hot or r.is_high_engagement or r.is_discussion_rich
        ]
        # Explore is shuffled: the same stories, in any order.
        assert sorted(feed.orders[f"explore:{age}"]) == sorted(
            r.story.id for r in in_age if r.is_uncertain or r.is_similar or r.is_novel
        )
        # Date: every story sent, newest first, under either Age tab.
        assert sorted(feed.orders[f"date:{age}"]) == sorted(ids)
        times = [by_id[sid].time for sid in feed.orders[f"date:{age}"]]
        assert times == sorted(times, reverse=True)

    # What the server sends is what the TUI parses.
    assert Feed.parse(json.loads(json.dumps(feed.to_dict()))) == feed
