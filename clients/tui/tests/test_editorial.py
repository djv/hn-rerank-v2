from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx
import pytest
from rich.color import Color
from rich.text import Text
from textual.widgets import Button, Input, Markdown, OptionList, Select, Static, Tabs

from hn_rerank.app import (
    EMPTY_NOTICE,
    Reader,
    Setup,
    diversify_order,
    headline,
    headline_domain,
    headline_points,
    story_age,
    story_metadata,
)
from hn_rerank.models import FeedStory

from .test_client import FakeServer


class UnreachableServer(FakeServer):
    def __init__(self) -> None:
        super().__init__()
        self.fail_feed = True

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail_feed and request.url.path.endswith("/api/feed"):
            raise httpx.ConnectError("unreachable", request=request)
        return await super().__call__(request)


class EditorialServer(FakeServer):
    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/tldr-detail"):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "tldr": (
                        "# A readable summary\n\n"
                        + "\n\n".join(
                            f"## Section {i}\n\nA paragraph with **emphasis** and a useful explanation.\n\n"
                            "- First point\n- Second point\n\n> A quoted observation.\n\n"
                            '```python\nprint("hello reader")\n```'
                            for i in range(5)
                        )
                    ),
                },
            )
        return await super().__call__(request)


@pytest.mark.parametrize("width", [60, 80, 100, 140])
async def test_editorial_filters_and_resize(width: int) -> None:
    fake = EditorialServer()
    fake.feed.stories[0] = replace(
        fake.feed.stories[0],
        title="A long headline about careful software design and readable terminal interfaces "
        * 2,
    )
    app = Reader(api=fake.api())
    async with app.run_test(size=(width, 35)) as pilot:
        await pilot.pause(0.6)
        listing = app.query_one(OptionList)
        summary = app.query_one(Markdown)
        assert app.query_one("#sort-tabs", Tabs).display == (width >= 100)
        assert app.query_one("#sort", Select).display == (width < 100)
        assert "example.org" in str(app.query_one("#story-heading", Static).content)
        assert listing.highlighted == 0
        assert listing.display and summary.display
        if width < 100:
            assert (
                app.query_one("#reading-pane").region.y
                > app.query_one("#headlines").region.y
            )
        if width >= 100:
            await pilot.click("#sort-popular")
        else:
            app.query_one("#sort", Select).value = "popular"
        await pilot.pause()
        assert [s.id for s in app.stories] == [1]
        assert app.query_one("#sort-tabs", Tabs).active == "sort-popular"
        summary.focus()
        await pilot.press("down", "down", "down")
        await pilot.pause()
        scroll = summary.scroll_y
        assert scroll > 0
        content = summary._markdown
        for new_width in (60, 80, 100, 140, width):
            await pilot.resize_terminal(new_width, 35)
            await pilot.pause()
            selected = app.selected()
            assert selected and selected.id == 1
            assert summary._markdown == content
            assert summary.scroll_y == scroll
            assert summary.has_focus
        # j/k only move the headline list, even while the summary is focused.
        summary.focus()
        await pilot.press("j")
        await pilot.pause()
        assert summary.scroll_y == scroll


@pytest.mark.parametrize("width", [60, 80, 100, 140])
async def test_setup_layout_and_error(width: int, tmp_path: Path) -> None:
    app = Reader(config_path=tmp_path / "missing.json")
    async with app.run_test(size=(width, 35)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, Setup)
        for name in ("link", "import", "server", "create", "quit"):
            widget = app.screen.query_one("#" + name)
            assert widget.region.right <= width
            assert 0 <= widget.region.y < 35
        assert app.screen.query_one("#link", Input).password
        await pilot.click("#import")
        await pilot.pause()
        message = app.screen.query_one("#setup-message", Static)
        assert message.has_class("error")
        assert "try again" in str(message.content)


async def test_footer_hints_spell_out_vote_directions() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause(0.6)
        hints = app.query_one("#shortcuts", Static)
        assert "1 up · 2 neutral · 3 down" in str(hints.content)
        assert "j/k move" in str(hints.content)
        assert app.query_one("#shortcuts").region.height == 1
        assert app.query_one("#status").region.height == 1


async def test_enter_does_not_toggle_view() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        listing = app.query_one(OptionList)
        summary = app.query_one(Markdown)
        assert listing.has_focus
        await pilot.press("enter")
        await pilot.pause()
        assert listing.has_focus
        assert not summary.has_focus
        assert listing.display and summary.display


async def test_read_mode_only_when_summary_overflows() -> None:
    short = FakeServer()
    app = Reader(api=short.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        hints = app.query_one("#shortcuts", Static)
        assert "Enter read" not in str(hints.content)
        listing = app.query_one(OptionList)
        await pilot.press("enter")
        await pilot.pause()
        assert listing.has_focus
        assert not app.reading

    long = EditorialServer()
    app = Reader(api=long.api())
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause(0.6)
        hints = app.query_one("#shortcuts", Static)
        for _ in range(20):
            if "Enter read" in str(hints.content):
                break
            await pilot.pause(0.2)
        assert "Enter read" in str(hints.content)
        listing = app.query_one(OptionList)
        await pilot.press("enter")
        await pilot.pause()
        assert app.reading
        assert not listing.display
        assert app.query_one(Markdown).has_focus
        await pilot.press("j")
        await pilot.pause()
        assert app.query_one(Markdown).scroll_y > 0
        await pilot.press("escape")
        await pilot.pause()
        assert not app.reading
        assert listing.display


async def test_empty_and_error_recovery() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause(0.6)
        fake.fail_vote = True
        app.action_vote("up")
        await pilot.pause(0.2)
        assert app.query_one("#status").has_class("error")
        assert "check before voting" in str(app.query_one("#status", Static).content)
        app.query_one("#age", Select).value = "archive"
        app.query_one("#sort", Select).value = "popular"
        await pilot.pause()
        assert not app.stories
        assert "No stories" in app.query_one(Markdown)._markdown
        assert not str(app.query_one("#story-heading", Static).content)


def test_story_age_guard_and_buckets() -> None:
    now = int(time.time())
    story = FeedStory(
        1,
        "Story",
        "https://example.org/a",
        "https://news.ycombinator.com/item?id=1",
        "hn",
        10,
        None,
        0,
        1.0,
        ["recent_mixed"],
        False,
        False,
    )
    # Placeholder timestamps must not render as "20000d ago".
    assert story_metadata(story) == "example.org · 10 pts · 0 comments"
    assert story_age(replace(story, time=now - 120)) == "2m"
    assert story_age(replace(story, time=now - 5 * 86400)) == "5d"
    assert story_age(replace(story, time=now - 45 * 86400)) == "1mo"
    assert story_age(replace(story, time=now - 400 * 86400)) == "1y"
    assert story_metadata(replace(story, time=now - 5 * 86400)).endswith("5d ago")


def test_headline_shows_badge_emoji() -> None:
    story = FeedStory(
        1,
        "Story",
        "https://example.org/a",
        "https://news.ycombinator.com/item?id=1",
        "hn",
        10,
        None,
        0,
        1.0,
        ["recent_mixed"],
        False,
        False,
        badges=["\U0001f525", "\U0001f3c6"],
    )
    assert headline(story).plain.startswith("🔥 🏆 Story")
    assert headline(story, selected=True).plain.startswith("> 🔥 🏆 Story")
    assert headline(replace(story, badges=[])).plain.startswith("Story")


def test_headline_hides_unknown_reddit_score_and_shows_subreddit() -> None:
    story = FeedStory(
        1,
        "Story",
        "https://www.reddit.com/r/LocalLLaMA/comments/abc/slug/",
        "https://www.reddit.com/r/LocalLLaMA/comments/abc/slug/",
        "rss_reddit_localllama",
        0,
        None,
        0,
        1.0,
        ["recent_mixed"],
        False,
        False,
    )
    rendered = headline(story).plain
    assert "r/localllama" in rendered
    assert "pts" not in rendered
    assert "· 5 ·" in headline(replace(story, points=5)).plain
    hn = replace(story, source="hn", article_url="https://example.org/a")
    assert "example.org" in headline(hn).plain
    assert "· 0 ·" in headline(hn).plain


def test_headline_separators_share_columns_across_stories() -> None:
    base = FeedStory(
        1,
        "Story",
        "https://example.org/a",
        "https://news.ycombinator.com/item?id=1",
        "hn",
        5,
        7,
        0,
        1.0,
        ["recent_mixed"],
        False,
        False,
    )
    reddit = replace(
        base,
        id=2,
        article_url="https://www.reddit.com/r/LocalLLaMA/comments/abc/slug/",
        source="rss_reddit_localllama",
        points=0,
        comments=1234,
    )
    widths = (
        max(len(headline_domain(base)), len(headline_domain(reddit))),
        max(len(headline_points(base)), len(headline_points(reddit))),
        max(len("7 comments"), len("1234 comments")),
    )
    first = headline(base, True, widths).plain.splitlines()[1]
    second = headline(reddit, False, widths).plain.splitlines()[1]
    dots = [i for i, char in enumerate(first) if char == "·"]
    assert dots == [i for i, char in enumerate(second) if char == "·"]
    assert "pts" not in second
    assert "r/localllama" in second


def test_diversify_order_caps_sources_and_total() -> None:
    lookup = {
        i: replace(
            FeedStory(
                1,
                "S",
                "https://example.org",
                "https://example.org/x",
                "hn",
                1,
                0,
                0,
                1.0,
                ["recent_mixed"],
                False,
                False,
            ),
            id=i,
            source="rss_lesswrong_com" if i % 2 else "hn",
        )
        for i in range(1, 41)
    }
    order = list(range(1, 41)) + [999]
    picked = diversify_order(order, lookup)
    assert len(picked) == 8  # 4 per source over 2 sources
    assert picked == [1, 2, 3, 4, 5, 6, 7, 8]
    many = {i: replace(lookup[1], id=i, source=f"rss_src_{i}") for i in range(100)}
    assert len(diversify_order(list(range(100)), many)) == 30


def test_headline_truncates_long_domains_to_fit() -> None:
    long_domain = FeedStory(
        1,
        "Story",
        "https://marginalrevolution.com/posts/abc/slug",
        "https://marginalrevolution.com/posts/abc/slug",
        "rss_marginalrevolution_com",
        5,
        7,
        0,
        1.0,
        ["recent_mixed"],
        False,
        False,
    )
    assert (
        headline(long_domain, True, (10, 1, 1)).plain.splitlines()[1]
        == "marginalr… · 5 · 7"
    )
    assert "marginalrevolution.com" in headline(long_domain).plain


async def test_failure_copy_in_reading_pane() -> None:
    server = UnreachableServer()
    app = Reader(api=server.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        assert app.feed is None
        markdown = app.query_one(Markdown)._markdown
        assert "# Could not reach server" in markdown
        assert "Connection failed" in markdown
        assert "**r** to retry" in markdown
        assert app.query_one("#status").has_class("error")
        assert not str(app.query_one("#story-heading", Static).content)
        server.fail_feed = False
        app.action_refresh()
        await pilot.pause(0.6)
        assert [s.id for s in app.stories] == [1, 2]
        assert "Could not reach server" not in app.query_one(Markdown)._markdown
        assert str(app.query_one("#status", Static).content) == "2 shown · +0 ~0 −0"


async def test_empty_notice_heading() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause(0.6)
        app.query_one("#age", Select).value = "archive"
        app.query_one("#sort", Select).value = "popular"
        await pilot.pause()
        assert not app.stories
        assert app.query_one(Markdown)._markdown == EMPTY_NOTICE
        assert not app.query_one("#reading-pane").has_class("has-story")


async def test_setup_modal_chrome(tmp_path: Path) -> None:
    app = Reader(config_path=tmp_path / "missing.json")
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, Setup)
        assert app.screen.styles.background.a < 1.0
        assert app.screen.query_one("#setup").styles.border_top[0] == "round"
        widths = {
            app.screen.query_one("#" + name, Button).region.width
            for name in ("import", "create", "quit")
        }
        assert len(widths) == 1


async def test_reading_measure_cap() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(220, 40)) as pilot:
        await pilot.pause(0.6)
        assert app.query_one("#reading-pane").region.width <= 100
        assert app.query_one("#story-heading").styles.border_bottom[0] == "solid"


async def test_summary_emphasis_uses_accent_color() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        block = app.query("#summary MarkdownBlock").first()
        strong = block.get_component_rich_style("strong")
        em = block.get_component_rich_style("em")
        assert strong.color == Color.parse("#FF914D")
        assert strong.bold
        assert em.color == Color.parse("#FF914D")


async def test_summary_headings_align_left_like_body() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        heading = app.query("#summary MarkdownH1").first()
        assert heading.styles.content_align == ("left", "top")


async def test_badge_legend_hotkey_and_escape_restore_story() -> None:
    app = Reader(api=EditorialServer().api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        await pilot.press("b")
        await pilot.pause()
        legend = app.query_one(Markdown)._markdown
        for badge in (
            "🔥 **Hot**",
            "🏆 **Top**",
            "💬 **Talk**",
            "🤔 **Unsure**",
            "✨ **Novel**",
            "🎯 **Similar**",
        ):
            assert badge in legend
        app.rebuild()
        await pilot.pause()
        assert "Badge legend" in app.query_one(Markdown)._markdown
        await pilot.press("escape")
        await pilot.pause(0.5)
        assert "Section 0" in app.query_one(Markdown)._markdown


async def test_help_escape_restores_story_view_and_survives_refresh() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        assert "Section 0" in app.query_one(Markdown)._markdown
        await pilot.press("?")
        await pilot.pause()
        assert "Shortcuts" in app.query_one(Markdown)._markdown
        app.rebuild()  # A feed refresh must not steal the open help pane.
        await pilot.pause()
        assert "Shortcuts" in app.query_one(Markdown)._markdown
        await pilot.press("escape")
        await pilot.pause(0.5)
        assert "Section 0" in app.query_one(Markdown)._markdown


async def test_reading_heading_tracks_refreshed_story_data() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        fake.feed.stories[0] = replace(fake.feed.stories[0], points=999, comments=123)
        app.action_refresh()
        await pilot.pause(0.6)
        selected = app.selected()
        assert selected and selected.id == 1
        heading = str(app.query_one("#story-heading", Static).content)
        assert "· 999 · 123" in heading


async def test_vote_statusline_confirms_without_toast() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await pilot.pause(0.5)
        assert [n.message for n in app._notifications] == []
        assert app.rated == {1}
        assert not app.query_one("#status").has_class("error")


async def test_footer_counts_follow_filters() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.6)
        status = app.query_one("#status", Static)
        assert str(status.content) == "2 shown · +0 ~0 −0"
        assert status.has_class("context")
        assert {"#A8C7A0", "#E5C07B", "#FFB4A6"} <= {
            str(span.style) for span in cast(Text, status.content).spans
        }
        assert app.query_one("#reading-pane").has_class("has-story")
        assert app.query_one("#footer").region.bottom == pilot.app.size.height
        listing = app.query_one(OptionList)
        styles = {
            str(span.style)
            for index in range(listing.option_count)
            for span in cast(Text, listing.get_option_at_index(index).prompt).spans
        }
        # Selected row keeps the ivory title; unselected rows are dimmed.
        # Metadata adds restrained color: blue domain, sage points, dim separators.
        assert {"bold #EEE8DD", "#D2CCC1", "#8AB4F8", "#A8C7A0"} <= styles
        app.query_one("#sort", Select).value = "popular"
        app.query_one("#age", Select).value = "archive"
        await pilot.pause()
        assert str(app.query_one("#status", Static).content) == "0 shown · +0 ~0 −0"
        app.status("Could not reach server.", error=True)
        assert str(app.query_one("#status", Static).content).startswith("✗ ")
        app.rebuild()
        assert "Could not reach server" in str(app.query_one("#status", Static).content)
