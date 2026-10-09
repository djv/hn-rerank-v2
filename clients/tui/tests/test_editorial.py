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
    LIGHT_PALETTE,
    Reader,
    VIEW_LIMIT,
    Setup,
    _cell_len,
    headline,
    headline_comments,
    headline_domain,
    headline_points,
    story_age,
    story_heading,
    story_metadata,
    theme_for_hour,
)
from hn_rerank.models import FeedStory

from ._settle import settle
from .test_client import FakeServer


@pytest.fixture(autouse=True)
def dark_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Colour assertions pin the dark palette regardless of the wall clock."""
    monkeypatch.setattr("hn_rerank.app.theme_for_hour", lambda _hour: "editorial")


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
                            for i in range(3)
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
        await settle(pilot)
        listing = app.query_one(OptionList)
        summary = app.query_one(Markdown)
        assert app.query_one("#sort-tabs", Tabs).display == (width >= 100)
        assert app.query_one("#sort", Select).display == (width < 100)
        sort_control = app.query_one("#sort-tabs" if width >= 100 else "#sort")
        window_control = app.query_one("#window", Select)
        assert window_control.display
        assert sort_control.region.y == window_control.region.y
        assert sort_control.region.right <= window_control.region.x
        assert window_control.region.right <= width
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
        # Every start width reaches every other through this sweep; run it once.
        for new_width in (60, 80, 100, 140, width) if width == 140 else ():
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


async def test_focused_pane_shows_accent_border() -> None:
    app = Reader(api=FakeServer().api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        headlines = app.query_one("#headlines", OptionList)
        headlines.focus()
        await pilot.pause()
        assert headlines.styles.border_top[1].hex.upper() == "#FF914D"
        app.query_one(Markdown).focus()
        await pilot.pause()
        assert headlines.styles.border_top[1].hex.upper() == "#171717"


@pytest.mark.parametrize("width", [60, 120])
async def test_filter_change_starts_at_first_story(width: int) -> None:
    app = Reader(api=FakeServer().api())
    async with app.run_test(size=(width, 35)) as pilot:
        await settle(pilot)
        listing = app.query_one("#headlines", OptionList)
        listing.highlighted = 1
        await pilot.pause()
        selected = app.selected()
        assert selected is not None and selected.id == 2
        if width < 100:
            app.query_one("#sort", Select).value = "explore"
        else:
            await pilot.click("#sort-explore")
        await pilot.pause()
        assert [story.id for story in app.stories] == [2]
        assert listing.highlighted == 0
        selected = app.selected()
        assert selected is not None and selected.id == 2
        if width < 100:
            app.query_one("#sort", Select).value = "recommended"
        else:
            await pilot.click("#sort-recommended")
        await pilot.pause()
        assert listing.highlighted == 0
        selected = app.selected()
        assert selected is not None and selected.id == 1
        assert listing.scroll_y == 0
        assert app.query_one("#story-heading", Static).outer_size.height <= 4


@pytest.mark.parametrize("width", [60, 80, 100, 140])
async def test_setup_layout_and_error(width: int, tmp_path: Path) -> None:
    app = Reader(config_path=tmp_path / "missing.json")
    async with app.run_test(size=(width, 35)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, Setup)
        for name in ("link", "import", "token", "use-token", "create", "quit"):
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
        await settle(pilot)
        hints = app.query_one("#shortcuts", Static)
        assert "1 up · 2 neutral · 3 down → next story" in str(hints.content)
        assert "j/k move" in str(hints.content)
        assert app.query_one("#shortcuts").region.height == 1
        assert app.query_one("#status").region.height == 1


async def test_narrow_footer_keeps_status_visible() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(51, 37)) as pilot:
        await settle(pilot)
        status = app.query_one("#status").region
        hints = app.query_one("#shortcuts").region
        # One row: the keys shrink so the counts line stays whole beside them.
        assert app.query_one("#footer").region.height == 1
        assert status.y == hints.y and status.right <= hints.x
        assert str(app.query_one("#status", Static).content).endswith("−0")
        assert str(app.query_one("#shortcuts", Static).content).endswith("? help")


async def test_wide_footer_is_one_row() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(146, 39)) as pilot:
        await settle(pilot)
        status = app.query_one("#status").region
        hints = app.query_one("#shortcuts").region
        assert status.y == hints.y
        assert status.height == hints.height == 1
        assert status.right <= hints.x
        assert "b badges" in str(app.query_one("#shortcuts", Static).content)
        # A long message keeps the row: the keys shrink to "? help".
        app.status("x" * 80)
        await pilot.pause()
        assert app.query_one("#footer").region.height == 1
        assert "b badges" not in str(app.query_one("#shortcuts", Static).content)
        assert str(app.query_one("#status", Static).content) == "x" * 80
        # The counts line brings the full keys back.
        app.status_mode = "context"
        app.context_status()
        await pilot.pause()
        assert "b badges" in str(app.query_one("#shortcuts", Static).content)


@pytest.mark.parametrize("width", [100, 240])
async def test_wide_layout_gives_list_a_third(width: int) -> None:
    app = Reader(api=FakeServer().api())
    async with app.run_test(size=(width, 40)) as pilot:
        await settle(pilot)
        available = app.query_one("#panes").region
        listing = app.query_one("#headlines").region
        pane = app.query_one("#reading-pane").region
        assert listing.y == pane.y
        assert listing.width + pane.width == available.width
        assert abs(listing.width - available.width / 3) <= 1


@pytest.mark.parametrize("size", [(51, 37), (80, 30), (140, 40)])
@pytest.mark.parametrize("long_summary", [False, True])
async def test_enter_zooms_tldr(size: tuple[int, int], long_summary: bool) -> None:
    fake = EditorialServer() if long_summary else FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=size) as pilot:
        await settle(pilot)
        listing = app.query_one(OptionList)
        summary = app.query_one(Markdown)
        hints = app.query_one("#shortcuts", Static)
        assert listing.has_focus
        assert "Enter zoom" in str(hints.content)
        for exit_key in ("enter", "escape"):
            selected = app.selected()
            await pilot.press("enter")
            await pilot.pause()
            assert app.reading
            assert not listing.display
            assert summary.has_focus
            pane = app.query_one("#reading-pane").region
            available = app.query_one("#panes").region
            assert pane.width == min(100, available.width)
            assert abs((pane.x - available.x) - (available.right - pane.right)) <= 1
            assert "Enter/Esc back" in str(hints.content)
            if long_summary:
                assert summary.max_scroll_y > 0
            # j/k change stories in zoom as in the list.
            await pilot.press("j")
            await settle(pilot)
            assert app.reading and summary.has_focus
            moved = app.selected()
            assert moved is not None and selected is not None
            assert moved.id != selected.id
            assert summary.scroll_y == 0
            await pilot.press("k")
            await settle(pilot)
            assert app.selected() == selected
            await pilot.press(exit_key)
            await pilot.pause()
            assert not app.reading
            assert listing.display and listing.has_focus
            assert app.selected() == selected
            assert "Enter zoom" in str(hints.content)


async def test_empty_and_error_recovery() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(100, 35)) as pilot:
        await settle(pilot)
        fake.fail_vote = True
        app.action_vote("up")
        await settle(pilot)
        assert app.query_one("#status").has_class("error")
        # Too long for 100 columns: cut in the middle, both ends readable.
        shown = str(app.query_one("#status", Static).content)
        assert shown.startswith("✗ Vote not confirmed") and "…" in shown
        assert shown.endswith("Press r to refresh; votes are not retried.")
        app.query_one("#window", Select).value = "12h"
        app.query_one("#sort", Select).value = "popular"
        await settle(pilot)
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
        badges=["\U0001f525", "\U0001f3c6", "\U0001f4ac"],
    )
    assert headline(story).plain.startswith("🔥 🏆 💬 Story")
    assert headline(story, selected=True).plain.startswith("> 🔥 🏆 💬 Story")
    assert headline(replace(story, badges=[])).plain.startswith("Story")


def test_headline_hides_unknown_feed_counts_and_shows_subreddit() -> None:
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
    )
    rendered = headline(story).plain
    assert rendered.splitlines()[1] == "r/localllama"
    assert headline(replace(story, points=5, comments=3)).plain.endswith(
        "r/localllama · ▲ 5 · 💬 3"
    )
    # HN counts are real even at zero.
    hn = replace(story, source="hn", article_url="https://example.org/a")
    assert headline(hn).plain.endswith("example.org · ▲ 0 · 💬 0")


def test_headline_names_the_feed_not_the_linked_domain() -> None:
    ainews = FeedStory(
        1,
        "Story",
        "https://x.com/someone/status/1",
        "https://news.smol.ai/issues/1",
        "rss_ainews",
        0,
        12,
        0,
        1.0,
        source_label="AINews",
    )
    assert headline(ainews).plain.splitlines()[1] == "AINews · 💬 12"
    hn = replace(ainews, source="hn", source_label="HN", points=3)
    assert headline_domain(hn) == "x.com"


def test_heading_says_why_a_story_is_recommended() -> None:
    story = FeedStory(
        1,
        "Story",
        "https://example.org/a",
        "https://news.ycombinator.com/item?id=1",
        "hn",
        5,
        7,
        0,
        1.0,
    )
    assert story_heading(story).plain == headline(story).plain
    because = replace(story, best_match_title="Earlier story")
    assert story_heading(because).plain.splitlines()[-1] == (
        "Because you upvoted: Earlier story"
    )


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
    )
    now = int(time.time())
    base = replace(base, time=now - 3 * 3600)
    reddit = replace(
        base,
        id=2,
        article_url="https://www.reddit.com/r/LocalLLaMA/comments/abc/slug/",
        source="rss_reddit_localllama",
        points=0,
        comments=1234,
        time=now - 12 * 86400,
    )
    blog = replace(base, id=3, source="rss_blog", points=0, comments=0)
    stories = (base, reddit, blog)
    widths = (
        max(_cell_len(headline_domain(s)) for s in stories),
        max(_cell_len(headline_points(s)) for s in stories),
        max(_cell_len(headline_comments(s)) for s in stories),
    )
    first, second, third = (
        headline(s, s is base, widths).plain.splitlines()[1] for s in stories
    )
    # Unknown counts leave blank columns: every row's age starts where the
    # first row's does, and blank segments drop their separator too.
    assert first.endswith("3h") and second.endswith("12d") and third.endswith("3h")
    rows = (first, second, third)
    assert len({_cell_len(row[: row.rindex("·")]) for row in rows}) == 1
    assert _cell_len(first[: first.index("💬")]) == _cell_len(
        second[: second.index("💬")]
    )
    assert "▲" not in second and "▲" not in third and "💬" not in third
    assert third.count("·") == 1
    assert "r/localllama" in second


async def test_every_sort_shows_at_most_view_limit(tmp_path: Path) -> None:
    """Each sort caps at 12; rated ones backfill."""

    def story(i: int) -> FeedStory:
        return FeedStory(
            i,
            f"S{i}",
            "https://example.org",
            "https://example.org/x",
            "hn",
            1,
            0,
            0,
            float(100 - i),
        )

    ids = list(range(1, 21))
    fake = FakeServer()
    fake.feed = replace(
        fake.feed,
        stories=[story(i) for i in ids],
        orders={sort: ids for sort in Reader.SORT_CYCLE},
    )
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        for _ in Reader.SORT_CYCLE:
            assert len(app.stories) == VIEW_LIMIT
            await pilot.press("s")
            await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == "recommended"
        assert [s.id for s in app.stories] == ids[:VIEW_LIMIT]
        app.rated.add(1)
        app.rebuild()
        assert [s.id for s in app.stories] == ids[1 : VIEW_LIMIT + 1]


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
    )
    assert (
        headline(long_domain, True, (10, 3, 3)).plain.splitlines()[1]
        == "marginalr… · ▲ 5 · 💬 7"
    )
    assert "marginalrevolution.com" in headline(long_domain).plain


async def test_failure_copy_in_reading_pane() -> None:
    server = UnreachableServer()
    app = Reader(api=server.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert app.feed is None
        markdown = app.query_one(Markdown)._markdown
        assert "# Could not reach server" in markdown
        assert "Connection failed" in markdown
        assert "Switch sort or time window to retry." in markdown
        assert app.query_one("#status").has_class("error")
        assert not str(app.query_one("#story-heading", Static).content)
        server.fail_feed = False
        # r cannot refetch a failed feed anymore; a filter switch retries it.
        app.query_one("#sort", Select).value = "popular"
        await settle(pilot)
        assert [s.id for s in app.stories] == [1]
        assert "Could not reach server" not in app.query_one(Markdown)._markdown


async def test_empty_notice_heading() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(100, 35)) as pilot:
        await settle(pilot)
        app.query_one("#window", Select).value = "12h"
        app.query_one("#sort", Select).value = "popular"
        await settle(pilot)
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


async def test_wide_story_heading_rule() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(220, 40)) as pilot:
        await settle(pilot)
        assert app.query_one("#story-heading").styles.border_bottom[0] == "solid"


async def test_summary_emphasis_uses_accent_color() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
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
        await settle(pilot)
        heading = app.query("#summary MarkdownH1").first()
        assert heading.styles.content_align == ("left", "top")


async def test_badge_legend_hotkey_and_escape_restore_story() -> None:
    app = Reader(api=EditorialServer().api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("b")
        await pilot.pause()
        legend = app.query_one(Markdown)._markdown
        for badge in (
            "🔥 **Hot**",
            "🏆 **Top**",
            "💬 **Talk**",
            "🤔 **Unsure**",
            "✨ **Novel**",
            "🎯 **Interest**",
        ):
            assert badge in legend
        app.rebuild()
        await pilot.pause()
        assert "Badge legend" in app.query_one(Markdown)._markdown
        await pilot.press("escape")
        await settle(pilot)
        assert "Section 0" in app.query_one(Markdown)._markdown


async def test_why_story_hotkey_and_escape_restore_summary() -> None:
    app = Reader(api=EditorialServer().api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        selected = app.selected()
        assert selected is not None
        app.stories[app.stories.index(selected)] = replace(
            selected,
            related_upvotes=["An *upvote*", "Another upvote"],
            ranking_factors=["Helped: Content model", "Hurt: Word model"],
        )
        await pilot.press("w")
        await pilot.pause()
        panel = app.query_one(Markdown)._markdown
        assert "# Why this story" in panel
        assert "learned preferences" in panel
        assert r"An \*upvote\*" in panel
        assert "Another upvote" in panel
        assert "Helped: Content model" in panel
        assert "Hurt: Word model" in panel
        await pilot.press("escape")
        await settle(pilot)
        assert "Section 0" in app.query_one(Markdown)._markdown


async def test_help_escape_restores_story_view_and_survives_refresh() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert "Section 0" in app.query_one(Markdown)._markdown
        await pilot.press("?")
        await pilot.pause()
        assert "Shortcuts" in app.query_one(Markdown)._markdown
        app.rebuild()  # A feed refresh must not steal the open help pane.
        await pilot.pause()
        assert "Shortcuts" in app.query_one(Markdown)._markdown
        await pilot.press("escape")
        await settle(pilot)
        assert "Section 0" in app.query_one(Markdown)._markdown


async def test_reading_heading_tracks_refreshed_story_data() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        fake.feed.stories[0] = replace(fake.feed.stories[0], points=999, comments=123)
        # r regenerates the summary without refetching the feed: the heading
        # keeps the served counts until a feed refresh carries the new ones.
        app.action_refresh()
        await settle(pilot)
        selected = app.selected()
        assert selected and selected.id == 1
        heading = str(app.query_one("#story-heading", Static).content)
        assert "· ▲ 100 · 💬 10" in heading
        app.refresh_feed(announce=False)
        await settle(pilot)
        heading = str(app.query_one("#story-heading", Static).content)
        assert "· ▲ 999 · 💬 123" in heading


async def test_vote_statusline_confirms_without_toast() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await settle(pilot)
        assert [n.message for n in app._notifications] == []
        assert app.rated == {1}
        assert not app.query_one("#status").has_class("error")


async def test_footer_counts_follow_filters() -> None:
    fake = EditorialServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
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
        app.query_one("#window", Select).value = "12h"
        await settle(pilot)
        assert str(app.query_one("#status", Static).content) == "0 shown · +0 ~0 −0"
        app.status("Could not reach server.", error=True)
        assert str(app.query_one("#status", Static).content).startswith("✗ ")
        app.rebuild()
        assert "Could not reach server" in str(app.query_one("#status", Static).content)


async def test_narrow_footer_fits_error_and_zoom_shortcuts() -> None:
    app = Reader(api=FakeServer().api())
    async with app.run_test(size=(51, 37)) as pilot:
        await settle(pilot)
        await pilot.press("enter")
        app.status("Connection failed. " * 10, error=True)
        await pilot.pause()
        status = app.query_one("#status").region
        hints = app.query_one("#shortcuts").region
        footer = app.query_one("#footer").content_region
        # A long message is cut to one row with an ellipsis, never wrapped.
        assert status.height == footer.height == 1
        assert status.y == hints.y and status.right <= hints.x
        # The cut is in the middle: what to do next stays readable.
        app.status("Vote not confirmed by the server. " * 3 + "Press r to retry.")
        await pilot.pause()
        shown = str(app.query_one("#status", Static).content)
        assert shown.startswith("Vote not confirm") and "…" in shown
        assert shown.endswith("Press r to retry.")
        assert len(shown) <= 51 - 2 - len("? help") - 2


def test_fit_middle_keeps_both_ends_within_width() -> None:
    from hn_rerank.app import _cell_len, fit_middle

    assert fit_middle("short", 20) == "short"
    cut = fit_middle("Could not save the vote on story 12345. Press r to retry.", 30)
    assert _cell_len(cut) <= 30
    assert cut.startswith("Could not") and cut.endswith("r to retry.")
    wide = fit_middle("🔥" * 30, 21)
    assert _cell_len(wide) <= 21 and "…" in wide


def test_theme_follows_local_hour() -> None:
    assert [theme_for_hour(h) for h in (5, 6, 19, 20)] == [
        "editorial",
        "editorial-light",
        "editorial-light",
        "editorial",
    ]


async def test_clock_switch_restyles_css_and_rich_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = Reader(api=EditorialServer().api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert app.theme == "editorial"
        monkeypatch.setattr(
            "hn_rerank.app.theme_for_hour", lambda _hour: "editorial-light"
        )
        app.apply_clock_theme()
        await pilot.pause()
        assert app.theme == "editorial-light"
        assert app.screen.styles.background.hex.upper() == LIGHT_PALETTE["bg"]
        # Top and bottom bars share a tint that sets them off from the page.
        for bar in ("#filters", "#footer"):
            tint = app.query_one(bar).styles.background.hex.upper()
            assert tint == LIGHT_PALETTE["chrome"] != LIGHT_PALETTE["bg"]
        listing = app.query_one(OptionList)
        styles = {
            str(span.style)
            for index in range(listing.option_count)
            for span in cast(Text, listing.get_option_at_index(index).prompt).spans
        }
        assert {LIGHT_PALETTE["link"], LIGHT_PALETTE["good"]} <= styles
        status = cast(Text, app.query_one("#status", Static).content)
        assert LIGHT_PALETTE["good"] in {str(span.style) for span in status.spans}
        # A manual theme pick holds until the clock's choice changes again.
        app.theme = "editorial"
        app.apply_clock_theme()
        assert app.theme == "editorial"


@pytest.mark.parametrize("zoom", [False, True])
async def test_space_pages_tldr_from_either_view(zoom: bool) -> None:
    """Space pages the TLDR down whether the list or the TLDR has focus, and
    leaves the selected headline alone."""
    app = Reader(api=EditorialServer().api())
    async with app.run_test(size=(120, 20)) as pilot:
        await settle(pilot)
        listing = app.query_one(OptionList)
        summary = app.query_one(Markdown)
        if zoom:
            await pilot.press("enter")
            await pilot.pause()
        assert summary.has_focus == zoom and listing.has_focus != zoom
        assert summary.max_scroll_y > summary.scrollable_content_region.height
        selected = app.selected()
        await pilot.press("space")
        await pilot.pause()
        first = summary.scroll_y
        assert first == summary.scrollable_content_region.height
        await pilot.press("space")
        await pilot.pause()
        assert summary.scroll_y == min(2 * first, summary.max_scroll_y)
        assert app.selected() == selected


async def test_a_slow_summary_counts_the_wait(tmp_path: Path) -> None:
    fake = FakeServer()
    fake.delay_summary = 2.5  # story 1, the first selected
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        summary = app.query_one("#summary", Markdown)
        for _ in range(100):
            if (
                summary._markdown.startswith("Loading summary… ")
                and summary._markdown[-1] == "s"
            ):
                break
            await pilot.pause(0.05)
        assert summary._markdown[len("Loading summary… ") : -1].isdigit()
        await settle(pilot)
        assert summary._markdown == "# Summary 1"


async def test_short_pane_keeps_footer_and_heading_compact() -> None:
    """A short pane (a small tmux split) reads the summary, not chrome: one
    footer row with "? help", and no "Because you upvoted" line. Both come
    back when the pane grows."""
    fake = FakeServer()
    fake.feed.stories[0] = replace(fake.feed.stories[0], best_match_title="Earlier")
    app = Reader(api=fake.api())
    async with app.run_test(size=(64, 23)) as pilot:
        await settle(pilot)
        await pilot.press("enter")
        await pilot.pause()
        status = app.query_one("#status").region
        hints = app.query_one("#shortcuts", Static)
        assert str(hints.content) == "? help"
        assert status.height == hints.region.height == 1
        assert status.y == hints.region.y and status.right <= hints.region.x
        heading = app.query_one("#story-heading", Static)
        assert "Because you upvoted" not in str(heading.content)
        # A long message is cut beside the hint, never stacked.
        app.status("Connection failed. " * 10, error=True)
        await pilot.pause()
        assert app.query_one("#status").region.y == hints.region.y
        await pilot.resize_terminal(64, 40)
        await pilot.pause()
        assert "Because you upvoted: Earlier" in str(heading.content)
        # The error keeps the row; the counts line brings the keys back.
        assert str(hints.content) == "? help"
        app.status_mode = "context"
        app.context_status()
        assert "Enter/Esc back" in str(hints.content)
