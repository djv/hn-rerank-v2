from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx
import pytest
from rich.text import Text
from textual.widgets import Button, Input, Markdown, OptionList, Select, Static, Tabs

from hn_rerank.app import EMPTY_NOTICE, Reader, Setup, story_age, story_metadata
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
async def test_editorial_filters_reading_and_resize(width: int) -> None:
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
        assert app.query_one("#sort-tabs", Tabs).display == (width >= 100)
        assert app.query_one("#sort", Select).display == (width < 100)
        assert "example.org" in str(app.query_one("#story-heading", Static).content)
        assert listing.highlighted == 0
        if width >= 100:
            await pilot.click("#sort-popular")
        else:
            app.query_one("#sort", Select).value = "popular"
        await pilot.pause()
        assert [s.id for s in app.stories] == [1]
        assert app.query_one("#sort-tabs", Tabs).active == "sort-popular"
        listing.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#reading-pane").has_pseudo_class("focus-within")
        summary = app.query_one(Markdown)
        await pilot.press("j", "j", "j")
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
        await pilot.press("escape")
        assert not app.query_one("#reading-pane").has_pseudo_class("focus-within")
        assert listing.has_focus and listing.display
        await pilot.press("enter")
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
