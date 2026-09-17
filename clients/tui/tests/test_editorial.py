from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from textual.widgets import Input, Markdown, OptionList, Select, Static, Tabs

from hn_rerank.app import Reader, Setup

from .test_client import FakeServer


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
