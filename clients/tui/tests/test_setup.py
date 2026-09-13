from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from textual.widgets import Input, OptionList

import hn_rerank.app as app_module
from hn_rerank.api import API, Profile, load_profile, save_profile
from hn_rerank.app import Reader, Setup
from test_client import FakeServer


async def test_import_validates_then_persists_and_relaunches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeServer()
    monkeypatch.setattr(
        app_module,
        "API",
        lambda server, token=None: API(server, token, httpx.MockTransport(fake)),
    )
    path = tmp_path / "profile.json"
    app = Reader(config_path=path)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, Setup)
        app.screen.query_one("#link", Input).value = "https://example.org/hn/u/test"
        await pilot.click("#import")
        for _ in range(100):
            if app.feed is not None:
                break
            await pilot.pause(0.1)
        assert load_profile(path) == Profile("https://example.org/hn/", "test")
        assert app.feed is not None
    app = Reader(config_path=path)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert not isinstance(app.screen, Setup)
        assert app.feed is not None


async def test_invalid_saved_profile_returns_to_setup_without_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(401)

    monkeypatch.setattr(
        app_module,
        "API",
        lambda server, token=None: API(server, token, httpx.MockTransport(handler)),
    )
    path = tmp_path / "profile.json"
    save_profile(Profile("https://example.org/hn/", "invalid"), path)
    app = Reader(config_path=path)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert isinstance(app.screen, Setup)
        assert calls == ["/hn/api/user"]
        assert load_profile(path) == Profile("https://example.org/hn/", "invalid")


async def test_server_override_never_sends_saved_credential(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    save_profile(Profile("https://example.org/hn/", "test"), path)
    app = Reader(server="https://other.example/hn/", config_path=path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, Setup)
        assert app.api is None


async def test_undo_restores_story_during_stale_refresh() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.2)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await pilot.pause(0.2)
        for order in fake.feed.orders.values():
            order[:] = [sid for sid in order if sid != 1]
        fake.feed.stories[:] = [story for story in fake.feed.stories if story.id != 1]
        fake.feed = type(fake.feed)(
            1,
            fake.feed.stories,
            fake.feed.orders,
            fake.feed.feedback_counts,
            0,
            1,
            False,
        )
        app.action_undo()
        await pilot.pause(0.3)
        selected = app.selected()
        assert selected and selected.id == 1
        assert [story.id for story in app.stories] == [1, 2]


async def test_profile_setup_cancels_pending_vote_state() -> None:
    fake = FakeServer()
    fake.delay_vote = 0.4
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.2)
        app.action_vote("up")
        await pilot.pause(0.05)
        app.setup("Profile changed")
        await pilot.pause(0.5)
        assert isinstance(app.screen, Setup)
        assert not app.pending and not app.rated and not app.history
