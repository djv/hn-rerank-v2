from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from textual.widgets import Input, OptionList, Select

import hn_rerank.app as app_module
from hn_rerank.api import API, Profile, load_profile, save_profile
from hn_rerank.app import Reader, Setup
from hn_rerank.models import Feed

from .test_client import FakeServer


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
    app = Reader(config_path=path, server="https://example.org/hn/")
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
    app = Reader(config_path=path, server="https://example.org/hn/")
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert not isinstance(app.screen, Setup)
        assert app.feed is not None


async def test_token_uses_default_server_then_persists(
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
        app.screen.query_one("#token", Input).value = "test"
        await pilot.click("#use-token")
        for _ in range(100):
            if app.feed is not None:
                break
            await pilot.pause(0.1)
        from hn_rerank.app import DEFAULT_SERVER

        assert load_profile(path) == Profile(DEFAULT_SERVER, "test")
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
    app = Reader(config_path=path, server="https://example.org/hn/")
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


async def test_default_server_keeps_saved_profile_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeServer()
    monkeypatch.setattr(
        app_module,
        "API",
        lambda server, token=None: API(server, token, httpx.MockTransport(fake)),
    )
    path = tmp_path / "profile.json"
    save_profile(Profile("http://localhost:8000/", "test"), path)
    # No --server flag: the saved profile's server wins over DEFAULT_SERVER.
    app = Reader(config_path=path)
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert not isinstance(app.screen, Setup)
        assert app.api is not None and app.api.server == "http://localhost:8000/"


async def test_rejected_saved_profile_reconnects_on_its_own_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --server, setup after a rejected token stays on the saved
    profile's deployment instead of falling back to DEFAULT_SERVER."""
    fake = FakeServer()
    hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.headers.get("cookie") == "hn_token=revoked":
            return httpx.Response(401)
        return await fake(request)

    monkeypatch.setattr(
        app_module,
        "API",
        lambda server, token=None: API(server, token, httpx.MockTransport(handler)),
    )
    path = tmp_path / "profile.json"
    save_profile(Profile("https://mine.example/hn/", "revoked"), path)
    app = Reader(config_path=path)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(0.2)
        assert isinstance(app.screen, Setup)
        app.screen.query_one("#token", Input).value = "test"
        await pilot.click("#use-token")
        for _ in range(100):
            if app.feed is not None:
                break
            await pilot.pause(0.1)
        assert app.feed is not None
    assert set(hosts) == {"mine.example"}
    assert load_profile(path) == Profile("https://mine.example/hn/", "test")


async def test_undo_puts_story_back_only_where_the_server_listed_it() -> None:
    """Undo during a stale refresh restores the story to the views it was
    voted from, at the server's position: not newest in Date, and not into
    Recommended when the server had left it out."""
    fake = FakeServer()
    stories = [
        replace(story, memberships=["recent_mixed"], popular=True, explore=False)
        for story in fake.feed.stories
    ]
    original = {
        "recommended:recent": [2, 3],
        "popular:recent": [1, 2, 3],
        "date:recent": [3, 2, 1],
    }
    fake.feed = Feed(
        1,
        stories,
        {k: list(v) for k, v in original.items()},
        fake.feed.feedback_counts,
        0,
        0,
        True,
    )
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.2)
        app.query_one("#sort", Select).value = "popular"
        await pilot.pause(0.2)
        selected = app.selected()
        assert selected and selected.id == 1
        app.action_vote("up")
        await pilot.pause(0.2)
        remaining = [story for story in stories if story.id != 1]
        orders = {k: [sid for sid in v if sid != 1] for k, v in original.items()}
        fake.feed = Feed(1, remaining, orders, fake.feed.feedback_counts, 0, 1, False)
        app.action_undo()
        await pilot.pause(0.3)
        assert app.feed is not None
        assert {k: v for k, v in app.feed.orders.items() if v} == original
