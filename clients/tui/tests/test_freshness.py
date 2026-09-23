from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
from textual.widgets import Static

from hn_rerank.app import Reader
from tests.test_client import FakeServer, sample_feed


@pytest.mark.parametrize("version", [0, 2, 3])
async def test_passive_poll_only_fetches_changed_versions(version: int) -> None:
    fake = FakeServer()
    fake.feed = sample_feed(2, 2)
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.5)
        await pilot.press("j")
        await pilot.pause()
        fake.requests.clear()
        fake.feed = sample_feed(version, version)
        fake.feed.stories[1] = replace(fake.feed.stories[1], comments=77)
        await app.poll_feed_version()
        await pilot.pause(0.5)
        fetches = [r for r in fake.requests if r.url.path.endswith("/api/feed")]
        assert len(fetches) == (0 if version == 2 else 1)
        selected = app.selected()
        assert selected is not None
        assert selected.id == 2
        if version != 2:
            assert "· 77" in str(app.query_one("#story-heading", Static).content)


@pytest.mark.parametrize("state", ["pending", "reading", "help_open", "setting_up"])
async def test_passive_poll_defers_during_interaction(state: str) -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        fake.requests.clear()
        setattr(app, state, True)
        await app.poll_feed_version()
        assert not fake.requests


async def test_passive_poll_failure_keeps_existing_deck() -> None:
    class OfflineProbe(FakeServer):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/ranking-ready"):
                raise httpx.ReadError("offline", request=request)
            return await super().__call__(request)

    fake = OfflineProbe()
    app = Reader(api=fake.api())
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        original = app.feed
        status = str(app.query_one("#status", Static).content)
        await app.poll_feed_version()
        assert app.feed is original
        assert str(app.query_one("#status", Static).content) == status
