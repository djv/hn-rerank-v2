from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from textual.widgets import Markdown, OptionList, Static

from hn_rerank.app import Reader
from tests.test_client import FakeServer, sample_feed
from ._settle import settle


@pytest.mark.parametrize("version", [0, 2, 3])
async def test_passive_poll_only_fetches_changed_versions(version: int) -> None:
    fake = FakeServer()
    fake.feed = sample_feed(2, 2)
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("j")
        await pilot.pause()
        fake.requests.clear()
        fake.feed = sample_feed(version, version)
        fake.feed.stories[1] = replace(fake.feed.stories[1], comments=77)
        await app.poll_feed_version()
        await settle(pilot)
        fetches = [r for r in fake.requests if r.url.path.endswith("/api/feed")]
        assert len(fetches) == (0 if version == 2 else 1)
        selected = app.selected()
        assert selected is not None
        assert selected.id == 2
        if version != 2:
            assert "· 💬 77" in str(app.query_one("#story-heading", Static).content)


@pytest.mark.parametrize("state", ["reading", "help_open", "setting_up"])
async def test_passive_poll_defers_during_interaction(state: str) -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test() as pilot:
        await settle(pilot)
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
        await settle(pilot)
        original = app.feed
        status = str(app.query_one("#status", Static).content)
        await app.poll_feed_version()
        assert app.feed is original
        assert str(app.query_one("#status", Static).content) == status


async def test_passive_version_change_leaves_the_open_summary_alone() -> None:
    """A new published version (4h regen, a vote from another device) must
    not blank or refetch the summary being read."""
    fake = FakeServer()
    fake.feed = sample_feed(2, 2)
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("j")
        await settle(pilot)
        assert app.summaries.get(2) == "# Summary 2"
        fake.requests.clear()
        fake.feed = sample_feed(3, 3)
        await app.poll_feed_version()
        await settle(pilot)
        assert app.feed is not None and app.feed.version == 3
        summarized = [
            int(r.url.path.rsplit("/", 1)[1])
            if "/api/tldr-cache/" in r.url.path
            else json.loads(r.content)["story_id"]
            for r in fake.requests
            if r.url.path.endswith("/api/tldr-detail")
            or "/api/tldr-cache/" in r.url.path
        ]
        assert 2 not in summarized
        assert "Summary 2" in app.query_one(Markdown)._markdown


async def test_poll_after_vote_waits_for_the_reranked_deck() -> None:
    """A vote marks the deck stale; the poller reloads once ranking lands."""
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await settle(pilot)
        assert app.feed is not None and not app.feed.ready
        assert app.feed.target_version == 1
        fake.feed = sample_feed(0, 1)  # still ranking
        fake.requests.clear()
        await app.poll_feed_version()
        await settle(pilot)
        assert not [r for r in fake.requests if r.url.path.endswith("/api/feed")]
        fake.feed = sample_feed(1, 1)  # reranked deck published
        await app.poll_feed_version()
        await settle(pilot)
        assert [r for r in fake.requests if r.url.path.endswith("/api/feed")]
        assert app.feed.ready and app.feed.version == 1
        assert 1 not in [s.id for s in app.stories]  # still voted


async def test_ranking_notice_clears_when_the_ready_deck_arrives() -> None:
    fake = FakeServer()
    fake.feed = sample_feed(0, 1)  # stale deck while ranking runs
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        status = app.query_one("#status", Static)
        assert "ranking updates" in str(status.content)
        fake.feed = sample_feed(1, 1)
        await app.poll_feed_version()
        await settle(pilot)
        assert app.feed is not None and app.feed.ready
        assert "ranking updates" not in str(status.content)
        assert "shown" in str(status.content)
