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
        # The selected window reloads; a new version also re-prefetches its
        # neighbours (1w's are 1d and 1m) in the background.
        fetches = fake.feed_requests()
        assert fetches.count("1w") == (0 if version == 2 else 1)
        assert sorted(w for w in fetches if w != "1w") == (
            [] if version == 2 else ["1d", "1m"]
        )
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


async def test_counts_version_refetches_counts_and_keeps_the_open_summary() -> None:
    """Between decks the server refreshes hot threads' counts: a new counts
    version refetches the selected window and its neighbors, shows the counts, keeps the open
    summary, and forgets kept summaries of other stories that gained
    comments so reopening asks the server again."""
    fake = FakeServer()
    fake.feed = sample_feed(2, 2)
    fake.counts_version = 5
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await app.poll_feed_version()  # Records the baseline only.
        await settle(pilot)
        open_summary = app.summaries[1]
        app.summaries[2] = "# Summary 2"
        fake.requests.clear()
        await app.poll_feed_version()
        assert not fake.feed_requests()
        fake.feed.stories[0] = replace(fake.feed.stories[0], points=444, comments=218)
        fake.feed.stories[1] = replace(fake.feed.stories[1], comments=60)
        fake.counts_version = 6
        await app.poll_feed_version()
        await settle(pilot)
        fetches = fake.feed_requests()
        assert fetches.count("1w") == 1
        assert sorted(window for window in fetches if window != "1w") == ["1d", "1m"]
        heading = str(app.query_one("#story-heading", Static).content)
        assert "▲ 444" in heading and "💬 218" in heading
        assert app.summaries[1] == open_summary
        assert "Summary 1" in app.query_one("#summary", Markdown).source
        assert 2 not in app.summaries


async def test_generated_summary_counts_update_the_row_and_heading() -> None:
    """r regenerates the summary from freshly fetched comments; the counts
    that came with it show at once instead of at the next deck."""

    class CountingServer(FakeServer):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/tldr-detail"):
                self.requests.append(request)
                story_id = json.loads(request.content)["story_id"]
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "tldr": f"# Fresh {story_id}",
                        "points": 444,
                        "comments": 218,
                    },
                )
            return await super().__call__(request)

    fake = CountingServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        await pilot.press("r")
        await settle(pilot)
        selected = app.selected()
        assert selected is not None and (selected.points, selected.comments) == (
            444,
            218,
        )
        heading = str(app.query_one("#story-heading", Static).content)
        assert "▲ 444" in heading and "💬 218" in heading
        row = app.query_one(OptionList).get_option(str(selected.id)).prompt
        assert "444" in str(row) and "218" in str(row)
        assert app.feed is not None
        assert {(s.id, s.points) for s in app.feed.stories} >= {(selected.id, 444)}
