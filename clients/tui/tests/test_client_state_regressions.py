"""Small state regressions using only the existing HTTP mock transport."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from textual.widgets import OptionList, Select

from hn_rerank.app import Reader

from ._settle import settle
from .test_client import FakeServer, sample_feed


class ReaderStateMismatch(AssertionError):
    """Reader state diverges from acknowledged actions or current metadata."""


@pytest.mark.parametrize("failed_attempt", [1, 2])
async def test_later_tui_vote_survives_an_earlier_failed_vote(
    failed_attempt: int,
) -> None:
    """An up/undo/down chain ends down even if its initial up fails."""

    class FirstVoteFails(FakeServer):
        def __init__(self) -> None:
            super().__init__()
            self.vote_attempts = 0
            self.first_vote = asyncio.Event()
            self.release_first_vote = asyncio.Event()
            self.saved_action: str | None = None

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if not request.url.path.endswith("/api/feedback"):
                return await super().__call__(request)
            self.requests.append(request)
            self.vote_attempts += 1
            if self.vote_attempts == 1:
                self.first_vote.set()
                await self.release_first_vote.wait()
            if self.vote_attempts == failed_attempt:
                raise httpx.ReadError("first vote failed", request=request)
            self.saved_action = json.loads(request.content)["action"]
            return httpx.Response(
                200, json={"ok": True, "target_version": self.vote_attempts}
            )

    fake = FirstVoteFails()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await asyncio.wait_for(fake.first_vote.wait(), timeout=2)
        app.action_undo()
        app.action_vote("down")
        assert 1 in app.rated
        fake.release_first_vote.set()
        await settle(pilot)
        assert fake.vote_attempts == 3
        assert fake.saved_action == "down"
        observed = (
            1 in app.rated,
            1 in [story.id for story in app.stories],
            [story.id for story in app.history],
        )
        if observed != (True, False, [1]):
            raise ReaderStateMismatch(f"Acknowledged final downvote lost: {observed!r}")


async def test_counts_refresh_covers_a_prefetched_window_when_selected() -> None:
    """A global counts change must not leave neighboring cached feeds stale."""
    fake = FakeServer()
    fake.feed = sample_feed(2, 2)
    fake.window_feeds["1d"] = replace(
        fake.feed, window="1d", stories=list(fake.feed.stories)
    )
    fake.counts_version = 5
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await app.poll_feed_version()
        await settle(pilot)
        assert "1d" in app.feeds  # The neighboring window was prefetched.
        fresh_story = replace(fake.feed.stories[0], points=444, comments=218)
        fake.feed.stories[0] = fresh_story
        fake.window_feeds["1d"].stories[0] = fresh_story
        fake.counts_version = 6
        await app.poll_feed_version()
        await settle(pilot)
        selected = app.selected()
        assert selected is not None
        assert (selected.points, selected.comments) == (444, 218)
        app.query_one("#window", Select).value = "1d"
        await settle(pilot)
        await app.poll_feed_version()
        await settle(pilot)
        selected = app.selected()
        assert selected is not None
        if (selected.points, selected.comments) != (444, 218):
            raise ReaderStateMismatch(
                f"Prefetched window restored stale counts: {selected.points, selected.comments}"
            )
