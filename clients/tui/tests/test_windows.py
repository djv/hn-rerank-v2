"""Time windows: switching, background neighbour prefetch, late responses."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from textual.pilot import Pilot
from textual.widgets import OptionList, Select

from hn_rerank.app import Reader
from hn_rerank.models import Feed, Window

from .test_client import FakeServer, sample_feed
from ._settle import settle


def window_feed(window: Window, ids: list[int]) -> Feed:
    base = sample_feed().stories[0]
    return replace(
        sample_feed(window=window),
        stories=[replace(base, id=i, title=f"Story {i}") for i in ids],
        orders={"recommended": list(ids)},
    )


def shown(app: Reader) -> list[int]:
    return [story.id for story in app.stories]


async def until(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    """Wait for *predicate* while a held request keeps the app busy."""
    deadline = asyncio.get_running_loop().time() + 5.0
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await pilot.pause(0.05)


async def test_select_and_d_key_switch_windows() -> None:
    fake = FakeServer()
    fake.window_feeds = {
        "12h": window_feed("12h", [10]),
        "1m": window_feed("1m", [20, 21]),
    }
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert app.selected_window() == "1w" and shown(app) == [1, 2]
        # 1w's neighbours load in the background, never their summaries.
        assert sorted(fake.feed_requests()) == ["1d", "1m", "1w"]
        summaries = [
            r
            for r in fake.requests
            if "/api/tldr" in r.url.path
            and (
                "tldr-cache" in r.url.path
                and int(r.url.path.rsplit("/", 1)[1]) in {20, 21}
                or r.method == "POST"
                and json.loads(r.content)["story_id"] in {20, 21}
            )
        ]
        assert not summaries
        fake.requests.clear()
        app.query_one(OptionList).focus()
        await pilot.press("d")  # 1w -> 1m, shown from the cache
        await settle(pilot)
        assert app.selected_window() == "1m" and shown(app) == [20, 21]
        assert app.feed is not None and app.feed.window == "1m"
        # Only 1m's other neighbour is new.
        assert fake.feed_requests() == ["archive"]
        fake.requests.clear()
        app.query_one("#window", Select).value = "12h"  # not cached: fetched
        await settle(pilot)
        assert shown(app) == [10]
        assert fake.feed_requests()[0] == "12h"
        await pilot.press("d")  # 12h -> 1d: an empty window, not widened
        await settle(pilot)
        assert app.selected_window() == "1d" and shown(app) == []
        for _ in range(4):
            await pilot.press("d")
        await settle(pilot)
        assert app.selected_window() == "12h"  # wraps after Archive
        assert shown(app) == [10]


class GatedServer(FakeServer):
    """Holds 1m feed answers until released; the answer is the deck as it
    was when the request arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        window = request.url.params.get("window", "1w")
        if request.url.path.endswith("/api/feed") and window != "1w":
            self.requests.append(request)
            answer = self.feed_for(window)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            try:
                if window == "1m":
                    await self.gate.wait()
                else:
                    await asyncio.sleep(0.05)
            finally:
                self.in_flight -= 1
            return httpx.Response(200, json=answer.to_dict())
        return await super().__call__(request)


async def test_neighbour_prefetch_is_serial_and_once_per_version() -> None:
    fake = GatedServer()
    fake.gate.set()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert sorted(fake.feed_requests()) == ["1d", "1m", "1w"]
        assert fake.peak == 1
        fake.requests.clear()
        app.reload(manual=False)  # same version: nothing refetched
        await settle(pilot)
        assert fake.feed_requests() == ["1w"]
        fake.requests.clear()
        fake.feed = sample_feed(1, 1)  # a new deck: its neighbours again
        app.reload(manual=False)
        await settle(pilot)
        assert sorted(fake.feed_requests()) == ["1d", "1m", "1w"]
        assert {w: f.version for w, f in app.feeds.items()} == {
            "1w": 1,
            "1d": 1,
            "1m": 1,
        }
        assert fake.peak == 1


async def test_late_neighbour_answer_never_restores_the_pre_vote_deck() -> None:
    fake = GatedServer()
    fake.window_feeds = {"1m": window_feed("1m", [1, 20])}
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await until(pilot, lambda: fake.feed_requests() == ["1w", "1m"])
        await until(pilot, lambda: shown(app) == [1, 2])
        # The 1m prefetch (version 0) is held; 1d waits behind it.
        await pilot.pause(0.2)
        assert fake.feed_requests() == ["1w", "1m"]
        app.query_one(OptionList).focus()
        app.action_vote("up")  # story 1
        await until(pilot, lambda: app.feed is not None and not app.feed.ready)
        # The reranked deck (version 1) lands and the poller loads it.
        fake.feed = replace(
            sample_feed(1, 1),
            stories=sample_feed().stories[1:],
            orders={"recommended": [2]},
        )
        fake.window_feeds = {"1m": window_feed("1m", [20, 22])}
        await app.poll_feed_version()
        await until(pilot, lambda: app.feed is not None and app.feed.version == 1)
        fake.gate.set()  # the version-0 1m answer arrives late
        await settle(pilot)
        assert app.feed is not None
        assert (app.feed.window, app.feed.version) == ("1w", 1)
        assert {f.version for f in app.feeds.values()} == {1}
        assert fake.peak == 1
        app.query_one("#window", Select).value = "1m"
        await settle(pilot)
        assert app.feed is not None and app.feed.version == 1
        assert shown(app) == [20, 22]


async def test_rapid_votes_and_undo_across_windows() -> None:
    fake = FakeServer()
    fake.delay_vote = 0.1
    fake.window_feeds = {"1m": window_feed("1m", [1, 20, 21])}
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.action_vote("up")  # story 1 in 1w
        await pilot.press("d")  # 1m, cached; story 1 stays hidden there too
        await pilot.pause()
        assert app.selected_window() == "1m" and shown(app) == [20, 21]
        app.action_vote("down")  # story 20 in 1m
        assert shown(app) == [21]
        app.action_undo()  # story 20 back, in 1m
        assert shown(app) == [20, 21]
        app.query_one("#window", Select).value = "1w"
        await pilot.pause()
        assert shown(app) == [2]
        app.action_undo()  # story 1 back, in 1w
        assert shown(app) == [1, 2]
        await settle(pilot)
        votes = [
            json.loads(r.content)
            for r in fake.requests
            if r.url.path.endswith("/api/feedback")
        ]
        assert votes == [
            {"story_id": 1, "action": "up"},
            {"story_id": 20, "action": "down"},
            {"story_id": 20, "action": "clear"},
            {"story_id": 1, "action": "clear"},
        ]
        assert not app.rated
        # The pending rerank marks whichever window is on screen.
        assert app.feed is not None and not app.feed.ready


async def test_last_window_opens_next_time(tmp_path: Path) -> None:
    """The window picked is saved and the next start opens on it."""
    window_file = tmp_path / "config" / "window"
    fake = FakeServer()
    fake.window_feeds = {"1m": window_feed("1m", [20, 21])}
    app = Reader(api=fake.api(), window_file=window_file)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert app.selected_window() == "1w"  # nothing saved yet
        assert not window_file.exists()
        await pilot.press("d")
        await settle(pilot)
        assert window_file.read_text(encoding="utf-8").strip() == "1m"

    fake = FakeServer()
    fake.window_feeds = {"1m": window_feed("1m", [20, 21])}
    app = Reader(api=fake.api(), window_file=window_file)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert app.selected_window() == "1m" and shown(app) == [20, 21]
        assert fake.feed_requests()[0] == "1m"  # no 1w detour first


@pytest.mark.parametrize("content", [b"", b"2w\n", b"\xff\xfe"])
async def test_unusable_saved_window_falls_back_to_default(
    tmp_path: Path, content: bytes
) -> None:
    window_file = tmp_path / "window"
    window_file.write_bytes(content)
    app = Reader(api=FakeServer().api(), window_file=window_file)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert app.selected_window() == "1w"
