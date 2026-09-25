"""Prefetch: speculative summaries ahead of the selection, cached for revisits."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from dataclasses import replace

import httpx
import pytest
from textual.pilot import Pilot
from textual.widgets import Markdown, OptionList

from hn_rerank.app import Reader

from .test_client import FakeServer, sample_feed


class PrefetchServer(FakeServer):
    """Records summary taps and can force stale or rate-limited responses."""

    def __init__(self) -> None:
        super().__init__()
        self.feed = replace(sample_feed(), orders={"recommended:recent": [1, 2, 3]})
        self.summary_ids: list[int] = []
        self.stale: set[int] = set()
        self.rate_limited: set[int] = set()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if (
            request.url.path.endswith("/api/tldr-detail")
            or "/api/tldr-cache/" in request.url.path
        ):
            story_id = (
                int(request.url.path.rsplit("/", 1)[1])
                if "/api/tldr-cache/" in request.url.path
                else json.loads(request.content)["story_id"]
            )
            self.summary_ids.append(story_id)
            if story_id in self.rate_limited:
                return httpx.Response(429, headers={"Retry-After": "30"})
            if story_id in self.stale:
                return httpx.Response(
                    200,
                    json={"ok": True, "tldr": f"# Stale {story_id}", "stale": True},
                )
        return await super().__call__(request)


async def wait_for(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    deadline = asyncio.get_running_loop().time() + 3.0
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not reached in time")
        await pilot.pause(0.05)


async def test_prefetch_warms_next_stories_then_serves_from_cache() -> None:
    fake = PrefetchServer()
    app = Reader(api=fake.api(), prefetch=2)
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: app.summaries.keys() >= {2, 3})
        assert fake.summary_ids[:3] == [1, 2, 3]  # active tap first, then in order
        generation = [
            r for r in fake.requests if r.url.path.endswith("/api/tldr-detail")
        ]
        assert len(generation) == 1
        assert json.loads(generation[0].content)["story_id"] == 1
        assert app.summaries[2] == "# Summary 2"
        # Revisiting a prefetched story renders from cache without another tap.
        app.query_one(OptionList).focus()
        await pilot.press("j")
        await pilot.pause()
        assert "Summary 2" in app.query_one(Markdown)._markdown
        assert fake.summary_ids.count(2) == 1


async def test_cache_miss_never_generates_until_selected() -> None:
    class MissServer(PrefetchServer):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if "/api/tldr-cache/" in request.url.path:
                self.requests.append(request)
                return httpx.Response(204)
            return await super().__call__(request)

    fake = MissServer()
    app = Reader(api=fake.api(), prefetch_generate=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: 2 in app.prefetch_retry_at)
        assert fake.summary_ids == [1]
        assert 2 not in app.summaries
        app.rebuild()
        await pilot.pause(0.1)
        assert (
            len([r for r in fake.requests if r.url.path.endswith("/tldr-cache/2")]) == 1
        )
        app.query_one(OptionList).focus()
        await pilot.press("j")
        await wait_for(pilot, lambda: 2 in app.summaries)
        assert fake.summary_ids == [1, 2]


async def test_prefetch_zero_disables_speculation() -> None:
    fake = PrefetchServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: app.summaries.get(1) == "# Summary 1")
        await pilot.pause(0.3)
        assert fake.summary_ids == [1]
        assert not app.prefetch_queue


async def test_prefetch_stops_on_rate_limit() -> None:
    fake = PrefetchServer()
    fake.rate_limited = {2}
    app = Reader(api=fake.api(), prefetch=2)
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: 2 in fake.summary_ids)
        await pilot.pause(0.3)
        before = list(fake.summary_ids)
        app.schedule_prefetch()
        await pilot.pause(0.1)
        assert fake.summary_ids == before  # no new work during cooldown
        assert 2 not in app.summaries  # nothing provisional cached
        assert app.prefetch_cooldown_until > 0


async def test_stale_summaries_are_shown_but_not_cached() -> None:
    fake = PrefetchServer()
    fake.stale = {2}
    app = Reader(api=fake.api(), prefetch=2)
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: fake.summary_ids.count(2) == 1)
        await wait_for(pilot, lambda: fake.summary_ids.count(3) == 1)
        assert 2 not in app.summaries  # provisional response, retry later
        app.rebuild()  # rebuilds must not re-queue a provisional story
        await pilot.pause(0.2)
        assert fake.summary_ids.count(2) == 1
        app.query_one(OptionList).focus()
        await pilot.press("j")
        await wait_for(pilot, lambda: fake.summary_ids.count(2) == 2)
        assert "Stale 2" in app.query_one(Markdown)._markdown


async def test_refresh_clears_the_summary_cache() -> None:
    fake = PrefetchServer()
    app = Reader(api=fake.api(), prefetch=2)
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: app.summaries.keys() >= {2, 3})
        taps_before = fake.summary_ids.count(1)
        app.action_refresh()
        await wait_for(pilot, lambda: fake.summary_ids.count(1) > taps_before)
        assert "Summary 1" in app.query_one(Markdown)._markdown


def test_cli_prefetch_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import hn_rerank.__main__ as main_module

    captured: dict[str, int] = {}

    class FakeReader:
        def __init__(
            self,
            server: str | None = None,
            prefetch: int = 2,
            prefetch_generate: int = 3,
        ) -> None:
            captured["prefetch"] = prefetch

        def run(self) -> None:
            pass

    monkeypatch.setattr(main_module, "Reader", FakeReader)
    monkeypatch.setattr(sys, "argv", ["hn-rerank", "--prefetch", "5"])
    main_module.main()
    assert captured["prefetch"] == 5
    monkeypatch.setattr(sys, "argv", ["hn-rerank", "--prefetch", "-1"])
    with pytest.raises(SystemExit):
        main_module.main()


class NavigationServer(PrefetchServer):
    """Uncached navigation targets plus a generation gate for catch-up races."""

    def __init__(self) -> None:
        super().__init__()
        base = self.feed.stories[0]
        self.feed = replace(
            self.feed,
            stories=[replace(base, id=i, title=f"Story {i}") for i in range(1, 31)],
            orders={
                "recommended:recent": list(range(1, 26)),
                "popular:recent": [26, 27],
                "explore:recent": [28, 29],
                "date:recent": [30],
            },
        )
        self.generated: list[int] = []
        self.cached: list[int] = []
        self.blocked: set[int] = set()
        self.release = asyncio.Event()
        self.active = 0
        self.peak = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if "/api/tldr-cache/" in request.url.path:
            self.requests.append(request)
            sid = int(request.url.path.rsplit("/", 1)[1])
            self.cached.append(sid)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.03)
                return httpx.Response(204)
            finally:
                self.active -= 1
        if request.url.path.endswith("/api/tldr-detail"):
            sid = json.loads(request.content)["story_id"]
            self.generated.append(sid)
            if sid in self.blocked:
                await self.release.wait()
        return await super().__call__(request)


async def test_navigation_prefetch_generates_nearby_and_other_sorts() -> None:
    fake = NavigationServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(
            pilot, lambda: app.summaries.keys() >= {1, 2, 3, 4, 26, 27, 28, 29, 30}
        )
        await wait_for(pilot, lambda: 12 in fake.cached)
        assert 5 not in fake.generated  # deeper cache misses wait until nearby
        assert 13 not in fake.cached  # past the 12-story view cap
        assert 1 < fake.peak <= 4
        await pilot.press("j")
        await wait_for(pilot, lambda: 5 in app.summaries)  # rolling window refills
        await pilot.press("k")
        assert "Summary 1" in app.query_one(Markdown)._markdown
        await pilot.press("s")
        await pilot.pause()
        assert "Summary 26" in app.query_one(Markdown)._markdown
        assert fake.generated.count(26) == 1
        await pilot.press("1")
        await wait_for(
            pilot, lambda: (story := app.selected()) is not None and story.id == 27
        )
        assert "Summary 27" in app.query_one(Markdown)._markdown
        assert fake.generated.count(27) == 1


async def test_selecting_inflight_generation_reuses_request() -> None:
    fake = NavigationServer()
    fake.blocked = {2}
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: 2 in fake.generated)
        await pilot.press("j")
        await pilot.pause(0.4)
        assert fake.generated.count(2) == 1
        fake.release.set()
        await wait_for(pilot, lambda: "Summary 2" in app.query_one(Markdown)._markdown)
        assert fake.generated.count(2) == 1


async def test_refresh_cancels_prefetch_generation() -> None:
    fake = NavigationServer()
    fake.blocked = {2, 3, 4}
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: {2, 3, 4} <= set(fake.generated))
        old = list(app.prefetch_requests.values())
        app.action_refresh()
        await pilot.pause(0.2)
        assert all(task.done() for task in old)
        fake.release.set()
        await wait_for(pilot, lambda: 2 in app.summaries)


async def test_prefetch_runs_while_selected_story_generates() -> None:
    fake = NavigationServer()
    fake.blocked = {1}
    app = Reader(api=fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await wait_for(pilot, lambda: 2 in app.summaries)
        assert 1 not in app.summaries
        fake.release.set()
        await wait_for(pilot, lambda: 1 in app.summaries)
