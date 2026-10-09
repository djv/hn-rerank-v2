"""Bounded prototype comparison: auto-refresh TUI sections on s/h/l.

Real Reader under Textual Pilot, reusing FakeServer with a 10-story deck and a
hold gate on 1w feed answers (cached synthetic summaries only; no DB/ranking):

  A - invalidate the departed sort, then GET the latest *existing* feed on return.
  B - explicit new ML rerank on leaving (static check: no client/server operation).
  C - GET the latest existing feed in the background on leaving, updating the
      shared window feed so a return is fresh.

Repo root:  uv run pytest docs/evaluations/section-refresh-20261009/ -q
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import replace
from typing import Any

import httpx
import pytest
from textual.widgets import Markdown, OptionList, Select

from clients.tui.src.hn_rerank.app import VIEW_LIMIT, Reader
from clients.tui.tests._settle import settle
from clients.tui.tests.test_client import FakeServer, sample_feed

N = 10
V0 = {
    "recommended": list(range(1, N + 1)),
    "popular": list(range(N, 0, -1)),
    "explore": [2, 4, 6, 8, 10, 1, 3, 5, 7, 9],
}


class HeldServer(FakeServer):
    def __init__(self) -> None:
        super().__init__()
        base = sample_feed().stories[0]
        stories = [
            replace(
                base,
                id=i,
                title=f"Story {i}",
                points=100 + i,
                time=i,
                rank_score=float(N - i),
            )
            for i in range(1, N + 1)
        ]
        self.feed = replace(sample_feed(), stories=stories, orders=dict(V0))
        self.hold = False
        self.gate = asyncio.Event()
        self.gate.set()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if (
            self.hold
            and request.url.path.endswith("/api/feed")
            and request.url.params.get("window", "") == "1w"
        ):
            await self.gate.wait()
        return await super().__call__(request)

    def ones(self) -> int:
        return self.feed_requests().count("1w")

    def forced(self) -> int:
        return sum(
            1
            for r in self.requests
            if r.url.path.endswith("/api/tldr-detail")
            and json.loads(r.content).get("force_refresh")
        )


def _bump(server: HeldServer) -> None:
    feed = server.feed
    stories = [replace(s, points=s.points + 100) for s in feed.stories]
    server.feed = replace(
        feed,
        version=1,
        target_version=1,
        stories=stories,
        orders={k: v[::-1] for k, v in feed.orders.items()},
    )


class ProtoReader(Reader):
    """Real Reader plus a sort-refresh policy; summaries/force_refresh untouched."""

    policy: str = "base"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.invalidated: set[str] = set()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value != event.select.value:
            return
        sort_change = getattr(event.select, "id", None) == "sort"
        departed = (
            self.view_key.split(":")[-1] if (sort_change and self.view_key) else None
        )
        if self.policy == "C" and sort_change and self.feed is not None:
            super().on_select_changed(event)
            self.refresh_feed(announce=False)
            return
        if self.policy == "A" and sort_change and departed:
            self.invalidated.add(departed)
        super().on_select_changed(event)
        if self.policy == "A" and sort_change and str(event.value) in self.invalidated:
            self.invalidated.discard(str(event.value))
            self.refresh_feed(announce=False)


def shown(app: Reader) -> list[int]:
    return [s.id for s in app.stories]


def check_consistent(app: Reader, server: HeldServer) -> None:
    assert len(app.stories) <= VIEW_LIMIT
    sel = app.selected()
    assert sel is not None and sel.id == app.stories[0].id
    assert f"Summary {sel.id}" in app.query_one(Markdown)._markdown
    assert server.forced() == 0


async def test_current_sort_switch_is_local_and_capped(tmp_path) -> None:
    """s/h/l today: one shared window feed, local rebuilds, 8-story cap, no GET."""
    server = HeldServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        assert shown(app) == V0["recommended"][:8]
        server.requests.clear()
        await pilot.press("s")
        await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == "popular"
        assert shown(app) == V0["popular"][:8]
        check_consistent(app, server)
        await pilot.press("s")
        await settle(pilot)
        assert len(app.stories) == 8 and set(shown(app)) <= set(V0["explore"])
        check_consistent(app, server)
        assert server.ones() == 0


def test_B_has_no_client_rerank_operation() -> None:
    from clients.tui.src.hn_rerank import api as api_mod
    from clients.tui.src.hn_rerank import app as app_mod

    api_src = inspect.getsource(api_mod.API).lower()
    assert "feedback" in api_src  # votes: the only path yielding target_version
    for endpoint in ("api/feedback", "api/interaction", "api/tldr-detail"):
        assert endpoint in api_src  # the client's only POST endpoints
    assert "api/rerank" not in api_src and "api/regen" not in api_src
    assert "rerank(" not in api_src and "regen(" not in api_src
    reader_src = inspect.getsource(app_mod.Reader).lower()
    assert "api/rerank" not in reader_src and "api/regen" not in reader_src
    # "reranked deck"/"regenerat*" mentions left are comments and the summary-regen
    # path, not a ranking trigger reachable from a sort switch.


@pytest.mark.parametrize("policy", ["A", "C"])
async def test_leave_return_picks_up_new_deck(tmp_path, policy: str) -> None:
    server = HeldServer()
    app = ProtoReader(api=server.api(), config_path=tmp_path / "p.json")
    app.policy = policy
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        if policy == "A":
            await pilot.press("s")  # leave: no fetch, departed marked
            await settle(pilot)
            assert server.ones() == 0 and "recommended" in app.invalidated
            _bump(server)
            await pilot.press("h")  # return: one GET of the existing feed
            await settle(pilot)
            assert server.ones() == 1
        else:
            _bump(server)
            await pilot.press("s")  # leave: one background GET
            await settle(pilot)
            assert server.ones() == 1
            assert shown(app) == V0["popular"][::-1][:8]
            await pilot.press("h")  # return: fresh already, but leaving pop
            await settle(pilot)  # fires one more GET: C fetches every switch
            assert server.ones() == 2
        assert app.feed is not None and app.feed.version == 1
        assert shown(app) == V0["recommended"][::-1][:8]
        sel = app.selected()
        assert sel is not None and sel.points == 200 + shown(app)[0]  # new counts
        check_consistent(app, server)


@pytest.mark.parametrize("policy", ["A", "C"])
async def test_held_feed_keeps_old_deck_until_release(tmp_path, policy: str) -> None:
    server = HeldServer()
    app = ProtoReader(api=server.api(), config_path=tmp_path / "p.json")
    app.policy = policy
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.hold = True
        server.gate = asyncio.Event()
        if policy == "A":
            await pilot.press("s")
            await settle(pilot)
            _bump(server)
            await pilot.press("h")  # return starts the held GET
        else:
            _bump(server)
            await pilot.press("s")  # leave starts the held background GET
        await pilot.pause(0.3)
        assert app.feed is not None and app.feed.version == 0  # old deck stays
        check_consistent(app, server)
        old = shown(app)
        server.gate.set()  # same held fixture for A and C
        await settle(pilot)
        assert app.feed is not None and app.feed.version == 1
        assert shown(app) != old
        check_consistent(app, server)
        server.hold = False


@pytest.mark.parametrize("policy", ["A", "C"])
async def test_rapid_sort_cycling_settles(tmp_path, policy: str) -> None:
    server = HeldServer()
    app = ProtoReader(api=server.api(), config_path=tmp_path / "p.json")
    app.policy = policy
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        for _ in range(4):
            app.action_cycle_sort()  # rapid s s s s, no settle between
        await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == "popular"
        assert shown(app) == V0["popular"][:8]
        check_consistent(app, server)
        n = server.ones()
        assert n <= (2 if policy == "A" else 4)  # A refetches invalidated returns only
        await settle(pilot)
        assert server.ones() == n  # no self-sustaining echo
