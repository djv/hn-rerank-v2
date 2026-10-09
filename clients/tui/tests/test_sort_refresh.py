"""Section background refresh, redefined r, and the stats-regeneration prompt.

Real production Reader under Textual Pilot against a controllable fake
server: arrival-counted held feed GETs, gated summary POSTs, live stats
overrides and known summary snapshots. No network, no provider, no DB.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from textual.widgets import Markdown, OptionList, Select, Static

from hn_rerank.app import Reader, StatsPrompt
from hn_rerank.models import Feed, FeedStory

from ._settle import settle
from .test_client import FakeServer, sample_feed


def _stories(base: int = 100) -> list[FeedStory]:
    template = sample_feed().stories[0]
    return [
        replace(
            template,
            id=i,
            title=f"Story {i}",
            article_url=f"https://example.org/{i}",
            comments_url=f"https://news.ycombinator.com/item?id={i}",
            points=base + i,
            comments=10,
            time=i,
            rank_score=float(10 - i),
        )
        for i in range(1, 7)
    ]


ORDERS = {
    "recommended": [1, 2, 3, 4, 5, 6],
    "popular": [6, 5, 4, 3, 2, 1],
    "explore": [2, 5, 1, 6, 3, 4],
}


def _deck(base: int, version: int, drop: tuple[int, ...] = ()) -> Feed:
    stories = [s for s in _stories(base) if s.id not in drop]
    ids = {s.id for s in stories}
    return Feed(
        2,
        "1w",
        stories,
        {k: [i for i in v if i in ids] for k, v in ORDERS.items()},
        {"up": 0, "neutral": 0, "down": 0},
        version,
        version,
        True,
    )


class RefreshServer(FakeServer):
    """FakeServer with a 6-story deck, START-counted held 1w GETs, gated
    summary POSTs, per-story live stats and known summary snapshots."""

    def __init__(self) -> None:
        super().__init__()
        self.feed = _deck(100, 0)
        self.arrivals = 0
        self.active = 0
        self.peak = 0
        self.hold_feed = False
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail_feed = False
        self.snapshot: int | None = 10
        self.text_version = 1
        self.live: dict[int, tuple[int, int]] = {}
        self.probe_calls: list[int] = []
        self.hold_summary = False
        self.summary_gate = asyncio.Event()
        self.summary_gate.set()
        self.forced_posts: list[int] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if (
            path.endswith("/api/feed")
            and request.url.params.get("window", "1w") == "1w"
        ):
            self.requests.append(request)
            self.arrivals += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if self.hold_feed:
                    await self.gate.wait()
                if self.fail_feed:
                    return httpx.Response(500, json={"error": "down"})
                return httpx.Response(200, json=self.feed_for("1w").to_dict())
            finally:
                self.active -= 1
        if path.endswith("/api/tldr-detail"):
            self.requests.append(request)
            body = json.loads(request.content)
            story_id = body["story_id"]
            if body.get("force_refresh"):
                self.forced_posts.append(story_id)
            if self.hold_summary:
                await self.summary_gate.wait()
            payload: dict[str, object] = {
                "ok": True,
                "tldr": f"# Summary {story_id} v{self.text_version}",
            }
            if self.snapshot is not None:
                payload["comments_summarized"] = self.snapshot
            return httpx.Response(200, json=payload)
        if "/api/tldr-cache/" in path:
            self.requests.append(request)
            story_id = int(path.rsplit("/", 1)[1])
            payload = {"tldr": f"# Summary {story_id} v{self.text_version}"}
            if self.snapshot is not None:
                payload["comments_summarized"] = self.snapshot
            return httpx.Response(200, json=payload)
        if path.endswith("/api/story-stats"):
            self.requests.append(request)
            story_id = int(request.url.params.get("story_id", "0"))
            story = next((s for s in self.feed.stories if s.id == story_id), None)
            if story is None:
                return httpx.Response(404, json={"error": "missing"})
            if story.source.startswith("rss_"):
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "story_id": story_id,
                        "points": story.points,
                        "points_live": False,
                        "comments": story.comments or 0,
                        "comments_live": False,
                        "reason": "unsupported_source",
                    },
                )
            self.probe_calls.append(story_id)
            points, comments = self.live.get(
                story_id, (story.points, story.comments or 0)
            )
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "story_id": story_id,
                    "points": points,
                    "points_live": True,
                    "comments": comments,
                    "comments_live": True,
                    "reason": "live_check",
                },
            )
        return await super().__call__(request)

    def bump(self, drop: tuple[int, ...] = ()) -> None:
        stories = [s for s in _stories(200) if s.id not in drop]
        ids = {s.id for s in stories}
        self.feed = Feed(
            2,
            "1w",
            stories,
            {k: [i for i in reversed(v) if i in ids] for k, v in ORDERS.items()},
            {"up": 0, "neutral": 0, "down": 0},
            1,
            1,
            True,
        )

    def forced(self) -> int:
        return len(self.forced_posts)


def shown(app: Reader) -> list[int]:
    return [s.id for s in app.stories]


async def wait_for(pilot: Any, predicate: Any, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await pilot.pause(0.05)


def modal_text(app: Reader) -> str:
    return " ".join(str(w.content) for w in app.screen.query(Static))


@pytest.mark.parametrize(
    ("key", "sort", "order", "kept"),
    [
        ("s", "popular", ORDERS["popular"][::-1], ORDERS["popular"][0]),
        ("l", "popular", ORDERS["popular"][::-1], ORDERS["popular"][0]),
        ("h", "explore", ORDERS["explore"][::-1], ORDERS["explore"][::-1][0]),
    ],
)
async def test_sort_key_background_refreshes_counts_and_orders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    sort: str,
    order: list[int],
    kept: int,
) -> None:
    server = RefreshServer()
    monkeypatch.setattr("random.shuffle", lambda order: order.reverse())
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        assert shown(app) == ORDERS["recommended"]
        server.bump()
        server.requests.clear()
        server.arrivals = 0
        await pilot.press(key)
        await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == sort
        assert server.arrivals == 1
        assert shown(app) == order  # explore is deterministically shuffled above
        selected = app.selected()
        # The background refresh preserves the open story by id (not the row
        # position) and applies the deck's new counts to it.
        assert selected is not None and selected.id == kept
        assert selected.points == 200 + kept
        assert f"Summary {kept}" in app.query_one(Markdown)._markdown
        assert server.forced() == 0


async def test_held_get_coalesces_rapid_keypresses(tmp_path: Path) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_feed = True
        server.gate = asyncio.Event()
        await pilot.press("s")
        await wait_for(pilot, lambda: server.arrivals == 1)
        await pilot.press("h")
        await pilot.press("s")
        await pilot.pause(0.3)
        assert server.arrivals == 1  # coalesced, never restarted per keypress
        assert server.peak == 1
        server.bump()
        server.gate.set()
        server.hold_feed = False
        await settle(pilot)
        assert server.arrivals == 2  # exactly one bounded follow-up
        assert shown(app) == ORDERS["popular"][::-1]  # s, h, s ends on popular v1
        assert app.feed is not None and app.feed.version == 1
        selected = app.selected()
        assert selected is not None
        assert f"Summary {selected.id}" in app.query_one(Markdown)._markdown
        assert server.forced() == 0


async def test_window_change_during_held_sort_get(tmp_path: Path) -> None:
    server = RefreshServer()
    base = sample_feed().stories[0]
    server.window_feeds["1m"] = Feed(
        2,
        "1m",
        [replace(base, id=i, title=f"Story {i}") for i in (20, 21)],
        {sort: [20, 21] for sort in ("recommended", "popular", "explore")},
        {"up": 0, "neutral": 0, "down": 0},
        0,
        0,
        True,
    )
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_feed = True
        server.gate = asyncio.Event()
        await pilot.press("s")  # held 1w GET starts
        await wait_for(pilot, lambda: server.arrivals == 1)
        await pilot.press("d")  # switch window while it is held
        await wait_for(pilot, lambda: app.feed is not None and app.feed.window == "1m")
        assert shown(app) == [20, 21]
        server.gate.set()
        server.hold_feed = False
        await settle(pilot)
        # The stale 1w reply never replaces the newer window's feed.
        assert app.feed is not None and app.feed.window == "1m"
        assert shown(app) == [20, 21]


async def test_profile_swap_drops_stale_reply_without_followup(
    tmp_path: Path,
) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_feed = True
        server.gate = asyncio.Event()
        await pilot.press("s")
        await wait_for(pilot, lambda: server.arrivals == 1)
        app.api = server.api()  # profile replaced mid-request
        server.gate.set()
        server.hold_feed = False
        await settle(pilot)
        assert server.arrivals == 1
        assert app.feed is not None and app.feed.version == 0
        assert shown(app) == ORDERS["popular"]


async def test_teardown_during_held_get_stays_quiet(tmp_path: Path) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_feed = True
        server.gate = asyncio.Event()
        await pilot.press("s")
        await wait_for(pilot, lambda: server.arrivals == 1)
    server.gate.set()
    await asyncio.sleep(0.5)
    assert server.arrivals == 1  # no follow-up after teardown


async def test_feed_error_keeps_deck_without_retry_loop(tmp_path: Path) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.fail_feed = True
        await pilot.press("s")
        await settle(pilot)
        assert shown(app) == ORDERS["popular"]  # local switch still applied
        assert server.arrivals == 1
        status = str(app.query_one("#status", Static).content)
        assert "✗" in status and "Press r to retry" not in status
        await settle(pilot)
        assert server.arrivals == 1


async def test_background_failure_names_retry_controls(tmp_path: Path) -> None:
    """A failed background refresh keeps the usable deck, claims no
    automatic retry (none is scheduled), and the next genuine sort
    change retries the fetch."""
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        assert shown(app) == ORDERS["recommended"]
        server.fail_feed = True
        server.arrivals = 0
        await pilot.press("s")
        await wait_for(pilot, lambda: server.arrivals == 1)
        await settle(pilot)
        status = str(app.query_one("#status", Static).content)
        assert "Background refresh failed" in status
        assert "automatically" not in status
        assert "Switch sort or time window to retry." in status
        assert shown(app) == ORDERS["popular"]  # local switch applied
        assert app.feed is not None  # usable deck stays on screen
        server.fail_feed = False
        server.arrivals = 0
        await pilot.press("h")
        await wait_for(pilot, lambda: server.arrivals == 1)
        await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == "recommended"
        assert shown(app) == ORDERS["recommended"]


async def test_r_regenerates_only_selected(tmp_path: Path) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        before = dict(app.summaries)
        assert len(before) > 1
        app.unavailable.add(6)
        app.rebuild(1)
        assert shown(app) == [1, 2, 3, 4, 5]
        server.requests.clear()
        server.arrivals = 0
        await pilot.press("r")
        await settle(pilot)
        assert not [r for r in server.requests if r.url.path.endswith("/api/feed")]
        assert app.unavailable == {6}
        assert shown(app) == [1, 2, 3, 4, 5]
        assert server.forced_posts == [1]
        assert app.summaries == before
        assert "Summary regenerated." in str(app.query_one("#status", Static).content)


async def test_repeated_r_coalesces_to_one_forced_post(tmp_path: Path) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_summary = True
        server.summary_gate = asyncio.Event()
        await pilot.press("r")
        await pilot.press("r")
        await pilot.pause(0.6)
        assert len(server.forced_posts) == 1
        server.summary_gate.set()
        server.hold_summary = False
        await settle(pilot)
        assert len(server.forced_posts) == 1


async def test_stats_growth_prompts_and_decline_dedupes(tmp_path: Path) -> None:
    server = RefreshServer()
    server.live = {1: (150, 25)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        assert "▲ 150" in str(app.query_one("#story-heading", Static).content)
        await pilot.press("r")
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        await wait_for(pilot, lambda: bool(app.screen.query("#stats-regen")))
        text = modal_text(app)
        assert "Story 1" in text and "10" in text and "25" in text
        before_posts = len(server.forced_posts)
        await pilot.press("escape")
        await settle(pilot)
        assert not isinstance(app.screen, StatsPrompt)
        assert len(server.forced_posts) == before_posts  # kept: no paid regen
        assert "Summary 1" in app.query_one(Markdown)._markdown
        await pilot.press("r")  # same observation must not nag again
        await settle(pilot)
        assert not isinstance(app.screen, StatsPrompt)


async def test_stats_accept_regenerates_selected_only(tmp_path: Path) -> None:
    server = RefreshServer()
    server.live = {1: (150, 25)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        await pilot.press("r")
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        server.text_version = 2
        server.snapshot = 25
        posts_before = len(server.forced_posts)
        await wait_for(pilot, lambda: bool(app.screen.query("#stats-regen")))
        await pilot.click("#stats-regen")
        await settle(pilot)
        assert not isinstance(app.screen, StatsPrompt)
        assert len(server.forced_posts) == posts_before + 1
        assert server.forced_posts[-1] == 1
        assert "Summary 1 v2" in app.query_one(Markdown)._markdown
        assert app._summary_snapshots.get(1) == 25


async def test_stats_regrowth_prompts_again(tmp_path: Path) -> None:
    server = RefreshServer()
    server.snapshot = 25
    server.live = {1: (150, 25)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        await pilot.press("r")
        await settle(pilot)
        assert not isinstance(app.screen, StatsPrompt)  # no growth over snapshot
        server.live = {1: (160, 40)}
        await pilot.press("r")
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        await wait_for(pilot, lambda: bool(app.screen.query("#stats-regen")))
        assert "40" in modal_text(app)


async def test_stats_stale_and_unsupported_skip_prompt(tmp_path: Path) -> None:
    server = RefreshServer()
    server.live = {1: (90, 5)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        await pilot.press("r")
        await settle(pilot)
        assert not isinstance(app.screen, StatsPrompt)
        assert "💬 5" in str(app.query_one("#story-heading", Static).content)


async def test_stats_unsupported_source_never_probes(tmp_path: Path) -> None:
    server = RefreshServer()
    base = sample_feed().stories[0]
    rss = replace(base, id=7, title="Story 7", source="rss_reddit_x")
    server.feed = Feed(
        2,
        "1w",
        [*_stories(), rss],
        {k: [*v, 7] for k, v in ORDERS.items()},
        {"up": 0, "neutral": 0, "down": 0},
        0,
        0,
        True,
    )
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        for _ in range(6):
            await pilot.press("j")
        await settle(pilot)
        selected = app.selected()
        assert selected is not None and selected.id == 7
        server.requests.clear()
        server.arrivals = 0
        server.probe_calls.clear()
        await pilot.press("r")
        await settle(pilot)
        assert 7 not in server.probe_calls
        assert not isinstance(app.screen, StatsPrompt)


async def test_stats_during_inflight_r_defers_to_returned_snapshot(
    tmp_path: Path,
) -> None:
    server = RefreshServer()
    server.live = {1: (150, 25)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_summary = True
        server.summary_gate = asyncio.Event()
        await pilot.press("r")
        await wait_for(pilot, lambda: len(server.forced_posts) == 1)
        await pilot.pause(0.6)  # fast stats result lands mid-regeneration
        assert not isinstance(app.screen, StatsPrompt)
        server.snapshot = 10
        server.summary_gate.set()
        server.hold_summary = False
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        await pilot.press("escape")
        await settle(pilot)


async def test_stats_returned_snapshot_covers_pending_growth(
    tmp_path: Path,
) -> None:
    server = RefreshServer()
    server.live = {1: (150, 25)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_summary = True
        server.summary_gate = asyncio.Event()
        await pilot.press("r")
        await wait_for(pilot, lambda: len(server.forced_posts) == 1)
        await pilot.pause(0.6)
        assert not isinstance(app.screen, StatsPrompt)
        server.snapshot = 30  # regeneration covers beyond the live count
        server.summary_gate.set()
        server.hold_summary = False
        await settle(pilot)
        await pilot.pause(0.5)
        assert not isinstance(app.screen, StatsPrompt)


async def test_stats_for_other_story_updates_silently(tmp_path: Path) -> None:
    server = RefreshServer()
    server.live = {3: (300, 33)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        selected = app.selected()
        assert selected is not None and selected.id == 1
        app.refresh_story_stats(3, app.selection_serial)
        await settle(pilot)
        assert not isinstance(app.screen, StatsPrompt)
        assert app.feed is not None
        row = next(s for s in app.feed.stories if s.id == 3)
        assert (row.points, row.comments) == (300, 33)


async def test_section_refresh_triggers_stats_check_without_r(
    tmp_path: Path,
) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        assert app._summary_snapshots.get(6) == 10
        server.live = {6: (160, 25)}
        server.requests.clear()
        server.arrivals = 0
        await pilot.press("s")  # background refresh, then tail stats check
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        assert server.forced() == 0  # prompt only: no implicit paid force
        await wait_for(pilot, lambda: bool(app.screen.query("#stats-regen")))
        text = modal_text(app)
        assert "Story 6" in text and "25" in text
        await pilot.press("escape")
        await settle(pilot)


async def test_carry_row_retains_open_story_on_membership_loss(
    tmp_path: Path,
) -> None:
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        selected = app.selected()
        assert selected is not None and selected.id == 1
        server.bump(drop=(1,))
        server.requests.clear()
        server.arrivals = 0
        app.refresh_feed(announce=False)  # background refresh, no navigation
        await settle(pilot)
        assert server.arrivals == 1
        assert shown(app)[0] == 1  # carried reader row, not a jump
        assert "Summary 1" in app.query_one(Markdown)._markdown
        await pilot.press("j")  # navigate: selection moves on
        await settle(pilot)
        await pilot.press("s")  # next rebuild drops the carried row
        await settle(pilot)
        assert 1 not in shown(app)
