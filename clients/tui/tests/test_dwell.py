from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rich.console import RenderableType
from textual import events
from textual.widgets import OptionList

import hn_rerank.app as reader_module
from hn_rerank.api import API, InteractionEvent
from hn_rerank.app import Reader
from hn_rerank.dwell import DwellContext, DwellTracker

from ._settle import settle
from .test_client import FakeServer


def make_ctx(story_id: int = 1, position: int = 0) -> DwellContext:
    return DwellContext(
        story_id=story_id,
        position=position,
        dashboard_version=0,
        sort_mode="recommended",
        window="1w",
        badges=(),
    )


def install_clocks(
    monkeypatch: pytest.MonkeyPatch,
    start_mono: float = 1000.0,
    start_wall: float = 1_700_000_000.0,
) -> dict[str, float]:
    state = {"mono": start_mono, "wall": start_wall}
    monkeypatch.setattr(reader_module, "_monotonic", lambda: state["mono"])
    monkeypatch.setattr(reader_module, "_wall_time", lambda: state["wall"])
    return state


def gate_impression_sleep(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Hold the 1 s impression delay open until the test releases it."""
    delay = asyncio.Event()

    async def controlled_sleep(seconds: float) -> None:
        if seconds == 1.0:
            await delay.wait()
        else:
            await asyncio.sleep(seconds)

    monkeypatch.setattr(
        reader_module,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": controlled_sleep}),
    )
    return delay


def interaction_events(fake: FakeServer) -> list[dict[str, Any]]:
    return [
        json.loads(request.content)["events"][0]
        for request in fake.requests
        if request.url.path.endswith("/api/interaction")
    ]


async def flush(app: Reader) -> None:
    await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)


# DwellTracker unit tests ----------------------------------------------------


def test_start_stop_basic() -> None:
    tracker = DwellTracker()
    assert tracker.start(make_ctx(1), 0.0) is None
    dwell = tracker.stop(5.0)
    assert dwell is not None
    assert dwell.ctx.story_id == 1
    assert dwell.duration_ms == 5000


def test_short_segment_dropped() -> None:
    tracker = DwellTracker()
    tracker.start(make_ctx(1), 0.0)
    assert tracker.stop(0.5) is None


def test_idle_caps_segment() -> None:
    tracker = DwellTracker()
    tracker.start(make_ctx(1), 0.0)
    dwell = tracker.stop(600.0)
    assert dwell is not None
    assert dwell.duration_ms == 300_000


def test_touch_idle_rollover_opens_same_ctx() -> None:
    tracker = DwellTracker()
    tracker.start(make_ctx(7), 0.0)
    dwell = tracker.touch(600.0)
    assert dwell is not None
    assert dwell.ctx.story_id == 7
    assert dwell.duration_ms == 300_000
    assert tracker.current is not None and tracker.current.story_id == 7
    again = tracker.stop(605.0)
    assert again is not None and again.duration_ms == 5000


def test_start_while_open_returns_old() -> None:
    tracker = DwellTracker()
    tracker.start(make_ctx(1), 0.0)
    old = tracker.start(make_ctx(2, position=1), 5.0)
    assert old is not None and old.ctx.story_id == 1
    assert old.duration_ms == 5000
    dwell = tracker.stop(7.0)
    assert dwell is not None and dwell.ctx.story_id == 2
    assert dwell.duration_ms == 2000


def test_empty_stop_and_touch() -> None:
    tracker = DwellTracker()
    assert tracker.stop(1.0) is None
    assert tracker.touch(1.0) is None


def test_backwards_touch_ignored() -> None:
    tracker = DwellTracker()
    tracker.start(make_ctx(1), 0.0)
    assert tracker.touch(5.0) is None
    assert tracker.touch(3.0) is None
    dwell = tracker.stop(6.0)
    assert dwell is not None and dwell.duration_ms == 6000


def test_duration_clamped_to_day() -> None:
    tracker = DwellTracker(idle_limit_s=200_000.0, min_s=0.0)
    tracker.start(make_ctx(1), 0.0)
    dwell = tracker.stop(100_000.0)
    assert dwell is not None and dwell.duration_ms == 86_400_000


# DwellTracker property tests (small budgets) --------------------------------


@settings(max_examples=25, deadline=None)
@given(st.data())
def test_dwell_durations_bounded(data: Any) -> None:
    tracker = DwellTracker()
    idle_ms = 300_000
    min_ms = 1000
    now = 0.0
    ctxs = [
        make_ctx(story_id=i + 1, position=i)
        for i in range(data.draw(st.integers(1, 4)))
    ]
    for _ in range(data.draw(st.integers(1, 25))):
        now += data.draw(st.floats(0.05, 20.0, allow_nan=False, allow_infinity=False))
        op = data.draw(st.integers(0, 2))
        if op == 0:
            dwell = tracker.start(data.draw(st.sampled_from(ctxs)), now)
        elif op == 1:
            dwell = tracker.touch(now)
        else:
            dwell = tracker.stop(now)
        if dwell is not None:
            assert min_ms <= dwell.duration_ms <= idle_ms
            assert 0 <= dwell.duration_ms <= 86_400_000


@settings(max_examples=25, deadline=None)
@given(st.data())
def test_dwell_total_never_exceeds_elapsed(data: Any) -> None:
    tracker = DwellTracker()
    first = 100.0
    now = first
    total_ms = 0
    started = False
    ctxs = [make_ctx(story_id=i + 1) for i in range(3)]
    for _ in range(data.draw(st.integers(1, 25))):
        now += data.draw(st.floats(0.05, 20.0, allow_nan=False, allow_infinity=False))
        op = data.draw(st.integers(0, 2))
        if op == 0:
            started = True
            dwell = tracker.start(data.draw(st.sampled_from(ctxs)), now)
        elif op == 1:
            dwell = tracker.touch(now)
        else:
            dwell = tracker.stop(now)
        if dwell is not None:
            total_ms += dwell.duration_ms
    if started:
        assert total_ms <= int((now - first) * 1000) + 1


def test_no_dwell_without_start() -> None:
    tracker = DwellTracker()
    now = 0.0
    for op in ("touch", "stop", "touch", "stop"):
        now += 1.0
        if op == "touch":
            assert tracker.touch(now) is None
        else:
            assert tracker.stop(now) is None


@settings(max_examples=25, deadline=None)
@given(st.data())
def test_dwell_ctx_matches_open_segment(data: Any) -> None:
    tracker = DwellTracker()
    open_ctx: DwellContext | None = None
    last_input: float | None = None
    now = 0.0
    ctxs = [make_ctx(story_id=100 + i) for i in range(3)]
    for _ in range(data.draw(st.integers(1, 25))):
        now += data.draw(st.floats(0.05, 20.0, allow_nan=False, allow_infinity=False))
        op = data.draw(st.integers(0, 2))
        if op == 0:
            ctx = data.draw(st.sampled_from(ctxs))
            dwell = tracker.start(ctx, now)
            if dwell is not None and open_ctx is not None:
                assert dwell.ctx == open_ctx
            open_ctx, last_input = ctx, now
        elif op == 1:
            dwell = tracker.touch(now)
            if open_ctx is None:
                assert dwell is None
            elif last_input is not None and now - last_input > 300.0:
                assert dwell is not None and dwell.ctx == open_ctx
                last_input = now
            else:
                assert dwell is None
                last_input = now
        else:
            dwell = tracker.stop(now)
            if open_ctx is None:
                assert dwell is None
            elif dwell is not None:
                assert dwell.ctx == open_ctx
            open_ctx, last_input = None, None


# InteractionEvent payload ---------------------------------------------------


async def test_interaction_payload_omits_none_duration() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.clear()
        seen.update(json.loads(request.content)["events"][0])
        return httpx.Response(200, json={"ok": True})

    api = API("https://example.org/hn/", "test", httpx.MockTransport(handler))
    try:
        await api.interaction(
            InteractionEvent(
                event_id=str(uuid4()),
                client_session_id=str(uuid4()),
                story_id=1,
                dashboard_version=0,
                position=0,
                sort_mode="recommended",
                window="1w",
                occurred_at=1_700_000_000.0,
                event_type="article_open",
            )
        )
        assert "duration_ms" not in seen
        assert seen["ranker_arm"] == "tui_observed"
        assert seen["source_filter"] == "mixed"
        await api.interaction(
            InteractionEvent(
                event_id=str(uuid4()),
                client_session_id=str(uuid4()),
                story_id=1,
                dashboard_version=0,
                position=0,
                sort_mode="recommended",
                window="1w",
                occurred_at=1_700_000_000.0,
                event_type="dwell",
                duration_ms=5000,
            )
        )
        assert seen["duration_ms"] == 5000
    finally:
        await api.close()


# Opens end to end -------------------------------------------------------------


async def test_open_keys_send_one_event_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(reader_module, "open_in_browser", opened.append)
    install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        await flush(app)
        fake.requests.clear()
        await pilot.press("o")
        await settle(pilot)
        await flush(app)
        all_events = interaction_events(fake)
        assert len(all_events) == 1
        assert all_events[0]["event_type"] == "article_open"
        assert all_events[0]["story_id"] == 1
        assert all_events[0]["position"] == 0
        assert all_events[0]["sort_mode"] == "recommended"
        assert all_events[0]["window"] == "1w"
        assert all_events[0]["dashboard_version"] == fake.feed.version
        assert all_events[0]["badges"] == []
        assert "duration_ms" not in all_events[0]
        assert opened == ["https://example.org/1"]
        fake.requests.clear()
        await pilot.press("c")
        await settle(pilot)
        await flush(app)
        all_events = interaction_events(fake)
        assert len(all_events) == 1
        assert all_events[0]["event_type"] == "comments_open"
        assert all_events[0]["story_id"] == 1
        assert "duration_ms" not in all_events[0]
        assert opened[-1] == "https://news.ycombinator.com/item?id=1"


async def test_open_without_link_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(reader_module, "open_in_browser", opened.append)
    install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        selected = app.selected()
        assert selected is not None
        monkeypatch.setattr(
            app, "selected", lambda: dataclasses.replace(selected, article_url="")
        )
        await flush(app)
        fake.requests.clear()
        app.action_open_url("article_url")
        await settle(pilot)
        assert opened == []
        assert interaction_events(fake) == []


async def test_open_comments_without_link_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(reader_module, "open_in_browser", opened.append)
    install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        selected = app.selected()
        assert selected is not None
        monkeypatch.setattr(
            app, "selected", lambda: dataclasses.replace(selected, comments_url="")
        )
        await flush(app)
        fake.requests.clear()
        app.action_open_url("comments_url")
        await settle(pilot)
        await flush(app)
        assert opened == []
        assert interaction_events(fake) == []


# Dwell end to end --------------------------------------------------------------


async def test_dwell_five_seconds_then_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        state["mono"] += 5.0
        state["wall"] += 5.0
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["story_id"] == 1
        assert dwells[0]["duration_ms"] == 5000
        assert dwells[0]["position"] == 0


async def test_dwell_under_a_second_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        state["mono"] += 0.5
        state["wall"] += 0.5
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        assert [e for e in interaction_events(fake) if e["event_type"] == "dwell"] == []


async def test_dwell_idle_caps_at_five_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        state["mono"] += 600.0
        state["wall"] += 600.0
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["duration_ms"] == 300_000


async def test_dwell_input_keeps_idle_open_across_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paging every 60 s for 8 min logs one 480 s dwell, not the 300 s cap."""
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        for _ in range(8):
            state["mono"] += 60.0
            state["wall"] += 60.0
            await pilot.press("space")
            await settle(pilot)
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["story_id"] == 1
        assert dwells[0]["duration_ms"] == 480_000


async def test_dwell_idle_then_input_splits_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle 10 min, space, 5 s, j logs 300 s and 5 s dwells in order."""
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        state["mono"] += 600.0
        state["wall"] += 600.0
        await pilot.press("space")
        await settle(pilot)
        await flush(app)
        state["mono"] += 5.0
        state["wall"] += 5.0
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert [d["duration_ms"] for d in dwells] == [300_000, 5_000]
        assert all(d["story_id"] == 1 for d in dwells)


async def test_blur_ends_and_focus_resumes_without_impression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install_clocks(monkeypatch)
    delay = gate_impression_sleep(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        delay.set()
        await flush(app)
        fake.requests.clear()
        state["mono"] += 5.0
        state["wall"] += 5.0
        app.post_message(events.AppBlur())
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["duration_ms"] == 5000
        impressions_before = [
            e for e in interaction_events(fake) if e["event_type"] == "impression"
        ]
        state["mono"] += 2.0
        state["wall"] += 2.0
        app.post_message(events.AppFocus())
        await settle(pilot)
        await flush(app)
        assert len(
            [e for e in interaction_events(fake) if e["event_type"] == "impression"]
        ) == len(impressions_before)
        # Resume must actually restart the segment: 3 s later moving logs a
        # second dwell for the same story and no new impression for it.
        impressions_story1 = len(
            [
                e
                for e in interaction_events(fake)
                if e["event_type"] == "impression" and e["story_id"] == 1
            ]
        )
        state["mono"] += 3.0
        state["wall"] += 3.0
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        story1 = [d for d in dwells if d["story_id"] == 1]
        assert len(story1) == 2
        assert story1[1]["duration_ms"] == 3000
        assert (
            len(
                [
                    e
                    for e in interaction_events(fake)
                    if e["event_type"] == "impression" and e["story_id"] == 1
                ]
            )
            == impressions_story1
        )


async def test_help_ends_dwell_and_escape_resumes_without_impression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install_clocks(monkeypatch)
    delay = gate_impression_sleep(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        delay.set()
        await flush(app)
        fake.requests.clear()
        state["mono"] += 5.0
        state["wall"] += 5.0
        await pilot.press("?")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["duration_ms"] == 5000
        impressions = len(
            [e for e in interaction_events(fake) if e["event_type"] == "impression"]
        )
        await pilot.press("escape")
        await settle(pilot)
        await flush(app)
        assert (
            len(
                [e for e in interaction_events(fake) if e["event_type"] == "impression"]
            )
            == impressions
        )
        impressions_story1 = len(
            [
                e
                for e in interaction_events(fake)
                if e["event_type"] == "impression" and e["story_id"] == 1
            ]
        )
        state["mono"] += 3.0
        state["wall"] += 3.0
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        story1 = [d for d in dwells if d["story_id"] == 1]
        assert len(story1) == 2
        assert story1[1]["duration_ms"] == 3000
        assert (
            len(
                [
                    e
                    for e in interaction_events(fake)
                    if e["event_type"] == "impression" and e["story_id"] == 1
                ]
            )
            == impressions_story1
        )


async def test_vote_logs_four_second_dwell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        state["mono"] += 4.0
        state["wall"] += 4.0
        app.action_vote("up")
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["story_id"] == 1
        assert dwells[0]["duration_ms"] == 4000


async def test_window_cycle_logs_dwell_with_old_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        state["mono"] += 4.0
        state["wall"] += 4.0
        app.action_cycle_window(1)
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["story_id"] == 1
        assert dwells[0]["duration_ms"] == 4000
        assert dwells[0]["window"] == "1w"


@pytest.mark.parametrize("panel", ["why", "legend"])
async def test_info_panels_end_dwell_like_help(
    monkeypatch: pytest.MonkeyPatch, panel: str
) -> None:
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        state["mono"] += 5.0
        state["wall"] += 5.0
        if panel == "why":
            app.action_why_story()
        else:
            app.action_badge_legend()
        await settle(pilot)
        await flush(app)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["story_id"] == 1
        assert dwells[0]["duration_ms"] == 5000


# Quit --------------------------------------------------------------------------


class ObservedReader(Reader):
    def __init__(self, api: API) -> None:
        super().__init__(api=api, prefetch=0)
        self.exited = asyncio.Event()

    def exit(
        self,
        result: None = None,
        return_code: int = 0,
        message: RenderableType | None = None,
    ) -> None:
        self.exited.set()
        super().exit(result, return_code, message)


async def test_quit_sends_open_dwell(monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_clocks(monkeypatch)
    fake = FakeServer()
    app = ObservedReader(fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        fake.requests.clear()
        state["mono"] += 5.0
        state["wall"] += 5.0
        await app.action_quit()
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        dwells = [e for e in interaction_events(fake) if e["event_type"] == "dwell"]
        assert len(dwells) == 1
        assert dwells[0]["story_id"] == 1
        assert dwells[0]["duration_ms"] == 5000
        assert app.quit_report == []


async def test_quit_with_hanging_telemetry_still_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingInteraction(FakeServer):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/interaction"):
                self.requests.append(request)
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return await super().__call__(request)

    install_clocks(monkeypatch)
    fake = HangingInteraction()
    app = ObservedReader(fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await app.action_quit()
        # Telemetry alone never holds a quit longer than ~1 s.
        await asyncio.wait_for(app.exited.wait(), timeout=1.5)
        assert app.quit_report == []


async def test_quit_with_failing_telemetry_keeps_vote_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    saved: list[tuple[int, str]] = []

    class VoteHeldFailTelemetry(FakeServer):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/interaction"):
                self.requests.append(request)
                return httpx.Response(500)
            if request.url.path.endswith("/api/feedback"):
                self.requests.append(request)
                entered.set()
                await release.wait()
                payload = json.loads(request.content)
                saved.append((payload["story_id"], payload["action"]))
                return httpx.Response(200, json={"ok": True, "target_version": 1})
            return await super().__call__(request)

    install_clocks(monkeypatch)
    fake = VoteHeldFailTelemetry()
    app = ObservedReader(fake.api())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.action_vote("up")
        await asyncio.wait_for(entered.wait(), timeout=2)
        await app.action_quit()
        await pilot.pause(0.05)
        assert not app.exited.is_set()
        release.set()
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        assert saved == [(1, "up")]
        assert app.quit_report == []
