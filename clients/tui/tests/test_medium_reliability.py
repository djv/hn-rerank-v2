"""Recoverable summary errors and vote-safe shutdown."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from rich.console import RenderableType
from textual.widgets import Markdown, Select

from hn_rerank.api import API, APIError, TransientError
from hn_rerank.app import Reader

from ._settle import settle
from .test_client import FakeServer


class HeldVotes(FakeServer):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.saved: list[tuple[int, str]] = []
        self.attempts = 0
        self.cancelled = False
        self.fail = fail

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/api/feedback"):
            return await super().__call__(request)
        self.requests.append(request)
        self.attempts += 1
        if self.attempts == 1:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.fail:
            raise httpx.ReadError("Unconfirmed vote", request=request)
        payload = json.loads(request.content)
        self.saved.append((payload["story_id"], payload["action"]))
        return httpx.Response(200, json={"ok": True, "target_version": 1})


class ObservedAPI(API):
    def __init__(self, fake: FakeServer) -> None:
        super().__init__("https://example.org/hn/", "test", httpx.MockTransport(fake))
        self.closed = asyncio.Event()

    async def close(self) -> None:
        await super().close()
        self.closed.set()


class ObservedReader(Reader):
    def __init__(self, api: ObservedAPI) -> None:
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


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_server_errors_are_transient(status: int) -> None:
    api = API(
        "https://example.org/hn/",
        "test",
        httpx.MockTransport(lambda _: httpx.Response(status)),
    )
    try:
        with pytest.raises(TransientError, match=str(status)):
            await api.feed()
    finally:
        await api.close()


async def test_missing_content_is_not_a_transient_server_error() -> None:
    api = API(
        "https://example.org/hn/",
        "test",
        httpx.MockTransport(lambda _: httpx.Response(404)),
    )
    try:
        with pytest.raises(APIError) as error:
            await api.feed()
        assert not isinstance(error.value, TransientError)
    finally:
        await api.close()


async def test_first_summary_500_keeps_story_and_can_retry() -> None:
    class RetrySummary(FakeServer):
        fail = True

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if "/api/tldr-cache/" in request.url.path:
                return httpx.Response(204)
            if self.fail and request.url.path.endswith("/api/tldr-detail"):
                return httpx.Response(500)
            return await super().__call__(request)

    fake = RetrySummary()
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert [story.id for story in app.stories] == [1, 2]
        assert not app.unavailable
        selected = app.selected()
        assert selected is not None and selected.id == 1
        assert "500" in app.query_one("#summary", Markdown)._markdown
        fake.fail = False
        app.action_refresh()
        await settle(pilot)
        assert "Summary 1" in app.query_one("#summary", Markdown)._markdown
        assert [story.id for story in app.stories] == [1, 2]


@pytest.mark.parametrize("key", ["q", "ctrl+q"])
@pytest.mark.parametrize("undo", [False, True])
async def test_quit_drains_serialized_votes_before_closing_api(
    key: str, undo: bool
) -> None:
    fake = HeldVotes()
    api = ObservedAPI(fake)
    app = ObservedReader(api)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.action_vote("up")
        await asyncio.wait_for(fake.entered.wait(), timeout=2)
        if undo:
            app.action_undo()
        app.action_vote("down")
        await pilot.press(key)
        await pilot.pause(0.05)
        assert not app.exited.is_set()
        assert not fake.cancelled
        fake.release.set()
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        expected = (
            [(1, "up"), (1, "clear"), (1, "down")]
            if undo
            else [
                (1, "up"),
                (2, "down"),
            ]
        )
        assert fake.saved == expected
        assert fake.attempts == len(expected)
        assert not fake.cancelled
        assert app.quit_report == []
    assert api.closed.is_set()


async def test_quit_held_vote_blocks_keys_and_exits_at_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hn_rerank.app.QUIT_WAIT_SECONDS", 0.8, raising=False)
    fake = HeldVotes()
    api = ObservedAPI(fake)
    app = ObservedReader(api)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.action_vote("up")
        await asyncio.wait_for(fake.entered.wait(), timeout=2)
        await app.action_quit()
        assert "Saving 1 vote" in app.status_text
        assert "q again" in app.status_text
        selected = app.selected()
        assert selected is not None
        before_id = selected.id
        before_rated = set(app.rated)
        before_history = list(app.history)
        before_window = str(app.query_one("#window", Select).value)
        before_sort = str(app.query_one("#sort", Select).value)
        await pilot.press("j")
        await pilot.press("1")
        await pilot.press("u")
        await pilot.press("s")
        await pilot.press("d")
        await pilot.pause(0.05)
        selected = app.selected()
        assert selected is not None and selected.id == before_id
        assert set(app.rated) == before_rated
        assert list(app.history) == before_history
        assert str(app.query_one("#window", Select).value) == before_window
        assert str(app.query_one("#sort", Select).value) == before_sort
        # The quit wait itself never cancels the saving vote.
        assert not fake.cancelled
        assert fake.attempts == 1
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        assert len(app.quit_report) == 1
        assert "Story 1" in app.quit_report[0]
        assert "still saving" in app.quit_report[0]
        assert "(up)" in app.quit_report[0]
        assert fake.attempts == 1
        assert fake.saved == []
    assert api.closed.is_set()


async def test_quit_second_q_exits_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hn_rerank.app.QUIT_WAIT_SECONDS", 2.0, raising=False)
    fake = HeldVotes()
    api = ObservedAPI(fake)
    app = ObservedReader(api)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.action_vote("up")
        await asyncio.wait_for(fake.entered.wait(), timeout=2)
        await app.action_quit()
        await pilot.pause(0.05)
        assert not app.exited.is_set()
        assert not fake.cancelled
        started = time.monotonic()
        await app.action_quit()
        await asyncio.wait_for(app.exited.wait(), timeout=0.5)
        assert time.monotonic() - started < 1.0
        assert len(app.quit_report) == 1
        assert "still saving" in app.quit_report[0]
        assert fake.attempts == 1
    assert api.closed.is_set()


async def test_vote_failure_during_quit_exits_with_report() -> None:
    fake = HeldVotes(fail=True)
    api = ObservedAPI(fake)
    app = ObservedReader(api)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.action_vote("up")
        await asyncio.wait_for(fake.entered.wait(), timeout=2)
        await app.action_quit()
        fake.release.set()
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        assert len(app.quit_report) == 1
        assert "Story 1" in app.quit_report[0]
        assert "Connection failed" in app.quit_report[0]
        assert "(up)" in app.quit_report[0]
        assert fake.attempts == 1
        assert not fake.saved
        assert 1 not in app.rated
    assert api.closed.is_set()


class FailSecondStory(HeldVotes):
    """Story 1's vote saves; story 2's vote fails."""

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/feedback"):
            self.fail = json.loads(request.content)["story_id"] == 2
        return await super().__call__(request)


async def test_quit_report_lists_only_the_failed_vote() -> None:
    """A vote that saved is not reported because another one failed."""
    fake = FailSecondStory()
    app = ObservedReader(ObservedAPI(fake))
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.action_vote("up")
        await asyncio.wait_for(fake.entered.wait(), timeout=2)
        app.action_vote("down")
        await app.action_quit()
        fake.release.set()
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        assert fake.saved == [(1, "up")]
        assert len(app.quit_report) == 1
        assert "(down)" in app.quit_report[0]
        assert "Story 2" in app.quit_report[0]


async def test_quit_without_pending_votes_closes_immediately() -> None:
    api = ObservedAPI(FakeServer())
    app = ObservedReader(api)
    async with app.run_test() as pilot:
        await settle(pilot)
        await app.action_quit()
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        assert "Saving" not in app.status_text
        assert "q again" not in app.status_text
        assert app.quit_report == []
    assert api.closed.is_set()


async def test_exit_request_keeps_vote_and_undo_blocked() -> None:
    fake = HeldVotes()
    app = ObservedReader(ObservedAPI(fake))
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await app.action_quit()
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        rated, history = app.rated.copy(), app.history.copy()
        app.action_vote("up")
        assert app.rated == rated
        assert app.history == history
        app.action_undo()
        await app.action_quit()
        assert app.rated == rated
        assert app.history == history
        assert not fake.attempts
        assert app.check_action("vote", ("up",)) is False
        assert app.check_action("undo", ()) is False
        assert app.check_action("quit", ()) is True
        assert app.check_action("move", (1,)) is False
        assert app.quit_report == []


async def test_cancelled_vote_during_quit_exits_with_report() -> None:
    fake = HeldVotes()
    app = ObservedReader(ObservedAPI(fake))
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.action_vote("up")
        await asyncio.wait_for(fake.entered.wait(), timeout=2)
        await app.action_quit()
        app.workers.cancel_group(app, "vote")
        await asyncio.wait_for(app.exited.wait(), timeout=2)
        assert fake.cancelled
        assert fake.attempts == 1
        assert len(app.quit_report) == 1
        assert "Story 1" in app.quit_report[0]
        assert "cancelled" in app.quit_report[0]
        assert "(up)" in app.quit_report[0]
