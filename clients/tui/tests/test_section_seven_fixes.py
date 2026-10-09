"""Regression tests for the seven second-review fixes (2026-10-09).

Real production Reader under Textual Pilot against RefreshServer (held
feed GETs, gated summary POSTs, live stats overrides). No network,
no provider, no DB.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Markdown, OptionList, Static

from hn_rerank.api import API, Profile
from hn_rerank.app import Reader, StatsPrompt

from ._settle import settle
from .test_sort_refresh import RefreshServer, modal_text, shown, wait_for


async def test_second_r_while_forced_running_prompts_on_stale_snapshot(
    tmp_path: Path,
) -> None:
    """A repeated r joins the already-started forced task (one paid request),
    consumes its declared intent, and the still-behind returned snapshot
    prompts instead of deferring to the joined intent forever."""
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
        await pilot.press("r")  # joins the running forced task
        await pilot.pause(0.6)
        assert len(server.forced_posts) == 1
        assert not isinstance(app.screen, StatsPrompt)  # stats deferred mid-flight
        server.snapshot = 10  # regeneration lands still behind live 25
        server.summary_gate.set()
        server.hold_summary = False
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        assert len(server.forced_posts) == 1  # one paid request, not two
        await wait_for(pilot, lambda: bool(app.screen.query("#stats-regen")))
        text = modal_text(app)
        assert "Discussion had 10 comments when summarized" in text
        assert "25 now" in text
        await pilot.press("escape")
        await settle(pilot)


async def test_stale_ready_feed_after_vote_ack_keeps_target_and_restored(
    tmp_path: Path,
) -> None:
    """An old ready response (constructed at V) released after the vote ack
    (target V+1) must not clear the pending target or the undone rows."""
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        selected = app.selected()
        assert selected is not None and selected.id == 1
        server.requests.clear()
        server.arrivals = 0
        server.hold_feed = True
        server.gate = asyncio.Event()
        app.refresh_feed(announce=False)  # old response, constructed at V0
        await wait_for(pilot, lambda: server.arrivals == 1)
        app.action_vote("up")
        await wait_for(
            pilot,
            lambda: app.feed is not None and app.feed.target_version == 1,
        )
        app.action_undo()
        await wait_for(pilot, lambda: 1 in app.restored)
        server.gate.set()
        server.hold_feed = False
        await settle(pilot)
        assert app.feed is not None
        assert app.feed.version == 0
        assert app.feed.target_version == 1
        assert app.feed.ready is False
        assert 1 in app.restored
        assert 1 in shown(app)


async def test_connect_resets_stats_state_and_allows_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Profile-scoped regeneration/stats state dies on reconnect: a held
    request of A's is cancelled, old serials/offers/pending/intents clear,
    its late reply changes nothing, and B's same observation may prompt."""
    server = RefreshServer()
    server.live = {1: (150, 25)}
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        await pilot.press("r")
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        await pilot.press("escape")
        await settle(pilot)
        assert app._stats_offered.get(1) == 25

        stats_gate = asyncio.Event()
        stats_started = asyncio.Event()
        orig_stats = API.story_stats
        calls: list[API] = []

        async def gated_stats(self: API, story_id: int) -> Any:
            stats_started.set()
            await stats_gate.wait()
            calls.append(self)
            return await orig_stats(self, story_id)

        monkeypatch.setattr(API, "story_stats", gated_stats)
        app.refresh_story_stats(1, app.selection_serial)
        await wait_for(pilot, lambda: stats_started.is_set())
        await wait_for(pilot, lambda: 1 in app._stats_requests)
        old_task = app._stats_requests[1]
        old_serial = app._stats_serials[1]

        fresh_api = server.api()
        monkeypatch.setattr("hn_rerank.app.API", lambda srv, tok: fresh_api)
        await app.connected(Profile("https://example.org/hn/", "other-token"))
        await pilot.pause(0.3)
        # A's held request is gone (the feed tail may already have started
        # B's own check for the open story on the new api).
        assert old_task.done()
        assert all(t is not old_task for t in app._stats_requests.values())
        assert app._stats_offered == {}
        assert app._stats_pending_live == {}
        assert app._stats_serials.get(1) != old_serial
        assert app._forced_intent == {}
        stats_gate.set()
        # B's own check on the new api (explicit, not relying on the feed
        # tail): A's cancelled request must never serve a reply.
        app.refresh_story_stats(1, app.selection_serial)
        await wait_for(pilot, lambda: len(calls) == 1)
        assert calls[0] is fresh_api
        await wait_for(
            pilot,
            lambda: next(s for s in app.stories if s.id == 1).points == 150,
        )
        await settle(pilot)

        await pilot.press("r")
        await wait_for(pilot, lambda: isinstance(app.screen, StatsPrompt))
        await pilot.press("escape")
        await settle(pilot)


async def test_hidden_story_hint_names_reconnect_recovery(tmp_path: Path) -> None:
    """The hide notice no longer claims r restores; reconnect is the way back."""
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.unavailable.discard(2)
        app._hide_story(2, "summary unavailable (gone)")
        assert 2 in app.unavailable
        status = str(app.query_one("#status", Static).content)
        assert "r restores" not in status
        assert "reconnect to see it again" in status
        assert "Summary 2" not in app.query_one(Markdown)._markdown


async def test_covering_ready_after_vote_ack_clears_restored(tmp_path: Path) -> None:
    """A later ready deck covering the acked target clears the pending
    target and the undone rows: the stale old-version reply must not, the
    covering one must."""
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await wait_for(
            pilot,
            lambda: app.feed is not None and app.feed.target_version == 1,
        )
        app.action_undo()
        await wait_for(pilot, lambda: 1 in app.restored)
        server.bump()  # reranked deck v1 covers the acked target
        app.refresh_feed(announce=False)
        await wait_for(pilot, lambda: app.feed is not None and app.feed.version == 1)
        await settle(pilot)
        assert app.feed is not None
        assert app.feed.version == 1
        assert app.feed.target_version == 1
        assert app.feed.ready is True
        assert app.restored == {}


async def test_lower_version_restart_lands_on_later_request(tmp_path: Path) -> None:
    """The vote-ack serial bump must not block restarts: a lower-version
    deck arriving on a later request (fresh serial) still replaces."""
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await wait_for(
            pilot,
            lambda: app.feed is not None and app.feed.target_version == 1,
        )
        # Server restarted: back to the version-0 deck (target 0), while
        # the client still waits for target 1.
        app.refresh_feed(announce=False)
        await wait_for(
            pilot,
            lambda: (
                app.feed is not None
                and app.feed.version == 0
                and app.feed.target_version == 0
            ),
        )
        await settle(pilot)
        assert app.feed is not None
        assert app.feed.version == 0
        assert app.feed.ready is True


async def test_stale_reply_after_vote_ack_still_runs_queued_sort_followup(
    tmp_path: Path,
) -> None:
    """A sort change queued mid-request is not stranded by the vote-ack
    serial bump: the discarded old reply drains it with exactly one
    follow-up, and no further requests follow."""
    server = RefreshServer()
    app = Reader(api=server.api(), config_path=tmp_path / "p.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        server.requests.clear()
        server.arrivals = 0
        server.hold_feed = True
        server.gate = asyncio.Event()
        app.refresh_feed(announce=False)  # old request, constructed at V0
        await wait_for(pilot, lambda: server.arrivals == 1)
        app.refresh_sort_background()  # sort change mid-flight queues one
        assert app._sort_refresh_pending is True
        app.action_vote("up")  # ack postdates the held request: bumps serial
        await wait_for(
            pilot,
            lambda: app.feed is not None and app.feed.target_version == 1,
        )
        server.gate.set()
        server.hold_feed = False
        await wait_for(pilot, lambda: server.arrivals == 2)
        await settle(pilot)
        assert server.arrivals == 2  # at most one follow-up, then quiet
        assert app._sort_refresh_pending is False
        assert app.feed is not None
        assert app.feed.version == 0
