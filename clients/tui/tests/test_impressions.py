from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from textual.widgets import OptionList

import hn_rerank.app as reader_module
from hn_rerank.app import Reader

from ._settle import settle
from .test_client import FakeServer


async def test_selected_impression_is_delayed_not_prefetched_and_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delay = asyncio.Event()

    async def controlled_sleep(seconds: float) -> None:
        if seconds == 1.0:
            await delay.wait()
        else:
            await asyncio.sleep(seconds)

    # Replace only the reader's clock, leaving Textual's event loop untouched.
    monkeypatch.setattr(
        reader_module,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "sleep": controlled_sleep}),
    )
    fake = FakeServer()  # /api/interaction returns 404: reading must continue.
    app = Reader(api=fake.api(), prefetch=2)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert not any(r.url.path.endswith("/api/interaction") for r in fake.requests)
        app.query_one(OptionList).focus()
        await pilot.press("j")
        await settle(pilot)
        assert not any(r.url.path.endswith("/api/interaction") for r in fake.requests)
        delay.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)
        events = [
            json.loads(r.content)["events"][0]
            for r in fake.requests
            if r.url.path.endswith("/api/interaction")
        ]
        assert len(events) == 1
        event = events[0]
        assert event["story_id"] == 2
        assert event["position"] == 1
        assert event["sort_mode"] == "recommended"
        assert event["window"] == "1w" and "age_filter" not in event
        assert event["event_type"] == "impression"
        assert event["dashboard_version"] == fake.feed.version
        assert event["badges"] == []  # the sample stories have none
        # A normal rebuild/poll must not inflate impressions for this selection.
        app.rebuild()
        await settle(pilot)
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)
        assert sum(r.url.path.endswith("/api/interaction") for r in fake.requests) == 1
        assert app.selected() is not None
