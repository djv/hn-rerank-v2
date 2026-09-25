from __future__ import annotations

import json

from textual.widgets import OptionList

from hn_rerank.app import Reader

from .test_client import FakeServer


async def test_selected_impression_is_delayed_not_prefetched_and_best_effort() -> None:
    fake = FakeServer()  # /api/interaction returns 404: reading must continue.
    app = Reader(api=fake.api(), prefetch=2)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.3)
        app.query_one(OptionList).focus()
        await pilot.press("j")
        await pilot.pause(1.2)
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
        assert event["age_filter"] == "recent"
        assert event["event_type"] == "impression"
        assert event["dashboard_version"] == fake.feed.version
        # A normal rebuild/poll must not inflate impressions for this selection.
        app.rebuild()
        await pilot.pause(1.1)
        assert sum(r.url.path.endswith("/api/interaction") for r in fake.requests) == 1
        assert app.selected() is not None
