from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest
from hypothesis import given, strategies as st
from textual.widgets import Markdown, OptionList, Select, Tabs

import hn_rerank.app as reader_module
from hn_rerank.app import Reader
from hn_rerank.api import Profile
from hn_rerank.models import Feed, FeedBadge, FeedStory
from hn_rerank.queue import QueueEntry, default_sort, extend_queue, queue_entries

from ._settle import settle
from .test_client import FakeServer
from .test_dwell import flush, gate_impression_sleep, install_clocks, interaction_events


def story(sid: int, kinds: tuple[str, ...] = ()) -> FeedStory:
    return FeedStory(
        sid,
        f"Story {sid}",
        f"https://example.org/{sid}",
        f"https://news.ycombinator.com/item?id={sid}",
        "hn",
        100,
        10,
        sid,
        float(sid),
        badge_details=[FeedBadge(kind, "", kind, "") for kind in kinds],
    )


def queue_feed() -> Feed:
    kinds = {
        2: ("interest",),
        3: ("uncertain",),
        4: ("discover",),
        5: ("discover", "interest"),
    }
    return Feed(
        2,
        "1w",
        [story(sid, kinds.get(sid, ())) for sid in range(1, 15)],
        {
            "recommended": [1, 1, 999],
            "explore": [2, 3, 4, 5, 6, 1, 4, 999],
            "popular": [999, 1, 4, 7, 7, 8, 9, 10, 11, 12],
        },
        {},
        0,
        0,
        True,
    )


def test_queue_groups_deduplicates_skips_missing_and_keeps_origins() -> None:
    feed = queue_feed()
    before = {key: list(order) for key, order in feed.orders.items()}
    assert queue_entries(feed) == [
        QueueEntry(1, "recommended", 0),
        QueueEntry(4, "explore", 2),
        QueueEntry(5, "explore", 3),
        QueueEntry(3, "explore", 1),
        QueueEntry(2, "explore", 0),
        QueueEntry(6, "explore", 4),
        QueueEntry(7, "popular", 3),
        QueueEntry(8, "popular", 5),
        QueueEntry(9, "popular", 6),
        QueueEntry(10, "popular", 7),
        QueueEntry(11, "popular", 8),
    ]
    assert feed.orders == before


@given(
    ids=st.sets(st.integers(1, 30), max_size=30),
    recommended=st.lists(st.integers(1, 35), max_size=40),
    explore=st.lists(st.integers(1, 35), max_size=40),
    popular=st.lists(st.integers(1, 35), max_size=40),
    badges=st.dictionaries(
        st.integers(1, 30),
        st.sets(st.sampled_from(["discover", "uncertain", "interest", "hot"])),
        max_size=30,
    ),
)
def test_queue_invariants(
    ids: set[int],
    recommended: list[int],
    explore: list[int],
    popular: list[int],
    badges: dict[int, set[str]],
) -> None:
    feed = replace(
        queue_feed(),
        stories=[
            story(sid, tuple(sorted(badges.get(sid, set())))) for sid in sorted(ids)
        ],
        orders={"recommended": recommended, "explore": explore, "popular": popular},
    )
    entries = queue_entries(feed)
    result = [item.story_id for item in entries]
    prefix = list(dict.fromkeys(sid for sid in recommended if sid in ids))
    assert result[: len(prefix)] == prefix
    assert len(result) == len(set(result))
    assert set(result) <= ids
    assert {sid for sid in explore if sid in ids} <= set(result)
    assert sum(item.origin == "popular" for item in entries) <= 5
    earlier = set(prefix) | {sid for sid in explore if sid in ids}
    expected_popular = list(
        dict.fromkeys(sid for sid in popular if sid in ids and sid not in earlier)
    )[:5]
    assert [
        item.story_id for item in entries if item.origin == "popular"
    ] == expected_popular
    for item in entries:
        assert feed.orders[item.origin][item.served_rank] == item.story_id
    assert queue_entries(feed) == entries


@given(ordinal=st.integers(1, 3652058))
def test_default_alternates_by_utc_day(ordinal: int) -> None:
    now = datetime.fromordinal(ordinal).replace(tzinfo=UTC)
    assert default_sort(now) == ("queue" if ordinal % 2 else "recommended")
    assert default_sort(now + timedelta(days=1)) != default_sort(now)


def test_default_converts_to_utc() -> None:
    now = datetime.fromordinal(739900).replace(tzinfo=timezone(timedelta(hours=5)))
    assert default_sort(now) == "queue"


@given(
    original=st.lists(st.integers(1, 50), unique=True, max_size=30),
    recommended=st.lists(st.integers(1, 60), max_size=50),
    available=st.sets(st.integers(1, 60), max_size=60),
)
def test_extended_queue_preserves_prefix_and_appends_only_recommended(
    original: list[int], recommended: list[int], available: set[int]
) -> None:
    snapshot = [QueueEntry(sid, "explore", rank) for rank, sid in enumerate(original)]
    before = list(snapshot)
    feed = replace(
        queue_feed(),
        stories=[story(sid) for sid in sorted(available)],
        orders={"recommended": recommended, "explore": [61], "popular": [62]},
    )
    extended = extend_queue(snapshot, feed)
    assert snapshot == before
    assert extended[: len(snapshot)] == snapshot
    ids = [entry.story_id for entry in extended]
    assert len(ids) == len(set(ids))
    assert ids[len(snapshot) :] == list(
        dict.fromkeys(
            sid for sid in recommended if sid in available and sid not in original
        )
    )
    for entry in extended[len(snapshot) :]:
        assert entry.origin == "recommended"
        assert recommended[entry.served_rank] == entry.story_id
    assert extend_queue(extended, feed) == extended


@pytest.mark.parametrize(
    "ordinal,expected", [(739899, "queue"), (739900, "recommended")]
)
async def test_startup_sort_and_controls(ordinal: int, expected: str) -> None:
    app = Reader(
        api=FakeServer().api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(ordinal).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert app.query_one("#sort", Select).value == expected
        assert app.query_one("#sort-tabs", Tabs).active == f"sort-{expected}"
        assert [item.id for item in app.stories] == [1, 2]
        await flush(app)


async def test_queue_events_keep_origin_and_full_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = install_clocks(monkeypatch)
    delay = gate_impression_sleep(monkeypatch)
    monkeypatch.setattr(reader_module, "open_in_browser", lambda url: None)
    fake = FakeServer()
    fake.feed = replace(
        queue_feed(),
        orders={
            "recommended": [1],
            "explore": [2, 3, 4, 5, 6],
            "popular": [1, 4, 7, 8, 9, 10, 11, 12],
        },
    )
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.apply_vote(app.stories[0])
        await settle(pilot)
        clock["mono"] += 1
        delay.set()
        await flush(app)
        await pilot.press("o", "c")
        clock["mono"] += 2
        await pilot.press("j")
        await settle(pilot)
        await flush(app)
        events = [event for event in interaction_events(fake) if event["story_id"] == 4]
        assert {event["event_type"] for event in events} == {
            "impression",
            "dwell",
            "article_open",
            "comments_open",
        }
        for event in events:
            assert event["surface"] == "queue"
            assert event["queue_position"] == 1
            assert event["sort_mode"] == "explore"
            assert event["position"] == 2
        popular = next(item for item in app.stories if item.id == 7)
        ctx = app._dwell_ctx(popular)
        assert ctx is not None
        assert (ctx.sort_mode, ctx.position, ctx.queue_position) == ("popular", 2, 6)
        app.query_one("#sort", Select).value = "recommended"
        await settle(pilot)
        app.rated.clear()
        app.rebuild()
        ctx = app._dwell_ctx(app.stories[0])
        assert ctx is not None and ctx.surface == "views" and ctx.queue_position is None
        await flush(app)


async def test_queue_done_undo_refresh_and_window_switch() -> None:
    fake = FakeServer()
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        first, second = app.stories
        app.apply_vote(first)
        app.apply_vote(second)
        await settle(pilot)
        assert (
            "Caught up for this window" in app.query_one("#summary", Markdown)._markdown
        )
        assert "queue" in app.vote_views[first.id][1]
        app.set_feed(
            replace(
                fake.feed,
                stories=[second],
                orders={"recommended": [2], "popular": [], "explore": [2]},
            )
        )
        app.apply_undo(first)
        await settle(pilot)
        assert app.view_order("queue") == [1, 2]
        assert [item.id for item in app.stories] == [1]
        assert app.view_ranks[first.id] == 0
        assert app.feed is not None and "queue" not in app.feed.orders
        app.apply_undo(second)
        assert [item.id for item in app.stories] == [1, 2]
        app.query_one("#window", Select).value = "archive"
        await settle(pilot)
        assert [item.id for item in app.stories] == [3]
        app.query_one("#window", Select).value = "1w"
        await settle(pilot)
        assert [item.id for item in app.stories] == [1, 2]
        await flush(app)


async def test_queue_is_unshuffled_and_advances_past_visible_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeServer()
    fake.feed = replace(queue_feed(), orders={"recommended": list(range(1, 15))})
    monkeypatch.setattr("random.shuffle", lambda order: order.reverse())
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert [item.id for item in app.stories] == list(range(1, 9))
        for _ in range(8):
            app.apply_vote(app.stories[0])
        await settle(pilot)
        assert [item.id for item in app.stories] == list(range(9, 15))
        for item in list(app.stories):
            app.unavailable.add(item.id)
        app.rebuild()
        assert (
            "Caught up for this window" in app.query_one("#summary", Markdown)._markdown
        )


async def test_same_story_switch_changes_event_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = install_clocks(monkeypatch)
    delay = gate_impression_sleep(monkeypatch)
    fake = FakeServer()
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        delay.set()
        await flush(app)
        clock["mono"] += 2
        app.query_one("#sort", Select).value = "recommended"
        await settle(pilot)
        await flush(app)
        events = interaction_events(fake)
        dwells = [event for event in events if event["event_type"] == "dwell"]
        assert dwells[-1]["surface"] == "queue"
        assert dwells[-1]["queue_position"] == 0
        impressions = [event for event in events if event["event_type"] == "impression"]
        assert [event["surface"] for event in impressions] == ["queue", "views"]
        assert "queue_position" not in impressions[-1]
        assert app.dwell_tracker.current is not None
        assert app.dwell_tracker.current.surface == "views"


async def test_frozen_queue_reaches_explore_popular_then_appended_recommended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_clocks(monkeypatch)
    gate_impression_sleep(monkeypatch).set()

    class RefillingServer(FakeServer):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/feedback"):
                import json

                sid = json.loads(request.content)["story_id"]
                version = self.feed.version + 1
                remaining = [item for item in self.feed.stories if item.id != sid]
                available = {item.id for item in remaining}
                # Two replacement Rec cards; fresh Explore/Popular stay out.
                rec = [item for item in [1, 4, 5] if item in available][:1]
                self.feed = replace(
                    self.feed,
                    stories=remaining,
                    orders={
                        "recommended": rec,
                        "explore": [item for item in [2, 6] if item in available],
                        "popular": [item for item in [3, 7] if item in available],
                    },
                    version=version,
                    target_version=version,
                )
                self.requests.append(request)
                return httpx.Response(200, json={"ok": True, "target_version": version})
            return await super().__call__(request)

    fake = RefillingServer()
    fake.feed = replace(
        queue_feed(),
        stories=[story(sid) for sid in range(1, 8)],
        orders={"recommended": [1], "explore": [2], "popular": [3]},
    )
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        for slot, (sid, origin) in enumerate(
            [
                (1, "recommended"),
                (2, "explore"),
                (3, "popular"),
                (4, "recommended"),
                (5, "recommended"),
            ]
        ):
            selected = app.selected()
            assert selected is not None and selected.id == sid, (
                slot,
                app.stories,
                app.queue_snapshots,
                app.feed,
                app.last_error,
            )
            ctx = app._dwell_ctx(selected)
            assert ctx is not None
            assert (ctx.queue_position, ctx.sort_mode, ctx.position) == (
                slot,
                origin,
                0,
            )
            # A same-window refresh cannot move this card's origin or slot.
            app.refresh_feed(announce=False)
            await settle(pilot)
            assert app._dwell_ctx(selected) == ctx
            await pilot.press("1")
            await settle(pilot)
            app.refresh_feed(announce=False)
            await settle(pilot)
        assert app.view_order("queue") == [1, 2, 3, 4, 5]
        assert not app.stories
        assert app.query_one("#summary", Markdown)._markdown.startswith(
            "# Caught up for this window"
        )
        await flush(app)


async def test_queue_skips_aged_out_slots_and_restores_missing_undo() -> None:
    fake = FakeServer()
    fake.feed = replace(
        queue_feed(),
        stories=[story(sid) for sid in range(1, 5)],
        orders={"recommended": [1, 2], "explore": [3], "popular": [4]},
    )
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        first = app.stories[0]
        app.apply_vote(first)
        app.set_feed(
            replace(fake.feed, stories=[story(4)], orders={"recommended": [4]})
        )
        app.rebuild()
        assert [item.id for item in app.stories] == [4]
        assert app.view_ranks[4] == 3
        app.apply_undo(first)
        await settle(pilot)
        assert [item.id for item in app.stories] == [1, 4]
        # A later ready deck may still omit the undone story.
        fake.feed = replace(fake.feed, stories=[story(4)], orders={"recommended": [4]})
        app.refresh_feed(announce=False)
        await settle(pilot)
        assert [item.id for item in app.stories] == [1, 4]
        assert app.view_ranks[1] == 0
        await flush(app)


async def test_queue_snapshots_survive_windows_but_reset_on_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeServer()
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        original = list(app.queue_snapshots["1w"])
        # Reading the queue's order must not change any stored origin or snapshot.
        before = dict(app.queue_origins)
        assert app.view_order("queue") == [1, 2]
        assert app.queue_origins == before
        app.query_one("#window", Select).value = "archive"
        await settle(pilot)
        assert app.queue_snapshots["1w"] == original
        app.query_one("#window", Select).value = "1w"
        await settle(pilot)
        assert app.queue_snapshots["1w"] == original
        fake.feed = replace(
            fake.feed,
            stories=[story(8)],
            orders={"recommended": [8]},
        )
        fresh_api = fake.api()
        monkeypatch.setattr(reader_module, "API", lambda srv, tok: fresh_api)
        await app.connected(Profile("https://example.org/hn/", "new-profile"))
        await settle(pilot)
        assert app.queue_snapshots == {"1w": [QueueEntry(8, "recommended", 0)]}
        assert [item.id for item in app.stories] == [8]
        await flush(app)
