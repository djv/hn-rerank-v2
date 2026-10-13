from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from textual.content import Content
from textual.widgets import Markdown, OptionList, Select, Tabs

import hn_rerank.app as reader_module
from hn_rerank.api import Profile
from hn_rerank.app import Reader
from hn_rerank.models import Feed, FeedBadge, FeedStory
from hn_rerank.queue import (
    QueueEntry,
    build_round,
    default_sort,
    in_window,
    queue_entries,
)

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
        int(time.time()),
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
        QueueEntry(1, "recommended", 0, 0),
        QueueEntry(4, "explore", 2, 0),
        QueueEntry(5, "explore", 3, 0),
        QueueEntry(3, "explore", 1, 0),
        QueueEntry(2, "explore", 0, 0),
        QueueEntry(6, "explore", 4, 0),
        QueueEntry(7, "popular", 3, 0),
        QueueEntry(8, "popular", 5, 0),
        QueueEntry(9, "popular", 6, 0),
        QueueEntry(10, "popular", 7, 0),
        QueueEntry(11, "popular", 8, 0),
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
    decks=st.lists(
        st.tuples(
            st.lists(st.integers(1, 40), unique=True, max_size=20),
            st.lists(st.integers(1, 40), unique=True, max_size=20),
            st.lists(st.integers(1, 40), unique=True, max_size=20),
            st.sets(st.integers(1, 40), max_size=20),
            st.integers(0, 1000),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_rounds_preserve_prefix_composition_versions_and_uniqueness(
    decks: list[tuple[list[int], list[int], list[int], set[int], int]],
) -> None:
    rounds: list[list[QueueEntry]] = []
    for rec, explore, popular, excluded, version in decks:
        feed = replace(
            queue_feed(),
            stories=[story(sid) for sid in range(1, 41)],
            orders={"recommended": rec, "explore": explore, "popular": popular},
            version=version,
        )
        before = [entry for batch in rounds for entry in batch]
        seen = {entry.story_id for entry in before}
        expected = queue_entries(feed, seen | excluded)
        updated = build_round(rounds, feed, excluded)
        flat = [entry for batch in updated for entry in batch]
        assert flat[: len(before)] == before
        assert flat[len(before) :] == expected
        assert len(flat) == len({entry.story_id for entry in flat})
        assert all(entry.served_version == version for entry in expected)
        assert updated[: len(rounds)] == rounds
        rounds = updated


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
        fake.feed,
        stories=[replace(item, time=int(time.time())) for item in fake.feed.stories],
    )
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
            assert event["dashboard_version"] == 0
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
    fake.feed = replace(
        fake.feed,
        stories=[replace(item, time=int(time.time())) for item in fake.feed.stories],
    )
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
    fake.feed = replace(
        fake.feed,
        stories=[replace(item, time=int(time.time())) for item in fake.feed.stories],
    )
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
    fake.feed = replace(
        fake.feed,
        stories=[replace(item, time=int(time.time())) for item in fake.feed.stories],
    )
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


async def test_rounds_reach_churned_cards_then_fresh_explore_and_popular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_clocks(monkeypatch)
    gate_impression_sleep(monkeypatch).set()

    class RefillingServer(FakeServer):
        voted: set[int]

        def __init__(self) -> None:
            super().__init__()
            self.voted = set()

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/feedback"):
                import json

                self.voted.add(json.loads(request.content)["story_id"])
                version = self.feed.version + 1
                # Frozen Explore 2 and Popular 3 churn OUT after the first vote.
                # Rec refill 5 lands behind the still-served original Rec 4.
                orders = {
                    "recommended": [sid for sid in [4, 5] if sid not in self.voted],
                    "explore": [sid for sid in [6] if sid not in self.voted],
                    "popular": [sid for sid in [7] if sid not in self.voted],
                }
                available = {sid for order in orders.values() for sid in order}
                self.feed = replace(
                    self.feed,
                    stories=[
                        replace(story(sid), points=100 + version)
                        for sid in sorted(available)
                    ],
                    orders=orders,
                    version=version,
                    target_version=version,
                )
                self.requests.append(request)
                return httpx.Response(200, json={"ok": True, "target_version": version})
            return await super().__call__(request)

    fake = RefillingServer()
    fake.feed = replace(
        queue_feed(),
        stories=[story(sid) for sid in [1, 4, 2, 3]],
        orders={"recommended": [1, 4], "explore": [2], "popular": [3]},
    )
    app = Reader(
        api=fake.api(),
        prefetch=0,
        utc_clock=lambda: datetime.fromordinal(739899).replace(tzinfo=UTC),
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        expected = [
            (1, "recommended", 0),
            (4, "recommended", 1),
            (2, "explore", 0),
            (3, "popular", 0),
            (5, "recommended", 0),
            (6, "explore", 0),
            (7, "popular", 0),
        ]
        for slot, (sid, origin, rank) in enumerate(expected):
            selected = app.selected()
            assert selected is not None and selected.id == sid
            if sid == 4:
                assert selected.points == 101  # present cards refresh their data
            if sid == 2:
                assert selected.points == 100  # churned cards retain frozen data
            ctx = app._dwell_ctx(selected)
            assert ctx is not None
            assert (ctx.queue_position, ctx.sort_mode, ctx.position) == (
                slot,
                origin,
                rank,
            )
            assert ctx.dashboard_version == (0 if slot < 4 else 3)
            app.refresh_feed(announce=False)
            await settle(pilot)
            refreshed = app.selected()
            assert refreshed is not None
            assert app._dwell_ctx(refreshed) == ctx
            await pilot.press("1")
            await settle(pilot)
            app.refresh_feed(announce=False)
            await settle(pilot)
        assert app.view_order("queue") == [1, 4, 2, 3, 5, 6, 7]
        assert len(app.queue_rounds["1w"]) == 2
        assert not app.stories
        assert app.query_one("#summary", Markdown)._markdown.startswith(
            "# Caught up for this window"
        )
        await flush(app)
        events = interaction_events(fake)
        for slot, (sid, origin, rank) in enumerate(expected):
            event = next(
                e
                for e in events
                if e["story_id"] == sid and e["event_type"] == "impression"
            )
            assert (
                event["dashboard_version"],
                event["sort_mode"],
                event["position"],
                event["queue_position"],
            ) == (0 if slot < 4 else 3, origin, rank, slot)
        # A later feed resumes automatically; the refill has a non-zero served rank.
        fake.feed = replace(
            fake.feed,
            stories=[story(7), story(8)],
            orders={"recommended": [7, 8]},
            version=8,
            target_version=8,
        )
        app.refresh_feed(announce=False)
        await settle(pilot)
        assert [item.id for item in app.stories] == [8]
        ctx = app._dwell_ctx(app.stories[0])
        assert ctx is not None
        assert (ctx.position, ctx.queue_position, ctx.dashboard_version) == (1, 7, 8)
        status = app.query_one("#status").render()
        assert isinstance(status, Content)
        assert "Round 3" in status.plain
        await flush(app)


async def test_queue_keeps_missing_cards_and_restores_missing_undo() -> None:
    fake = FakeServer()
    fake.feed = replace(
        fake.feed,
        stories=[replace(item, time=int(time.time())) for item in fake.feed.stories],
    )
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
        assert [item.id for item in app.stories] == [2, 3, 4]
        assert app.view_ranks[4] == 3
        app.apply_undo(first)
        await settle(pilot)
        assert [item.id for item in app.stories] == [1, 2, 3, 4]
        # A later ready deck may still omit the undone story.
        fake.feed = replace(fake.feed, stories=[story(4)], orders={"recommended": [4]})
        app.refresh_feed(announce=False)
        await settle(pilot)
        assert [item.id for item in app.stories] == [1, 2, 3, 4]
        assert app.view_ranks[1] == 0
        await flush(app)


async def test_queue_snapshots_survive_windows_but_reset_on_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeServer()
    fake.feed = replace(
        fake.feed,
        stories=[replace(item, time=int(time.time())) for item in fake.feed.stories],
    )
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
        assert app.queue_snapshots == {"1w": [QueueEntry(8, "recommended", 0, 0)]}
        assert [item.id for item in app.stories] == [8]
        await flush(app)


@pytest.mark.parametrize(
    "window,seconds", [("12h", 43200), ("1d", 86400), ("1w", 604800), ("1m", 2592000)]
)
def test_age_boundaries(window: str, seconds: int) -> None:
    from typing import cast

    from hn_rerank.models import Window

    assert in_window(100, cast(Window, window), 100 + seconds)
    assert not in_window(100, cast(Window, window), 101 + seconds)
    assert in_window(100, cast(Window, window), 99)
    assert in_window(100, "archive", 10**12)


async def test_age_out_restored_and_queue_cards_and_views_vote_metadata() -> None:
    fake = FakeServer()
    # Just past 1w: still served by the deck below, then aged out of it.
    fresh = replace(story(1), time=int(time.time()) - 604801)
    fake.feed = replace(queue_feed(), stories=[fresh], orders={"recommended": [1]})
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        # A carried views row must not fabricate position zero.
        app.view_ranks.clear()
        assert app._dwell_ctx(app.stories[0]) is None
        app.rebuild()
        app.apply_vote(app.stories[0])
        assert "queue" not in app.vote_views[1][1]
        app.apply_undo(fresh)
        old = fresh
        app.restored[1] = old
        # The server stops serving a story once it ages out of the window.
        fake.feed = replace(fake.feed, stories=[], orders={})
        app.set_feed(fake.feed)
        app.rebuild()
        assert not app.stories
        assert not app.restored
        app.query_one("#sort", Select).value = "queue"
        await settle(pilot)
        assert not app.stories
        await flush(app)


def test_restored_rows_do_not_count_in_served_ranks() -> None:
    feed = replace(queue_feed(), orders={"recommended": [1, 2, 3]}, version=7)
    assert build_round([], feed, set(), {2}) == [
        [QueueEntry(1, "recommended", 0, 7), QueueEntry(3, "recommended", 1, 7)]
    ]
