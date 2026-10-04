from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from werkzeug.test import TestResponse

from bs4 import BeautifulSoup

from clients.tui.src.hn_rerank.models import FEED_API_VERSION, WINDOWS, Feed
from database import Database, Story, User
from pipeline import Config, RankedStory, WindowDeck, WindowViews
from server import DeckState, Handler, create_app


def payload(response: TestResponse) -> Any:
    data = response.get_json()
    assert isinstance(data, dict)
    return data


def page_feed(client: Any) -> Any:
    """The deck the page carries for the client."""
    script = BeautifulSoup(client.get("/").data, "html.parser").select_one(
        "script#feed-data"
    )
    assert script is not None
    return json.loads(script.get_text())


class _Runtime(Handler):
    _decks = {}
    _dashboard_versions = {}
    _cold_deck = WindowDeck()
    _pool_generation = 1

    @classmethod
    def _trigger_warm(
        cls,
        user: User,
        version: int,
        delay_s: float = 0.0,
        *,
        expedite: bool = True,
    ) -> None:
        pass


def _runtime(db: Database) -> type[Handler]:
    class Runtime(_Runtime):
        _decks = {}
        _dashboard_versions = {}

    Runtime.db = db
    Runtime.config = Config(db_path=db.db_path)
    Runtime.regen_event = threading.Event()
    return Runtime


def _story(i: int, age_s: float) -> Story:
    return Story(
        i,
        f"Story {i}",
        f"https://example.org/{i}",
        i * 10,
        int(time.time() - age_s),
        "text",
        comment_count=i,
        discussion_url=f"https://news.ycombinator.com/item?id={i}",
    )


def test_feed_parity_authentication_stale_cache_and_eviction(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "feed.db"))
    Runtime = _runtime(db)
    user = db.create_user("feed-user")
    other = db.create_user("other-user")
    ranked = [
        RankedStory(
            _story(i, 3600 * i),
            score=float(4 - i),
            best_match_title="",
            is_hot=i == 1,
            is_novel=i == 2,
        )
        for i in (1, 2, 3)
    ]
    week = WindowViews(
        recommended=tuple(ranked[1:]),
        popular=(ranked[0],),
        explore=(ranked[1],),
    )
    deck = WindowDeck({"1w": week})
    Runtime._cold_deck = deck
    client = create_app(Runtime).test_client()
    assert client.get("/api/feed").status_code == 401
    client.set_cookie("hn_token", user.token)
    response = client.get("/api/feed")
    feed = payload(response)
    parsed = Feed.parse(feed)
    assert parsed.api_version == FEED_API_VERSION and parsed.window == "1w"
    # One card per story, badges from every view of the window it is in.
    assert [story.id for story in parsed.stories] == [2, 3, 1]
    assert [story.badges for story in parsed.stories] == [
        ["\u2728"],
        [],
        ["\U0001f525"],
    ]
    assert parsed.orders == {"recommended": [2, 3], "popular": [1], "explore": [2]}
    hot = parsed.stories[2].badge_details[0]
    assert (hot.kind, hot.icon) == ("hot", "🔥") and hot.tooltip
    assert parsed.stories[0].domain == "example.org"
    assert response.headers["Cache-Control"] == "no-store"
    # No votes: the shared cold deck is this user's current deck.
    assert feed["version"] == 1 and feed["ready"] is True
    assert page_feed(client) == feed
    Runtime._decks[user.id] = DeckState(deck, time.time(), 1)
    for item in ranked:
        db.upsert_story(item.story)
    vote = client.post("/api/feedback", json={"story_id": 1, "action": "up"})
    assert payload(vote)["target_version"] == 2
    # Repeating a vote changes nothing, so nothing new is ranked.
    again = client.post("/api/feedback", json={"story_id": 1, "action": "up"})
    assert payload(again)["target_version"] == 2
    stale = payload(client.get("/api/feed"))
    assert not Feed.parse(stale).ready
    # The stale deck is served without the story just voted on, in every view.
    assert [s["id"] for s in stale["stories"]] == [2, 3]
    assert stale["orders"]["popular"] == []
    assert stale["version"] == 1 and stale["target_version"] == 2 and not stale["ready"]
    assert page_feed(client) == stale
    client.set_cookie("hn_token", other.token)
    assert payload(client.get("/api/feed"))["feedback_counts"]["up"] == 0
    client.set_cookie("hn_token", user.token)
    Runtime._decks[user.id] = DeckState(deck, time.time(), 2)
    assert payload(client.get("/api/feed"))["orders"]["recommended"] == [2, 3]
    assert payload(client.get("/api/feed"))["feedback_counts"]["up"] == 1
    assert (
        payload(client.post("/api/feedback", json={"story_id": 1, "action": "clear"}))[
            "target_version"
        ]
        == 3
    )
    Runtime._MAX_CACHED_DECKS = 1
    Runtime._decks[other.id] = DeckState(deck, time.time() + 1, 1)
    with Runtime._dashboard_versions_guard:
        Runtime._evict_old_decks_locked()
    assert user.id not in Runtime._decks
    # No votes left: the shared cold deck is current, even when it is empty.
    Runtime._cold_deck = WindowDeck()
    empty = payload(client.get("/api/feed"))
    assert not Feed.parse(empty).stories
    assert empty["ready"] and empty["stories"] == []
    assert empty["version"] == empty["target_version"] == 3
    db.close()


def test_feed_serves_the_requested_window_only(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "windows.db"))
    Runtime = _runtime(db)
    user = db.create_user("window-user")
    ages = {"12h": 3600, "1d": 20 * 3600, "1w": 3 * 86400, "1m": 20 * 86400}
    ages["archive"] = 90 * 86400
    stories = {
        w: RankedStory(_story(i, ages[w]), 1.0, "") for i, w in enumerate(ages, 1)
    }
    # Nested windows: every window up to 1m holds the stories younger than it.
    Runtime._cold_deck = WindowDeck(
        {
            w: WindowViews(
                recommended=tuple(stories[v] for v in WINDOWS[: WINDOWS.index(w) + 1])
                if w != "archive"
                else (stories["archive"],)
            )
            for w in WINDOWS
        }
    )
    client = create_app(Runtime).test_client()
    client.set_cookie("hn_token", user.token)
    default = payload(client.get("/api/feed"))
    assert default == payload(client.get("/api/feed?window=1w"))
    assert default["window"] == "1w"
    for i, window in enumerate(WINDOWS, 1):
        feed = Feed.parse(payload(client.get(f"/api/feed?window={window}")))
        assert feed.window == window
        expected = [i] if window == "archive" else list(range(1, i + 1))
        assert feed.orders["recommended"] == expected
        assert [s.id for s in feed.stories] == expected
    for bad in ("2d", "", "1W", "recent"):
        response = client.get(f"/api/feed?window={bad}")
        assert response.status_code == 400, bad
        assert "window" in payload(response)["error"]
    # The page embeds the default window.
    assert page_feed(client)["window"] == "1w"
    db.close()


def test_feed_serves_stored_counts_not_the_deck_snapshot(tmp_path: Path) -> None:
    """Decks hold story snapshots from the last pool build; points and
    comments refreshed since (hot refresh, TLDR hydration) are served from
    the DB by /api/feed and the page, while order stays as ranked."""
    db = Database(str(tmp_path / "counts.db"))
    Runtime = _runtime(db)
    user = db.create_user("counts-user")
    stored, unstored = _story(1, 3600), _story(2, 7200)
    db.upsert_story(stored)
    ranked = [
        RankedStory(s, score=1.0, best_match_title="") for s in (stored, unstored)
    ]
    Runtime._cold_deck = WindowDeck({"1w": WindowViews(recommended=tuple(ranked))})
    client = create_app(Runtime).test_client()
    client.set_cookie("hn_token", user.token)
    assert db.update_story_counts(1, 444, 218)

    def counts(feed: Any) -> list[tuple[int, int, int]]:
        return [(s["id"], s["points"], s["comments"]) for s in feed["stories"]]

    feed = payload(client.get("/api/feed"))
    # A story missing from the DB keeps its snapshot counts.
    assert counts(feed) == [(1, 444, 218), (2, 20, 2)]
    assert feed["orders"]["recommended"] == [1, 2]
    assert counts(page_feed(client)) == counts(feed)
    db.close()
