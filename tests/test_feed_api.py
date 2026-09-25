from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from werkzeug.test import TestResponse

from bs4 import BeautifulSoup

from clients.tui.src.hn_rerank.models import Feed
from database import Database, Story, User
from pipeline import Config, RankedStory
from server import DeckState, Handler, create_app


def payload(response: TestResponse) -> Any:
    data = response.get_json()
    assert isinstance(data, dict)
    return data


def test_feed_parity_authentication_stale_cache_and_eviction(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "feed.db"))

    class Runtime(Handler):
        _decks = {}
        _dashboard_versions = {}
        _cold_stories = []

        @classmethod
        def _trigger_warm(cls, user: User, version: int, delay_s: float = 0.0) -> None:
            pass

    Runtime.db = db
    Runtime.config = Config(db_path=db.db_path)
    Runtime.regen_event = threading.Event()
    user = db.create_user("feed-user")
    other = db.create_user("other-user")
    ranked = [
        RankedStory(
            Story(
                i,
                f"Story {i}",
                f"https://example.org/{i}",
                i * 10,
                i * 100,
                "text",
                comment_count=i,
                discussion_url=f"https://news.ycombinator.com/item?id={i}",
            ),
            score=float(4 - i),
            best_match_title="",
            combo_keys="recent_hn recent_mixed"
            if i < 3
            else "archive_hn archive_mixed",
            is_hot=i == 1,
            is_novel=i == 2,
        )
        for i in (1, 2, 3)
    ]
    Runtime._cold_stories = ranked
    client = create_app(Runtime).test_client()
    assert client.get("/api/feed").status_code == 401
    client.set_cookie("hn_token", user.token)
    response = client.get("/api/feed")
    feed = payload(response)
    parsed = Feed.parse(feed)
    assert parsed.api_version == 1
    assert [story.badges for story in parsed.stories] == [
        ["\U0001f525"],
        ["\u2728"],
        [],
    ]
    assert parsed.stories[0].badge_details[0].kind == "hot"
    assert parsed.stories[0].badge_details[0].icon == "🔥"
    assert parsed.stories[0].badge_details[0].tooltip
    assert parsed.stories[0].domain == "example.org"
    assert [story.id for story in parsed.stories] == [
        story["id"] for story in feed["stories"]
    ]
    assert response.headers["Cache-Control"] == "no-store"
    # No votes: the shared cold deck is this user's current deck.
    assert feed["version"] == 1 and feed["ready"] is True
    html = client.get("/").data
    cards = BeautifulSoup(html, "html.parser").select(".story-card")
    assert [int(str(card["data-story-id"])) for card in cards] == [
        story["id"] for story in feed["stories"]
    ]
    for age in ("recent", "archive"):
        for sort in ("recommended", "popular", "explore", "date"):
            matching = [
                c
                for c in cards
                if f"{age}_mixed" in str(c["data-combo"]).split()
                and (
                    sort not in {"popular", "explore"} or c[f"data-sort-{sort}"] == "1"
                )
            ]
            matching.sort(
                key=lambda c: float(
                    str(c["data-time" if sort == "date" else "data-score"])
                ),
                reverse=True,
            )
            assert feed["orders"][f"{sort}:{age}"] == [
                int(str(c["data-story-id"])) for c in matching
            ]
    Runtime._decks[user.id] = DeckState(ranked, time.time(), 1)
    for item in ranked:
        db.upsert_story(item.story)
    vote = client.post("/api/feedback", json={"story_id": 1, "action": "up"})
    assert payload(vote)["target_version"] == 2
    stale = payload(client.get("/api/feed"))
    assert not Feed.parse(stale).ready
    assert stale["stories"] == feed["stories"]
    assert stale["version"] == 1 and stale["target_version"] == 2 and not stale["ready"]
    assert (
        BeautifulSoup(client.get("/").data, "html.parser").select(".story-card")
        == cards
    )
    client.set_cookie("hn_token", other.token)
    assert payload(client.get("/api/feed"))["feedback_counts"]["up"] == 0
    client.set_cookie("hn_token", user.token)
    Runtime._decks[user.id] = DeckState(ranked[1:], time.time(), 2)
    assert payload(client.get("/api/feed"))["orders"]["recommended:recent"] == [2]
    assert payload(client.get("/api/feed"))["feedback_counts"]["up"] == 1
    assert (
        payload(client.post("/api/feedback", json={"story_id": 1, "action": "clear"}))[
            "target_version"
        ]
        == 3
    )
    Runtime._MAX_CACHED_DECKS = 1
    Runtime._decks[other.id] = DeckState(ranked, time.time() + 1, 1)
    with Runtime._dashboard_versions_guard:
        Runtime._evict_old_decks_locked()
    assert user.id not in Runtime._decks
    Runtime._cold_stories = []
    empty = payload(client.get("/api/feed"))
    assert not Feed.parse(empty).stories
    assert not empty["ready"] and empty["stories"] == []
    db.close()
