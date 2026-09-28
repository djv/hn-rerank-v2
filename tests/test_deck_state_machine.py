"""Stateful model of per-user decks and versions (server.Handler).

Random sequences of votes, clears, regens, restarts, reads and finished
warms, checked against what clients rely on:

- a user's version never goes down, across restarts too;
- the page, `/api/feed` and `/api/ranking-ready` agree on the deck version;
- `ready` means the served deck has caught up;
- a stale deck always has a warm queued for the current version;
- a user without votes gets the shared cold deck and is never cached;
- a voted story is never served;
- a vote that changes nothing leaves the version alone.

Warms run synchronously: a fake scheduler records requests the way
`WarmScheduler` coalesces them (newest version wins), and a rule lands them
through the real `_run_warm_job`. Ranking reads votes and the pool when it
starts, and a regen or vote can land before it finishes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pytest
from flask.testing import FlaskClient
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    run_state_machine_as_test,
)

import pipeline
import server
from database import Database, Story, User
from pipeline import Config, Embedder, RankedStory, WindowDeck, WindowViews
from server import Handler, create_app

STORY_IDS = list(range(1, 9))
POOLS = st.lists(st.sampled_from(STORY_IDS), unique=True, max_size=len(STORY_IDS))
USERS = 3
STEP_MS = 1000


class _FakeScheduler:
    """Records warm requests the way WarmScheduler coalesces them."""

    def __init__(self) -> None:
        self.pending: dict[int, tuple[User, int]] = {}

    def request(
        self,
        key: int,
        payload: User,
        version: int,
        delay_s: float = 0.0,
        *,
        expedite: bool = True,
    ) -> None:
        queued = self.pending.get(key)
        if queued is None or version > queued[1]:
            self.pending[key] = (payload, version)


def _ranked(story_id: int) -> RankedStory:
    posted = int(time.time()) - 3600 - story_id
    story = Story(story_id, f"Story {story_id}", None, 10, posted, "text")
    return RankedStory(story, score=float(story_id), best_match_title="")


def _deck(pool: list[RankedStory]) -> WindowDeck:
    """The pool as the default window's Recommended view."""
    return WindowDeck({"1w": WindowViews(recommended=tuple(pool))})


class DeckMachine(RuleBasedStateMachine):
    db: Database

    def __init__(self) -> None:
        super().__init__()
        with self.db.conn() as conn:
            with conn:
                conn.execute("DELETE FROM feedback")
                conn.execute("DELETE FROM users")
        self.clock_ms = 1_000_000

        class Runtime(Handler):
            pass

        Runtime.config = Config(
            db_path=self.db.db_path,
            server_port=0,
            article_fetch_max_per_run=0,
            tldr_prefetch_per_view=0,
            tldr_prefetch_stale_per_run=0,
        )
        Runtime.db = self.db
        Runtime.embedder = cast(Embedder, None)
        Runtime.regen_event = threading.Event()
        Runtime._feedback_warm_counts = {}
        Runtime._feedback_warm_guard = threading.Lock()
        Runtime.reset_public_demo_limiter()
        self.runtime = Runtime
        self.pool = [_ranked(i) for i in STORY_IDS[:5]]
        self.boot()

        app = create_app(Runtime)
        self.users = [self.db.create_user(f"user-{i}") for i in range(USERS)]
        self.clients: list[FlaskClient] = []
        for user in self.users:
            client = app.test_client()
            client.set_cookie("hn_token", user.token)
            self.clients.append(client)
        self.votes: list[dict[int, str]] = [{} for _ in self.users]
        self.seen = [0 for _ in self.users]
        self.during_rank: Callable[[], None] | None = None
        self.ranked_at = 0

    # Model helpers -----------------------------------------------------

    def boot(self) -> None:
        runtime = self.runtime
        runtime._decks = {}
        runtime._dashboard_versions = {}
        runtime._pool_generation = self.clock_ms
        runtime._scheduler = cast(Any, _FakeScheduler())
        runtime._feedback_warm_counts = {}
        runtime._cold_deck = _deck(self.pool)

    @property
    def scheduler(self) -> _FakeScheduler:
        return cast(_FakeScheduler, self.runtime._scheduler)

    def current(self, index: int) -> int:
        return self.runtime._dashboard_version(self.users[index].id)

    def rank(self, user_id: int) -> WindowDeck:
        """`fast_rerank_for_user`: this user's pool minus their votes."""
        self.ranked_at = self.runtime._dashboard_version(user_id)
        voted = pipeline._voted_story_ids(self.db, user_id)
        deck = _deck([r for r in self.pool if r.story.id not in voted])
        hook, self.during_rank = self.during_rank, None
        if hook is not None:
            hook()
        return deck

    def post_vote(self, index: int, story_id: int, action: str) -> None:
        before = self.current(index)
        response = self.clients[index].post(
            "/api/feedback", json={"story_id": story_id, "action": action}
        )
        assert response.status_code == 200, response.get_data(as_text=True)
        target = response.get_json()["target_version"]
        votes = self.votes[index]
        changed = votes.get(story_id) != (None if action == "clear" else action)
        if action == "clear":
            votes.pop(story_id, None)
        else:
            votes[story_id] = action
        assert target == self.current(index)
        assert target > before if changed else target == before

    # Rules --------------------------------------------------------------

    @rule(
        index=st.integers(0, USERS - 1),
        story_id=st.sampled_from(STORY_IDS),
        action=st.sampled_from(["up", "neutral", "down"]),
    )
    def vote(self, index: int, story_id: int, action: str) -> None:
        self.post_vote(index, story_id, action)

    @rule(index=st.integers(0, USERS - 1), story_id=st.sampled_from(STORY_IDS))
    def clear(self, index: int, story_id: int) -> None:
        self.post_vote(index, story_id, "clear")

    @rule(ids=POOLS)
    def regen(self, ids: list[int]) -> None:
        self.pool = [_ranked(i) for i in ids]
        self.runtime._pool_changed()

    @rule()
    def restart(self) -> None:
        self.boot()

    @rule(
        index=st.integers(0, USERS - 1),
        regen_ids=st.none() | POOLS,
        vote_on=st.none() | st.sampled_from(STORY_IDS),
    )
    def warm_lands(
        self, index: int, regen_ids: list[int] | None, vote_on: int | None
    ) -> None:
        user_id = self.users[index].id
        queued = self.scheduler.pending.pop(user_id, None)
        if queued is None:
            return

        def meanwhile() -> None:
            if regen_ids is not None:
                self.regen(regen_ids)
            if vote_on is not None:
                self.post_vote(index, vote_on, "up")

        self.during_rank = meanwhile
        self.runtime._run_warm_job(*queued)
        # Labeled no older than the votes and pool it was ranked from.
        assert self.runtime._decks[user_id].version >= self.ranked_at

    @rule(index=st.integers(0, USERS - 1))
    def read(self, index: int) -> None:
        client = self.clients[index]
        feed = client.get("/api/feed").get_json()
        version, target = feed["version"], feed["target_version"]
        ready = client.get(f"/api/ranking-ready?min_version={target}").get_json()
        page = client.get("/").get_data(as_text=True)

        assert ready["current_version"] == target == self.current(index)
        assert feed["ready"] is ready["ready"] is (version >= target)
        if feed["stories"] or version:
            assert page == f"v={version}/{target}"
        else:
            # An empty cold deck while the first warm runs.
            assert page == server.SKELETON_HTML.decode()
        served = {story["id"] for story in feed["stories"]}
        assert not served & set(self.votes[index])

        user_id = self.users[index].id
        if not self.votes[index]:
            assert version == target
            assert served == {r.story.id for r in self.pool}
            assert user_id not in self.runtime._decks
        elif version < target:
            queued = self.scheduler.pending.get(user_id)
            assert queued is not None and queued[1] == target

    # Invariants ---------------------------------------------------------

    @invariant()
    def tick(self) -> None:
        # Each step takes a second, so a restart boots at a later time.
        self.clock_ms += STEP_MS

    @invariant()
    def versions_never_decrease(self) -> None:
        for index in range(len(self.users)):
            current = self.current(index)
            assert current >= self.seen[index]
            self.seen[index] = current

    @invariant()
    def decks_are_never_ahead_and_stale_ones_have_a_warm(self) -> None:
        for user in self.users:
            deck = self.runtime._decks.get(user.id)
            if deck is None:
                continue
            current = self.runtime._dashboard_version(user.id)
            assert deck.version <= current
            if deck.version < current:
                queued = self.scheduler.pending.get(user.id)
                assert queued is not None and queued[1] == current


@pytest.fixture
def deck_db() -> Iterator[Database]:
    with TemporaryDirectory() as temp_dir:
        db = Database(str(Path(temp_dir) / "deck_machine.db"))
        for story_id in STORY_IDS:
            db.upsert_story(_ranked(story_id).story)
        yield db
        db.close()


def test_deck_versions_state_machine(
    deck_db: Database,
    monkeypatch: pytest.MonkeyPatch,
    hypothesis_examples: Callable[[int], int],
) -> None:
    def fake_rank(
        db: Database, config: Config, embedder: Embedder, user_id: int, **_: Any
    ) -> WindowDeck:
        assert Machine.live is not None
        return Machine.live.rank(user_id)

    def fake_render(deck: WindowDeck, *_: Any, **kwargs: Any) -> bytes:
        return f"v={kwargs['dashboard_version']}/{kwargs['dashboard_latest_version']}".encode()

    class Machine(DeckMachine):
        db = deck_db
        live: DeckMachine | None = None

        def __init__(self) -> None:
            Machine.live = self
            super().__init__()

        @classmethod
        def current_pool(cls) -> list[RankedStory]:
            assert cls.live is not None
            return cls.live.pool

    def rebuild_cold_deck(cls: type[Handler]) -> None:
        cls._cold_deck = _deck(Machine.current_pool())

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_rank)
    monkeypatch.setattr(pipeline, "generate_dashboard_bytes", fake_render)
    monkeypatch.setattr(Handler, "_rebuild_cold_deck", classmethod(rebuild_cold_deck))
    monkeypatch.setattr(
        Handler, "_collect_after_warm_attempt", classmethod(lambda cls: None)
    )
    run_state_machine_as_test(
        Machine,
        settings=settings(
            max_examples=hypothesis_examples(60),
            stateful_step_count=30,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow],
        ),
    )
