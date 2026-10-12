"""One summary generation per story, shared by taps and the warm prefetch."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

import server
from database import Database, Story, User
from pipeline import Config, RankedStory, WindowDeck, WindowViews
from server import Handler, TldrReply, TldrResult, create_app
from single_flight import Flight, SingleFlight

STORY = Story(
    4242,
    "Shared summary story",
    None,
    10,
    1_600_000_000,
    "text",
    source="hn",
    article_body="An article long enough to summarize without fetching anything.",
)


class _CountedFlights(SingleFlight[int, TldrReply]):
    """Signals each time a request joins a flight someone else leads."""

    def __init__(self) -> None:
        super().__init__()
        self.joined = threading.Semaphore(0)

    def join_or_lead(
        self, key: int, *, forced: bool = False
    ) -> tuple[Flight[TldrReply], bool]:
        flight, leading = super().join_or_lead(key, forced=forced)
        if not leading:
            self.joined.release()
        return flight, leading

    def followup_or_lead(
        self, ordinary: Flight[TldrReply], key: int
    ) -> tuple[Flight[TldrReply], bool]:
        flight, leading = super().followup_or_lead(ordinary, key)
        if not leading:
            self.joined.release()
        return flight, leading


class _Llm:
    """A generation that blocks until released, counting calls."""

    def __init__(self, result: TldrResult | None = None) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.result = result or TldrResult(kind="ok", tldr="## Summary\nshared")

    async def __call__(self, title: str, **_: str) -> TldrResult:
        self.calls += 1
        self.started.set()
        await asyncio.to_thread(self.release.wait, 5.0)
        if self.result.kind == "llm_error" and self.result.error_text == "raise":
            raise RuntimeError("provider exploded")
        return self.result


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[tuple[type[Handler], Database, User]]:
    db = Database(str(tmp_path / "flight.db"))
    db.upsert_story(STORY)

    class Runtime(Handler):
        _tldr_flights = _CountedFlights()

    # One generation slot and one uncached summary per user: a request that
    # joined must use neither.
    Runtime.config = Config(
        db_path=db.db_path,
        tldr_max_concurrent_generations=1,
        tldr_uncached_per_user_limit=1,
        tldr_uncached_per_user_window_seconds=3600,
        tldr_uncached_global_limit=100,
    )
    Runtime.db = db
    Runtime._tldr_generations = 0
    Runtime.reset_public_demo_limiter()
    yield Runtime, db, db.create_user("flight-user")
    db.close()


def _client(runtime: type[Handler], user: User) -> FlaskClient:
    client = create_app(runtime).test_client()
    client.set_cookie("hn_token", user.token)
    return client


def _tap(
    runtime: type[Handler], user: User, out: list[Any], force: bool = False
) -> threading.Thread:
    def run() -> None:
        response = _client(runtime, user).post(
            "/api/tldr-detail", json={"story_id": STORY.id, "force_refresh": force}
        )
        out.append((response.status_code, response.get_json()))

    thread = threading.Thread(target=run)
    thread.start()
    return thread


@pytest.mark.parametrize("followers", [1, 3])
def test_concurrent_taps_share_one_generation(
    runtime: tuple[type[Handler], Database, User],
    monkeypatch: pytest.MonkeyPatch,
    followers: int,
) -> None:
    handler, db, user = runtime
    llm = _Llm()
    monkeypatch.setattr(server, "generate_detailed_tldr", llm)
    flights = handler._tldr_flights
    assert isinstance(flights, _CountedFlights)
    replies: list[Any] = []

    leader = _tap(handler, user, replies)
    assert llm.started.wait(5.0)
    others = [_tap(handler, user, replies) for _ in range(followers)]
    for _ in others:
        assert flights.joined.acquire(timeout=5.0)
    llm.release.set()
    for thread in [leader, *others]:
        thread.join(5.0)

    assert llm.calls == 1
    assert len(replies) == followers + 1
    assert {status for status, _ in replies} == {200}
    assert all(body["tldr"] == "## Summary\nshared" for _, body in replies)
    assert handler._tldr_generations == 0
    assert not flights.in_flight(STORY.id)
    # The next tap is a plain cache hit.
    after = _client(handler, user).post("/api/tldr-detail", json={"story_id": STORY.id})
    assert after.get_json()["cached"] is True and llm.calls == 1


def test_forced_tap_leads_bounded_followup_after_ordinary(
    runtime: tuple[type[Handler], Database, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced tap never joins an ordinary flight as satisfaction: it waits
    once, then leads exactly one fresh follow-up. Two forced followers
    coalesce onto that follow-up (2 generations total, never 3+), and the
    fresh text wins the cache with its own snapshot."""
    import pipeline

    handler, db, user = runtime
    handler.config = replace(
        handler.config,
        tldr_uncached_per_user_limit=5,
        tldr_max_concurrent_generations=2,
    )

    async def fake_fetch_story(
        client: object,
        sid: int,
        db_: Database,
        *,
        force: bool = False,
        strict: bool = False,
    ) -> Story | None:
        assert force and strict
        row = db_.get_story(sid)
        assert row is not None
        return replace(
            row,
            top_comments="Fresh hydrated comments.",
            comment_count=42,
            comment_count_at_fetch=42,
        )

    monkeypatch.setattr(pipeline, "fetch_story", fake_fetch_story)

    made: list[str] = []
    first_started = threading.Event()
    release_leader = threading.Event()

    async def seq_llm(title: str, **_: str) -> TldrResult:
        made.append(title)
        if len(made) == 1:
            first_started.set()
            await asyncio.to_thread(release_leader.wait, 10.0)
            return TldrResult(kind="ok", tldr="## Summary ordinary")
        return TldrResult(kind="ok", tldr="## Summary forced fresh")

    monkeypatch.setattr(server, "generate_detailed_tldr", seq_llm)
    flights = handler._tldr_flights
    assert isinstance(flights, _CountedFlights)
    replies: list[Any] = []

    leader = _tap(handler, user, replies)
    assert first_started.wait(5.0)
    first = [_tap(handler, user, replies, force=True) for _ in range(2)]
    assert flights.joined.acquire(timeout=5.0)
    assert flights.joined.acquire(timeout=5.0)
    release_leader.set()
    # Exactly one follow-up starts; the second forced caller joins it.
    assert flights.joined.acquire(timeout=5.0)
    for thread in [leader, *first]:
        thread.join(10.0)

    assert len(made) == 2
    assert len(replies) == 3
    assert {status for status, _ in replies} == {200}
    assert replies[0][1]["tldr"] == "## Summary ordinary"
    assert replies[1][1]["tldr"] == "## Summary forced fresh"
    assert replies[2][1]["tldr"] == "## Summary forced fresh"
    assert not flights.in_flight(STORY.id)
    fresh_key = server._tldr_cache_key(
        title=STORY.title,
        self_text=STORY.self_text or "",
        top_comments="Fresh hydrated comments.",
        article_body=STORY.article_body or "",
    )
    assert db.get_tldr_cache(STORY.id, fresh_key) == "## Summary forced fresh"
    assert db.get_tldr_cache_snapshot(STORY.id, fresh_key) == 42


def test_late_forced_waiter_shares_completed_followup(
    runtime: tuple[type[Handler], Database, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced waiter that wakes after the fresh follow-up already landed
    still shares it: exactly ordinary + one force (2 generations), both
    forced replies carry the fresh text, and the cache stays fresh."""
    import pipeline

    handler, db, user = runtime
    handler.config = replace(
        handler.config,
        tldr_uncached_per_user_limit=5,
        tldr_max_concurrent_generations=2,
    )

    async def fake_fetch_story(
        client: object,
        sid: int,
        db_: Database,
        *,
        force: bool = False,
        strict: bool = False,
    ) -> Story | None:
        assert force and strict
        row = db_.get_story(sid)
        assert row is not None
        return replace(
            row,
            top_comments="Fresh hydrated comments.",
            comment_count=42,
            comment_count_at_fetch=42,
        )

    monkeypatch.setattr(pipeline, "fetch_story", fake_fetch_story)

    made: list[str] = []
    first_started = threading.Event()
    release_leader = threading.Event()

    async def seq_llm(title: str, **_: str) -> TldrResult:
        made.append(title)
        if len(made) == 1:
            first_started.set()
            await asyncio.to_thread(release_leader.wait, 10.0)
            return TldrResult(kind="ok", tldr="## Summary ordinary")
        return TldrResult(kind="ok", tldr="## Summary forced fresh")

    monkeypatch.setattr(server, "generate_detailed_tldr", seq_llm)
    flights = handler._tldr_flights
    assert isinstance(flights, _CountedFlights)
    fresh_key = server._tldr_cache_key(
        title=STORY.title,
        self_text=STORY.self_text or "",
        top_comments="Fresh hydrated comments.",
        article_body=STORY.article_body or "",
    )

    # Hold the second forced waiter inside followup_or_lead until the
    # first waiter's follow-up has fully landed (fresh cache written,
    # key released): no schedule luck, the slow path is forced.
    calls = 0

    def gated(ordinary: Flight[TldrReply], key: int) -> tuple[Flight[TldrReply], bool]:
        nonlocal calls
        calls += 1
        if calls == 2:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if db.get_tldr_cache(
                    STORY.id, fresh_key
                ) == "## Summary forced fresh" and not flights.in_flight(STORY.id):
                    break
                time.sleep(0.01)
        flight, leading = SingleFlight.followup_or_lead(flights, ordinary, key)
        if not leading:
            flights.joined.release()
        return flight, leading

    monkeypatch.setattr(flights, "followup_or_lead", gated)
    replies: list[Any] = []

    leader = _tap(handler, user, replies)
    assert first_started.wait(5.0)
    first = [_tap(handler, user, replies, force=True) for _ in range(2)]
    assert flights.joined.acquire(timeout=5.0)
    assert flights.joined.acquire(timeout=5.0)
    release_leader.set()
    for thread in [leader, *first]:
        thread.join(10.0)
        assert not thread.is_alive()

    assert len(made) == 2
    assert len(replies) == 3
    assert {status for status, _ in replies} == {200}
    assert replies[0][1]["tldr"] == "## Summary ordinary"
    assert replies[1][1]["tldr"] == "## Summary forced fresh"
    assert replies[2][1]["tldr"] == "## Summary forced fresh"
    assert not flights.in_flight(STORY.id)
    assert db.get_tldr_cache(STORY.id, fresh_key) == "## Summary forced fresh"
    assert db.get_tldr_cache_snapshot(STORY.id, fresh_key) == 42


def test_prefetch_skipped_while_forced_flight_runs(
    runtime: tuple[type[Handler], Database, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary work never starts while a forced flight exists (shared key)."""
    import pipeline

    handler, db, user = runtime
    monkeypatch.setattr(Handler, "_tldr_flights", handler._tldr_flights)
    monkeypatch.setattr(Handler, "db", db, raising=False)

    async def fake_fetch_story(
        client: object,
        sid: int,
        db_: Database,
        *,
        force: bool = False,
        strict: bool = False,
    ) -> Story | None:
        return db_.get_story(sid)

    monkeypatch.setattr(pipeline, "fetch_story", fake_fetch_story)
    llm = _Llm()
    monkeypatch.setattr(server, "generate_detailed_tldr", llm)
    replies: list[Any] = []
    generated: list[int] = []
    tap = _tap(handler, user, replies, force=True)
    assert llm.started.wait(5.0)
    prefetch = _prefetch(generated)
    prefetch.join(5.0)
    assert generated == [0] and llm.calls == 1
    llm.release.set()
    tap.join(5.0)
    assert replies[0][0] == 200


def test_a_failed_leader_releases_the_story_and_followers_retry(
    runtime: tuple[type[Handler], Database, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, db, user = runtime
    handler.config = replace(handler.config, tldr_uncached_per_user_limit=5)
    llm = _Llm(TldrResult(kind="llm_error", error_text="raise"))
    monkeypatch.setattr(server, "generate_detailed_tldr", llm)
    flights = handler._tldr_flights
    assert isinstance(flights, _CountedFlights)
    replies: list[Any] = []

    leader = _tap(handler, user, replies)
    assert llm.started.wait(5.0)
    follower = _tap(handler, user, replies)
    assert flights.joined.acquire(timeout=5.0)
    llm.release.set()
    leader.join(5.0)
    follower.join(5.0)

    assert sorted(status for status, _ in replies) == [500, 503]
    assert not flights.in_flight(STORY.id)
    llm.result = TldrResult(kind="ok", tldr="second try")
    retry = _client(handler, user).post("/api/tldr-detail", json={"story_id": STORY.id})
    assert retry.get_json()["tldr"] == "second try"
    assert llm.calls == 2


def _prefetch(out: list[int]) -> threading.Thread:
    # In the default window's Recommended view, which the prefetch walks.
    recent = replace(STORY, time=int(time.time()) - 3600)
    deck = WindowDeck({"1w": WindowViews(recommended=(RankedStory(recent, 1.0, ""),))})

    def run() -> None:
        out.append(
            asyncio.run(
                server._prefetch_tldrs_for_ranked(
                    deck, Handler.db, per_view=1, stagger_s=0
                )
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def test_prefetch_and_taps_never_generate_the_same_story_twice(
    runtime: tuple[type[Handler], Database, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler, db, user = runtime
    # The prefetch uses the base Handler's table and DB.
    monkeypatch.setattr(Handler, "_tldr_flights", handler._tldr_flights)
    monkeypatch.setattr(Handler, "db", db, raising=False)
    flights = handler._tldr_flights
    assert isinstance(flights, _CountedFlights)

    # A tap is generating: the prefetch leaves the story alone.
    llm = _Llm()
    monkeypatch.setattr(server, "generate_detailed_tldr", llm)
    replies: list[Any] = []
    generated: list[int] = []
    tap = _tap(handler, user, replies)
    assert llm.started.wait(5.0)
    prefetch = _prefetch(generated)
    prefetch.join(5.0)
    assert flights.joined.acquire(timeout=0)  # it found the tap's flight
    llm.release.set()
    tap.join(5.0)
    assert generated == [0] and llm.calls == 1

    # The prefetch is generating: a tap waits for it.
    db.upsert_story(replace(STORY, article_body=STORY.article_body + " More."))
    llm = _Llm(TldrResult(kind="ok", tldr="from the prefetch"))
    monkeypatch.setattr(server, "generate_detailed_tldr", llm)
    prefetch = _prefetch(generated)
    assert llm.started.wait(5.0)
    tap = _tap(handler, user, replies)
    assert flights.joined.acquire(timeout=5.0)
    llm.release.set()
    prefetch.join(5.0)
    tap.join(5.0)
    assert generated == [0, 1] and llm.calls == 1
    assert replies[-1] == (
        200,
        {
            "ok": True,
            "tldr": "from the prefetch",
            "cached": False,
            "comment_count_summarized": 0,
            "comments_summarized": 0,
        },
    )


def test_prefetch_join_of_provisional_half_carries_no_snapshot(
    runtime: tuple[type[Handler], Database, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A salvaged half (uncacheable, served retryably) joins with unknown
    provenance: no snapshot keys, so the client never treats it as coverage."""
    handler, db, user = runtime
    monkeypatch.setattr(Handler, "_tldr_flights", handler._tldr_flights)
    monkeypatch.setattr(Handler, "db", db, raising=False)
    flights = handler._tldr_flights
    assert isinstance(flights, _CountedFlights)

    db.upsert_story(replace(STORY, article_body=STORY.article_body + " Half."))
    llm = _Llm(TldrResult(kind="ok", tldr="half text", cacheable=False))
    monkeypatch.setattr(server, "generate_detailed_tldr", llm)
    replies: list[Any] = []
    generated: list[int] = []
    prefetch = _prefetch(generated)
    assert llm.started.wait(5.0)
    tap = _tap(handler, user, replies)
    assert flights.joined.acquire(timeout=5.0)
    llm.release.set()
    prefetch.join(5.0)
    tap.join(5.0)
    assert llm.calls == 1
    assert replies[-1] == (
        200,
        {"ok": True, "tldr": "half text", "cached": False, "retryable": True},
    )
