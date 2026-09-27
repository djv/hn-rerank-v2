"""One summary generation per story, shared by taps and the warm prefetch."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

import server
from database import Database, Story, User
from pipeline import Config, RankedStory
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

    def join_or_lead(self, key: int) -> tuple[Flight[TldrReply], bool]:
        flight, leading = super().join_or_lead(key)
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
    others = [_tap(handler, user, replies, force=i == 0) for i in range(followers)]
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
    ranked = [RankedStory(STORY, 1.0, "", combo_keys="recent_hn recent_mixed")]

    def run() -> None:
        out.append(
            asyncio.run(
                server._prefetch_tldrs_for_ranked(
                    ranked, Handler.db, per_combo=1, date_top_n=0, stagger_s=0
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
        {"ok": True, "tldr": "from the prefetch", "cached": False},
    )
