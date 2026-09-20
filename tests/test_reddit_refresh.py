from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pipeline

from database import Database, Story
from pipeline import Config, RssConfig
from pipeline.ranking import Embedder
from reddit_limiter import RedditCircuitSnapshot, RedditRateLimiter
from reddit_refresh import RedditRefreshWorker


def test_reddit_feed_snapshot_and_failure_survive_reopen(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = Database(str(path))
    story = Story(
        id=-1, title="r", url="https://reddit/r/x", score=0, time=1, text_content="r"
    )
    db.upsert_story(story)
    db.record_reddit_feed_success("feed", [story.id], 100.0)
    db.record_reddit_feed_failure("feed", "rate limited", 200.0)
    db.close()

    reopened = Database(str(path))
    try:
        state = reopened.get_reddit_feed_state("feed")
        assert state is not None
        assert state.last_success_at == 100.0
        assert state.failure_count == 1
        assert state.next_retry_at == 500.0
        assert state.item_count == 1
    finally:
        reopened.close()


def test_reddit_limiter_snapshot_restore_preserves_open_circuit() -> None:
    limiter = RedditRateLimiter()
    limiter.on_429()
    limiter.on_429()
    limiter.on_429()
    snapshot = limiter.snapshot(wall_time=time.time())

    restored = RedditRateLimiter()
    restored.restore(RedditCircuitSnapshot(snapshot.consecutive_429, snapshot.retry_at))

    assert restored.circuit_open is True


def test_reddit_worker_coalesces_one_pending_refresh(monkeypatch) -> None:
    db = Database(":memory:")
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = 0

    def fake_refresh(config, worker_db, embedder):
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=2.0)
        if calls == 2:
            completed.set()
        return SimpleNamespace(changed=False)

    monkeypatch.setattr(pipeline, "refresh_reddit_candidates", fake_refresh)
    worker = RedditRefreshWorker(
        Config(), db, cast(Embedder, SimpleNamespace()), lambda: None
    )
    try:
        worker.submit()
        assert started.wait(timeout=1.0)
        worker.submit()
        worker.submit()
        release.set()
        assert completed.wait(timeout=2.0)
        assert calls == 2
    finally:
        worker.shutdown()
        db.close()


def test_non_hn_candidates_use_only_configured_recent_feeds() -> None:
    db = Database(":memory:")
    now = 2_000_000
    configured = "https://lobste.rs/top/rss"
    config = replace(
        Config(days=30, recent_candidate_hn_limit=0, recent_candidate_rss_limit=10),
        rss=RssConfig(enabled=True, feeds=(configured,)),
    )
    db.upsert_story(
        Story(
            id=-1,
            title="configured",
            url=None,
            score=0,
            time=now,
            source="rss_lobste_rs",
            text_content="x",
            self_text="x",
        )
    )
    db.upsert_story(
        Story(
            id=-2,
            title="removed",
            url=None,
            score=0,
            time=now,
            source="rss_example_com",
            text_content="x",
            self_text="x",
        )
    )
    db.upsert_story(
        Story(
            id=-3,
            title="old",
            url=None,
            score=0,
            time=now - 31 * 86400,
            source="rss_lobste_rs",
            text_content="x",
            self_text="x",
        )
    )
    try:
        stories = pipeline.load_production_candidate_stories(
            db, config, user_id=None, exclude_feedback=False, now_ts=now
        )
        assert [story.id for story in stories] == [-1]
    finally:
        db.close()
