"""Tests for the process-wide candidate pool cache.

See pipeline/candidate_cache.py docstring and WORKLOG 2026-07-28 for why
this exists: candidate SQL load + embedding materialization dominated
rank time (median 2-8s across ~8,900 rows) even though the SVM fit itself
takes single-digit milliseconds. This cache makes that load-once,
mask-per-user.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

import pipeline.candidate_cache as candidate_cache
from database import Database, Story
from pipeline import Config, Embedder
from pipeline.candidate_cache import get_candidate_pool, invalidate_candidate_pool


class CountingEmbedder(Embedder):
    """Records how many times `encode` is called; no real model load."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts: list[str], batch_size: int | None = None) -> Any:
        self.encode_calls += 1
        return np.zeros((len(texts), 384), dtype=np.float32)


@pytest.fixture
def db():
    db_instance = Database(":memory:")
    yield db_instance
    db_instance.close()


@pytest.fixture(autouse=True)
def _reset_pool_cache():
    """Each test starts from a clean (uncached) pool state."""
    invalidate_candidate_pool()
    yield
    invalidate_candidate_pool()


def _story(sid: int, score: int = 10, time_ts: int | None = None) -> Story:
    return Story(
        id=sid,
        title=f"Story {sid}",
        url=f"https://example.com/{sid}",
        score=score,
        time=time_ts if time_ts is not None else int(time.time()) - 3600,
        text_content="body text",
        source="hn",
        comment_count=1,
    )


def test_get_candidate_pool_builds_once_and_caches(db: Database) -> None:
    for sid in range(1, 6):
        db.upsert_story(_story(sid))
    embedder = CountingEmbedder()
    config = Config()

    pool1 = get_candidate_pool(db, config, embedder)
    calls_after_first = embedder.encode_calls
    pool2 = get_candidate_pool(db, config, embedder)

    assert calls_after_first > 0
    assert embedder.encode_calls == calls_after_first  # no second encode
    assert pool1 is pool2
    assert len(pool1.stories) == 5
    assert pool1.embeddings.shape == (5, 384)


def test_invalidate_candidate_pool_forces_rebuild(db: Database) -> None:
    for sid in range(1, 4):
        db.upsert_story(_story(sid))
    embedder = CountingEmbedder()
    config = Config()

    pool1 = get_candidate_pool(db, config, embedder)
    invalidate_candidate_pool()
    pool2 = get_candidate_pool(db, config, embedder)

    assert pool1 is not pool2
    assert pool2.generation > pool1.generation


def test_without_feedback_preserves_alignment_and_excludes_voted(
    db: Database,
) -> None:
    for sid in range(1, 6):
        db.upsert_story(_story(sid, score=sid * 10))
    embedder = CountingEmbedder()
    config = Config()
    pool = get_candidate_pool(db, config, embedder)
    # Distinguish embeddings by row so alignment is verifiable.
    fake = np.arange(len(pool.stories) * 384, dtype=np.float32).reshape(-1, 384)
    pool = candidate_cache.CandidatePool(
        stories=pool.stories,
        embeddings=fake,
        index_by_id=pool.index_by_id,
        generation=pool.generation,
        built_at=pool.built_at,
    )

    voted = frozenset({2, 4})
    kept_stories, kept_embeddings = pool.without_feedback(voted)

    # Excludes exactly the voted IDs, preserves the pool's relative order
    # for everything else.
    assert {s.id for s in kept_stories} == {1, 3, 5}
    original_order = [s.id for s in pool.stories if s.id not in voted]
    assert [s.id for s in kept_stories] == original_order
    for story, emb in zip(kept_stories, kept_embeddings):
        expected_row = pool.index_by_id[story.id]
        assert np.array_equal(emb, fake[expected_row])


def test_without_feedback_empty_voted_returns_full_pool(db: Database) -> None:
    for sid in range(1, 4):
        db.upsert_story(_story(sid))
    embedder = CountingEmbedder()
    pool = get_candidate_pool(db, Config(), embedder)

    kept_stories, kept_embeddings = pool.without_feedback(frozenset())

    assert [s.id for s in kept_stories] == [s.id for s in pool.stories]
    assert np.array_equal(kept_embeddings, pool.embeddings)


def test_concurrent_get_candidate_pool_builds_once(db: Database) -> None:
    for sid in range(1, 4):
        db.upsert_story(_story(sid))
    embedder = CountingEmbedder()
    config = Config()
    pools: list[candidate_cache.CandidatePool] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait(timeout=5)
        pools.append(get_candidate_pool(db, config, embedder))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(pools) == 8
    assert all(p is pools[0] for p in pools)
    assert embedder.encode_calls == 1
