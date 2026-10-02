from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from database import Database, Story
from pipeline import Config, Embedder, ModelConfig
from pipeline.ranking import RankScoreContext, RankTrace, _score_and_rank
from pipeline.side_embeddings import (
    SideEmbedder,
    SideVectorCache,
    model_space,
    side_by_side,
    side_input,
    side_text_hash,
)

SIDE_DIM = 8


@pytest.fixture(autouse=True)
def _fresh_side_cache() -> None:
    """Ranking reads through the module cache; tests reuse story IDs."""
    from pipeline import side_embeddings

    side_embeddings._CACHE.clear()


def _story(sid: int, text: str = "plain words about nothing") -> Story:
    return Story(
        id=sid,
        title=f"story {sid}",
        url=f"https://example.com/{sid}",
        score=10,
        time=1_600_000_000 + sid,
        text_content=f"{text} {sid}",
    )


def _unit(index: int, rng: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    vec = np.zeros(SIDE_DIM, dtype=np.float32)
    vec[index] = 1.0
    vec += noise * rng.standard_normal(SIDE_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _config(enabled: bool, coverage: float = 0.98) -> Config:
    return Config(
        side_embedding_dim=SIDE_DIM,
        model=ModelConfig(
            min_up_for_svm=2,
            min_down_for_svm=2,
            side_embedding_enabled=enabled,
            side_embedding_min_coverage=coverage,
        ),
    )


def test_side_vectors_round_trip_and_skip_stale_text_or_wrong_dim() -> None:
    db = Database(":memory:")
    try:
        stories = [_story(1), _story(2), _story(3)]
        for story in stories:
            db.upsert_story(story)
        rng = np.random.default_rng(0)
        vecs = [_unit(i, rng) for i in range(3)]
        db.upsert_side_embeddings(
            "side-v1",
            [
                (1, side_text_hash(stories[0]), vecs[0]),
                (2, "stale-hash", vecs[1]),
                (3, side_text_hash(stories[2]), np.ones(4, dtype=np.float32)),
            ],
        )
        hashes = {s.id: side_text_hash(s) for s in stories}

        found = db.get_side_embeddings_batch([1, 2, 3], "side-v1", hashes, SIDE_DIM)
        other_model = db.get_side_embeddings_batch([1], "side-v2", hashes, SIDE_DIM)

        assert list(found) == [1]
        assert np.array_equal(found[1], vecs[0])
        assert other_model == {}
    finally:
        db.close()


def test_read_only_db_without_side_table_reads_as_empty(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.db"
    sqlite3.connect(path).close()
    db = Database(str(path), read_only=True)
    try:
        assert db.get_side_embeddings_batch([1], "v", {1: "h"}, SIDE_DIM) == {}
    finally:
        db.close()


def test_side_by_side_keeps_unit_rows_unit_and_checks_rows() -> None:
    rng = np.random.default_rng(1)
    stored = np.stack([_unit(0, rng), _unit(1, rng)])
    side = np.stack([_unit(2, rng), _unit(3, rng)])

    joined = side_by_side(stored, side)

    assert joined.shape == (2, 2 * SIDE_DIM)
    assert np.allclose(np.linalg.norm(joined, axis=1), 1.0, atol=1e-6)
    # Cosine of joined rows is the mean of the two models' cosines.
    expected = (stored[0] @ stored[1] + side[0] @ side[1]) / 2
    assert joined[0] @ joined[1] == pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError):
        side_by_side(stored, side[:1])


def test_model_space_zero_fills_missing_and_gates_on_coverage() -> None:
    db = Database(":memory:")
    try:
        candidates = [_story(i) for i in range(1, 11)]
        feedback = [_story(i) for i in range(11, 21)]
        for story in candidates + feedback:
            db.upsert_story(story)
        rng = np.random.default_rng(2)
        side = {s.id: _unit(s.id % SIDE_DIM, rng) for s in candidates + feedback}
        db.upsert_side_embeddings(
            "embeddinggemma-300m|sentence|128|classification",
            [
                (sid, side_text_hash(s), side[sid])
                for s in candidates + feedback
                for sid in [s.id]
                if sid != 3
            ],
        )
        stored_c = np.eye(10, 4, dtype=np.float32)
        stored_f = np.eye(10, 4, dtype=np.float32)

        space = model_space(
            candidates,
            stored_c,
            feedback,
            stored_f,
            db,
            _config(True, 0.9),
            cache=SideVectorCache(),
        )
        gated = model_space(
            candidates,
            stored_c,
            feedback,
            stored_f,
            db,
            _config(True, 0.99),
            cache=SideVectorCache(),
        )

        assert space is not None and space.coverage == pytest.approx(19 / 20)
        assert space.candidates.shape == (10, 4 + SIDE_DIM)
        assert np.allclose(space.candidates[2, 4:], 0.0)  # story 3: no side vector
        assert np.allclose(space.candidates[0, 4:], side[1] / np.sqrt(2))
        assert np.allclose(space.feedback[0, :4], stored_f[0] / np.sqrt(2))
        assert gated is None
    finally:
        db.close()


def test_side_vector_cache_reads_each_story_text_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = Database(":memory:")
    try:
        stories = [_story(1), _story(2)]
        for story in stories:
            db.upsert_story(story)
        config = _config(True)
        db.upsert_side_embeddings(
            config.side_embedding_model_version,
            [(1, side_text_hash(stories[0]), np.ones(SIDE_DIM, dtype=np.float32))],
        )
        asked: list[list[int]] = []
        real = db.get_side_embeddings_batch

        def counting(ids: list[int], *args: Any, **kwargs: Any) -> Any:
            asked.append(sorted(ids))
            return real(ids, *args, **kwargs)

        monkeypatch.setattr(db, "get_side_embeddings_batch", counting)
        cache = SideVectorCache()

        cache.get(db, config, stories)
        cache.get(db, config, stories)
        edited = replace(stories[0], text_content="new text")
        cache.get(db, config, [edited])

        # Story 2 has no vector, so it is asked again; story 1 only when
        # its text changes.
        assert asked == [[1, 2], [2], [1]]
    finally:
        db.close()


def test_side_embedder_restores_input_order_and_normalizes() -> None:
    class Tokenizer:
        def __call__(self, texts: list[str], **_: Any) -> dict[str, np.ndarray]:
            lengths = np.array([[len(t)] for t in texts], dtype=np.int64)
            return {"input_ids": lengths, "attention_mask": np.ones_like(lengths)}

    class Output:
        name = "sentence_embedding"

    class Session:
        def get_outputs(self) -> list[Output]:
            return [Output()]

        def run(self, _: None, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            n = feed["input_ids"].astype(np.float32)
            return [np.hstack([n, 2 * n])]

    embedder = object.__new__(SideEmbedder)
    embedder.max_tokens = 128
    embedder.batch_size = 2
    embedder.tokenizer = Tokenizer()
    embedder.session = Session()  # type: ignore[assignment]  # stub session
    embedder.input_names = {"input_ids"}

    vectors = embedder.encode(["aaa", "a", "aaaaa"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors, np.tile([1, 2], (3, 1)) / np.sqrt(5))
    assert side_input(_story(1)).startswith("task: classification | query: ")


class _NoiseEmbedder(Embedder):
    """Stored vectors carry no signal; the side vectors do."""

    def __init__(self) -> None:
        self.batch_size = 32
        self.max_tokens = 4096

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(len(texts))
        vecs = rng.standard_normal((len(texts), 384)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _seed_votes(db: Database, *, skip_side: int = 0) -> tuple[int, list[Story]]:
    """40 votes: up stories' side vectors near e0, down stories' near e1;
    identical text, so TF-IDF and the stored noise carry no signal."""
    user = db.create_user("side")
    rng = np.random.default_rng(3)
    config = _config(True)
    rows = []
    for i in range(40):
        up = i % 2 == 0
        story = _story(1000 + i)
        db.upsert_story(story)
        action: Literal["up", "down"] = "up" if up else "down"
        db.upsert_feedback(user.id, story.id, action)
        if i >= skip_side:
            rows.append((story.id, side_text_hash(story), _unit(0 if up else 1, rng)))
    candidates = [_story(1), _story(2)]
    for story in candidates:
        db.upsert_story(story)
    rows.append((1, side_text_hash(candidates[0]), _unit(1, rng)))
    rows.append((2, side_text_hash(candidates[1]), _unit(0, rng)))
    db.upsert_side_embeddings(config.side_embedding_model_version, rows)
    return user.id, candidates


def _rank(
    db: Database, user_id: int, candidates: list[Story], config: Config
) -> tuple[dict[int, float], RankTrace, RankScoreContext]:
    """The SVM's P(up) per candidate (the final score still blends in
    gravity at 40 votes), the trace and the score context."""
    trace = RankTrace()
    context = RankScoreContext()
    stored = np.random.default_rng(99).standard_normal((2, 384)).astype(np.float32)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    ranked = _score_and_rank(
        candidates,
        stored,
        db,
        config,
        _NoiseEmbedder(),
        user_id=user_id,
        trace=trace,
        score_context=context,
    )
    return {r.story.id: float(r.prob_up or 0.0) for r in ranked}, trace, context


def test_side_vectors_steer_the_svm_but_not_the_score_context() -> None:
    db = Database(":memory:")
    try:
        user_id, candidates = _seed_votes(db)
        off, _, off_context = _rank(db, user_id, candidates, _config(False))
        on, trace, on_context = _rank(db, user_id, candidates, _config(True))

        assert trace.labels["side_embeddings"] == "on"
        # Candidate 2's side vector sits with the upvotes, 1's with the
        # downvotes; the stored noise alone cannot tell them apart.
        assert on[2] - on[1] > 0.3
        assert abs(off[2] - off[1]) < 0.1
        assert off_context.cand_closest_up is not None
        assert on_context.cand_closest_up is not None
        assert np.allclose(on_context.cand_closest_up, off_context.cand_closest_up)
        assert on_context.fb_up_embeddings is not None
        assert on_context.fb_up_embeddings.shape[1] == 384
    finally:
        db.close()


def test_low_side_coverage_ranks_on_stored_vectors_alone() -> None:
    db = Database(":memory:")
    try:
        user_id, candidates = _seed_votes(db, skip_side=10)
        off, _, _ = _rank(db, user_id, candidates, _config(False))
        gated, trace, _ = _rank(db, user_id, candidates, _config(True))

        assert trace.labels["side_embeddings"] == "off"
        assert gated == pytest.approx(off)
    finally:
        db.close()


def test_config_reads_side_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[hn_rewrite]\n"
        'side_embedding_model_dir = "/models/gemma"\n'
        "side_embedding_max_tokens = 256\n"
        "[hn_rewrite.model]\n"
        "side_embedding_enabled = true\n"
        "side_embedding_min_coverage = 0.9\n"
    )

    config = Config.load(str(path))

    assert config.side_embedding_model_dir == "/models/gemma"
    assert config.side_embedding_max_tokens == 256
    assert config.model.side_embedding_enabled
    assert config.model.side_embedding_min_coverage == 0.9
    assert not Config().model.side_embedding_enabled
    with pytest.raises(ValueError, match="side_embedding_dim"):
        Config(side_embedding_dim=0)


def test_encoder_queue_puts_votes_first_then_newest_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import embed_side_vectors

    db = Database(":memory:")
    try:
        user = db.create_user("queue")
        voted, old, new, done = _story(10), _story(20), _story(30), _story(40)
        for story in (voted, old, new, done):
            db.upsert_story(story)
        db.upsert_feedback(user.id, voted.id, "up")
        config = _config(True)
        db.upsert_side_embeddings(
            config.side_embedding_model_version,
            [(done.id, side_text_hash(done), np.ones(SIDE_DIM, dtype=np.float32))],
        )
        monkeypatch.setattr(
            embed_side_vectors,
            "load_production_candidate_stories",
            lambda *_, **__: [old, done, new, voted],
        )

        todo, considered = embed_side_vectors.stories_to_encode(db, config)

        assert [s.id for s in todo] == [10, 30, 20]
        assert considered == 4
    finally:
        db.close()
