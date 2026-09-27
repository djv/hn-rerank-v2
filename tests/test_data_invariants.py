from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from cachetools import LRUCache
from hypothesis import example, given, settings, strategies as st
from numpy.typing import NDArray

from database import Action, Database, Story
from pipeline import Config, Embedder, ModelConfig, RankTrace
import pipeline.ranking as ranking


FeedbackOperation = tuple[int, int, Literal["up", "down", "neutral", "clear"]]


@given(
    operations=st.lists(
        st.tuples(
            st.integers(0, 2),
            st.integers(0, 3),
            st.sampled_from(["up", "down", "neutral", "clear"]),
        ),
        min_size=1,
        max_size=25,
    )
)
@example(
    operations=[
        (0, 0, "up"),
        (1, 0, "down"),
        (0, 0, "clear"),
        (0, 0, "clear"),
        (0, 0, "neutral"),
    ]
)
@settings(deadline=None)
def test_feedback_sequences_preserve_user_isolation(
    operations: list[FeedbackOperation],
) -> None:
    db = Database(":memory:")
    try:
        users = [db.create_user(f"property-{index}") for index in range(3)]
        ids = [1, -1, 2, -2]
        for sid in ids:
            db.upsert_story(
                Story(
                    id=sid,
                    title=f"Story {sid}",
                    url=None,
                    score=1,
                    time=1,
                    text_content=f"Body {sid}",
                )
            )
        expected: dict[tuple[int, int], Action] = {}
        label = {"down": 0, "neutral": 1, "up": 2}
        for user_index, story_index, action in operations:
            uid, sid = users[user_index].id, ids[story_index]
            if action == "clear":
                existed = (uid, sid) in expected
                assert db.delete_feedback(uid, sid) == existed
                expected.pop((uid, sid), None)
            else:
                db.upsert_feedback(uid, sid, action)
                expected[uid, sid] = action
            for user in users:
                wanted = {
                    sid: vote
                    for (owner, sid), vote in expected.items()
                    if owner == user.id
                }
                records = db.get_all_feedback(user.id)
                assert len(records) == len(wanted)
                assert {row.story_id: row.action for row in records} == wanted
                assert db.count_feedback_by_action(user.id) == {
                    vote: sum(action == vote for action in wanted.values())
                    for vote in label
                }
                stories, labels, times = db.get_feedback_for_training(user.id)
                assert len(stories) == len(labels) == len(times) == len(wanted)
                assert {s.id: value for s, value in zip(stories, labels)} == {
                    sid: label[vote] for sid, vote in wanted.items()
                }
                assert {s.id for s in db.get_feedback_stories(user.id, ("up",))} == {
                    sid for sid, vote in wanted.items() if vote == "up"
                }
            assert {s.id for s in db.get_stories(ids)} == set(ids)
    finally:
        db.close()


class LookupEmbedder(Embedder):
    """Deterministic model boundary; no tokenizer or ONNX session."""

    def __init__(self, vectors: dict[str, NDArray[np.float32]]) -> None:
        self.vectors = vectors
        self.model_version = "property-v1"
        self.calls: list[list[str]] = []

    def encode(
        self, texts: list[str], batch_size: int | None = None
    ) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        return np.stack([self.vectors[text] for text in texts])


@given(
    states=st.lists(
        st.sampled_from(["hit", "missing", "text", "model", "dimension"]),
        min_size=2,
        max_size=8,
    ),
    data=st.data(),
)
@settings(deadline=None)
def test_embedding_cache_rejects_stale_rows_and_preserves_batch_alignment(
    states: list[str],
    data: st.DataObject,
) -> None:
    db = Database(":memory:")
    try:
        basis = np.eye(384, dtype=np.float32)
        stories = [
            Story(
                id=i + 1,
                title=f"Story {i}",
                url=None,
                score=1,
                time=1,
                text_content=f"text-{i}",
            )
            for i in range(len(states))
        ]
        embedder = LookupEmbedder(
            {s.text_content: basis[i] for i, s in enumerate(stories)}
        )
        for story, state in zip(stories, states):
            db.upsert_story(story)
            if state != "missing":
                expected = embedder.vectors[story.text_content]
                db.upsert_embedding(
                    story.id,
                    "stale-model" if state == "model" else embedder.model_version,
                    "stale-hash"
                    if state == "text"
                    else hashlib.sha256(story.text_content.encode()).hexdigest(),
                    expected
                    if state == "hit"
                    else basis[-1, :383]
                    if state == "dimension"
                    else basis[-1],
                )
        order = data.draw(st.permutations(range(len(stories))), label="request order")
        requested = [stories[index] for index in order]
        actual = ranking.get_or_compute_embeddings(requested, embedder, db)
        np.testing.assert_array_equal(
            actual, np.stack([embedder.vectors[s.text_content] for s in requested])
        )
        misses = [
            stories[index].text_content for index in order if states[index] != "hit"
        ]
        assert embedder.calls == ([misses] if misses else [])
        embedder.calls.clear()
        # Metadata-only changes and a different request order must reuse vectors.
        reordered = [replace(s, score=999) for s in reversed(requested)]
        again = ranking.get_or_compute_embeddings(reordered, embedder, db)
        np.testing.assert_array_equal(again, actual[::-1])
        assert embedder.calls == []
        # Changing one text recomputes exactly that row, not the whole batch.
        updated = replace(reordered[0], text_content="changed text")
        embedder.vectors[updated.text_content] = basis[-2]
        reordered[0] = updated
        refreshed = ranking.get_or_compute_embeddings(reordered, embedder, db)
        np.testing.assert_array_equal(
            refreshed, np.stack([embedder.vectors[s.text_content] for s in reordered])
        )
        assert embedder.calls == [[updated.text_content]]
        # A new model invalidates every row even with unchanged story text.
        embedder.calls.clear()
        embedder.model_version = "property-v2"
        embedder.vectors = {text: -vector for text, vector in embedder.vectors.items()}
        migrated = ranking.get_or_compute_embeddings(reordered, embedder, db)
        np.testing.assert_array_equal(migrated, -refreshed)
        assert embedder.calls == [[s.text_content for s in reordered]]
    finally:
        db.close()


@pytest.mark.parametrize("change", ["text", "source", "model"])
@pytest.mark.parametrize("seed", [0, 42, 2**31 - 1])
def test_model_cache_agrees_with_fresh_fit_after_training_input_change(
    change: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(9, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    embedder = LookupEmbedder({f"text-{i}": vector for i, vector in enumerate(vectors)})
    db = Database(":memory:")
    try:
        user = db.create_user("cache-property")
        stories = [
            Story(
                id=i + 1,
                title=f"Story {i}",
                url=None,
                score=10,
                time=1,
                text_content=f"text-{i}",
            )
            for i in range(8)
        ]
        for story in stories:
            db.upsert_story(story)
        for story, action in zip(stories[:4], ("up", "up", "down", "down")):
            db.upsert_feedback(user.id, story.id, action)
        config = Config(model=ModelConfig(min_up_for_svm=2, min_down_for_svm=2))

        def rank() -> list[tuple[int, float, float, float, float]]:
            trace = RankTrace()
            results = ranking._score_and_rank(
                stories[4:],
                vectors[4:8],
                db,
                config,
                embedder,
                user_id=user.id,
                trace=trace,
            )
            assert trace.labels.get("model_cache") in ("hit", "miss")
            observed: list[tuple[int, float, float, float, float]] = []
            for row in results:
                assert row.prob_down is not None
                assert row.prob_neutral is not None
                assert row.prob_up is not None
                observed.append(
                    (
                        row.story.id,
                        row.score,
                        row.prob_down,
                        row.prob_neutral,
                        row.prob_up,
                    )
                )
            return sorted(observed)

        with (
            patch.object(ranking, "_MODEL_CACHE", LRUCache(maxsize=20)),
            patch("time.time", return_value=2_000_000_000.0),
        ):
            initial = rank()
            np.testing.assert_allclose(rank(), initial, rtol=1e-6, atol=1e-6)
            # These change fitted model inputs without changing the user's votes.
            if change == "text":
                db.upsert_story(replace(stories[0], text_content="text-8"))
            elif change == "source":
                db.upsert_story(replace(stories[0], source="ch_seed"))
            else:
                config = replace(config, model=replace(config.model, svm_gamma=0.4))
            cached = rank()
            with patch.object(ranking, "_get_cached_model", return_value=None):
                fresh = rank()
            np.testing.assert_allclose(cached, fresh, rtol=1e-6, atol=1e-6)
    finally:
        db.close()
