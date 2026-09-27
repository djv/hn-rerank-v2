from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path
from unittest.mock import MagicMock

from database import Database, Story
from pipeline.publication import identity, publication_features


def story(sid: int, url: str | None = None) -> Story:
    return Story(sid, "Story", url or f"https://publisher.test/{sid}", 0, 0, "text")


def test_chronological_features_exclude_self_future_and_ties() -> None:
    feedback = [story(1), story(2), story(3)]
    train, candidate = publication_features(feedback, [2, 2, 0], [1, 1, 2], [story(4)])
    np.testing.assert_array_equal(train[:2], np.zeros((2, 4)))
    assert train[2, 2] > 0  # its own dislike is not visible
    changed, _ = publication_features(feedback, [2, 2, 2], [1, 1, 2], [])
    np.testing.assert_array_equal(train, changed)
    assert candidate[0, 3] == pytest.approx(3 / 13)


def test_duplicates_count_once_and_entire_own_group_is_excluded() -> None:
    a, duplicate = story(1), story(2, "http://www.publisher.test/1#comments")
    train, cand = publication_features([a, duplicate], [2, 0], [1, 2], [a, story(3)])
    np.testing.assert_array_equal(train, np.zeros((2, 4)))
    np.testing.assert_array_equal(cand[0], np.zeros(4))
    assert cand[1, 0] > 0  # latest vote replaces, not adds to, old evidence
    assert cand[1, 3] == pytest.approx(1 / 11)


def test_missing_and_unseen_publications_are_neutral() -> None:
    _, cand = publication_features(
        [story(1)],
        [2],
        [1],
        [story(2, "https://new.test/"), story(3, "https://reddit.com/r/a")],
    )
    np.testing.assert_allclose(cand, np.zeros((2, 4)), atol=1e-7)
    assert identity(story(4, "https://alice.substack.com/post")) != identity(
        story(5, "https://bob.substack.com/post")
    )


def test_consistent_likes_increase_confidence_without_touching_other_publishers() -> (
    None
):
    _, one = publication_features([story(1)], [2], [1], [story(99)])
    _, many = publication_features(
        [story(i) for i in range(13)], [2] * 13, list(range(13)), [story(99)]
    )
    assert many[0, 3] > one[0, 3]
    assert many[0, 2] > 0
    assert abs(float(many[0, :3].sum())) < 1e-6


def test_affinity_integrates_with_svm_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from pipeline import ranking
    from pipeline.config import Config

    db = Database(str(tmp_path / "rank.db"))
    try:
        user = db.create_user("test-publication")
        feedback = [story(i) for i in range(1, 61)]
        for i, item in enumerate(feedback):
            db.upsert_story(item)
            db.upsert_feedback(user.id, item.id, ("up", "neutral", "down")[i % 3])
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(100, 384)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

        def embeddings(
            stories: list[Story], embedder: ranking.Embedder, database: Database
        ) -> np.ndarray:
            return matrix[[s.id for s in stories]]

        monkeypatch.setattr(ranking, "get_or_compute_embeddings", embeddings)
        embedder = MagicMock(spec=ranking.Embedder)
        embedder.model_version = "test"
        config = Config()
        config = replace(
            config, model=replace(config.model, publication_affinity_enabled=True)
        )
        candidates = [story(70), story(71, "https://other.test/71")]
        first = ranking._score_and_rank(
            candidates, matrix[[70, 71]], db, config, embedder, user.id
        )
        second = ranking._score_and_rank(
            candidates, matrix[[70, 71]], db, config, embedder, user.id
        )
        assert len(first) == 2
        assert all(r.prob_up is not None for r in first + second)
        assert [(r.story.id, r.score) for r in first] == [
            (r.story.id, r.score) for r in second
        ]
        assert all(np.isfinite(r.score) for r in first)
    finally:
        db.close()


@pytest.mark.parametrize("strength", [0, -1, float("nan")])
def test_invalid_strength(strength: float) -> None:
    with pytest.raises(ValueError):
        publication_features([], [], [], [], strength=strength)
