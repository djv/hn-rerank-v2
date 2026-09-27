from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from database import Database, Story


def _story(sid: int, points: int, comments: int) -> Story:
    return Story(
        sid,
        f"Story {sid}",
        f"https://example.test/{sid}",
        points,
        0,
        "text",
        comment_count=comments,
    )


def test_engagement_features_let_the_svm_learn_a_points_preference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With engagement_features_enabled, a user who upvotes low-points stories
    and downvotes high-points ones (embeddings carry no signal) gets a
    low-points candidate ranked above a high-points one."""
    from pipeline import ranking
    from pipeline.config import Config

    db = Database(str(tmp_path / "rank.db"))
    try:
        user = db.create_user("test-engagement")
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(200, 384)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        for sid in range(1, 91):
            action = ("up", "neutral", "down")[sid % 3]
            points = {"up": 5, "neutral": 60, "down": 900}[action]
            item = _story(sid, points + sid % 7, points // 3)
            db.upsert_story(item)
            db.upsert_feedback(user.id, sid, action)

        def embeddings(
            stories: list[Story], embedder: ranking.Embedder, database: Database
        ) -> np.ndarray:
            return matrix[[s.id for s in stories]]

        monkeypatch.setattr(ranking, "get_or_compute_embeddings", embeddings)
        embedder = MagicMock(spec=ranking.Embedder)
        embedder.model_version = "test"
        config = Config()
        config = replace(
            config, model=replace(config.model, engagement_features_enabled=True)
        )
        low = [_story(sid, 6, 2) for sid in range(150, 170)]
        high = [_story(sid, 900, 300) for sid in range(170, 190)]
        candidates = low + high
        ranked = ranking._score_and_rank(
            candidates,
            matrix[[s.id for s in candidates]],
            db,
            config,
            embedder,
            user.id,
        )
        position = {r.story.id: i for i, r in enumerate(ranked)}
        low_first = sum(position[a.id] < position[b.id] for a in low for b in high) / (
            len(low) * len(high)
        )
        assert low_first > 0.7  # ~0.5 with the flag off (embeddings are noise)
        assert all(np.isfinite(r.score) for r in ranked)
    finally:
        db.close()
