from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from database import Database, Story
from pipeline import RankTrace, story_embedding_text
from scripts import benchmark_rank_cold_cache as bench


def _cache_embedding(db: Database, story: Story) -> None:
    text_hash = hashlib.sha256(story_embedding_text(story).encode("utf-8")).hexdigest()
    db.upsert_embedding(
        story.id,
        bench.EMBEDDING_MODEL_VERSION,
        text_hash,
        np.zeros(384, dtype=np.float32),
    )


def test_benchmark_rank_cold_cache_outputs_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "bench.db"
    db = Database(str(db_path))
    now = int(time.time())
    user = db.create_user("bench-user")
    candidate = Story(
        id=1,
        title="Candidate",
        url=None,
        score=10,
        time=now,
        text_content="candidate text",
        source="hn",
        comment_count=1,
    )
    feedback = Story(
        id=2,
        title="Feedback",
        url=None,
        score=10,
        time=now,
        text_content="feedback text",
        source="hn",
        comment_count=1,
    )
    db.upsert_story(candidate)
    db.upsert_story(feedback)
    db.upsert_feedback(user.id, feedback.id, "up")
    _cache_embedding(db, candidate)
    _cache_embedding(db, feedback)
    db.close()

    def fake_fast_rerank_for_user(
        db_arg: Database,
        config_arg: Any,
        embedder_arg: object,
        user_id: int,
        trace: RankTrace | None = None,
    ) -> list[object]:
        _ = (config_arg, embedder_arg)
        assert user_id == user.id
        assert db_arg.read_only
        if trace is not None:
            trace.set_label("model_cache", "miss")
            trace.add_timing("candidate_sql", 1.0)
        return [object()]

    monkeypatch.setattr(bench, "Embedder", lambda _model_dir: object())
    monkeypatch.setattr(bench, "fast_rerank_for_user", fake_fast_rerank_for_user)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_rank_cold_cache.py",
            "--db",
            str(db_path),
            "--cold-runs",
            "1",
            "--warm-runs",
            "0",
            "--json-only",
        ],
    )

    bench.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["user_id"] == user.id
    assert payload["preflight"]["candidates"] == 1
    assert payload["preflight"]["feedback"] == 1
    assert payload["cold_runs"][0]["model_cache"] == "miss"
    assert payload["cold_runs"][0]["stories"] == 1
