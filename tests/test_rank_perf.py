from __future__ import annotations

import json
import time

from database import Database, RankPerfSample
from scripts.perf_report import StageStat, aggregate, parse_stage_timings


def _make_sample(
    *,
    recorded_at: float | None = None,
    user_id: int = 1,
    version: int = 1,
    rank_total_ms: float = 123.4,
    html_ms: float = 5.6,
    candidates: int = 42,
    feedback_total: int = 10,
    model_cache: str = "hit",
    stories: int = 8,
    fields: dict[str, int | float | str] | None = None,
) -> RankPerfSample:
    return RankPerfSample(
        recorded_at=recorded_at if recorded_at is not None else time.time(),
        user_id=user_id,
        version=version,
        rank_total_ms=rank_total_ms,
        html_ms=html_ms,
        candidates=candidates,
        feedback_total=feedback_total,
        model_cache=model_cache,
        stories=stories,
        fields=fields
        if fields is not None
        else {"rank_total_ms": 123.4, "candidates": 42, "model_cache": "hit"},
    )


def test_insert_rank_perf_round_trip() -> None:
    db = Database(":memory:")
    sample = _make_sample()

    db.insert_rank_perf(sample)

    rows = db.execute(
        "SELECT recorded_at, user_id, version, rank_total_ms, html_ms, "
        "candidates, feedback_total, model_cache, stories, fields_json "
        "FROM rank_perf"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == sample.recorded_at
    assert row[1] == sample.user_id
    assert row[2] == sample.version
    assert row[3] == sample.rank_total_ms
    assert row[4] == sample.html_ms
    assert row[5] == sample.candidates
    assert row[6] == sample.feedback_total
    assert row[7] == sample.model_cache
    assert row[8] == sample.stories
    assert json.loads(row[9]) == sample.fields


def test_parse_stage_timings_includes_html_ms_synthesized() -> None:
    fields_json = json.dumps(
        {"rank_total_ms": 100.0, "svm_fit_ms": 40.0, "candidates": 8, "model_cache": "hit"}
    )

    stages = parse_stage_timings(fields_json, html_ms=12.5)

    assert stages == {"rank_total_ms": 100.0, "svm_fit_ms": 40.0, "html_ms": 12.5}


def test_aggregate_computes_p50_p95_max_per_stage_and_cache() -> None:
    rows = [
        ("hit", 10.0, json.dumps({"rank_total_ms": ms}))
        for ms in [100.0, 200.0, 300.0, 400.0, 500.0]
    ] + [
        ("skipped", 1.0, json.dumps({"rank_total_ms": 50.0})),
    ]

    by_cache = aggregate(rows)

    assert set(by_cache.keys()) == {"hit", "skipped"}

    hit_stats = {s.stage: s for s in by_cache["hit"]}
    assert set(hit_stats.keys()) == {"rank_total_ms", "html_ms"}
    rank_stat = hit_stats["rank_total_ms"]
    assert rank_stat.n == 5
    assert rank_stat.max == 500.0
    assert rank_stat.p50 == 300.0

    skipped_stats = {s.stage: s for s in by_cache["skipped"]}
    assert skipped_stats["rank_total_ms"] == StageStat(
        "rank_total_ms", 1, 50.0, 50.0, 50.0
    )
