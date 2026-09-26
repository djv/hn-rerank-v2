from __future__ import annotations

import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, strategies as st

from database import Story
from pipeline import Config
from scripts import calibrate_rankings
from scripts.calibrate_rankings import (
    Scored,
    _discordant,
    cached_scores,
    disagreement_pool,
    next_batch,
    pair_agreement,
    parse_order,
    report,
)


def test_parse_order_accepts_only_permutations() -> None:
    assert parse_order("31524", 5) == [2, 0, 4, 1, 3]
    assert parse_order("3 1 5 2 4", 5) == [2, 0, 4, 1, 3]
    for bad in ("3152", "31554", "3152x", "315246", ""):
        assert parse_order(bad, 5) is None


@given(st.permutations(range(5)))
def test_pair_agreement_is_full_for_matching_scores_and_zero_for_reversed(
    order: list[int],
) -> None:
    # Scores that rank exactly like the user's order (best first = highest).
    scores = [0.0] * 5
    for position, item in enumerate(order):
        scores[item] = float(5 - position)
    assert pair_agreement(order, scores) == (10, 10)
    assert pair_agreement(order, [-s for s in scores]) == (0, 10)


def _scored(sid: int, production: float, challenger: float) -> Scored:
    return Scored(Story(sid, f"S{sid}", None, 1, 0, ""), production, challenger)


def test_disagreement_pool_keeps_top_stories_most_disputed_first() -> None:
    scored = [_scored(i, i / 999, i / 999) for i in range(1000)]
    scored[995] = _scored(995, 0.99, 0.10)  # production top, challenger low
    scored[3] = _scored(3, 0.05, 0.97)  # challenger top, production low
    scored[500] = _scored(500, 0.5, 0.2)  # disputed but in neither top 60
    pool = disagreement_pool(scored, seen={3})
    ids = [s.story.id for s in pool]
    assert ids[0] == 995
    assert 3 not in ids and 500 not in ids
    assert all(max(s.production, s.challenger) >= 1 - 60 / 1000 for s in pool)


@given(st.integers(0, 2**32 - 1))
def test_next_batch_picks_pairs_the_rankers_order_differently(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Ten stories the rankers order oppositely, buried among forty they agree on.
    agreed = [_scored(i, 0.5 + i / 100, 0.5 + i / 100) for i in range(40)]
    reversed_ = [_scored(100 + i, 0.9 + i / 100, 0.99 - i / 100) for i in range(10)]
    shuffled = [(agreed + reversed_)[int(i)] for i in rng.permutation(50)]
    pool = sorted(shuffled, key=lambda s: -abs(s.production - s.challenger))
    batch = next_batch(pool, rng)
    assert len({s.story.id for s in batch}) == 5
    assert all(_discordant(a, b) for a, b in combinations(batch, 2))


def test_next_batch_spreads_ties_across_topics() -> None:
    def item(sid: int, axis: int) -> Scored:
        vector = np.zeros(8)
        vector[axis] = 1.0
        return replace(_scored(sid, 0.9, 0.9), embedding=vector)

    # No story is disputed, so topic spread decides: 5 topics, 4 copies each.
    pool = [item(topic * 10 + copy, topic) for topic in range(5) for copy in range(4)]
    batch = next_batch(pool, np.random.default_rng(0))
    assert len({s.story.id // 10 for s in batch}) == 5


def test_cached_scores_reuse_until_the_snapshot_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "snapshot.db"
    db.write_bytes(b"x")
    calls: list[int] = []

    def fake_score(_db: Path, _config: Config, _user: int) -> list[Scored]:
        calls.append(1)
        story = Story(7, "T", "https://a.example/x", 3, 0, "long " * 200)
        return [Scored(story, 0.9, 0.1, np.ones(4) / 2)]

    monkeypatch.setattr(calibrate_rankings, "score_candidates", fake_score)
    cache = tmp_path / "scores.json"
    first = cached_scores(db, "missing.toml", Config(), 1, cache)
    second = cached_scores(db, "missing.toml", Config(), 1, cache)
    assert len(calls) == 1
    assert (second[0].story.id, second[0].production, second[0].challenger) == (
        7,
        0.9,
        0.1,
    )
    cached, original = second[0].embedding, first[0].embedding
    assert cached is not None and original is not None
    assert np.allclose(cached, original)
    db.write_bytes(b"xy")  # new snapshot -> rescore
    cached_scores(db, "missing.toml", Config(), 1, cache)
    assert len(calls) == 2


def test_report_counts_pairs_and_batch_winners(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "calibration.jsonl"
    record = {
        "story_ids": [1, 2, 3, 4, 5],
        "order": [0, 1, 2, 3, 4],
        "production": [5.0, 4.0, 3.0, 2.0, 1.0],  # matches the user exactly
        "challenger": [1.0, 2.0, 3.0, 4.0, 5.0],  # exactly reversed
    }
    log.write_text(json.dumps(record) + "\n")
    report(log)
    out = capsys.readouterr().out
    assert "production  agrees with you on 10/10 pairs (100%)" in out
    assert "challenger  agrees with you on 0/10 pairs (0%)" in out
    assert "production better 1" in out
