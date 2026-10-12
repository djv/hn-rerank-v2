from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.simulate_interleaving import (
    ArmSpec,
    compare,
    load_blocks,
    simulate_days,
)


def _dump(path: Path, scores: dict[str, list[float]], labels: list[int]) -> Path:
    fold = {
        "fold": 1,
        "ids": list(range(100, 100 + len(labels))),
        "labels": labels,
        "scores": scores,
    }
    path.write_text(json.dumps({"config": {}, "folds": [fold]}))
    return path


def test_replay_credits_each_vote_once_and_finds_the_better_arm(
    tmp_path: Path,
) -> None:
    labels = [2] * 20 + [0] * 20 + [-1] * 5
    good = [1.0] * 20 + [0.0] * 20 + [0.5] * 5
    bad = [0.0] * 20 + [1.0] * 20 + [0.5] * 5
    file = _dump(tmp_path / "s.json", {"control": bad, "challenger": good}, labels)
    arms = [ArmSpec("c", "control", (file,)), ArmSpec("x", "challenger", (file,))]

    blocks = load_blocks(arms, challenger_weight=1.0)
    assert len(blocks) == 1 and len(blocks[0].ids) == 40  # unjudged dropped
    credits = simulate_days(
        blocks,
        days=3,
        votes_per_day=40,
        deck_size=6,
        votes_per_deck=2,
        rng=np.random.default_rng(0),
    )
    assert len(credits) == 120
    for day in range(3):
        # Each vote of a day is a distinct story of the block.
        assert sorted(c.label for c in credits if c.day == day).count(2) == 20
    result = compare(credits, 1)
    assert result.up_rate > result.control_up_rate
    assert result.deck_wins > result.deck_losses

    same = load_blocks(arms, challenger_weight=0.0)
    assert same[0].rankings[0] == same[0].rankings[1]


def test_blocks_must_share_stories_and_labels(tmp_path: Path) -> None:
    a = _dump(tmp_path / "a.json", {"v": [1.0, 0.0]}, [2, 0])
    b = _dump(tmp_path / "b.json", {"v": [1.0, 0.0]}, [2, 1])
    with pytest.raises(ValueError, match="stories or labels differ"):
        load_blocks(
            [ArmSpec("a", "v", (a,)), ArmSpec("b", "v", (b,))], challenger_weight=1
        )
