from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "fast_route_controls",
    Path(__file__).parents[1]
    / "docs/evaluations/decision1-20261010/fast_route_controls.py",
)
assert spec is not None and spec.loader is not None
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def test_cache_hash_changes_with_every_answer_input() -> None:
    original = controls.request_hash({"x": 1}, {"yes": "instruction"}, "rev1", "fp32")
    assert original == controls.request_hash(
        {"x": 1}, {"yes": "instruction"}, "rev1", "fp32"
    )
    for value in [
        controls.request_hash({"x": 2}, {"yes": "instruction"}, "rev1", "fp32"),
        controls.request_hash({"x": 1}, {"yes": "new instruction"}, "rev1", "fp32"),
        controls.request_hash({"x": 1}, {"yes": "instruction"}, "rev2", "fp32"),
        controls.request_hash({"x": 1}, {"yes": "instruction"}, "rev1", "bf16"),
    ]:
        assert value != original


def test_cache_hash_ignores_dictionary_order() -> None:
    assert controls.request_hash(
        {"a": 1, "b": 2}, {}, "r", "fp32"
    ) == controls.request_hash({"b": 2, "a": 1}, {}, "r", "fp32")


def test_pairwise_auc_counts_ties_half_and_excludes_unjudged() -> None:
    assert controls.auc(np.array([0.7, 0.7, 0.9, 0.1]), np.array([2, 0, -1, 1])) == 0.75
    assert controls.auc(np.array([0.8, 0.9]), np.array([2, -1])) is None


def test_ndcg_perfect_order_and_rank_percentiles() -> None:
    assert controls.ndcg40(np.array([0.9, 0.8, 0.1]), np.array([2, 2, 0])) == 1
    assert controls.ndcg40(np.array([0.9, 0.8, 0.1]), np.array([0, 2, 2])) < 1
    np.testing.assert_allclose(
        controls.pct(np.array([2, 2, 1, 3])), [0.625, 0.625, 0.25, 1]
    )
    with pytest.raises(ValueError):
        controls.ndcg40(np.array([0.9, 0.1]), np.array([0, 1]))
