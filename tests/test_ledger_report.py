"""Unit tests for the pure metric functions in scripts/ledger_report.py.

No DB, no sklearn — these are the bucketing/aggregation/AUC/drift helpers
that ledger_report.py's five report sections are built from.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from scripts.ledger_report import (
    bucket_position,
    cap_dwell_ms,
    centroid,
    cosine_distance,
    percentile,
    rank_auc,
)


class TestBucketPosition:
    def test_zero_is_bucket_zero(self) -> None:
        assert bucket_position(0) == 0

    def test_within_width_stays_in_bucket(self) -> None:
        assert bucket_position(4, width=5) == 0
        assert bucket_position(5, width=5) == 5
        assert bucket_position(9, width=5) == 5

    def test_negative_position_rejected(self) -> None:
        with pytest.raises(ValueError):
            bucket_position(-1)

    @given(st.integers(min_value=0, max_value=10_000), st.integers(min_value=1, max_value=50))
    def test_bucket_is_always_a_multiple_of_width_and_le_position(
        self, position: int, width: int
    ) -> None:
        bucket = bucket_position(position, width=width)
        assert bucket % width == 0
        assert bucket <= position < bucket + width


class TestCapDwellMs:
    def test_within_cap_unchanged(self) -> None:
        assert cap_dwell_ms(500, cap_ms=1000) == 500

    def test_over_cap_clamped(self) -> None:
        assert cap_dwell_ms(5000, cap_ms=1000) == 1000

    def test_negative_clamped_to_zero(self) -> None:
        assert cap_dwell_ms(-5, cap_ms=1000) == 0

    @given(st.integers(min_value=-10_000, max_value=10_000), st.integers(min_value=1, max_value=10_000))
    def test_result_always_in_bounds(self, duration_ms: int, cap_ms: int) -> None:
        result = cap_dwell_ms(duration_ms, cap_ms=cap_ms)
        assert 0 <= result <= cap_ms


class TestPercentile:
    def test_empty_is_nan(self) -> None:
        assert np.isnan(percentile([], 50))

    def test_single_value(self) -> None:
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 0) == 42.0
        assert percentile([42.0], 100) == 42.0

    def test_p50_of_sorted_range(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 50) == 3.0

    def test_p100_is_max(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 100) == 3.0

    def test_invalid_quantile_rejected(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0], 101)

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1))
    def test_result_is_one_of_the_input_values(self, values: list[float]) -> None:
        result = percentile(values, 50)
        assert result in values


class TestRankAuc:
    def test_perfect_separation(self) -> None:
        # positives all score higher than negatives
        assert rank_auc([1.0, 2.0, 3.0], [False, False, True]) == 1.0

    def test_perfect_anti_separation(self) -> None:
        assert rank_auc([1.0, 2.0, 3.0], [True, False, False]) == 0.0

    def test_random_is_half(self) -> None:
        # symmetric interleave -> exactly 0.5
        assert rank_auc([1.0, 1.0, 2.0, 2.0], [True, False, True, False]) == 0.5

    def test_empty_class_is_nan(self) -> None:
        assert np.isnan(rank_auc([1.0, 2.0], [True, True]))
        assert np.isnan(rank_auc([], []))

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError):
            rank_auc([1.0], [True, False])

    @given(
        st.lists(
            st.tuples(st.floats(min_value=-100, max_value=100, allow_nan=False), st.booleans()),
            min_size=2,
            max_size=30,
        )
    )
    def test_invariant_under_strictly_increasing_transform(
        self, pairs: list[tuple[float, bool]]
    ) -> None:
        scores = [s for s, _ in pairs]
        labels = [lbl for _, lbl in pairs]
        assume(any(labels) and not all(labels))  # need both classes present
        base = rank_auc(scores, labels)
        # Rank-transform: strictly order-preserving regardless of the
        # original floats' spacing (an affine map can collapse distinct
        # near-zero floats back together at float64 precision).
        order = {s: i for i, s in enumerate(sorted(set(scores)))}
        transformed = [float(order[s]) for s in scores]
        assert rank_auc(transformed, labels) == pytest.approx(base)

    @given(
        st.lists(
            st.tuples(st.floats(min_value=-100, max_value=100, allow_nan=False), st.booleans()),
            min_size=2,
            max_size=30,
        )
    )
    def test_flipping_labels_gives_one_minus_auc(
        self, pairs: list[tuple[float, bool]]
    ) -> None:
        scores = [s for s, _ in pairs]
        labels = [lbl for _, lbl in pairs]
        assume(any(labels) and not all(labels))
        base = rank_auc(scores, labels)
        flipped = [not lbl for lbl in labels]
        assert rank_auc(scores, flipped) == pytest.approx(1.0 - base)


class TestCentroid:
    def test_single_vector_is_itself(self) -> None:
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(centroid(v), v[0])

    def test_mean_of_two(self) -> None:
        v = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(centroid(v), [1.0, 2.0])

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            centroid(np.zeros((0, 3), dtype=np.float32))


class TestCosineDistance:
    def test_identical_vectors_zero_distance(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_distance_one(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_zero_vector_is_nan(self) -> None:
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert np.isnan(cosine_distance(a, b))
