"""Legacy feature assembler for offline eval / ablation scripts.

`_augment_features` was the original 15-parameter feature builder that
preceded `_svm_personalization_features` (in `pipeline.py`). The production
personalization path uses the slim version; this richer assembler is kept
for the offline eval scripts (`eval.py`, `eval_rss.py`, `eval_no_hn_features.py`)
and the feature-ablation script (`scripts/feature_ablation.py`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


_LOG_POINTS_SCALE = 8.0  # log1p(~3000) ≈ 8
_LOG_COMMENTS_SCALE = 7.0  # log1p(~1000) ≈ 6.9
_LOG_QUALITY_SCALE = 8.0  # log1p(score/(age_hours+1)) rarely exceeds 8
_LOG_VELOCITY_SCALE = 8.0  # log1p(score/max(age_h, 0.1)) rarely exceeds 8


def _augment_features(
    embeddings: NDArray[np.float32],
    scores: list[int] | np.ndarray,
    age_seconds: list[float] | np.ndarray,
    comment_counts: np.ndarray | None = None,
    text_lengths: np.ndarray | None = None,
    hn_quality: np.ndarray | None = None,
    score_velocity: np.ndarray | None = None,
    comment_velocity: np.ndarray | None = None,
    sim_to_upvoted: np.ndarray | None = None,
    sim_to_downvoted: np.ndarray | None = None,
    closest_upvoted: np.ndarray | None = None,
    closest_downvoted: np.ndarray | None = None,
    comment_score_ratio: np.ndarray | None = None,
    is_hn_live: np.ndarray | None = None,
    is_archive: np.ndarray | None = None,
    is_reddit: np.ndarray | None = None,
    is_rss: np.ndarray | None = None,
    engagement_ratio: np.ndarray | None = None,
) -> NDArray[np.float32]:
    n = len(scores)
    n_meta = 1
    for f in (
        comment_counts,
        text_lengths,
        hn_quality,
        comment_score_ratio,
        is_hn_live,
        is_archive,
        is_reddit,
        is_rss,
        engagement_ratio,
    ):
        if f is not None:
            n_meta += 1
    if score_velocity is not None and comment_velocity is not None:
        n_meta += 2
    if sim_to_upvoted is not None:
        n_meta += 4

    meta = np.zeros((n, n_meta), dtype=np.float32)
    col = 0

    # log points
    meta[:, col] = (
        np.clip(np.log1p(np.maximum(scores, 0)), 0, _LOG_POINTS_SCALE)
        / _LOG_POINTS_SCALE
    )
    col += 1

    if comment_counts is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(comment_counts, 0)), 0, _LOG_COMMENTS_SCALE)
            / _LOG_COMMENTS_SCALE
        )
        col += 1

    if text_lengths is not None:
        meta[:, col] = np.clip(np.log1p(np.maximum(text_lengths, 0)), 0, 12.0) / 12.0
        col += 1

    if hn_quality is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(hn_quality, 0)), 0, _LOG_QUALITY_SCALE)
            / _LOG_QUALITY_SCALE
        )
        col += 1

    if comment_score_ratio is not None:
        meta[:, col] = comment_score_ratio
        col += 1

    if score_velocity is not None and comment_velocity is not None:
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(score_velocity, 0)), 0, _LOG_VELOCITY_SCALE)
            / _LOG_VELOCITY_SCALE
        )
        col += 1
        meta[:, col] = (
            np.clip(np.log1p(np.maximum(comment_velocity, 0)), 0, _LOG_VELOCITY_SCALE)
            / _LOG_VELOCITY_SCALE
        )
        col += 1

    if sim_to_upvoted is not None:
        assert (
            sim_to_downvoted is not None
            and closest_upvoted is not None
            and closest_downvoted is not None
        )
        meta[:, col] = (np.clip(sim_to_upvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(sim_to_downvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(closest_upvoted, -1, 1) + 1) / 2
        col += 1
        meta[:, col] = (np.clip(closest_downvoted, -1, 1) + 1) / 2
        col += 1

    if is_hn_live is not None:
        meta[:, col] = is_hn_live
        col += 1
    if is_archive is not None:
        meta[:, col] = is_archive
        col += 1
    if is_reddit is not None:
        meta[:, col] = is_reddit
        col += 1
    if is_rss is not None:
        meta[:, col] = is_rss
        col += 1

    if engagement_ratio is not None:
        meta[:, col] = engagement_ratio
        col += 1

    return np.concatenate([embeddings, meta], axis=1)
