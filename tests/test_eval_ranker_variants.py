"""Tests for scripts/eval_ranker_variants.py."""

from pathlib import Path

import numpy as np
from hypothesis import given, settings, strategies as st
import pytest

from database import Story
from pipeline import Config


@pytest.mark.parametrize(
    "argv",
    [
        ["--split", "stratified"],  # retired: chronological evaluation only
        ["--leak-check", "--leak-seeds", "0"],
        ["--folds", "0"],
    ],
)
def test_cli_rejects_invalid_evaluation_setups(argv: list[str]) -> None:
    """Bad setups fail at argument parsing, before any DB or model work."""
    from scripts.eval_ranker_variants import main

    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def _eval_story(sid: int) -> Story:
    return Story(
        id=sid,
        title=f"Eval {sid}",
        url=f"https://example.com/eval/{sid}",
        score=1,
        time=1,
        text_content="story text",
        source="hn",
        comment_count=1,
    )


def test_external_embedding_snapshot_validates_candidate_identity(
    tmp_path: Path,
) -> None:
    from scripts.eval_ranker_variants import (
        _embedding_text_hashes,
        _load_external_embeddings,
    )

    stories = [_eval_story(1), _eval_story(2)]
    path = tmp_path / "embeddings.npz"
    expected = np.eye(2, 384, dtype=np.float32)
    np.savez(
        path,
        story_ids=np.array([1, 2], dtype=np.int64),
        text_hashes=_embedding_text_hashes(stories),
        embeddings=expected,
    )

    actual = _load_external_embeddings(path, stories)

    assert np.array_equal(actual, expected)


def test_external_embedding_snapshot_rejects_stale_text(tmp_path: Path) -> None:
    from scripts.eval_ranker_variants import _load_external_embeddings

    stories = [_eval_story(1)]
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        story_ids=np.array([1], dtype=np.int64),
        text_hashes=np.array(["0" * 64], dtype="<U64"),
        embeddings=np.ones((1, 384), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="text hashes"):
        _load_external_embeddings(path, stories)


def test_external_embedding_snapshot_freezes_candidate_membership(
    tmp_path: Path,
) -> None:
    import json
    from dataclasses import asdict

    from scripts.eval_ranker_variants import (
        _embedding_text_hashes,
        _snapshot_candidate_stories,
    )

    stories = [_eval_story(2), _eval_story(1)]
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        story_ids=np.array([2, 1], dtype=np.int64),
        text_hashes=_embedding_text_hashes(stories),
        embeddings=np.eye(2, 384, dtype=np.float32),
        stories_json=np.array(json.dumps([asdict(story) for story in stories])),
    )

    loaded = _snapshot_candidate_stories(path)

    assert [story.id for story in loaded] == [2, 1]


def test_external_embedding_snapshot_freezes_feedback_labels_and_times(
    tmp_path: Path,
) -> None:
    import json
    from dataclasses import asdict

    from scripts.eval_ranker_variants import _snapshot_feedback

    stories = [_eval_story(2), _eval_story(1)]
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        feedback_story_ids=np.array([2, 1], dtype=np.int64),
        feedback_stories_json=np.array(
            json.dumps([asdict(story) for story in stories])
        ),
        feedback_labels=np.array([2, 0], dtype=np.int8),
        feedback_vote_times=np.array([10.0, 20.0], dtype=np.float64),
    )

    loaded_stories, labels, vote_times = _snapshot_feedback(path)

    assert [story.id for story in loaded_stories] == [2, 1]
    assert labels.tolist() == [2, 0]
    assert vote_times.tolist() == [10.0, 20.0]


def test_candidate_cap_retains_feedback_story_ids() -> None:
    from scripts.eval_ranker_variants import _candidate_indices_with_feedback

    candidates = [_eval_story(sid) for sid in range(1, 11)]

    indices = _candidate_indices_with_feedback(
        candidates,
        max_candidates=4,
        required_story_ids={2, 8},
    )

    selected_ids = {candidates[index].id for index in indices}
    assert len(indices) == 4
    assert {2, 8} <= selected_ids


def test_make_fold_removes_training_feedback_but_keeps_held_out() -> None:
    from scripts.eval_ranker_variants import _make_fold

    candidates = [_eval_story(1), _eval_story(2), _eval_story(3)]
    cand_emb = np.zeros((3, 384), dtype=np.float32)
    cand_emb[0, 0] = 1.0
    cand_emb[1, 1] = 1.0
    cand_emb[2, 2] = 1.0
    fb_to_cand = np.array([0, 1, 2], dtype=int)
    valid_positions = np.array([0, 1, 2], dtype=int)

    fold = _make_fold(
        candidates,
        cand_emb,
        candidates,
        fb_to_cand,
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([2, 0, 2], dtype=int),
        valid_positions,
        np.array([0, 1], dtype=int),
        np.array([2], dtype=int),
        Config(),
    )

    assert [story.id for story in fold.candidates] == [3]
    assert [story.id for story in fold.train_stories] == [1, 2]
    assert [story.id for story in fold.test_stories] == [3]


@pytest.mark.parametrize("judged_only", [False, True])
def test_fold_groups_crossposts_and_replay_keeps_all_test_classes(
    judged_only: bool,
) -> None:
    from dataclasses import replace
    from scripts.eval_ranker_variants import _make_fold

    rows = [_eval_story(i) for i in range(1, 8)]
    original_url = rows[0].url
    assert original_url is not None
    rows[1] = replace(rows[1], url=original_url + "?utm_source=other")
    rows[3] = replace(rows[3], url=rows[2].url)
    emb = np.eye(7, 384, dtype=np.float32)
    fold = _make_fold(
        rows,
        emb,
        rows,
        np.arange(7),
        np.arange(7, dtype=float),
        np.array([2, 2, 0, 2, 1, 2, 0]),
        np.arange(7),
        np.array([0]),
        np.array([1, 2, 3, 4, 5]),
        Config(),
        feedback_embeddings=emb,
        needs_experimental=False,
        judged_only=judged_only,
    )
    assert [s.id for s in fold.test_stories] == [3, 5, 6]
    assert fold.test_actions.tolist() == [0, 1, 2]
    assert [s.id for s in fold.candidates] == (
        [3, 5, 6] if judged_only else [3, 5, 6, 7]
    )
    np.testing.assert_array_equal(fold.cand_emb[0], emb[2])


def _metric_fold(test_ids: list[int], test_actions: list[int]):
    from scripts.eval_ranker_variants import FoldData

    candidates = [_eval_story(sid) for sid in range(1, 51)]
    cand_emb = np.zeros((50, 384), dtype=np.float32)
    for row in range(50):
        cand_emb[row, row % 384] = 1.0
    empty_2d = np.empty((0, 0), dtype=np.float32)
    return FoldData(
        candidates=candidates,
        cand_emb=cand_emb,
        train_stories=[],
        test_stories=[_eval_story(sid) for sid in test_ids],
        test_actions=np.array(test_actions, dtype=int),
        train_vote_times=np.empty(0, dtype=np.float64),
        x_train_base=empty_2d,
        x_cand_base=empty_2d,
        y_train=np.empty(0, dtype=int),
        tier2_scores=np.zeros(50, dtype=np.float32),
    )


def test_metrics_include_time_forward_dashboard_keys() -> None:
    from scripts.eval_ranker_variants import _metrics

    fold = _metric_fold([1, 5, 12, 41], [2, 0, 2, 1])
    scores = -np.arange(50, dtype=np.float32)

    metrics = _metrics(scores, fold, Config())["raw"]

    ideal = 1.0 + (1.0 / np.log2(3))
    actual = 1.0 + (1.0 / np.log2(13))
    assert metrics["ndcg_at_12"] == pytest.approx(actual / ideal)
    assert metrics["up_recall_at_12"] == 1.0
    assert metrics["up_recall_at_40"] == 1.0
    assert metrics["hit_at_40"] == 0.75
    assert metrics["known_upvote_fraction_at_40"] == 2 / 40
    assert metrics["known_downvote_fraction_at_40"] == 1 / 40


def test_metrics_zero_up_recall_when_no_held_out_upvotes() -> None:
    from scripts.eval_ranker_variants import _metrics

    fold = _metric_fold([1, 2], [0, 1])
    scores = -np.arange(50, dtype=np.float32)

    metrics = _metrics(scores, fold, Config())["raw"]

    assert metrics["up_recall_at_12"] is None
    assert metrics["up_recall_at_40"] is None


def test_temporal_splits_train_only_on_prior_feedback() -> None:
    from scripts.eval_ranker_variants import _temporal_splits

    y = np.array([2, 0, 1, 2, 0, 1, 2, 0], dtype=int)
    vote_times = np.array([80, 10, 70, 20, 60, 30, 50, 40], dtype=np.float64)

    splits = _temporal_splits(y, vote_times, folds=2)

    assert len(splits) == 2
    for split in splits:
        assert np.max(vote_times[split.train_pos]) < np.min(vote_times[split.test_pos])
    assert set(splits[0].train_pos) < set(splits[1].train_pos)


def test_report_aggregation_shape_includes_new_metrics_and_baselines() -> None:
    from scripts.eval_ranker_variants import _aggregate_results

    _metrics_row = {
        "raw": {
            "ndcg_at_12": 0.1,
            "ndcg_at_100": 0.2,
            "ndcg_at_40": 0.3,
            "ndcg_at_200": 0.4,
            "map": 0.5,
            "known_upvote_fraction_at_40": 0.6,
            "up_recall_at_12": 0.7,
            "up_recall_at_40": 0.8,
            "known_downvote_fraction_at_40": 0.9,
            "hit_at_40": 1.0,
            "hit_at_100": 1.0,
            "median_rank": 2.0,
            "p25_rank": 1.0,
            "p75_rank": 3.0,
            "brier_up": 0.11,
        },
        "mmr": {
            "ndcg_at_12": 0.1,
            "ndcg_at_100": 0.2,
            "ndcg_at_40": 0.3,
            "ndcg_at_200": 0.4,
            "map": 0.5,
            "known_upvote_fraction_at_40": 0.6,
            "up_recall_at_12": 0.7,
            "up_recall_at_40": 0.8,
            "known_downvote_fraction_at_40": 0.9,
            "hit_at_40": 1.0,
            "hit_at_100": 1.0,
            "median_rank": 2.0,
            "p25_rank": 1.0,
            "p75_rank": 3.0,
            "brier_up": 0.11,
        },
    }

    report = {
        "variants": _aggregate_results({"margin3_up": [_metrics_row]}),
        "baselines": _aggregate_results({"candidate_order": [_metrics_row]}),
    }

    for section in ("variants", "baselines"):
        payload = next(iter(report[section].values()))
        assert "ndcg_at_12" in payload["mean"]["raw"]
        assert "up_recall_at_40" in payload["std"]["raw"]
        assert "hit_at_40" in payload["per_fold"][0]["raw"]


@settings(max_examples=40, deadline=None)
@given(
    labels=st.lists(st.sampled_from([0, 1, 2]), min_size=2, max_size=50),
    seed=st.integers(0, 2**32 - 1),
)
def test_metrics_auc_matches_sklearn_on_judged_cards(
    labels: list[int], seed: int
) -> None:
    """auc_up_vs_rest/down equal sklearn's ROC-AUC over the judged cards
    (distinct scores, so no ties), and are None without both classes."""
    from sklearn.metrics import roc_auc_score

    from scripts.eval_ranker_variants import _metrics

    ids = list(range(1, len(labels) + 1))
    fold = _metric_fold(ids, labels)
    scores = np.random.default_rng(seed).permutation(50).astype(np.float32)
    metrics = _metrics(scores, fold, Config())["raw"]

    judged_scores = scores[: len(labels)]
    y = np.array(labels)
    for name, mask in (("rest", np.ones(len(y), bool)), ("down", y != 1)):
        target = y[mask] == 2
        if target.all() or not target.any():
            assert metrics[f"auc_up_vs_{name}"] is None
        else:
            assert metrics[f"auc_up_vs_{name}"] == pytest.approx(
                roc_auc_score(target, judged_scores[mask])
            )


def test_model_override_spec_is_typed_by_field_default() -> None:
    from scripts.eval_ranker_variants import _parse_model_overrides

    assert _parse_model_overrides("svm_c=2;knn_k=20;svm_gamma=0.05") == {
        "svm_c": 2.0,
        "knn_k": 20,
        "svm_gamma": 0.05,
    }
    assert _parse_model_overrides("deduplicate_training_feedback=true") == {
        "deduplicate_training_feedback": True
    }
    with pytest.raises(ValueError):
        _parse_model_overrides("no_such_field=1")


def test_replay_embeddings_require_every_story_with_unchanged_text(
    tmp_path: Path,
) -> None:
    from scripts.eval_ranker_variants import _load_replay_embeddings

    path = tmp_path / "replay.npz"
    np.savez(
        path,
        story_ids=np.array([1, 2]),
        text_hashes=np.array(["a", "b"]),
        embeddings=np.eye(2, dtype=np.float32),
    )
    vectors = _load_replay_embeddings(path, {2: "b", 1: "a"})
    np.testing.assert_array_equal(vectors[2], [0.0, 1.0])
    with pytest.raises(ValueError, match="1 missing"):
        _load_replay_embeddings(path, {1: "a", 3: "c"})
    with pytest.raises(ValueError, match="1 with changed text"):
        _load_replay_embeddings(path, {1: "a", 2: "changed"})


def test_validated_embeddings_accept_one_shared_dimension() -> None:
    from scripts.eval_ranker_variants import _validated_embeddings

    stories = [_eval_story(1), _eval_story(2)]
    wide = {1: np.eye(768, dtype=np.float32)[0], 2: np.eye(768, dtype=np.float32)[1]}
    assert _validated_embeddings(stories, wide).shape == (2, 768)
    mixed = {1: wide[1], 2: np.eye(384, dtype=np.float32)[0]}
    with pytest.raises(ValueError, match="1/2 valid"):
        _validated_embeddings(stories, mixed)


def test_concatenated_replay_embeddings_stay_unit_length(tmp_path: Path) -> None:
    from scripts.eval_ranker_variants import _load_concatenated_replay

    paths = []
    for name, dim in (("a", 3), ("b", 2)):
        path = tmp_path / f"{name}.npz"
        vectors = np.zeros((2, dim), dtype=np.float32)
        vectors[:, 0] = 1.0
        np.savez(
            path,
            story_ids=np.array([1, 2]),
            text_hashes=np.array(["h1", "h2"]),
            embeddings=vectors,
        )
        paths.append(path)
    joined = _load_concatenated_replay(paths, {1: "h1", 2: "h2"})
    assert joined[1].shape == (5,)
    assert np.linalg.norm(joined[2]) == pytest.approx(1.0)
