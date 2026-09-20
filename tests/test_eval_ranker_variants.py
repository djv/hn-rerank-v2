"""AST-based checks for scripts/eval_ranker_variants.py."""

import ast
from pathlib import Path

import numpy as np
import pytest

from database import Story
from pipeline import Config

SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_ranker_variants.py"


def test_leak_check_flag_in_help() -> None:
    """--leak-check must be wired into argparse.

    Parses the script's source for `argparse.ArgumentParser.add_argument`
    calls containing the literal `--leak-check`. Cheaper than booting a
    subprocess (the script imports sklearn + onnx at top, ~2.5s).
    """
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--leak-check":
                    return
    pytest.fail("--leak-check not in any argparse add_argument call")


def test_split_flag_in_help() -> None:
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--split":
                    return
    pytest.fail("--split not in any argparse add_argument call")


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
