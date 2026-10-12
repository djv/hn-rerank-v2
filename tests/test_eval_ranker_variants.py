"""Tests for scripts/eval_ranker_variants.py."""

from pathlib import Path
from typing import Any

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
        ["--candidate-pool", "impressions"],  # needs --holdout-after
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


def test_metrics_auc_up_vs_all_counts_unjudged_cards_as_not_up() -> None:
    from scripts.eval_ranker_variants import _metrics

    fold = _metric_fold([1, 5, 12, 41], [2, 0, 2, 1])
    scores = -np.arange(50, dtype=np.float32)

    metrics = _metrics(scores, fold, Config())["raw"]

    # Upvotes at ranks 0 and 11 of 50: 48 + 38 of 2 * 48 pairs up-first.
    assert metrics["auc_up_vs_all"] == pytest.approx(86 / 96)
    assert metrics["auc_up_vs_rest"] == pytest.approx(3 / 4)
    assert metrics["known_upvote_fraction_at_12"] == 2 / 12


def test_metrics_score_each_slice_on_its_own_stories() -> None:
    from dataclasses import replace

    from scripts.eval_ranker_variants import _metrics

    fold = replace(
        _metric_fold([1, 5, 12, 41], [2, 0, 2, 1]),
        slices={"feed_popular": {5, 12, 30, 99}},
    )
    scores = -np.arange(50, dtype=np.float32)

    sliced = _metrics(scores, fold, Config())["raw_feed_popular"]

    # Order within the slice is 5, 12, 30 (99 is not a candidate): one
    # upvote (12) ranked above one unvoted card and below one downvote.
    assert sliced["returned_cards"] == 3
    assert sliced["eligible_positives"] == 1
    assert sliced["auc_up_vs_all"] == pytest.approx(1 / 2)
    assert sliced["auc_up_vs_down"] == 0.0
    assert sliced["known_upvote_fraction_at_12"] == 1 / 3


def test_first_shown_feeds_group_stories_by_first_impression(tmp_path: Path) -> None:
    from database import Database, InteractionEvent
    from scripts.eval_ranker_variants import _first_shown_feeds

    db = Database(str(tmp_path / "f.db"))
    for sid in (1, 2):
        db.upsert_story(_eval_story(sid))

    def shown(eid: str, sid: int, at: float, mode: str, user: int = 7):
        return InteractionEvent(
            event_id=eid,
            client_session_id="s",
            user_id=user,
            story_id=sid,
            event_type="impression",
            dashboard_version=0,
            position=0,
            sort_mode=mode,
            age_filter="1d",
            source_filter="all",
            ranker_arm="tui_observed",
            occurred_at=at,
        )

    db.insert_interaction_events(
        [
            shown("a", 1, 20.0, "recommended"),
            shown("b", 1, 10.0, "popular"),
            shown("c", 2, 30.0, "explore"),
            shown("d", 2, 5.0, "popular", user=8),  # another user's impression
        ]
    )

    assert _first_shown_feeds(db, 7) == {"feed_popular": {1}, "feed_explore": {2}}
    db.close()


def test_window_gravity_ages_stories_at_the_blocks_median_vote() -> None:
    from dataclasses import replace

    from scripts.eval_ranker_variants import _scores_window_gravity

    fold = replace(
        _metric_fold([1], [2]),
        candidates=[
            replace(_eval_story(1), score=100, time=0),
            replace(_eval_story(2), score=40, time=14 * 3600),
        ],
        test_vote_times=np.array([10.0, 16 * 3600.0, 30 * 3600.0]),
    )

    scores = _scores_window_gravity(fold, 8.0)

    # Now = 16 h: ages 16 h and 2 h on an 8-hour clock.
    assert scores == pytest.approx([100 / (2 + 2) ** 1.8, 40 / (0.25 + 2) ** 1.8])
    assert scores[1] > scores[0]


def test_impression_pools_take_each_blocks_unvoted_shown_stories() -> None:
    from scripts.eval_ranker_variants import (
        SkippedPool,
        _holdout_splits,
        _impression_pools,
    )

    vote_times = np.array([5, 10, 20, 30], dtype=np.float64)
    voted_ids = [1, 2, 3, 4]
    splits = _holdout_splits(vote_times, 10, blocks=2)
    shown = {100: 12, 101: 31, 102: 3, 3: 11, 4: 15, 1: 25}
    pool = SkippedPool(
        stories=[_eval_story(sid) for sid in shown],
        first_shown=np.array(list(shown.values()), dtype=np.float64),
        emb=np.zeros((len(shown), 384), dtype=np.float32),
    )

    pools = _impression_pools(pool, splits, voted_ids, vote_times, 10)

    # 102 was shown before the holdout, 4 is voted in block 2 and 1 in
    # training, so neither may enter block 1 as an unvoted story.
    assert pools == {1: {100, 3}, 2: {101}}


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


@given(
    times=st.lists(st.integers(0, 40), min_size=2, max_size=60),
    after=st.integers(0, 40),
    blocks=st.integers(1, 6),
)
def test_holdout_splits_cover_held_votes_with_prior_training(
    times: list[int], after: int, blocks: int
) -> None:
    from scripts.eval_ranker_variants import _holdout_splits

    vote_times = np.array(times, dtype=np.float64)
    held_groups = len(np.unique(vote_times[vote_times >= after]))
    if not 1 <= blocks <= held_groups:
        with pytest.raises(ValueError):
            _holdout_splits(vote_times, after, blocks=blocks)
        return

    splits = _holdout_splits(vote_times, after, blocks=blocks)

    assert len(splits) == blocks
    tested = np.concatenate([split.test_pos for split in splits])
    assert sorted(tested) == list(np.flatnonzero(vote_times >= after))
    for split in splits:
        assert not set(split.train_pos) & set(split.test_pos)
        if len(split.train_pos):
            assert vote_times[split.train_pos].max() < vote_times[split.test_pos].min()
    assert set(splits[0].train_pos) == set(np.flatnonzero(vote_times < after))


@given(
    times=st.lists(st.integers(0, 40), min_size=2, max_size=60),
    after=st.integers(1, 40),
    blocks=st.integers(1, 4),
    limit=st.integers(1, 30),
)
def test_recent_training_keeps_only_the_latest_prior_votes(
    times: list[int], after: int, blocks: int, limit: int
) -> None:
    from scripts.eval_ranker_variants import _holdout_splits, _recent_training

    vote_times = np.array(times, dtype=np.float64)
    if not 1 <= blocks <= len(np.unique(vote_times[vote_times >= after])):
        return
    splits = _holdout_splits(vote_times, after, blocks=blocks)

    for full, short in zip(
        splits, _recent_training(splits, vote_times, limit), strict=True
    ):
        assert short.fold_no == full.fold_no
        np.testing.assert_array_equal(short.test_pos, full.test_pos)
        assert len(short.train_pos) == min(limit, len(full.train_pos))
        assert set(short.train_pos) <= set(full.train_pos)
        dropped = sorted(set(full.train_pos) - set(short.train_pos))
        if dropped and len(short.train_pos):
            # Nothing dropped is newer than anything kept.
            assert vote_times[dropped].max() <= vote_times[short.train_pos].min()


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


def test_metrics_count_annoying_meh_and_discovery_cards_in_top_12() -> None:
    from dataclasses import replace

    from scripts.eval_ranker_variants import _metrics

    fold = _metric_fold([1, 30, 5, 7], [2, 2, 0, 1])
    # One training upvote whose similarity to candidate row r is r / 50, so
    # rows below the median (ids 1-25) are the less familiar half.
    up = (np.arange(384, dtype=np.float32) / 50.0)[None, :]
    up[0, 50:] = 0.0
    fold = replace(
        fold,
        train_stories=[_eval_story(999)],
        train_emb=up,
        train_vote_times=np.array([0.0]),
        y_train=np.array([2]),
    )
    scores = -np.arange(50, dtype=np.float32)  # ids 1..12 are the top 12

    metrics = _metrics(scores, fold, Config())["raw"]

    assert metrics["known_downvote_fraction_at_12"] == 1 / 12
    assert metrics["known_neutral_fraction_at_12"] == 1 / 12
    assert metrics["discovery_upvotes_at_12"] == 1  # id 1: upvoted, unfamiliar
    assert metrics["non_hn_upvotes_at_12"] == 0
    # Reversed ranking: the top 12 are ids 39-50, all familiar, so the
    # upvote there (id 40) is not a discovery.
    reversed_fold = replace(
        fold, test_stories=[_eval_story(40)], test_actions=np.array([2])
    )
    flipped = _metrics(np.arange(50, dtype=np.float32), reversed_fold, Config())["raw"]
    assert flipped["up_recall_at_12"] == 1.0
    assert flipped["discovery_upvotes_at_12"] == 0


def test_out_of_fold_label_rates_ignore_own_vote() -> None:
    """With every key seen once, a row's rate is only the prior from the
    other folds, identical for all rows (stratified folds). Leave-one-out
    would give up rows a lower up-rate than down rows: a label leak."""
    from scripts.eval_ranker_variants import _out_of_fold_label_rates

    y = np.random.default_rng(0).permutation(np.repeat([0, 1, 2], 20))
    rates = _out_of_fold_label_rates([f"k{i}" for i in range(60)], y)
    np.testing.assert_allclose(rates, np.tile(rates[0], (60, 1)))
    np.testing.assert_allclose(rates[0], [1 / 3, 1 / 3])


def test_smoothed_label_rates_shrink_toward_global_rate() -> None:
    from scripts.eval_ranker_variants import (
        _STACK_PRIOR_STRENGTH,
        _smoothed_label_rates,
    )

    keys = ["a", "b", "b"]
    y = np.array([2, 0, 2])
    base = np.array([2 / 3, 1 / 3])
    k = _STACK_PRIOR_STRENGTH
    rates = _smoothed_label_rates(keys, y, ["a", "b", "c"])
    np.testing.assert_allclose(rates[0], (np.array([1, 0]) + k * base) / (1 + k))
    np.testing.assert_allclose(rates[1], (np.array([1, 1]) + k * base) / (2 + k))
    np.testing.assert_allclose(rates[2], base)


def _signal_fold(n: int = 150, n_train: int = 120) -> Any:
    """Up votes share an embedding direction, a domain and a title word."""
    from dataclasses import replace

    from scripts.eval_ranker_variants import _make_fold

    rng = np.random.default_rng(0)
    y = rng.permutation(np.repeat([0, 1, 2], n // 3))
    emb = rng.normal(size=(n, 384)).astype(np.float32) * 0.3
    emb[:, 0] += np.where(y == 2, 2.0, np.where(y == 0, -2.0, 0.0))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    domains = {0: "down.example", 1: "mid.example", 2: "up.example"}
    words = {0: "crypto", 1: "misc", 2: "compilers"}
    stories = [
        replace(
            _eval_story(i + 1),
            url=f"https://{domains[int(label)]}/{i}",
            title=f"Story {i} about {words[int(label)]}",
        )
        for i, label in enumerate(y)
    ]
    return _make_fold(
        stories,
        emb,
        stories,
        np.arange(n),
        np.arange(n, dtype=float) + 10.0,
        y,
        np.arange(n),
        np.arange(n_train),
        np.arange(n_train, n),
        Config(),
        feedback_embeddings=emb,
        judged_only=True,
    )


def _ups_above_downs(fold: Any, scores: np.ndarray) -> bool:
    by_id = dict(zip((s.id for s in fold.candidates), scores))
    pairs = list(zip((s.id for s in fold.test_stories), fold.test_actions))
    ups = [by_id[i] for i, a in pairs if a == 2]
    downs = [by_id[i] for i, a in pairs if a == 0]
    return min(ups) > max(downs)


@pytest.mark.parametrize(
    ("mode", "feats"),
    [
        ("gbm", "both"),
        ("ordinal", "both"),
        ("pair", "both"),
        ("lr", "both"),
        ("lr", "content"),
        ("lr", "meta"),
    ],
)
def test_stack_ranks_learnable_signal_first(mode: str, feats: str) -> None:
    """Every stack mode and feature set puts held-out ups above downs."""
    from scripts.eval_ranker_variants import _scores_stack

    fold = _signal_fold()
    scores, _ = _scores_stack(fold, Config(), 1.0, mode=mode, feats=feats)
    assert _ups_above_downs(fold, scores)


@pytest.mark.parametrize("target", ["updown", "up", "updown_only"])
def test_logreg_targets_rank_learnable_signal_first(target: str) -> None:
    from scripts.eval_ranker_variants import _scores_logreg_target

    fold = _signal_fold()
    assert _ups_above_downs(fold, _scores_logreg_target(fold, Config(), target))


def test_tfidf_learns_title_words_and_domains() -> None:
    from scripts.eval_ranker_variants import _scores_tfidf

    fold = _signal_fold()
    assert _ups_above_downs(fold, _scores_tfidf(fold, 4.0))


def test_skipped_stories_join_training_only_before_cutoff() -> None:
    """Shown-but-unvoted stories become training votes only when first shown
    before the fold's training cutoff and no story of the same URL group is
    trained or tested on."""
    from dataclasses import replace

    from scripts.eval_ranker_variants import SkippedPool, _with_skipped

    fold = _signal_fold()
    cutoff = float(fold.train_vote_times.max())
    extra = [
        replace(_eval_story(1000 + i), url=f"https://x.example/{i}") for i in range(2)
    ]
    # A cross-post (new ID, same URL) of a held-out story is the same article.
    cross_post = replace(_eval_story(2000), url=fold.candidates[0].url)
    pool = SkippedPool(
        stories=[
            extra[0],
            extra[1],
            fold.train_stories[0],
            fold.candidates[0],
            cross_post,
        ],
        first_shown=np.array([cutoff - 1, cutoff + 1, cutoff - 1, cutoff - 1, 0.0]),
        emb=np.tile(np.eye(384, dtype=np.float32)[1], (5, 1)),
    )
    grown = _with_skipped(fold, pool, 0, 1.0, Config())
    added = grown.train_stories[len(fold.train_stories) :]
    assert [s.id for s in added] == [1000]
    assert grown.y_train[-1] == 0 and len(grown.y_train) == len(fold.y_train) + 1
    assert grown.x_train_base.shape[0] == len(grown.train_stories)
    assert grown.x_cand_base.shape == fold.x_cand_base.shape
    assert grown.candidates == fold.candidates and grown.runtime_db is None
    assert _with_skipped(fold, pool, 0, 0.0, Config()) is fold


def test_per_embedding_production_rejects_dims_that_do_not_split() -> None:
    from scripts.eval_ranker_variants import _per_embedding_production

    with pytest.raises(ValueError, match="do not split"):
        _per_embedding_production(_signal_fold(), Config(), None, [100, 200])


def test_auc_counts_tied_opposite_labels_as_half() -> None:
    """An upvote and a downvote with the same score are one coin flip: AUC
    0.5 whichever comes first in the list, not 0 or 1."""
    from scripts.eval_ranker_variants import _metrics

    for first, second in ((1, 2), (2, 1)):
        fold = _metric_fold(
            [first, second], [2 if first == 1 else 0, 0 if first == 1 else 2]
        )
        scores = np.zeros(50, dtype=np.float32)
        raw = _metrics(scores, fold, Config())["raw"]
        assert raw["auc_up_vs_rest"] == 0.5
        assert raw["auc_up_vs_down"] == 0.5
    # Distinct scores still count strictly.
    fold = _metric_fold([1, 2], [2, 0])
    raw = _metrics(-np.arange(50, dtype=np.float32), fold, Config())["raw"]
    assert raw["auc_up_vs_rest"] == 1.0


def test_percentile_scores_share_rank_on_ties() -> None:
    from scripts.eval_ranker_variants import _percentile_scores

    np.testing.assert_allclose(
        _percentile_scores(np.array([3.0, 1.0, 1.0, 2.0])), [1.0, 1 / 6, 1 / 6, 2 / 3]
    )
    np.testing.assert_allclose(_percentile_scores(np.zeros(3)), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(_percentile_scores(np.array([7.0])), [1.0])


@pytest.mark.parametrize(
    ("text", "char"),
    [("full", False), ("title", False), ("titledom", False), ("full", True)],
)
def test_tfidf_inputs_learn_the_signal(text: str, char: bool) -> None:
    from scripts.eval_ranker_variants import _scores_tfidf

    fold = _signal_fold()
    assert _ups_above_downs(fold, _scores_tfidf(fold, 4.0, text=text, char=char))


def test_joint_words_and_embeddings_learn_the_signal() -> None:
    from scripts.eval_ranker_variants import _scores_joint

    fold = _signal_fold()
    assert _ups_above_downs(fold, _scores_joint(fold, Config(), 1.0, half_life=30.0))


def test_source_prior_follows_training_vote_rates() -> None:
    from dataclasses import replace

    from scripts.eval_ranker_variants import _scores_source_prior

    fold = _signal_fold()
    source = {0: "rss_down", 1: "rss_mid", 2: "rss_up"}
    train = [
        replace(s, source=source[int(y)])
        for s, y in zip(fold.train_stories, fold.y_train)
    ]
    labels = dict(zip((s.id for s in fold.test_stories), fold.test_actions))
    cands = [replace(s, source=source[int(labels[s.id])]) for s in fold.candidates]
    scores = _scores_source_prior(replace(fold, train_stories=train, candidates=cands))
    by_label = {int(labels[s.id]): float(v) for s, v in zip(cands, scores, strict=True)}
    assert by_label[2] > by_label[1] > by_label[0]


def test_prodlr_live_shape_matches_the_live_blend() -> None:
    """prodlr with the live weights ranks exactly like production with the
    live blend on at full ramp (a heavy voter), and its production part
    carries no blend of its own."""
    from dataclasses import replace

    from scripts.eval_ranker_variants import _production_scores, _scores_prodlr

    fold = _signal_fold()
    base = Config()
    live = replace(
        base,
        model=replace(
            base.model,
            linear_blend_enabled=True,
            linear_blend_dense_weight=0.2,
            linear_blend_tfidf_weight=0.3,
            linear_blend_ramp=False,
        ),
    )
    expected = _production_scores(fold, live)[0]
    got = _scores_prodlr(
        fold,
        live,
        base,
        None,
        {"lr_weight": "0.2", "tfidf_weight": "0.3", "tfidf_c": "4.0"},
        None,
    )
    assert np.array_equal(
        np.argsort(-expected, kind="stable"), np.argsort(-got, kind="stable")
    )
    np.testing.assert_allclose(got, expected, atol=1e-5)
