"""Behavioral checks for the canonical retrospective evaluator."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from database import Database, Story
from pipeline import Config, build_cold_deck, finalize_ranked_deck, rerank_candidates
from pipeline.hn_dupes import (
    _load_feedback_context,
    _matches_feedback,
    canonicalize_hn_dupes,
)
from pipeline.ranking import _loocv_knn_features, _score_and_rank
from scripts import eval_ranker_variants as evaluator

NOW = 2_000_000_000.0


def story(sid: int, *, age_days: int = 1, source: str = "hn") -> Story:
    return Story(
        id=sid,
        title=f"Distinct topic number {sid}",
        url=f"https://site{sid}.com/item",
        score=100 + sid,
        time=int(NOW - age_days * 86400),
        text_content=f"Text content {sid}",
        source=source,
        comment_count=sid % 40,
    )


def make_fold() -> evaluator.FoldData:
    stories = [
        story(
            i,
            age_days=60 if i % 3 == 0 else 1,
            source="rss_test" if i % 3 == 1 else "hn",
        )
        for i in range(1, 91)
    ]
    vectors = np.random.default_rng(17).normal(size=(90, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return evaluator._make_fold(
        stories[12:],
        vectors[12:],
        stories[:18],
        np.full(18, -1),
        np.arange(18, dtype=float) + NOW - 100,
        np.array([0, 1, 2] * 6),
        np.arange(18),
        np.arange(12),
        np.arange(12, 18),
        Config(),
        feedback_embeddings=vectors[:18],
    )


@pytest.mark.parametrize(
    "bad", [None, np.zeros(384), np.ones(10), np.full(384, np.nan), np.ones(384)]
)
def test_strict_embedding_coverage(bad: np.ndarray | None) -> None:
    cached = {} if bad is None else {1: bad}
    with pytest.raises(ValueError, match="0/1 valid"):
        evaluator._validated_embeddings([story(1)], cached)


def test_independent_training_and_singleton_self_exclusion() -> None:
    fold = make_fold()
    assert {s.id for s in fold.train_stories}.isdisjoint(s.id for s in fold.candidates)
    assert fold.train_emb is not None and fold.train_emb.shape == (12, 384)
    vectors = np.eye(3, 384, dtype=np.float32)
    mean, closest = _loocv_knn_features(vectors, vectors[:1], np.array([0]), 10)
    np.testing.assert_array_equal(mean, np.zeros(3))
    np.testing.assert_array_equal(closest, np.zeros(3))
    means = evaluator._loocv_similarity_features(vectors, np.array([2, 0, 1]), 10)
    assert all(np.isfinite(values).all() for values in means)


def test_timestamp_ties_and_cutoff_decay() -> None:
    times = np.repeat(np.arange(10, dtype=float), 3)
    splits = evaluator._temporal_splits(np.zeros(30), times, folds=3)
    for split in splits:
        assert times[split.train_pos].max() < times[split.test_pos].min()
        for timestamp in times[split.test_pos]:
            assert set(np.flatnonzero(times == timestamp)) <= set(split.test_pos)
    decay = evaluator._recency_decay(86400, np.array([0, 86400]), 1)
    np.testing.assert_allclose(decay, [0.5, 1.0])
    with pytest.raises(ValueError, match="timestamp groups"):
        evaluator._temporal_splits(np.zeros(5), np.ones(5), folds=2)


@pytest.mark.parametrize("precomputed", [False, True])
def test_production_scores_features_and_recommended_parity(precomputed: bool) -> None:
    fold = make_fold()
    config = replace(
        Config(),
        model=replace(
            Config().model,
            min_up_for_svm=2,
            min_down_for_svm=2,
            tier2_blend_window=1,
            tier3_blend_window=1,
            svm_precomputed_enabled=precomputed,
        ),
    )
    # Duplicate URLs exercise score-dependent survivor selection.
    fold.candidates[1] = replace(fold.candidates[1], url=fold.candidates[0].url)
    with (
        patch("time.time", return_value=NOW),
        evaluator._fold_database(fold, config, None) as db,
    ):
        embedder = evaluator._FrozenEmbedder(config.embedding_model_version)
        actual_scores, probs = evaluator._production_scores(fold, config)
        expected = _score_and_rank(fold.candidates, fold.cand_emb, db, config, embedder)
        by_id = {r.story.id: r for r in expected}
        np.testing.assert_array_equal(
            actual_scores, [by_id[s.id].score for s in fold.candidates]
        )
        assert probs is not None and probs.shape == (78, 3)
        context = _load_feedback_context(
            db, user_id=1, actions=config.model.dedup_exclude_actions
        )
        expected_deck = rerank_candidates(
            db,
            config,
            embedder,
            fold.candidates,
            fold.cand_emb,
            is_feedback_match=lambda s: _matches_feedback(s, context),
        )
        expected_deck = finalize_ranked_deck(
            expected_deck, fold.candidates, fold.cand_emb, db, config, embedder, 1
        )
        actual_deck = evaluator._recommended(actual_scores, fold, config, probs, None)
        assert actual_deck == expected_deck
        metrics = evaluator._metrics(actual_scores, fold, config, probs)
        for age in ("recent", "archive"):
            for source in ("mixed", "hn", "non-hn"):
                combo = f"{age}_{source}"
                assert metrics[f"recommended_{combo}"]["returned_cards"] == sum(
                    combo in r.combo_keys.split() for r in expected_deck
                )
        assert metrics["raw"]["brier_up"] is None  # softmax scores are not calibrated


def test_cold_deck_parity() -> None:
    fold = replace(
        make_fold(),
        train_stories=[],
        y_train=np.empty(0, dtype=int),
        train_vote_times=np.empty(0),
        train_emb=np.empty((0, 384)),
    )
    config = Config()
    with (
        patch("time.time", return_value=NOW),
        evaluator._fold_database(fold, config, None) as db,
    ):
        expected = canonicalize_hn_dupes(
            build_cold_deck(db, config, candidates=fold.candidates),
            db,
            selected_limit=config.count,
            user_id=1,
        )
        actual = evaluator._recommended(
            np.arange(78, dtype=float), fold, config, None, None
        )
        assert actual == expected
        assert not any(r.is_novel or r.is_uncertain or r.is_similar for r in actual)


def test_failed_production_fit_aborts() -> None:
    config = replace(
        Config(), model=replace(Config().model, min_up_for_svm=1, min_down_for_svm=1)
    )
    with patch("pipeline.ranking.SVC.fit", side_effect=RuntimeError("broken fold")):
        with (
            patch("time.time", return_value=NOW),
            pytest.raises(RuntimeError, match="Production fit failed"),
        ):
            evaluator._production_scores(make_fold(), config)


def test_null_metrics_unjudged_and_empty_results() -> None:
    fold = make_fold()
    with patch("time.time", return_value=NOW):
        metrics = evaluator._metrics(np.arange(78, dtype=float), fold, Config())["raw"]
    assert metrics["returned_cards"] == 78
    assert metrics["judged_cards"] == 6
    assert metrics["judged_coverage"] == 6 / 78
    assert metrics["known_neutral_cards"] == metrics["known_downvote_cards"] == 2
    empty = replace(fold, candidates=[], cand_emb=np.empty((0, 384)))
    metrics = evaluator._metrics(np.empty(0), empty, Config())["raw"]
    assert metrics["returned_cards"] == 0
    assert metrics["up_recall_at_40"] is None
    assert metrics["median_rank"] is None
    assert metrics["brier_up"] is None
    result = evaluator._aggregate_results(
        {"x": [{"raw": {"metric": None}}, {"raw": {"metric": 2}}]}
    )
    assert result["x"]["mean"]["raw"]["metric"] == 2
    assert result["x"]["defined_folds"]["raw"]["metric"] == 1


def fixture_database(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "input.db"
    db = Database(str(path))
    user = db.get_user_by_token("default")
    assert user is not None
    for sid in range(1, 61):
        item = story(sid, age_days=1000 if sid <= 3 else 1)
        db.upsert_story(item)
        vec = np.eye(1, 384, k=sid, dtype=np.float32)[0]
        db.upsert_embedding(
            sid,
            Config().embedding_model_version,
            hashlib.sha256(evaluator.story_embedding_text(item).encode()).hexdigest(),
            vec,
        )
        if sid <= 40:
            db.upsert_feedback(user.id, sid, ("down", "neutral", "up")[sid % 3])
    with db.conn() as conn:
        conn.execute(
            "UPDATE feedback SET updated_at = ? + CAST((story_id - 1) / 2 AS INT)",
            (NOW - 100,),
        )
        conn.commit()
    db.close()
    config = tmp_path / "config.toml"
    config.write_text(f'[hn_rewrite]\ndb_path = "{path}"\n')
    return path, config


def test_consistent_read_only_backup_includes_wal(tmp_path: Path) -> None:
    path, _ = fixture_database(tmp_path)
    before = path.read_bytes()
    with evaluator.frozen_database(str(path)) as (snapshot, digest):
        assert digest == evaluator._db_sha256(snapshot.db_path)
        with snapshot.conn() as conn:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("UPDATE stories SET score=0")
        assert snapshot.get_story(1) is not None
    assert path.read_bytes() == before


def test_entrypoint_reproducibility_and_confirmation(tmp_path: Path) -> None:
    path, config = fixture_database(tmp_path)
    before = path.read_bytes()
    outputs = []
    for _ in range(2):
        output = tmp_path / f"report{len(outputs)}.json"
        evaluator.main(
            [
                "--config",
                str(config),
                "--folds",
                "2",
                "--now",
                str(NOW),
                "--output",
                str(output),
            ]
        )
        outputs.append(json.loads(output.read_text()))
    assert outputs[0] == outputs[1]
    assert outputs[0]["config"]["n_feedback_valid"] == 32
    assert outputs[0]["config"]["confirmation_start"] == NOW - 84
    output = tmp_path / "confirmation.json"
    evaluator.main(
        [
            "--config",
            str(config),
            "--folds",
            "2",
            "--now",
            str(NOW),
            "--confirmation",
            "--output",
            str(output),
        ]
    )
    confirmed = json.loads(output.read_text())
    assert confirmed["config"]["confirmation"] is True
    assert confirmed["config"]["n_feedback_valid"] == 40
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "args",
    [
        ["--split", "stratified"],
        ["--candidate-cap-seed", "5"],
        ["--exclude-sources", "hn"],
        ["--k-values", "500"],
        ["--max-candidates", "0"],
    ],
)
def test_unsupported_cli_semantics_fail(args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        evaluator.main(args)
    assert error.value.code == 2


def test_cached_scaling_preserves_values_and_training_only_fit() -> None:
    fold = make_fold()
    train, cand = fold.x_train_base, fold.x_cand_base.copy()
    expected = evaluator._fit_scale(train, cand, 384)
    with evaluator._reuse_preprocessing():
        first = evaluator._fit_scale(train, cand, 384)
        second = evaluator._fit_scale(train, cand, 384)
        assert first is second
        np.testing.assert_array_equal(first[0], expected[0])
        np.testing.assert_array_equal(first[1], expected[1])
    cand[:, 384:] += 1000
    changed = evaluator._fit_scale(train, cand, 384)
    np.testing.assert_array_equal(changed[0], expected[0])
    np.testing.assert_array_equal(changed[1][:, :384], cand[:, :384])


def test_bakeoff_captures_independent_feedback_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import bakeoff_embedding_models as bakeoff

    path, config = fixture_database(tmp_path)
    before = path.read_bytes()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test fixture model")
    monkeypatch.setattr(bakeoff, "_model_paths", lambda spec: ("unused", model))

    def encode(texts: list[str], **kwargs: object) -> tuple[np.ndarray, float]:
        vectors = (
            np.random.default_rng(33).normal(size=(len(texts), 384)).astype(np.float32)
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors, 0.1

    monkeypatch.setattr(bakeoff, "_encode", encode)
    output_dir = tmp_path / "snapshots"
    monkeypatch.setattr(
        "sys.argv",
        [
            "bakeoff",
            "--config",
            str(config),
            "--models",
            "minilm",
            "--output-dir",
            str(output_dir),
        ],
    )
    with patch("time.time", return_value=NOW):
        bakeoff.main()
    snapshot = output_dir / "minilm-tokens256.npz"
    with np.load(snapshot, allow_pickle=False) as data:
        assert len(data["feedback_embeddings"]) == 40
        assert 1 not in data["story_ids"]  # outside the candidate window, still trained
        assert 1 in data["feedback_story_ids"]
    report_path = tmp_path / "snapshot-report.json"
    evaluator.main(
        [
            "--embeddings-file",
            str(snapshot),
            "--folds",
            "2",
            "--output",
            str(report_path),
        ]
    )
    first = json.loads(report_path.read_text())
    # Changing the live source cannot affect the snapshot's canonicalization context.
    writable = Database(str(path))
    writable.upsert_story(replace(story(55), title="Changed live metadata"))
    writable.close()
    evaluator.main(
        [
            "--embeddings-file",
            str(snapshot),
            "--folds",
            "2",
            "--output",
            str(report_path),
        ]
    )
    assert json.loads(report_path.read_text()) == first
    assert before != path.read_bytes()
    # The second encoder uses exactly the first encoder's frozen rows and clock.
    monkeypatch.setattr(
        "sys.argv",
        [
            "bakeoff",
            "--models",
            "mxbai_xsmall",
            "--reference-snapshot",
            str(snapshot),
            "--output-dir",
            str(output_dir),
        ],
    )
    bakeoff.main()
    second = output_dir / "mxbai_xsmall-tokens256.npz"
    assert evaluator._snapshot_context(second) == evaluator._snapshot_context(snapshot)


def test_canonical_replacement_uses_snapshot_target(tmp_path: Path) -> None:
    from database import HnDupeResolution

    fold = make_fold()
    source = Database(str(tmp_path / "canonical.db"))
    target = story(1001)
    source.upsert_story(target)
    source_id = next(
        s.id for s in fold.candidates if s.source == "hn" and s.time > NOW - 86400 * 30
    )
    source.upsert_story(next(s for s in fold.candidates if s.id == source_id))
    source.upsert_hn_dupe_resolution(
        HnDupeResolution(source_id, target.id, "canonical", NOW, NOW + 3600, 0, "")
    )
    scores = np.zeros(len(fold.candidates))
    scores[[s.id for s in fold.candidates].index(source_id)] = 10
    with patch("time.time", return_value=NOW):
        deck = evaluator._recommended(scores, fold, Config(), None, source)
    assert target.id in {r.story.id for r in deck}
    assert source_id not in {r.story.id for r in deck}
    source.close()


def test_confirmation_groups_survive_feedback_sampling(tmp_path: Path) -> None:
    _, config = fixture_database(tmp_path)
    report = tmp_path / "sampled.json"
    evaluator.main(
        [
            "--config",
            str(config),
            "--folds",
            "2",
            "--max-feedback-per-class",
            "8",
            "--now",
            str(NOW),
            "--output",
            str(report),
        ]
    )
    result = json.loads(report.read_text())
    assert result["config"]["confirmation_start"] == NOW - 84
    assert all(fold["test_end"] < NOW - 84 for fold in result["config"]["folds"])


def test_empty_candidate_fold_and_unavailable_positive_ranks() -> None:
    vectors = np.eye(2, 384, dtype=np.float32)
    fold = evaluator._make_fold(
        [],
        np.empty((0, 384)),
        [story(1), story(2)],
        np.array([-1, -1]),
        np.array([1, 2]),
        np.array([0, 2]),
        np.array([0, 1]),
        np.array([0]),
        np.array([1]),
        Config(),
        feedback_embeddings=vectors,
    )
    metrics = evaluator._metrics(np.empty(0), fold, Config())
    assert all(
        side["median_rank"] is None and side["returned_cards"] == 0
        for side in metrics.values()
    )
    assert metrics["raw"]["excluded_positives"] == 1


def test_brier_only_when_probability_estimate_is_available() -> None:
    fold = make_fold()
    probabilities = np.tile([0.2, 0.3, 0.5], (len(fold.candidates), 1))
    with patch("time.time", return_value=NOW):
        metrics = evaluator._metrics(
            np.arange(len(fold.candidates), dtype=float),
            fold,
            Config(),
            probabilities,
            calibration_available=True,
        )
    assert metrics["raw"]["brier_up"] == 0.25


def test_snapshot_is_consistent_while_source_wal_changes(tmp_path: Path) -> None:
    path, _ = fixture_database(tmp_path)
    source = Database(str(path))
    source.upsert_story(story(999))
    with evaluator.frozen_database(str(path)) as (snapshot, _):
        assert snapshot.get_story(999) == source.get_story(999)
        source.upsert_story(replace(story(999), title="Newer committed WAL row"))
        assert snapshot.get_story(999) != source.get_story(999)
    source.close()


def test_production_scores_are_deterministic() -> None:
    fold = make_fold()
    config = replace(
        Config(), model=replace(Config().model, min_up_for_svm=2, min_down_for_svm=2)
    )
    tuned = replace(config, model=replace(config.model, svm_c=0.7))
    with patch("time.time", return_value=NOW):
        first, _ = evaluator._production_scores(fold, tuned)
        second, _ = evaluator._production_scores(fold, tuned)
    np.testing.assert_array_equal(first, second)


def test_failed_production_probability_mapping_aborts() -> None:
    config = replace(
        Config(), model=replace(Config().model, min_up_for_svm=2, min_down_for_svm=2)
    )
    with (
        patch("time.time", return_value=NOW),
        patch(
            "pipeline.ranking._softmax_rows",
            side_effect=lambda values: np.zeros((len(values), 2)),
        ),
    ):
        with pytest.raises(RuntimeError, match="probability mapping failed"):
            evaluator._production_scores(make_fold(), config)
