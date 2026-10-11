"""Synthetic paired-evaluation gates: no database feedback, model or network."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import pytest

EVALUATION = Path(__file__).parents[1] / "docs/evaluations/decision1-20261010"


@dataclass
class SyntheticCard:
    id: int
    title: str
    domain: str = "example.test"
    source: str = "hn"
    text: str = "Synthetic article excerpt"


@dataclass
class SyntheticFold:
    fold: int
    candidates: list[SyntheticCard]
    labels: list[int]
    production: list[float]
    profile_up: list[str]
    profile_down: list[str]
    profile_ids: list[int]
    train_counts: dict[str, int]


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EVALUATION))
    for name in ("score_cache", "synthetic_decision1_paired_eval"):
        filename = "score_cache.py" if name == "score_cache" else "eval_decision1.py"
        spec = importlib.util.spec_from_file_location(name, EVALUATION / filename)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        spec.loader.exec_module(loaded)
    monkeypatch.setattr(loaded, "BACKEND", "strands")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("synthetic evaluation must not invoke a model or network")

    monkeypatch.setattr(loaded, "_http", forbidden)
    monkeypatch.setattr(loaded.subprocess, "run", forbidden)
    return loaded


@pytest.fixture
def folds() -> list[SyntheticFold]:
    return [
        SyntheticFold(
            fold=number,
            candidates=[
                SyntheticCard(number * 10 + i, f"Synthetic story {number}:{i}")
                for i in range(1, 6)
            ],
            labels=[2, 1, 0, -1, -1],
            production=[0.6, 0.4, 0.2, 0.9, 0.1],
            profile_up=[f"Synthetic preferred training story {number}"],
            profile_down=[f"Synthetic disliked training story {number}"],
            profile_ids=[100 + number],
            train_counts={"up": 1, "neutral": 0, "down": 1},
        )
        for number in range(2)
    ]


def write_folds(work: Path, folds: list[SyntheticFold]) -> None:
    (work / "folds.json").write_text(json.dumps([asdict(fold) for fold in folds]))


def populate_scores(module: ModuleType, work: Path) -> list[Path]:
    cache = sys.modules["score_cache"]
    identity = module.strands_identity()
    paths = []
    for fold in module.checked_folds(work, None):
        for card, label in zip(fold.candidates, fold.labels, strict=True):
            for variant in ("real", "empty"):
                real = {2: 0.9, 1: 0.3, 0: 0.1, -1: 0.99}[label]
                score = real if variant == "real" else 1 - real
                path = module.strict_cache_path(work, fold, card, variant, identity)
                cache.write_cached_score(
                    path,
                    score,
                    module.profile_state(fold, card, variant),
                    module.QUESTION,
                    identity,
                )
                paths.append(path)
    return paths


def make_snapshot(
    path: Path, folds: list[SyntheticFold], issue: str | None = None
) -> Path:
    records = []
    for fold in folds:
        for card, label in zip(fold.candidates, fold.labels, strict=True):
            points, comments = (1.0, 100.0) if label == 2 else (100.0, 1.0)
            if card.id == folds[0].candidates[0].id:
                if issue == "missing_story":
                    continue
                if issue == "null_points":
                    points = None
                elif issue == "null_comments":
                    comments = None
                elif issue == "nonfinite_points":
                    points = float("inf")
                elif issue == "nonfinite_comments":
                    comments = float("inf")
            records.append((card.id, points, comments))
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE stories(id INTEGER PRIMARY KEY, score REAL, comment_count REAL)"
        )
        connection.executemany("INSERT INTO stories VALUES (?, ?, ?)", records)
        connection.commit()
    return path


@pytest.mark.parametrize("cache_kind", ["missing", "legacy", "strict"])
def test_prepare_keeps_complete_paired_scope_with_existing_scores(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
    cache_kind: str,
) -> None:
    write_folds(tmp_path, folds)
    if cache_kind == "legacy":
        old = tmp_path / "cache_strands"
        old.mkdir()
        for fold in folds:
            for card in fold.candidates:
                (old / f"{fold.fold}_{card.id}.json").write_text('{"p_yes": 0.99}')
    elif cache_kind == "strict":
        populate_scores(module, tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    path = module.prepare_strands_jobs(tmp_path)
    jobs = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {
        (fold.fold, card.id, variant)
        for fold in folds
        for card in fold.candidates
        for variant in ("real", "empty")
    }
    actual = [(job["fold"], job["story_id"], job["variant"]) for job in jobs]
    assert len(actual) == len(expected) == 20
    assert set(actual) == expected
    assert Counter(job["variant"] for job in jobs) == {"real": 10, "empty": 10}
    assert len({job["out"] for job in jobs}) == 20
    assert module.prepare_strands_jobs(tmp_path) == path
    assert all(saved == filename.read_bytes() for filename, saved in before.items())
    for job in jobs:
        profile = job["state"]["reader_profile"]
        assert bool(profile["recently_upvoted"]) is (job["variant"] == "real")
        assert bool(profile["recently_downvoted"]) is (job["variant"] == "real")


def test_prepare_sample_and_fold_scope_preserve_all_judged_pairs(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    write_folds(tmp_path, folds)
    path = module.prepare_strands_jobs(tmp_path, sample=1, seed=7, folds_arg={1})
    jobs = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(jobs) == 8
    assert {job["fold"] for job in jobs} == {1}
    ids = {job["story_id"] for job in jobs}
    assert {card.id for card in folds[1].candidates[:3]} <= ids
    assert len(ids) == 4
    assert (
        module.prepare_strands_jobs(tmp_path, 1, 7, {1}).read_bytes()
        == path.read_bytes()
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_real",
        "missing_empty",
        "stale_identity",
        "stale_request",
        "legacy_record",
        "malformed",
    ],
)
def test_incomplete_or_stale_pairs_block_report_before_any_metrics(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    write_folds(tmp_path, folds)
    paths = populate_scores(module, tmp_path)
    path = paths[1] if damage == "missing_empty" else paths[0]
    if damage.startswith("missing"):
        path.unlink()
    elif damage == "malformed":
        path.write_text("not JSON")
    else:
        record = json.loads(path.read_text())
        if damage == "stale_identity":
            record["cache_identity"]["dtype"] = "stale-dtype"
        elif damage == "stale_request":
            record["request_sha256"] = "0" * 64
        else:
            record = {"p_yes": 0.4}
        path.write_text(json.dumps(record))
    audit = module.audit_strands_cache(tmp_path, None, 0, None)
    assert audit["complete"] is False
    assert audit["metrics_computed"] is False
    assert sum(row["valid"] for row in audit["coverage"]) == 19
    key = "missing" if damage.startswith("missing") else "invalid"
    assert sum(row[key] for row in audit["coverage"]) == 1
    previous_report = tmp_path / "report_strands_v2.json"
    if damage == "stale_request":
        previous_report.write_text('{"previous": "must remain unchanged"}')

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("metrics must not run before complete provenance audit")

    monkeypatch.setattr(module, "metrics", forbidden)
    monkeypatch.setattr(module, "_story_bootstrap_auc", forbidden)
    with pytest.raises(ValueError, match="paired score coverage"):
        module.cmd_report(tmp_path, None, 0, None)
    if damage == "stale_request":
        assert previous_report.read_text() == '{"previous": "must remain unchanged"}'
    else:
        assert not previous_report.exists()


def test_old_cache_does_not_count_as_valid_pair(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    write_folds(tmp_path, folds)
    old = tmp_path / "cache_strands"
    old.mkdir()
    for fold in folds:
        for card in fold.candidates:
            (old / f"{fold.fold}_{card.id}.json").write_text('{"p_yes": 0.99}')
    audit = module.audit_strands_cache(tmp_path, None, 0, None)
    assert audit["complete"] is False
    assert sum(row["missing"] for row in audit["coverage"]) == 20
    assert sum(row["legacy_present_not_reused"] for row in audit["coverage"]) == 10
    assert sum(row["valid"] for row in audit["coverage"]) == 0


def test_profile_change_invalidates_real_scores_without_reusing_old_scope(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    write_folds(tmp_path, folds)
    paths = populate_scores(module, tmp_path)
    saved = {path: path.read_bytes() for path in paths}
    folds[0].profile_up[0] = "Different synthetic training preference"
    write_folds(tmp_path, folds)
    audit = module.audit_strands_cache(tmp_path, None, 0, None)
    assert audit["complete"] is False
    first = [row for row in audit["coverage"] if row["fold"] == 0]
    assert {row["profile"]: row["missing"] for row in first} == {"real": 5, "empty": 0}
    assert sum(row["valid"] for row in audit["coverage"]) == 15
    assert all(path.read_bytes() == contents for path, contents in saved.items())


def test_report_requires_strands_backend_before_metrics(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_folds(tmp_path, folds)
    populate_scores(module, tmp_path)
    monkeypatch.setattr(module, "BACKEND", "decision1")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy backend cannot produce paired metrics")

    monkeypatch.setattr(module, "metrics", forbidden)
    with pytest.raises(ValueError, match="Strands"):
        module.cmd_report(tmp_path, None, 0, None)
    assert not (tmp_path / "report_strands_v2.json").exists()


@pytest.mark.parametrize("missing_path", [False, True])
def test_report_requires_existing_snapshot_before_metrics(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
    monkeypatch: pytest.MonkeyPatch,
    missing_path: bool,
) -> None:
    write_folds(tmp_path, folds)
    populate_scores(module, tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing popularity controls cannot reach metrics")

    monkeypatch.setattr(module, "metrics", forbidden)
    snapshot = tmp_path / "absent-synthetic-snapshot.db" if missing_path else None
    with pytest.raises(ValueError, match="snapshot"):
        module.cmd_report(tmp_path, None, 0, None, snapshot=snapshot)
    assert not (tmp_path / "report_strands_v2.json").exists()


@pytest.mark.parametrize(
    "issue",
    [
        "missing_story",
        "null_points",
        "null_comments",
        "nonfinite_points",
        "nonfinite_comments",
    ],
)
def test_report_rejects_incomplete_snapshot_values_before_metrics(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
    monkeypatch: pytest.MonkeyPatch,
    issue: str,
) -> None:
    write_folds(tmp_path, folds)
    populate_scores(module, tmp_path)
    snapshot = make_snapshot(tmp_path / "synthetic-snapshot.db", folds, issue)
    original = snapshot.read_bytes()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid popularity controls cannot reach metrics")

    monkeypatch.setattr(module, "metrics", forbidden)
    with pytest.raises(ValueError, match="popularity"):
        module.cmd_report(tmp_path, None, 0, None, snapshot=snapshot)
    assert snapshot.read_bytes() == original
    assert not (tmp_path / "report_strands_v2.json").exists()


def test_complete_report_is_fixed_weight_and_real_versus_empty_paired(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    write_folds(tmp_path, folds)
    paths = populate_scores(module, tmp_path)
    snapshot = make_snapshot(tmp_path / "synthetic-snapshot.db", folds)
    original = {
        path: path.read_bytes() for path in [snapshot, *paths, tmp_path / "folds.json"]
    }
    audit = module.audit_strands_cache(tmp_path, None, 23, None)
    assert audit["complete"] is True
    module.cmd_report(tmp_path, None, 23, None, snapshot=snapshot)
    report = json.loads((tmp_path / "report_strands_v2.json").read_text())
    assert report["primary_blend_weight"] == 0.2
    assert "weight_selection" not in report
    assert set(report["summary"]) == {
        "production",
        "strands_real",
        "strands_empty",
        "blend_real_w0.2",
        "blend_empty_w0.2",
        "snapshot_points",
        "snapshot_comments",
    }
    primary = report["primary_real_minus_empty_auc"]
    assert primary["diff"] == pytest.approx(1.0)
    assert primary["ci95"][0] <= primary["diff"] <= primary["ci95"][1]
    assert report["summary"]["strands_real"]["auc_up_vs_rest"]["mean"] == 1
    assert report["summary"]["strands_empty"]["auc_up_vs_rest"]["mean"] == 0
    assert report["summary"]["snapshot_points"]["auc_up_vs_rest"]["mean"] == 0
    assert report["summary"]["snapshot_comments"]["auc_up_vs_rest"]["mean"] == 1
    assert report["coverage_complete"] is True
    assert report["confirmation_used"] is False
    assert (
        report["folds_sha256"]
        == hashlib.sha256((tmp_path / "folds.json").read_bytes()).hexdigest()
    )
    assert [fold["scored_candidates"] for fold in report["folds"]] == [5, 5]
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert not (tmp_path / "report_strands.json").exists()


def test_report_undefined_primary_auc_stays_unknown_instead_of_zero(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    for fold in folds:
        fold.labels = [2] * len(fold.labels)
    write_folds(tmp_path, folds)
    populate_scores(module, tmp_path)
    snapshot = make_snapshot(tmp_path / "synthetic-snapshot.db", folds)
    module.cmd_report(tmp_path, None, 0, None, snapshot=snapshot)
    report = json.loads((tmp_path / "report_strands_v2.json").read_text())
    assert report["primary_real_minus_empty_auc"] is None
    assert report["summary"]["strands_real"]["auc_up_vs_rest"]["mean"] is None
    assert (
        report["summary"]["strands_real"]["pooled_auc_up_vs_rest_diff_story_bootstrap"]
        is None
    )


@pytest.mark.parametrize(
    "issue",
    [
        "candidate_ids",
        "fold_numbers",
        "profile_leakage",
        "repeated_judged",
        "shape",
        "nonfinite",
        "invalid_label",
        "empty_profile",
    ],
)
def test_checked_folds_rejects_invalid_or_leaky_manifest(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
    issue: str,
) -> None:
    if issue == "candidate_ids":
        folds[0].candidates[1].id = folds[0].candidates[0].id
    elif issue == "fold_numbers":
        folds[1].fold = folds[0].fold
    elif issue == "profile_leakage":
        folds[0].profile_ids.append(folds[0].candidates[0].id)
    elif issue == "repeated_judged":
        folds[1].candidates[0].id = folds[0].candidates[0].id
    elif issue == "shape":
        folds[0].labels.pop()
    elif issue == "nonfinite":
        folds[0].production[0] = float("nan")
    elif issue == "invalid_label":
        folds[0].labels[0] = 99
    else:
        folds[0].profile_up.clear()
        folds[0].profile_down.clear()
    write_folds(tmp_path, folds)
    with pytest.raises(ValueError):
        module.checked_folds(tmp_path, None)


def test_checked_folds_allows_shared_unjudged_but_not_absent_fold(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    folds[1].candidates[-1].id = folds[0].candidates[-1].id
    write_folds(tmp_path, folds)
    assert len(module.checked_folds(tmp_path, None)) == 2
    with pytest.raises(ValueError, match="absent"):
        module.checked_folds(tmp_path, {9})
    with pytest.raises(ValueError, match="no evaluation folds"):
        module.checked_folds(tmp_path, set())


def test_prepare_rejects_negative_sampling(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    write_folds(tmp_path, folds)
    with pytest.raises(ValueError, match="negative"):
        module.prepare_strands_jobs(tmp_path, sample=-1)


def test_cli_audit_refuses_incomplete_pairs_without_metrics(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_folds(tmp_path, folds)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("CLI audit cannot compute efficacy metrics")

    monkeypatch.setattr(module, "metrics", forbidden)
    with pytest.raises(SystemExit, match="incomplete paired coverage"):
        module.main(["audit", "--work", str(tmp_path), "--backend", "strands"])
    output = json.loads(capsys.readouterr().out)
    assert output["complete"] is False
    assert output["metrics_computed"] is False
    assert not (tmp_path / "report_strands_v2.json").exists()


def test_cli_prepare_applies_options_after_phase(
    module: ModuleType,
    tmp_path: Path,
    folds: list[SyntheticFold],
) -> None:
    write_folds(tmp_path, folds)
    module.main(
        [
            "prepare",
            "--work",
            str(tmp_path),
            "--backend",
            "strands",
            "--sample-per-fold",
            "0",
            "--folds",
            "1",
        ]
    )
    paths = list((tmp_path / "paired_plans").glob("*/jobs.jsonl"))
    assert len(paths) == 1
    jobs = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(jobs) == 6
    assert {job["fold"] for job in jobs} == {1}
    assert {job["story_id"] for job in jobs} == {
        card.id for card in folds[1].candidates[:3]
    }
