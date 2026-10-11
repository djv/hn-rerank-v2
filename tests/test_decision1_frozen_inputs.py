"""Synthetic create-only provenance receipts; no real snapshot, scores or inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Barrier
from types import ModuleType

import pytest

EVALUATION = Path(__file__).parents[1] / "docs/evaluations/decision1-20261010"


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "synthetic_decision1_frozen_inputs", EVALUATION / "frozen_inputs.py"
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    folds = tmp_path / "folds.json"
    folds.write_bytes(b'[{"fold":0,"labels":[2,0]}]\n')
    snapshot = tmp_path / "synthetic-snapshot.db"
    snapshot.write_bytes(b"Synthetic frozen snapshot bytes, not a database")
    return folds, snapshot


def test_freeze_records_full_input_hashes_and_repeated_calls_preserve_receipt(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
) -> None:
    folds, snapshot = inputs
    original = {path: path.read_bytes() for path in inputs}
    receipt = module.freeze_inputs(tmp_path, snapshot)
    assert asdict(receipt) == {
        "schema": 1,
        "folds_sha256": hashlib.sha256(folds.read_bytes()).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    path = tmp_path / "frozen_inputs_v1.json"
    before = path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns
    assert module.read_frozen_inputs(tmp_path) == receipt
    assert module.validate_frozen_inputs(tmp_path) == receipt
    assert (
        module.validate_frozen_inputs(tmp_path, snapshot, require_snapshot=True)
        == receipt
    )
    assert module.freeze_inputs(tmp_path, snapshot) == receipt
    assert before == (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
    assert all(path.read_bytes() == saved for path, saved in original.items())
    assert not list(tmp_path.glob(".frozen-inputs-*"))


@pytest.mark.parametrize("target", ["folds", "snapshot"])
def test_changed_input_cannot_be_refrozen_and_preserves_existing_receipt(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    target: str,
) -> None:
    folds, snapshot = inputs
    module.freeze_inputs(tmp_path, snapshot)
    path = tmp_path / module.RECEIPT_FILENAME
    receipt_bytes = path.read_bytes()
    changed = folds if target == "folds" else snapshot
    changed.write_bytes(changed.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="differs"):
        module.validate_frozen_inputs(tmp_path, snapshot, require_snapshot=True)
    with pytest.raises(ValueError, match="differs"):
        module.freeze_inputs(tmp_path, snapshot)
    assert path.read_bytes() == receipt_bytes


def test_snapshot_is_required_for_metrics_and_optional_only_for_fold_validation(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
) -> None:
    _, snapshot = inputs
    receipt = module.freeze_inputs(tmp_path, snapshot)
    with pytest.raises(ValueError, match="snapshot is required"):
        module.validate_frozen_inputs(tmp_path, require_snapshot=True)
    snapshot.unlink()
    assert module.validate_frozen_inputs(tmp_path, require_snapshot=False) == receipt
    with pytest.raises(ValueError, match="cannot hash"):
        module.validate_frozen_inputs(tmp_path, snapshot)


def test_validation_refuses_missing_receipt_without_creating_any_files(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
) -> None:
    _, snapshot = inputs
    with pytest.raises(ValueError, match="cannot read frozen-input receipt"):
        module.validate_frozen_inputs(tmp_path, snapshot, require_snapshot=True)
    assert not (tmp_path / module.RECEIPT_FILENAME).exists()


@pytest.mark.parametrize("target", ["folds", "snapshot", "snapshot_directory"])
def test_initial_freeze_requires_two_readable_regular_inputs(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    target: str,
) -> None:
    folds, snapshot = inputs
    if target == "folds":
        folds.unlink()
    else:
        snapshot.unlink()
        if target == "snapshot_directory":
            snapshot.mkdir()
    with pytest.raises(ValueError):
        module.freeze_inputs(tmp_path, snapshot)
    assert not (tmp_path / module.RECEIPT_FILENAME).exists()
    assert not list(tmp_path.glob(".frozen-inputs-*"))


@pytest.mark.parametrize(
    "record",
    [
        [],
        {},
        {"schema": 1, "folds_sha256": "0" * 64},
        {"schema": True, "folds_sha256": "0" * 64, "snapshot_sha256": "1" * 64},
        {"schema": 2, "folds_sha256": "0" * 64, "snapshot_sha256": "1" * 64},
        {"schema": 1, "folds_sha256": "0" * 40, "snapshot_sha256": "1" * 64},
        {"schema": 1, "folds_sha256": "A" * 64, "snapshot_sha256": "1" * 64},
        {"schema": 1, "folds_sha256": "0" * 64, "snapshot_sha256": None},
        {"schema": 1, "folds_sha256": "0" * 64, "snapshot_sha256": "not-a-digest"},
        {
            "schema": 1,
            "folds_sha256": "0" * 64,
            "snapshot_sha256": "1" * 64,
            "extra": 1,
        },
    ],
)
def test_invalid_receipt_is_never_repaired_or_replaced(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    record: object,
) -> None:
    _, snapshot = inputs
    path = tmp_path / module.RECEIPT_FILENAME
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        module.read_frozen_inputs(tmp_path)
    with pytest.raises(ValueError):
        module.freeze_inputs(tmp_path, snapshot)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        b"not JSON",
        b"\xff",
        b'{"schema":1,"schema":1,"folds_sha256":"x","snapshot_sha256":"y"}',
        b'{"schema":NaN,"folds_sha256":"x","snapshot_sha256":"y"}',
        b'{"schema":Infinity,"folds_sha256":"x","snapshot_sha256":"y"}',
        b" " * 4097,
    ],
)
def test_malformed_nonfinite_or_duplicate_receipt_is_preserved(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    content: bytes,
) -> None:
    _, snapshot = inputs
    path = tmp_path / module.RECEIPT_FILENAME
    path.write_bytes(content)
    with pytest.raises(ValueError):
        module.validate_frozen_inputs(tmp_path, snapshot, require_snapshot=True)
    with pytest.raises(ValueError):
        module.freeze_inputs(tmp_path, snapshot)
    assert path.read_bytes() == content


def test_receipt_symlink_is_rejected_and_preserved(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
) -> None:
    _, snapshot = inputs
    other = tmp_path / "other-receipt.json"
    other.write_text("{}")
    path = tmp_path / module.RECEIPT_FILENAME
    path.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        module.freeze_inputs(tmp_path, snapshot)
    assert path.is_symlink()
    assert other.read_text() == "{}"


def test_hash_streams_large_input_without_read_bytes(
    module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic-block" * 200_000
    path = tmp_path / "large-synthetic-input"
    path.write_bytes(payload)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("input hashing must not read the whole snapshot at once")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert module.hash_file(path) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("replacement", [False, True])
def test_hash_rejects_modification_or_replacement_during_streaming(
    module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: bool,
) -> None:
    path = tmp_path / "synthetic-input"
    path.write_bytes(b"before")
    sha256 = module.hashlib.sha256

    class MutatingDigest:
        def __init__(self) -> None:
            self.real = sha256()

        def update(self, chunk: bytes) -> None:
            self.real.update(chunk)
            if replacement:
                other = tmp_path / "replacement"
                other.write_bytes(b"after")
                other.replace(path)
            else:
                path.write_bytes(b"after")

        def hexdigest(self) -> str:
            return self.real.hexdigest()

    monkeypatch.setattr(module.hashlib, "sha256", MutatingDigest)
    with pytest.raises(ValueError, match="changed"):
        module.hash_file(path)


def test_fold_change_during_snapshot_hash_prevents_initial_receipt(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds, snapshot = inputs
    original = module.hash_file

    def mutate_fold(path: Path) -> str:
        result = original(path)
        if path == snapshot:
            folds.write_bytes(b"different fold labels")
        return result

    monkeypatch.setattr(module, "hash_file", mutate_fold)
    with pytest.raises(ValueError, match="folds.json changed"):
        module.freeze_inputs(tmp_path, snapshot)
    assert not (tmp_path / module.RECEIPT_FILENAME).exists()


def test_atomic_publication_failure_leaves_no_receipt_or_temporary_files(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, snapshot = inputs

    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="publication failure"):
        module.freeze_inputs(tmp_path, snapshot)
    assert not (tmp_path / module.RECEIPT_FILENAME).exists()
    assert not list(tmp_path.glob(".frozen-inputs-*"))


def test_concurrent_identical_freezes_publish_one_complete_receipt(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, snapshot = inputs
    barrier = Barrier(2)
    original = os.link

    def race_link(source: Path, destination: Path) -> None:
        barrier.wait(timeout=5)
        original(source, destination)

    monkeypatch.setattr(module.os, "link", race_link)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(module.freeze_inputs, tmp_path, snapshot) for _ in range(2)
        ]
        first, second = [future.result(timeout=10) for future in futures]
    assert first == second == module.read_frozen_inputs(tmp_path)
    assert not list(tmp_path.glob(".frozen-inputs-*"))


def test_concurrent_different_snapshots_cannot_overwrite_winning_receipt(
    module: ModuleType,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, snapshot = inputs
    second = tmp_path / "different-synthetic-snapshot.db"
    second.write_bytes(b"different snapshot")
    barrier = Barrier(2)
    original = os.link

    def race_link(source: Path, destination: Path) -> None:
        barrier.wait(timeout=5)
        original(source, destination)

    monkeypatch.setattr(module.os, "link", race_link)
    outcomes = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(module.freeze_inputs, tmp_path, item)
            for item in (snapshot, second)
        ]
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except module.FrozenInputsError as exc:
                outcomes.append(exc)
    errors = [item for item in outcomes if isinstance(item, ValueError)]
    assert len(errors) == 1
    receipt = module.read_frozen_inputs(tmp_path)
    assert receipt in outcomes
    assert receipt.snapshot_sha256 in {
        module.hash_file(snapshot),
        module.hash_file(second),
    }
    assert not list(tmp_path.glob(".frozen-inputs-*"))
