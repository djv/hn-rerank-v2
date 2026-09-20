"""Unit tests for the embedding model manifest baseline (no ONNX needed)."""

from __future__ import annotations

from pathlib import Path

from pipeline.model_manifest import (
    MANIFESTED_FILES,
    check_model_dir,
    hash_model_files,
    read_manifest,
    verify_model_dir,
    write_manifest,
)


def _seed_model_dir(directory: Path, payload: bytes = b"model-bytes") -> None:
    for name in MANIFESTED_FILES:
        (directory / name).write_bytes(payload + name.encode())


def test_manifest_roundtrip(tmp_path: Path) -> None:
    _seed_model_dir(tmp_path)
    manifest = write_manifest(tmp_path, "repo/x", "rev1")
    assert manifest.files["model.onnx"] != "<missing>"
    reread = read_manifest(tmp_path)
    assert reread == manifest


def test_missing_manifest_creates_baseline_without_mismatches(
    tmp_path: Path,
) -> None:
    _seed_model_dir(tmp_path)
    manifest, mismatches, baselined = check_model_dir(tmp_path, "repo/x", "rev1")
    assert baselined is True
    assert mismatches == []
    assert read_manifest(tmp_path) == manifest


def test_tampered_file_reports_mismatch_and_serves_on(tmp_path: Path) -> None:
    _seed_model_dir(tmp_path)
    write_manifest(tmp_path, "repo/x", "rev1")
    (tmp_path / "model.onnx").write_bytes(b"tampered")
    stored, mismatches, baselined = check_model_dir(tmp_path, "repo/x", "rev1")
    assert baselined is False
    assert any(m.startswith("model.onnx:") for m in mismatches)
    # The stored baseline still describes the old bytes (evidence kept).
    assert stored.files["model.onnx"] != hash_model_files(tmp_path)["model.onnx"]


def test_corrupt_manifest_rebaselines(tmp_path: Path) -> None:
    _seed_model_dir(tmp_path)
    (tmp_path / "model_manifest.json").write_text("{not json", encoding="utf-8")
    manifest, mismatches, baselined = check_model_dir(tmp_path, "repo/x", "rev1")
    assert baselined is True
    assert mismatches == []
    assert manifest.files["model.onnx"] != "<missing>"


def test_identity_drift_reported(tmp_path: Path) -> None:
    _seed_model_dir(tmp_path)
    write_manifest(tmp_path, "repo/x", "rev1")
    _, mismatches, _ = check_model_dir(tmp_path, "repo/y", "rev2")
    assert any("identity drift" in m for m in mismatches)


def test_verify_model_dir_never_writes(tmp_path: Path) -> None:
    _seed_model_dir(tmp_path)
    stored, mismatches = verify_model_dir(tmp_path, "repo/x", "rev1")
    assert stored is None
    assert mismatches == []
    assert not (tmp_path / "model_manifest.json").exists()

    write_manifest(tmp_path, "repo/x", "rev1")
    stored, mismatches = verify_model_dir(tmp_path, "repo/x", "rev1")
    assert stored is not None
    assert mismatches == []
