"""Baseline manifest for the production ONNX embedding model directory.

The manifest records which exact bytes the model directory held when it was
last provisioned (or first seen), so a later swap shows up as an explicit
mismatch instead of silently invalidating every stored vector. It is purely
additive: existing embedding rows keep matching regardless of manifest
state (see WORKLOG 2026-09-07 — no re-embed rotation).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MANIFEST_FILENAME = "model_manifest.json"

# Files covered by the manifest. Pico CSS is deliberately excluded: it is a
# frontend asset, not model bytes, and lives per-checkout.
MANIFESTED_FILES = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


@dataclass(frozen=True)
class ModelManifest:
    repo: str
    revision: str
    files: dict[str, str]
    created_at: float


def hash_model_files(model_dir: str | Path) -> dict[str, str]:
    """SHA-256 every manifested file; missing files hash as "<missing>"."""
    directory = Path(model_dir)
    digests: dict[str, str] = {}
    for name in MANIFESTED_FILES:
        path = directory / name
        if not path.is_file():
            digests[name] = "<missing>"
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digests[name] = digest.hexdigest()
    return digests


def manifest_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / MANIFEST_FILENAME


def write_manifest(model_dir: str | Path, repo: str, revision: str) -> ModelManifest:
    """Atomically (re)write the manifest from the directory's current bytes."""
    manifest = ModelManifest(
        repo=repo,
        revision=revision,
        files=hash_model_files(model_dir),
        created_at=time.time(),
    )
    dest = manifest_path(model_dir)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    tmp.replace(dest)
    return manifest


def read_manifest(model_dir: str | Path) -> ModelManifest | None:
    """Parse the manifest, or None when absent/unparseable (grandfathered)."""
    try:
        raw = json.loads(manifest_path(model_dir).read_text(encoding="utf-8"))
        return ModelManifest(
            repo=str(raw["repo"]),
            revision=str(raw["revision"]),
            files={str(k): str(v) for k, v in raw["files"].items()},
            created_at=float(raw["created_at"]),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def check_model_dir(
    model_dir: str | Path, repo: str, revision: str
) -> tuple[ModelManifest, list[str], bool]:
    """Compare the directory against its manifest.

    Returns (manifest, mismatches, baselined). When no usable manifest
    exists, the current bytes are recorded as the baseline (grandfathering:
    existing vectors stay valid) and mismatches is empty. A non-empty
    mismatches list means the bytes changed under a recorded baseline —
    the caller decides the policy (production warn-and-serves).

    Writing belongs to explicit operator actions (setup_model.py, tests):
    request-time code must use verify_model_dir below, which never writes.
    """
    stored = read_manifest(model_dir)
    if stored is None:
        return write_manifest(model_dir, repo, revision), [], True
    return stored, _diff_manifest(stored, model_dir, repo, revision), False


def verify_model_dir(
    model_dir: str | Path, repo: str, revision: str
) -> tuple[ModelManifest | None, list[str]]:
    """Read-only comparison: (manifest or None, mismatches). Never writes,
    so test doubles and request paths can't alter the recorded baseline."""
    stored = read_manifest(model_dir)
    if stored is None:
        return None, []
    return stored, _diff_manifest(stored, model_dir, repo, revision)


def _diff_manifest(
    stored: ModelManifest, model_dir: str | Path, repo: str, revision: str
) -> list[str]:
    current = hash_model_files(model_dir)
    mismatches: list[str] = []
    if stored.repo != repo or stored.revision != revision:
        mismatches.append(
            f"identity drift: manifest {stored.repo}@{stored.revision} "
            f"vs expected {repo}@{revision}"
        )
    for name in MANIFESTED_FILES:
        if current.get(name) != stored.files.get(name):
            mismatches.append(
                f"{name}: manifest {stored.files.get(name)} vs now {current.get(name)}"
            )
    return mismatches
