"""Create-only provenance receipts for offline folds and snapshot bytes.

The receipt is ``work/frozen_inputs_v1.json``. Initial freezing requires both
inputs; a receipt is never upgraded or rewritten. Reusing a prepared plan may
validate folds alone, but metrics must validate the supplied snapshot as well.
This contract proves byte identity, not authenticity against manual receipt
replacement outside this API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

FROZEN_INPUTS_SCHEMA = 1
RECEIPT_FILENAME = "frozen_inputs_v1.json"
HASH_CHUNK_BYTES = 1024 * 1024
_RECEIPT_MAX_BYTES = 4096
_RECEIPT_FIELDS = frozenset({"schema", "folds_sha256", "snapshot_sha256"})


class FrozenInputsError(ValueError):
    """The files cannot prove the exact initially frozen input bytes."""


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FrozenInputsError(f"{field} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    schema: int
    folds_sha256: str
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != FROZEN_INPUTS_SCHEMA:
            raise FrozenInputsError("missing or unsupported frozen-input schema")
        _digest(self.folds_sha256, "folds_sha256")
        _digest(self.snapshot_sha256, "snapshot_sha256")


def _stamp(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def hash_file(path: Path) -> str:
    """Stream a regular file and reject modification or replacement during hashing."""
    digest = hashlib.sha256()
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise FrozenInputsError(f"frozen input is not a regular file: {path}")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _stamp(before) != _stamp(opened):
                raise FrozenInputsError(f"frozen input changed before hashing: {path}")
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        if _stamp(opened) != _stamp(after) or _stamp(after) != _stamp(path.stat()):
            raise FrozenInputsError(f"frozen input changed during hashing: {path}")
    except OSError as exc:
        raise FrozenInputsError(f"cannot hash frozen input: {path}") from exc
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FrozenInputsError(f"duplicate frozen-input receipt field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise FrozenInputsError(f"nonfinite frozen-input receipt value: {value}")


def read_frozen_inputs(work: Path) -> FrozenInputs:
    """Read a complete, strictly versioned receipt without writing or checking inputs."""
    path = work / RECEIPT_FILENAME
    try:
        if path.is_symlink():
            raise FrozenInputsError(f"frozen-input receipt cannot be a symlink: {path}")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > _RECEIPT_MAX_BYTES:
            raise FrozenInputsError(f"invalid frozen-input receipt file: {path}")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            content = handle.read(_RECEIPT_MAX_BYTES + 1)
            after = os.fstat(handle.fileno())
        if len(content) > _RECEIPT_MAX_BYTES:
            raise FrozenInputsError(f"oversized frozen-input receipt: {path}")
        # A concurrent publisher removing its temporary hardlink changes ctime
        # without changing the receipt bytes. Preserve inode/size/mtime checks.
        if not (
            _stamp(before)[:4]
            == _stamp(opened)[:4]
            == _stamp(after)[:4]
            == _stamp(path.stat())[:4]
        ):
            raise FrozenInputsError(
                f"frozen-input receipt changed while reading: {path}"
            )
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FrozenInputsError(f"cannot read frozen-input receipt: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != _RECEIPT_FIELDS:
        raise FrozenInputsError(
            "frozen-input receipt fields are missing or unsupported"
        )
    if type(raw["schema"]) is not int or raw["schema"] != FROZEN_INPUTS_SCHEMA:
        raise FrozenInputsError("missing or unsupported frozen-input schema")
    return FrozenInputs(
        schema=FROZEN_INPUTS_SCHEMA,
        folds_sha256=_digest(raw["folds_sha256"], "folds_sha256"),
        snapshot_sha256=_digest(raw["snapshot_sha256"], "snapshot_sha256"),
    )


def validate_frozen_inputs(
    work: Path,
    snapshot: Path | None = None,
    *,
    require_snapshot: bool = False,
) -> FrozenInputs:
    """Prove folds and, whenever supplied or required, snapshot against the receipt."""
    receipt = read_frozen_inputs(work)
    if hash_file(work / "folds.json") != receipt.folds_sha256:
        raise FrozenInputsError("folds.json differs from the initially frozen input")
    if snapshot is None:
        if require_snapshot:
            raise FrozenInputsError("snapshot is required to validate frozen inputs")
    elif hash_file(snapshot) != receipt.snapshot_sha256:
        raise FrozenInputsError("snapshot differs from the initially frozen input")
    return receipt


def freeze_inputs(work: Path, snapshot: Path) -> FrozenInputs:
    """Atomically publish one full receipt, preserving every existing receipt byte."""
    if not isinstance(snapshot, Path):
        raise FrozenInputsError("snapshot path is required for initial input freezing")
    target = work / RECEIPT_FILENAME
    if target.exists() or target.is_symlink():
        return validate_frozen_inputs(work, snapshot, require_snapshot=True)
    receipt = FrozenInputs(
        schema=FROZEN_INPUTS_SCHEMA,
        folds_sha256=hash_file(work / "folds.json"),
        snapshot_sha256=hash_file(snapshot),
    )
    # Snapshot hashing can take time; do not publish if folds changed meanwhile.
    if hash_file(work / "folds.json") != receipt.folds_sha256:
        raise FrozenInputsError("folds.json changed during initial input freezing")
    payload = json.dumps(asdict(receipt), sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=work,
            prefix=".frozen-inputs-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            # A concurrent publisher is accepted only for these exact inputs.
            return validate_frozen_inputs(work, snapshot, require_snapshot=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt
