"""Content-bound, versioned offline score caches; no model or network dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_SCHEMA = 1
_RESERVED = frozenset({"cache_schema", "cache_identity", "request_sha256", "p_yes"})


class CacheValidationError(ValueError):
    """An input or cache cannot prove an exact, valid score request."""


@dataclass(frozen=True)
class CacheIdentity:
    backend: str
    model_id: str
    model_revision: str
    dtype: str
    scorer_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise CacheValidationError(
                    f"cache identity {name} must be a nonempty string"
                )
        if re.fullmatch(r"[0-9a-f]{64}", self.scorer_sha256) is None:
            raise CacheValidationError(
                "scorer_sha256 must be a lowercase SHA256 digest"
            )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise CacheValidationError("cache input must be finite JSON data") from exc


def request_sha256(state: object, question: object, identity: CacheIdentity) -> str:
    """Bind complete prompt inputs and the exact scorer/model configuration."""
    payload = {
        "cache_schema": CACHE_SCHEMA,
        "cache_identity": asdict(identity),
        "state": state,
        "question": question,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def cache_path(
    directory: Path, state: object, question: object, identity: CacheIdentity
) -> Path:
    return directory / f"{request_sha256(state, question, identity)}.json"


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CacheValidationError("p_yes must be a finite numeric probability")
    try:
        probability = float(value)
    except OverflowError as exc:
        raise CacheValidationError(
            "p_yes must be a finite numeric probability"
        ) from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise CacheValidationError("p_yes must be finite and within [0,1]")
    return probability


def cache_record(
    p_yes: float,
    state: object,
    question: object,
    identity: CacheIdentity,
    *,
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    extra = dict(extra_fields or {})
    if _RESERVED.intersection(extra):
        raise CacheValidationError("extra fields cannot replace cache contract fields")
    record: dict[str, object] = {
        **extra,
        "cache_schema": CACHE_SCHEMA,
        "cache_identity": asdict(identity),
        "request_sha256": request_sha256(state, question, identity),
        "p_yes": _probability(p_yes),
    }
    _canonical_json(record)
    return record


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CacheValidationError(f"duplicate cache JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CacheValidationError(f"nonfinite cache JSON constant: {value}")


def read_cached_score(
    path: Path, state: object, question: object, identity: CacheIdentity
) -> float:
    try:
        record = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CacheValidationError(f"cannot read valid score cache: {path}") from exc
    if not isinstance(record, dict):
        raise CacheValidationError(f"cache must be a JSON object: {path}")
    if (
        type(record.get("cache_schema")) is not int
        or record["cache_schema"] != CACHE_SCHEMA
    ):
        raise CacheValidationError(f"missing/unsupported cache schema: {path}")
    if record.get("cache_identity") != asdict(identity):
        raise CacheValidationError(f"cache identity is missing or stale: {path}")
    if record.get("request_sha256") != request_sha256(state, question, identity):
        raise CacheValidationError(f"cache request digest is missing or stale: {path}")
    return _probability(record.get("p_yes"))


def write_cached_score(
    path: Path,
    p_yes: float,
    state: object,
    question: object,
    identity: CacheIdentity,
    *,
    extra_fields: Mapping[str, object] | None = None,
) -> None:
    """Publish a complete record atomically without replacing any existing entry.

    Hard-link publication is create-only: a concurrent writer cannot make this
    operation overwrite an existing cache. A valid existing entry is preserved;
    invalid/legacy/stale entries raise rather than being repaired or replaced.
    """
    payload = (
        _canonical_json(
            cache_record(p_yes, state, question, identity, extra_fields=extra_fields)
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".score-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            read_cached_score(path, state, question, identity)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
