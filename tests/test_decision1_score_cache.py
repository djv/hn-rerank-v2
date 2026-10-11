from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

SPEC = importlib.util.spec_from_file_location(
    "decision1_score_cache",
    Path(__file__).parents[1] / "docs/evaluations/decision1-20261010/score_cache.py",
)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache
SPEC.loader.exec_module(cache)

IDENTITY = cache.CacheIdentity("strands", "org/model", "a" * 40, "fp32", "b" * 64)
STATE = {"profile": ["synthetic liked item"], "story": {"title": "synthetic story"}}
QUESTION = {"upvote": {"instructions": "synthetic decision"}}


@given(
    backend=st.text(min_size=1).filter(
        lambda value: bool(value.strip()) and value != "strands"
    ),
    model=st.text(min_size=1).filter(
        lambda value: bool(value.strip()) and value != "org/model"
    ),
    revision=st.text(min_size=1).filter(
        lambda value: bool(value.strip()) and value != "a" * 40
    ),
    dtype=st.text(min_size=1).filter(
        lambda value: bool(value.strip()) and value != "fp32"
    ),
    scorer=st.text(alphabet="0123456789abcdef", min_size=64, max_size=64).filter(
        lambda value: value != "b" * 64
    ),
)
def test_every_identity_field_changes_request_digest(
    backend: str, model: str, revision: str, dtype: str, scorer: str
) -> None:
    original = cache.request_sha256(STATE, QUESTION, IDENTITY)
    for field, value in {
        "backend": backend,
        "model_id": model,
        "model_revision": revision,
        "dtype": dtype,
        "scorer_sha256": scorer,
    }.items():
        assert (
            cache.request_sha256(STATE, QUESTION, replace(IDENTITY, **{field: value}))
            != original
        )


def test_prompt_content_changes_digest_but_key_order_does_not() -> None:
    original = cache.request_sha256(STATE, QUESTION, IDENTITY)
    assert original == cache.request_sha256(
        {"story": {"title": "synthetic story"}, "profile": ["synthetic liked item"]},
        QUESTION,
        IDENTITY,
    )
    assert original != cache.request_sha256(
        {**STATE, "profile": []}, QUESTION, IDENTITY
    )
    assert original != cache.request_sha256(
        STATE, {"upvote": {"instructions": "different question"}}, IDENTITY
    )


def test_round_trip_records_identity_and_request_without_private_input(
    tmp_path: Path,
) -> None:
    path = cache.cache_path(tmp_path, STATE, QUESTION, IDENTITY)
    cache.write_cached_score(
        path, 0.73, STATE, QUESTION, IDENTITY, extra_fields={"n_tokens": 4}
    )
    assert path.name == f"{cache.request_sha256(STATE, QUESTION, IDENTITY)}.json"
    assert cache.read_cached_score(path, STATE, QUESTION, IDENTITY) == 0.73
    record = json.loads(path.read_text())
    assert record["cache_identity"] == asdict(IDENTITY)
    assert record["n_tokens"] == 4
    assert "state" not in record and "question" not in record
    assert not list(tmp_path.glob(".score-*"))


@pytest.mark.parametrize("probability", [0, 1, 0.5])
def test_probability_boundaries(probability: float) -> None:
    assert (
        cache.cache_record(probability, STATE, QUESTION, IDENTITY)["p_yes"]
        == probability
    )


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, float("nan"), float("inf"), -float("inf"), True, "0.5", None],
)
def test_invalid_probabilities_rejected(probability: object) -> None:
    with pytest.raises(cache.CacheValidationError):
        cache.cache_record(probability, STATE, QUESTION, IDENTITY)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"p_yes":0.7}',
        "[]",
        "null",
        "broken json",
        '{"p_yes":NaN}',
        '{"x":1,"x":2}',
    ],
)
def test_malformed_and_legacy_files_rejected(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "existing.json"
    path.write_text(payload)
    before = path.read_bytes()
    with pytest.raises(cache.CacheValidationError):
        cache.read_cached_score(path, STATE, QUESTION, IDENTITY)
    with pytest.raises(cache.CacheValidationError):
        cache.write_cached_score(path, 0.6, STATE, QUESTION, IDENTITY)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".score-*"))


def test_missing_file_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(cache.CacheValidationError):
        cache.read_cached_score(tmp_path / "missing.json", STATE, QUESTION, IDENTITY)


@pytest.mark.parametrize(
    "field", ["backend", "model_id", "model_revision", "dtype", "scorer_sha256"]
)
def test_stale_identity_rejected_and_preserved(tmp_path: Path, field: str) -> None:
    path = tmp_path / "existing.json"
    cache.write_cached_score(path, 0.6, STATE, QUESTION, IDENTITY)
    before = path.read_bytes()
    changed = replace(
        IDENTITY, **{field: "c" * (64 if field == "scorer_sha256" else 40)}
    )
    with pytest.raises(cache.CacheValidationError):
        cache.read_cached_score(path, STATE, QUESTION, changed)
    with pytest.raises(cache.CacheValidationError):
        cache.write_cached_score(path, 0.1, STATE, QUESTION, changed)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["state", "question", "digest", "schema", "probability"]
)
def test_stale_prompt_or_corrupt_record_rejected(tmp_path: Path, change: str) -> None:
    path = tmp_path / "existing.json"
    record = cache.cache_record(0.6, STATE, QUESTION, IDENTITY)
    state, question = STATE, QUESTION
    if change == "state":
        state = {**STATE, "profile": []}
    elif change == "question":
        question = {"upvote": {"instructions": "different"}}
    elif change == "digest":
        record["request_sha256"] = "0" * 64
    elif change == "schema":
        record["cache_schema"] = True
    else:
        record["p_yes"] = 1.1
    path.write_text(json.dumps(record))
    with pytest.raises(cache.CacheValidationError):
        cache.read_cached_score(path, state, question, IDENTITY)


def test_valid_existing_record_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "existing.json"
    cache.write_cached_score(path, 0.6, STATE, QUESTION, IDENTITY)
    before = path.read_bytes()
    cache.write_cached_score(path, 0.1, STATE, QUESTION, IDENTITY)
    assert path.read_bytes() == before
    assert cache.read_cached_score(path, STATE, QUESTION, IDENTITY) == 0.6


@pytest.mark.parametrize(
    "reserved", ["cache_schema", "cache_identity", "request_sha256", "p_yes"]
)
def test_extra_fields_cannot_override_contract(reserved: str) -> None:
    with pytest.raises(cache.CacheValidationError):
        cache.cache_record(
            0.5, STATE, QUESTION, IDENTITY, extra_fields={reserved: "spoof"}
        )


@pytest.mark.parametrize("state", [{"x": float("nan")}, {"x": object()}])
def test_non_json_or_nonfinite_input_rejected(state: object) -> None:
    with pytest.raises(cache.CacheValidationError):
        cache.request_sha256(state, QUESTION, IDENTITY)


@pytest.mark.parametrize(
    "field", ["backend", "model_id", "model_revision", "dtype", "scorer_sha256"]
)
def test_blank_identity_fields_rejected(field: str) -> None:
    with pytest.raises(cache.CacheValidationError):
        replace(IDENTITY, **{field: " "})


def test_invalid_source_digest_rejected() -> None:
    with pytest.raises(cache.CacheValidationError):
        replace(IDENTITY, scorer_sha256="not a SHA256")
