from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from database import Database, Story
from scripts import benchmark_embeddings as bench


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_dir: str) -> FakeTokenizer:
        assert model_dir
        return cls()

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        assert truncation
        encoded = [
            [max(1, (ord(char) % 97) + 1) for char in text[:max_length]]
            for text in texts
        ]
        if not padding:
            return {"input_ids": encoded}

        width = max((len(row) for row in encoded), default=1)
        input_ids = np.zeros((len(encoded), width), dtype=np.int64)
        attention_mask = np.zeros((len(encoded), width), dtype=np.int64)
        for idx, row in enumerate(encoded):
            input_ids[idx, : len(row)] = row
            attention_mask[idx, : len(row)] = 1
        if return_tensors == "np":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }


class FakeSessionOptions:
    def __init__(self) -> None:
        self.enable_cpu_mem_arena: bool | None = None
        self.enable_mem_pattern: bool | None = None
        self.intra_op_num_threads: int | None = None
        self.inter_op_num_threads: int | None = None
        self.graph_optimization_level: str | None = None
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


class FakeSession:
    options_seen: list[FakeSessionOptions] = []

    def __init__(
        self,
        model_path: str,
        *,
        sess_options: FakeSessionOptions,
        providers: list[str],
    ) -> None:
        assert model_path.endswith("model.onnx")
        assert providers == ["CPUExecutionProvider"]
        self.options_seen.append(sess_options)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(
        self,
        output_names: object,
        onnx_inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert output_names is None
        input_ids = onnx_inputs["input_ids"].astype(np.float32)
        dims = np.arange(bench.EMBEDDING_DIM, dtype=np.float32) / 1000.0
        token_embeddings = input_ids[:, :, None] + dims[None, None, :]
        return [token_embeddings.astype(np.float32)]


@pytest.fixture(autouse=True)
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSession.options_seen = []
    monkeypatch.setattr(bench, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(
        bench,
        "ort",
        SimpleNamespace(
            SessionOptions=FakeSessionOptions,
            InferenceSession=FakeSession,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ),
    )


def _create_db(path: Path) -> None:
    db = Database(str(path))
    db.upsert_story(
        Story(
            id=1,
            title="First title",
            url=None,
            score=10,
            time=100,
            text_content="short text",
            source="hn",
        )
    )
    db.upsert_story(
        Story(
            id=2,
            title="Second title",
            url=None,
            score=20,
            time=200,
            text_content="",
            source="hn",
            self_text="self text",
            top_comments="comment text",
        )
    )
    db.upsert_story(
        Story(
            id=3,
            title="",
            url=None,
            score=0,
            time=300,
            text_content="",
            source="hn",
        )
    )
    db.close()


def test_benchmark_embeddings_outputs_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "bench.db"
    output_path = tmp_path / "out.json"
    _create_db(db_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_embeddings.py",
            "--db",
            str(db_path),
            "--sample-size",
            "2",
            "--runs",
            "1",
            "--variants",
            "current",
            "--batch-sizes",
            "1,32",
            "--json-only",
            "--write-output",
            str(output_path),
        ],
    )

    bench.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline"] == {"variant": "current", "batch_size": 32}
    assert payload["token_length_percentiles"]["max"] > 0
    assert [row["batch_size"] for row in payload["results"]] == [1, 32]
    assert payload["results"][0]["output_shape"] == [2, bench.EMBEDDING_DIM]
    assert payload["results"][0]["finite"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_sample_texts_are_deterministic_and_filter_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.db"
    _create_db(db_path)
    db = Database(str(db_path), read_only=True)
    try:
        first = bench.load_sample_texts(db, sample_size=2, seed=7)
        second = bench.load_sample_texts(db, sample_size=2, seed=7)
        other = bench.load_sample_texts(db, sample_size=2, seed=8)
    finally:
        db.close()

    assert first == second
    assert all(text.strip() for text in first)
    assert len(first) == 2
    assert set(first) == set(other)


def test_variant_selection_sets_expected_session_options() -> None:
    _ = bench.BenchmarkEmbedder("onnx_model", bench.VARIANT_SPECS["spin_off_auto_threads"])

    options = FakeSession.options_seen[-1]
    assert options.entries == {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }
    assert options.graph_optimization_level == "all"
    assert options.intra_op_num_threads == 0
    assert options.inter_op_num_threads == 1


def test_drift_calculation_flags_equivalent_and_changed_vectors() -> None:
    baseline = np.eye(bench.EMBEDDING_DIM, dtype=np.float32)[:2]
    same = baseline.copy()
    changed = baseline.copy()
    changed[1] *= -1

    same_drift = bench.drift_against_baseline(baseline, same)
    changed_drift = bench.drift_against_baseline(baseline, changed)

    assert same_drift["acceptable"] is True
    assert same_drift["min_cosine_to_baseline"] == pytest.approx(1.0)
    assert changed_drift["acceptable"] is False
    assert changed_drift["min_cosine_to_baseline"] == pytest.approx(-1.0)


def test_database_is_opened_read_only_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "bench.db"
    _create_db(db_path)
    opened_read_only: list[bool] = []
    real_database = bench.Database

    class TrackingDatabase(real_database):
        def __init__(self, path: str = "hn_rewrite.db", *, read_only: bool = False) -> None:
            opened_read_only.append(read_only)
            super().__init__(path, read_only=read_only)

    monkeypatch.setattr(bench, "Database", TrackingDatabase)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_embeddings.py",
            "--db",
            str(db_path),
            "--sample-size",
            "1",
            "--runs",
            "0",
            "--variants",
            "current",
            "--batch-sizes",
            "32",
            "--json-only",
        ],
    )

    bench.main()

    assert opened_read_only == [True]
    assert json.loads(capsys.readouterr().out)["results"][0]["sample_size"] == 1
