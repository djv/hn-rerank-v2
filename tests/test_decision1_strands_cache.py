"""Synthetic worker/cache preflight tests: no torch, model, downloads or inference."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

EVALUATION = Path(__file__).parents[1] / "docs/evaluations/decision1-20261010"


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch):
    cache_spec = importlib.util.spec_from_file_location(
        "score_cache", EVALUATION / "score_cache.py"
    )
    assert cache_spec is not None and cache_spec.loader is not None
    cache = importlib.util.module_from_spec(cache_spec)
    monkeypatch.setitem(sys.modules, "score_cache", cache)
    cache_spec.loader.exec_module(cache)

    def forbidden(*args, **kwargs):
        raise AssertionError("real runtime/inference must never run in synthetic tests")

    torch = ModuleType("torch")
    setattr(torch, "inference_mode", lambda: lambda function: function)
    setattr(torch, "set_num_threads", lambda count: None)
    setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    strands = ModuleType("strands_decider")
    infer = ModuleType("strands_decider.infer")
    setattr(infer, "_expand_cache", forbidden)
    setattr(infer, "load_engine", forbidden)
    setattr(
        infer,
        "SystemOneEngine",
        type("SystemOneEngine", (), {"_upcast_torso_for_cpu": forbidden}),
    )
    modeling = ModuleType("strands_decider.modeling")
    setattr(
        modeling,
        "StrandsDeciderModel",
        type("StrandsDeciderModel", (), {"_load_torso": staticmethod(forbidden)}),
    )
    for name in [
        "apply_temperature",
        "gather_options",
        "masked_log_softmax",
        "pool_last_token",
    ]:
        setattr(modeling, name, forbidden)
    prompting = ModuleType("strands_decider.prompting")
    for name in ["read_noul", "render_question", "render_state"]:
        setattr(prompting, name, forbidden)
    schema = ModuleType("strands_decider.schema")
    setattr(schema, "NoulQuestion", forbidden)
    setattr(schema, "SystemOneRequest", forbidden)
    setattr(strands, "infer", infer)
    for name, module in {
        "strands_decider": strands,
        "strands_decider.infer": infer,
        "strands_decider.modeling": modeling,
        "strands_decider.prompting": prompting,
        "strands_decider.schema": schema,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "synthetic_strands_worker", EVALUATION / "strands_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_for(worker, directory: Path) -> tuple[dict, object]:
    identity = worker.worker_identity(
        worker.MODEL_ID, worker.MODEL_REVISION, "bf16w-fp32"
    )
    state = {"profile": ["synthetic preference"], "story": "synthetic story"}
    question = {"upvote": {"instructions": "synthetic decision"}}
    digest = worker.request_sha256(state, question, identity)
    job = {
        "out": str(directory / f"{digest}.json"),
        "state": state,
        "question": question,
        "cache_identity": asdict(identity),
        "request_sha256": digest,
    }
    return job, identity


def test_missing_is_pending_and_valid_cache_is_skipped(worker, tmp_path: Path) -> None:
    job, identity = job_for(worker, tmp_path)
    assert worker.validated_pending_jobs([job], identity) == [job]
    worker.write_cached_score(
        Path(job["out"]),
        0.4,
        job["state"],
        job["question"],
        identity,
        extra_fields={
            "prefix_cached": False,
            "parity_validation": {"schema": 1, "method": "plain"},
        },
    )
    assert worker.validated_pending_jobs([job], identity) == []


@pytest.mark.parametrize(
    "change", ["identity", "request", "filename", "legacy", "malformed", "profile"]
)
def test_worker_rejects_bad_requests_and_existing_outputs(
    worker, tmp_path: Path, change: str
) -> None:
    job, identity = job_for(worker, tmp_path)
    path = Path(job["out"])
    if change == "identity":
        job["cache_identity"] = {**job["cache_identity"], "dtype": "int8"}
    elif change == "request":
        job["request_sha256"] = "0" * 64
    elif change == "filename":
        job["out"] = str(tmp_path / "legacy-story-id.json")
    elif change == "legacy":
        path.write_text('{"p_yes":0.4}')
    elif change == "malformed":
        path.write_text("broken JSON")
    else:
        job["state"] = {**job["state"], "profile": []}
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(worker.CacheValidationError):
        worker.validated_pending_jobs([job], identity)
    if before is not None:
        assert path.read_bytes() == before


def test_worker_identity_rejects_mutable_revision(worker) -> None:
    with pytest.raises(worker.CacheValidationError):
        worker.worker_identity(worker.MODEL_ID, "main", "bf16w-fp32")


def test_package_versions_must_match_preserved_environment(
    worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        worker, "installed_version", lambda name: worker.EXPECTED_RUNTIME_PACKAGES[name]
    )
    assert worker.validate_runtime_packages() == worker.EXPECTED_RUNTIME_PACKAGES
    monkeypatch.setattr(worker, "installed_version", lambda name: "wrong-version")
    with pytest.raises(worker.CacheValidationError, match="expected pinned"):
        worker.validate_runtime_packages()


def test_missing_runtime_package_fails_without_install(
    worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    def absent(name: str) -> str:
        raise worker.PackageNotFoundError(name)

    monkeypatch.setattr(worker, "installed_version", absent)
    with pytest.raises(worker.CacheValidationError, match="is absent"):
        worker.validate_runtime_packages()


def test_local_snapshot_requests_exact_revision_and_offline_only(
    worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / worker.MODEL_REVISION
    snapshot.mkdir(parents=True)
    calls = []

    def resolve(**kwargs) -> str:
        calls.append(kwargs)
        return str(snapshot)

    hub = ModuleType("huggingface_hub")
    setattr(hub, "snapshot_download", resolve)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert worker.local_snapshot(worker.MODEL_ID, worker.MODEL_REVISION) == snapshot
    assert calls == [
        {
            "repo_id": worker.MODEL_ID,
            "revision": worker.MODEL_REVISION,
            "local_files_only": True,
        }
    ]


def test_local_snapshot_refuses_mutable_or_absent_resolution(
    worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = ModuleType("huggingface_hub")
    setattr(hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(worker.CacheValidationError):
        worker.local_snapshot(worker.MODEL_ID, worker.MODEL_REVISION)

    def absent(**kwargs):
        raise FileNotFoundError("not cached")

    setattr(hub, "snapshot_download", absent)
    with pytest.raises(worker.CacheValidationError, match="no download allowed"):
        worker.local_snapshot(worker.MODEL_ID, worker.MODEL_REVISION)


def test_engine_adapter_and_base_receive_local_paths(
    worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, base = tmp_path / "adapter", tmp_path / "base"
    observed = []

    def load_torso(config, device_map, attention):
        observed.append(
            (config.base_model, config.base_revision, device_map, attention)
        )
        return object()

    monkeypatch.setattr(
        worker.StrandsDeciderModel, "_load_torso", staticmethod(load_torso)
    )

    def load_engine(path: str, *, device: str):
        assert path == str(adapter) and device == "cpu"
        config = SimpleNamespace(
            base_model=worker.BASE_MODEL_ID, base_revision="mutable-main"
        )
        worker.StrandsDeciderModel._load_torso(config, None, None)
        return SimpleNamespace(model=SimpleNamespace(config=config), tok=object())

    monkeypatch.setattr(worker, "load_engine", load_engine)
    scorer = worker.Scorer(str(adapter), "float32", base_snapshot=base)
    assert scorer.model.config.base_model == str(base)
    assert observed == [(str(base), None, None, None)]
    assert worker.StrandsDeciderModel._load_torso is load_torso


def test_engine_that_ignores_base_pin_fails(
    worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        worker,
        "load_engine",
        lambda *args, **kwargs: SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(base_model=worker.BASE_MODEL_ID)
            ),
            tok=object(),
        ),
    )
    with pytest.raises(worker.CacheValidationError, match="did not use"):
        worker.Scorer("adapter", "float32", base_snapshot=tmp_path / "base")


def test_main_all_valid_entries_does_not_resolve_or_load(
    worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    job, identity = job_for(worker, tmp_path)
    worker.write_cached_score(
        Path(job["out"]),
        0.4,
        job["state"],
        job["question"],
        identity,
        extra_fields={
            "prefix_cached": False,
            "parity_validation": {"schema": 1, "method": "plain"},
        },
    )
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    monkeypatch.setattr(
        sys, "argv", ["strands_worker", str(jobs_path), "--dtype", "bf16w-fp32"]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no model resolution or loading for cached requests")

    monkeypatch.setattr(worker, "local_snapshot", forbidden)
    monkeypatch.setattr(worker, "Scorer", forbidden)
    worker.main()


def run_synthetic_worker(
    worker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scores: list[float],
    plain_scores: list[float],
    groups: list[str] | None = None,
    options: list[str] | None = None,
):
    import json

    jobs = []
    for i in range(len(scores)):
        job, identity = job_for(worker, tmp_path)
        job["state"]["story"] = f"synthetic story {i}"
        digest = worker.request_sha256(job["state"], job["question"], identity)
        job.update(
            out=str(tmp_path / f"{digest}.json"),
            request_sha256=digest,
            group=groups[i] if groups else "real",
        )
        jobs.append(job)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strands_worker",
            str(jobs_path),
            "--dtype",
            "bf16w-fp32",
            "--batch",
            "4",
            *(options or []),
        ],
    )
    monkeypatch.setattr(worker, "validate_runtime_packages", lambda: {})
    monkeypatch.setattr(worker, "local_snapshot", lambda *args: tmp_path)
    monkeypatch.setattr(worker, "render_state", lambda state: state["story"])
    monkeypatch.setattr(worker, "common_prefix", lambda texts: "prefix")
    plain_calls = []

    class SyntheticScorer:
        def __init__(self, *args, **kwargs):
            pass

        def prefix(self, text):
            return [1], object()

        def batch(self, prefix_ids, cache, rows):
            return [(scores[int(text.rsplit(" ", 1)[1])], 10) for text, _ in rows]

        def plain(self, state, question):
            index = int(state["story"].rsplit(" ", 1)[1])
            plain_calls.append(index)
            return plain_scores[index]

    monkeypatch.setattr(worker, "Scorer", SyntheticScorer)
    return jobs, plain_calls


@pytest.mark.parametrize(
    "scores,plain",
    [
        ([0.95, 0.95], [0.05, 0.05]),
        ([0.4, float("nan")], [0.4, 0.4]),
        ([0.4, 0.5], [0.4, float("inf")]),
        ([0.4, 1.5], [0.4, 0.5]),
    ],
)
def test_parity_or_nonfinite_failure_publishes_nothing_in_chunk(
    worker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scores: list[float],
    plain: list[float],
) -> None:
    jobs, _ = run_synthetic_worker(worker, tmp_path, monkeypatch, scores, plain)
    with pytest.raises(worker.CacheValidationError):
        worker.main()
    assert all(not Path(job["out"]).exists() for job in jobs)


def test_every_profile_group_is_checked_before_its_scores_are_published(
    worker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, calls = run_synthetic_worker(
        worker,
        tmp_path,
        monkeypatch,
        [0.4, 0.4, 0.95, 0.95],
        [0.4, 0.4, 0.05, 0.05],
        ["real", "real", "empty", "empty"],
    )
    with pytest.raises(worker.CacheValidationError, match="parity failed"):
        worker.main()
    assert calls == [1, 3]
    assert all(Path(job["out"]).exists() for job in jobs[:2])
    assert all(not Path(job["out"]).exists() for job in jobs[2:])


def test_later_chunk_mismatch_stops_before_publishing_that_chunk(
    worker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, calls = run_synthetic_worker(
        worker, tmp_path, monkeypatch, [0.4] * 4 + [0.95] * 2, [0.4] * 4 + [0.05] * 2
    )
    with pytest.raises(worker.CacheValidationError, match="parity failed"):
        worker.main()
    assert calls == [3, 5]
    assert all(Path(job["out"]).exists() for job in jobs[:4])
    assert all(not Path(job["out"]).exists() for job in jobs[4:])


def test_finite_parity_within_tolerance_publishes_valid_scores(
    worker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, calls = run_synthetic_worker(
        worker, tmp_path, monkeypatch, [0.4, 0.5], [0.4, 0.50005]
    )
    worker.main()
    assert calls == [1]
    assert all(Path(job["out"]).exists() for job in jobs)
    identity = worker.worker_identity(
        worker.MODEL_ID, worker.MODEL_REVISION, "bf16w-fp32"
    )
    for job, expected in zip(jobs, [0.4, 0.5], strict=True):
        assert (
            worker.read_cached_score(
                Path(job["out"]),
                job["state"],
                job["question"],
                identity,
                require_parity=True,
            )
            == expected
        )


@pytest.mark.parametrize(
    "options",
    [
        ["--check", "0"],
        ["--check", "-1"],
        ["--parity-tolerance", "nan"],
        ["--parity-tolerance", "inf"],
        ["--parity-tolerance", "-0.1"],
        ["--parity-tolerance", "1e9"],
    ],
)
def test_invalid_parity_options_stop_before_runtime(
    worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    jobs, calls = run_synthetic_worker(
        worker, tmp_path, monkeypatch, [0.4, 0.5], [0.4, 0.5], options=options
    )
    with pytest.raises(worker.CacheValidationError, match="check must be positive"):
        worker.main()
    assert not calls
    assert all(not Path(job["out"]).exists() for job in jobs)
