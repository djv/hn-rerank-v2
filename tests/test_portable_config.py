"""Configuration paths must work outside the owner's deployment layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "config.py"
_PATH_KEYS = ("HN_ONNX_MODEL_DIR", "HN_SIDE_MODEL_DIR", "HN_ENV_FILE")


def _read_defaults(config_path: Path, overrides: dict[str, str]) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key not in _PATH_KEYS
    }
    environment.update(overrides)
    script = (
        "import json, runpy, sys; config = runpy.run_path(sys.argv[1]); "
        "print(json.dumps({key: config[key] for key in "
        "('DEFAULT_ONNX_MODEL_DIR', 'DEFAULT_SIDE_EMBEDDING_MODEL_DIR', "
        "'DEFAULT_ENV_PATH')}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("shared_exists", [False, True])
def test_path_defaults_preserve_shared_layout_or_use_local_models(
    tmp_path: Path, shared_exists: bool
) -> None:
    config_path = tmp_path / "checkout" / "pipeline" / "config.py"
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(_CONFIG_PATH, config_path)
    shared = tmp_path / "shared"
    if shared_exists:
        shared.mkdir()

    defaults = _read_defaults(config_path, {})

    assert defaults["DEFAULT_ONNX_MODEL_DIR"] == (
        str(shared / "mxbai-embed-xsmall-v1") if shared_exists else "onnx_model"
    )
    assert defaults["DEFAULT_SIDE_EMBEDDING_MODEL_DIR"] == (
        str(shared / "embeddinggemma-300m-onnx")
        if shared_exists
        else "onnx_model_embeddinggemma"
    )
    assert defaults["DEFAULT_ENV_PATH"] == str(shared / ".env")


def test_process_environment_overrides_model_and_secrets_paths(tmp_path: Path) -> None:
    overrides = {key: str(tmp_path / key.lower()) for key in _PATH_KEYS}

    defaults = _read_defaults(_CONFIG_PATH, overrides)

    assert defaults == {
        "DEFAULT_ONNX_MODEL_DIR": overrides["HN_ONNX_MODEL_DIR"],
        "DEFAULT_SIDE_EMBEDDING_MODEL_DIR": overrides["HN_SIDE_MODEL_DIR"],
        "DEFAULT_ENV_PATH": overrides["HN_ENV_FILE"],
    }


@pytest.mark.parametrize("server", [None, "https://example.com/hn/"])
def test_tui_server_default_is_local_and_can_be_overridden(server: str | None) -> None:
    environment = dict(os.environ)
    environment.pop("HN_RERANK_SERVER", None)
    # A reused development environment may have an editable install elsewhere.
    source = _CONFIG_PATH.parents[1] / "clients" / "tui" / "src"
    environment["PYTHONPATH"] = str(source)
    if server is not None:
        environment["HN_RERANK_SERVER"] = server
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hn_rerank.app import DEFAULT_SERVER; print(DEFAULT_SERVER)",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == (server or "http://127.0.0.1:8766/")
