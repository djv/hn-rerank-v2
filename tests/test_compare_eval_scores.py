from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.compare_eval_scores import (
    Fold,
    _auc,
    _average_precision,
    _bootstrap,
    _delong,
    _holm,
    _pair,
    _sign_flip_p,
    main,
)


def _scores(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = rng.random(300) < 0.25
    a = rng.normal(size=300) + positive * 0.8
    b = a + rng.normal(scale=0.5, size=300)
    a[:20] = a[20:40]  # ties get half credit
    return positive, a, b


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auc_and_average_precision_match_sklearn(seed: int) -> None:
    positive, a, b = _scores(seed)
    assert _auc(a, positive) == pytest.approx(roc_auc_score(positive, a))
    assert _average_precision(b, positive) == pytest.approx(
        average_precision_score(positive, b)
    )


def test_delong_matches_stratified_bootstrap() -> None:
    positive, a, b = _scores(1)
    fold = Fold("dev:1", positive, (a, b))
    delta, variance = _delong(fold)
    boot = _bootstrap([fold], "AUC", 1500, np.random.default_rng(0))
    assert delta == pytest.approx(_auc(b, positive) - _auc(a, positive))
    assert np.sqrt(variance) == pytest.approx(boot.std(), rel=0.15)


def test_identical_runs_have_no_difference() -> None:
    positive, a, _b = _scores(0)
    fold = Fold("dev:1", positive, (a, a.copy()))
    assert _delong(fold) == (0.0, 0.0)
    assert np.all(_bootstrap([fold], "P@12", 50, np.random.default_rng(0)) == 0)


def test_sign_flip_is_exact_and_holm_is_monotone() -> None:
    assert _sign_flip_p(np.full(12, 0.01)) == pytest.approx(2 / 4096)
    assert _sign_flip_p(np.array([0.01, -0.01])) == 1.0
    assert _holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_pair_rejects_runs_over_different_stories(tmp_path: Path) -> None:
    def dump(name: str, ids: list[int]) -> Path:
        path = tmp_path / name
        fold = {
            "fold": 1,
            "ids": ids,
            "labels": [2, 0, 1],
            "scores": {"production": [0.3, 0.2, 0.1]},
        }
        path.write_text(json.dumps({"folds": [fold]}))
        return path

    base = dump("dev-a.scores.json", [1, 2, 3])
    assert (
        len(_pair(base, dump("dev-b.scores.json", [1, 2, 3]), "production", "dev:"))
        == 1
    )
    with pytest.raises(SystemExit):
        _pair(base, dump("dev-c.scores.json", [1, 2, 4]), "production", "dev:")


def test_cli_retains_periods_with_the_same_filename_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: list[Path] = []
    for period in ("development", "fresh"):
        for kind in ("base", "other"):
            path = tmp_path / f"broad-{period}-{kind}.json"
            path.write_text(
                json.dumps(
                    {
                        "folds": [
                            {
                                "fold": 1,
                                "ids": list(range(16)),
                                "labels": [2, 0] * 8,
                                "scores": {"production": list(range(16))},
                            }
                        ]
                    }
                )
            )
            paths.append(path)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_eval_scores.py",
            f"base={paths[0]},{paths[2]}",
            f"other={paths[1]},{paths[3]}",
            "--boot",
            "10",
            "--block-boot",
            "10",
            "--json",
            str(output),
        ],
    )
    main()
    scopes = json.loads(output.read_text())["production"]["other"]
    assert set(scopes) == {
        "broad-development-base",
        "broad-fresh-base",
        "pooled",
    }
    assert scopes["broad-development-base"]["folds"] == 1
    assert scopes["broad-fresh-base"]["folds"] == 1
    assert scopes["pooled"]["folds"] == 2
