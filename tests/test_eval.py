import json
import subprocess
from pathlib import Path

REPORT = Path(__file__).parent.parent / "eval_report.json"
EVAL_PY = Path(__file__).parent.parent / "eval.py"


def test_report_exists():
    assert REPORT.exists(), "Run `uv run python eval.py` first."


def test_candidate_cap_flag_in_help() -> None:
    """--candidate-cap must be exposed in --help output.

    Locks the memory-bounding CLI flag in place; regression test against
    accidental removal in future refactors. Added 2026-06-28 when this
    flag was discovered during eval.py memory profiling (see WORKLOG).
    """
    result = subprocess.run(
        ["uv", "run", "python", str(EVAL_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "--candidate-cap" in result.stdout, (
        "--candidate-cap not in --help output; memory-bounding flag is missing"
    )
    assert "--candidate-cap-seed" in result.stdout, (
        "--candidate-cap-seed not in --help output"
    )


def test_report_has_expected_formulas():
    r = json.loads(REPORT.read_text())
    expected = {"current", "up_only", "hn_baseline"}
    assert expected.issubset(r["formulas"].keys())


def test_report_has_5_folds():
    r = json.loads(REPORT.read_text())
    for formula in r["formulas"].values():
        assert len(formula["per_fold"]) == 5


def test_svm_better_than_random():
    r = json.loads(REPORT.read_text())
    up_only = r["formulas"]["up_only"]["mean"]["mmr"]["ndcg_at_40"]
    hn = r["formulas"]["hn_baseline"]["mean"]["mmr"]["ndcg_at_40"]
    assert up_only > hn, f"SVM NDCG@40 ({up_only:.3f}) <= HN baseline ({hn:.3f})"


def test_report_has_map_and_brier():
    r = json.loads(REPORT.read_text())
    for formula_data in r["formulas"].values():
        for variant in ("mmr", "raw"):
            metrics = formula_data["mean"][variant]
            assert "map" in metrics, f"map missing from {variant}"
            assert "brier_up" in metrics, f"brier_up missing from {variant}"


def test_map_in_valid_range():
    r = json.loads(REPORT.read_text())
    for formula_data in r["formulas"].values():
        for variant in ("mmr", "raw"):
            map_val = formula_data["mean"][variant]["map"]
            assert 0.0 <= map_val <= 1.0, f"map {map_val} out of [0,1]"


def test_svm_map_better_than_hn_baseline():
    r = json.loads(REPORT.read_text())
    svm_map = r["formulas"]["up_only"]["mean"]["mmr"]["map"]
    hn_map = r["formulas"]["hn_baseline"]["mean"]["mmr"]["map"]
    assert svm_map > hn_map, (
        f"SVM MAP ({svm_map:.3f}) <= HN baseline MAP ({hn_map:.3f})"
    )


def test_final_queue_present():
    r = json.loads(REPORT.read_text())
    assert "final_queue" in r, "final_queue key missing from report"
    fq = r["final_queue"]["mean"]["mmr"]
    assert "ndcg_at_40" in fq
    assert "hit_at_40" in fq
    assert "map" in fq
    assert "brier_up" in fq


def test_final_queue_per_source_present():
    r = json.loads(REPORT.read_text())
    fq = r.get("final_queue", {})
    assert "per_source" in fq, "final_queue.per_source missing"
    ps = fq["per_source"]
    assert isinstance(ps, dict)
    if ps:
        source = next(iter(ps))
        assert "n_test" in ps[source]
        assert "mean" in ps[source]
        assert "mmr" in ps[source]["mean"]
        assert "ndcg_at_40" in ps[source]["mean"]["mmr"]
