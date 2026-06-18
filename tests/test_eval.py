import json
from pathlib import Path

REPORT = Path(__file__).parent.parent / "eval_report.json"


def test_report_exists():
    assert REPORT.exists(), "Run `uv run python eval.py` first."


def test_report_has_3_formulas():
    r = json.loads(REPORT.read_text())
    assert r["formulas"].keys() == {"soft", "up_only", "hn_baseline"}


def test_report_has_5_folds():
    r = json.loads(REPORT.read_text())
    for formula in r["formulas"].values():
        assert len(formula["per_fold"]) == 5


def test_svm_better_than_random():
    r = json.loads(REPORT.read_text())
    up_only = r["formulas"]["up_only"]["mean"]["ndcg_at_10"]
    hn = r["formulas"]["hn_baseline"]["mean"]["ndcg_at_10"]
    assert up_only > hn, f"SVM NDCG@10 ({up_only:.3f}) <= HN baseline ({hn:.3f})"
