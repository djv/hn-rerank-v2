import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Database, Story
from eval import MIN_EMBEDDING_COVERAGE, _load_candidates

REPORT = Path(__file__).parent.parent / "eval_report.json"
EVAL_PY = Path(__file__).parent.parent / "eval.py"
MODEL_VERSION = "mxbai-embed-xsmall-v1|mean|norm|4096"


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


# ---------------------------------------------------------------------------
# _load_candidates: model_version must be threaded in, not hardcoded
# (regression coverage for the 2026-08-27 fix -- see WORKLOG). A prior
# hardcoded MODEL_VERSION constant disagreed with the configured encoder,
# so most embedding lookups silently missed and fell back to zero vectors.
# ---------------------------------------------------------------------------


def _seed_story_with_embedding(
    db: Database, story_id: int, text: str, model_version: str | None
) -> None:
    import hashlib

    db.upsert_story(
        Story(
            id=story_id,
            title=f"story {story_id}",
            url=None,
            score=1,
            time=1,
            text_content=text,
            source="hn",
        )
    )
    if model_version is not None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        db.upsert_embedding(
            story_id, model_version, text_hash, np.zeros(384, dtype=np.float32)
        )


def test_load_candidates_uses_the_passed_model_version_not_a_hardcoded_one():
    """A story embedded under a different model_version must count as a
    miss, not a hit -- proves the lookup key comes from the caller."""
    db = Database(":memory:")
    _seed_story_with_embedding(db, 1, "story one", MODEL_VERSION)
    _seed_story_with_embedding(db, 2, "story two", "some-other-model|v1")

    with pytest.raises(RuntimeError, match="coverage"):
        _load_candidates(db, MODEL_VERSION)


def test_load_candidates_raises_on_low_embedding_coverage():
    """Below MIN_EMBEDDING_COVERAGE, fail loudly rather than silently
    filling the gap with zero vectors (the pre-fix behavior)."""
    assert 0.1 < MIN_EMBEDDING_COVERAGE, "test assumes 1/10 coverage is a failure"
    db = Database(":memory:")
    for i in range(10):
        # Only embed 1 of 10 -- well under MIN_EMBEDDING_COVERAGE.
        _seed_story_with_embedding(
            db, i, f"story {i}", MODEL_VERSION if i == 0 else None
        )

    with pytest.raises(RuntimeError, match="coverage"):
        _load_candidates(db, MODEL_VERSION)


def test_load_candidates_succeeds_with_full_coverage():
    db = Database(":memory:")
    for i in range(10):
        _seed_story_with_embedding(db, i, f"story {i}", MODEL_VERSION)

    stories, embeddings = _load_candidates(db, MODEL_VERSION)
    assert len(stories) == 10
    assert embeddings.shape == (10, 384)


def test_load_candidates_empty_db_does_not_raise():
    """Zero candidate stories is a degenerate-but-valid input (coverage
    over an empty set is defined as 1.0), not a coverage failure."""
    db = Database(":memory:")
    stories, embeddings = _load_candidates(db, MODEL_VERSION)
    assert stories == []
    assert embeddings.shape == (0,)
