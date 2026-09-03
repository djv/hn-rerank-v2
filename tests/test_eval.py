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


# ---------------------------------------------------------------------------
# Eval/prod feature parity: eval must measure the 394-d production feature
# set, including the positive-cluster column (regression coverage for the
# 2026-09 fix -- eval previously passed positive_cluster_similarity=None,
# ranking a 393-d proxy of the 394-d served model).
# ---------------------------------------------------------------------------


def _clustered_embeddings(
    seed: int, n_up: int = 20, n_down: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated 384-d blobs: ups around e0, downs around e1."""
    rng = np.random.default_rng(seed)
    up = np.zeros((n_up, 384), dtype=np.float32)
    up[:, 0] = 1.0
    up += rng.normal(0, 0.05, size=up.shape).astype(np.float32)
    down = np.zeros((n_down, 384), dtype=np.float32)
    down[:, 1] = 1.0
    down += rng.normal(0, 0.05, size=down.shape).astype(np.float32)
    for row in list(up) + list(down):
        row /= max(np.linalg.norm(row), 1e-12)
    fb_train = np.concatenate([up, down], axis=0).astype(np.float32)
    return fb_train, up.astype(np.float32)


def test_eval_cluster_feature_matches_production_schema() -> None:
    """Cluster column is populated (not nulled), 394-d, and discriminative."""
    from pipeline import (
        _positive_cluster_centers,
        _similarity_to_positive_cluster_centers,
        _svm_personalization_features,
    )

    fb_train, fb_up = _clustered_embeddings(seed=0)
    centers = _positive_cluster_centers(fb_up, 4)
    sims = _similarity_to_positive_cluster_centers(fb_train, centers)

    n = len(fb_train)
    zeros = np.zeros(n, dtype=np.float32)
    X = _svm_personalization_features(
        fb_train,
        text_lengths=np.full(n, 100),
        sim_to_upvoted=zeros,
        sim_to_downvoted=zeros,
        closest_upvoted=zeros,
        closest_downvoted=zeros,
        positive_cluster_similarity=sims,
        is_hn_live=zeros,
        is_archive=zeros,
        is_reddit=zeros,
        is_rss=zeros,
    )
    assert X.shape == (n, 394)
    cluster_col = X[:, 384 + 5]
    assert bool((cluster_col > 0).any()), "cluster column must not be all-zero"
    assert float(cluster_col.min()) >= 0.0 and float(cluster_col.max()) <= 1.0
    # Ups (around e0, where centers were fit) outrank downs on this column.
    assert float(cluster_col[: len(fb_up)].mean()) > float(
        cluster_col[len(fb_up) :].mean()
    )


def test_eval_cluster_feature_empty_up_class_yields_zeros() -> None:
    """No up-voted feedback degrades to a zero column, matching production
    (`_similarity_to_positive_cluster_centers` with empty centers)."""
    from pipeline import (
        _positive_cluster_centers,
        _similarity_to_positive_cluster_centers,
    )

    fb_train, _ = _clustered_embeddings(seed=1)
    centers = _positive_cluster_centers(np.zeros((0, 384), dtype=np.float32), 4)
    assert centers.shape == (0, 384)
    sims = _similarity_to_positive_cluster_centers(fb_train, centers)
    assert sims.shape == (len(fb_train),)
    assert bool((sims == 0).all())


def test_eval_py_does_not_null_the_cluster_column() -> None:
    """Locks the fix in place: eval.py must thread real cluster similarities
    into both `_svm_personalization_features` calls (train + candidates)."""
    src = EVAL_PY.read_text()
    assert "positive_cluster_similarity=None" not in src
    assert src.count("positive_cluster_similarity=fb_cluster_sim") == 1
    assert src.count("positive_cluster_similarity=cand_cluster_sim") == 1
