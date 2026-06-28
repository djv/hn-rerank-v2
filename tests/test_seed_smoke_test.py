import io
from pathlib import Path


from scripts.seed_smoke_test import (
    _normalize,
    build_idx,
    compute_diff,
    load_jsonl,
    print_summary,
)


def test_load_jsonl_skips_meta_header(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    f.write_text(
        '{"_meta": {"source": "bq", "rows": 2}}\n'
        '{"id": 1, "title": "A"}\n'
        '{"id": 2, "title": "B"}\n'
    )
    rows = load_jsonl(f)
    assert len(rows) == 2
    assert rows[0]["id"] == 1


def test_load_jsonl_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    assert load_jsonl(f) == []


def test_load_jsonl_skip_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "blank.jsonl"
    f.write_text('{"id": 1}\n\n\n{"id": 2}\n')
    rows = load_jsonl(f)
    assert len(rows) == 2


def test_normalize_strips_html() -> None:
    assert _normalize("<b>Title</b>") == "Title"
    assert _normalize("<a href='x'>Link</a>") == "Link"
    assert _normalize("  spaced  text  ") == "spaced text"
    assert _normalize("") == ""


def test_build_idx() -> None:
    rows = [{"id": 10, "title": "A"}, {"id": 20, "title": "B"}]
    idx = build_idx(rows)
    assert idx == {10: rows[0], 20: rows[1]}


def test_build_idx_skips_rows_without_id() -> None:
    rows = [{"title": "A"}, {"id": 1, "title": "B"}]
    idx = build_idx(rows)
    assert idx == {1: rows[1]}


def test_compute_diff_empty_inputs() -> None:
    report = compute_diff([], [])
    assert report["counts"] == {
        "bq": 0,
        "ch": 0,
        "intersection": 0,
        "bq_only": 0,
        "ch_only": 0,
    }


def test_compute_diff_identical() -> None:
    rows = [
        {
            "id": 1,
            "title": "A",
            "score": 100,
            "descendants": 10,
            "created_at_i": 1760000000,
            "url": "https://x.com",
        },
        {
            "id": 2,
            "title": "B",
            "score": 200,
            "descendants": 20,
            "created_at_i": 1760000001,
            "url": "https://y.com",
        },
    ]
    report = compute_diff(rows, rows)
    assert report["counts"]["intersection"] == 2
    for field, info in report["field_agreement"].items():
        assert info["match"] == 2, f"{field} should have 2 matches"
        assert info["differ"] == 0


def test_compute_diff_bq_only() -> None:
    bq = [
        {
            "id": 1,
            "title": "A",
            "score": 100,
            "descendants": 0,
            "created_at_i": 0,
            "url": "",
        }
    ]
    ch: list[dict] = []
    report = compute_diff(bq, ch)
    assert report["counts"]["bq_only"] == 1
    assert report["counts"]["intersection"] == 0
    assert len(report["bq_only_stories"]) == 1


def test_compute_diff_ch_only() -> None:
    ch = [
        {
            "id": 2,
            "title": "B",
            "score": 200,
            "descendants": 0,
            "created_at_i": 0,
            "url": "",
        }
    ]
    bq: list[dict] = []
    report = compute_diff(bq, ch)
    assert report["counts"]["ch_only"] == 1
    assert report["counts"]["intersection"] == 0
    assert len(report["ch_only_stories"]) == 1


def test_compute_diff_score_within_delta() -> None:
    bq = [
        {
            "id": 1,
            "title": "A",
            "score": 100,
            "descendants": 10,
            "created_at_i": 1760000000,
            "url": "",
        }
    ]
    ch = [
        {
            "id": 1,
            "title": "A",
            "score": 103,
            "descendants": 10,
            "created_at_i": 1760000000,
            "url": "",
        }
    ]
    report = compute_diff(bq, ch, score_delta=5)
    assert report["field_agreement"]["score"]["match"] == 1
    assert report["field_agreement"]["score"]["differ"] == 0


def test_compute_diff_score_exceeds_delta() -> None:
    bq = [
        {
            "id": 1,
            "title": "A",
            "score": 100,
            "descendants": 10,
            "created_at_i": 1760000000,
            "url": "",
        }
    ]
    ch = [
        {
            "id": 1,
            "title": "A",
            "score": 120,
            "descendants": 10,
            "created_at_i": 1760000000,
            "url": "",
        }
    ]
    report = compute_diff(bq, ch, score_delta=5)
    assert report["field_agreement"]["score"]["differ"] == 1
    assert report["top_score_diffs"][0]["delta"] == -20


def test_compute_diff_title_normalized() -> None:
    bq = [
        {
            "id": 1,
            "title": "<b>Title</b>",
            "score": 100,
            "descendants": 0,
            "created_at_i": 0,
            "url": "",
        }
    ]
    ch = [
        {
            "id": 1,
            "title": "Title",
            "score": 100,
            "descendants": 0,
            "created_at_i": 0,
            "url": "",
        }
    ]
    report = compute_diff(bq, ch)
    assert report["field_agreement"]["title"]["match"] == 1


def test_compute_diff_url_trailing_slash() -> None:
    bq = [
        {
            "id": 1,
            "title": "A",
            "score": 100,
            "descendants": 0,
            "created_at_i": 0,
            "url": "https://x.com/",
        }
    ]
    ch = [
        {
            "id": 1,
            "title": "A",
            "score": 100,
            "descendants": 0,
            "created_at_i": 0,
            "url": "https://x.com",
        }
    ]
    report = compute_diff(bq, ch)
    assert report["field_agreement"]["url"]["match"] == 1


def test_print_summary_smoke() -> None:
    report = {
        "counts": {"bq": 5, "ch": 5, "intersection": 5, "bq_only": 0, "ch_only": 0},
        "field_agreement": {
            "title": {"match": 5, "differ": 0, "diffs": []},
            "url": {"match": 5, "differ": 0, "diffs": []},
            "score": {
                "match": 4,
                "differ": 1,
                "diffs": [{"id": 1, "bq": 100, "ch": 110, "delta": -10}],
            },
            "descendants": {"match": 5, "differ": 0, "diffs": []},
            "created_at_i": {"match": 5, "differ": 0, "diffs": []},
        },
        "top_score_diffs": [],
        "bq_only_stories": [],
        "ch_only_stories": [],
        "errors": {},
    }
    buf = io.StringIO()
    print_summary(report, buf)
    output = buf.getvalue()
    assert "5/5" in output
    assert "4/5" in output
