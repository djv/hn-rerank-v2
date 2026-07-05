from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            rows.append(obj)
    return rows


def _normalize(s: str) -> str:
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_idx(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(r["id"]): r for r in rows if r.get("id")}


def compute_diff(
    bq: list[dict[str, Any]],
    ch: list[dict[str, Any]],
    *,
    score_delta: int = 5,
    descendants_delta: int = 5,
) -> dict[str, Any]:
    bq_idx = build_idx(bq)
    ch_idx = build_idx(ch)

    bq_ids = set(bq_idx)
    ch_ids = set(ch_idx)
    intersection = bq_ids & ch_ids
    bq_only = bq_ids - ch_ids
    ch_only = ch_ids - bq_ids

    field_agreement: dict[str, Any] = {}
    top_score_diffs: list[dict[str, Any]] = []

    for field, delta in [
        ("score", score_delta),
        ("descendants", descendants_delta),
        ("created_at_i", 0),
        ("title", 0),
        ("url", 0),
    ]:
        match_count = 0
        differ_count = 0
        diffs: list[dict[str, Any]] = []
        for sid in sorted(intersection):
            bv = bq_idx[sid].get(field)
            cv = ch_idx[sid].get(field)
            if field in ("title",):
                bn, cn = _normalize(str(bv or "")), _normalize(str(cv or ""))
                if bn == cn:
                    match_count += 1
                else:
                    differ_count += 1
                    diffs.append({"id": sid, "bq": bv, "ch": cv})
            elif field == "url":
                bu = (bv or "").rstrip("/")
                cu = (cv or "").rstrip("/")
                if bu == cu:
                    match_count += 1
                else:
                    differ_count += 1
                    diffs.append({"id": sid, "bq": bv, "ch": cv})
            else:
                bv_i = int(bv) if bv is not None else 0
                cv_i = int(cv) if cv is not None else 0
                if abs(bv_i - cv_i) <= delta:
                    match_count += 1
                else:
                    differ_count += 1
                    diffs.append(
                        {"id": sid, "bq": bv_i, "ch": cv_i, "delta": bv_i - cv_i}
                    )

        field_agreement[field] = {
            "match": match_count,
            "differ": differ_count,
            "diffs": diffs,
        }
        if field == "score" and diffs:
            top_score_diffs = sorted(
                diffs, key=lambda d: abs(d["delta"]), reverse=True
            )[:10]

    bq_only_stories = sorted(
        (
            {
                "id": sid,
                "title": bq_idx[sid].get("title", ""),
                "score": bq_idx[sid].get("score", 0),
            }
            for sid in bq_only
        ),
        key=lambda x: int(x["score"] or 0),
        reverse=True,
    )
    ch_only_stories = sorted(
        (
            {
                "id": sid,
                "title": ch_idx[sid].get("title", ""),
                "score": ch_idx[sid].get("score", 0),
            }
            for sid in ch_only
        ),
        key=lambda x: int(x["score"] or 0),
        reverse=True,
    )

    return {
        "counts": {
            "bq": len(bq),
            "ch": len(ch),
            "intersection": len(intersection),
            "bq_only": len(bq_only),
            "ch_only": len(ch_only),
        },
        "field_agreement": field_agreement,
        "top_score_diffs": top_score_diffs,
        "bq_only_stories": bq_only_stories,
        "ch_only_stories": ch_only_stories,
    }


def print_summary(report: dict[str, Any], out: Any = sys.stdout) -> None:
    c = report["counts"]
    out.write("== BQ vs ClickHouse smoke test ==\n")
    out.write(f"  BQ rows: {c['bq']}\n")
    out.write(f"  CH rows: {c['ch']}\n")
    out.write(f"  Intersection: {c['intersection']}\n")
    out.write(f"  BQ-only: {c['bq_only']} stories\n")
    out.write(f"  CH-only: {c['ch_only']} stories\n\n")

    out.write("  Field agreement (on intersection):\n")
    for field, info in report["field_agreement"].items():
        total = info["match"] + info["differ"]
        if total == 0:
            continue
        pct = info["match"] / total * 100
        parts = [f"    {field:16s} {info['match']:3d}/{total} match ({pct:5.1f}%)"]
        if info["differ"]:
            max_d = (
                max(abs(d.get("delta", 0)) for d in info["diffs"])
                if info["diffs"]
                else 0
            )
            parts.append(f"  {info['differ']} differ")
            if max_d:
                parts.append(f"  max delta={max_d}")
        out.write("".join(parts) + "\n")

    if report["top_score_diffs"]:
        out.write("\n  Top score deltas:\n")
        for d in report["top_score_diffs"][:5]:
            out.write(
                f"    id={d['id']} bq={d['bq']} ch={d['ch']} delta={d['delta']}\n"
            )

    if report.get("errors"):
        for src, err in report["errors"].items():
            if err:
                out.write(f"\n  {src}: {err}\n")


def fetch_bq(
    limit: int | None, months: int, min_score: int
) -> tuple[list[dict[str, Any]], str | None]:
    from scripts.seed_hn_from_bq import run_bq_query

    try:
        rows = run_bq_query(limit=limit, months=months, min_score=min_score)
        return rows, None
    except Exception as e:
        return [], str(e)


def fetch_ch(
    limit: int | None, months: int, min_score: int
) -> tuple[list[dict[str, Any]], str | None]:
    from scripts.seed_hn_from_clickhouse import run_ch_query

    try:
        rows = run_ch_query(limit=limit, months=months, min_score=min_score)
        return rows, None
    except Exception as e:
        return [], str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BQ vs ClickHouse seeder output for the same query parameters."
    )
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--min-score", type=int, default=100)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--bq-file",
        type=Path,
        default=None,
        help="Load BQ rows from prior dry-run JSONL",
    )
    parser.add_argument(
        "--ch-file",
        type=Path,
        default=None,
        help="Load CH rows from prior dry-run JSONL",
    )
    parser.add_argument("--output", type=Path, default=None, help="Report output path")
    parser.add_argument(
        "--skip-bq", action="store_true", help="Skip live BQ query (requires --ch-file)"
    )
    parser.add_argument("--score-delta", type=int, default=5)
    parser.add_argument("--descendants-delta", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    errors: dict[str, str | None] = {}

    bq_rows: list[dict[str, Any]] = []
    if args.bq_file:
        bq_rows = load_jsonl(args.bq_file)
        logger.info("Loaded %s BQ rows from %s", len(bq_rows), args.bq_file)
    elif not args.skip_bq:
        logger.info(
            "Fetching BQ rows (months=%s, min_score=%s, limit=%s)...",
            args.months,
            args.min_score,
            args.limit,
        )
        bq_rows, err = fetch_bq(args.limit, args.months, args.min_score)
        errors["bq"] = err
        if err:
            logger.warning("BQ query failed: %s", err)
        else:
            logger.info("Fetched %s BQ rows", len(bq_rows))
    else:
        errors["bq"] = "skipped (--skip-bq)"

    ch_rows: list[dict[str, Any]] = []
    if args.ch_file:
        ch_rows = load_jsonl(args.ch_file)
        logger.info("Loaded %s CH rows from %s", len(ch_rows), args.ch_file)
    else:
        logger.info(
            "Fetching CH rows (months=%s, min_score=%s, limit=%s)...",
            args.months,
            args.min_score,
            args.limit,
        )
        ch_rows, err = fetch_ch(args.limit, args.months, args.min_score)
        errors["ch"] = err
        if err:
            logger.warning("CH query failed: %s", err)
        else:
            logger.info("Fetched %s CH rows", len(ch_rows))

    if not bq_rows and not args.skip_bq and not args.bq_file:
        logger.error(
            "No BQ data available and --skip-bq not set. Use --bq-file or --skip-bq."
        )
        sys.exit(1)

    report = compute_diff(
        bq_rows,
        ch_rows,
        score_delta=args.score_delta,
        descendants_delta=args.descendants_delta,
    )
    report["errors"] = errors
    report["params"] = {
        "months": args.months,
        "min_score": args.min_score,
        "limit": args.limit,
        "score_delta": args.score_delta,
        "descendants_delta": args.descendants_delta,
    }

    print_summary(report)

    out_path = args.output or Path(f"smoke_report_{int(time.time())}.json")
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written to %s", out_path)


if __name__ == "__main__":
    main()
