#!/usr/bin/env python3
"""Report p50/p95/max per rank_perf stage, split by model_cache.

Reads the rank_perf table O2 writes on every warm (server.py's
_run_warm_attempt) and prints a compact stage-timing breakdown. This is
the before/after instrument for the P1/P2/P3 performance work in
fable_plan.md — run before and after each change and diff the numbers.

Usage:
    uv run python scripts/perf_report.py --window-days 7
    uv run python scripts/perf_report.py --window-days 1 --user 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Database  # noqa: E402
from pipeline import Config  # noqa: E402


@dataclass(frozen=True)
class StageStat:
    stage: str
    n: int
    p50: float
    p95: float
    max: float


def parse_stage_timings(fields_json: str, html_ms: float) -> dict[str, float]:
    """Every `*_ms` key logged on the trace, plus html_ms (timed outside
    the trace in server.py, so it never appears in fields_json itself)."""
    fields = json.loads(fields_json)
    stages = {k: float(v) for k, v in fields.items() if k.endswith("_ms")}
    stages.setdefault("html_ms", html_ms)
    return stages


def _stat(stage: str, values: list[float]) -> StageStat:
    n = len(values)
    if n == 1:
        (v,) = values
        return StageStat(stage, n, v, v, v)
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return StageStat(stage, n, qs[49], qs[94], max(values))


def aggregate(
    rows: list[tuple[str, float, str]],
) -> dict[str, list[StageStat]]:
    """rows: (model_cache, html_ms, fields_json). Returns model_cache ->
    per-stage stats sorted by p95 descending (hot stage first)."""
    by_cache: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for model_cache, html_ms, fields_json in rows:
        for stage, ms in parse_stage_timings(fields_json, html_ms).items():
            by_cache[model_cache][stage].append(ms)

    result: dict[str, list[StageStat]] = {}
    for model_cache, stage_values in by_cache.items():
        stats = [_stat(stage, values) for stage, values in stage_values.items()]
        stats.sort(key=lambda s: s.p95, reverse=True)
        result[model_cache] = stats
    return result


def format_report(
    by_cache: dict[str, list[StageStat]], total_warms: int, window_days: float
) -> str:
    lines = [f"rank_perf report — last {window_days:g} day(s), {total_warms} warm(s)", ""]
    for model_cache in sorted(by_cache):
        stats = by_cache[model_cache]
        n = stats[0].n if stats else 0
        lines.append(f"model_cache={model_cache!r} ({n} warms)")
        lines.append(f"  {'stage':<28}{'n':>6}{'p50':>10}{'p95':>10}{'max':>10}")
        for s in stats:
            lines.append(
                f"  {s.stage:<28}{s.n:>6}{s.p50:>10.1f}{s.p95:>10.1f}{s.max:>10.1f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report rank_perf p50/p95/max per stage."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--window-days", type=float, default=7)
    parser.add_argument("--user", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    db = Database(config.db_path, read_only=True)
    try:
        cutoff = time.time() - args.window_days * 86400
        sql = "SELECT model_cache, html_ms, fields_json FROM rank_perf WHERE recorded_at >= ?"
        params: tuple[float | int, ...] = (cutoff,)
        if args.user is not None:
            sql += " AND user_id = ?"
            params += (args.user,)
        rows = db.execute(sql, params)
    finally:
        db.close()

    if not rows:
        print(f"No rank_perf rows in the last {args.window_days:g} days.")
        return

    by_cache = aggregate(rows)
    print(format_report(by_cache, len(rows), args.window_days))


if __name__ == "__main__":
    main()
