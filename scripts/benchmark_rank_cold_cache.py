from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database, Story  # noqa: E402
from pipeline import (  # noqa: E402
    BQ_ARCHIVE_CANDIDATE_LIMIT,
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_CANDIDATE_LIMIT,
    CH_ARCHIVE_SOURCE,
    Config,
    Embedder,
    RankTrace,
    _MODEL_CACHE,
    _MODEL_CACHE_LOCK,
    fast_rerank_for_user,
    is_summarizable,
    story_embedding_text,
)

EMBEDDING_MODEL_VERSION = "all-MiniLM-L6-v2|mean|norm|256"


def _clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _heaviest_user_id(db: Database) -> int:
    rows = db.execute(
        """
        SELECT user_id, COUNT(*) AS n
        FROM feedback
        GROUP BY user_id
        ORDER BY n DESC
        LIMIT 1
        """
    )
    if not rows:
        raise SystemExit("No feedback rows found; nothing to benchmark.")
    return int(rows[0][0])


def _candidate_stories(db: Database, config: Config, user_id: int) -> list[Story]:
    now_ts = int(time.time())
    cutoff_ts = now_ts - (config.days * 86400)
    recent_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, "
        "       CASE WHEN self_text != '' OR top_comments != '' OR article_body != '' "
        "            THEN '1' ELSE '' END AS self_text, "
        "       '' AS top_comments, '' AS article_body "
        "FROM stories WHERE time >= ? AND source NOT IN (?, ?) "
        "AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?)",
        (cutoff_ts, BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE, user_id),
    )
    archive_rows = db.execute(
        "SELECT id, title, url, score, time, text_content, source, comment_count, "
        "       discussion_url, comment_count_at_fetch, "
        "       CASE WHEN self_text != '' OR top_comments != '' OR article_body != '' "
        "            THEN '1' ELSE '' END AS self_text, "
        "       '' AS top_comments, '' AS article_body "
        "FROM stories INDEXED BY idx_stories_archive_score_time "
        f"WHERE source IN ('{BQ_ARCHIVE_SOURCE}', '{CH_ARCHIVE_SOURCE}') "
        "AND text_content != '' "
        "AND id NOT IN (SELECT story_id FROM feedback WHERE user_id = ?) "
        "ORDER BY score DESC, time DESC LIMIT ?",
        (
            user_id,
            BQ_ARCHIVE_CANDIDATE_LIMIT + CH_ARCHIVE_CANDIDATE_LIMIT,
        ),
    )
    stories = [Database._row_to_story(row) for row in recent_rows + archive_rows]
    return [s for s in stories if is_summarizable(s)]


def _missing_embedding_count(db: Database, stories: list[Story]) -> int:
    hashes = {
        story.id: hashlib.sha256(
            story_embedding_text(story).encode("utf-8")
        ).hexdigest()
        for story in stories
    }
    cached = db.get_embeddings_batch(
        [story.id for story in stories], EMBEDDING_MODEL_VERSION, hashes
    )
    return len(stories) - len(cached)


def _preflight_read_only_embeddings(
    db: Database, config: Config, user_id: int
) -> dict[str, int]:
    candidates = _candidate_stories(db, config, user_id)
    feedback_stories, _labels, _vote_times = db.get_feedback_for_training(user_id)
    candidate_missing = _missing_embedding_count(db, candidates)
    feedback_missing = _missing_embedding_count(db, feedback_stories)
    return {
        "candidates": len(candidates),
        "feedback": len(feedback_stories),
        "candidate_missing_embeddings": candidate_missing,
        "feedback_missing_embeddings": feedback_missing,
    }


def _run_once(db: Database, config: Config, embedder: Embedder, user_id: int) -> dict:
    trace = RankTrace()
    with trace.stage("rank_total"):
        ranked = fast_rerank_for_user(db, config, embedder, user_id, trace=trace)
    fields = trace.to_log_fields()
    fields["stories"] = len(ranked)
    return fields


def _summarize(values: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "rank_total_ms",
        "candidate_sql_ms",
        "candidate_embedding_ms",
        "feedback_embedding_ms",
        "svm_candidate_feature_prep_ms",
        "svm_training_feature_prep_ms",
        "svm_candidate_scale_ms",
        "svm_fit_ms",
        "decision_ms",
        "tier2_ms",
        "badge_similarity_ms",
        "dedup_ms",
    ]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        samples = [float(row.get(key, 0.0)) for row in values]
        if not samples:
            continue
        sorted_samples = sorted(samples)
        p95_idx = min(len(sorted_samples) - 1, int(round((len(samples) - 1) * 0.95)))
        summary[key] = {
            "min": round(min(samples), 1),
            "p50": round(statistics.median(samples), 1),
            "p95": round(sorted_samples[p95_idx], 1),
            "max": round(max(samples), 1),
        }
    return summary


def _print_table(label: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{label}")
    print("stage                         min      p50      p95      max")
    for key, stats in summary.items():
        print(
            f"{key:<28} {stats['min']:>7.1f} {stats['p50']:>8.1f} "
            f"{stats['p95']:>8.1f} {stats['max']:>8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cold vs warm personalized ranking for a real user."
    )
    parser.add_argument("--db", default="hn_rewrite.db")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--cold-runs", type=int, default=1)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument(
        "--hn-candidate-limit",
        type=int,
        default=None,
        help="Override Config.recent_candidate_hn_limit for this run.",
    )
    parser.add_argument(
        "--rss-candidate-limit",
        type=int,
        default=None,
        help="Override Config.recent_candidate_rss_limit for this run.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Override Config.embedding_batch_size for this run.",
    )
    parser.add_argument(
        "--embedding-ort-variant",
        choices=[
            "current",
            "spin_off",
            "spin_off_graph_all",
            "spin_off_auto_threads",
        ],
        default=None,
        help="Override Config.embedding_ort_variant for this run.",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.hn_candidate_limit is not None:
        config = replace(config, recent_candidate_hn_limit=args.hn_candidate_limit)
    if args.rss_candidate_limit is not None:
        config = replace(config, recent_candidate_rss_limit=args.rss_candidate_limit)
    if args.embedding_batch_size is not None:
        config = replace(config, embedding_batch_size=args.embedding_batch_size)
    if args.embedding_ort_variant is not None:
        config = replace(
            config,
            embedding_ort_variant=cast(Any, args.embedding_ort_variant),
        )
    db = Database(args.db, read_only=not args.allow_writes)
    try:
        user_id = args.user_id if args.user_id is not None else _heaviest_user_id(db)
        preflight = _preflight_read_only_embeddings(db, config, user_id)
        if args.preflight_only:
            result = {
                "user_id": user_id,
                "preflight": preflight,
                "cold_runs": [],
                "warm_runs": [],
                "cold_summary": {},
                "warm_summary": {},
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        if not args.allow_writes and (
            preflight["candidate_missing_embeddings"] > 0
            or preflight["feedback_missing_embeddings"] > 0
        ):
            raise SystemExit(
                "Read-only benchmark would need to write missing embeddings: "
                f"{preflight}. Run `uv run python scripts/embed_remaining.py` first, "
                "or pass --allow-writes for an explicit write-enabled benchmark."
            )

        embedder = Embedder(
            config.onnx_model_dir,
            batch_size=config.embedding_batch_size,
            ort_variant=config.embedding_ort_variant,
        )
        cold: list[dict[str, Any]] = []
        warm: list[dict[str, Any]] = []

        for _ in range(max(args.cold_runs, 0)):
            _clear_model_cache()
            cold.append(_run_once(db, config, embedder, user_id))
        for _ in range(max(args.warm_runs, 0)):
            warm.append(_run_once(db, config, embedder, user_id))

        result: dict[str, Any] = {
            "user_id": user_id,
            "preflight": preflight,
            "cold_runs": cold,
            "warm_runs": warm,
            "cold_summary": _summarize(cold),
            "warm_summary": _summarize(warm),
        }
        if not args.json_only:
            print(
                f"user_id={user_id} candidates={preflight['candidates']} "
                f"feedback={preflight['feedback']}"
            )
            _print_table("cold", result["cold_summary"])
            _print_table("warm", result["warm_summary"])
            print("\njson")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        db.close()


if __name__ == "__main__":
    main()
