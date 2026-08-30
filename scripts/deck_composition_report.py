#!/usr/bin/env python3
"""Report the HN / non-HN composition of the ranked deck at every stage.

Diagnoses "the deck is too short and looks HN-only" by printing, for one
user, how many candidates survive each stage between the raw SQL pool and
the final rendered deck: per-leg pool size, per-combo pool/primary/badge
counts, and pre/post-dedup non-HN survivor counts. These are the counters
added to ``RankTrace`` (see ``pool_hn``/``pool_rss``/``pool_archive``,
``pool_rss_oldest_age_h``, ``combo_pool_*``/``combo_primary_*``/
``combo_badges_*``, ``deck_nonhn_pre_dedup``/``deck_nonhn_post_dedup``/
``deck_nonhn_final`` in ``pipeline/ranking.py`` and ``pipeline/__init__.py``).

Two modes:

* Default: re-runs ``fast_rerank_for_user`` for ``--user-id`` against a
  scratch copy of the DB (never the live file — see AGENTS.md's "never
  destructively modify the local database"; the embedding-cache write path
  means this needs a writable copy) and prints the fresh trace.
* ``--from-rank-perf``: reads the newest ``rank_perf`` row(s) instead of
  re-ranking, so the live service can be inspected read-only with no
  re-rank cost. Works directly against the configured DB (read-only).

Usage:
    uv run python scripts/deck_composition_report.py --user-id 1
    uv run python scripts/deck_composition_report.py --from-rank-perf --user-id 1 -n 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config, Embedder, RankTrace, fast_rerank_for_user  # noqa: E402

# Combo ids as produced by pipeline.ranking._assemble_combo_deck's COMBO_DEFS.
_COMBO_IDS = ("recent_hn", "recent_nonhn", "archive_hn", "archive_nonhn")


def _print_fields(fields: Mapping[str, object]) -> None:
    print(f"  pool_hn              = {fields.get('pool_hn', '?')}")
    print(f"  pool_archive         = {fields.get('pool_archive', '?')}")
    print(f"  pool_rss             = {fields.get('pool_rss', '?')}")
    if "pool_rss_oldest_age_h" in fields:
        print(
            f"  pool_rss_oldest_age_h= {fields['pool_rss_oldest_age_h']}"
            "  (hours of the configured `days` window the RSS leg's"
            " LIMIT actually reached)"
        )
    print()
    print(f"  {'combo':<16}{'pool':>8}{'primary':>10}{'badges':>10}")
    for combo_id in _COMBO_IDS:
        pool = fields.get(f"combo_pool_{combo_id}", "-")
        primary = fields.get(f"combo_primary_{combo_id}", "-")
        badges = fields.get(f"combo_badges_{combo_id}", "-")
        print(f"  {combo_id:<16}{pool!s:>8}{primary!s:>10}{badges!s:>10}")
    print()
    print(f"  deck_nonhn_pre_dedup  = {fields.get('deck_nonhn_pre_dedup', '?')}")
    print(f"  deck_nonhn_post_dedup = {fields.get('deck_nonhn_post_dedup', '?')}")
    print(f"  deck_nonhn_final      = {fields.get('deck_nonhn_final', '?')}")
    print(f"  candidates (total)    = {fields.get('candidates', '?')}")
    print(f"  stories (final total) = {fields.get('stories', fields.get('n', '?'))}")


def _from_rank_perf(config: Config, user_id: int, n: int) -> None:
    db = Database(config.db_path, read_only=True)
    try:
        rows = db.execute(
            "SELECT recorded_at, version, stories, fields_json FROM rank_perf "
            "WHERE user_id = ? ORDER BY recorded_at DESC LIMIT ?",
            (user_id, n),
        )
    finally:
        db.close()

    if not rows:
        print(f"No rank_perf rows found for user_id={user_id}.")
        return

    for recorded_at, version, stories, fields_json in rows:
        fields = json.loads(fields_json)
        fields.setdefault("stories", stories)
        print(f"=== rank_perf version={version} recorded_at={recorded_at:.0f} ===")
        _print_fields(fields)
        print()


def _rerank_fresh(config: Config, user_id: int) -> None:
    # Never open the live DB read-write: fast_rerank_for_user's embedding
    # cache path (get_or_compute_embeddings) can write missing rows, and
    # AGENTS.md forbids any write path against the production file. Work
    # on a scratch copy instead.
    with tempfile.TemporaryDirectory(prefix="deck_composition_report_") as tmpdir:
        scratch_db = Path(tmpdir) / "scratch.db"
        shutil.copy2(config.db_path, scratch_db)

        db = Database(str(scratch_db), read_only=False)
        try:
            embedder = Embedder(
                config.onnx_model_dir,
                model_version=config.embedding_model_version,
                max_tokens=config.embedding_max_tokens,
                batch_size=config.embedding_batch_size,
                ort_variant=config.embedding_ort_variant,
            )
            trace = RankTrace()
            ranked = fast_rerank_for_user(db, config, embedder, user_id, trace=trace)
            fields = trace.to_log_fields()
            fields["stories"] = len(ranked)
            print(f"=== fresh rerank (scratch copy) user_id={user_id} ===")
            _print_fields(fields)
        finally:
            db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report per-stage HN/non-HN deck composition."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument(
        "--from-rank-perf",
        action="store_true",
        help="Read the newest rank_perf row(s) instead of re-ranking.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        help="With --from-rank-perf, how many recent rows to show (default 1).",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.from_rank_perf:
        _from_rank_perf(config, args.user_id, args.n)
    else:
        _rerank_fresh(config, args.user_id)


if __name__ == "__main__":
    main()
