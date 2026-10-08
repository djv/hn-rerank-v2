#!/usr/bin/env python3
"""Compare Reddit RSS top feeds with their Arctic Shift replacement.

For each Reddit feed in the config (or each ``--feed``), fetches the RSS
top list and the list that ``reddit_source = "arctic_shift"`` would store,
then reports how many stories match by story id (keyed by permalink).
Arctic Shift scores posts only after ~36 h, so posts younger than that
are expected among the RSS-only ones. Run while RSS still works (until
2026-11-13), from the VPS: Reddit throttles the laptop's RSS requests.

Reads no database. RSS requests are spaced by ``--rss-spacing`` so the
live service's Reddit budget on the same IP is not starved.

    uv run python scripts/compare_reddit_sources.py
    uv run python scripts/compare_reddit_sources.py --feed URL --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import Story
from pipeline.config import Config
from pipeline.enrichment import (
    _fetch_and_parse_feed,
    _fetch_arctic_topfeed,
    reddit_top_query,
)

YOUNG_POST_HOURS = 36.0


@dataclass(frozen=True)
class OnlyIn:
    title: str
    url: str
    age_hours: float


@dataclass(frozen=True)
class FeedComparison:
    feed: str
    rss_count: int
    arctic_count: int
    shared: int
    rss_only: list[OnlyIn]
    arctic_only: list[OnlyIn]

    @property
    def rss_only_young(self) -> int:
        return sum(item.age_hours < YOUNG_POST_HOURS for item in self.rss_only)


def compare_feed(
    feed: str, rss: Sequence[Story], arctic: Sequence[Story], now: float
) -> FeedComparison:
    """Match the two lists by story id; list each side's extras."""
    rss_ids = {s.id for s in rss}
    arctic_ids = {s.id for s in arctic}

    def only(stories: Sequence[Story], other: set[int]) -> list[OnlyIn]:
        return [
            OnlyIn(s.title, s.url or "", round((now - s.time) / 3600, 1))
            for s in stories
            if s.id not in other
        ]

    return FeedComparison(
        feed=feed,
        rss_count=len(rss_ids),
        arctic_count=len(arctic_ids),
        shared=len(rss_ids & arctic_ids),
        rss_only=only(rss, arctic_ids),
        arctic_only=only(arctic, rss_ids),
    )


async def _compare_all(
    feeds: Sequence[str],
    per_feed: int,
    days: int,
    rss_spacing: float,
    rss_retry_wait: float,
) -> list[FeedComparison]:
    now = time.time()
    cutoff = now - days * 86400
    results: list[FeedComparison] = []
    for index, feed in enumerate(feeds):
        if index:
            await asyncio.sleep(rss_spacing)
        rss = await _fetch_and_parse_feed(feed, per_feed, cutoff, now, set())
        if not rss and rss_retry_wait > 0:
            # Usually a 429; Reddit throttles bursts per IP.
            await asyncio.sleep(rss_retry_wait)
            rss = await _fetch_and_parse_feed(feed, per_feed, cutoff, now, set())
        arctic = await _fetch_arctic_topfeed(feed, per_feed, cutoff, now, set())
        result = compare_feed(feed, rss, arctic, now)
        results.append(result)
        print(
            f"{feed.split('/r/')[1].split('/')[0]:24s} rss={result.rss_count:3d} "
            f"arctic={result.arctic_count:3d} shared={result.shared:3d} "
            f"rss_only={len(result.rss_only):3d} "
            f"(<{YOUNG_POST_HOURS:.0f}h {result.rss_only_young:3d}) "
            f"arctic_only={len(result.arctic_only):3d}",
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--feed",
        action="append",
        default=[],
        help="Reddit /top feed URL (repeatable); default: every one in --config",
    )
    parser.add_argument(
        "--rss-spacing",
        type=float,
        default=15.0,
        help="seconds between feeds (default 15)",
    )
    parser.add_argument(
        "--rss-retry-wait",
        type=float,
        default=60.0,
        help="wait before one retry of an empty RSS answer (default 60; 0 = no retry)",
    )
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    config = Config.load(str(args.config))
    feeds = args.feed or [f for f in config.rss.feeds if reddit_top_query(f)]
    if not feeds:
        parser.error("no Reddit /top feeds to compare")
    results = asyncio.run(
        _compare_all(
            feeds,
            config.rss.per_feed_limit,
            config.days,
            args.rss_spacing,
            args.rss_retry_wait,
        )
    )
    total_rss = sum(r.rss_count for r in results)
    shared = sum(r.shared for r in results)
    young = sum(r.rss_only_young for r in results)
    print(
        f"total: shared {shared}/{total_rss} RSS stories; "
        f"{young} RSS-only posts are under {YOUNG_POST_HOURS:.0f} h old; "
        f"{sum(r.rss_count == 0 for r in results)} feeds returned no RSS"
    )
    if args.json:
        args.json.write_text(
            json.dumps(
                [asdict(r) | {"rss_only_young": r.rss_only_young} for r in results],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
