#!/usr/bin/env python3
"""Preview the AINews source: the per-topic cards regen would store.

Uses ``pipeline.ainews`` (the live source) without touching the DB: fetches
the feed, splits every ``[AINews]`` issue into topic cards, optionally
fetches their tweets from fxtwitter, and prints one line per card.

Usage:
    uv run python scripts/ainews_topics.py
    uv run python scripts/ainews_topics.py --no-tweets
    uv run python scripts/ainews_topics.py --show 1 --jsonl /tmp/ainews.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config  # noqa: E402
from pipeline.ainews import (  # noqa: E402
    USER_AGENT,
    Tweet,
    fetch_tweets,
    format_tweets,
    parse_feed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview AINews per-topic cards (read-only)."
    )
    parser.add_argument("--feed", default=Config().ainews_feed_url)
    parser.add_argument("--no-tweets", action="store_true", help="skip fxtwitter")
    parser.add_argument("--jsonl", type=Path, help="write cards here")
    parser.add_argument("--show", type=int, default=0, help="print N full cards")
    args = parser.parse_args()

    resp = httpx.get(
        args.feed, headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True
    )
    resp.raise_for_status()
    cards = parse_feed(resp.text, cutoff=0)
    tweets: dict[str, Tweet] = {}
    if not args.no_tweets:
        tweets = asyncio.run(fetch_tweets([i for c in cards for i in c.tweet_ids]))

    rows = []
    for c in cards:
        card_tweets = [tweets[i] for i in c.tweet_ids if i in tweets]
        comments = format_tweets(card_tweets)
        likes = sum(t.likes for t in card_tweets)
        print(
            f"{likes:>7}♥ {len(card_tweets):>3}/{len(c.tweet_ids):<3} tw "
            f"{len(c.body) + len(comments):>6}ch  {c.title[:90]}"
        )
        rows.append(
            {
                "title": c.title,
                "url": c.url,
                "time": c.published,
                "likes": likes,
                "issue": c.issue_title,
                "self_text": c.body,
                "top_comments": comments,
            }
        )
    for c in cards[: args.show]:
        card_tweets = [tweets[i] for i in c.tweet_ids if i in tweets]
        print(f"\n===== {c.title}\n{c.url}\n\n{c.body[:1500]}")
        print(f"\n--- tweets\n{format_tweets(card_tweets)[:1500]}")
    if args.jsonl:
        with args.jsonl.open("w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"wrote {len(rows)} cards to {args.jsonl}")


if __name__ == "__main__":
    main()
