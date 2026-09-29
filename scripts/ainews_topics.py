#!/usr/bin/env python3
"""Prototype: split AINews issues into one card per topic.

AINews (smol.ai) now ships as ``[AINews]`` posts in the Latent Space feed.
Each issue is one long page: an "AI Twitter Recap" of 4-7 topics (bold
paragraph headings, sometimes a "Top Story:" with h2 subsections), then an
"AI Reddit Recap". This splits the Twitter recap into topic cards and
fetches the text of every tweet a topic links to from ``api.fxtwitter.com``
(a public mirror, so nothing calls x.com). The Reddit recap and the
"Top tweets" grab-bag are skipped.

Read-only: prints cards, optionally writes JSONL. Nothing touches the DB.

Usage:
    uv run python scripts/ainews_topics.py --issues 2
    uv run python scripts/ainews_topics.py --issues 5 --jsonl /tmp/ainews.jsonl
    uv run python scripts/ainews_topics.py --no-tweets
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import feedparser
import httpx
from bs4 import BeautifulSoup, Tag

FEED_URL = "https://www.latent.space/feed"
FXTWITTER_URL = "https://api.fxtwitter.com/status/{id}"
USER_AGENT = "hn-rerank-ainews-prototype/0.1"
DEFAULT_CACHE = Path.home() / ".cache" / "hn-rerank" / "fxtwitter.json"

TWEET_RE = re.compile(r"(?:x|twitter)\.com/(\w+)/status/(\d+)")
SKIP_TOPICS = re.compile(r"^top tweets", re.I)
TOP_STORY = re.compile(r"^top story:\s*", re.I)


@dataclass(frozen=True)
class Tweet:
    id: str
    author: str
    text: str
    likes: int
    replies: int
    created: int
    quote_text: str = ""


@dataclass
class TopicCard:
    issue_title: str
    issue_url: str
    published: int
    title: str
    body: str
    tweet_ids: list[str] = field(default_factory=list)
    tweets: list[Tweet] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable id: same topic reposted in another issue collapses."""
        return hashlib.sha1(f"{self.title}\n{self.body}".encode()).hexdigest()[:12]

    @property
    def url(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")[:60]
        return f"{self.issue_url}#{slug}"

    @property
    def score(self) -> int:
        return sum(t.likes for t in self.tweets)

    def text_content(self) -> str:
        """What would be embedded and summarized."""
        parts = [self.title, self.body]
        for t in self.tweets:
            quote = f"\n  > {t.quote_text}" if t.quote_text else ""
            parts.append(f"@{t.author}: {t.text}{quote}")
        return "\n\n".join(parts)


def _is_heading_para(el: Tag) -> bool:
    """A short <p> that is nothing but bold text: AINews's topic heading.

    Bold paragraphs ending in ":" are sub-labels ("Facts vs. opinions:"),
    and ones ending in "." are a top story's bold lead sentence.
    """
    if el.name != "p":
        return False
    bold = el.find(["strong", "b"])
    if bold is None:
        return False
    text = el.get_text(" ", strip=True)
    return (
        text == bold.get_text(" ", strip=True)
        and len(text) <= 160
        and not text.endswith((":", "."))
    )


def _tweet_ids(el: Tag) -> list[str]:
    ids: list[str] = []
    for a in el.find_all("a", href=True):
        m = TWEET_RE.search(str(a["href"]))
        if m and m.group(2) not in ids:
            ids.append(m.group(2))
    return ids


def split_issue(
    html: str, *, issue_title: str, issue_url: str, published: int
) -> list[TopicCard]:
    """Split one issue's Twitter recap into topic cards."""
    soup = BeautifulSoup(html, "html.parser")
    cards: list[TopicCard] = []
    in_recap = False
    current: TopicCard | None = None
    chunks: list[str] = []

    def flush() -> None:
        if current is not None and chunks and not SKIP_TOPICS.match(current.title):
            current.body = "\n".join(chunks)
            cards.append(current)

    for el in soup.children:
        if not isinstance(el, Tag):
            continue
        if el.name == "h1":
            if in_recap:
                break  # next top-level section (Reddit recap): done
            in_recap = "twitter" in el.get_text().lower()
            continue
        if not in_recap:
            continue
        text = el.get_text(" ", strip=True)
        if _is_heading_para(el):
            flush()
            current = TopicCard(
                issue_title=issue_title,
                issue_url=issue_url,
                published=published,
                title=TOP_STORY.sub("", text),
                body="",
            )
            chunks = []
            continue
        if current is None or not text:
            continue
        chunks.append(f"## {text}" if el.name in ("h2", "h3") else text)
        for tid in _tweet_ids(el):
            if tid not in current.tweet_ids:
                current.tweet_ids.append(tid)
    flush()
    return cards


def parse_tweet(tid: str, payload: dict[str, Any]) -> Tweet | None:
    """Normalize an fxtwitter response; None when the tweet is gone."""
    t = payload.get("tweet")
    if payload.get("code") != 200 or not isinstance(t, dict):
        return None
    quote = t.get("quote")
    return Tweet(
        id=tid,
        author=str((t.get("author") or {}).get("screen_name", "")),
        text=str(t.get("text", "")),
        likes=int(t.get("likes") or 0),
        replies=int(t.get("replies") or 0),
        created=int(t.get("created_timestamp") or 0),
        quote_text=str(quote.get("text", "")) if isinstance(quote, dict) else "",
    )


async def fetch_tweets(
    ids: list[str], cache_path: Path, *, concurrency: int = 4
) -> dict[str, Tweet]:
    cache: dict[str, dict[str, Any] | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    missing = [i for i in dict.fromkeys(ids) if i not in cache]
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, tid: str) -> None:
        async with sem:
            try:
                r = await client.get(FXTWITTER_URL.format(id=tid))
                tw = parse_tweet(tid, r.json())
            except (httpx.HTTPError, ValueError) as exc:
                print(f"  tweet {tid}: {exc!r}", file=sys.stderr)
                return  # not cached: retried next run
            cache[tid] = asdict(tw) if tw else None

    if missing:
        async with httpx.AsyncClient(
            timeout=20, headers={"User-Agent": USER_AGENT}
        ) as client:
            await asyncio.gather(*(one(client, i) for i in missing))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return {i: Tweet(**v) for i, v in cache.items() if i in ids and v}


def load_issues(feed_url: str, limit: int) -> list[TopicCard]:
    feed = feedparser.parse(feed_url, agent=USER_AGENT)
    seen: set[str] = set()
    cards: list[TopicCard] = []
    issues = [e for e in feed.entries if str(e.get("title", "")).startswith("[AINews]")]
    for e in issues[:limit]:
        content = e.get("content") or []
        if not content:
            continue
        parsed = e.get("published_parsed")
        for card in split_issue(
            str(content[0].get("value", "")),
            issue_title=str(e.title).removeprefix("[AINews]").strip(),
            issue_url=str(e.link),
            published=calendar.timegm(parsed) if parsed else 0,
        ):
            if card.key not in seen:  # reposted issues repeat topics
                seen.add(card.key)
                cards.append(card)
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype: split AINews issues into per-topic cards."
    )
    parser.add_argument("--feed", default=FEED_URL)
    parser.add_argument("--issues", type=int, default=2, help="newest N issues")
    parser.add_argument("--no-tweets", action="store_true", help="skip fxtwitter")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--jsonl", type=Path, help="write cards here")
    parser.add_argument("--show", type=int, default=0, help="print N full cards")
    args = parser.parse_args()

    cards = load_issues(args.feed, args.issues)
    if not args.no_tweets:
        ids = [i for c in cards for i in c.tweet_ids]
        tweets = asyncio.run(fetch_tweets(ids, args.cache))
        for c in cards:
            c.tweets = [tweets[i] for i in c.tweet_ids if i in tweets]

    for c in cards:
        print(
            f"{c.score:>7}♥ {len(c.tweets):>3}/{len(c.tweet_ids):<3} tw "
            f"{len(c.text_content()):>6}ch  {c.title[:90]}"
        )
    for c in cards[: args.show]:
        print(f"\n===== {c.title}\n{c.url}\n\n{c.text_content()[:3000]}")
    if args.jsonl:
        with args.jsonl.open("w") as fh:
            for c in cards:
                row = {
                    "key": c.key,
                    "title": c.title,
                    "url": c.url,
                    "time": c.published,
                    "score": c.score,
                    "issue": c.issue_title,
                    "text_content": c.text_content(),
                }
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {len(cards)} cards to {args.jsonl}")


if __name__ == "__main__":
    main()
