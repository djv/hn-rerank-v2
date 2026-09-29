"""AINews source: one card per topic of each AINews issue.

AINews (smol.ai) ships as ``[AINews]`` posts in the Latent Space feed. An
issue is one long page: an "AI Twitter Recap" of 4-7 topics (bold paragraph
headings; a "Top Story:" may carry h2 subsections), then an "AI Reddit
Recap". Each Twitter-recap topic becomes a story: the topic's bullets are
its ``self_text`` and the text of every tweet it links to, fetched from
``api.fxtwitter.com`` (a public mirror, so nothing calls x.com), is its
``top_comments``. The Reddit recap and the "Top tweets" grab-bag are
skipped; the generic RSS path drops the whole ``[AINews]`` entries.
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup, Tag

from database import Database, Story, StoryIdentityConflict

from .ranking import compose_story_text

AINEWS_SOURCE = "rss_ainews"
AINEWS_TITLE_PREFIX = "[AINews]"
FXTWITTER_URL = "https://api.fxtwitter.com/status/{id}"
USER_AGENT = "hn-rewrite/1.0 (+https://github.com/local/hn-rewrite)"

_TWEET_RE = re.compile(r"(?:x|twitter)\.com/(\w+)/status/(\d+)")
# Story ids hash this plus the topic's identity URL. The first layout
# (2026-09-29 04:06-, no namespace) stored that URL as the story URL; a
# negative id cannot change its URL, so the tweet/topic links needed new ids.
_ID_NAMESPACE = "ainews-v2:"
_SKIP_TOPICS = re.compile(r"^top tweets", re.I)
_TOP_STORY = re.compile(r"^top story:\s*", re.I)


@dataclass(frozen=True)
class Tweet:
    id: str
    author: str
    text: str
    likes: int
    replies: int
    created: int
    quote_text: str = ""
    # Expanded non-media links in the tweet (t.co resolved by fxtwitter).
    links: tuple[str, ...] = ()


@dataclass
class TopicCard:
    issue_title: str
    issue_url: str
    published: int
    title: str
    body: str
    tweet_ids: list[str] = field(default_factory=list)
    first_tweet_url: str = ""

    @property
    def key(self) -> str:
        """Content identity: a topic reposted in another issue collapses."""
        return hashlib.sha1(f"{self.title}\n{self.body}".encode()).hexdigest()[:12]

    @property
    def identity_url(self) -> str:
        """Stable per-topic key; never opened."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")[:60]
        return f"{self.issue_url}#{slug}"

    @property
    def topic_url(self) -> str:
        """The issue, scrolled to this topic's heading by a text fragment."""
        start = " ".join(self.title.split()[:8])
        # "-" is fragment syntax (prefix-/-suffix), so it must be escaped too.
        text = quote(start, safe="").replace("-", "%2D")
        return f"{self.issue_url}#:~:text={text}"

    @property
    def story_url(self) -> str:
        """Article link: the topic's first linked tweet, else the topic."""
        return self.first_tweet_url or self.topic_url


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
        if current is not None and chunks and not _SKIP_TOPICS.match(current.title):
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
                title=_TOP_STORY.sub("", text),
                body="",
            )
            chunks = []
            continue
        if current is None or not text:
            continue
        chunks.append(f"## {text}" if el.name in ("h2", "h3") else text)
        for a in el.find_all("a", href=True):
            m = _TWEET_RE.search(str(a["href"]))
            if m and m.group(2) not in current.tweet_ids:
                current.tweet_ids.append(m.group(2))
                if not current.first_tweet_url:
                    current.first_tweet_url = (
                        f"https://x.com/{m.group(1)}/status/{m.group(2)}"
                    )
    flush()
    return cards


def parse_feed(content: str, *, cutoff: float) -> list[TopicCard]:
    """Topic cards of every ``[AINews]`` entry newer than ``cutoff``.

    Newest issue first; a topic repeated by a reposted issue keeps its
    first (newest) copy.
    """
    seen: set[str] = set()
    cards: list[TopicCard] = []
    for entry in feedparser.parse(content).entries:
        title = str(entry.get("title", ""))
        link = entry.get("link")
        body = entry.get("content") or []
        if not title.startswith(AINEWS_TITLE_PREFIX) or not link or not body:
            continue
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        published = calendar.timegm(parsed) if parsed else 0
        if published < cutoff:
            continue
        for card in split_issue(
            str(body[0].get("value", "")),
            issue_title=title.removeprefix(AINEWS_TITLE_PREFIX).strip(),
            issue_url=str(link),
            published=published,
        ):
            if card.key not in seen:
                seen.add(card.key)
                cards.append(card)
    return cards


def parse_tweet(tid: str, payload: dict[str, Any]) -> Tweet | None:
    """Normalize an fxtwitter response; None when the tweet is gone."""
    t = payload.get("tweet")
    if payload.get("code") != 200 or not isinstance(t, dict):
        return None
    author = t.get("author")
    quote = t.get("quote")
    raw = t.get("raw_text")
    facets = raw.get("facets") if isinstance(raw, dict) else None
    links = tuple(
        str(f["replacement"])
        for f in facets or []
        if isinstance(f, dict) and f.get("type") == "url" and f.get("replacement")
    )
    return Tweet(
        id=tid,
        author=str(author.get("screen_name", "")) if isinstance(author, dict) else "",
        text=str(t.get("text", "")),
        likes=int(t.get("likes") or 0),
        replies=int(t.get("replies") or 0),
        created=int(t.get("created_timestamp") or 0),
        quote_text=str(quote.get("text", "")) if isinstance(quote, dict) else "",
        links=links,
    )


def tweet_id_from_url(url: str) -> str | None:
    """Status id of an x.com/twitter.com tweet URL, else None."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if host not in {"x.com", "twitter.com", "mobile.twitter.com", "mobile.x.com"}:
        return None
    m = _TWEET_RE.search(url)
    return m.group(2) if m else None


async def fetch_tweets(ids: list[str], *, concurrency: int = 4) -> dict[str, Tweet]:
    """Fetch tweets from fxtwitter; missing or failed ids are left out."""
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, Tweet] = {}

    async def one(client: httpx.AsyncClient, tid: str) -> None:
        async with sem:
            try:
                resp = await client.get(FXTWITTER_URL.format(id=tid))
                tweet = parse_tweet(tid, resp.json())
            except (httpx.HTTPError, ValueError) as exc:
                logging.warning("ainews: tweet %s fetch failed: %r", tid, exc)
                return
            if tweet is not None:
                out[tid] = tweet

    async with httpx.AsyncClient(
        timeout=15.0, headers={"User-Agent": USER_AGENT}
    ) as client:
        await asyncio.gather(*(one(client, tid) for tid in dict.fromkeys(ids)))
    return out


def format_tweets(tweets: list[Tweet]) -> str:
    parts = []
    for t in tweets:
        quote = f"\n> quoting: {t.quote_text}" if t.quote_text else ""
        parts.append(f"@{t.author} ({t.likes} likes): {t.text}{quote}")
    return "\n\n".join(parts)


def _story_id(card: TopicCard) -> int:
    """Negative synthetic id, like generic RSS rows, from the topic key."""
    key = f"{_ID_NAMESPACE}{card.identity_url}".encode()
    val = int.from_bytes(hashlib.md5(key).digest()[:4], "big")
    return -(val % (2**31))


async def fetch_ainews_stories(
    feed_url: str,
    days: int,
    exclude_urls: set[str],
    db: Database,
    *,
    max_tweets_per_run: int,
    now: float,
) -> list[Story]:
    """Fetch the feed, split issues into topic stories, upsert them.

    A story's URL (``o`` in the reader) is the topic's first linked tweet;
    its discussion URL (``c``) is the issue scrolled to the topic. Tweets
    are fetched once per story: a stored row that already has tweet
    text is reused as is. At most ``max_tweets_per_run`` tweets are fetched
    per call; cards past the cap are stored without tweets and filled in
    on a later run.
    """
    from http_fetch import fetch_with_urllib_fallback

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            status, content, _ = await fetch_with_urllib_fallback(
                client, feed_url, {"User-Agent": USER_AGENT}
            )
    except httpx.HTTPError as exc:
        logging.warning("ainews: feed fetch failed: %r", exc)
        return []
    if status != 200:
        logging.warning("ainews: feed returned HTTP %s", status)
        return []

    cards = [
        c
        for c in parse_feed(content, cutoff=now - days * 86400)
        if c.story_url not in exclude_urls and c.topic_url not in exclude_urls
    ]
    stored = {s.id: s for s in db.get_stories([_story_id(c) for c in cards])}
    to_fetch: list[str] = []
    for card in cards:
        prior = stored.get(_story_id(card))
        if prior is None or not prior.top_comments:
            to_fetch.extend(card.tweet_ids)
    tweets = await fetch_tweets(list(dict.fromkeys(to_fetch))[:max_tweets_per_run])

    stories: list[Story] = []
    for card in cards:
        sid = _story_id(card)
        prior = stored.get(sid)
        card_tweets = [tweets[i] for i in card.tweet_ids if i in tweets]
        if prior is not None and prior.top_comments and not card_tweets:
            stories.append(prior)
            continue
        comments = format_tweets(card_tweets)
        story = Story(
            id=sid,
            title=card.title,
            url=card.story_url,
            score=0,
            time=card.published,
            text_content=compose_story_text(card.title, card.body, comments),
            source=AINEWS_SOURCE,
            discussion_url=card.topic_url,
            comment_count=len(card_tweets),
            comment_count_at_fetch=len(card_tweets),
            self_text=card.body,
            top_comments=comments,
        )
        try:
            db.upsert_story(story)
        except StoryIdentityConflict:
            logging.warning(
                "ainews_identity_conflict story_id=%s url=%s", sid, card.story_url
            )
            continue
        stories.append(story)
    logging.info(
        "ainews: %d topic stories, %d tweets fetched", len(stories), len(tweets)
    )
    return stories
