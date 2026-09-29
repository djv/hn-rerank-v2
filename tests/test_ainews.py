from __future__ import annotations

import asyncio
from collections.abc import Iterator
from email.utils import formatdate
from html import escape

import pytest

import pipeline.ainews as ainews
from database import Database, Story
from pipeline import Config, load_production_candidate_stories
from pipeline.config import RssConfig
from pipeline.ainews import (
    AINEWS_SOURCE,
    TopicCard,
    Tweet,
    fetch_ainews_stories,
    parse_feed,
    parse_tweet,
    split_issue,
)
from pipeline.enrichment import fetch_rss_feeds, select_rss_article_prewarm

NOW = 2_000_000_000

ISSUE = """
<p><strong>Sponsored by Someone</strong></p>
<blockquote><p>AI News for 9/28. We checked 544 Twitters.</p></blockquote>
<h1>AI Twitter Recap</h1>
<p><strong>Top Story: Model X launch and reactions</strong></p>
<p><strong>Lab shipped Model X, the second model in the family.</strong></p>
<h2>What happened</h2>
<ul><li>Launch: <a href="https://x.com/lab/status/111">Lab</a> and
<a href="https://twitter.com/fan/status/222">a fan</a>.</li></ul>
<p><strong>Facts vs. opinions:</strong></p>
<ul><li>Again <a href="https://x.com/lab/status/111">Lab</a>.</li></ul>
<p><strong>Agents and Tooling</strong></p>
<ul><li>A harness <a href="https://x.com/dev/status/333">dev</a>.</li></ul>
<p><strong>Top tweets (by engagement)</strong></p>
<ul><li>Meme <a href="https://x.com/meme/status/444">meme</a>.</li></ul>
<h1>AI Reddit Recap</h1>
<h3>1. Local model release</h3>
<p>Reddit text <a href="https://x.com/r/status/555">tweet</a></p>
"""


def _item(title: str, link: str, ts: int, html: str) -> str:
    return (
        f"<item><title>{escape(title)}</title><link>{link}</link>"
        f"<pubDate>{formatdate(ts, usegmt=True)}</pubDate>"
        f"<content:encoded>{escape(html)}</content:encoded></item>"
    )


FEED = (
    '<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">'
    "<channel>"
    + _item("[AINews] Model X", "https://l.space/p/ainews-x", NOW - 3600, ISSUE)
    + _item("[AINews] Repost", "https://l.space/p/ainews-repost", NOW - 7200, ISSUE)
    + _item("A podcast", "https://l.space/p/podcast", NOW - 3600, "<p>talk</p>")
    + _item("[AINews] Old", "https://l.space/p/ainews-old", NOW - 90 * 86400, ISSUE)
    + "</channel></rss>"
)


def _split() -> list[TopicCard]:
    return split_issue(
        ISSUE, issue_title="Issue", issue_url="https://l/p/i", published=1
    )


def _tweet(tid: str) -> Tweet:
    return Tweet(
        id=tid, author=f"u{tid}", text=f"tweet {tid}", likes=5, replies=0, created=1
    )


@pytest.fixture
def db() -> Iterator[Database]:
    instance = Database(":memory:")
    yield instance
    instance.close()


@pytest.fixture
def fake_net(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Serve FEED for every feed fetch; record each fetch_tweets id list."""
    calls: list[list[str]] = []

    async def fake_fetch(
        *args: object, **kwargs: object
    ) -> tuple[int, str, dict[str, str]]:
        return 200, FEED, {}

    async def fake_tweets(ids: list[str], **kwargs: object) -> dict[str, Tweet]:
        calls.append(list(ids))
        return {i: _tweet(i) for i in ids}

    monkeypatch.setattr("http_fetch.fetch_with_urllib_fallback", fake_fetch)
    monkeypatch.setattr(ainews, "fetch_tweets", fake_tweets)
    return calls


def _run(db: Database, max_tweets: int = 100) -> list[Story]:
    return asyncio.run(
        fetch_ainews_stories(
            "https://l.space/feed",
            30,
            set(),
            db,
            max_tweets_per_run=max_tweets,
            now=NOW,
        )
    )


def test_split_issue_yields_twitter_recap_topics_only() -> None:
    cards = _split()
    assert [c.title for c in cards] == [
        "Model X launch and reactions",
        "Agents and Tooling",
    ]


def test_top_story_keeps_lead_subsections_and_sub_labels() -> None:
    top = _split()[0]
    assert "Lab shipped Model X" in top.body
    assert "## What happened" in top.body
    assert "Facts vs. opinions:" in top.body
    assert top.tweet_ids == ["111", "222"]


def test_cards_link_first_tweet_and_topic_heading() -> None:
    a, b = _split()
    assert a.story_url == "https://x.com/lab/status/111"
    assert a.topic_url == ("https://l/p/i#:~:text=Model%20X%20launch%20and%20reactions")
    assert b.story_url == "https://x.com/dev/status/333"
    assert a.identity_url != b.identity_url and a.key != b.key


def test_card_without_tweets_links_the_topic() -> None:
    issue = (
        "<h1>AI Twitter Recap</h1><p><strong>Quiet Topic</strong></p>"
        "<ul><li>No tweets, just <a href='https://example.com/a'>a link</a>.</li></ul>"
    )
    (card,) = split_issue(
        issue, issue_title="I", issue_url="https://l/p/q", published=1
    )
    assert card.story_url == card.topic_url == "https://l/p/q#:~:text=Quiet%20Topic"


def test_topic_url_escapes_fragment_syntax() -> None:
    issue = "<h1>AI Twitter Recap</h1><p><strong>GPT-6, Astra &amp; More</strong></p><p>x</p>"
    (card,) = split_issue(
        issue, issue_title="I", issue_url="https://l/p/q", published=1
    )
    assert card.topic_url.endswith("#:~:text=GPT%2D6%2C%20Astra%20%26%20More")


def test_parse_feed_keeps_new_ainews_issues_and_drops_reposted_topics() -> None:
    cards = parse_feed(FEED, cutoff=NOW - 30 * 86400)
    assert [(c.issue_url, c.title) for c in cards] == [
        ("https://l.space/p/ainews-x", "Model X launch and reactions"),
        ("https://l.space/p/ainews-x", "Agents and Tooling"),
    ]
    assert cards[0].published == NOW - 3600


def test_parse_tweet_normalizes_and_rejects_missing() -> None:
    payload = {
        "code": 200,
        "tweet": {
            "text": "hello",
            "likes": 7,
            "replies": 2,
            "created_timestamp": 5,
            "author": {"screen_name": "a"},
            "quote": {"text": "quoted"},
        },
    }
    tw = parse_tweet("9", payload)
    assert tw is not None
    assert (tw.author, tw.likes, tw.quote_text) == ("a", 7, "quoted")
    assert parse_tweet("9", {"code": 404, "message": "NOT_FOUND"}) is None


def test_fetch_stores_topic_stories_with_bullets_and_tweets(
    db: Database, fake_net: list[list[str]]
) -> None:
    stories = _run(db)
    assert [s.title for s in stories] == [
        "Model X launch and reactions",
        "Agents and Tooling",
    ]
    top = db.get_story(stories[0].id)
    assert top is not None and top.source == AINEWS_SOURCE
    assert top.url == "https://x.com/lab/status/111"
    assert top.discussion_url is not None and "#:~:text=Model%20X" in top.discussion_url
    assert "Lab shipped Model X" in top.self_text
    assert "@u111 (5 likes): tweet 111" in top.top_comments
    assert "tweet 222" in top.top_comments and top.comment_count == 2
    assert top.text_content.startswith("Model X launch and reactions.")
    assert "tweet 222" in top.text_content
    assert fake_net == [["111", "222", "333"]]


def test_refetch_reuses_stored_tweets(db: Database, fake_net: list[list[str]]) -> None:
    first = _run(db)
    second = _run(db)
    assert [s.id for s in second] == [s.id for s in first]
    assert fake_net[1] == []  # nothing new to fetch


def test_tweet_cap_defers_the_rest_to_a_later_run(
    db: Database, fake_net: list[list[str]]
) -> None:
    stories = _run(db, max_tweets=2)
    assert fake_net[0] == ["111", "222"]
    tooling = db.get_story(stories[1].id)
    assert tooling is not None and tooling.top_comments == ""
    _run(db, max_tweets=2)
    assert fake_net[1] == ["333"]
    tooling = db.get_story(stories[1].id)
    assert tooling is not None and "tweet 333" in tooling.top_comments


def test_generic_rss_skips_whole_ainews_issues(
    db: Database, fake_net: list[list[str]]
) -> None:
    stories = asyncio.run(
        fetch_rss_feeds(
            ["https://l.space/feed"],
            10,
            10**6,
            set(),
            db,
            skip_title_prefixes=("[AINews]",),
        )
    )
    assert [s.title for s in stories] == ["A podcast"]


def test_candidates_include_topics_and_drop_old_layout_rows(
    db: Database, fake_net: list[list[str]]
) -> None:
    topics = _run(db)
    # A topic row from the first layout: issue#slug URL, no discussion link.
    db.upsert_story(
        Story(
            id=-3,
            title="Model X launch and reactions",
            url="https://l.space/p/ainews-x#model-x-launch-and-reactions",
            score=0,
            time=NOW - 60,
            text_content="t",
            source=AINEWS_SOURCE,
            self_text="text",
        )
    )
    for sid, title in ((-1, "[AINews] Whole issue"), (-2, "A podcast")):
        db.upsert_story(
            Story(
                id=sid,
                title=title,
                url=f"https://l.space/p/{-sid}",
                score=0,
                time=NOW - 60,
                text_content=title,
                source="rss_latent_space",
                self_text="text",
            )
        )
    config = Config(days=30, rss=RssConfig(feeds=("https://www.latent.space/feed",)))
    ids = {
        s.id
        for s in load_production_candidate_stories(
            db, config, user_id=None, exclude_feedback=False, now_ts=NOW
        )
    }
    assert {s.id for s in topics} | {-2} <= ids
    assert -1 not in ids and -3 not in ids


def test_article_prewarm_never_fetches_the_issue_page(
    db: Database, fake_net: list[list[str]]
) -> None:
    topics = _run(db)
    assert select_rss_article_prewarm(topics, db, max_per_run=10, now_ts=NOW) == []


def test_parse_tweet_keeps_expanded_links() -> None:
    payload = {
        "code": 200,
        "tweet": {
            "text": "decryptor https://github.com/x/y",
            "author": {"screen_name": "a"},
            "raw_text": {
                "facets": [
                    {"type": "url", "replacement": "https://github.com/x/y"},
                    {
                        "type": "media",
                        "replacement": "https://x.com/a/status/9/photo/1",
                    },
                ]
            },
        },
    }
    tw = parse_tweet("9", payload)
    assert tw is not None and tw.links == ("https://github.com/x/y",)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://twitter.com/h0t_max/status/1549155542786080774",
            "1549155542786080774",
        ),
        ("https://x.com/a/status/12?s=20", "12"),
        ("https://mobile.twitter.com/a/status/34", "34"),
        ("https://example.com/x.com/a/status/56", None),
        ("https://x.com/a", None),
    ],
)
def test_tweet_id_from_url(url: str, expected: str | None) -> None:
    assert ainews.tweet_id_from_url(url) == expected
