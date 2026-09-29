from __future__ import annotations

from scripts.ainews_topics import TopicCard, parse_tweet, split_issue

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


def _split() -> list[TopicCard]:
    return split_issue(
        ISSUE, issue_title="Issue", issue_url="https://l/p/i", published=1
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


def test_cards_get_distinct_urls_and_keys() -> None:
    a, b = _split()
    assert a.url == "https://l/p/i#model-x-launch-and-reactions"
    assert a.url != b.url and a.key != b.key
    assert b.tweet_ids == ["333"]


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
