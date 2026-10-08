"""Arctic Shift client and the Reddit source adapter built on it."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import arctic_shift
import pipeline
import server
from arctic_shift import ArcticComment, ArcticPost, ArcticShiftError
from database import Database
from pipeline.config import Config, RssConfig
from pipeline.enrichment import (
    _fetch_and_parse_feed,
    build_reddit_prewarm_factories,
    build_reddit_topfeed_factories,
    reddit_post_self_text,
    reddit_post_story,
    reddit_top_query,
)
from reddit_feed_cache import cache as reddit_feed_cache

WEEK = 7 * 86400
NOW = 1_800_000_000
FEED = "https://www.reddit.com/r/LocalLLaMA/top/.rss?t=week&limit=25"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(arctic_shift, "REQUEST_SPACING_SECONDS", 0.0)
    monkeypatch.setattr(arctic_shift, "RETRY_DELAY_SECONDS", 0.0)
    arctic_shift.reset_throttle()
    yield
    arctic_shift.reset_throttle()


def _serve(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=arctic_shift.ARCTIC_SHIFT_BASE_URL,
            transport=httpx.MockTransport(record),
        )

    monkeypatch.setattr(arctic_shift, "_client", client)
    return seen


def _raw_post(
    post_id: str,
    score: int,
    created: int,
    *,
    title: str = "A post",
    selftext: str = "",
    url: str | None = None,
    is_self: bool = True,
    removed_by: str | None = None,
) -> dict[str, object]:
    permalink = f"/r/LocalLLaMA/comments/{post_id}/slug_{post_id}/"
    return {
        "id": post_id,
        "permalink": permalink,
        "title": title,
        "selftext": selftext,
        "url": url or f"https://www.reddit.com{permalink}",
        "is_self": is_self,
        "author": "alice",
        "score": score,
        "num_comments": 3,
        "created_utc": created,
        "removed_by_category": removed_by,
    }


def _archive(posts: list[dict[str, object]], page_size: int = 1000) -> Handler:
    """A fake archive: time-cursor search (inclusive ``after``) and ids."""
    by_id = {str(p["id"]): p for p in posts}

    def handle(request: httpx.Request) -> httpx.Response:
        params = parse_qs(urlparse(str(request.url)).query)
        if request.url.path == "/api/posts/search":
            after = int(params["after"][0])
            before = int(params["before"][0])
            rows = sorted(
                (p for p in posts if after <= int(str(p["created_utc"])) < before),
                key=lambda p: int(str(p["created_utc"])),
            )[:page_size]
            fields = params["fields"][0].split(",")
            data = [{f: p[f] for f in fields} for p in rows]
            return httpx.Response(200, json={"data": data})
        if request.url.path == "/api/posts/ids":
            ids = params["ids"][0].split(",")
            return httpx.Response(
                200, json={"data": [by_id[i] for i in ids if i in by_id]}
            )
        return httpx.Response(404)

    return handle


def _post(
    post_id: str = "abc12",
    *,
    selftext: str = "Body text of the post.",
    url: str = "",
    is_self: bool = True,
    created: int = NOW - 3600,
) -> ArcticPost:
    return ArcticPost(
        id=post_id,
        permalink=f"/r/LocalLLaMA/comments/{post_id}/slug/",
        title="Q&A thread",
        url=url or f"https://www.reddit.com/r/LocalLLaMA/comments/{post_id}/slug/",
        selftext=selftext,
        author="alice",
        score=500,
        num_comments=40,
        created_utc=created,
        is_self=is_self,
        removed=False,
    )


async def test_top_posts_ranks_by_score_and_drops_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleted/removed text drops a post; a stale ``removed_by_category``
    (moderators approve AutoModerator-held posts later) does not."""
    posts = [
        _raw_post("a", 50, NOW - 5 * 86400, title="Q&amp;A &lt;3"),
        _raw_post("b", 900, NOW - 4 * 86400, selftext="[deleted]"),
        _raw_post("c", 300, NOW - 3 * 86400),
        _raw_post("d", 300, NOW - 2 * 86400, removed_by="automod_filtered"),
        _raw_post("e", 2000, NOW - 8 * 86400),
        _raw_post("f", 1, NOW - 600, selftext="[removed]"),
    ]
    seen = _serve(monkeypatch, _archive(posts))

    top = await arctic_shift.top_posts(
        "LocalLLaMA", window_seconds=WEEK, limit=3, now=NOW
    )

    assert [p.id for p in top] == ["d", "c", "a"]
    assert top[2].title == "Q&A <3"
    search = parse_qs(urlparse(str(seen[0].url)).query)
    assert search["subreddit"] == ["LocalLLaMA"]
    assert search["after"] == [str(NOW - WEEK)]
    assert search["sort"] == ["asc"]


async def test_top_posts_loads_more_records_when_leaders_were_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts = [
        _raw_post("r1", 900, NOW - 3600, selftext="[removed]"),
        _raw_post("r2", 800, NOW - 3600, selftext="[deleted]"),
        _raw_post("r3", 700, NOW - 3600, selftext="[removed]"),
        _raw_post("ok1", 600, NOW - 3600),
        _raw_post("ok2", 500, NOW - 3600),
        _raw_post("ok3", 400, NOW - 3600),
    ]
    seen = _serve(monkeypatch, _archive(posts))

    top = await arctic_shift.top_posts("x", window_seconds=WEEK, limit=2, now=NOW)

    assert [p.id for p in top] == ["ok1", "ok2"]
    assert [r.url.path for r in seen].count("/api/posts/ids") == 2


async def test_top_posts_refuses_a_search_it_cannot_page_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated search would rank only the oldest posts, so it fails and
    the refresh keeps the feed's previous stories."""
    page = arctic_shift.AUTO_PAGE_MIN_ROWS
    # More posts within one second than a page holds: time paging stalls.
    tied = [_raw_post(f"t{i}", 5, NOW - 3600) for i in range(page + 20)]
    _serve(monkeypatch, _archive(tied, page))
    with pytest.raises(ArcticShiftError, match="full page"):
        await arctic_shift.top_posts("x", window_seconds=WEEK, limit=5, now=NOW)

    spread = [_raw_post(f"s{i}", 5, NOW - WEEK + 60 * i) for i in range(3 * page)]
    _serve(monkeypatch, _archive(spread, page))
    monkeypatch.setattr(arctic_shift, "MAX_SEARCH_PAGES", 2)
    with pytest.raises(ArcticShiftError, match="more than 2 pages"):
        await arctic_shift.top_posts("x", window_seconds=WEEK, limit=5, now=NOW)


async def test_top_posts_raises_on_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(
        monkeypatch,
        lambda _r: httpx.Response(200, json={"data": None, "error": "bad field"}),
    )
    with pytest.raises(ArcticShiftError, match="bad field"):
        await arctic_shift.top_posts("x", window_seconds=WEEK, limit=5, now=NOW)


async def test_get_retries_transient_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = [422, 200]

    def handle(_request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        return httpx.Response(status, json={"data": []})

    seen = _serve(monkeypatch, handle)
    assert await arctic_shift.top_posts("x", window_seconds=WEEK, limit=5) == []
    assert len(seen) == 2

    # Background top lists wait out overload longer than thread fetches,
    # which serve card taps.
    seen = _serve(monkeypatch, lambda _r: httpx.Response(429))
    with pytest.raises(ArcticShiftError, match="HTTP 429"):
        await arctic_shift.top_posts("x", window_seconds=WEEK, limit=5)
    assert len(seen) == arctic_shift.TOP_POSTS_RETRY.attempts
    seen.clear()
    with pytest.raises(ArcticShiftError, match="HTTP 429"):
        await arctic_shift.comment_tree("abc")
    assert len(seen) == arctic_shift.THREAD_RETRY.attempts


def test_overload_wait_follows_the_server_reset_up_to_each_cap() -> None:
    """The archive's overload window resets on the minute: background top
    lists wait it out, thread fetches for card taps do not."""
    overloaded = httpx.Response(422, headers={"x-ratelimit-reset": "51"})
    assert (
        arctic_shift._retry_wait(
            overloaded, arctic_shift.TOP_POSTS_RETRY.max_wait_seconds
        )
        == 51.0
    )
    assert (
        arctic_shift._retry_wait(overloaded, arctic_shift.THREAD_RETRY.max_wait_seconds)
        == arctic_shift.THREAD_RETRY.max_wait_seconds
    )


# The autouse throttle fixture only zeroes module constants, so sharing it
# across examples is safe.
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    groups=st.lists(
        st.tuples(st.integers(0, WEEK - 1), st.integers(1, 3)),
        unique_by=lambda g: g[0],
        max_size=400,
    ),
    page_size=st.integers(arctic_shift.AUTO_PAGE_MIN_ROWS, 250),
)
def test_search_pages_reach_every_post_once(
    groups: list[tuple[int, int]], page_size: int
) -> None:
    """Time-cursor paging returns each post once for any page size, with
    a few posts sharing a second at page boundaries."""
    posts = [
        _raw_post(f"p{offset}x{k}", 1, NOW - WEEK + offset)
        for offset, count in groups
        for k in range(count)
    ]

    async def run() -> list[str]:
        async with httpx.AsyncClient(
            base_url=arctic_shift.ARCTIC_SHIFT_BASE_URL,
            transport=httpx.MockTransport(_archive(posts, page_size)),
        ) as client:
            found = await arctic_shift._search_scores(client, "x", NOW - WEEK, NOW)
        return [p.id for p in found]

    ids = asyncio.run(run())
    assert sorted(ids) == sorted(str(p["id"]) for p in posts)


async def test_comment_tree_flattens_with_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def comment(cid: str, score: int, replies: list[object]) -> dict[str, object]:
        return {
            "kind": "t1",
            "data": {
                "id": cid,
                "author": f"user_{cid}",
                "body": f"Body &gt; {cid}",
                "score": score,
                "created_utc": NOW,
                "replies": {"data": {"children": replies}} if replies else "",
            },
        }

    tree = [
        comment("a", 5, [comment("a1", 9, [comment("a1x", 1, [])])]),
        {"kind": "more", "data": {"count": 12}},
        comment("b", 7, []),
    ]
    seen = _serve(monkeypatch, lambda _r: httpx.Response(200, json={"data": tree}))

    comments = await arctic_shift.comment_tree("xyz")

    assert [(c.author, c.depth) for c in comments] == [
        ("user_a", 0),
        ("user_a1", 1),
        ("user_a1x", 2),
        ("user_b", 0),
    ]
    assert comments[0].body == "Body > a"
    assert parse_qs(urlparse(str(seen[0].url)).query)["link_id"] == ["xyz"]


@pytest.mark.parametrize(
    ("feed_url", "expected"),
    [
        (FEED, ("LocalLLaMA", float(WEEK), 25)),
        ("https://www.reddit.com/r/haskell/top/.rss", ("haskell", 86400.0, 25)),
        ("https://old.reddit.com/r/x/top/.rss?t=month&limit=10", ("x", 2592000.0, 10)),
        # All-time leaders come before the age cutoff on Reddit; a windowed
        # archive search cannot reproduce that, so t=all is unsupported.
        ("https://www.reddit.com/r/x/top/.rss?t=all", None),
        ("https://www.reddit.com/r/x/hot/.rss", None),
        ("https://example.com/r/x/top/.rss", None),
    ],
)
def test_reddit_top_query(
    feed_url: str, expected: tuple[str, float, int] | None
) -> None:
    assert reddit_top_query(feed_url) == expected


def test_self_text_keeps_external_links_only() -> None:
    assert (
        reddit_post_self_text(
            _post(url="https://example.com/a", is_self=False, selftext="")
        )
        == "https://example.com/a"
    )
    assert (
        reddit_post_self_text(
            _post(url="https://v.redd.it/xyz", is_self=False, selftext="Look")
        )
        == "Look"
    )
    assert reddit_post_self_text(_post()) == "Body text of the post."


async def test_arctic_story_matches_rss_story_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post keeps its story id, URL and source when the Reddit source
    switches from RSS to Arctic Shift, so its votes and caches carry over."""
    post = _post()
    link = f"https://www.reddit.com{post.permalink}"
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>top</title>
<entry><title>{post.title.replace("&", "&amp;")}</title><link href="{link}" />
<id>t3_{post.id}</id><published>2027-01-15T07:00:00+00:00</published>
<content type="html">&lt;p&gt;Body&lt;/p&gt;</content></entry></feed>"""

    class MockResp:
        status_code = 200
        text = rss
        content = rss.encode()
        headers: dict[str, str] = {}

    class MockClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> MockClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(
            self, _url: str, headers: dict[str, str] | None = None
        ) -> MockResp:
            return MockResp()

    monkeypatch.setattr("pipeline.enrichment.httpx.AsyncClient", MockClient)
    rss_stories = await _fetch_and_parse_feed(FEED, 25, 0.0, NOW, set())
    assert len(rss_stories) == 1

    arctic = reddit_post_story(post, rss_stories[0].source)
    assert (arctic.id, arctic.url, arctic.title, arctic.source) == (
        rss_stories[0].id,
        rss_stories[0].url,
        rss_stories[0].title,
        "rss_reddit_localllama",
    )
    assert (arctic.score, arctic.comment_count) == (0, 0)


async def test_arctic_topfeed_factory_bypasses_reddit_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float, int]] = []
    voted = _post("voted")
    # Older than the 30-day cutoff, which the factory takes from the clock.
    old = _post("old", created=int(time.time()) - 40 * 86400)

    async def fake_top_posts(
        subreddit: str, *, window_seconds: float, limit: int, now: float | None
    ) -> list[ArcticPost]:
        calls.append((subreddit, window_seconds, limit))
        return [_post("keep"), voted, old]

    async def no_reddit(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("Arctic Shift fetches must not use reddit_limiter")

    monkeypatch.setattr(arctic_shift, "top_posts", fake_top_posts)
    monkeypatch.setattr(pipeline.enrichment.reddit_limiter, "acquire", no_reddit)

    factories, feeds = build_reddit_topfeed_factories(
        [FEED, "https://example.com/feed.xml"],
        per_feed=10,
        days=30,
        exclude_urls={f"https://www.reddit.com{voted.permalink}"},
        reddit_source="arctic_shift",
    )
    assert feeds == [FEED]
    await factories[0]()

    cached = reddit_feed_cache.get(FEED)
    assert cached is not None
    assert [s.url for s in cached] == [
        "https://www.reddit.com/r/LocalLLaMA/comments/keep/slug/"
    ]
    assert calls == [("LocalLLaMA", float(WEEK), 10)]


async def test_arctic_prewarm_ignores_open_reddit_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = Database(":memory:")
    try:
        story = reddit_post_story(_post(), "rss_reddit_localllama")
        db.upsert_story(story)

        async def fake_context(url: str | None) -> server.RedditRssContext:
            assert url == story.url
            return server.RedditRssContext(
                self_text="Body text of the post.",
                top_comments="/u/bob: A long enough archived comment to keep.",
                comment_count=1,
            )

        monkeypatch.setattr(server, "_fetch_reddit_arctic_context", fake_context)
        monkeypatch.setattr(
            type(pipeline.enrichment.reddit_limiter),
            "circuit_open",
            property(lambda _self: True),
        )

        factories, updated = build_reddit_prewarm_factories(
            [story.id], db, reddit_source="arctic_shift"
        )
        await factories[0]()

        assert updated == [story.id]
        stored = db.get_story(story.id)
        assert stored is not None
        assert "archived comment" in stored.top_comments
    finally:
        db.close()


def test_refresh_uses_arctic_source_and_ignores_kept_longer_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In arctic mode the refresh passes the source to both builders and
    paces the queue by ``reddit_arctic_stride_seconds``. A snapshot whose
    text is shorter than the stored row (upsert keeps the longer) is not a
    change."""
    db = Database(":memory:")
    try:
        config = replace(
            Config(db_path=":memory:"),
            reddit_source="arctic_shift",
            reddit_arctic_stride_seconds=3.0,
            rss=RssConfig(enabled=True, per_feed_limit=70, feeds=(FEED,)),
        )
        fetched = reddit_post_story(_post(selftext="Short."), "rss_reddit_localllama")
        db.upsert_story(
            replace(fetched, self_text="Short. submitted by /u/alice [link]")
        )
        sources: list[str] = []
        strides: list[float | None] = []

        def fake_topfeed(
            feeds: list[str],
            per_feed: int,
            days: int,
            exclude_urls: set[str],
            reddit_source: str = "rss",
        ) -> tuple[list[object], list[str]]:
            sources.append(reddit_source)
            reddit_feed_cache.set(FEED, [fetched])

            async def factory() -> None:
                return None

            return [factory], [FEED]

        def fake_prewarm(
            story_ids: list[int], db_: Database, reddit_source: str = "rss"
        ) -> tuple[list[object], list[int]]:
            sources.append(reddit_source)

            async def factory() -> None:
                return None

            return [factory], []

        class _Queue:
            def enqueue_all_reddit_fetches(
                self,
                topfeed: list[object],
                prewarm: list[object],
                *,
                min_stride_seconds: float | None = None,
            ) -> None:
                strides.append(min_stride_seconds)

            def wait_until_empty(self, timeout: float = 5400.0) -> bool:
                return True

        monkeypatch.setattr(pipeline, "build_reddit_topfeed_factories", fake_topfeed)
        monkeypatch.setattr(pipeline, "build_reddit_prewarm_factories", fake_prewarm)
        monkeypatch.setattr("reddit_fetch_queue.queue", _Queue())

        result = pipeline.refresh_reddit_candidates(config, db, None)

        assert sources == ["arctic_shift", "arctic_shift"]
        assert strides == [3.0, 3.0]
        assert result.changed_stories == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.reddit.com/r/x/comments/1wymgu6/slug/", "1wymgu6"),
        ("https://old.reddit.com/r/x/comments/abc", "abc"),
        ("https://www.reddit.com/r/x/top/", None),
        ("https://example.com/r/x/comments/abc/", None),
        (None, None),
    ],
)
def test_reddit_post_id(url: str | None, expected: str | None) -> None:
    assert server._reddit_post_id(url) == expected


async def test_arctic_context_orders_and_filters_comments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def c(author: str, body: str, score: int, depth: int = 0) -> ArcticComment:
        return ArcticComment(
            author=author, body=body, score=score, created_utc=NOW, depth=depth
        )

    long = "has enough words to count as a real comment"
    comments = [
        c("low", f"Top-level low {long}", 5),
        c("reply", f"Reply with high score {long}", 900, depth=1),
        c("high", f"Top-level high  {long}", 50),
        c("AutoModerator", f"Rules reminder {long}", 99),
        c("[deleted]", f"Ghost {long}", 80),
        c("short", "lol", 70),
    ]

    async def fake_posts(ids: list[str]) -> list[ArcticPost]:
        assert ids == ["abc12"]
        return [_post()]

    async def fake_tree(post_id: str) -> list[ArcticComment]:
        return comments

    monkeypatch.setattr(arctic_shift, "posts", fake_posts)
    monkeypatch.setattr(arctic_shift, "comment_tree", fake_tree)
    monkeypatch.setattr(server, "REDDIT_COMMENT_LIMIT", 3)

    ctx = await server._fetch_reddit_arctic_context(
        "https://www.reddit.com/r/LocalLLaMA/comments/abc12/slug/"
    )

    assert ctx is not None
    assert ctx.self_text == "Body text of the post."
    assert ctx.top_comments == " ".join(
        [
            f"/u/high: Top-level high {long}",
            f"/u/low: Top-level low {long}",
            f"/u/reply: Reply with high score {long}",
        ]
    )
    assert ctx.comment_count == 3


async def test_arctic_context_gives_up_at_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow archive must not hold a card tap past the deadline (clients
    give up after 150 s; retries alone could take minutes)."""

    async def slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json={"data": []})

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=arctic_shift.ARCTIC_SHIFT_BASE_URL,
            transport=httpx.MockTransport(slow),
        )

    monkeypatch.setattr(arctic_shift, "_client", client)
    monkeypatch.setattr(server, "ARCTIC_THREAD_DEADLINE_SECONDS", 0.2)
    started = time.monotonic()

    ctx = await server._fetch_reddit_arctic_context(
        "https://www.reddit.com/r/x/comments/abc/slug/"
    )

    assert ctx is None
    assert time.monotonic() - started < 5


async def test_arctic_context_is_none_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing(_ids: list[str]) -> list[ArcticPost]:
        raise ArcticShiftError("down")

    monkeypatch.setattr(arctic_shift, "posts", failing)
    assert (
        await server._fetch_reddit_arctic_context(
            "https://www.reddit.com/r/x/comments/abc/slug/"
        )
        is None
    )


def test_config_rejects_unknown_reddit_source(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[hn_rewrite]\nreddit_source = "json"\n')
    with pytest.raises(ValueError, match="reddit_source"):
        Config.load(str(path))


def test_payload_shape_errors_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, lambda _r: httpx.Response(200, content=json.dumps([1])))
    with pytest.raises(ArcticShiftError, match="not an object"):
        asyncio.run(arctic_shift.comment_tree("abc"))


def test_compare_feed_matches_by_story_id() -> None:
    from scripts.compare_reddit_sources import compare_feed

    shared = reddit_post_story(_post("same"), "rss_reddit_localllama")
    young = reddit_post_story(
        _post("young", created=NOW - 3600), "rss_reddit_localllama"
    )
    archived = reddit_post_story(
        _post("archived", created=NOW - 3 * 86400), "rss_reddit_localllama"
    )

    result = compare_feed(FEED, [shared, young], [shared, archived], NOW)

    assert (result.rss_count, result.arctic_count, result.shared) == (2, 2, 1)
    assert [o.url for o in result.rss_only] == [young.url]
    assert result.rss_only_young == 1
    assert [o.age_hours for o in result.arctic_only] == [72.0]
