"""Boundary invariants: wire parsing and deployment URLs must fail safely."""

from __future__ import annotations

import unicodedata
from dataclasses import fields, replace
from typing import cast

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hn_rerank.api import API, APIError, normalize_server
from hn_rerank.models import Feed, FeedBadge, FeedStory

from .test_client import sample_feed

# Arbitrary JSON, including shapes a proxy or a wrong deployment can return.
_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=6), children, max_size=3)
    ),
    max_leaves=8,
)


# Text as a hostile feed or LLM could send it: plain text interleaved with
# terminal escape sequences and raw C0/C1 control characters.
_UNTRUSTED_TEXT = st.lists(
    st.one_of(
        st.text(max_size=6),
        st.sampled_from(
            ["\x1b[31m", "\x1b]0;PWNED\x1b\\", "\x1b]52;c;eA==\x07", "\x9b2J"]
        ),
        st.characters(max_codepoint=0x9F),
    ),
    max_size=6,
).map("".join)


def _strip_controls(text: str) -> str:
    """Spec: Unicode Cc (C0, DEL, C1) are removed, except tab and newline."""
    return "".join(c for c in text if c in "\t\n" or unicodedata.category(c) != "Cc")


def _terminal_view(feed: Feed) -> Feed:
    """The feed with every displayed string passed through the spec."""

    def clean(obj: FeedStory | FeedBadge) -> FeedStory | FeedBadge:
        changes: dict[str, object] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if isinstance(value, str):
                changes[f.name] = _strip_controls(value)
            elif isinstance(value, list):
                changes[f.name] = [
                    _strip_controls(item) if isinstance(item, str) else clean(item)
                    for item in value
                ]
        return replace(obj, **changes)

    return replace(feed, stories=[cast(FeedStory, clean(s)) for s in feed.stories])


@st.composite
def _feed_stories(draw: st.DrawFn) -> FeedStory:
    return FeedStory(
        id=draw(st.integers(min_value=0)),
        title=draw(_UNTRUSTED_TEXT),
        article_url=draw(st.text(max_size=20)),
        comments_url=draw(st.text(max_size=20)),
        source=draw(st.text(max_size=10)),
        points=draw(st.integers()),
        comments=draw(st.one_of(st.none(), st.integers())),
        time=draw(st.integers()),
        rank_score=draw(st.floats(allow_nan=False, allow_infinity=False)),
        memberships=draw(st.lists(st.text(max_size=8), max_size=3)),
        popular=draw(st.booleans()),
        explore=draw(st.booleans()),
        badges=draw(st.lists(_UNTRUSTED_TEXT, max_size=2)),
        badge_details=draw(
            st.lists(
                st.builds(
                    FeedBadge,
                    _UNTRUSTED_TEXT,
                    _UNTRUSTED_TEXT,
                    _UNTRUSTED_TEXT,
                    _UNTRUSTED_TEXT,
                ),
                max_size=2,
            )
        ),
        domain=draw(_UNTRUSTED_TEXT),
    )


@st.composite
def _feeds(draw: st.DrawFn) -> Feed:
    stories = draw(
        st.lists(_feed_stories(), max_size=4, unique_by=lambda story: story.id)
    )
    ids = [story.id for story in stories]
    orders = {
        "recommended:recent": draw(st.lists(st.sampled_from(ids), max_size=3))
        if ids
        else []
    }
    counts = {
        action: draw(st.integers(min_value=0)) for action in ("up", "neutral", "down")
    }
    return Feed(
        1,
        stories,
        orders,
        counts,
        draw(st.integers(min_value=0)),
        draw(st.integers(min_value=0)),
        draw(st.booleans()),
    )


@given(_feeds())
@settings(deadline=None)
def test_parse_round_trips_valid_wire_payloads(feed: Feed) -> None:
    """Valid payloads parse to the same feed, minus terminal control
    characters in any displayed string; everything else is kept."""
    assert Feed.parse(feed.to_dict()) == _terminal_view(feed)


@st.composite
def _malformed_payloads(draw: st.DrawFn) -> object:
    payload = draw(_feeds()).to_dict()
    payload[draw(st.sampled_from(sorted(payload)))] = draw(_JSON)
    stories = payload["stories"]
    if isinstance(stories, list) and stories and draw(st.booleans()):
        story = stories[draw(st.integers(min_value=0, max_value=len(stories) - 1))]
        if isinstance(story, dict) and story:
            fields = cast("dict[str, object]", story)
            fields[draw(st.sampled_from(sorted(fields)))] = draw(_JSON)
    return payload


@given(st.one_of(_JSON, _malformed_payloads()))
@settings(deadline=None)
def test_parse_raises_only_value_error_on_malformed_payloads(payload: object) -> None:
    try:
        Feed.parse(payload)
    except ValueError:
        pass


def test_parse_rejects_unrepresentable_rank_score() -> None:
    # JSON numbers are arbitrary-precision; math.isfinite overflows on huge ints.
    story = replace(sample_feed().stories[0], rank_score=10**400)
    with pytest.raises(ValueError, match="Invalid feed response"):
        Feed.parse(replace(sample_feed(), stories=[story]).to_dict())


@pytest.mark.parametrize("rank_score", [True, False])
def test_parse_rejects_boolean_rank_score(rank_score: bool) -> None:
    story = replace(sample_feed().stories[0], rank_score=rank_score)
    with pytest.raises(ValueError, match="Invalid story"):
        Feed.parse(replace(sample_feed(), stories=[story]).to_dict())


@pytest.mark.parametrize("badges", [None, [42], [{"kind": 7}], "hot"])
def test_parse_rejects_malformed_badge_details(badges: object) -> None:
    payload = sample_feed().to_dict()
    stories = cast("list[dict[str, object]]", payload["stories"])
    stories[0]["badge_details"] = badges
    with pytest.raises(ValueError):
        Feed.parse(payload)


def test_parse_tolerates_missing_badges_from_older_servers() -> None:
    payload = sample_feed().to_dict()
    stories = cast("list[dict[str, object]]", payload["stories"])
    for story in stories:
        del story["badges"]
    assert [story.badges for story in Feed.parse(payload).stories] == [[], [], []]


@pytest.mark.parametrize(
    "server",
    [
        "https://example.org:abc/hn/",
        "https://example.org:65536/hn/",
        "https://example.org:-1/hn/",
    ],
)
def test_normalize_server_rejects_unrequestable_ports(server: str) -> None:
    with pytest.raises(ValueError, match="valid port"):
        normalize_server(server)


@pytest.mark.parametrize(
    "server",
    [
        "https://exa\x00mple.org/hn/",
        "https://127.0.0.1i\u00a1\x83|\U00076966V/hn/",
    ],
)
def test_normalize_server_rejects_hosts_httpx_cannot_parse(server: str) -> None:
    with pytest.raises(ValueError, match="client can request"):
        normalize_server(server)


@st.composite
def _server_urls(draw: st.DrawFn) -> str:
    scheme = draw(st.one_of(st.just("http"), st.just("https")))
    host = draw(
        st.one_of(
            st.just("localhost"),
            st.just("127.0.0.1"),
            st.just("[::1]"),
            st.from_regex(r"[a-z0-9]([a-z0-9.-]{0,20}[a-z0-9])?", fullmatch=True),
        )
    )
    port = draw(
        st.one_of(
            st.none(),
            st.integers(min_value=0, max_value=70000),
            st.from_regex(r"[0-9a-f]{0,5}", fullmatch=True),
        )
    )
    path = draw(
        st.one_of(
            st.just(""),
            st.just("/"),
            st.from_regex(r"/[a-zA-Z0-9/._%-]{0,20}", fullmatch=True),
        )
    )
    authority = host if port is None else f"{host}:{port}"
    return f"{scheme}://{authority}{path}"


@given(st.one_of(st.text(), _server_urls()))
@settings(deadline=None)
def test_accepted_servers_are_requestable_and_idempotent(value: str) -> None:
    try:
        normalized = normalize_server(value)
    except ValueError:
        return
    httpx.URL(normalized)  # The HTTP layer must be able to build this request.
    assert normalize_server(normalized) == normalized


@pytest.mark.parametrize("payload", [[], 42, "nope", None])
async def test_feed_rejects_non_object_payloads(payload: object) -> None:
    api = API(
        "https://example.org/hn/",
        "token",
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    try:
        with pytest.raises(APIError, match="Invalid feed response"):
            await api.feed()
    finally:
        await api.close()


async def test_request_converts_invalid_url_to_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = API("https://example.org/hn/", "token")

    async def invalid(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.InvalidURL("Invalid port: 'abc'")

    monkeypatch.setattr(api.client, "request", invalid)
    try:
        with pytest.raises(APIError, match="Connection failed"):
            await api.validate()
    finally:
        await api.close()


async def test_summaries_drop_terminal_control_sequences() -> None:
    hostile = "# Title\x1b]0;PWNED\x1b\\\n\n\x1b[31mred\x9b2J text\r\n\tend\x07"
    api = API(
        "https://example.org/hn/",
        "token",
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ok": True, "tldr": hostile})
        ),
    )
    try:
        for summary in (await api.summary(1), await api.cached_summary(1)):
            assert summary is not None
            assert summary.text == _strip_controls(hostile)
            assert "\x1b" not in summary.text and "\x9b" not in summary.text
            assert "\n\tend" in summary.text  # layout whitespace survives
    finally:
        await api.close()
