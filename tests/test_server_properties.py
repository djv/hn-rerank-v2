"""Property tests for server request-edge helpers (rate limiting, client IP)."""

from __future__ import annotations

import ipaddress
import itertools
from collections import deque

import pytest
from flask import Flask
from hypothesis import given, settings, strategies as st

import server
from server import FixedWindowLimiter

_KEYS = st.sampled_from(["ip:a", "ip:a", "ip:a", "ip:b"])


@st.composite
def _limiter_ops(draw: st.DrawFn) -> list[tuple[float, str]]:
    """Non-decreasing timestamps paired with bucket keys. Gaps mix bursts
    with pauses on the scale of the windows and of the 300 s idle-bucket
    sweep, so hits straddle window edges exactly when a sweep runs."""
    gaps = draw(
        st.lists(
            st.one_of(st.integers(0, 5), st.integers(100, 450)),
            min_size=1,
            max_size=40,
        )
    )
    now = 0.0
    ops = []
    for gap in gaps:
        now += gap
        ops.append((now, draw(_KEYS)))
    return ops


@settings(max_examples=400)
@given(
    ops=_limiter_ops(),
    limit=st.integers(1, 4),
    window=st.sampled_from([60, 400]),
)
def test_fixed_window_limiter_matches_sliding_window_model(
    ops: list[tuple[float, str]], limit: int, window: int
) -> None:
    """Each decision equals an independent sliding-window model's, so the
    idle-bucket sweep never forgets a hit that still counts."""
    limiter = FixedWindowLimiter()
    model: dict[str, deque[float]] = {}
    for now, key in ops:
        hits = model.setdefault(key, deque())
        while hits and hits[0] <= now - window:
            hits.popleft()
        expected = len(hits) < limit
        if expected:
            hits.append(now)
        result = limiter.try_acquire([(key, limit, window)], now=now)
        assert result.allowed is expected
        if not expected:
            assert 1 <= result.retry_after_seconds <= window


@given(
    ops=_limiter_ops(),
    per_key=st.integers(1, 3),
    global_limit=st.integers(1, 5),
)
def test_limiter_multi_bucket_check_is_all_or_nothing(
    ops: list[tuple[float, str]], per_key: int, global_limit: int
) -> None:
    """A denied request consumes no quota from any of its buckets."""
    limiter = FixedWindowLimiter()
    model: dict[str, deque[float]] = {}
    for now, key in ops:
        checks = [(key, per_key), ("global", global_limit)]
        for bucket_key, _ in checks:
            hits = model.setdefault(bucket_key, deque())
            while hits and hits[0] <= now - 60:
                hits.popleft()
        expected = all(len(model[bucket_key]) < limit for bucket_key, limit in checks)
        if expected:
            for bucket_key, _ in checks:
                model[bucket_key].append(now)
        result = limiter.try_acquire(
            [(key, per_key, 60), ("global", global_limit, 60)], now=now
        )
        assert result.allowed is expected
        for bucket_key, _ in checks:
            assert list(limiter._buckets[bucket_key]) == list(model[bucket_key])


_app = Flask(__name__)
_PUBLIC_V4 = st.ip_addresses(v=4).filter(lambda ip: ip.is_global)
_LOOPBACK = st.sampled_from(["127.0.0.1", "::1", "127.0.0.53"])


@given(
    spoofed=st.lists(_PUBLIC_V4.map(str), max_size=3),
    real=_PUBLIC_V4.map(str),
    proxies=st.lists(_LOOPBACK, min_size=1, max_size=3),
)
def test_client_ip_is_the_rightmost_non_proxy_hop(
    spoofed: list[str], real: str, proxies: list[str]
) -> None:
    """Whatever a client writes into X-Forwarded-For ends up left of the hop
    our edge appended; only that appended hop is trusted."""
    header = ", ".join([*spoofed, real, *proxies])
    with _app.test_request_context(
        "/",
        headers={"X-Forwarded-For": header},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    ):
        assert server._flask_client_ip() == real


@given(proxies=st.lists(_LOOPBACK | st.just(""), max_size=3))
def test_client_ip_falls_back_to_socket_address(proxies: list[str]) -> None:
    header = ", ".join(proxies)
    with _app.test_request_context(
        "/",
        headers={"X-Forwarded-For": header},
        environ_base={"REMOTE_ADDR": "10.1.2.3"},
    ):
        assert server._flask_client_ip() == "10.1.2.3"


@given(ip=st.ip_addresses(v=4))
def test_is_public_ip_agrees_for_ipv4_mapped_form(ip: ipaddress.IPv4Address) -> None:
    """An IPv4-mapped IPv6 address must not slip past the SSRF check."""
    import http_fetch

    mapped = ipaddress.IPv6Address(f"::ffff:{ip}")
    assert http_fetch._is_public_ip(str(ip)) == http_fetch._is_public_ip(str(mapped))


@given(
    ip=st.one_of(
        *[
            st.ip_addresses(network=net)
            for net in (
                "10.0.0.0/8",
                "127.0.0.0/8",
                "169.254.0.0/16",
                "172.16.0.0/12",
                "192.168.0.0/16",
                "100.64.0.0/10",
                "fd00::/8",
                "fe80::/10",
            )
        ]
    )
)
def test_is_public_ip_rejects_private_ranges(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> None:
    import http_fetch

    assert not http_fetch._is_public_ip(str(ip))


_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
    lambda inner: (
        st.lists(inner, max_size=3)
        | st.dictionaries(st.text(max_size=8), inner, max_size=3)
    ),
    max_leaves=10,
)
_VALID_EVENT = st.fixed_dictionaries(
    {
        "event_id": st.uuids().map(str),
        "client_session_id": st.uuids().map(str),
        "story_id": st.integers(1, 2**40) | st.integers(-(2**40), -1),
        "event_type": st.sampled_from(["impression", "article_open", "comments_open"]),
        "dashboard_version": st.integers(0, 10**6),
        "position": st.integers(0, 500),
        "sort_mode": st.text(min_size=1, max_size=64),
        "window": st.text(min_size=1, max_size=64),
        "source_filter": st.text(min_size=1, max_size=64),
        "ranker_arm": st.text(min_size=1, max_size=64),
        "occurred_at": st.floats(1, 4e9),
    }
)


@given(
    raw=_JSON
    | _VALID_EVENT.flatmap(
        lambda event: st.dictionaries(
            st.sampled_from(sorted(event)), _JSON, max_size=3
        ).map(lambda patch: {**event, **patch})
    )
)
def test_interaction_event_parser_only_raises_value_error(raw: object) -> None:
    """Untrusted JSON either parses or is rejected with ValueError (a 400),
    never another exception type (a 500)."""
    try:
        server._parse_interaction_event(raw, user_id=1)
    except ValueError:
        pass


@given(event=_VALID_EVENT)
def test_valid_interaction_events_round_trip(event: dict[str, object]) -> None:
    parsed = server._parse_interaction_event(event, user_id=7)
    assert parsed.user_id == 7
    for field in ("event_id", "story_id", "event_type", "position", "sort_mode"):
        assert getattr(parsed, field) == event[field]
    # The window is stored in the ledger's age_filter column.
    assert parsed.age_filter == event["window"]


@pytest.mark.parametrize("deck_size", [0, 2])
@pytest.mark.parametrize("knobs", list(itertools.product([0, 1], repeat=3)))
def test_background_tasks_spawn_exactly_when_they_have_work(
    knobs: tuple[int, int, int], deck_size: int
) -> None:
    """The spawn check and `_warm_background_tasks` agree: a thread starts
    exactly when it would fetch articles or prefetch summaries. (All 16
    on/off combinations: random draws rarely hit "only one knob on".)"""
    from unittest import mock

    import pipeline
    from background_cadence import BackgroundCadence
    from database import Story
    from pipeline import Config, RankedStory, WindowDeck, WindowViews

    fetch, per_view, stale = knobs
    config = Config(
        article_fetch_max_per_run=fetch,
        tldr_prefetch_per_view=per_view,
        tldr_prefetch_stale_per_run=stale,
    )
    deck = WindowDeck(
        {
            "archive": WindowViews(
                recommended=tuple(
                    RankedStory(Story(i, "t", None, 1, 1, "x"), 1.0, "")
                    for i in range(deck_size)
                )
            )
        }
    )
    work: list[str] = []

    async def fetch_bodies(**_: object) -> None:
        work.append("articles")

    async def prefetch(*_: object, **__: object) -> int:
        work.append("summaries")
        return 0

    class Runtime(server.Handler):
        _tldr_prefetch_gate = BackgroundCadence()

    with (
        mock.patch.object(
            pipeline,
            "select_article_fetch_candidates",
            lambda **kw: [r.story for r in kw["ranked"]],
        ),
        mock.patch.object(pipeline, "fetch_and_cache_article_bodies", fetch_bodies),
        mock.patch.object(server, "_prefetch_tldrs_for_ranked", prefetch),
    ):
        Runtime._warm_background_tasks(deck, mock.Mock(), mock.Mock(), config)
    assert bool(work) == server._wants_background_tasks(config, deck)
