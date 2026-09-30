"""Card links from untrusted feeds: only http(s) may reach href/window.open.
Window feeds follow the view rules both clients rely on."""

from __future__ import annotations

import json

from hypothesis import given, settings, strategies as st

import numpy as np

from clients.tui.src.hn_rerank.models import VIEWS, WINDOWS, Feed
from database import Story
from pipeline import Config, RankedStory, is_hn_source
from pipeline.ranking import (
    EXPLORE_PER_BADGE,
    GRAVITY_TIME_SCALE,
    VIEW_SIZE,
    WINDOW_SECONDS,
    ExploreContext,
    assemble_window_deck,
    hn_gravity,
    in_window,
    serve_window,
)
from pipeline.render import _web_url, build_feed

_SCHEMES = st.sampled_from(
    ["http", "https", "javascript", "data", "vbscript", "file", "ftp", "blob"]
)


def _browser_scheme(url: str) -> str | None:
    """Spec (WHATWG URL parsing, the parts that matter for a scheme): strip
    leading/trailing C0 controls and spaces, drop every tab/CR/LF, then the
    scheme is an ASCII letter followed by letters, digits, + - . up to ':'."""
    url = url.strip("".join(map(chr, range(0x21))))
    url = url.replace("\t", "").replace("\n", "").replace("\r", "")
    head, sep, _ = url.partition(":")
    if not sep or not head or not head[0].isascii() or not head[0].isalpha():
        return None
    if not all(c.isascii() and (c.isalnum() or c in "+-.") for c in head):
        return None
    return head.lower()


@st.composite
def _hostile_urls(draw: st.DrawFn) -> str:
    scheme = draw(_SCHEMES)
    # Mixed case and embedded tab/newline: browsers still read the scheme.
    scheme = "".join(
        draw(st.sampled_from([c.lower(), c.upper()]))
        + draw(st.sampled_from(["", "", "\t", "\n"]))
        for c in scheme
    )
    lead = draw(st.text(alphabet=" \t\n\x00\x01\x1f", max_size=3))
    rest = draw(st.sampled_from(["//example.com/a", "alert(1)", "text/html,<b>"]))
    return lead + scheme + ":" + rest


@given(st.one_of(_hostile_urls(), st.text(max_size=30)))
def test_card_links_are_http_or_empty(url: str) -> None:
    out = _web_url(url)
    assert out in ("", url)  # never rewritten, only dropped
    if out:
        assert _browser_scheme(out) in ("http", "https")


@given(_hostile_urls())
def test_real_web_links_survive(url: str) -> None:
    if _browser_scheme(url) in ("http", "https"):
        assert _web_url(url) == url


_NOW = 1_800_000_000.0
_DAY = 86400


@st.composite
def _pools(draw: st.DrawFn) -> tuple[list[RankedStory], ExploreContext | None]:
    """Scored candidate pools as the ranker emits them: stories spread over
    two months (so every window, and both sides of every boundary, get
    some), HN and non-HN, with coarse scores so ties are common, and the
    per-candidate similarity arrays Explore reads (absent on the cold path)."""
    ids = draw(st.lists(st.integers(1, 10**6), unique=True, max_size=120))
    boundaries = [12 * 3600, _DAY, 7 * _DAY, 30 * _DAY]
    personalized = draw(st.booleans())
    pool = []
    for story_id in ids:
        age = draw(
            st.one_of(
                st.integers(0, 60 * _DAY),
                st.sampled_from(boundaries).map(lambda b: b + draw(st.integers(-2, 2))),
            )
        )
        story = Story(
            id=story_id,
            title=f"Story {story_id}",
            url=draw(st.none() | st.just(f"https://example.com/{story_id}")),
            score=draw(st.integers(0, 500)),
            time=int(_NOW) - age,
            text_content="",
            source=draw(st.sampled_from(["hn", "hn", "rss_example_com", "ch_seed"])),
            comment_count=draw(st.integers(0, 600)),
        )
        prob = draw(st.floats(0.01, 0.98)) if personalized else None
        pool.append(
            RankedStory(
                story=story,
                score=draw(st.integers(-3, 3)) / 2,
                best_match_title="",
                prob_down=prob,
                prob_neutral=None if prob is None else (1 - prob) / 2,
                prob_up=None if prob is None else (1 - prob) / 2,
            )
        )
    if not personalized:
        return pool, None
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    n_interests = int(rng.integers(0, 6))
    return pool, ExploreContext(
        cand_max_sim=rng.random(len(pool), dtype=np.float32),
        cand_interest=rng.integers(0, max(n_interests, 1), len(pool)),
        interest_sizes=rng.integers(1, 50, n_interests),
        row_of={r.story.id: i for i, r in enumerate(pool)},
    )


_POPULAR_ICONS = {"\U0001f525", "\U0001f3c6", "\U0001f4ac"}  # 🔥 🏆 💬
_EXPLORE_KINDS = ("uncertain", "novel", "interest")


@settings(max_examples=80, deadline=None)
@given(
    pool_and_context=_pools(),
    served_later=st.sampled_from([0, 3600, 5 * _DAY]),
    version=st.integers(0, 10**13),
    target=st.integers(0, 10**13),
)
def test_window_views_hold_only_their_window_and_keep_their_orders(
    pool_and_context: tuple[list[RankedStory], ExploreContext | None],
    served_later: int,
    version: int,
    target: int,
) -> None:
    pool, context = pool_and_context
    config = Config()
    deck = assemble_window_deck(pool, config=config, now=_NOW, explore=context)
    now = _NOW + served_later
    for window in WINDOWS:
        feed = build_feed(
            deck,
            window,
            config,
            {"up": 1, "neutral": 0, "down": 2},
            version,
            target,
            now=now,
        )
        by_id = {story.id: story for story in feed.stories}
        assert feed.window == window and set(feed.orders) == set(VIEWS)
        assert (feed.version, feed.target_version, feed.ready) == (
            version,
            target,
            version >= target,
        )
        # Orders are authoritative: every sent story is in one, each once.
        assert set(by_id) == {sid for order in feed.orders.values() for sid in order}
        for name, order in feed.orders.items():
            assert len(order) == len(set(order)) <= VIEW_SIZE, name
            # Every story lies inside the window at serve time.
            assert all(in_window(by_id[sid].time, window, now) for sid in order)

        recommended = feed.orders["recommended"]
        scores = [by_id[sid].rank_score for sid in recommended]
        assert scores == sorted(scores, reverse=True)
        if not served_later:
            # Pure model score over the window, no source quota.
            in_w = [r for r in pool if in_window(r.story.time, window, now)]
            best = sorted(in_w, key=lambda r: r.score, reverse=True)[:VIEW_SIZE]
            assert recommended == [r.story.id for r in best]

        popular = feed.orders["popular"]
        assert all(is_hn_source(by_id[sid].source) for sid in popular)
        gravity = [
            hn_gravity(
                by_id[sid].points, by_id[sid].time, _NOW, GRAVITY_TIME_SCALE[window]
            )
            for sid in popular
        ]
        assert gravity == sorted(gravity, reverse=True)
        for sid in popular:
            icons = _POPULAR_ICONS & set(by_id[sid].badges)
            assert len(icons) == 1, by_id[sid].badges
            story = by_id[sid]
            if "\U0001f525" not in icons:  # not Hot: Talk iff comments >= points
                assert ("\U0001f4ac" in icons) == (
                    (story.comments or 0) >= story.points
                )

        explore = feed.orders["explore"]
        if context is None:
            assert explore == []  # cold: nothing to personalize against
        assert not set(explore) & set(recommended)
        scores = [by_id[sid].rank_score for sid in explore]
        assert scores == sorted(scores, reverse=True)
        for kind in _EXPLORE_KINDS:
            assert (
                sum(
                    any(b.kind == kind for b in by_id[sid].badge_details)
                    for sid in explore
                )
                <= EXPLORE_PER_BADGE
            )

        # What the server sends is what the TUI parses.
        assert Feed.parse(json.loads(json.dumps(feed.to_dict()))) == feed


def test_window_boundaries_are_inclusive_and_archive_is_the_rest() -> None:
    now = _NOW
    for window, seconds in WINDOW_SECONDS.items():
        assert in_window(int(now) - seconds, window, now)
        assert not in_window(int(now) - seconds - 1, window, now)
    month = WINDOW_SECONDS["1m"]
    assert not in_window(int(now) - month, "archive", now)
    assert in_window(int(now) - month - 1, "archive", now)
    # A story from the future (clock skew) is brand new, not archived.
    assert in_window(int(now) + 60, "12h", now)
    assert not in_window(int(now) + 60, "archive", now)


@given(age=st.integers(-2 * _DAY, 400 * _DAY), now=st.integers(10**9, 2 * 10**9))
def test_windows_are_nested_and_archive_is_exactly_the_rest(age: int, now: int) -> None:
    member = [in_window(now - age, window, float(now)) for window in WINDOWS]
    *nested, archive = member
    # 12h in 1d in 1w in 1m: once a story is in a window, it is in every longer one.
    assert nested == sorted(nested)
    assert archive != nested[-1]


def _is_subsequence(part: list[int], whole: list[int]) -> bool:
    items = iter(whole)
    return all(x in items for x in part)


@settings(max_examples=60, deadline=None)
@given(
    pool_and_context=_pools(),
    data=st.data(),
    served_later=st.integers(0, 40 * _DAY),
)
def test_votes_and_ageing_only_remove_stories_never_add_or_reorder(
    pool_and_context: tuple[list[RankedStory], ExploreContext | None],
    data: st.DataObject,
    served_later: int,
) -> None:
    pool, context = pool_and_context
    deck = assemble_window_deck(pool, config=Config(), now=_NOW, explore=context)
    ids = sorted(r.story.id for r in pool)
    voted = set(data.draw(st.lists(st.sampled_from(ids), unique=True)) if ids else [])
    hidden = deck.without(voted)
    now = _NOW + served_later
    for window in WINDOWS:
        cached = deck.window(window)
        served_all = serve_window(cached, window, now)
        served = serve_window(hidden.window(window), window, now)
        for view in VIEWS:
            before = [r.story.id for r in cached.view(view)]
            after = [r.story.id for r in hidden.window(window).view(view)]
            # A vote hides the story in every window and keeps the rest in order.
            assert after == [sid for sid in before if sid not in voted]
            shown = [r.story.id for r in served.view(view)]
            assert not set(shown) & voted
            assert set(shown) <= set(before)
            if view != "explore":  # Explore is re-sorted by score when served
                assert _is_subsequence(shown, before)
            # Ageing drops stories that left the window; nothing else joins.
            assert all(
                in_window(r.story.time, window, now) for r in served_all.view(view)
            )


def test_an_empty_window_is_served_empty_not_widened() -> None:
    old = RankedStory(
        Story(1, "Old", None, 50, int(_NOW) - 3 * _DAY, "", comment_count=1),
        1.0,
        "",
    )
    deck = assemble_window_deck([old], config=Config(), now=_NOW)
    for window in ("12h", "1d"):
        feed = build_feed(deck, window, Config(), {}, 1, 1, now=_NOW)
        assert feed.stories == [] and all(o == [] for o in feed.orders.values())
    assert build_feed(deck, "1w", Config(), {}, 1, 1, now=_NOW).orders[
        "recommended"
    ] == [1]
    # Served a day later, a story that aged out of a window leaves it.
    later = _NOW + 5 * _DAY
    assert build_feed(deck, "1w", Config(), {}, 1, 1, now=later).stories == []


def test_a_week_of_popular_favours_its_big_stories_over_fresh_small_ones() -> None:
    fresh = Story(1, "Fresh", None, 100, int(_NOW) - 3 * 3600, "", source="hn")
    big = Story(2, "Big", None, 1000, int(_NOW) - 3 * _DAY, "", source="hn")
    pool = [RankedStory(fresh, 0.0, ""), RankedStory(big, 0.0, "")]
    deck = assemble_window_deck(pool, config=Config(), now=_NOW)
    assert [r.story.id for r in deck.window("1w").popular] == [2, 1]
    assert [r.story.id for r in deck.window("1d").popular] == [1]
    # HN's own clock (1 hour) would put the fresh story first in every window.
    assert hn_gravity(100, fresh.time, _NOW) > hn_gravity(1000, big.time, _NOW)
