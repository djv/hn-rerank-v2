from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
import time

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader

from database import Database, StoryCounts
from clients.tui.src.hn_rerank.models import (
    DEFAULT_WINDOW,
    FEED_API_VERSION,
    VIEWS,
    WINDOW_LABELS,
    Feed,
    FeedBadge,
    FeedStory,
    Window,
)
from .config import (
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_SOURCE,
    Config,
)
from .ranking import RankedStory, WindowDeck, card_attribution, serve_window

# Side-rail sort tabs: (value, label).
SORT_TABS: tuple[tuple[str, str], ...] = (
    ("recommended", "Recommended"),
    ("popular", "Popular"),
    ("explore", "Explore"),
)


@dataclass(frozen=True)
class VoteCountsView:
    up: int
    neutral: int
    down: int


def source_label_filter(source: str) -> str:
    if not source:
        return ""
    if source == "hn":
        return "HN"
    if source == BQ_ARCHIVE_SOURCE:
        return "BQ Seed"
    if source == CH_ARCHIVE_SOURCE:
        return "CH Seed"

    label = source
    if label.startswith("rss_"):
        label = label[4:]
    # Historical rows from feeds hosted at rss.* were stored as rss_rss_*.
    if label.startswith("rss_"):
        label = label[4:]
    if label.startswith("reddit_"):
        subreddit = label[len("reddit_") :]
        return f"r/{subreddit}"

    known = {
        "slashdot_org": "Slashdot",
        "mshibanami_github_io": "GitHub Trending",
        "tildes_net": "Tildes",
        "lesswrong_com": "LessWrong",
        "lobste_rs": "Lobsters",
        "discourse_haskell_org": "Haskell Discourse",
        "latent_space": "Latent Space",
        "ainews": "AINews",
        "scottaaronson_blog": "Scott Aaronson",
        "simonwillison_net": "Simon Willison",
        "lwn_net": "LWN",
        "openai_com": "OpenAI",
        "huggingface_co": "Hugging Face",
        "blog_cloudflare_com": "Cloudflare",
        "blog_janestreet_com": "Jane Street",
        "well-typed_com": "Well-Typed",
        "tweag_io": "Tweag",
        "ocaml_org": "OCaml",
        "quantamagazine_org": "Quanta",
        "www_worksinprogress_news": "Works in Progress",
        "erictopol_substack_com": "Ground Truths",
        "theskepticalcardiologist_substack_com": "Skeptical Cardiologist",
        "sciencebasedmedicine_org": "Science-Based Medicine",
    }
    if label in known:
        return known[label]

    return label.replace("_", ".")


_pico_css_cache: str | None = None


def _get_pico_css() -> str:
    global _pico_css_cache
    if _pico_css_cache is None:
        path = Path("templates/pico.min.css")
        _pico_css_cache = path.read_text(encoding="utf-8") if path.exists() else ""
    return _pico_css_cache


def _web_url(url: str | None) -> str:
    """The URL if it is http(s), else "": card links go into href and
    window.open, and third-party feeds are untrusted (javascript:, data:)."""
    if not url:
        return ""
    try:
        scheme = urlparse(url.strip()).scheme.lower()
    except ValueError:
        return ""
    return url if scheme in ("http", "https") else ""


def _domain_of(*urls: str) -> str:
    """First registrable-looking hostname across the given URLs.

    Lowercased, www-stripped, punycode left as-is (honest, no network).
    Empty when no URL carries a hostname — the template hides the chip.
    """
    for url in urls:
        if not url:
            continue
        try:
            host = urlparse(url).hostname or ""
        except ValueError:
            continue
        host = host.lower().removeprefix("www.")
        if host:
            return host
    return ""


# Badge legend for the side rail: icon + short label per kind, in a stable
# display order. Tooltips stay on the card badges themselves; the legend is
# a reminder, not documentation.
BADGE_LEGEND: tuple[tuple[str, str], ...] = (
    ("🔥", "Hot"),
    ("🏆", "Top"),
    ("💬", "Talk"),
    ("🤔", "Unsure"),
    ("✨", "Novel"),
    ("🎯", "Interest"),
)


def _build_badges(item: RankedStory, *, hot_badge_percentile: int) -> list[FeedBadge]:
    badges: list[FeedBadge] = []
    if item.is_uncertain:
        badges.append(
            FeedBadge(
                kind="uncertain",
                icon="🤔",
                label="Unsure",
                tooltip="Model is highly uncertain about this story (high entropy distribution)",
            )
        )
    if item.is_novel:
        badges.append(
            FeedBadge(
                kind="novel",
                icon="✨",
                label="Novel",
                tooltip="Semantically distant from anything you've voted on",
            )
        )
    if item.is_discussion_rich:
        badges.append(
            FeedBadge(
                kind="talk",
                icon="💬",
                label="Talk-worthy",
                tooltip="At least as many HN comments as points",
            )
        )
    if item.is_high_engagement:
        badges.append(
            FeedBadge(
                kind="top",
                icon="🏆",
                label="Top",
                tooltip="Popular on HN, not rising fast and not mostly discussion",
            )
        )
    if item.is_hot:
        badges.append(
            FeedBadge(
                kind="hot",
                icon="🔥",
                label="Hot",
                tooltip=(
                    f"Top {hot_badge_percentile}% by engagement velocity "
                    "(points/hour) and score ≥ 20"
                ),
            )
        )
    if item.is_interest:
        badges.append(
            FeedBadge(
                kind="interest",
                icon="🎯",
                label="Interest",
                tooltip="Best story from one of your interests that Recommended misses",
            )
        )
    return badges


def _feed_story(
    item: RankedStory, *, hot_badge_percentile: int, counts: StoryCounts | None = None
) -> FeedStory:
    story = item.story
    badges = _build_badges(item, hot_badge_percentile=hot_badge_percentile)
    return FeedStory(
        id=story.id,
        title=story.title,
        article_url=_web_url(story.url),
        comments_url=_web_url(story.discussion_url),
        source=story.source,
        points=story.score if counts is None else counts.score,
        comments=story.comment_count if counts is None else counts.comment_count,
        time=story.time,
        rank_score=item.score,
        badges=[badge.icon for badge in badges],
        badge_details=badges,
        best_match_title=card_attribution(item),
        source_label=source_label_filter(story.source),
        domain=_domain_of(story.url or "", story.discussion_url or ""),
        enriched=len(story.text_content) >= 1000,
    )


def build_feed(
    deck: WindowDeck,
    window: Window,
    config: Config,
    counts: dict[str, int],
    version: int,
    target: int,
    *,
    now: float | None = None,
    live_counts: Callable[[Sequence[int]], Mapping[int, StoryCounts]] | None = None,
) -> Feed:
    """The `/api/feed` snapshot of one window of a deck as served at *now*
    (``serve_window``): its stories and each view's order, which is
    authoritative (Recommended and Explore by model score, Popular by HN
    gravity; clients shuffle Explore themselves).

    Points and comments come from *live_counts* when given: the deck's
    stories are snapshots from the last pool build (hourly regen), and
    counts refreshed since then (hot-thread refresh, TLDR hydration) live
    only in the database. Order and badges stay as ranked."""
    views = serve_window(
        deck.window(window), window, time.time() if now is None else now
    )
    hot_badge_percentile = int(round(config.model.hot_badge_percentile))
    items = views.stories()
    stored = live_counts([item.story.id for item in items]) if live_counts else {}
    return Feed(
        FEED_API_VERSION,
        window,
        [
            _feed_story(
                item,
                hot_badge_percentile=hot_badge_percentile,
                counts=stored.get(item.story.id),
            )
            for item in items
        ],
        {name: [r.story.id for r in views.view(name)] for name in VIEWS},
        counts,
        version,
        target,
        version >= target,
    )


@functools.cache
def _template_env() -> Environment:
    """One Jinja environment per process, so compiled templates are reused
    across renders (a fresh environment recompiled index.html every time).
    FileSystemLoader's default auto_reload still picks up edited templates."""
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["source_label"] = source_label_filter
    return env


def generate_dashboard_bytes(
    deck: WindowDeck,
    config: Config,
    db: Database,
    user_id: int | None = None,
    user_token: str | None = None,
    dashboard_version: int | None = None,
    dashboard_latest_version: int | None = None,
) -> bytes:
    """Render dashboard to bytes without writing to disk."""
    env = _template_env()
    pico_css = _get_pico_css()

    raw_vote_counts = (
        db.count_feedback_by_action(user_id)
        if user_id
        else {"up": 0, "neutral": 0, "down": 0}
    )
    vote_counts = VoteCountsView(
        up=raw_vote_counts["up"],
        neutral=raw_vote_counts["neutral"],
        down=raw_vote_counts["down"],
    )
    # The page carries the default window as the same JSON /api/feed serves;
    # the client builds every card from it.
    feed = build_feed(
        deck,
        DEFAULT_WINDOW,
        config,
        raw_vote_counts,
        dashboard_version or 0,
        dashboard_latest_version or 0,
        live_counts=db.get_story_counts,
    )
    template = env.get_template("index.html")
    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        feed=feed.to_dict(),
        sort_tabs=SORT_TABS,
        windows=tuple(WINDOW_LABELS.items()),
        default_window=DEFAULT_WINDOW,
        badge_legend=BADGE_LEGEND,
        server_port=config.server_port,
        pico_css=pico_css,
        user_id=user_id,
        user_token=user_token,
        vote_counts=vote_counts,
    )
    return html_content.encode("utf-8")
