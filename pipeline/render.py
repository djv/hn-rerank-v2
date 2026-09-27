from __future__ import annotations

import functools

from datetime import datetime
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader

from database import Database, Story
from clients.tui.src.hn_rerank.models import Feed, FeedBadge, FeedStory
from .config import (
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_SOURCE,
    Config,
)
from .ranking import RankedStory

# Recommended shows at most this many cards per age (top by score);
# Popular and Explore keep their own badge quotas. Cards in no view are
# dropped so the server sends a short deck. Date lists every card that is in
# another view, newest first, for both Age tabs. Clients show 12 per view
# (so Date is the 12 newest in the deck); the rest are the backfill that
# slides in as cards are voted.
RECOMMENDED_LIMIT = 24


@dataclass(frozen=True)
class BadgeView:
    kind: str
    icon: str
    label: str
    tooltip: str


@dataclass(frozen=True)
class TabView:
    value: str
    label_html: str
    active: bool = False


@dataclass(frozen=True)
class TabGroupView:
    key: str
    aria_label: str
    css_class: str
    data_attr: str
    segmented: bool
    tabs: tuple[TabView, ...]


@dataclass(frozen=True)
class VoteCountsView:
    up: int
    neutral: int
    down: int


@dataclass(frozen=True)
class DashboardCardView:
    story: Story
    score: float
    best_match_title: str
    badges: tuple[BadgeView, ...]
    combo_keys: str
    is_enriched: bool
    sort_popular_attr: str
    sort_explore_attr: str
    sort_date_attr: str
    sort_recommended_attr: str
    article_url: str
    comments_url: str
    domain: str
    source_label: str


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
    ("🎯", "Similar"),
)


def _build_badges(
    item: RankedStory, *, hot_badge_percentile: int
) -> tuple[BadgeView, ...]:
    badges: list[BadgeView] = []
    if item.is_uncertain:
        badges.append(
            BadgeView(
                kind="uncertain",
                icon="🤔",
                label="Unsure",
                tooltip="Model is highly uncertain about this story (high entropy distribution)",
            )
        )
    if item.is_novel:
        badges.append(
            BadgeView(
                kind="novel",
                icon="✨",
                label="Novel",
                tooltip="Semantically distant from anything you've voted on",
            )
        )
    if item.is_discussion_rich:
        badges.append(
            BadgeView(
                kind="talk",
                icon="💬",
                label="Talk-worthy",
                tooltip="High HN comment count for its age cohort",
            )
        )
    if item.is_high_engagement:
        badges.append(
            BadgeView(
                kind="top",
                icon="🏆",
                label="Top",
                tooltip="High HN score for its age cohort",
            )
        )
    if item.is_hot:
        badges.append(
            BadgeView(
                kind="hot",
                icon="🔥",
                label="Hot",
                tooltip=(
                    f"Top {hot_badge_percentile}% by engagement velocity "
                    "(points/hour) and score ≥ 20"
                ),
            )
        )
    if item.is_similar:
        badges.append(
            BadgeView(
                kind="similar",
                icon="🎯",
                label="Similar",
                tooltip="Most similar to your upvoted stories for its age cohort",
            )
        )
    return tuple(badges)


def _build_dashboard_cards(
    ranked: list[RankedStory], *, hot_badge_percentile: int
) -> list[DashboardCardView]:
    recommended_ids: set[int] = set()
    for age in ("recent", "archive"):
        in_age = [r for r in ranked if f"{age}_mixed" in r.combo_keys.split()]
        in_age.sort(key=lambda r: r.score, reverse=True)
        recommended_ids.update(r.story.id for r in in_age[:RECOMMENDED_LIMIT])
    # Cards outside any age deck (no *_mixed key) aren't subject to the cap.
    recommended_ids.update(
        r.story.id
        for r in ranked
        if not any(key.endswith("_mixed") for key in r.combo_keys.split())
    )
    cards: list[DashboardCardView] = []
    for item in ranked:
        story = item.story
        cards.append(
            DashboardCardView(
                story=story,
                score=item.score,
                best_match_title=item.best_match_title,
                badges=_build_badges(item, hot_badge_percentile=hot_badge_percentile),
                combo_keys=item.combo_keys,
                is_enriched=len(story.text_content) >= 1000,
                sort_popular_attr=(
                    "1"
                    if item.is_hot or item.is_high_engagement or item.is_discussion_rich
                    else "0"
                ),
                sort_explore_attr=(
                    "1"
                    if item.is_uncertain or item.is_similar or item.is_novel
                    else "0"
                ),
                sort_recommended_attr=(
                    "1" if item.story.id in recommended_ids else "0"
                ),
                # Every card kept below is in another view, so in Date.
                sort_date_attr="1",
                article_url=_web_url(story.url),
                comments_url=_web_url(story.discussion_url),
                domain=_domain_of(story.url or "", story.discussion_url or ""),
                source_label=source_label_filter(story.source),
            )
        )
    return [
        c
        for c in cards
        if "1" in (c.sort_recommended_attr, c.sort_popular_attr, c.sort_explore_attr)
    ]


def _build_tab_groups() -> tuple[TabGroupView, ...]:
    # Source filter (Mixed/HN/Non-HN) is temporarily disabled. The claim
    # that non-HN sources are absent from the candidate pool is stale —
    # the RSS/Reddit/LessWrong leg has been enabled since well before this
    # comment was last touched (config.non_hn_candidates_enabled=true;
    # see WORKLOG 2026-08-28/2026-08-30) — but re-enabling this UI still
    # needs client-side work: an Archive+Non-HN selection currently has no
    # matching combo (archive_nonhn is structurally always empty, see
    # PRIMARY_RECENT_NONHN/PRIMARY_ARCHIVE_HN in pipeline/ranking.py) and
    # would need the same kind of guard the client already has for
    # Popular+Non-HN. Deferred (2026-08-30 user decision); re-add the
    # TabGroupView below alongside that client-side guard.
    return (
        TabGroupView(
            key="sort",
            aria_label="Sort order",
            css_class="tab-bar tab-bar--sort",
            data_attr="sort",
            segmented=False,
            tabs=(
                TabView("recommended", "Recommended", True),
                TabView("popular", "Popular"),
                TabView("explore", "Explore"),
                TabView("date", "Date"),
            ),
        ),
        TabGroupView(
            key="age",
            aria_label="Age filter",
            css_class="tab-bar tab-bar--segmented",
            data_attr="age",
            segmented=True,
            tabs=(
                TabView("recent", "Recent", True),
                TabView("archive", "<u>A</u>rchive"),
            ),
        ),
    )


def prepare_feed(
    cards: list[DashboardCardView], counts: dict[str, int], version: int, target: int
) -> Feed:
    stories = [
        FeedStory(
            id=c.story.id,
            title=c.story.title,
            article_url=c.article_url,
            comments_url=c.comments_url,
            source=c.story.source,
            points=c.story.score,
            comments=c.story.comment_count,
            time=c.story.time,
            rank_score=c.score,
            memberships=c.combo_keys.split(),
            popular=c.sort_popular_attr == "1",
            explore=c.sort_explore_attr == "1",
            badges=[badge.icon for badge in c.badges],
            badge_details=[
                FeedBadge(badge.kind, badge.icon, badge.label, badge.tooltip)
                for badge in c.badges
            ],
            best_match_title=c.best_match_title,
            source_label=c.source_label,
            domain=c.domain,
            enriched=c.is_enriched,
        )
        for c in cards
    ]
    recommended_ids = {c.story.id for c in cards if c.sort_recommended_attr == "1"}
    date_ids = {c.story.id for c in cards if c.sort_date_attr == "1"}
    orders: dict[str, list[int]] = {}
    for age in ("recent", "archive"):
        for sort in ("recommended", "popular", "explore", "date"):
            selected = [
                s
                for s in stories
                # Date is one list (the deck, newest first) that ignores the
                # Age axis, so both ages carry it.
                if (
                    s.id in date_ids
                    if sort == "date"
                    else f"{age}_mixed" in s.memberships
                    and (sort != "popular" or s.popular)
                    and (sort != "explore" or s.explore)
                    and (sort != "recommended" or s.id in recommended_ids)
                )
            ]
            selected.sort(
                key=lambda s: s.time if sort == "date" else s.rank_score, reverse=True
            )
            if sort == "explore":
                random.shuffle(selected)
            orders[f"{sort}:{age}"] = [s.id for s in selected]
    return Feed(1, stories, orders, counts, version, target, version >= target)


def build_feed(
    ranked: list[RankedStory],
    config: Config,
    counts: dict[str, int],
    version: int,
    target: int,
) -> Feed:
    """The `/api/feed` snapshot of a deck, without rendering the page."""
    cards = _build_dashboard_cards(
        ranked, hot_badge_percentile=int(round(config.model.hot_badge_percentile))
    )
    return prepare_feed(cards, counts, version, target)


@functools.cache
def _template_env() -> Environment:
    """One Jinja environment per process, so compiled templates are reused
    across renders (a fresh environment recompiled index.html every time).
    FileSystemLoader's default auto_reload still picks up edited templates."""
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["source_label"] = source_label_filter
    return env


def generate_dashboard_bytes(
    ranked: list[RankedStory],
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
    hot_badge_percentile = int(round(config.model.hot_badge_percentile))

    cards = _build_dashboard_cards(ranked, hot_badge_percentile=hot_badge_percentile)
    # The page carries the deck as the same JSON /api/feed serves; the client
    # builds every card from it.
    feed = prepare_feed(
        cards, raw_vote_counts, dashboard_version or 0, dashboard_latest_version or 0
    )
    template = env.get_template("index.html")
    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        feed=feed.to_dict(),
        tab_groups=_build_tab_groups(),
        badge_legend=BADGE_LEGEND,
        server_port=config.server_port,
        pico_css=pico_css,
        user_id=user_id,
        user_token=user_token,
        vote_counts=vote_counts,
    )
    return html_content.encode("utf-8")
