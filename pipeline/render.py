from __future__ import annotations

from datetime import datetime
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from database import Database, Story
from .config import (
    BQ_ARCHIVE_SOURCE,
    CH_ARCHIVE_SOURCE,
    Config,
)
from .ranking import RankedStory


@dataclass(frozen=True)
class BadgeView:
    kind: str
    icon: str
    label: str
    tooltip: str

    @property
    def css_class(self) -> str:
        return f"badge badge--{self.kind}"


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
    is_hn_attr: str
    sort_popular_attr: str
    sort_explore_attr: str
    is_recent_attr: str
    article_url: str
    comments_url: str
    source_label: str
    time_ago: str
    show_source_badge: bool
    show_score: bool


def time_ago_filter(seconds: int) -> str:
    diff = int(time.time()) - seconds
    if diff < 0:
        return "now"
    if diff < 60:
        return f"{diff}s ago"
    minutes = diff // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


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
                is_hn_attr="0" if item.is_non_hn else "1",
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
                is_recent_attr="1" if item.is_recent else "0",
                article_url=story.url or "",
                comments_url=story.discussion_url or "",
                source_label=source_label_filter(story.source),
                time_ago=time_ago_filter(story.time),
                show_source_badge=story.source != "hn",
                show_score=story.score > 0,
            )
        )
    return cards


def _build_tab_groups() -> tuple[TabGroupView, ...]:
    return (
        TabGroupView(
            key="sort",
            aria_label="Sort order",
            css_class="tab-bar tab-bar--sort",
            data_attr="sort",
            segmented=False,
            tabs=(
                TabView("recommended", "<u>R</u>ecommended", True),
                TabView("popular", "<u>P</u>opular"),
                TabView("explore", "E<u>x</u>plore"),
                TabView("date", "<u>D</u>ate"),
            ),
        ),
        TabGroupView(
            key="age",
            aria_label="Age filter",
            css_class="tab-bar tab-bar--segmented",
            data_attr="age",
            segmented=True,
            tabs=(
                TabView("recent", "R<u>e</u>cent", True),
                TabView("archive", "<u>A</u>rchive"),
            ),
        ),
        TabGroupView(
            key="source",
            aria_label="Source filter",
            css_class="tab-bar tab-bar--segmented",
            data_attr="source",
            segmented=True,
            tabs=(
                TabView("mixed", "<u>M</u>ixed", True),
                TabView("hn", "<u>H</u>N"),
                TabView("non-hn", "<u>N</u>on-HN"),
            ),
        ),
    )


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
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    env.filters["time_ago"] = time_ago_filter
    env.filters["source_label"] = source_label_filter

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

    template = env.get_template("index.html")
    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        cards=_build_dashboard_cards(ranked, hot_badge_percentile=hot_badge_percentile),
        tab_groups=_build_tab_groups(),
        server_port=config.server_port,
        pico_css=pico_css,
        user_id=user_id,
        user_token=user_token,
        vote_counts=vote_counts,
        dashboard_version=dashboard_version or 0,
        dashboard_latest_version=dashboard_latest_version or 0,
    )
    return html_content.encode("utf-8")
