"""Feed wire contract; deliberately free of terminal/backend dependencies."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, get_args

# Feed schema number. 2 (2026-09-28): one time window per response, with
# "recommended" / "popular" / "explore" orders; no memberships, no Date.
FEED_API_VERSION = 2

# Time windows, nested by age ("1d" includes the "12h" stories; "1m" is the
# last 30 days), plus "archive": older than 30 days.
Window = Literal["12h", "1d", "1w", "1m", "archive"]
WINDOWS: tuple[Window, ...] = get_args(Window)
DEFAULT_WINDOW: Window = "1w"
WINDOW_LABELS: dict[Window, str] = {
    "12h": "12 hours",
    "1d": "1 day",
    "1w": "1 week",
    "1m": "1 month",
    "archive": "Archive",
}
# Popular's gravity clock per window (the server's GRAVITY_TIME_SCALE): age
# counts in units of this many hours. Used to re-insert an undone story.
GRAVITY_TIME_SCALE: dict[Window, float] = {
    "12h": 4.0,
    "1d": 8.0,
    "1w": 56.0,
    "1m": 360.0,
    "archive": 2920.0,
}
# The three views of a window, the keys of Feed.orders.
View = Literal["recommended", "popular", "explore"]
VIEWS: tuple[View, ...] = get_args(View)


# C0 controls except tab/newline, DEL, and C1 controls. Server text (titles
# from third-party feeds, LLM summaries) is untrusted; ESC/CSI/OSC sequences
# would otherwise reach the terminal raw (Rich only strips a few of these).
_TERMINAL_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def terminal_safe(text: str) -> str:
    """Drop control characters that a terminal would interpret."""
    return _TERMINAL_CONTROLS.sub("", text)


def _terminal_safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return terminal_safe(value)
    if isinstance(value, list):
        return [
            terminal_safe(item) if isinstance(item, str) else item for item in value
        ]
    return value


@dataclass(frozen=True)
class FeedBadge:
    kind: str
    icon: str
    label: str
    tooltip: str


@dataclass(frozen=True)
class FeedStory:
    id: int
    title: str
    article_url: str
    comments_url: str
    source: str
    points: int
    comments: int | None
    time: int
    rank_score: float
    badges: list[str] = field(default_factory=list)
    badge_details: list[FeedBadge] = field(default_factory=list)
    best_match_title: str = ""
    source_label: str = ""
    domain: str = ""
    enriched: bool = False


# Unknown keys are ignored so a server can add optional fields within an
# api_version without breaking already-installed clients.
_STORY_FIELDS = frozenset(f.name for f in fields(FeedStory))
_BADGE_FIELDS = frozenset(f.name for f in fields(FeedBadge))


@dataclass(frozen=True)
class Feed:
    """One time window of a deck: its stories and, per view, their order."""

    api_version: int
    window: Window
    stories: list[FeedStory]
    orders: dict[str, list[int]]
    feedback_counts: dict[str, int]
    version: int
    target_version: int
    ready: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def parse(cls, data: Any) -> Feed:
        if not isinstance(data, dict):
            # Parse failures are ValueError by contract (tests/test_boundaries.py).
            raise ValueError("Invalid feed response")  # noqa: TRY004
        if data.get("api_version") != FEED_API_VERSION:
            raise ValueError("Unsupported feed API; update hn-rerank.")
        try:
            stories = []
            for raw_story in data["stories"]:
                story_data: dict[str, Any] = {
                    key: _terminal_safe_value(value)
                    for key, value in dict(raw_story).items()
                    if key in _STORY_FIELDS
                }
                story_data["badge_details"] = [
                    FeedBadge(
                        **{
                            key: _terminal_safe_value(value)
                            for key, value in dict(badge).items()
                            if key in _BADGE_FIELDS
                        }
                    )
                    for badge in story_data.get("badge_details", [])
                ]
                stories.append(FeedStory(**story_data))
            result = cls(
                FEED_API_VERSION,
                data["window"],
                stories,
                data["orders"],
                data["feedback_counts"],
                data["version"],
                data["target_version"],
                data["ready"],
            )
            if (
                type(result.version) is not int
                or result.version < 0
                or type(result.target_version) is not int
                or result.target_version < 0
                or type(result.ready) is not bool
                or result.window not in WINDOWS
            ):
                raise ValueError("Invalid ranking state")
            for story in stories:
                if (
                    type(story.id) is not int
                    or any(
                        not isinstance(value, str)
                        for value in (
                            story.title,
                            story.article_url,
                            story.comments_url,
                            story.source,
                        )
                    )
                    or type(story.points) is not int
                    or type(story.time) is not int
                    or (story.comments is not None and type(story.comments) is not int)
                    or isinstance(story.rank_score, bool)
                    or not isinstance(story.rank_score, (int, float))
                    or not math.isfinite(story.rank_score)
                    or not isinstance(story.badges, list)
                    or any(not isinstance(badge, str) for badge in story.badges)
                    or not isinstance(story.badge_details, list)
                    or any(
                        not isinstance(badge, FeedBadge)
                        or any(
                            not isinstance(value, str)
                            for value in (
                                badge.kind,
                                badge.icon,
                                badge.label,
                                badge.tooltip,
                            )
                        )
                        for badge in story.badge_details
                    )
                    or any(
                        not isinstance(value, str)
                        for value in (
                            story.best_match_title,
                            story.source_label,
                            story.domain,
                        )
                    )
                    or type(story.enriched) is not bool
                ):
                    raise ValueError("Invalid story")
            ids = {story.id for story in stories}
            if not isinstance(result.orders, dict) or any(
                key not in VIEWS
                or not isinstance(order, list)
                or any(type(sid) is not int or sid not in ids for sid in order)
                for key, order in result.orders.items()
            ):
                raise ValueError("Invalid filter order")
            if len(ids) != len(stories) or any(
                type(value) is not int or value < 0
                for value in result.feedback_counts.values()
            ):
                raise ValueError("Invalid feedback counts or duplicate stories")
            return result
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("Invalid feed response") from exc
