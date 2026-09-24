from __future__ import annotations

import asyncio
import time
import unicodedata
import webbrowser
from collections import deque
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit
from uuid import uuid4

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.timer import Timer
from textual.widgets import (
    Button,
    Input,
    Label,
    Markdown,
    OptionList,
    Select,
    Static,
    Tab,
    Tabs,
)

# Private module: Dropdown reuses Select's internals, so pyproject pins
# textual to the tested minor release.
from textual.widgets._select import (
    NonSelectableStatic,
    SelectCurrent,
    SelectOverlay,
)
from textual.widgets.option_list import Option

from .api import (
    API,
    APIError,
    InvalidProfile,
    Impression,
    Profile,
    Summary as SummaryResult,
    TransientError,
    load_profile,
    normalize_server,
    profile_path,
    save_profile,
)
from .models import Feed, FeedStory

DEFAULT_SERVER = "https://ubuntu-8gb-nbg1-1.tailca4726.ts.net:8443/hn/"


def open_in_firefox(url: str) -> None:
    """Open a URL in the running Firefox window, launching one if needed."""
    import shutil
    import subprocess

    firefox = shutil.which("firefox")
    if firefox is None:
        webbrowser.open(url)
        return
    # A missing pgrep/wmctrl or a failed launch must not crash the reader.
    try:
        running = (
            subprocess.run(
                ["pgrep", "-x", "firefox"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).returncode
            == 0
        )
        subprocess.Popen(
            [firefox, "--new-tab", url] if running else [firefox, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        webbrowser.open(url)
        return
    if shutil.which("wmctrl") is not None:
        try:
            subprocess.run(
                ["wmctrl", "-x", "-a", "Navigator.firefox"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            pass  # Focusing the window is cosmetic.


class ArrowLeftCurrent(SelectCurrent):
    """SelectCurrent with the toggle arrow ahead of the label."""

    def compose(self) -> ComposeResult:
        yield NonSelectableStatic("▼", classes="arrow down-arrow")
        yield NonSelectableStatic("▲", classes="arrow up-arrow")
        yield NonSelectableStatic(self.placeholder, id="label")


class Dropdown(Select):
    """Select that reads arrow-first; overlay behavior unchanged."""

    def compose(self) -> ComposeResult:
        yield ArrowLeftCurrent(self.prompt)
        yield SelectOverlay(type_to_search=self._type_to_search).data_bind(
            compact=Select.compact
        )


# Hacker News launched in 2006; earlier timestamps are missing or placeholder data.
EARLIEST_STORY_TIME = 1_136_073_600

# Pause new speculative prefetch after background API errors.
PREFETCH_COOLDOWN_SECONDS = 60.0
DEFAULT_PREFETCH = 20
DEFAULT_PREFETCH_GENERATE = 3
PREFETCH_CONCURRENCY = 4


def story_age(story: FeedStory) -> str:
    """Compact relative age; empty when the timestamp is missing or implausible."""
    if story.time < EARLIEST_STORY_TIME:
        return ""
    seconds = max(0, int(time.time()) - story.time)
    for ceiling, divisor, unit in (
        (3600, 60, "m"),
        (86400, 3600, "h"),
        (2592000, 86400, "d"),
        (31536000, 2592000, "mo"),
    ):
        if seconds < ceiling:
            return f"{seconds // divisor}{unit}"
    return f"{seconds // 31536000}y"


def story_metadata(story: FeedStory) -> str:
    domain = urlsplit(story.article_url).hostname or story.source
    parts = [domain, f"{story.points} pts", f"{story.comments or 0} comments"]
    age = story_age(story)
    if age:
        parts.append(f"{age} ago")
    return " · ".join(parts)


RECOMMENDED_LIMIT = 30


def limit_recommended(
    order: list[int], lookup: dict[int, FeedStory], limit: int = RECOMMENDED_LIMIT
) -> list[int]:
    """First *limit* in rank order, plus any popular stories cut off.

    Popular-flagged stories are never dropped by the truncation.
    """
    head, tail = order[:limit], order[limit:]
    extras = [
        sid for sid in tail if (story := lookup.get(sid)) is not None and story.popular
    ]
    return head + extras


def _cell_len(text: str) -> int:
    """Terminal cell width (wide emoji count double)."""
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text
    )


def _pad_cells(text: str, width: int) -> str:
    padding = width - _cell_len(text)
    return text + " " * padding if padding > 0 else text


def headline_domain(story: FeedStory) -> str:
    if story.source.startswith("rss_reddit_") and len(story.source) > 11:
        return f"r/{story.source[11:]}"
    domain = urlsplit(story.article_url).hostname or story.source
    return domain[4:] if domain.startswith("www.") else domain


def headline_points(story: FeedStory) -> str:
    # Reddit RSS carries no scores (0/8487 rows have one): 0 means unknown,
    # not zero. The web card already hides zero scores; match that here.
    if story.points > 0 or not story.source.startswith("rss_reddit_"):
        return f"▲ {story.points}"
    return ""


def headline(
    story: FeedStory,
    selected: bool | None = None,
    widths: tuple[int, int, int] = (0, 0, 0),
) -> Text:
    """Headline with column-aligned `·` separators when *widths* is given.

    *widths* holds the (domain, points, comments) segment widths across the
    visible list; each row pads its segments so the separators line up.
    Unknown Reddit scores pad as blank space to preserve the columns.
    """
    text = Text()
    if selected is not None:
        text.append("> " if selected else "  ", style="bold #FF914D")
    if story.badges:
        text.append(" ".join(story.badges) + " ")
    text.append(
        story.title, style="bold #EEE8DD" if selected is not False else "#D2CCC1"
    )
    text.append("\n")
    domain = headline_domain(story)
    if widths[0] and _cell_len(domain) > widths[0]:
        domain = domain[: widths[0] - 1] + "…" if widths[0] > 1 else "…"
    elif widths[0]:
        domain = _pad_cells(domain, widths[0])
    text.append(domain, style="#8AB4F8")
    points = headline_points(story)
    if points or widths[1]:
        text.append(" · ", style="#6B655D")
        text.append(
            _pad_cells(points, widths[1]) if widths[1] else points,
            style="#A8C7A0",
        )
    comments = f"💬 {story.comments or 0}"
    age = story_age(story)
    text.append(" · ", style="#6B655D")
    if age and widths[2]:
        comments = _pad_cells(comments, widths[2])
    text.append(comments, style="#C6C1B8")
    if age:
        text.append(" · ", style="#6B655D")
        text.append(age, style="#8F897F")
    return text


EMPTY_NOTICE = (
    "# Nothing here\n\nNo stories in this filter. "
    "Change filters or press **r** to refresh."
)


def feed_failure_notice(detail: str) -> str:
    return f"# Could not reach server\n\n{detail}\n\nPress **r** to retry."


class Summary(Markdown):
    can_focus = True
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("down", "scroll_down", "Down"),
        ("up", "scroll_up", "Up"),
        ("pagedown", "page_down", "Page down"),
        ("pageup", "page_up", "Page up"),
    ]


class Setup(ModalScreen[Profile | None]):
    CSS = """
    Setup { align: center middle; background: rgba(14,14,14,0.7); color: #EEE8DD; }
    #setup { width: 70; max-width: 95%; height: auto; max-height: 100%;
             overflow-y: auto; padding: 1 2; background: #1C1B19;
             border: round #44403B; }
    #setup-title { text-style: bold; }
    #setup-message { height: auto; margin: 1 0; color: #AAA399; }
    #setup-message.error { color: #FFB4A6; }
    .setup-section { margin-top: 1; text-style: bold; }
    Setup Input { margin: 0 0 1 0; background: #222222; border: tall #44403B; }
    Setup Input:focus { border: tall #FF914D; }
    Setup Button { width: 1fr; background: #292724; color: #EEE8DD; border: none; }
    Setup Button:focus { background: #2E2B27; color: #FF914D; text-style: bold; }
    #quit { margin-top: 1; background: #1C1B19; color: #AAA399; }
    """

    def __init__(
        self,
        path: Path,
        server: str | None,
        message: str = "",
        explicit_server: str | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.server = server
        self.message = message
        self.explicit_server = explicit_server
        self.pending = False

    def compose(self) -> ComposeResult:
        with Vertical(id="setup"):
            yield Label("HN Rerank", id="setup-title")
            yield Static(
                self.message
                or "Import a profile link, or enter a server URL and create a new profile.",
                id="setup-message",
                markup=False,
            )
            yield Label("Import an existing profile", classes="setup-section")
            yield Input(placeholder="https://host/hn/u/TOKEN", password=True, id="link")
            yield Button("Import profile", id="import")
            yield Label("Use an existing token", classes="setup-section")
            yield Input(placeholder="Profile token", password=True, id="token")
            yield Button("Use token", id="use-token")
            yield Label("Start a new profile", classes="setup-section")
            yield Button("Create new profile", id="create")
            yield Button("Quit", id="quit")

    @work(exclusive=True)
    async def connect(self, mode: str) -> None:
        api: API | None = None
        try:
            if mode == "create":
                api = API(self.server or DEFAULT_SERVER)
                profile = await api.create()
            elif mode == "use-token":
                profile = Profile(
                    self.server or DEFAULT_SERVER,
                    self.query_one("#token", Input).value.strip(),
                )
                api = API(profile.server, profile.token)
                profile = await api.validate()
            else:
                profile = Profile.from_link(self.query_one("#link", Input).value)
                if (
                    self.explicit_server
                    and normalize_server(self.explicit_server) != profile.server
                ):
                    raise ValueError(
                        "Profile link does not match --server. Use the matching deployment."
                    )
                api = API(profile.server, profile.token)
                profile = await api.validate()
            await (
                api.feed()
            )  # Verify API compatibility before persisting the credential.
            await asyncio.to_thread(save_profile, profile, self.path)
            self.dismiss(profile)
        except (APIError, ValueError, OSError) as exc:
            message = (
                str(exc)
                if not isinstance(exc, OSError)
                else "Could not save profile configuration. Check directory permissions."
            )
            self.query_one("#setup-message", Static).update(
                message + " Check your details and try again."
            )
            self.query_one("#setup-message").add_class("error")
        finally:
            if api:
                await api.close()
            self.pending = False
            if self.is_mounted:
                for button in self.query(Button):
                    button.disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        elif event.button.id in ("import", "use-token", "create") and not self.pending:
            self.pending = True
            for button in self.query(Button):
                button.disabled = True
            self.query_one("#setup-message", Static).update("Connecting…")
            self.query_one("#setup-message").remove_class("error")
            self.connect(str(event.button.id))


class Reader(App[None]):
    TITLE = "HN Rerank"
    CSS = """
    Screen { background: #171717; color: #EEE8DD; }
    #filters { height: 2; align-vertical: top; }
    Select { width: auto; height: auto; display: none; }
    .filter-caption { width: auto; height: 1; padding: 0 1 0 2; color: #8F897F; display: none; }
    .narrow .filter-caption { display: block; }
    SelectCurrent { background: transparent; border: none; height: 1; width: auto; padding: 0 2; }
    SelectCurrent .arrow { padding: 0 1 0 0; }
    SelectCurrent Static#label { width: auto; }
    Select:focus-within > SelectCurrent { background: #FF914D; }
    Select:focus-within Static#label { color: #171717; }
    Select:focus-within .arrow { color: #171717; }
    .narrow #filters { height: 1; }
    Tabs { width: auto; }
    #sort-tabs { width: 58; }
    #age-tabs { width: 20; }
    Tab { color: #AAA399; padding: 0 1; }
    Tab.-active { color: #FF914D; text-style: bold; }
    Tabs:focus Tab.-active { text-style: bold underline; }
    Underline > .underline--bar { color: #FF914D; background: #171717; }
    #panes { height: 1fr; }
    #headlines { width: 1fr; height: 1fr; background: #171717;
                 border: solid #171717; padding: 0; }
    #headlines:focus { border: solid #FF914D; }
    #headlines > .option-list--option { padding: 0 1; }
    #headlines > .option-list--option-highlighted {
        background: #5A3A12; color: #FFFFFF; text-style: bold;
    }
    #headlines:focus > .option-list--option-highlighted { text-style: none;
        border-left: solid #FF914D; }
    #reading-pane { width: 2fr; height: 1fr; border-left: solid #44403B;
                    max-width: 100; }
    #reading-pane.has-story:focus-within { border-left: solid #FF914D; }
    #story-heading { height: auto; max-height: 6; padding: 0 1;
                     border-bottom: solid #2A2825; }
    #summary { width: 1fr; height: 1fr; padding: 0 2; overflow-y: auto;
               background: #171717; color: #EEE8DD; }
    MarkdownH1, MarkdownH2, MarkdownH3, MarkdownH4, MarkdownH5, MarkdownH6 {
        margin: 1 0 0 0; padding: 0;
        border: none; background: #171717; color: #EEE8DD; text-style: bold;
        content-align: left top; }
    MarkdownParagraph, MarkdownBulletList, MarkdownOrderedList { margin: 0; }
    MarkdownBlockQuote { border-left: solid #AAA399; background: #222222; margin: 0 0 1 0; }
    MarkdownFence { background: #222222; margin: 0 0 1 0; padding: 1; }
    #summary MarkdownBlock > .strong { color: #FF914D; text-style: bold; }
    #summary MarkdownBlock > .em { color: #FF914D; }
    #footer { dock: bottom; layout: vertical; height: auto; background: #1D1C1A;
              border-top: solid #2A2825; }
    #status { width: 1fr; height: auto; max-height: 3; padding: 0 1; color: #AAA399; }
    #status.context { color: #C6C1B8; }
    #status.error { color: #FFB4A6; text-style: bold; }
    #shortcuts { width: 1fr; height: auto; padding: 0 1; color: #8F897F; }
    .narrow Tabs { display: none; }
    .narrow Select { display: block; }
    .narrow #panes { layout: vertical; }
    .narrow #headlines { width: 1fr; height: 1fr; }
    .narrow #reading-pane { width: 1fr; height: 3fr; border-left: none;
                            border-top: solid #44403B; }
    /* The wide focus rule outranks `.narrow #reading-pane`; cancel its left edge. */
    .narrow #reading-pane.has-story:focus-within { border-top: solid #FF914D;
                                                   border-left: none; }
    .reading #headlines { display: none; }
    .reading #panes { align-horizontal: center; }
    .reading #reading-pane { width: 1fr; max-width: 100; }
    .narrow.reading #reading-pane { height: 1fr; }
    """
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("j", "move(1)", "Down"),
        ("k", "move(-1)", "Up"),
        ("1", "vote('up')", "+"),
        ("2", "vote('neutral')", "~"),
        ("3", "vote('down')", "−"),
        ("u", "undo", "Undo"),
        ("o", "open_url('article_url')", "Article"),
        ("c", "open_url('comments_url')", "Comments"),
        ("r", "refresh", "Refresh"),
        ("s", "cycle_sort", "Sort"),
        ("enter", "read", "Read"),
        ("escape", "headlines", "Back"),
        ("?", "help", "Help"),
        ("b", "badge_legend", "Badges"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        server: str | None = None,
        config_path: Path | None = None,
        api: API | None = None,
        prefetch: int = DEFAULT_PREFETCH,
        prefetch_generate: int = DEFAULT_PREFETCH_GENERATE,
    ) -> None:
        super().__init__()
        self.register_theme(
            Theme(
                name="editorial",
                primary="#AAA399",
                secondary="#AAA399",
                accent="#FF914D",
                foreground="#EEE8DD",
                background="#171717",
                surface="#222222",
                panel="#292724",
                error="#FFB4A6",
                success="#A8C7A0",
                warning="#E5C07B",
                dark=True,
                variables={
                    "scrollbar": "#44403B",
                    "scrollbar-hover": "#6B655D",
                    "scrollbar-active": "#FF914D",
                    "scrollbar-background": "#1D1C1A",
                    "scrollbar-background-hover": "#1D1C1A",
                    "scrollbar-background-active": "#1D1C1A",
                },
            )
        )
        self.theme = "editorial"
        self.explicit_server = normalize_server(server) if server else None
        self.server = self.explicit_server or DEFAULT_SERVER
        self.config_path = config_path or profile_path()
        self.api = api
        self.feed: Feed | None = None
        self.stories: list[FeedStory] = []
        # Headline row state as last rendered: column widths and marked story.
        self._row_widths: tuple[int, int, int] | None = None
        self._marked_id: int | None = None
        self.rated: set[int] = set()
        self.unavailable: set[int] = set()
        self.restored: dict[int, FeedStory] = {}
        self.history: list[FeedStory] = []
        self.pending = False
        self.target: int | None = None
        self.selection_serial = 0
        self.interaction_session = str(uuid4())
        self.summary_story_id: int | None = None
        self.force_summary_id: int | None = None
        self.reading = False
        self.can_read = False
        self._read_timers: list[Timer] = []
        self.help_open = False
        self.setting_up = False
        self.status_mode = "context"
        self.last_error: str | None = None
        self.prefetch = max(0, prefetch)
        self.prefetch_generate = min(self.prefetch, max(0, prefetch_generate))
        self.prefetch_requests: dict[int, asyncio.Task[SummaryResult | None]] = {}
        self.prefetch_cache_misses: set[int] = set()
        self.summaries: dict[int, str] = {}
        self.prefetching: set[int] = set()
        self.prefetch_queue: deque[int] = deque()
        self.prefetch_retry_at: dict[int, float] = {}
        self.prefetch_active = False
        self.closing = False
        self.prefetch_cooldown_until = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(id="filters"):
            yield Tabs(
                *(
                    Tab(s.title(), id=f"sort-{s}")
                    for s in ("recommended", "popular", "explore", "date")
                ),
                id="sort-tabs",
            )
            yield Tabs(
                Tab("Recent", id="age-recent"),
                Tab("Archive", id="age-archive"),
                id="age-tabs",
            )
            yield Static("Sort", classes="filter-caption")
            yield Dropdown(
                [(s.title(), s) for s in ("recommended", "popular", "explore", "date")],
                value="recommended",
                allow_blank=False,
                id="sort",
            )
            yield Static("Age", classes="filter-caption")
            yield Dropdown(
                [("Recent", "recent"), ("Archive", "archive")],
                value="recent",
                allow_blank=False,
                id="age",
            )
        with Horizontal(id="panes"):
            yield OptionList(id="headlines")
            with Vertical(id="reading-pane"):
                yield Static("", id="story-heading", markup=False)
                yield Summary("Connecting…", id="summary", open_links=False)
        with Horizontal(id="footer"):
            yield Static("Loading…", id="status", markup=False)
            yield Static("", id="shortcuts", markup=False)

    def on_mount(self) -> None:
        self.layout_panes()
        self.set_interval(1.0, self.refresh_read_state)
        self.set_interval(60.0, self.poll_feed_version)
        self.query_one("#headlines", OptionList).focus()
        self.start()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if self.setting_up or isinstance(self.focused, Input):
            return False
        # Zoom is available for any selected story.
        if (
            action == "read"
            and not self.reading
            and (not self.can_read or self.help_open)
        ):
            return False
        # A focused selector owns typing keys, but focus movement and quit must
        # stay reachable or the keyboard gets stuck on the dropdown.
        return not (
            isinstance(self.focused, Select)
            and action
            in {"move", "vote", "undo", "read", "refresh", "open_url", "cycle_sort"}
        )

    def status(self, message: str, *, error: bool = False) -> None:
        """Show a transient message; context_status() restores the counts line."""
        self.status_mode = "error" if error else "message"
        widget = self.query_one("#status", Static)
        widget.update(("✗ " if error else "") + message)
        widget.set_class(error, "error")
        widget.set_class(False, "context")

    def context_status(self) -> None:
        """Counts line for the current filter; only replaces an earlier counts line."""
        if self.status_mode != "context" or not self.query("#status"):
            return
        counts = self.feed.feedback_counts if self.feed else {}
        line = Text()
        line.append(f"{len(self.stories)} shown", style="#C6C1B8")
        line.append(" · ", style="#6B655D")
        line.append(f"+{counts.get('up', 0)}", style="#A8C7A0")
        line.append(" ", style="#6B655D")
        line.append(f"~{counts.get('neutral', 0)}", style="#E5C07B")
        line.append(" ", style="#6B655D")
        line.append(f"−{counts.get('down', 0)}", style="#FFB4A6")
        widget = self.query_one("#status", Static)
        widget.update(line)
        widget.set_class(False, "error")
        widget.set_class(True, "context")

    def show_failure(self, detail: str) -> None:
        """Failure copy in the reading pane when no feed has loaded yet."""
        self.last_error = detail
        if self.feed is not None or not self.query("#summary"):
            return
        self.query_one("#story-heading", Static).update("")
        self.summary_story_id = None
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.query_one("#summary", Markdown).update(feed_failure_notice(detail))
        self.schedule_read_state()

    @work(group="startup", exclusive=True)
    async def start(self) -> None:
        try:
            if self.api is None:
                profile = load_profile(self.config_path)
                # Only an explicit --server overrides the saved profile's server;
                # the default must not force setup for a profile saved elsewhere.
                if profile is None or (
                    self.explicit_server and self.explicit_server != profile.server
                ):
                    self.setup()
                    return
                self.api = API(profile.server, profile.token)
            await self.api.validate()
            self.refresh_feed()
        except InvalidProfile as exc:
            self.setup(str(exc))
        except APIError as exc:
            self.status(str(exc) + " Press r to retry.", error=True)
            self.show_failure(str(exc))

    def setup(self, message: str = "") -> None:
        if self.setting_up:
            return
        self.setting_up = True
        self.help_open = False
        self.summary_story_id = None
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.workers.cancel_group(self, "prefetch")
        self.workers.cancel_group(self, "refresh")
        self.workers.cancel_group(self, "vote")
        self.workers.cancel_group(self, "impression")
        self.pending = False
        self.push_screen(
            Setup(
                self.config_path,
                self.server,
                message,
                explicit_server=self.explicit_server,
            ),
            self.connected,
        )

    async def connected(self, profile: Profile | None) -> None:
        if profile is None:
            self.exit()
            return
        if self.api:
            await self.api.close()
        self.api = API(profile.server, profile.token)
        self.interaction_session = str(uuid4())
        self.feed = None
        self.rated.clear()
        self.unavailable.clear()
        self.restored.clear()
        self.history.clear()
        self.summaries.clear()
        self.prefetch_queue.clear()
        self.prefetching.clear()
        self.prefetch_retry_at.clear()
        self.prefetch_cache_misses.clear()
        self.prefetch_cooldown_until = 0.0
        self.workers.cancel_group(self, "prefetch")
        self.target = None
        self.help_open = False
        self.summary_story_id = None
        self.setting_up = False
        self.refresh_feed()

    def selected(self) -> FeedStory | None:
        index = self.query_one("#headlines", OptionList).highlighted
        return (
            self.stories[index]
            if index is not None and index < len(self.stories)
            else None
        )

    def meta_widths(self, available: int = 0) -> tuple[int, int, int]:
        """Per-segment widths so headline `·` separators share columns.

        When *available* (metadata content width) is positive, the domain
        column is capped with ellipsis so the whole row fits on one line.
        """
        widths = [
            max((_cell_len(headline_domain(s)) for s in self.stories), default=0),
            max((_cell_len(headline_points(s)) for s in self.stories), default=0),
            max(
                (_cell_len(f"💬 {s.comments or 0}") for s in self.stories),
                default=0,
            ),
        ]
        age_w = max((len(story_age(s)) for s in self.stories), default=0)
        total = widths[0] + 3 + widths[1] + 3 + widths[2] + 3 + age_w
        if available > 0 and total > available and widths[0] > 8:
            widths[0] = max(8, widths[0] - (total - available))
        return (widths[0], widths[1], widths[2])

    def rebuild(self, select_id: int | None = None) -> None:
        # Teardown removes nodes before the final messages drain; ignore late
        # rebuilds rather than raising NoMatches.
        if not self.query("#headlines"):
            return
        old = self.selected()
        if select_id is None and old:
            select_id = old.id
        sort = self.query_one("#sort", Select).value
        age = self.query_one("#age", Select).value
        lookup = {story.id: story for story in self.feed.stories} if self.feed else {}
        order = self.feed.orders.get(f"{sort}:{age}", []) if self.feed else []
        if sort == "recommended":
            order = limit_recommended(order, lookup)
        self.stories = [
            lookup[sid]
            for sid in order
            if sid not in self.rated and sid not in self.unavailable
        ]
        headlines = self.query_one("#headlines", OptionList)
        headlines.clear_options()
        # Option padding (1 each side) plus the 2-cell selection marker.
        widths = self.meta_widths(max(0, headlines.size.width - 4))
        self._row_widths = widths
        self._marked_id = select_id
        headlines.add_options(
            [
                Option(
                    headline(s, s.id == select_id, widths),
                    id=str(s.id),
                )
                for s in self.stories
            ]
        )
        if self.stories:
            headlines.highlighted = next(
                (i for i, s in enumerate(self.stories) if s.id == select_id), 0
            )
            if select_id == -1:  # Filter change: no story has this ID.
                headlines.scroll_home(animate=False)
            # Fresh feed data can carry new points/comments; keep the reading
            # heading in step even when the selection id has not changed.
            if selected := self.selected():
                self.query_one("#story-heading", Static).update(headline(selected))
            self.schedule_summary()
        else:
            self.query_one("#story-heading", Static).update("")
            self.summary_story_id = None
            self.selection_serial += 1
            self.workers.cancel_group(self, "summary")
            self.query_one("#summary", Markdown).update(
                feed_failure_notice(self.last_error)
                if self.feed is None and self.last_error
                else EMPTY_NOTICE
            )
            self.schedule_read_state()
        self.query_one("#reading-pane").set_class(bool(self.stories), "has-story")
        self.context_status()

    def on_select_changed(self, event: Select.Changed) -> None:
        # Queued changes can outlive their value after rapid sort cycling.
        # Replaying them into Tabs would start an endless two-way echo.
        if event.value != event.select.value:
            return
        tabs_id = f"#{event.select.id}-tabs"
        if event.select.id in {"sort", "age"} and self.query(tabs_id):
            self.query_one(tabs_id, Tabs).active = f"{event.select.id}-{event.value}"
        self.rebuild(select_id=-1)

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if not event.tab.id or event.tab.id != event.tabs.active:
            return
        group, value = event.tab.id.split("-", 1)
        # Teardown and the first layout pass can activate a tab before the
        # matching selector is queryable; ignore rather than raise NoMatches.
        if not self.query(f"#{group}"):
            return
        self.query_one(f"#{group}", Select).value = value

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if self.setting_up or not self.query("#headlines"):
            return
        listing = self.query_one("#headlines", OptionList)
        widths = self.meta_widths(max(0, listing.size.width - 4))
        current = self.selected()
        current_id = current.id if current else None
        # Only the old and new marker rows change, unless a resize moved the
        # column widths; re-rendering every row on each keypress is wasted work.
        if widths == self._row_widths:
            changed = {self._marked_id, current_id}
            rows = [s for s in self.stories if s.id in changed]
        else:
            rows = self.stories
        for story in rows:
            listing.replace_option_prompt(
                str(story.id), headline(story, story.id == current_id, widths)
            )
        self._row_widths = widths
        self._marked_id = current_id
        self.schedule_summary()

    def schedule_summary(self) -> None:
        story = self.selected()
        if self.help_open:
            return
        if story and story.id != self.summary_story_id:
            self.query_one("#story-heading", Static).update(headline(story))
            self.summary_story_id = story.id
            self.selection_serial += 1
            if self.feed is not None:
                self.record_impression(
                    Impression(
                        event_id=str(uuid4()),
                        client_session_id=self.interaction_session,
                        story_id=story.id,
                        dashboard_version=self.feed.version,
                        position=self.stories.index(story),
                        sort_mode=str(self.query_one("#sort", Select).value),
                        age_filter=str(self.query_one("#age", Select).value),
                        occurred_at=time.time(),
                    ),
                    self.selection_serial,
                )
            force_refresh = story.id == self.force_summary_id
            self.force_summary_id = None
            cached = None if force_refresh else self.summaries.get(story.id)
            if cached is not None:
                self.query_one("#summary", Markdown).update(cached)
                self.schedule_prefetch()
            else:
                if story.id not in self.summaries:
                    self.query_one("#summary", Markdown).update("Loading summary…")
                self.load_summary(
                    story.id, self.selection_serial, force_refresh=force_refresh
                )
            self.query_one("#summary", Markdown).scroll_home(animate=False)
            self.schedule_read_state()

    @work(group="impression", exclusive=True)
    async def record_impression(self, event: Impression, serial: int) -> None:
        # Only a selected card visible for >=1s counts, not prefetched stories.
        api = self.api
        await asyncio.sleep(1.0)
        selected = self.selected() if self.query("#headlines") else None
        if (
            api is None
            or api is not self.api
            or serial != self.selection_serial
            or self.setting_up
            or self.help_open
            or selected is None
            or selected.id != event.story_id
            or self.feed is None
            or self.feed.version != event.dashboard_version
            or str(self.query_one("#sort", Select).value) != event.sort_mode
            or str(self.query_one("#age", Select).value) != event.age_filter
        ):
            return
        try:
            await api.impression(event)
        except APIError:
            # Best effort; telemetry must never interrupt reading or voting.
            pass

    @work(group="summary", exclusive=True)
    async def load_summary(
        self, story_id: int, serial: int, *, force_refresh: bool = False
    ) -> None:
        await asyncio.sleep(0.3)
        if (
            not self.api
            or serial != self.selection_serial
            or not self.query("#summary")
        ):
            return
        previous = self.summaries.get(story_id)
        if previous is None:
            self.query_one("#summary", Markdown).update("Loading summary…")
        else:
            self.query_one("#summary", Markdown).update(previous)
            self.status("Regenerating summary…")
        self.schedule_read_state()
        self.schedule_prefetch()
        try:
            # Reuse background work when navigation catches up with it.
            request = self.prefetch_requests.get(story_id)
            if not force_refresh and story_id in self.summaries:
                summary = SummaryResult(self.summaries[story_id])
            elif not force_refresh and request is not None:
                summary = await asyncio.shield(request)
                if summary is None:
                    summary = await self.api.summary(story_id)
            else:
                summary = await self.api.summary(story_id, force_refresh=force_refresh)
            if serial == self.selection_serial and self.query("#summary"):
                if summary.empty:
                    # Server found nothing summarizable: same session-hide
                    # as undisplayable failures, never rendered or cached.
                    self._hide_story(story_id, "no summarizable content")
                    return
                if not summary.provisional:
                    self.summaries[story_id] = summary.text
                self.query_one("#summary", Markdown).update(summary.text)
                if summary.provisional:
                    self.status(
                        "Summary may be outdated or incomplete. Press r to retry."
                    )
                elif force_refresh:
                    self.status("Summary regenerated.")
                self.schedule_read_state()
                self.schedule_prefetch()
        except InvalidProfile as exc:
            self.setup(str(exc))
        except APIError as exc:
            if serial == self.selection_serial and self.query("#summary"):
                if previous is not None:
                    self.query_one("#summary", Markdown).update(previous)
                    self.status(
                        f"Kept previous summary — refresh failed ({exc}).", error=True
                    )
                    self.schedule_read_state()
                    return
                if isinstance(exc, TransientError):
                    # A dropped connection or rate limit says nothing about
                    # this story; hiding here would drain the deck while
                    # the user keeps moving through a cooldown.
                    self.query_one("#summary", Markdown).update(
                        f"# Summary unavailable\n\n{exc}\n\n"
                        "Move away and back, or press **r**, to try again."
                    )
                    self.status(str(exc), error=True)
                    self.schedule_read_state()
                    return
                # Undisplayable summaries leave the deck: the failure may be
                # transient (quota/cooldown), so this hides for the session
                # only — refresh restores. InvalidProfile goes to setup above.
                self._hide_story(story_id, f"summary unavailable ({exc})")

    def _hide_story(self, story_id: int, reason: str) -> None:
        """Drop a story from the deck for this session; refresh restores."""
        self.unavailable.add(story_id)
        current = [s.id for s in self.stories]
        try:
            advance: int | None = current[current.index(story_id) + 1]
        except (ValueError, IndexError):
            advance = next((sid for sid in reversed(current) if sid != story_id), None)
        self.rebuild(select_id=advance)
        self.status(f"Skipped story {story_id} — {reason}. r restores hidden stories.")

    def prefetch_targets(self, depth: int) -> list[int]:
        """Upcoming stories, nearby history, and entry points into other sorts."""
        if depth <= 0 or not self.feed or not self.query("#headlines"):
            return []
        story = self.selected()
        if story is None:
            return []
        index = self.stories.index(story)
        ids = [item.id for item in self.stories[index + 1 : index + 1 + depth]]
        ids.extend(
            item.id for item in reversed(self.stories[max(0, index - 3) : index])
        )
        lookup = {item.id: item for item in self.feed.stories}
        age = self.query_one("#age", Select).value
        sort = self.query_one("#sort", Select).value
        for other in self.SORT_CYCLE:
            if other == sort:
                continue
            order = self.feed.orders.get(f"{other}:{age}", [])
            if other == "recommended":
                order = limit_recommended(order, lookup)
            eligible = [
                sid
                for sid in order
                if sid not in self.rated and sid not in self.unavailable
            ]
            ids.extend(eligible[: min(depth, 3)])
        return list(
            dict.fromkeys(
                sid
                for sid in ids
                if sid != story.id
                and sid not in self.rated
                and sid not in self.unavailable
            )
        )

    def schedule_prefetch(self) -> None:
        """Refill the bounded rolling window, retaining cached summaries across sorts."""
        if self.prefetch <= 0 or not self.api or self.setting_up:
            return
        now = time.monotonic()
        if now < self.prefetch_cooldown_until:
            return
        generate = set(self.prefetch_targets(self.prefetch_generate))
        self.prefetch_queue = deque(
            sid
            for sid in self.prefetch_targets(self.prefetch)
            if sid not in self.summaries
            and sid not in self.prefetching
            and (
                self.prefetch_retry_at.get(sid, 0.0) <= now
                or (sid in self.prefetch_cache_misses and sid in generate)
            )
        )
        self.start_prefetch()

    def start_prefetch(self) -> None:
        if (
            self.prefetch_active
            or self.closing
            or not self.prefetch_queue
            or self.setting_up
            or not self.is_mounted
        ):
            return
        self.prefetch_active = True
        self.prefetch_summaries()

    async def fetch_prefetched_summary(self, story_id: int) -> SummaryResult | None:
        api = self.api
        if api is None:
            return None
        summary = await api.cached_summary(story_id)
        if summary is None:
            self.prefetch_cache_misses.add(story_id)
            # Recheck the current window after I/O: a sort change can abandon it.
            if (
                story_id in self.prefetch_targets(self.prefetch_generate)
                and time.monotonic() >= self.prefetch_cooldown_until
            ):
                summary = await api.summary(story_id)
        return summary

    async def prefetch_loop(self) -> None:
        while (
            self.prefetch_queue
            and self.api is not None
            and time.monotonic() >= self.prefetch_cooldown_until
        ):
            story_id = self.prefetch_queue.popleft()
            if story_id in self.summaries or story_id in self.prefetching:
                continue
            self.prefetching.add(story_id)
            request = asyncio.create_task(self.fetch_prefetched_summary(story_id))
            self.prefetch_requests[story_id] = request
            try:
                summary = await request
            except APIError:
                # Already-running requests may finish; no new work during cooldown.
                self.prefetch_cooldown_until = (
                    time.monotonic() + PREFETCH_COOLDOWN_SECONDS
                )
                self.prefetch_queue.clear()
                return
            finally:
                self.prefetching.discard(story_id)
                self.prefetch_requests.pop(story_id, None)
            if summary is None or summary.provisional or summary.empty:
                self.prefetch_retry_at[story_id] = (
                    time.monotonic() + PREFETCH_COOLDOWN_SECONDS
                )
                if summary is not None:
                    self.prefetch_cache_misses.discard(story_id)
            else:
                self.prefetch_cache_misses.discard(story_id)
                self.summaries[story_id] = summary.text

    @work(group="prefetch")
    async def prefetch_summaries(self) -> None:
        try:
            async with asyncio.TaskGroup() as group:
                for _ in range(PREFETCH_CONCURRENCY):
                    group.create_task(self.prefetch_loop())
        finally:
            self.prefetch_active = False
            if (
                self.prefetch_queue
                and not self.closing
                and not self.setting_up
                and self.is_mounted
                and time.monotonic() >= self.prefetch_cooldown_until
            ):
                self.start_prefetch()

    def can_poll_feed(self) -> bool:
        return bool(
            self.api
            and self.feed
            and not self.setting_up
            and not self.pending
            and not self.reading
            and not self.help_open
            and not any(
                worker.group in {"refresh", "vote"} and worker.is_running
                for worker in self.workers
            )
        )

    async def poll_feed_version(self) -> None:
        """Observe published versions without interrupting reading or voting."""
        if not self.can_poll_feed():
            return
        api, feed = self.api, self.feed
        assert api is not None and feed is not None
        try:
            _, current = await api.ready(feed.version)
        except APIError:
            # Passive checks must not replace a usable deck with an error.
            # Manual refresh retains its visible error/retry behavior.
            return
        if (
            self.api is api
            and self.feed is feed
            and self.can_poll_feed()
            and current != feed.version
        ):
            # A lower version is a server restart, not an obsolete response.
            # Drop local summaries too: the new generation may have new text.
            # Hidden stories stay hidden; only a manual r restores them.
            self.action_refresh(force_summary=False, restore_hidden=False)

    @work(group="refresh", exclusive=True)
    async def refresh_feed(self) -> None:
        if not self.api or self.setting_up:
            return
        self.status("Refreshing…")
        try:
            for attempt in range(30):
                feed = await self.api.feed()
                self.feed = feed
                if feed.ready:
                    self.restored.clear()
                else:
                    for restored in self.restored.values():
                        self.restore_story(restored)
                # Versions are process-local; a lower target is a valid server reset.
                if self.target is None or feed.target_version < self.target:
                    self.target = feed.target_version
                else:
                    self.target = max(self.target, feed.target_version)
                self.rebuild()
                if feed.ready and feed.version >= self.target:
                    self.status_mode = "context"
                    self.last_error = None
                    self.context_status()
                    return
                self.status("Showing available stories while ranking updates…")
                await asyncio.sleep(1)
                ready, current = await self.api.ready(self.target)
                self.target = min(self.target, current)
                if not ready and attempt == 29:
                    self.status("Ranking is still updating. Press r to check again.")
        except InvalidProfile as exc:
            self.setup(str(exc))
        except APIError as exc:
            self.status(
                ("Showing stale stories. " if self.feed else "")
                + str(exc)
                + " Press r to retry.",
                error=True,
            )
            self.show_failure(str(exc))

    SORT_CYCLE: ClassVar[tuple[str, ...]] = (
        "recommended",
        "popular",
        "explore",
        "date",
    )

    def action_cycle_sort(self) -> None:
        """Advance the sort selector one step (wraps to recommended)."""
        select = self.query_one("#sort", Select)
        try:
            index = self.SORT_CYCLE.index(str(select.value))
        except ValueError:
            index = -1
        select.value = self.SORT_CYCLE[(index + 1) % len(self.SORT_CYCLE)]

    def action_refresh(
        self, *, force_summary: bool = True, restore_hidden: bool = True
    ) -> None:
        story = self.selected()
        self.force_summary_id = story.id if force_summary and story else None
        self.help_open = False
        self.summary_story_id = None
        # Invalidate an older request immediately, not only after feed refresh.
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        previous = self.summaries.get(story.id) if story and force_summary else None
        self.summaries.clear()
        if previous is not None and story is not None:
            self.summaries[story.id] = previous
        if restore_hidden:
            self.unavailable.clear()
        self.prefetch_queue.clear()
        self.prefetch_retry_at.clear()
        self.prefetch_cache_misses.clear()
        self.prefetch_cooldown_until = 0.0
        self.workers.cancel_group(self, "prefetch")
        self.refresh_feed()

    def action_move(self, delta: int) -> None:
        if self.reading:
            self.query_one("#summary", Markdown).scroll_relative(
                y=delta * 3, animate=False
            )
            return
        listing = self.query_one("#headlines", OptionList)
        if self.stories:
            listing.highlighted = max(
                0, min(len(self.stories) - 1, (listing.highlighted or 0) + delta)
            )

    def action_vote(self, action: str) -> None:
        story = self.selected()
        if story and not self.pending:
            self.pending = True
            self.submit(story, action)

    def action_undo(self) -> None:
        if self.history and not self.pending:
            self.pending = True
            self.submit(self.history[-1], "clear")

    @work(group="vote")
    async def submit(self, story: FeedStory, action: str) -> None:
        if not self.api:
            self.pending = False
            return
        self.workers.cancel_group(self, "refresh")
        self.status("Saving vote…")
        try:
            self.target = await self.api.vote(story.id, action)
            if action == "clear":
                self.history.pop()
                self.rated.discard(story.id)
                self.restored[story.id] = story
                self.restore_story(story)
                self.rebuild(story.id)
            else:
                index = next(
                    (i for i, item in enumerate(self.stories) if item.id == story.id), 0
                )
                remaining = [item for item in self.stories if item.id != story.id]
                next_id = (
                    remaining[min(index, len(remaining) - 1)].id if remaining else None
                )
                self.rated.add(story.id)
                self.restored.pop(story.id, None)
                self.history.append(story)
                self.rebuild(next_id)
            self.status("Vote cleared." if action == "clear" else "Vote saved.")
            self.refresh_feed()
        except InvalidProfile as exc:
            self.setup(str(exc))
        except APIError as exc:
            self.status(
                str(exc)
                + " Vote not confirmed; r refreshes, then check before voting again.",
                error=True,
            )
        finally:
            self.pending = False

    def restore_story(self, story: FeedStory) -> None:
        if not self.feed:
            return
        if all(item.id != story.id for item in self.feed.stories):
            self.feed.stories.append(story)
        for age in ("recent", "archive"):
            for sort in ("recommended", "popular", "explore", "date"):
                if (
                    f"{age}_mixed" in story.memberships
                    and (sort != "popular" or story.popular)
                    and (sort != "explore" or story.explore)
                ):
                    order = self.feed.orders.setdefault(f"{sort}:{age}", [])
                    if story.id not in order:
                        order.insert(0, story.id)

    def action_open_url(self, field: str) -> None:
        story = self.selected()
        url = getattr(story, field, "") if story else ""
        if urlsplit(url).scheme in {"http", "https"}:
            open_in_firefox(url)
        else:
            self.status("No link available for this story.")

    def layout_panes(self, width: int | None = None) -> None:
        narrow = (self.size.width if width is None else width) < 100
        self.set_class(narrow, "narrow")
        self.set_class(self.reading, "reading")
        votes = "1 up · 2 neutral · 3 down → next story"
        if narrow:
            # Narrow hints may wrap; the badge key stays listed in ? help.
            if self.reading:
                hints = f"j/k scroll · Enter/Esc back · {votes}"
            elif self.can_read:
                hints = f"Enter zoom · {votes} · ? help"
            else:
                hints = f"j/k move · {votes} · ? help"
        elif self.reading:
            hints = (
                f"j/k scroll · Enter/Esc back · {votes} · b badges · ? help · q quit"
            )
        elif self.can_read:
            hints = f"j/k move · Enter zoom · {votes} · b badges · ? help · q quit"
        else:
            hints = f"j/k move · {votes} · b badges · ? help · q quit"
        self.query_one("#shortcuts", Static).update(hints)

    def schedule_read_state(self) -> None:
        """Refresh zoom availability after deferred selection and content updates."""
        self.call_after_refresh(self.refresh_read_state)
        for timer in self._read_timers:
            timer.stop()
        self._read_timers = [
            self.set_timer(delay, self.refresh_read_state) for delay in (0.1, 0.3, 0.6)
        ]

    def refresh_read_state(self) -> None:
        """Offer zoom whenever a story is selected, regardless of summary length."""
        if self.reading or not self.query("#summary") or not self.query("#headlines"):
            return
        can_read = self.selected() is not None
        if can_read != self.can_read:
            self.can_read = can_read
            self.layout_panes()

    def on_resize(self, event: events.Resize) -> None:
        if self.query("#panes"):
            self.layout_panes(event.size.width)
            self.schedule_read_state()

    def focus_summary(self) -> None:
        self.query_one("#summary", Markdown).focus()

    def action_read(self) -> None:
        if not self.can_read and not self.reading:
            return
        self.reading = not self.reading
        self.layout_panes()
        if self.reading:
            self.query_one("#summary", Markdown).focus()
        else:
            self.query_one("#headlines", OptionList).focus()
            self.schedule_read_state()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_read()

    def action_headlines(self) -> None:
        was_reading = self.reading
        self.reading = False
        self.layout_panes()
        self.query_one("#headlines", OptionList).focus()
        if self.help_open:
            self.help_open = False
            self.schedule_summary()
        elif was_reading:
            self.schedule_read_state()

    def action_badge_legend(self) -> None:
        self.help_open = True
        self.summary_story_id = None
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.query_one("#summary", Markdown).update(
            "# Badge legend\n\n"
            "🔥 **Hot** — rising fast · 🏆 **Top** — high score · 💬 **Talk** — many comments\n\n"
            "🤔 **Unsure** — model uncertain · ✨ **Novel** — unlike your votes · 🎯 **Similar** — matches your upvotes\n\n"
            "Escape: return to the story. ?: shortcuts."
        )
        self.focus_summary()
        self.schedule_read_state()

    def action_help(self) -> None:
        self.help_open = True
        self.summary_story_id = None
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.query_one("#summary", Markdown).update(
            "# Shortcuts\n\nj/k: move the headline list. Arrows: scroll the focused pane. Tab: focus. Escape: close this help.\n\nEnter: hide the article list and zoom the TLDR pane. Enter or Escape: return to the article list. In zoom mode, j/k scroll the TLDR. 1/2/3: up / neutral / down. u: undo latest vote. o/c: article / comments. r: refresh and regenerate selected summary. s: cycle sort. b: badge legend. ?: this help. q: quit.\n\nUse the selectors for Recommended, Popular, Explore, Date and Recent / Archive. Votes are never automatically retried after network errors."
        )
        self.focus_summary()
        self.schedule_read_state()

    async def on_unmount(self) -> None:
        self.closing = True
        self.prefetch_queue.clear()
        self.selection_serial += 1
        self.workers.cancel_all()
        if self.api:
            await self.api.close()
