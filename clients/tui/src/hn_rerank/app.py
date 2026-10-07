from __future__ import annotations

import asyncio
import random
import time
import unicodedata
import webbrowser
from dataclasses import replace
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
    Impression,
    InvalidProfile,
    Profile,
    TransientError,
    load_profile,
    load_window,
    normalize_server,
    profile_path,
    save_profile,
    save_window,
)
from .api import Summary as SummaryResult
from .models import (
    DEFAULT_WINDOW,
    GRAVITY_TIME_SCALE,
    WINDOW_LABELS,
    WINDOWS,
    Feed,
    FeedStory,
)

DEFAULT_SERVER = "https://ubuntu-8gb-nbg1-1.tailca4726.ts.net:8443/hn/"

DARK_PALETTE: dict[str, str] = {
    "bg": "#171717",
    "fg": "#EEE8DD",
    "title-dim": "#D2CCC1",
    "soft": "#C6C1B8",
    "muted": "#AAA399",
    "faint": "#8F897F",
    "sep": "#6B655D",
    "accent": "#FF914D",
    "link": "#8AB4F8",
    "good": "#A8C7A0",
    "warn": "#E5C07B",
    "bad": "#FFB4A6",
    "surface": "#222222",
    "panel": "#292724",
    "bar": "#1D1C1A",
    "chrome": "#262420",
    "modal": "#1C1B19",
    "button-focus": "#2E2B27",
    "border": "#44403B",
    "rule": "#2A2825",
    "select-bg": "#5A3A12",
    "select-fg": "#FFFFFF",
    "overlay": "rgba(14,14,14,0.7)",
}
LIGHT_PALETTE: dict[str, str] = {
    "bg": "#FAF7F2",
    "fg": "#1F1D1A",
    "title-dim": "#3A3631",
    "soft": "#4A453F",
    "muted": "#5F5850",
    "faint": "#7A7369",
    "sep": "#A8A195",
    "accent": "#C4520F",
    "link": "#1F5FBF",
    "good": "#2E7D32",
    "warn": "#9A6700",
    "bad": "#B3261E",
    "surface": "#F0EBE3",
    "panel": "#E8E2D8",
    "bar": "#F2EDE5",
    "chrome": "#ECE5D9",
    "modal": "#F2EDE5",
    "button-focus": "#E0D8CB",
    "border": "#CFC7BA",
    "rule": "#E3DDD3",
    "select-bg": "#F6DCC4",
    "select-fg": "#1F1D1A",
    "overlay": "rgba(60,50,40,0.35)",
}
PALETTES: dict[str, dict[str, str]] = {
    "editorial": DARK_PALETTE,
    "editorial-light": LIGHT_PALETTE,
}
# Rich Text styles are baked at render time, so they read the active palette.
PALETTE: dict[str, str] = dict(DARK_PALETTE)
LIGHT_HOURS = range(6, 20)  # Local 06:00-19:59 uses the light theme.


def theme_for_hour(hour: int) -> str:
    return "editorial-light" if hour in LIGHT_HOURS else "editorial"


def editorial_theme(name: str) -> Theme:
    p = PALETTES[name]
    return Theme(
        name=name,
        primary=p["muted"],
        secondary=p["muted"],
        accent=p["accent"],
        foreground=p["fg"],
        background=p["bg"],
        surface=p["surface"],
        panel=p["panel"],
        error=p["bad"],
        success=p["good"],
        warning=p["warn"],
        dark=p is DARK_PALETTE,
        variables={
            "scrollbar": p["border"],
            "scrollbar-hover": p["sep"],
            "scrollbar-active": p["accent"],
            "scrollbar-background": p["bar"],
            "scrollbar-background-hover": p["bar"],
            "scrollbar-background-active": p["bar"],
            **{f"hn-{key}": value for key, value in p.items()},
        },
    )


def copy_with_system_tool(text: str) -> bool:
    """Copy text with wl-copy, xclip, xsel or pbcopy; False when none worked."""
    import os
    import shutil
    import subprocess

    commands = [["pbcopy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]]
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.insert(0, ["wl-copy"])
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(
                command,
                input=text.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return True
    return False


def open_in_browser(url: str) -> None:
    """Open a URL in `$HN_RERANK_BROWSER` (a command, e.g. `surf`) when set,
    else in Chrome, bringing its window forward. A running Chrome adds the
    URL as a tab in its current window; otherwise it starts one."""
    import os
    import shlex
    import shutil
    import subprocess

    if custom := shlex.split(os.environ.get("HN_RERANK_BROWSER", "")):
        try:
            subprocess.Popen(
                [*custom, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError):
            webbrowser.open(url)
        return
    chrome = next(
        (
            path
            for name in ("google-chrome", "google-chrome-stable", "chromium")
            if (path := shutil.which(name))
        ),
        None,
    )
    if chrome is None:
        webbrowser.open(url)
        return
    # A failed launch must not crash the reader.
    try:
        subprocess.Popen(
            [chrome, url],
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
                ["wmctrl", "-x", "-a", "google-chrome.Google-chrome"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass  # Focusing the window is cosmetic.


def dig_deeper_prompt(story: FeedStory) -> str:
    """The opening prompt for a rabbit-hole agent session on one story."""
    links = [
        f"{label}: {url}"
        for label, url in (
            ("Article", story.article_url),
            ("Discussion", story.comments_url),
        )
        if urlsplit(url).scheme in {"http", "https"}
    ]
    return "\n".join(
        [
            f"Dig deeper into this story from my news reader: {story.title}",
            *links,
            "",
            (
                "Read the article and the discussion. Tell me the key ideas, what is"
                " new or surprising, where commenters push back or add expertise, and"
                " background or related work worth reading next. Then wait for my"
                " follow-up questions."
            ),
        ]
    )


def open_agent_session(prompt: str) -> bool:
    """Start `$HN_RERANK_AGENT` (default `claude`) on the prompt in a tmux
    pane split beside the reader's, in the home directory; False outside
    tmux or on failure."""
    import os
    import shlex
    import shutil
    import subprocess

    if not os.environ.get("TMUX"):
        return False
    agent = shlex.split(os.environ.get("HN_RERANK_AGENT", "")) or ["claude"]
    # tmux reports success even when the pane's command cannot start (the
    # pane just closes), so resolve it here, on the reader's PATH.
    executable = shutil.which(agent[0])
    if executable is None:
        return False
    agent[0] = executable
    # With several arguments tmux runs the command directly, so the prompt
    # needs no shell quoting.
    command = ["tmux", "split-window", "-h", "-c", str(Path.home())]
    if pane := os.environ.get("TMUX_PANE"):
        command += ["-t", pane]
    try:
        subprocess.run(
            [*command, *agent, prompt],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


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


# Every sort shows at most this many stories (matches the web client);
# unrated stories past the cap slide in as ones ahead are rated.
VIEW_LIMIT = 12
# Panes shorter than this many rows get one footer row (status and
# "? help") and no "Because you upvoted" line in the reading heading.
COMPACT_HEIGHT = 30


def _cell_len(text: str) -> int:
    """Terminal cell width (wide emoji count double)."""
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text
    )


def fit_middle(text: str, width: int) -> str:
    """*text* in at most *width* cells, cut from the middle with `…`: the
    start names what happened, the end often says what to do."""
    if _cell_len(text) <= width or width < 8:
        return text
    tail_budget = (width - 1) * 3 // 5
    head_budget = width - 1 - tail_budget
    head = ""
    for char in text:
        if _cell_len(head + char) > head_budget:
            break
        head += char
    tail = ""
    for char in reversed(text):
        if _cell_len(char + tail) > tail_budget:
            break
        tail = char + tail
    return head.rstrip() + "…" + tail.lstrip()


def _pad_cells(text: str, width: int) -> str:
    padding = width - _cell_len(text)
    return text + " " * padding if padding > 0 else text


def headline_domain(story: FeedStory) -> str:
    # Feeds name themselves (AINews, not the x.com its items link to), as
    # the web card's source badge does.
    if story.source != "hn" and story.source_label:
        return story.source_label
    if story.source.startswith("rss_reddit_") and len(story.source) > 11:
        return f"r/{story.source[11:]}"
    domain = urlsplit(story.article_url).hostname or story.source
    return domain.removeprefix("www.")


def headline_points(story: FeedStory) -> str:
    # Most feeds carry no scores (Reddit RSS: 0/8487 rows have one), so a
    # non-HN 0 means unknown, not zero. The web card hides zero scores too.
    if story.points > 0 or story.source == "hn":
        return f"▲ {story.points}"
    return ""


def headline_comments(story: FeedStory) -> str:
    # Likewise a non-HN 0 is a feed without comment counts (Slashdot, blogs).
    if story.comments or story.source == "hn":
        return f"💬 {story.comments or 0}"
    return ""


def headline(
    story: FeedStory,
    selected: bool | None = None,
    widths: tuple[int, int, int] = (0, 0, 0),
) -> Text:
    """Headline with column-aligned `·` separators when *widths* is given.

    *widths* holds the (domain, points, comments) segment widths across the
    visible list; each row pads its segments so the separators line up.
    Unknown scores and comment counts pad as blank space, separator included,
    to preserve the columns; without *widths* they are left out.
    """
    text = Text()
    if selected is not None:
        text.append("> " if selected else "  ", style=f"bold {PALETTE['accent']}")
    if story.badges:
        text.append(" ".join(story.badges) + " ")
    text.append(
        story.title,
        style=f"bold {PALETTE['fg']}"
        if selected is not False
        else PALETTE["title-dim"],
    )
    text.append("\n")
    domain = headline_domain(story)
    if widths[0] and _cell_len(domain) > widths[0]:
        domain = domain[: widths[0] - 1] + "…" if widths[0] > 1 else "…"
    elif widths[0]:
        domain = _pad_cells(domain, widths[0])
    text.append(domain, style=PALETTE["link"])
    age = story_age(story)
    segments = (
        (headline_points(story), widths[1], PALETTE["good"]),
        # The last column needs no padding when nothing follows it.
        (headline_comments(story), widths[2] if age else 0, PALETTE["soft"]),
        (age, 0, PALETTE["faint"]),
    )
    for value, width, style in segments:
        if value:
            text.append(" · ", style=PALETTE["sep"])
            text.append(_pad_cells(value, width), style=style)
        elif width:
            text.append(" " * (3 + width))
    return text


def story_heading(story: FeedStory, *, attribution: bool = True) -> Text:
    """The reading pane's heading: the headline, plus why the story is
    recommended when the server says (as the web card does) and there is
    room (*attribution*)."""
    text = headline(story)
    if attribution and story.best_match_title:
        text.append("\nBecause you upvoted: ", style=PALETTE["faint"])
        text.append(story.best_match_title, style=PALETTE["soft"])
    return text


EMPTY_NOTICE = (
    "# Nothing here\n\nNo stories in this view. "
    "Try another sort or time window, or press **r** to refresh."
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
    Setup { align: center middle; background: $hn-overlay; color: $hn-fg; }
    #setup { width: 70; max-width: 95%; height: auto; max-height: 100%;
             overflow-y: auto; padding: 1 2; background: $hn-modal;
             border: round $hn-border; }
    #setup-title { text-style: bold; }
    #setup-message { height: auto; margin: 1 0; color: $hn-muted; }
    #setup-message.error { color: $hn-bad; }
    .setup-section { margin-top: 1; text-style: bold; }
    Setup Input { margin: 0 0 1 0; background: $hn-surface; border: tall $hn-border; }
    Setup Input:focus { border: tall $hn-accent; }
    Setup Button { width: 1fr; background: $hn-panel; color: $hn-fg; border: none; }
    Setup Button:focus { background: $hn-button-focus; color: $hn-accent; text-style: bold; }
    #quit { margin-top: 1; background: $hn-modal; color: $hn-muted; }
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
    Screen { background: $hn-bg; color: $hn-fg; }
    #filters { height: 2; align-vertical: top; background: $hn-chrome; }
    Select { width: auto; height: auto; display: none; }
    .filter-caption { width: auto; height: 1; padding: 0 1 0 2; color: $hn-faint; display: none; }
    .narrow .filter-caption { display: block; }
    SelectCurrent { background: transparent; border: none; height: 1; width: auto; padding: 0 2; }
    SelectCurrent .arrow { padding: 0 1 0 0; }
    SelectCurrent Static#label { width: auto; }
    Select:focus-within > SelectCurrent { background: $hn-accent; }
    Select:focus-within Static#label { color: $hn-bg; }
    Select:focus-within .arrow { color: $hn-bg; }
    .narrow #filters { height: 1; }
    Tabs { width: auto; }
    #sort-tabs { width: 45; }
    #window { display: block; }
    Tab { color: $hn-muted; padding: 0 1; }
    Tab.-active { color: $hn-accent; text-style: bold; }
    Tabs:focus Tab.-active { text-style: bold underline; }
    Underline > .underline--bar { color: $hn-accent; background: $hn-chrome; }
    #panes { height: 1fr; }
    #headlines { width: 1fr; height: 1fr; background: $hn-bg;
                 border: solid $hn-bg; padding: 0; }
    #headlines:focus { border: solid $hn-accent; }
    #headlines > .option-list--option { padding: 0 1; }
    #headlines > .option-list--separator { color: $hn-rule; }
    #headlines > .option-list--option-highlighted {
        background: $hn-select-bg; color: $hn-select-fg; text-style: bold;
    }
    #headlines:focus > .option-list--option-highlighted { text-style: none;
        border-left: solid $hn-accent; }
    #reading-pane { width: 2fr; height: 1fr; border-left: solid $hn-border;
                    max-width: 100; }
    #reading-pane.has-story:focus-within { border-left: solid $hn-accent; }
    #story-heading { height: auto; max-height: 6; padding: 0 1;
                     border-bottom: solid $hn-rule; }
    #summary { width: 1fr; height: 1fr; padding: 0 2; overflow-y: auto;
               background: $hn-bg; color: $hn-fg; }
    MarkdownH1, MarkdownH2, MarkdownH3, MarkdownH4, MarkdownH5, MarkdownH6 {
        margin: 1 0 0 0; padding: 0;
        border: none; background: $hn-bg; color: $hn-fg; text-style: bold;
        content-align: left top; }
    MarkdownParagraph, MarkdownBulletList, MarkdownOrderedList { margin: 0; }
    MarkdownBlockQuote { border-left: solid $hn-muted; background: $hn-surface; margin: 0 0 1 0; }
    MarkdownFence { background: $hn-surface; margin: 0 0 1 0; padding: 1; }
    #summary MarkdownBlock > .strong { color: $hn-accent; text-style: bold; }
    #summary MarkdownBlock > .em { color: $hn-accent; }
    #footer { dock: bottom; layout: horizontal; height: 1; background: $hn-chrome; }
    #status { width: 1fr; height: 1; padding: 0 1; color: $hn-muted;
              text-wrap: nowrap; text-overflow: ellipsis; }
    #status.context { color: $hn-soft; }
    #status.error { color: $hn-bad; text-style: bold; }
    #shortcuts { width: auto; height: 1; padding: 0 1; color: $hn-faint;
                 text-wrap: nowrap; }
    .narrow Tabs { display: none; }
    .narrow Select { display: block; }
    .narrow #panes { layout: vertical; }
    .narrow #headlines { width: 1fr; height: 1fr; }
    .narrow #reading-pane { width: 1fr; height: 3fr; border-left: none;
                            border-top: solid $hn-border; }
    /* The wide focus rule outranks `.narrow #reading-pane`; cancel its left edge. */
    .narrow #reading-pane.has-story:focus-within { border-top: solid $hn-accent;
                                                   border-left: none; }
    .reading #headlines { display: none; }
    .reading #panes { align-horizontal: center; }
    .reading #reading-pane { width: 1fr; max-width: 100; }
    .narrow.reading #reading-pane { height: 1fr; }
    """
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("j", "move(1)", "Down"),
        ("k", "move(-1)", "Up"),
        ("space", "page_summary", "Page down"),
        ("1", "vote('up')", "+"),
        ("2", "vote('neutral')", "~"),
        ("3", "vote('down')", "−"),
        ("u", "undo", "Undo"),
        ("o", "open_url('article_url')", "Article"),
        ("c", "open_url('comments_url')", "Comments"),
        ("y", "copy_url", "Copy link"),
        ("a", "dig_deeper", "Ask Claude"),
        ("r", "refresh", "Refresh"),
        ("s", "cycle_sort", "Sort"),
        ("h", "cycle_sort(-1)", "Prev sort"),
        ("l", "cycle_sort(1)", "Next sort"),
        ("d", "cycle_window(1)", "Window"),
        ("D", "cycle_window(-1)", "Prev window"),
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
        window_file: Path | None = None,
    ) -> None:
        super().__init__()
        # The last window picked opens next time; None (tests) keeps 1w and
        # never writes.
        self.window_file = window_file
        self.saved_window = (
            load_window(window_file) if window_file else None
        ) or DEFAULT_WINDOW
        for name in PALETTES:
            self.register_theme(editorial_theme(name))
        self.clock_theme = ""
        self.apply_clock_theme()
        self.explicit_server = normalize_server(server) if server else None
        self.server = self.explicit_server or DEFAULT_SERVER
        self.config_path = config_path or profile_path()
        self.api = api
        # The selected window's feed; None while a window switch fetches it.
        self.feed: Feed | None = None
        # Feeds of the current version by window: the selected one and
        # neighbours prefetched in the background.
        self.feeds: dict[str, Feed] = {}
        # Neighbour prefetches tried, by (window, version): at most once each.
        self.window_attempts: set[tuple[str, int]] = set()
        self.window_prefetching = False
        self.stories: list[FeedStory] = []
        # Headline row state as last rendered: column widths and marked story.
        self._row_widths: tuple[int, int, int] | None = None
        self._marked_id: int | None = None
        self.rated: set[int] = set()
        self.unavailable: set[int] = set()
        # Explore's client-side shuffle per "<window>:explore" view, kept
        # stable across rebuilds (votes, polls, feed refreshes) and dropped
        # when the user leaves that view, so each visit gets a fresh order.
        self.explore_orders: dict[str, list[int]] = {}
        self.view_key: str | None = None
        self.restored: dict[int, FeedStory] = {}
        self.history: list[FeedStory] = []
        # The window and views (sorts) each voted story was listed in, so
        # undo puts it back only where the server had it.
        self.vote_views: dict[int, tuple[str, list[str]]] = {}
        self.vote_revisions: dict[int, int] = {}
        self.vote_lock = asyncio.Lock()
        self.selection_serial = 0
        self.interaction_session = str(uuid4())
        self.summary_story_id: int | None = None
        self.reading = False
        self.help_open = False
        self.setting_up = False
        self.status_mode = "context"
        # Footer texts: the keys shrink through hint_options (longest first)
        # so the status and keys share the single footer row.
        self.status_text = ""
        self.hint_options: list[str] = ["? help"]
        self.hints_text = "? help"
        self.last_error: str | None = None
        # The server's counts version at the last poll (None until seen).
        self.counts_version: int | None = None
        self.window_generation = 0
        self.prefetch = max(0, prefetch)
        self.prefetch_generate = min(self.prefetch, max(0, prefetch_generate))
        # Complete summaries, and at most one request per story in flight. The
        # selection and prefetch share that request; only refresh, setup and
        # quit cancel it, never a caller that stopped waiting.
        self.summaries: dict[int, str] = {}
        self.summary_requests: dict[int, asyncio.Task[SummaryResult | None]] = {}
        # Prefetch backoff: provisional/empty/failed results wait until the
        # time; a cache miss (time of the miss) may still generate once the
        # story is near the selection.
        self.prefetch_retry_at: dict[int, float] = {}
        self.prefetch_misses: dict[int, float] = {}
        self.prefetch_cooldown_until = 0.0
        self.prefetch_slots = asyncio.Semaphore(PREFETCH_CONCURRENCY)
        self.closing = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="filters"):
            yield Tabs(
                *(Tab(s.title(), id=f"sort-{s}") for s in self.SORT_CYCLE),
                id="sort-tabs",
            )
            yield Static("Sort", classes="filter-caption")
            yield Dropdown(
                [(s.title(), s) for s in self.SORT_CYCLE],
                value="recommended",
                allow_blank=False,
                id="sort",
            )
            yield Static("Window", classes="filter-caption")
            yield Dropdown(
                [(WINDOW_LABELS[w], w) for w in WINDOWS],
                value=self.saved_window,
                allow_blank=False,
                id="window",
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
        self.set_interval(60.0, self.poll_feed_version)
        self.set_interval(60.0, self.apply_clock_theme)
        self.theme_changed_signal.subscribe(self, lambda _theme: self.restyle())
        self.query_one("#headlines", OptionList).focus()
        self.start()

    def apply_clock_theme(self) -> None:
        """Follow the local clock, but only when the clock's pick changes, so a
        theme chosen from the command palette holds until the next boundary."""
        name = theme_for_hour(time.localtime().tm_hour)
        if name == self.clock_theme:
            return
        self.clock_theme = name
        self.theme = name
        PALETTE.update(PALETTES[name])

    def restyle(self) -> None:
        """Re-render Rich-styled text (headlines, heading, counts) in the
        palette of the current theme; CSS-styled widgets follow on their own."""
        PALETTE.update(PALETTES.get(self.theme, DARK_PALETTE))
        if not self.query("#headlines"):
            return
        headlines = self.query_one("#headlines", OptionList)
        widths = self._row_widths or (0, 0, 0)
        for story in self.stories:
            headlines.replace_option_prompt(
                str(story.id), headline(story, story.id == self._marked_id, widths)
            )
        if selected := self.selected():
            self.query_one("#story-heading", Static).update(self.heading(selected))
        self.context_status()

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
            in {
                "move",
                "vote",
                "undo",
                "read",
                "refresh",
                "open_url",
                "copy_url",
                "dig_deeper",
                "cycle_sort",
                "cycle_window",
            }
        )

    def status(self, message: str, *, error: bool = False) -> None:
        """Show a transient message; context_status() restores the counts line."""
        self.status_mode = "error" if error else "message"
        text = ("✗ " if error else "") + message
        widget = self.query_one("#status", Static)
        widget.set_class(error, "error")
        widget.set_class(False, "context")
        self.fit_footer(status=text)
        # Cut the middle to fit beside the keys, so the subject and an
        # actionable tail ("Press r to retry.") both survive.
        room = self.size.width - 2 - (_cell_len(self.hints_text) + 2)
        widget.update(fit_middle(text, room))

    def context_status(self) -> None:
        """Counts line for the current filter; only replaces an earlier counts line."""
        if self.status_mode != "context" or not self.query("#status"):
            return
        counts = self.feed.feedback_counts if self.feed else {}
        line = Text()
        line.append(f"{len(self.stories)} shown", style=PALETTE["soft"])
        line.append(" · ", style=PALETTE["sep"])
        line.append(f"+{counts.get('up', 0)}", style=PALETTE["good"])
        line.append(" ", style=PALETTE["sep"])
        line.append(f"~{counts.get('neutral', 0)}", style=PALETTE["warn"])
        line.append(" ", style=PALETTE["sep"])
        line.append(f"−{counts.get('down', 0)}", style=PALETTE["bad"])
        widget = self.query_one("#status", Static)
        widget.update(line)
        widget.set_class(False, "error")
        widget.set_class(True, "context")
        self.fit_footer(status=line.plain)

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
                # A rejected token reopens setup on the profile's deployment.
                self.server = profile.server
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
        self.cancel_summary_requests()
        self.workers.cancel_group(self, "refresh")
        self.workers.cancel_group(self, "vote")
        self.workers.cancel_group(self, "impression")
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
        self.server = profile.server
        self.api = API(profile.server, profile.token)
        self.interaction_session = str(uuid4())
        self.feed = None
        self.feeds.clear()
        self.window_attempts.clear()
        self.rated.clear()
        self.unavailable.clear()
        self.restored.clear()
        self.history.clear()
        self.vote_views.clear()
        self.reset_summaries()
        self.prefetch_cooldown_until = 0.0
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
            max((_cell_len(headline_comments(s)) for s in self.stories), default=0),
        ]
        age_w = max((len(story_age(s)) for s in self.stories), default=0)
        total = widths[0] + 3 + widths[1] + 3 + widths[2] + 3 + age_w
        if available > 0 and total > available and widths[0] > 8:
            widths[0] = max(8, widths[0] - (total - available))
        return (widths[0], widths[1], widths[2])

    def selected_window(self) -> str:
        return str(self.query_one("#window", Select).value)

    def view_order(self, sort: str) -> list[int]:
        """Story order for a sort of the current window as the user sees it."""
        order = self.feed.orders.get(sort, []) if self.feed else []
        if sort != "explore" or self.feed is None:
            return order
        # Explore is a discovery deck: shuffle client-side, since the server
        # sends it in model-score order and would pin the deck for hours.
        # Stories already placed keep their position; new ones are shuffled
        # in after them. Copies: feed.orders is shared.
        key = f"{self.feed.window}:{sort}"
        members = set(order)
        kept = [sid for sid in self.explore_orders.get(key, []) if sid in members]
        placed = set(kept)
        fresh = [sid for sid in order if sid not in placed]
        random.shuffle(fresh)
        self.explore_orders[key] = kept + fresh
        return list(self.explore_orders[key])

    def rebuild(self, select_id: int | None = None) -> None:
        # Teardown removes nodes before the final messages drain; ignore late
        # rebuilds rather than raising NoMatches.
        if not self.query("#headlines"):
            return
        old = self.selected()
        if select_id is None and old:
            select_id = old.id
        sort = str(self.query_one("#sort", Select).value)
        lookup = {story.id: story for story in self.feed.stories} if self.feed else {}
        self.view_key = f"{self.selected_window()}:{sort}"
        order = self.view_order(sort)
        self.stories = [
            lookup[sid]
            for sid in order
            if sid not in self.rated and sid not in self.unavailable
        ][:VIEW_LIMIT]
        headlines = self.query_one("#headlines", OptionList)
        headlines.clear_options()
        # Option padding (1 each side) plus the 2-cell selection marker.
        widths = self.meta_widths(max(0, headlines.size.width - 4))
        self._row_widths = widths
        self._marked_id = select_id
        # A `None` between options draws a faint rule row; it does not take an
        # option index, so `highlighted` still indexes `self.stories`.
        options: list[Option | None] = []
        for s in self.stories:
            if options:
                options.append(None)
            options.append(Option(headline(s, s.id == select_id, widths), id=str(s.id)))
        headlines.add_options(options)
        if self.stories:
            headlines.highlighted = next(
                (i for i, s in enumerate(self.stories) if s.id == select_id), 0
            )
            if select_id == -1:  # Filter change: no story has this ID.
                headlines.scroll_home(animate=False)
            # Fresh feed data can carry new points/comments; keep the reading
            # heading in step even when the selection id has not changed.
            if selected := self.selected():
                self.query_one("#story-heading", Static).update(self.heading(selected))
            self.schedule_summary()
        else:
            self.query_one("#story-heading", Static).update("")
            self.summary_story_id = None
            self.selection_serial += 1
            self.workers.cancel_group(self, "summary")
            self.query_one("#summary", Markdown).update(
                EMPTY_NOTICE
                if self.feed is not None
                else feed_failure_notice(self.last_error)
                if self.last_error
                else "Loading…"
            )
        self.query_one("#reading-pane").set_class(bool(self.stories), "has-story")
        self.layout_panes()
        self.context_status()

    def on_select_changed(self, event: Select.Changed) -> None:
        # Queued changes can outlive their value after rapid sort cycling.
        # Replaying them into Tabs would start an endless two-way echo.
        if event.value != event.select.value:
            return
        if event.select.id == "sort" and self.query("#sort-tabs"):
            self.query_one("#sort-tabs", Tabs).active = f"sort-{event.value}"
        if self.view_key is not None:
            self.explore_orders.pop(self.view_key, None)  # next visit reshuffles
        if event.select.id == "window":
            if self.window_file is not None and event.value != self.saved_window:
                self.saved_window = str(event.value)
                save_window(self.saved_window, self.window_file)
            self.show_window(str(event.value))
        else:
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
            self.query_one("#story-heading", Static).update(self.heading(story))
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
                        window=self.feed.window,
                        occurred_at=time.time(),
                        badges=[badge.kind for badge in story.badge_details],
                    ),
                    self.selection_serial,
                )
            cached = self.summaries.get(story.id)
            if cached is not None:
                self.query_one("#summary", Markdown).update(cached)
                self.schedule_prefetch()
            else:
                self.query_one("#summary", Markdown).update("Loading summary…")
                self.load_summary(story.id, self.selection_serial)
            self.query_one("#summary", Markdown).scroll_home(animate=False)

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
            or self.feed.window != event.window
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
        started = time.monotonic()
        await asyncio.sleep(0.3)
        if (
            not self.api
            or serial != self.selection_serial
            or not self.query("#summary")
        ):
            return
        previous = self.summaries.get(story_id)
        ticker: Timer | None = None
        if previous is None:
            self.query_one("#summary", Markdown).update("Loading summary…")

            # A summary written on request takes ~15 s; count the wait so a
            # slow one reads as working, not stuck.
            def tick() -> None:
                if serial == self.selection_serial and self.query("#summary"):
                    elapsed = int(time.monotonic() - started)
                    self.query_one("#summary", Markdown).update(
                        f"Loading summary… {elapsed}s"
                    )

            ticker = self.set_interval(1.0, tick)
        else:
            self.query_one("#summary", Markdown).update(previous)
            self.status("Regenerating summary…")
        try:
            summary = await self.await_summary(story_id, force=force_refresh)
            if serial == self.selection_serial and self.query("#summary"):
                if summary.empty:
                    # Server found nothing summarizable: same session-hide
                    # as undisplayable failures, never rendered or cached.
                    self._hide_story(story_id, "no summarizable content")
                    return
                self.query_one("#summary", Markdown).update(summary.text)
                if summary.provisional:
                    self.status(
                        "Summary may be outdated or incomplete. Press r to retry."
                    )
                elif force_refresh:
                    self.status("Summary regenerated.")
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
                    return
                # Undisplayable summaries leave the deck: the failure may be
                # transient (quota/cooldown), so this hides for the session
                # only — refresh restores. InvalidProfile goes to setup above.
                self._hide_story(story_id, f"summary unavailable ({exc})")
        finally:
            if ticker is not None:
                ticker.stop()

    async def await_summary(
        self, story_id: int, *, force: bool = False
    ) -> SummaryResult:
        """The summary for the selected story, joining any request in flight."""
        if not force and (text := self.summaries.get(story_id)) is not None:
            return SummaryResult(text)
        task = self.summary_task(story_id, force=force)
        # Start prefetch after the selected request, so it goes out first.
        self.schedule_prefetch()
        # Shielded: moving to another story must not cancel shared work.
        summary = await asyncio.shield(task)
        if summary is None:
            # Speculation came back empty-handed (skipped, cache miss, or a
            # failed cache read); ask for this story directly.
            summary = await asyncio.shield(self.summary_task(story_id))
        if summary is None:
            raise TransientError("Not connected. Press r to retry.")
        return summary

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
        sort = self.query_one("#sort", Select).value
        for other in self.SORT_CYCLE:
            if other == sort:
                continue
            order = self.view_order(other)
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
        """Start background requests for stories around the selection."""
        if self.prefetch <= 0 or not self.api or self.setting_up or self.closing:
            return
        now = time.monotonic()
        if now < self.prefetch_cooldown_until:
            return
        generate = set(self.prefetch_targets(self.prefetch_generate))
        for sid in self.prefetch_targets(self.prefetch):
            missed = self.prefetch_misses.get(sid)
            if (
                sid in self.summaries
                or sid in self.summary_requests
                or self.prefetch_retry_at.get(sid, 0.0) > now
                or (
                    missed is not None
                    and now - missed < PREFETCH_COOLDOWN_SECONDS
                    and sid not in generate
                )
            ):
                continue
            self.summary_task(sid, background=True)

    def summary_task(
        self, story_id: int, *, background: bool = False, force: bool = False
    ) -> asyncio.Task[SummaryResult | None]:
        """The one request for a story: reuse it, or start it."""
        task = self.summary_requests.get(story_id)
        if task is not None and not task.done() and not force:
            return task
        if task is not None:
            task.cancel()
        task = asyncio.create_task(
            self.fetch_summary(story_id, background=background, force=force)
        )
        self.summary_requests[story_id] = task

        def done(task: asyncio.Task[SummaryResult | None]) -> None:
            if self.summary_requests.get(story_id) is task:
                del self.summary_requests[story_id]
            if not task.cancelled():
                task.exception()  # Awaiting callers handle it; none may be left.

        task.add_done_callback(done)
        return task

    def cancel_summary_requests(self, keep: int | None = None) -> None:
        for sid, task in list(self.summary_requests.items()):
            if sid != keep:
                task.cancel()

    def reset_summaries(
        self, *, keep_text: int | None = None, keep_request: int | None = None
    ) -> None:
        """Forget summaries (a new deck may carry new text) and prefetch state."""
        kept = self.summaries.get(keep_text) if keep_text is not None else None
        self.summaries.clear()
        if keep_text is not None and kept is not None:
            self.summaries[keep_text] = kept
        self.cancel_summary_requests(keep=keep_request)
        self.prefetch_retry_at.clear()
        self.prefetch_misses.clear()

    async def fetch_summary(
        self, story_id: int, *, background: bool, force: bool
    ) -> SummaryResult | None:
        api = self.api
        if api is None:
            return None
        if background:
            async with self.prefetch_slots:
                summary = await self.speculate(api, story_id)
        else:
            summary = await api.summary(story_id, force_refresh=force)
        if summary is not None and api is self.api:
            if summary.points is not None:
                self.patch_counts(story_id, summary.points, summary.comments)
            if summary.provisional or summary.empty:
                # Shown (or hidden) by the selection, never cached.
                self.prefetch_retry_at[story_id] = (
                    time.monotonic() + PREFETCH_COOLDOWN_SECONDS
                )
            else:
                self.summaries[story_id] = summary.text
                self.prefetch_retry_at.pop(story_id, None)
        return summary

    def speculation_wanted(self, story_id: int, *, generate: bool = False) -> bool:
        """Still worth fetching: selected, or inside the current window."""
        selected = self.selected() if self.query("#headlines") else None
        if selected is not None and selected.id == story_id:
            return True
        if time.monotonic() < self.prefetch_cooldown_until:
            return False
        depth = self.prefetch_generate if generate else self.prefetch
        return story_id in self.prefetch_targets(depth)

    async def speculate(self, api: API, story_id: int) -> SummaryResult | None:
        """Read the server cache; generate only for stories near the selection.

        Returns None when there is nothing to show yet, so a selection waiting
        on this request asks for the story directly.
        """
        # Recheck after waiting for a slot: navigation can abandon the target.
        if not self.speculation_wanted(story_id):
            return None
        try:
            summary = await api.cached_summary(story_id)
            if summary is None:
                self.prefetch_misses[story_id] = time.monotonic()
                if self.speculation_wanted(story_id, generate=True):
                    summary = await api.summary(story_id)
        except APIError as exc:
            # Requests already running may finish; nothing new for a minute.
            self.prefetch_cooldown_until = time.monotonic() + PREFETCH_COOLDOWN_SECONDS
            if isinstance(exc, (InvalidProfile, TransientError)):
                raise
            return None
        return summary

    def can_poll_feed(self) -> bool:
        return bool(
            self.api
            and self.feed
            and not self.setting_up
            and not self.reading
            and not self.help_open
            and not any(
                worker.group in {"refresh", "vote"} and worker.is_running
                for worker in self.workers
            )
        )

    async def poll_feed_version(self) -> None:
        """Every minute: reload when the server has a newer deck than ours,
        or refetch the feed when only stored points/comments changed."""
        if not self.can_poll_feed():
            return
        api, feed = self.api, self.feed
        assert api is not None and feed is not None
        # A current deck waits for any new version (regen, a vote elsewhere);
        # a stale one waits for the reranked deck it is missing.
        wanted = feed.version if feed.ready else feed.target_version
        try:
            readiness = await api.ready(wanted)
        except APIError:
            # Passive checks must not replace a usable deck with an error.
            return
        if not (self.api is api and self.feed is feed and self.can_poll_feed()):
            return
        current = readiness.current
        # A lower version is a server restart, not an obsolete response.
        if feed.ready:
            newer = current != feed.version
        else:
            newer = readiness.ready or current < wanted
        seen = self.counts_version
        counts_changed = (
            seen is not None
            and readiness.counts_version is not None
            and readiness.counts_version != seen
        )
        if seen is None:
            self.counts_version = readiness.counts_version
        if counts_changed:
            self.feeds.clear()
            self.window_attempts.clear()
            self.window_generation += 1
        if newer:
            self.reload(manual=False)
        elif counts_changed:
            # Same deck, fresher counts: summaries and the open story stay.
            self.refresh_feed(announce=False, counts_version=readiness.counts_version)

    @work(group="refresh", exclusive=True)
    async def refresh_feed(
        self, *, announce: bool = True, counts_version: int | None = None
    ) -> None:
        """Fetch the selected window's feed and show it."""
        if not self.api or self.setting_up:
            return
        api, window = self.api, self.selected_window()
        if announce:
            self.status("Refreshing…")
        try:
            feed = await api.feed(window)
        except InvalidProfile as exc:
            self.setup(str(exc))
            return
        except APIError as exc:
            if announce or self.feed is None:
                self.status(
                    ("Showing stale stories. " if self.feed else "")
                    + str(exc)
                    + " Press r to retry.",
                    error=True,
                )
                self.show_failure(str(exc))
            return
        if api is not self.api:
            return
        if counts_version is not None:
            self.counts_version = counts_version
        if window != self.selected_window():
            # The user moved to a cached window meanwhile: keep this one
            # only as a cache entry of the version on screen.
            if self.feed is not None and feed.version == self.feed.version:
                self.feeds[window] = feed
            return
        self.forget_outgrown_summaries(feed)
        self.set_feed(feed)
        self.last_error = None
        if feed.ready:
            self.restored.clear()
        else:
            # Until the reranked deck lands, it may lack undone stories.
            for restored in self.restored.values():
                self.restore_story(restored)
        if announce or (feed.ready and self.status_mode != "error"):
            # A current deck replaces the ranking notice; errors stay put.
            self.status_mode = "context"
        self.rebuild()
        if announce and not feed.ready:
            self.status("Showing available stories while ranking updates…")
        self.schedule_window_prefetch()

    def forget_outgrown_summaries(self, feed: Feed) -> None:
        """Drop kept summaries of stories that gained comments, except the
        open one, so reopening asks the server again (it rewrites a busy
        thread's summary once enough comments arrived)."""
        if self.feed is None:
            return
        before = {story.id: story.comments or 0 for story in self.feed.stories}
        for story in feed.stories:
            if story.id != self.summary_story_id and (story.comments or 0) > before.get(
                story.id, story.comments or 0
            ):
                self.summaries.pop(story.id, None)

    def patch_counts(self, story_id: int, points: int, comments: int | None) -> None:
        """Show the counts a summary reply carried; the feed's may be older."""
        for feed in self.feeds.values():
            for i, item in enumerate(feed.stories):
                if item.id == story_id:
                    feed.stories[i] = replace(item, points=points, comments=comments)
        for i, item in enumerate(self.stories):
            if item.id != story_id or (item.points, item.comments) == (
                points,
                comments,
            ):
                continue
            story = replace(item, points=points, comments=comments)
            self.stories[i] = story
            if not self.query("#headlines"):
                return
            listing = self.query_one("#headlines", OptionList)
            # A count gaining a digit can widen the metadata columns.
            widths = self.meta_widths(max(0, listing.size.width - 4))
            for row in [story] if widths == self._row_widths else self.stories:
                listing.replace_option_prompt(
                    str(row.id), headline(row, row.id == self._marked_id, widths)
                )
            self._row_widths = widths
            selected = self.selected()
            if selected is not None and selected.id == story_id:
                self.query_one("#story-heading", Static).update(self.heading(story))

    def set_feed(self, feed: Feed) -> None:
        """Make *feed* the selected window's; a new version drops the cached
        windows of other versions."""
        self.feeds = {
            window: cached
            for window, cached in self.feeds.items()
            if cached.version == feed.version
        }
        self.feeds[feed.window] = feed
        self.feed = feed

    def show_window(self, window: str) -> None:
        """Switch to *window*: its cached feed of the version on screen, or
        fetch it."""
        current = self.feed
        cached = self.feeds.get(window)
        if current is not None and current.window == window:
            self.rebuild(select_id=-1)
        elif (
            current is not None
            and cached is not None
            and cached.version == current.version
        ):
            # Same deck, so the ranking state (a pending vote's target) and
            # counts on screen apply to it too.
            self.set_feed(
                replace(
                    cached,
                    target_version=current.target_version,
                    ready=current.ready,
                    feedback_counts=current.feedback_counts,
                )
            )
            self.rebuild(select_id=-1)
            self.schedule_window_prefetch()
        else:
            self.feed = None
            self.rebuild(select_id=-1)
            self.refresh_feed(announce=False)

    def next_window_prefetch(self) -> str | None:
        """A neighbour of the selected window not cached or tried for the
        version on screen."""
        feed = self.feed
        if feed is None or feed.window not in WINDOWS:
            return None
        index = WINDOWS.index(feed.window)
        for other in (index + 1, index - 1):
            if not 0 <= other < len(WINDOWS):
                continue
            window = WINDOWS[other]
            cached = self.feeds.get(window)
            if (cached is not None and cached.version == feed.version) or (
                window,
                feed.version,
            ) in self.window_attempts:
                continue
            return window
        return None

    def schedule_window_prefetch(self) -> None:
        """Fetch the neighbouring windows' feeds in the background, one at a
        time. Only feeds: never summaries."""
        if (
            self.window_prefetching
            or self.api is None
            or self.setting_up
            or self.closing
            or self.next_window_prefetch() is None
        ):
            return
        self.window_prefetching = True
        self.prefetch_windows()

    @work(group="window-prefetch")
    async def prefetch_windows(self) -> None:
        try:
            while (window := self.next_window_prefetch()) is not None:
                api, feed = self.api, self.feed
                if api is None or feed is None:
                    return
                self.window_attempts.add((window, feed.version))
                generation = self.window_generation
                try:
                    fetched = await api.feed(window)
                except APIError:
                    continue
                # A late answer is only a cache entry of the version on
                # screen: it never replaces the selected window's feed.
                if (
                    api is self.api
                    and generation == self.window_generation
                    and self.feed is not None
                    and fetched.version == self.feed.version
                    and window != self.feed.window
                    and window not in self.feeds
                ):
                    self.feeds[window] = fetched
        finally:
            self.window_prefetching = False

    SORT_CYCLE: ClassVar[tuple[str, ...]] = ("recommended", "popular", "explore")

    def action_cycle_window(self, delta: int = 1) -> None:
        """Move the window selector by *delta* steps, wrapping at either end."""
        select = self.query_one("#window", Select)
        try:
            index = WINDOWS.index(str(select.value))
        except ValueError:
            index = 0 if delta < 0 else -1
        select.value = WINDOWS[(index + delta) % len(WINDOWS)]

    def action_cycle_sort(self, delta: int = 1) -> None:
        """Move the sort selector by *delta* steps, wrapping at either end."""
        select = self.query_one("#sort", Select)
        try:
            index = self.SORT_CYCLE.index(str(select.value))
        except ValueError:
            index = -1 if delta > 0 else 0
        select.value = self.SORT_CYCLE[(index + delta) % len(self.SORT_CYCLE)]

    def action_refresh(self) -> None:
        self.reload(manual=True)

    def reload(self, *, manual: bool) -> None:
        """Reload the feed; cached summaries go (a new deck may carry new text).

        The poller's refresh leaves the open story, its summary and scroll
        alone. A manual r also regenerates the selected summary and restores
        stories hidden this session.
        """
        story = self.selected()
        keep = story.id if story else None
        # The old text stays on screen while a forced request regenerates it.
        self.reset_summaries(keep_text=keep, keep_request=None if manual else keep)
        if manual:
            self.help_open = False
            self.unavailable.clear()
            self.prefetch_cooldown_until = 0.0
            self.selection_serial += 1
            self.summary_story_id = keep
            if story is None:
                self.workers.cancel_group(self, "summary")
            else:
                self.load_summary(story.id, self.selection_serial, force_refresh=True)
        self.refresh_feed(announce=manual)

    def action_move(self, delta: int) -> None:
        """Next/previous story, in the list and in zoom alike."""
        listing = self.query_one("#headlines", OptionList)
        if self.stories:
            listing.highlighted = max(
                0, min(len(self.stories) - 1, (listing.highlighted or 0) + delta)
            )

    def action_page_summary(self) -> None:
        """Page the TLDR down from either view, whichever pane has focus."""
        self.query_one("#summary", Markdown).scroll_page_down(animate=False)

    def action_vote(self, action: str) -> None:
        """Hide the story and move on now; the server hears about it next."""
        story = self.selected()
        if story is None:
            return
        self.apply_vote(story)
        revision = self.vote_revisions.get(story.id, 0) + 1
        self.vote_revisions[story.id] = revision
        self.submit(story, action, revision)

    def action_undo(self) -> None:
        if self.history:
            story = self.history[-1]
            self.apply_undo(story)
            revision = self.vote_revisions.get(story.id, 0) + 1
            self.vote_revisions[story.id] = revision
            self.submit(story, "clear", revision)

    def apply_vote(self, story: FeedStory) -> None:
        index = next((i for i, s in enumerate(self.stories) if s.id == story.id), 0)
        remaining = [item for item in self.stories if item.id != story.id]
        next_id = remaining[min(index, len(remaining) - 1)].id if remaining else None
        self.rated.add(story.id)
        self.restored.pop(story.id, None)
        self.history.append(story)
        if self.feed is not None:
            self.vote_views[story.id] = (
                self.feed.window,
                [key for key, order in self.feed.orders.items() if story.id in order],
            )
        self.rebuild(next_id)

    def apply_undo(self, story: FeedStory) -> None:
        if story in self.history:
            self.history.remove(story)
        self.rated.discard(story.id)
        self.restored[story.id] = story
        self.restore_story(story)
        self.rebuild(story.id)

    @work(group="vote")
    async def submit(self, story: FeedStory, action: str, revision: int) -> None:
        api = self.api
        if api is None:
            return
        self.status("Saving vote…")
        # One at a time, so the server sees votes and undos in the order made.
        async with self.vote_lock:
            try:
                target = await api.vote(story.id, action)
            except InvalidProfile as exc:
                self.setup(str(exc))
                return
            except APIError as exc:
                if api is self.api:
                    # Never retried: the server may or may not have it.
                    latest = self.vote_revisions.get(story.id) == revision
                    if latest:
                        if action == "clear":
                            self.apply_vote(story)
                        else:
                            self.apply_undo(story)
                    notice = (
                        "Vote not confirmed, so the story is back; check before voting again."
                        if latest
                        else "Earlier vote not confirmed; your latest choice is still shown."
                    )
                    self.status(f"{notice} {exc}", error=True)
                return
        if api is not self.api:
            return
        if self.feed is not None and target > self.feed.version:
            # The poller loads the reranked deck once the server has it.
            self.set_feed(replace(self.feed, target_version=target, ready=False))
        self.status("Vote cleared." if action == "clear" else "Vote saved.")

    def restore_story(self, story: FeedStory) -> None:
        """Put an undone story back in the views of the window it was voted
        from, where the server orders it: Popular by HN gravity on that
        window's clock, the rest by rank score."""
        window, keys = self.vote_views.get(story.id, ("", []))
        feed = self.feeds.get(window)
        if feed is None:
            return
        if all(item.id != story.id for item in feed.stories):
            feed.stories.append(story)
        by_id = {item.id: item for item in feed.stories}
        now = time.time()
        scale = GRAVITY_TIME_SCALE.get(window, 1.0)
        for key in keys:
            order = feed.orders.setdefault(key, [])
            if story.id in order:
                continue
            by_gravity = key == "popular"

            def rank(item: FeedStory, by_gravity: bool = by_gravity) -> float:
                if by_gravity:
                    age_h = max(now - item.time, 0) / 3600
                    return item.points / (age_h / scale + 2) ** 1.8
                return item.rank_score

            index = next(
                (i for i, sid in enumerate(order) if rank(by_id[sid]) < rank(story)),
                len(order),
            )
            order.insert(index, story.id)

    def action_open_url(self, field: str) -> None:
        story = self.selected()
        url = getattr(story, field, "") if story else ""
        if urlsplit(url).scheme in {"http", "https"}:
            open_in_browser(url)
        else:
            self.status("No link available for this story.")

    async def action_dig_deeper(self) -> None:
        """Open a Claude Code session on the story, or copy its prompt."""
        story = self.selected()
        if story is None:
            self.status("No story selected.")
            return
        prompt = dig_deeper_prompt(story)
        # tmux answers at once, but a wedged server must not freeze the reader.
        if await asyncio.to_thread(open_agent_session, prompt):
            self.status(f"Opened Claude on: {story.title}")
            return
        self.copy_to_clipboard(prompt)
        await asyncio.to_thread(copy_with_system_tool, prompt)
        self.status("Could not open a tmux pane; copied the Claude prompt.")

    async def action_copy_url(self) -> None:
        """Copy the comments link, or the article link when there is none."""
        story = self.selected()
        url = next(
            (
                u
                for u in ((story.comments_url, story.article_url) if story else ())
                if urlsplit(u).scheme in {"http", "https"}
            ),
            "",
        )
        if not url:
            self.status("No link available for this story.")
            return
        # OSC 52 needs terminal support; a system clipboard tool covers the rest.
        self.copy_to_clipboard(url)
        # A hung clipboard owner must not freeze the reader.
        if await asyncio.to_thread(copy_with_system_tool, url):
            self.status(f"Copied {url}")
        else:
            self.status(f"Sent to terminal clipboard (needs OSC 52): {url}")

    def layout_panes(self, width: int | None = None, height: int | None = None) -> None:
        narrow = (self.size.width if width is None else width) < 100
        compact = (self.size.height if height is None else height) < COMPACT_HEIGHT
        compact_changed = compact != self.has_class("compact")
        self.set_class(narrow, "narrow")
        self.set_class(compact, "compact")
        self.set_class(self.reading, "reading")
        votes = "1 up · 2 neutral · 3 down → next story"
        if compact:
            # A short pane shows only ? help, which lists the keys.
            keys: list[str] = []
        elif self.reading:
            keys = [
                (
                    f"j/k story · Space page · Enter/Esc back · {votes} · b badges"
                    " · ? help · q quit"
                ),
                f"j/k story · Space page · Enter/Esc back · {votes} · ? help",
                f"Enter/Esc back · {votes} · ? help",
                "Enter/Esc back · 1/2/3 vote · ? help",
                "Enter/Esc back · ? help",
            ]
        elif self.can_read:
            keys = [
                f"j/k move · Enter zoom · {votes} · b badges · ? help · q quit",
                f"j/k move · Enter zoom · {votes} · ? help",
                f"Enter zoom · {votes} · ? help",
                "Enter zoom · 1/2/3 vote · ? help",
                "Enter zoom · ? help",
            ]
        else:
            keys = [
                f"j/k move · {votes} · b badges · ? help · q quit",
                f"j/k move · {votes} · ? help",
                f"{votes} · ? help",
                "1/2/3 vote · ? help",
            ]
        self.hint_options = [*keys, "? help"]
        self.fit_footer(width=width)
        if compact_changed and (selected := self.selected()):
            self.query_one("#story-heading", Static).update(self.heading(selected))

    def heading(self, story: FeedStory) -> Text:
        """The reading pane's heading; a short pane leaves out why the story
        is recommended."""
        return story_heading(story, attribution=not self.has_class("compact"))

    def fit_footer(
        self,
        *,
        status: str | None = None,
        width: int | None = None,
    ) -> None:
        """One footer row: status left, the longest key hints that fit right.
        The keys shrink to "? help" before the status is cut."""
        if status is not None:
            self.status_text = status
        width = self.size.width if width is None else width
        # Each widget pads one cell per side; keep a gap of two between them.
        room = width - _cell_len(self.status_text) - 6
        self.hints_text = next(
            (h for h in self.hint_options if _cell_len(h) <= room),
            self.hint_options[-1],
        )
        if self.query("#shortcuts"):
            self.query_one("#shortcuts", Static).update(self.hints_text)

    @property
    def can_read(self) -> bool:
        """Zoom is offered whenever a story is selected."""
        return bool(self.query("#headlines")) and self.selected() is not None

    def on_resize(self, event: events.Resize) -> None:
        if self.query("#panes"):
            self.layout_panes(event.size.width, event.size.height)

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

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_read()

    def action_headlines(self) -> None:
        self.reading = False
        self.layout_panes()
        self.query_one("#headlines", OptionList).focus()
        if self.help_open:
            self.help_open = False
            self.schedule_summary()

    def action_badge_legend(self) -> None:
        self.help_open = True
        self.summary_story_id = None
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.query_one("#summary", Markdown).update(
            "# Badge legend\n\n"
            "🔥 **Hot** — rising fast · 🏆 **Top** — at least 100 HN points · "
            "💬 **Talk** — at least 50 comments and comments ≥ points\n\n"
            "Badges combine when several apply.\n\n"
            "🤔 **Unsure** — model uncertain · ✨ **Novel** — unlike your votes · 🎯 **Interest** — an interest Recommended misses\n\n"
            "Escape: return to the story. ?: shortcuts."
        )
        self.focus_summary()

    def action_help(self) -> None:
        self.help_open = True
        self.summary_story_id = None
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.query_one("#summary", Markdown).update(
            "# Shortcuts\n\n"
            "## Move\n\n"
            "- `j` / `k`: next / previous story (list or zoom view)\n"
            "- `Tab`: switch focus between panes\n"
            "- Arrow keys: scroll the focused pane\n\n"
            "## Read\n\n"
            "- `Enter`: zoom the TLDR pane (hide the article list)\n"
            "- `Enter` / `Escape`: return to the article list\n"
            "- `Space`: page the TLDR down (list or zoom view)\n\n"
            "## Vote\n\n"
            "- `1` / `2` / `3`: up / neutral / down (advances to next story)\n"
            "- `u`: undo latest vote\n\n"
            "## Sort\n\n"
            "- `s`: cycle sort (Recommended → Popular → Explore)\n"
            "- `h` / `l`: previous / next sort\n"
            "- `d` / `D`: next / previous time window (12 hours → 1 day →"
            " 1 week → 1 month → Archive)\n"
            "- Selectors: sort and time window\n\n"
            "## Other\n\n"
            "- `o` / `c`: open article / comments\n"
            "- `y`: copy comments link (article link if none)\n"
            "- `a`: dig deeper: Claude Code in a tmux pane beside this one\n"
            "- `r`: refresh and regenerate selected summary\n"
            "- `b`: badge legend\n"
            "- `?`: this help\n"
            "- `q`: quit\n"
            "- `Escape`: close this help\n\n"
            "Votes are never automatically retried after network errors."
        )
        self.focus_summary()

    async def on_unmount(self) -> None:
        self.closing = True
        self.selection_serial += 1
        self.workers.cancel_all()
        self.cancel_summary_requests()
        if self.api:
            await self.api.close()
