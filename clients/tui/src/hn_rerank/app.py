from __future__ import annotations

import asyncio
import webbrowser
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    Markdown,
    OptionList,
    Select,
    Static,
)
from textual.widgets.option_list import Option

from .api import (
    API,
    APIError,
    InvalidProfile,
    Profile,
    load_profile,
    normalize_server,
    profile_path,
    save_profile,
)
from .models import Feed, FeedStory


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
    Setup { align: center middle; }
    #setup { width: 70; max-width: 95%; height: auto; padding: 1 2; border: round $accent; }
    #setup-message { height: auto; margin: 1 0; }
    Input { margin-bottom: 1; }
    """

    def __init__(self, path: Path, server: str | None, message: str = "") -> None:
        super().__init__()
        self.path = path
        self.server = server
        self.message = message
        self.pending = False

    def compose(self) -> ComposeResult:
        with Vertical(id="setup"):
            yield Label("HN Rerank — connect your profile")
            yield Static(
                self.message
                or "Import a profile link, or enter a server URL and create a new profile.",
                id="setup-message",
                markup=False,
            )
            yield Input(placeholder="https://host/hn/u/TOKEN", password=True, id="link")
            yield Button("Import profile", id="import")
            yield Input(
                value=self.server or "", placeholder="https://host/hn/", id="server"
            )
            yield Button("Create new profile", id="create")
            yield Button("Quit", id="quit")

    @work(exclusive=True)
    async def connect(self, create: bool) -> None:
        api: API | None = None
        try:
            if create:
                api = API(self.query_one("#server", Input).value)
                profile = await api.create()
            else:
                profile = Profile.from_link(self.query_one("#link", Input).value)
                if self.server and normalize_server(self.server) != profile.server:
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
            self.query_one("#setup-message", Static).update(message)
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
        elif not self.pending:
            self.pending = True
            for button in self.query(Button):
                button.disabled = True
            self.query_one("#setup-message", Static).update("Connecting…")
            self.connect(event.button.id == "create")


class Reader(App[None]):
    TITLE = "HN Rerank"
    CSS = """
    #filters { height: 3; }
    Select { width: 1fr; }
    #panes { height: 1fr; }
    #headlines { width: 1fr; height: 1fr; }
    #summary { width: 2fr; height: 1fr; padding: 0 2; overflow-y: auto; }
    #status { height: auto; max-height: 3; padding: 0 1; color: $text-muted; }
    .narrow #headlines, .narrow #summary { width: 1fr; }
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
        ("enter", "read", "Read"),
        ("escape", "headlines", "Back"),
        ("?", "help", "Help"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        server: str | None = None,
        config_path: Path | None = None,
        api: API | None = None,
    ) -> None:
        super().__init__()
        self.server = normalize_server(server) if server else None
        self.config_path = config_path or profile_path()
        self.api = api
        self.feed: Feed | None = None
        self.stories: list[FeedStory] = []
        self.rated: set[int] = set()
        self.restored: dict[int, FeedStory] = {}
        self.history: list[FeedStory] = []
        self.pending = False
        self.target: int | None = None
        self.selection_serial = 0
        self.summary_story_id: int | None = None
        self.reading = False
        self.setting_up = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="filters"):
            yield Select(
                [(s.title(), s) for s in ("recommended", "popular", "explore", "date")],
                value="recommended",
                allow_blank=False,
                id="sort",
            )
            yield Select(
                [("Recent", "recent"), ("Archive", "archive")],
                value="recent",
                allow_blank=False,
                id="age",
            )
        with Horizontal(id="panes"):
            yield OptionList(id="headlines")
            yield Summary("Connecting…", id="summary", open_links=False)
        yield Static("Loading…", id="status", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.layout_panes()
        self.query_one("#headlines", OptionList).focus()
        self.start()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        return not self.setting_up and not isinstance(self.focused, (Input, Select))

    def status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    @work(group="startup", exclusive=True)
    async def start(self) -> None:
        try:
            if self.api is None:
                profile = load_profile(self.config_path)
                if profile is None or (self.server and self.server != profile.server):
                    self.setup()
                    return
                self.api = API(profile.server, profile.token)
            await self.api.validate()
            self.refresh_feed()
        except InvalidProfile as exc:
            self.setup(str(exc))
        except APIError as exc:
            self.status(str(exc))

    def setup(self, message: str = "") -> None:
        if self.setting_up:
            return
        self.setting_up = True
        self.summary_story_id = None
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.workers.cancel_group(self, "refresh")
        self.workers.cancel_group(self, "vote")
        self.pending = False
        self.push_screen(Setup(self.config_path, self.server, message), self.connected)

    async def connected(self, profile: Profile | None) -> None:
        if profile is None:
            self.exit()
            return
        if self.api:
            await self.api.close()
        self.api = API(profile.server, profile.token)
        self.feed = None
        self.rated.clear()
        self.restored.clear()
        self.history.clear()
        self.target = None
        self.setting_up = False
        self.refresh_feed()

    def selected(self) -> FeedStory | None:
        index = self.query_one("#headlines", OptionList).highlighted
        return (
            self.stories[index]
            if index is not None and index < len(self.stories)
            else None
        )

    def rebuild(self, select_id: int | None = None) -> None:
        old = self.selected()
        if select_id is None and old:
            select_id = old.id
        sort = self.query_one("#sort", Select).value
        age = self.query_one("#age", Select).value
        lookup = {story.id: story for story in self.feed.stories} if self.feed else {}
        order = self.feed.orders.get(f"{sort}:{age}", []) if self.feed else []
        self.stories = [lookup[sid] for sid in order if sid not in self.rated]
        headlines = self.query_one("#headlines", OptionList)
        headlines.clear_options()
        headlines.add_options(
            [
                Option(
                    Text(
                        f"{s.title}\n{s.points} pts · {s.comments or 0} comments · {s.source}"
                    ),
                    id=str(s.id),
                )
                for s in self.stories
            ]
        )
        if self.stories:
            headlines.highlighted = next(
                (i for i, s in enumerate(self.stories) if s.id == select_id), 0
            )
            self.schedule_summary()
        else:
            self.summary_story_id = None
            self.selection_serial += 1
            self.workers.cancel_group(self, "summary")
            self.query_one("#summary", Markdown).update(
                "No stories in this filter. Change filters or press **r** to refresh."
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        self.rebuild()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        self.schedule_summary()

    def schedule_summary(self) -> None:
        story = self.selected()
        if story and story.id != self.summary_story_id:
            self.summary_story_id = story.id
            self.selection_serial += 1
            self.load_summary(story.id, self.selection_serial)

    @work(group="summary", exclusive=True)
    async def load_summary(self, story_id: int, serial: int) -> None:
        await asyncio.sleep(0.3)
        if (
            not self.api
            or serial != self.selection_serial
            or not self.query("#summary")
        ):
            return
        self.query_one("#summary", Markdown).update("Loading summary…")
        try:
            summary = await self.api.summary(story_id)
            if serial == self.selection_serial and self.query("#summary"):
                self.query_one("#summary", Markdown).update(summary)
        except InvalidProfile as exc:
            self.setup(str(exc))
        except APIError as exc:
            if serial == self.selection_serial and self.query("#summary"):
                self.query_one("#summary", Markdown).update(str(exc))

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
                    counts = feed.feedback_counts
                    self.status(
                        f"{len(self.stories)} stories · +{counts.get('up', 0)} ~{counts.get('neutral', 0)} −{counts.get('down', 0)}"
                    )
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
            self.status(("Showing stale stories. " if self.feed else "") + str(exc))

    def action_refresh(self) -> None:
        self.summary_story_id = None
        self.refresh_feed()

    def action_move(self, delta: int) -> None:
        if self.focused is self.query_one("#summary", Markdown) or (
            self.size.width < 100 and self.reading
        ):
            self.query_one("#summary", Markdown).scroll_relative(
                y=delta * 3, animate=False
            )
        else:
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
            self.status(str(exc))
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
            webbrowser.open(url)
        else:
            self.status("No link available for this story.")

    def layout_panes(self, width: int | None = None) -> None:
        narrow = (self.size.width if width is None else width) < 100
        self.set_class(narrow, "narrow")
        self.query_one("#headlines").display = not narrow or not self.reading
        self.query_one("#summary").display = not narrow or self.reading

    def on_resize(self, event: events.Resize) -> None:
        if self.query("#panes"):
            self.layout_panes(event.size.width)

    def action_read(self) -> None:
        self.reading = True
        self.layout_panes()
        self.query_one("#summary", Markdown).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_read()

    def action_headlines(self) -> None:
        self.reading = False
        self.layout_panes()
        self.query_one("#headlines", OptionList).focus()

    def action_help(self) -> None:
        self.selection_serial += 1
        self.workers.cancel_group(self, "summary")
        self.query_one("#summary", Markdown).update(
            "# Shortcuts\n\nj/k or arrows: navigate / scroll. Tab: focus. Enter: read. Escape: headlines.\n\n1/2/3: positive / neutral / negative. u: undo latest vote. o/c: article / comments. r: refresh. q: quit.\n\nUse the selectors for Recommended, Popular, Explore, Date and Recent / Archive. Votes are never automatically retried after network errors."
        )
        self.action_read()

    async def on_unmount(self) -> None:
        self.selection_serial += 1
        self.workers.cancel_all()
        if self.api:
            await self.api.close()
