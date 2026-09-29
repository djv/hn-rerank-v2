from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import cast

import httpx
import pytest
from textual.widgets import Input, Markdown, OptionList, Select, Static, Tabs

from hn_rerank.api import (
    API,
    APIError,
    InvalidProfile,
    Profile,
    load_profile,
    save_profile,
)
from hn_rerank.app import Reader, Setup
from hn_rerank.models import Feed, FeedStory, Window
from ._settle import settle


def sample_feed(version: int = 0, target: int = 0, window: Window = "1w") -> Feed:
    """The 1w window holds stories 1 and 2; "archive" holds story 3."""
    stories = [
        FeedStory(
            i,
            f"Story {i}",
            f"https://example.org/{i}",
            f"https://news.ycombinator.com/item?id={i}",
            "hn",
            100,
            10,
            i,
            float(4 - i),
        )
        for i in (1, 2, 3)
    ]
    if window == "archive":
        stories, orders = stories[2:], {"recommended": [3], "popular": [3]}
    elif window == "1w":
        stories = stories[:2]
        orders = {"recommended": [1, 2], "popular": [1], "explore": [2]}
    else:
        stories, orders = [], {}
    return Feed(
        2,
        window,
        stories,
        orders,
        {"up": 0, "neutral": 0, "down": 0},
        version,
        target,
        version >= target,
    )


class FakeServer:
    """One deck: `feed` is its 1w window; other windows come from
    `window_feeds` (or `sample_feed`) at the deck's version and state."""

    def __init__(self) -> None:
        self.feed = sample_feed()
        self.window_feeds: dict[Window, Feed] = {}
        self.requests: list[httpx.Request] = []
        self.fail_vote = False
        self.delay_vote = 0.0
        self.delay_summary = 0.0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/api/user"):
            return httpx.Response(200, json={"user_id": 1, "token": "test"})
        if path.endswith("/api/feed"):
            window = cast(Window, request.url.params.get("window", "1w"))
            return httpx.Response(200, json=self.feed_for(window).to_dict())
        if path.endswith("/api/tldr-detail") or "/api/tldr-cache/" in path:
            story_id = (
                int(path.rsplit("/", 1)[1])
                if "/api/tldr-cache/" in path
                else json.loads(request.content)["story_id"]
            )
            await asyncio.sleep(self.delay_summary if story_id == 1 else 0)
            return httpx.Response(
                200, json={"ok": True, "tldr": f"# Summary {story_id}"}
            )
        if path.endswith("/api/feedback"):
            await asyncio.sleep(self.delay_vote)
            if self.fail_vote:
                raise httpx.ReadError("secret URL must not appear", request=request)
            return httpx.Response(200, json={"ok": True, "target_version": 1})
        if path.endswith("/api/ranking-ready"):
            return httpx.Response(
                200,
                json={
                    "ready": self.feed.ready,
                    "current_version": self.feed.target_version,
                },
            )
        return httpx.Response(404)

    def feed_for(self, window: Window) -> Feed:
        if window == self.feed.window:
            return self.feed
        base = self.window_feeds.get(window) or sample_feed(window=window)
        return replace(
            base,
            version=self.feed.version,
            target_version=self.feed.target_version,
            ready=self.feed.ready,
            feedback_counts=self.feed.feedback_counts,
        )

    def feed_requests(self) -> list[str]:
        """The window of every feed request, in order."""
        return [
            r.url.params.get("window", "")
            for r in self.requests
            if r.url.path.endswith("/api/feed")
        ]

    def api(self) -> API:
        return API("https://example.org/hn/", "test", httpx.MockTransport(self))


async def test_navigation_resize_filters_and_late_summary(tmp_path: Path) -> None:
    fake = FakeServer()
    fake.delay_summary = 0.7
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        listing = app.query_one(OptionList)
        listing.focus()
        assert [s.id for s in app.stories] == [1, 2]
        await pilot.press("j")
        await pilot.pause(1)
        selected = app.selected()
        assert selected and selected.id == 2
        assert "Summary 2" in app.query_one(Markdown)._markdown
        await pilot.resize_terminal(80, 25)
        await pilot.pause()
        assert listing.display
        assert app.query_one(Markdown).display
        assert "narrow" in app.classes
        assert (
            app.query_one("#reading-pane").region.y
            > app.query_one("#headlines").region.y
        )
        app.query_one("#window", Select).value = "archive"
        await settle(pilot)
        assert [s.id for s in app.stories] == [3]
        await pilot.resize_terminal(120, 35)
        assert listing.display and app.query_one(Markdown).display


async def test_rapid_votes_apply_in_order_undo_and_failure_revert(
    tmp_path: Path,
) -> None:
    fake = FakeServer()
    fake.delay_vote = 0.15
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        # Both apply at once; the second lands on the next story, not dropped.
        app.action_vote("up")
        assert [s.id for s in app.stories] == [2]
        app.action_vote("down")
        assert not app.stories and app.rated == {1, 2}
        await settle(pilot)
        votes = [
            json.loads(r.content)
            for r in fake.requests
            if r.url.path.endswith("/feedback")
        ]
        assert votes == [
            {"story_id": 1, "action": "up"},
            {"story_id": 2, "action": "down"},
        ]
        app.action_refresh()
        await settle(pilot)
        assert not app.stories  # voted stories stay out of a refreshed deck
        app.action_undo()
        await settle(pilot)
        assert app.rated == {1} and [s.id for s in app.history] == [1]
        selected = app.selected()
        assert selected and selected.id == 2
        # A failed vote puts the story back, selected, and is not retried.
        fake.fail_vote = True
        fake.requests.clear()
        app.action_vote("down")
        assert app.rated == {1, 2}
        await settle(pilot)
        assert app.rated == {1} and [s.id for s in app.history] == [1]
        selected = app.selected()
        assert selected and selected.id == 2
        assert len([r for r in fake.requests if r.url.path.endswith("/feedback")]) == 1
        status = str(app.query_one("#status", Static).content)
        assert "not confirmed" in status and "secret" not in status


async def test_setup_typing_does_not_trigger_actions(tmp_path: Path) -> None:
    app = Reader(config_path=tmp_path / "missing.json")
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, Setup)
        entry = app.screen.query_one("#link", Input)
        entry.focus()
        await pilot.press("q", "1", "u", "r")
        assert entry.value == "q1ur"
        assert not app.rated
        assert not (tmp_path / "missing.json").exists()


async def test_narrow_select_focus_keeps_escape_and_focus_moves_reachable() -> None:
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test(size=(80, 30)) as pilot:
        await settle(pilot)
        # Narrow keeps both panes visible, so the summary sits between the
        # headline list and the selectors in the focus chain.
        await pilot.press("tab", "tab")
        selector = app.query_one("#sort", Select)
        assert app.focused is selector
        assert app.check_action("focus_next", ()) is True
        assert app.check_action("quit", ()) is True
        assert app.check_action("vote", ("up",)) is False
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is not selector
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is app.query_one("#headlines", OptionList)


def test_profile_persistence_and_redaction(tmp_path: Path) -> None:
    profile = Profile.from_link("https://example.org/hn/u/secret_token")
    assert profile.server == "https://example.org/hn/"
    assert "secret_token" not in repr(profile)
    path = tmp_path / "config" / "profile.json"
    save_profile(profile, path)
    assert load_profile(path) == profile
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    for value in (
        "https://example.org/hn/u/token?x=1",
        "http://remote/u/token",
        "https://user:pass@example.org/u/token",
    ):
        with pytest.raises(ValueError):
            Profile.from_link(value)


async def test_credentials_do_not_follow_redirects_or_create_after_invalid() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://evil.test/"})

    api = API("https://example.org/hn/", "secret", httpx.MockTransport(handler))
    with pytest.raises(APIError, match="redirected"):
        await api.validate()
    assert len(requests) == 1
    assert requests[0].url.path == "/hn/api/user"
    assert requests[0].headers["Cookie"] == "hn_token=secret"
    await api.close()
    api = API(
        "https://example.org/",
        "bad",
        httpx.MockTransport(lambda r: httpx.Response(401)),
    )
    with pytest.raises(InvalidProfile):
        await api.validate()
    await api.close()


async def test_rate_limit_and_zero_reset() -> None:
    api = API(
        "https://example.org/",
        "test",
        httpx.MockTransport(
            lambda r: httpx.Response(429, headers={"Retry-After": "7"})
        ),
    )
    with pytest.raises(APIError, match="7 seconds"):
        await api.feed()
    await api.close()
    fake = FakeServer()
    app = Reader(api=fake.api())
    async with app.run_test() as pilot:
        await settle(pilot)
        assert app.feed and app.feed.ready and app.feed.version == 0


class FailSummaryServer(FakeServer):
    """Fails foreground taps for chosen stories; cache reads always miss."""

    def __init__(self, fail_ids: set[int]) -> None:
        super().__init__()
        self.fail_ids = fail_ids

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/tldr-detail"):
            story_id = json.loads(request.content)["story_id"]
            if story_id in self.fail_ids:
                return httpx.Response(500)
        if "/api/tldr-cache/" in request.url.path:
            return httpx.Response(204)
        return await super().__call__(request)


async def test_summary_failure_hides_story_until_refresh(
    tmp_path: Path,
) -> None:
    fake = FailSummaryServer({1})
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        deadline = asyncio.get_running_loop().time() + 5.0
        while 1 not in app.unavailable:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("story 1 was not hidden")
            await pilot.pause(0.05)
        assert [s.id for s in app.stories] == [2]
        selected = app.selected()
        assert selected is not None and selected.id == 2
        assert "Skipped story 1" in str(app.query_one("#status", Static).content)
        deadline = asyncio.get_running_loop().time() + 5.0
        while "Summary 2" not in app.query_one(Markdown)._markdown:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("story 2 summary did not load")
            await pilot.pause(0.05)
        app.action_refresh()
        deadline = asyncio.get_running_loop().time() + 5.0
        while [s.id for s in app.stories] != [1, 2]:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("refresh did not restore story 1")
            await pilot.pause(0.05)


class EmptySummaryServer(FakeServer):
    """Returns the no-content flag for chosen stories; cache reads miss."""

    def __init__(self, empty_ids: set[int]) -> None:
        super().__init__()
        self.empty_ids = empty_ids

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/tldr-detail"):
            story_id = json.loads(request.content)["story_id"]
            if story_id in self.empty_ids:
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "tldr": "No article body or discussion available.",
                        "cached": False,
                        "retryable": True,
                        "empty": True,
                    },
                )
        if "/api/tldr-cache/" in request.url.path:
            return httpx.Response(204)
        return await super().__call__(request)


async def test_empty_summary_hides_story_until_refresh(
    tmp_path: Path,
) -> None:
    fake = EmptySummaryServer({1})
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        deadline = asyncio.get_running_loop().time() + 5.0
        while 1 not in app.unavailable:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("empty story 1 was not hidden")
            await pilot.pause(0.05)
        assert [s.id for s in app.stories] == [2]
        assert 1 not in app.summaries  # placeholder never cached
        assert "Skipped story 1" in str(app.query_one("#status", Static).content)
        app.action_refresh()
        deadline = asyncio.get_running_loop().time() + 5.0
        while [s.id for s in app.stories] != [1, 2]:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("refresh did not restore story 1")
            await pilot.pause(0.05)


def test_open_in_browser_opens_a_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as stdlib_subprocess

    import hn_rerank.app as app_module

    monkeypatch.delenv("HN_RERANK_BROWSER", raising=False)
    calls: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: (
            calls.append(a[0]) or stdlib_subprocess.CompletedProcess(a[0], 0)
        ),
    )
    monkeypatch.setattr("subprocess.Popen", lambda argv, **k: calls.append(argv))
    app_module.open_in_browser("https://example.org/x")
    assert calls == [
        ["/usr/bin/google-chrome", "https://example.org/x"],
        ["wmctrl", "-x", "-a", "google-chrome.Google-chrome"],
    ]


def test_open_in_browser_without_chrome_uses_default_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hn_rerank.app as app_module

    monkeypatch.delenv("HN_RERANK_BROWSER", raising=False)
    opened: list[str] = []
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(app_module.webbrowser, "open", opened.append)
    app_module.open_in_browser("https://example.org/x")
    assert opened == ["https://example.org/x"]


def test_hn_rerank_browser_overrides_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    import hn_rerank.app as app_module

    monkeypatch.setenv("HN_RERANK_BROWSER", "surf -z 1.2")
    calls: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr("subprocess.Popen", lambda argv, **k: calls.append(argv))
    app_module.open_in_browser("https://example.org/x")
    assert calls == [["surf", "-z", "1.2", "https://example.org/x"]]


async def test_stale_poll_does_not_cancel_summary_for_same_selection() -> None:
    fake = FakeServer()
    fake.feed = sample_feed(0, 1)
    fake.delay_summary = 1.4
    app = Reader(api=fake.api(), prefetch=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(2.1)
        assert "Summary 1" in app.query_one(Markdown)._markdown
        assert (
            len([r for r in fake.requests if r.url.path.endswith("tldr-detail")]) == 1
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL verification")
def test_windows_profile_has_private_acl(tmp_path: Path) -> None:
    import subprocess

    path = tmp_path / "private" / "profile.json"
    save_profile(Profile("https://example.org/hn/", "private"), path)
    script = """
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
foreach ($p in @($env:HN_RERANK_CONFIG_PATH, (Split-Path $env:HN_RERANK_CONFIG_PATH))) {
    $acl = Get-Acl -LiteralPath $p
    if (-not $acl.AreAccessRulesProtected -or $acl.Access.Count -ne 1) { exit 1 }
    if ($acl.Access[0].IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -ne $sid) { exit 2 }
}
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        check=False,
        env={**os.environ, "HN_RERANK_CONFIG_PATH": str(path)},
    )
    assert result.returncode == 0


@pytest.mark.parametrize("token", [None, 42, [], {}])
async def test_malformed_profile_response_returns_safe_error(token: object) -> None:
    api = API(
        "https://example.org/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"token": token})
        ),
    )
    try:
        with pytest.raises(InvalidProfile, match="invalid profile"):
            await api.validate()
    finally:
        await api.close()


@pytest.mark.parametrize("provisional", [False, True])
async def test_refresh_keeps_readable_summary_and_reports_failure(
    tmp_path: Path, provisional: bool
) -> None:
    class RefreshServer(FakeServer):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/api/tldr-detail") and json.loads(
                request.content
            ).get("force_refresh"):
                self.requests.append(request)
                await asyncio.sleep(0.3)
                if provisional:
                    return httpx.Response(
                        200,
                        json={"tldr": "# Summary 1", "cached": True, "retryable": True},
                    )
                return httpx.Response(503, json={"error": "Provider unavailable"})
            return await super().__call__(request)

    fake = RefreshServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.8)
        assert app.summaries[1] == "# Summary 1"
        app.query_one(OptionList).focus()
        await pilot.press("r")
        await pilot.pause(0.1)
        assert "Summary 1" in app.query_one("#summary", Markdown)._markdown
        await pilot.pause(0.9)
        selected = app.selected()
        assert selected is not None and selected.id == 1
        assert 1 not in app.unavailable
        assert "Summary 1" in app.query_one("#summary", Markdown)._markdown
        status = str(app.query_one("#status", Static).content)
        assert ("outdated or incomplete" if provisional else "refresh failed") in status


async def test_refresh_forces_only_selected_summary(tmp_path: Path) -> None:
    fake = FakeServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        fake.requests.clear()
        await pilot.press("r")
        await pilot.pause(0.8)
        forced = [
            json.loads(r.content)
            for r in fake.requests
            if r.url.path.endswith("/api/tldr-detail")
            and json.loads(r.content).get("force_refresh")
        ]
        assert forced == [{"story_id": 1, "force_refresh": True}]
        fake.requests.clear()
        app.reload(manual=False)
        await pilot.pause(0.8)
        assert not any(
            json.loads(r.content).get("force_refresh")
            for r in fake.requests
            if r.url.path.endswith("/api/tldr-detail")
        )


@pytest.mark.parametrize("width", [73, 146])
@pytest.mark.parametrize("origin", ["cycle", "tabs"])
async def test_rapid_filter_changes_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, origin: str
) -> None:
    app = Reader(api=FakeServer().api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(width, 38)) as pilot:
        await settle(pilot)
        rebuilds = 0
        original = app.rebuild

        def counted_rebuild(select_id: int | None = None) -> None:
            nonlocal rebuilds
            rebuilds += 1
            original(select_id)

        monkeypatch.setattr(app, "rebuild", counted_rebuild)
        # No yielding: reproduce several queued changes before either widget
        # has handled its peer's messages.
        if origin == "cycle":
            for _ in range(4):
                app.action_cycle_sort()
        else:
            for value in ("popular", "explore", "popular", "explore"):
                app.query_one("#sort-tabs", Tabs).active = f"sort-{value}"
        await settle(pilot)
        expected = "popular" if origin == "cycle" else "explore"
        assert app.query_one("#sort", Select).value == expected
        assert app.query_one("#sort-tabs", Tabs).active == f"sort-{expected}"
        settled = rebuilds
        assert 0 < settled <= 4
        await settle(pilot)
        assert rebuilds == settled  # No self-sustaining Select/Tabs echo.


async def test_s_cycles_sort_modes(tmp_path: Path) -> None:
    fake = FakeServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        assert str(app.query_one("#sort", Select).value) == "recommended"
        assert [s.id for s in app.stories] == [1, 2]
        for expected, story_ids in (
            ("popular", [1]),
            ("explore", [2]),
            ("recommended", [1, 2]),
        ):
            await pilot.press("s")
            await settle(pilot)
            assert str(app.query_one("#sort", Select).value) == expected
            assert [s.id for s in app.stories] == story_ids


async def test_explore_sort_is_shuffled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explore reshuffles client-side instead of following server rank order."""
    from dataclasses import replace

    fake = FakeServer()
    fake.feed = replace(fake.feed, orders={**fake.feed.orders, "explore": [1, 2]})

    def reverse(order: list[int]) -> None:
        order[:] = order[::-1]

    monkeypatch.setattr("random.shuffle", reverse)
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        assert [s.id for s in app.stories] == [1, 2]
        await pilot.press("s")  # popular: untouched by the shuffle.
        await settle(pilot)
        assert [s.id for s in app.stories] == [1]
        await pilot.press("s")  # explore: reversed server order.
        await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == "explore"
        assert [s.id for s in app.stories] == [2, 1]
        # Rebuild must copy: the shared server order stays intact for
        # prefetch entry points into other sorts.
        assert fake.feed.orders["explore"] == [1, 2]


async def test_explore_order_is_stable_within_a_visit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebuilds (votes, polls) keep Explore's order; leaving and coming back
    reshuffles."""
    from dataclasses import replace

    fake = FakeServer()
    fake.feed = replace(
        fake.feed,
        stories=sample_feed(window="archive").stories + fake.feed.stories,
        orders={**fake.feed.orders, "explore": [1, 2, 3]},
    )
    shuffles = 0

    def rotate(order: list[int]) -> None:
        # Each full draw rotates one step further, so a redraw is visible.
        nonlocal shuffles
        if len(order) == 3:
            shuffles += 1
            k = shuffles % 3
            order[:] = order[k:] + order[:k]

    monkeypatch.setattr("random.shuffle", rotate)
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        await pilot.press("s", "s")
        await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == "explore"
        first = [s.id for s in app.stories]
        assert sorted(first) == [1, 2, 3]
        for _ in range(3):
            app.rebuild()
        assert [s.id for s in app.stories] == first
        assert shuffles == 1
        app.rated.add(first[0])
        app.rebuild()
        assert [s.id for s in app.stories] == first[1:]
        app.rated.clear()
        await pilot.press("s", "s", "s")  # full cycle back to explore
        await settle(pilot)
        assert str(app.query_one("#sort", Select).value) == "explore"
        assert [s.id for s in app.stories] != first


class FlakySummaryServer(FakeServer):
    """Rate-limits foreground taps for story 1; cache reads always miss."""

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if (
            request.url.path.endswith("/api/tldr-detail")
            and json.loads(request.content)["story_id"] == 1
        ):
            return httpx.Response(429, headers={"Retry-After": "30"})
        if "/api/tldr-cache/" in request.url.path:
            return httpx.Response(204)
        return await super().__call__(request)


async def test_transient_summary_failure_keeps_story(tmp_path: Path) -> None:
    fake = FlakySummaryServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        deadline = asyncio.get_running_loop().time() + 5.0
        while "Rate limited" not in app.query_one(Markdown)._markdown:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("rate limit was not shown")
            await pilot.pause(0.05)
        assert 1 not in app.unavailable
        assert [s.id for s in app.stories] == [1, 2]
        selected = app.selected()
        assert selected is not None and selected.id == 1


async def test_version_poll_keeps_hidden_stories_hidden(tmp_path: Path) -> None:
    fake = FakeServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.unavailable.add(1)
        app.reload(manual=False)
        await settle(pilot)
        assert [s.id for s in app.stories] == [2]
        app.action_refresh()
        await settle(pilot)
        assert [s.id for s in app.stories] == [1, 2]


def test_open_in_browser_falls_back_on_launch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hn_rerank.app as app_module

    monkeypatch.delenv("HN_RERANK_BROWSER", raising=False)
    opened: list[str] = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/google-chrome")

    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("google-chrome")

    monkeypatch.setattr("subprocess.Popen", missing)
    monkeypatch.setattr(app_module.webbrowser, "open", opened.append)
    app_module.open_in_browser("https://example.org/x")
    assert opened == ["https://example.org/x"]


async def test_y_copies_comments_link_else_article_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses

    import hn_rerank.app as app_module

    copied: list[str] = []

    def copy(url: str) -> bool:
        copied.append(url)
        return True

    monkeypatch.setattr(app_module, "copy_with_system_tool", copy)
    fake = FakeServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("y")
        assert copied == ["https://news.ycombinator.com/item?id=1"]
        selected = app.selected()
        assert selected is not None
        no_comments = dataclasses.replace(selected, comments_url="")
        monkeypatch.setattr(app, "selected", lambda: no_comments)
        await pilot.press("y")
        assert copied[-1] == "https://example.org/1"
        monkeypatch.setattr(
            app,
            "selected",
            lambda: dataclasses.replace(no_comments, article_url="javascript:x"),
        )
        await pilot.press("y")
        assert len(copied) == 2


def test_copy_with_system_tool_skips_failing_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    import hn_rerank.app as app_module

    calls: list[list[str]] = []
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    def run(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        if command[0] != "xsel":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("subprocess.run", run)
    assert app_module.copy_with_system_tool("u") is True
    assert [c[0] for c in calls] == ["pbcopy", "xclip", "xsel"]


async def test_failed_refresh_does_not_force_a_later_regeneration(
    tmp_path: Path,
) -> None:
    """r arms a forced regeneration; if the feed fetch fails it must disarm."""

    class FeedFailServer(FakeServer):
        fail_feed = False

        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if self.fail_feed and request.url.path.endswith("/api/feed"):
                return httpx.Response(500)
            return await super().__call__(request)

    fake = FeedFailServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        fake.fail_feed = True
        app.action_refresh()
        await settle(pilot)
        fake.fail_feed = False
        app.reload(manual=False)
        await settle(pilot)
        app.query_one(OptionList).focus()
        await pilot.press("j", "k")
        await settle(pilot)
        forced = [
            r
            for r in fake.requests
            if r.url.path.endswith("/api/tldr-detail")
            and json.loads(r.content).get("force_refresh")
        ]
        assert len(forced) == 1  # only the r itself


async def test_only_summary_generation_gets_the_long_timeout() -> None:
    from hn_rerank.api import GENERATION_TIMEOUT, REQUEST_TIMEOUT

    seen: dict[str, float] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path.rsplit("/", 1)[-1]] = request.extensions["timeout"][
            "read"
        ]
        if request.url.path.endswith("/api/tldr-detail"):
            return httpx.Response(200, json={"ok": True, "tldr": "# T"})
        return httpx.Response(204)

    api = API("https://example.org/hn/", "test", httpx.MockTransport(handler))
    await api.summary(1)
    await api.cached_summary(1)
    await api.close()
    assert seen == {"tldr-detail": GENERATION_TIMEOUT, "1": REQUEST_TIMEOUT}


async def test_failed_undo_hides_the_story_again(tmp_path: Path) -> None:
    fake = FakeServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        await settle(pilot)
        fake.fail_vote = True
        app.action_undo()
        assert not app.rated  # shown at once
        await settle(pilot)
        assert app.rated == {1} and [s.id for s in app.history] == [1]
        assert [s.id for s in app.stories] == [2]
        assert "not confirmed" in str(app.query_one("#status", Static).content)


async def test_h_and_l_step_through_sorts_and_wrap(tmp_path: Path) -> None:
    fake = FakeServer()
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        app.query_one(OptionList).focus()
        sort = app.query_one("#sort", Select)
        seen = []
        for key in "lllhh":
            await pilot.press(key)
            await pilot.pause()
            seen.append(str(sort.value))
        assert seen == ["popular", "explore", "recommended", "explore", "popular"]
        assert app.query_one("#sort-tabs", Tabs).active == "sort-popular"
