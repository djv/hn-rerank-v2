from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
from textual.widgets import Input, Markdown, OptionList, Select, Static

from hn_rerank.api import (
    API,
    APIError,
    InvalidProfile,
    Profile,
    load_profile,
    save_profile,
)
from hn_rerank.app import Reader, Setup
from hn_rerank.models import Feed, FeedStory


def sample_feed(version: int = 0, target: int = 0) -> Feed:
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
            ["recent_mixed"] if i < 3 else ["archive_mixed"],
            i == 1,
            i == 2,
        )
        for i in (1, 2, 3)
    ]
    return Feed(
        1,
        stories,
        {
            "recommended:recent": [1, 2],
            "recommended:archive": [3],
            "popular:recent": [1],
            "explore:recent": [2],
            "date:recent": [2, 1],
        },
        {"up": 0, "neutral": 0, "down": 0},
        version,
        target,
        version >= target,
    )


class FakeServer:
    def __init__(self) -> None:
        self.feed = sample_feed()
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
            return httpx.Response(200, json=self.feed.to_dict())
        if path.endswith("/api/tldr-detail"):
            story_id = json.loads(request.content)["story_id"]
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

    def api(self) -> API:
        return API("https://example.org/hn/", "test", httpx.MockTransport(self))


async def test_navigation_resize_filters_and_late_summary(tmp_path: Path) -> None:
    fake = FakeServer()
    fake.delay_summary = 0.7
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.4)
        listing = app.query_one(OptionList)
        listing.focus()
        assert [s.id for s in app.stories] == [1, 2]
        await pilot.press("j")
        await pilot.pause(1)
        selected = app.selected()
        assert selected and selected.id == 2
        assert "Summary 2" in app.query_one(Markdown)._markdown
        await pilot.resize_terminal(80, 25)
        await pilot.press("enter")
        assert not listing.display
        assert app.query_one(Markdown).display
        await pilot.press("escape")
        assert listing.display
        app.query_one("#age", Select).value = "archive"
        await pilot.pause()
        assert [s.id for s in app.stories] == [3]
        await pilot.resize_terminal(120, 35)
        assert listing.display and app.query_one(Markdown).display


async def test_vote_duplicate_failure_undo_and_stale_exclusion(tmp_path: Path) -> None:
    fake = FakeServer()
    fake.delay_vote = 0.15
    app = Reader(api=fake.api(), config_path=tmp_path / "profile.json")
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.15)
        app.query_one(OptionList).focus()
        app.action_vote("up")
        app.action_vote("down")
        await pilot.pause(0.3)
        votes = [r for r in fake.requests if r.url.path.endswith("/feedback")]
        assert len(votes) == 1
        assert app.rated == {1}
        assert [s.id for s in app.stories] == [2]
        app.action_refresh()
        await pilot.pause(0.1)
        assert [s.id for s in app.stories] == [2]
        app.action_undo()
        await pilot.pause(0.3)
        assert not app.rated and not app.history
        selected = app.selected()
        assert selected and selected.id == 1
        fake.fail_vote = True
        app.action_vote("down")
        await pilot.pause(0.3)
        assert not app.rated and not app.history
        selected = app.selected()
        assert selected and selected.id == 1
        assert "secret" not in str(app.query_one("#status", Static).content)


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
    app.target = 12
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        assert app.target == 0
        assert app.feed and app.feed.ready and app.feed.version == 0


async def test_stale_poll_does_not_cancel_summary_for_same_selection() -> None:
    fake = FakeServer()
    fake.feed = sample_feed(0, 1)
    fake.delay_summary = 1.4
    app = Reader(api=fake.api())
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
