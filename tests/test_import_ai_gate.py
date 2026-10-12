from __future__ import annotations

from dataclasses import replace
from email.utils import formatdate
from pathlib import Path

import httpx
import pytest

from database import Database, Story, User
from pipeline import Config, RankedStory, WindowDeck, WindowViews
from scripts.check_import_ai_recommended import (
    Issue,
    check_issue,
    fetch_publisher,
    latest_issue,
    main,
)
from scripts import check_import_ai_recommended
from server import DeckState, Handler, create_app

NOW = 1_800_000_000
ISSUE = Issue(
    "Import AI 500: Integration fixture", "https://jack-clark.net/issue-500/", NOW - 60
)


def _runtime(db: Database, user: User, position: int, ready: bool) -> type[Handler]:
    class Runtime(Handler):
        _decks = {}
        _dashboard_versions = {}
        _pool_generation = 1 if ready else 2

        @classmethod
        def _trigger_warm(
            cls,
            user: User,
            version: int,
            delay_s: float = 0.0,
            *,
            expedite: bool = True,
        ) -> None:
            pass

    Runtime.db = db
    Runtime.config = Config(db_path=db.db_path)
    db.upsert_story(
        Story(
            200,
            "Previously voted unrelated story",
            "https://example.org/old",
            1,
            NOW - 86400,
            "",
        )
    )
    db.upsert_feedback(user.id, 200, "up")
    stories = [
        Story(i, f"Other {i}", f"https://example.org/{i}", 1, NOW - 60, "")
        for i in range(1, 13)
    ]
    stories[position - 1] = Story(
        100,
        ISSUE.title,
        ISSUE.url,
        0,
        ISSUE.published_at,
        "",
        source="rss_jack-clark_net",
    )
    for story in stories:
        db.upsert_story(story)
    other = db.create_user("other-profile")
    db.upsert_feedback(other.id, 100, "up")
    ranked = tuple(RankedStory(s, score=1.0, best_match_title="") for s in stories)
    deck = WindowDeck(
        {w: WindowViews(recommended=ranked) for w in ("12h", "1d", "1w", "1m")}
    )
    Runtime._decks = {user.id: DeckState(deck, float(NOW), 1)}
    return Runtime


@pytest.mark.parametrize(
    ("position", "ready", "expected"),
    [(8, True, "PASS"), (9, True, "FAIL"), (1, False, "FAIL")],
)
def test_gate_checks_actual_served_api_and_visible_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    position: int,
    ready: bool,
    expected: str,
) -> None:
    monkeypatch.setattr("time.time", lambda: float(NOW))
    path = tmp_path / "gate.db"
    db = Database(str(path))
    user = db.create_user("fixture-secret-not-for-output")
    runtime = _runtime(db, user, position, ready)
    try:
        before = db.execute("SELECT COUNT(*) FROM interaction_events")
        readonly = Database(str(path), read_only=True)
        try:
            with httpx.Client(
                base_url="http://localhost",
                transport=httpx.WSGITransport(app=create_app(runtime)),
            ) as client:
                result = check_issue(readonly, user.id, ISSUE, client, NOW)
        finally:
            readonly.close()
        assert result.status == expected
        assert len(result.windows) == 4
        assert result.windows[0].position == (position if position <= 8 else None)
        assert db.execute("SELECT COUNT(*) FROM interaction_events") == before
        assert user.token not in repr(result)
    finally:
        db.close()


@pytest.mark.parametrize("state", ["seen", "voted", "old", "missing"])
def test_gate_applicability_and_ingestion(tmp_path: Path, state: str) -> None:
    db = Database(str(tmp_path / "gate.db"))
    user = db.create_user("user")
    other = db.create_user("other")
    story = Story(100, ISSUE.title, ISSUE.url, 0, ISSUE.published_at, "")
    if state != "missing":
        db.upsert_story(story)
    if state == "voted":
        db.upsert_feedback(user.id, story.id, "neutral")
    if state == "seen":
        db.execute(
            "INSERT INTO interaction_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "seen",
                "session",
                user.id,
                100,
                "impression",
                1,
                0,
                "recommended",
                "1w",
                "all",
                "",
                NOW,
                None,
                NOW,
                "[]",
            ),
        )
    issue = replace(ISSUE, published_at=NOW - 8 * 86400) if state == "old" else ISSUE
    # Another profile's vote must not change this profile's eligibility.
    if state != "missing":
        db.upsert_feedback(other.id, story.id, "up")

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Inapplicable/missing gate must not call feed API")

    try:
        with httpx.Client(
            base_url="http://localhost", transport=httpx.MockTransport(forbidden)
        ) as client:
            result = check_issue(db, user.id, issue, client, NOW)
        assert result.status == ("FAIL" if state == "missing" else "NOT_APPLICABLE")
    finally:
        db.close()


def test_publisher_issue_selection_requires_date_and_weekly_issue_title() -> None:
    def item(title: str, at: int) -> str:
        return f"<item><title>{title}</title><link>https://jack-clark.net/{at}/</link><pubDate>{formatdate(at, usegmt=True)}</pubDate></item>"

    content = (
        "<rss><channel>"
        + item("Import AI 499: Previous", NOW - 7 * 86400)
        + item("Import AI 500: Current", NOW - 60)
        + item("Special interview", NOW)
        + "</channel></rss>"
    ).encode()
    assert latest_issue(content, NOW).title == "Import AI 500: Current"
    with pytest.raises(ValueError, match="no dated"):
        latest_issue(b"<rss><channel/></rss>", NOW)
    with pytest.raises(ValueError, match="URL/date"):
        latest_issue(
            b"<rss><channel><item><title>Import AI 500: Missing</title><link>https://jack-clark.net/500/</link></item></channel></rss>",
            NOW,
        )


@pytest.mark.parametrize(
    "origin",
    ["https://example.org", "http://user@localhost:8766", "http://localhost:8766/api"],
)
def test_nonlocal_or_ambiguous_origin_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid origin must not contact publisher or profile API")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    path = tmp_path / "must-not-create.db"
    with pytest.raises(SystemExit) as exc:
        main(["--db", str(path), "--user-id", "1", "--base-url", origin])
    assert exc.value.code == 2
    assert not path.exists()


@pytest.mark.asyncio
async def test_publisher_transport_failure_cannot_be_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(
        client: httpx.AsyncClient, url: str, headers: dict[str, str]
    ) -> tuple[int, str, dict[str, str]]:
        return 503, "", {}

    monkeypatch.setattr(
        check_import_ai_recommended, "fetch_with_urllib_fallback", unavailable
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await fetch_publisher()
