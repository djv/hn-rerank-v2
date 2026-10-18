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


def _rss(issue: Issue) -> str:
    return f"<rss><channel><item><title>{issue.title}</title><link>{issue.url}</link><pubDate>{formatdate(issue.published_at, usegmt=True)}</pubDate></item></channel></rss>"


@pytest.mark.parametrize(
    ("publisher_age", "stored_age", "omit", "expected", "windows"),
    [
        (14 * 3600, 3 * 3600, "12h", "FAIL", 4),
        (3 * 3600, 14 * 3600, "", "PASS", 3),
        (2 * 86400, 2 * 86400, "", "PASS", 2),
        (60, 40 * 86400, "", "FAIL", 0),
    ],
)
def test_windows_follow_stored_crosspost_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher_age: int,
    stored_age: int,
    omit: str,
    expected: str,
    windows: int,
) -> None:
    monkeypatch.setattr("time.time", lambda: float(NOW))
    db = Database(str(tmp_path / "gate.db"))
    try:
        user = db.create_user("fixture")
        runtime = _runtime(db, user, 1, True, age=stored_age, omit=omit)
        with httpx.Client(
            base_url="http://localhost",
            transport=httpx.WSGITransport(app=create_app(runtime)),
        ) as client:
            result = check_issue(
                db,
                user.id,
                replace(ISSUE, published_at=NOW - publisher_age),
                client,
                NOW,
            )
        assert result.status == expected
        assert len(result.windows) == windows
    finally:
        db.close()


def test_future_newest_issue_fails_instead_of_selecting_previous() -> None:
    content = _rss(ISSUE).replace(
        "</channel>",
        _rss(replace(ISSUE, title="Import AI 501: Future", published_at=NOW + 30))
        .split("<channel>")[1]
        .split("</channel>")[0]
        + "</channel>",
    )
    with pytest.raises(ValueError, match="future"):
        latest_issue(content, NOW)


def test_offhost_publisher_issue_fails() -> None:
    with pytest.raises(ValueError, match="URL/date"):
        latest_issue(_rss(replace(ISSUE, url="https://example.org/500")), NOW)


@pytest.mark.parametrize("state", ["visible", "voted", "seen"])
def test_duplicate_url_ids_share_visibility_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    monkeypatch.setattr("time.time", lambda: float(NOW))
    db = Database(str(tmp_path / "gate.db"))
    try:
        user = db.create_user("fixture")
        runtime = _runtime(db, user, 1, True, story_id=101, age=3 * 3600)
        db.upsert_story(Story(100, ISSUE.title, ISSUE.url, 0, NOW - 14 * 3600, ""))
        if state == "voted":
            db.upsert_feedback(user.id, 100, "neutral")
        elif state == "seen":
            db.execute(
                """INSERT INTO interaction_events (
                    event_id, client_session_id, user_id, story_id, event_type,
                    dashboard_version, position, sort_mode, age_filter,
                    source_filter, ranker_arm, occurred_at, duration_ms,
                    received_at, badges
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        with httpx.Client(
            base_url="http://localhost",
            transport=httpx.WSGITransport(app=create_app(runtime)),
        ) as client:
            result = check_issue(db, user.id, ISSUE, client, NOW)
        assert result.status == ("PASS" if state == "visible" else "NOT_APPLICABLE")
        if state == "visible":
            assert len(result.windows) == 4
    finally:
        db.close()


def test_profile_mismatch_fails_before_feed(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "gate.db"))
    try:
        user = db.create_user("fixture")
        db.upsert_story(Story(100, ISSUE.title, ISSUE.url, 0, NOW - 60, ""))

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/user"
            return httpx.Response(200, json={"user_id": user.id + 1})

        with httpx.Client(
            base_url="http://localhost", transport=httpx.MockTransport(respond)
        ) as client:
            assert check_issue(db, user.id, ISSUE, client, NOW).status == "FAIL"
    finally:
        db.close()


@pytest.mark.parametrize(
    ("state", "exit_code"),
    [("visible", 0), ("missing", 1), ("voted", 3), ("future", 1)],
)
def test_cli_exits_using_actual_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: str,
    exit_code: int,
) -> None:
    monkeypatch.setattr("time.time", lambda: float(NOW))
    path = tmp_path / "gate.db"
    db = Database(str(path))
    try:
        user = db.create_user("fixture-secret")
        runtime = _runtime(db, user, 9 if state == "missing" else 1, True)
        if state == "voted":
            db.upsert_feedback(user.id, 100, "neutral")

        async def publisher() -> str:
            return _rss(
                replace(ISSUE, published_at=NOW + 30) if state == "future" else ISSUE
            )

        real_client = httpx.Client

        def client(**kwargs: object) -> httpx.Client:
            assert kwargs["trust_env"] is False
            assert kwargs["follow_redirects"] is False
            return real_client(
                base_url="http://localhost",
                transport=httpx.WSGITransport(app=create_app(runtime)),
            )

        monkeypatch.setattr(check_import_ai_recommended, "fetch_publisher", publisher)
        monkeypatch.setattr(check_import_ai_recommended.httpx, "Client", client)
        with pytest.raises(SystemExit) as exc:
            main(["--db", str(path), "--user-id", str(user.id)])
        assert exc.value.code == exit_code
        captured = capsys.readouterr()
        assert user.token not in captured.out + captured.err
        assert not db.execute("SELECT 1 FROM interaction_events")
    finally:
        db.close()


def _runtime(
    db: Database,
    user: User,
    position: int,
    ready: bool,
    *,
    age: int = 60,
    omit: str = "",
    story_id: int = 100,
) -> type[Handler]:
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
        story_id,
        ISSUE.title,
        ISSUE.url,
        0,
        NOW - age,
        "",
        source="rss_jack-clark_net",
    )
    for story in stories:
        db.upsert_story(story)
    other = db.create_user("other-profile")
    db.upsert_feedback(other.id, story_id, "up")
    ranked = tuple(RankedStory(s, score=1.0, best_match_title="") for s in stories)
    deck = WindowDeck(
        {
            w: WindowViews(
                recommended=tuple(
                    r for r in ranked if w != omit or r.story.id != story_id
                )
            )
            for w in ("12h", "1d", "1w", "1m")
        }
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
            """INSERT INTO interaction_events (
                event_id, client_session_id, user_id, story_id, event_type,
                dashboard_version, position, sort_mode, age_filter,
                source_filter, ranker_arm, occurred_at, duration_ms,
                received_at, badges
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
