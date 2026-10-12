"""Read-only live gate: a new, unseen, unvoted Import AI must be visible.

Exit 0: PASS; 1: FAIL/error; 3: not applicable (old, seen, or voted).
This checks actual served rankings; it never boosts a source or clears history.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import feedparser
import httpx

from clients.tui.src.hn_rerank.models import Feed, Window
from database import Database
from http_fetch import fetch_with_urllib_fallback
from pipeline.enrichment import RSS_USER_AGENT
from pipeline.ranking import WINDOW_SECONDS, in_window

PUBLISHER_FEED = "https://jack-clark.net/feed/"
VISIBLE_CARDS = 8


@dataclass(frozen=True)
class Issue:
    title: str
    url: str
    published_at: int


@dataclass(frozen=True)
class WindowCheck:
    window: Window
    position: int | None
    ready: bool


@dataclass(frozen=True)
class GateResult:
    status: Literal["PASS", "FAIL", "NOT_APPLICABLE"]
    reason: str
    issue: Issue
    windows: tuple[WindowCheck, ...] = ()


def latest_issue(content: bytes | str, now: float) -> Issue:
    issues: list[Issue] = []
    for entry in feedparser.parse(content).entries:
        title = entry.get("title", "")
        if not isinstance(title, str) or not re.match(r"^Import AI \d+:", title):
            continue
        url = entry.get("link")
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if (
            not isinstance(url, str)
            or urlsplit(url).hostname
            not in {
                "jack-clark.net",
                "www.jack-clark.net",
            }
            or published is None
        ):
            raise ValueError("Import AI entry lacks a valid publisher URL/date")
        at = calendar.timegm(published)
        if at <= now:
            issues.append(Issue(title, url, at))
    if not issues:
        raise ValueError("Publisher feed contains no dated Import AI issue")
    return max(issues, key=lambda issue: issue.published_at)


async def fetch_publisher() -> str:
    """Use the ordinary RSS transport, but fail explicitly on fetch errors."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        status, content, _ = await fetch_with_urllib_fallback(
            client, PUBLISHER_FEED, {"User-Agent": RSS_USER_AGENT}
        )
    if status != 200:
        raise RuntimeError(f"Publisher RSS fetch returned HTTP {status}")
    return content


def check_issue(
    db: Database, user_id: int, issue: Issue, client: httpx.Client, now: float
) -> GateResult:
    user = db.get_user_by_id(user_id)
    if user is None:
        raise ValueError("Existing profile not found; gate never creates profiles")
    if not 0 <= now - issue.published_at <= WINDOW_SECONDS["1w"]:
        return GateResult(
            "NOT_APPLICABLE", "Latest issue is outside the fresh week", issue
        )
    # Exact publisher URL covers an HN crosspost too; its vote/view also counts.
    rows = db.execute("SELECT id FROM stories WHERE url = ?", (issue.url,))
    if not rows:
        return GateResult("FAIL", "Fresh issue has not been ingested", issue)
    ids = {int(row[0]) for row in rows}
    for story_id in ids:
        if db.execute(
            "SELECT 1 FROM feedback WHERE user_id = ? AND story_id = ? LIMIT 1",
            (user_id, story_id),
        ):
            return GateResult("NOT_APPLICABLE", "User has voted on this issue", issue)
        if db.execute(
            "SELECT 1 FROM interaction_events "
            "WHERE user_id = ? AND story_id = ? LIMIT 1",
            (user_id, story_id),
        ):
            return GateResult(
                "NOT_APPLICABLE", "User has already seen this issue", issue
            )
    checks: list[WindowCheck] = []
    client.cookies.set("hn_token", user.token)
    for window in WINDOW_SECONDS:
        if not in_window(issue.published_at, window, now):
            continue
        response = client.get("/api/feed", params={"window": window})
        response.raise_for_status()
        feed = Feed.parse(response.json())
        if feed.window != window:
            raise ValueError("Feed returned a different window")
        order = feed.orders["recommended"][:VISIBLE_CARDS]
        position = next((i + 1 for i, sid in enumerate(order) if sid in ids), None)
        checks.append(WindowCheck(window, position, feed.ready))
    if not all(check.ready and check.position is not None for check in checks):
        return GateResult(
            "FAIL",
            "Issue missing from visible Recommended or feed not ready",
            issue,
            tuple(checks),
        )
    return GateResult(
        "PASS", "Fresh unseen issue is visible in Recommended", issue, tuple(checks)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    args = parser.parse_args(argv)
    origin = urlsplit(args.base_url)
    if (
        origin.scheme not in {"http", "https"}
        or origin.hostname not in {"127.0.0.1", "localhost", "::1"}
        or origin.username is not None
        or origin.password is not None
        or origin.path not in {"", "/"}
        or origin.query
        or origin.fragment
    ):
        parser.error(
            "--base-url must be a loopback origin (profile cookie stays local)"
        )
    try:
        now = time.time()
        issue = latest_issue(asyncio.run(fetch_publisher()), now)
        db = Database(str(args.db), read_only=True)
        try:
            with httpx.Client(
                base_url=args.base_url,
                timeout=30,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                result = check_issue(db, args.user_id, issue, client, now)
        finally:
            db.close()
        print(json.dumps(asdict(result), indent=2))
    except Exception:
        logging.exception("Import AI Recommended gate failed")
        raise SystemExit(1) from None
    raise SystemExit({"PASS": 0, "FAIL": 1, "NOT_APPLICABLE": 3}[result.status])


if __name__ == "__main__":
    main()
