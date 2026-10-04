from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from database import Database, InteractionEvent, Story
from scripts.badge_yield_report import load_report, render_table, wilson


def _impression(
    eid: str, story_id: int, at: float, sort_mode: str, badges: str = ""
) -> InteractionEvent:
    return InteractionEvent(
        event_id=eid,
        client_session_id="s",
        user_id=1,
        story_id=story_id,
        event_type="impression",
        dashboard_version=0,
        position=0,
        sort_mode=sort_mode,
        age_filter="1d",
        source_filter="all",
        ranker_arm="tui_observed",
        occurred_at=at,
        badges=badges,
    )


def test_badge_yield_uses_first_impression_and_counts_each_badge(
    tmp_path: Path,
) -> None:
    now = time.time()
    path = tmp_path / "b.db"
    db = Database(str(path))
    for sid, source in [(1, "hn"), (2, "hn"), (3, "rss_a"), (4, "hn")]:
        db.upsert_story(
            Story(
                id=sid,
                title="t",
                url=f"https://x/{sid}",
                score=1,
                time=int(now - 3600),
                text_content="t",
                source=source,
            )
        )
    db.insert_interaction_events(
        [
            # Story 1: first shown in explore with two badges, later in
            # recommended; only the first impression counts.
            _impression("a", 1, now - 50, "explore", "novel,hot"),
            _impression("b", 1, now - 10, "recommended"),
            _impression("c", 2, now - 40, "popular", "hot"),
            _impression("d", 3, now - 30, "recommended"),
            _impression("e", 4, now - 90 * 86400, "recommended"),  # before window
        ]
    )
    db.upsert_feedback(1, 1, "up")
    db.upsert_feedback(1, 2, "down")
    db.close()

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        report = load_report(conn, now - 86400, 1)
    finally:
        conn.close()

    assert {k: (v.shown, v.up, v.down) for k, v in report.by_feed.items()} == {
        "explore": (1, 1, 0),
        "popular": (1, 0, 1),
        "recommended": (1, 0, 0),
    }
    assert {k: (v.shown, v.up, v.down) for k, v in report.by_badge.items()} == {
        "novel": (1, 1, 0),
        "hot": (2, 1, 1),
        "(none)": (1, 0, 0),
    }
    assert (report.by_source["hn"].shown, report.by_source["other"].shown) == (2, 1)
    assert "50.0%" in render_table("badge", report.by_badge)


def test_wilson_interval_brackets_the_rate() -> None:
    low, high = wilson(1, 78)
    assert low < 1 / 78 < high
    assert low == pytest.approx(0.0023, abs=1e-3)
    assert high == pytest.approx(0.0690, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)
