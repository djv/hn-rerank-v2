from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from database import Database, InteractionEvent, InteractionEventType, Story
from scripts.source_yield_report import load_yields, render


def _event(
    eid: str, user_id: int, story_id: int, kind: InteractionEventType, at: float
) -> InteractionEvent:
    return InteractionEvent(
        event_id=eid,
        client_session_id="s",
        user_id=user_id,
        story_id=story_id,
        event_type=kind,
        dashboard_version=0,
        position=0,
        sort_mode="rank",
        age_filter="all",
        source_filter="all",
        ranker_arm="prod",
        occurred_at=at,
    )


def test_yield_counts_shown_pairs_votes_and_window(tmp_path: Path) -> None:
    now = time.time()
    path = tmp_path / "y.db"
    db = Database(str(path))
    for sid, source in [(1, "hn"), (2, "hn"), (3, "rss_a"), (4, "rss_a")]:
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
    db.upsert_story(
        Story(
            id=5,
            title="old",
            url="https://x/5",
            score=1,
            time=int(now - 90 * 86400),
            text_content="t",
            source="rss_old",
        )
    )
    db.insert_interaction_events(
        [
            _event("a", 1, 1, "impression", now),
            _event("b", 1, 1, "impression", now),  # same pair shown twice
            _event("c", 2, 1, "impression", now),
            _event("d", 1, 2, "impression", now),
            _event("e", 1, 3, "impression", now),
            _event("f", 1, 3, "article_open", now),
            _event("g", 1, 5, "impression", now - 60 * 86400),  # outside window
        ]
    )
    db.upsert_feedback(1, 1, "up")
    db.upsert_feedback(2, 1, "down")
    db.upsert_feedback(1, 3, "up")
    db.upsert_feedback(1, 4, "neutral")  # voted without an impression
    db.close()

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        by_source = {y.source: y for y in load_yields(conn, now - 30 * 86400, None)}
        only_user_2 = {y.source: y for y in load_yields(conn, now - 30 * 86400, 2)}
    finally:
        conn.close()

    hn = by_source["hn"]
    assert (hn.fetched, hn.shown, hn.voted, hn.up_shown) == (2, 3, 2, 1)
    assert (hn.up, hn.down) == (1, 1)
    assert hn.up_rate == 1 / 3
    rss = by_source["rss_a"]
    assert (rss.shown, rss.opened, rss.up_shown, rss.up, rss.neutral) == (1, 1, 1, 1, 1)
    assert "rss_old" not in by_source
    assert (only_user_2["hn"].shown, only_user_2["hn"].up_rate) == (1, 0.0)

    table = render(list(by_source.values()), min_shown=1)
    assert "33%" in table and "100%" in table
    assert render(list(only_user_2.values()), min_shown=1).count("<- low") == 1
