"""Queue telemetry boundaries and additive migration, using memory-only DBs."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from database import Database, Story
from server import _parse_interaction_event


def raw_event() -> dict[str, object]:
    return {
        "event_id": "11111111-1111-4111-8111-111111111111",
        "client_session_id": "22222222-2222-4222-8222-222222222222",
        "story_id": 501,
        "event_type": "impression",
        "dashboard_version": 0,
        "position": 3,
        "sort_mode": "explore",
        "window": "1w",
        "source_filter": "mixed",
        "ranker_arm": "tui_observed",
        "occurred_at": 1_700_000_000.0,
    }


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"surface": "views"},
        {"surface": "queue", "queue_position": 0},
        {"surface": "queue", "queue_position": 2**63 - 1},
    ],
)
def test_accept_surface_fields(extra: dict[str, object]) -> None:
    event = _parse_interaction_event({**raw_event(), **extra}, 7)
    assert event.surface == extra.get("surface", "")
    assert event.queue_position == extra.get("queue_position")
    assert event.position == 3 and event.sort_mode == "explore"


@pytest.mark.parametrize(
    "extra",
    [
        {"surface": ""},
        {"surface": "other"},
        {"surface": None},
        {"surface": []},
        {"surface": {}},
        {"surface": True},
        {"surface": "queue"},
        {"queue_position": 0},
        {"surface": "views", "queue_position": 0},
        {"surface": "views", "queue_position": None},
        *(
            {"surface": "queue", "queue_position": value}
            for value in [None, True, False, -1, 2**63, 1.5, "0", [], {}]
        ),
    ],
)
def test_reject_surface_fields(extra: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _parse_interaction_event({**raw_event(), **extra}, 7)


@given(position=st.integers())
def test_queue_position_signed_64_boundary(position: int) -> None:
    raw = {**raw_event(), "surface": "queue", "queue_position": position}
    if 0 <= position < 2**63:
        assert _parse_interaction_event(raw, 7).queue_position == position
    else:
        with pytest.raises(ValueError):
            _parse_interaction_event(raw, 7)


@pytest.mark.parametrize("legacy", [False, True])
def test_surface_round_trip_and_old_schema_migration(legacy: bool) -> None:
    class LegacyDatabase(Database):
        def _create_tables(self) -> None:
            with self.conn() as conn:
                conn.execute("""
                    CREATE TABLE interaction_events (
                        event_id TEXT PRIMARY KEY, client_session_id TEXT NOT NULL,
                        user_id INTEGER NOT NULL, story_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL, dashboard_version INTEGER NOT NULL,
                        position INTEGER NOT NULL, sort_mode TEXT NOT NULL,
                        age_filter TEXT NOT NULL, source_filter TEXT NOT NULL,
                        ranker_arm TEXT NOT NULL, occurred_at REAL NOT NULL,
                        duration_ms INTEGER, received_at REAL NOT NULL,
                        badges TEXT NOT NULL DEFAULT ''
                    ) STRICT
                """)
                conn.execute("""
                    INSERT INTO interaction_events VALUES (
                        'old', 'session', 7, 501, 'impression', 0, 3, 'explore',
                        '1w', 'mixed', 'tui_observed', 1700000000, NULL,
                        1700000001, 'discover'
                    )
                """)
                conn.commit()
            super()._create_tables()

    db = (LegacyDatabase if legacy else Database)(":memory:")
    try:
        db.upsert_story(Story(501, "Story", None, 1, 1, "Body"))
        event = _parse_interaction_event(
            {**raw_event(), "surface": "queue", "queue_position": 8}, 7
        )
        assert db.insert_interaction_events([event]) == (1, 0, 0)
        assert db.execute(
            "SELECT surface, queue_position, position, sort_mode FROM interaction_events WHERE event_id=?",
            (event.event_id,),
        ) == [("queue", 8, 3, "explore")]
        if legacy:
            assert db.execute(
                "SELECT badges, surface, queue_position FROM interaction_events WHERE event_id='old'"
            ) == [("discover", "", None)]
        # Re-running initialization is idempotent and preserves both old/new rows.
        Database._create_tables(db)
        assert db.execute("SELECT COUNT(*) FROM interaction_events") == [
            (2 if legacy else 1,)
        ]
    finally:
        db.close()
