"""A malformed integer event must not discard its valid batch neighbor."""

from __future__ import annotations

import pytest
from hypothesis import example, given, settings, strategies as st

from database import Database, Story
from pipeline import Config
from server import Handler, create_app


class InteractionBatchInvariantError(AssertionError):
    """The API rejected a whole batch instead of just its malformed event."""


_SIGNED_64_OVERFLOW_IDS = st.one_of(
    st.integers(min_value=2**63, max_value=2**128),
    st.integers(min_value=-(2**128), max_value=-(2**63) - 1),
)
_VALID_EVENT_ID = "11111111-1111-4111-8111-111111111111"
_BAD_EVENT_ID = "22222222-2222-4222-8222-222222222222"


@pytest.mark.parametrize(
    "field", ["story_id", "dashboard_version", "position", "occurred_at"]
)
@settings(deadline=None)
@given(overflow_story_id=_SIGNED_64_OVERFLOW_IDS, bad_first=st.booleans())
@example(overflow_story_id=2**63, bad_first=False)
def test_oversized_integer_does_not_discard_a_valid_neighbor(
    field: str, overflow_story_id: int, bad_first: bool
) -> None:
    """Storage and timestamp overflow must preserve the valid batch neighbor.

    Exercise parsing, batch rejection, and SQLite persistence together;
    expected counts come from the two input events, not server helpers.
    """
    db = Database(":memory:")

    class Runtime(Handler):
        pass

    Runtime.db = db
    Runtime.config = Config(db_path=":memory:")
    Runtime.reset_public_demo_limiter()
    try:
        story = Story(99991, "Stored neighbor", None, 1, 1, "Article text")
        db.upsert_story(story)
        user = db.create_user("interaction-integer-property")
        client = create_app(Runtime).test_client()
        client.set_cookie("hn_token", user.token)
        assert client.get("/api/user").status_code == 200
        valid: dict[str, object] = {
            "event_id": _VALID_EVENT_ID,
            "client_session_id": "33333333-3333-4333-8333-333333333333",
            "story_id": story.id,
            "event_type": "impression",
            "dashboard_version": 0,
            "position": 0,
            "sort_mode": "recommended",
            "window": "1d",
            "source_filter": "all",
            "ranker_arm": "production",
            "occurred_at": 1_700_000_000.0,
        }
        bad_value = (
            overflow_story_id * 2**2048 if field == "occurred_at" else overflow_story_id
        )
        bad = {**valid, "event_id": _BAD_EVENT_ID, field: bad_value}
        events = [bad, valid] if bad_first else [valid, bad]
        response = client.post("/api/interaction", json={"events": events})
        body = response.get_json()
        persisted = db.execute("SELECT event_id, story_id FROM interaction_events")
        expected_body = {
            "ok": True,
            "inserted": 1,
            "duplicates": 0,
            "rejected": 1,
        }
        if (
            response.status_code != 200
            or body != expected_body
            or persisted != [(_VALID_EVENT_ID, story.id)]
        ):
            raise InteractionBatchInvariantError(
                "Expected HTTP 200, inserted=1, rejected=1, and the valid event "
                f"persisted; got status={response.status_code}, body={body!r}, "
                f"persisted={persisted!r} for bad_id={overflow_story_id} "
                f"and bad_first={bad_first}"
            )
    finally:
        db.close()
