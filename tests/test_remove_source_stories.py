from __future__ import annotations

from pathlib import Path

from database import Database, Story
from scripts import remove_source_stories


def _story(sid: int, source: str) -> Story:
    return Story(sid, f"Story {sid}", None, 1, 0, "", source=source)


def test_remove_source_keeps_voted_stories_and_their_votes(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "remove.db"))
    for sid in (1, 2, 3):
        db.upsert_story(_story(sid, "rss_gone"))
    db.upsert_story(_story(4, "hn"))
    user = db.create_user("token")
    db.upsert_feedback(user.id, 2, "up")
    db.upsert_feedback(user.id, 4, "down")

    remove_source_stories.main(["--source", "rss_gone", "--yes"], db=db)

    with db.conn() as conn:
        left = {r[0] for r in conn.execute("SELECT id FROM stories")}
        votes = set(conn.execute("SELECT story_id, action FROM feedback"))
    assert left == {2, 4}
    assert votes == {(2, "up"), (4, "down")}
