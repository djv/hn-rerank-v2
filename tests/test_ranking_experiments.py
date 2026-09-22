from __future__ import annotations

from dataclasses import replace

from database import Story
from pipeline.feedback import deduplicate_feedback


def item(sid: int) -> Story:
    return Story(sid, "Newsletter", f"https://example.org/{sid}", 0, 0, "legacy")


def test_training_dedup_latest_vote_and_missing_urls() -> None:
    a = item(1)
    b = replace(item(2), url=a.url + "?utm_source=rss" if a.url else None)
    c = replace(item(3), url=None)
    d = replace(item(4), url=None)
    rows, labels, times = deduplicate_feedback(
        [a, b, c, d], [2, 0, 1, 2], [1.0, 2.0, 3.0, 4.0]
    )
    assert [s.id for s in rows] == [2, 3, 4]
    assert labels == [0, 1, 2]
    assert times == [2.0, 3.0, 4.0]


def test_training_dedup_tie_break_is_order_independent() -> None:
    a = item(1)
    b = replace(item(2), url=a.url)
    first = deduplicate_feedback([a, b], [0, 2], [1.0, 1.0])
    second = deduplicate_feedback([b, a], [2, 0], [1.0, 1.0])
    assert first == second == ([b], [2], [1.0])
