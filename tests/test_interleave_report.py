from __future__ import annotations

from database import InterleaveDeck
from scripts.interleave_report import (
    Impression,
    Vote,
    credit_votes,
    summarize,
)


def _deck(
    version: int, picks: dict[str, tuple[tuple[int, str], ...]]
) -> InterleaveDeck:
    return InterleaveDeck(151, version, 0.0, ("production", "joined_all"), picks)


def test_votes_credit_the_arm_of_the_last_recommended_impression() -> None:
    decks = [
        _deck(1, {"1w": ((10, "joined_all"), (11, "production"))}),
        _deck(2, {"1w": ((11, "joined_all"),), "1d": ((12, "production"),)}),
    ]
    impressions = [
        Impression(10, 100.0, 1, "recommended", "1w"),
        Impression(11, 100.0, 1, "recommended", "1w"),
        # Seen again in a later version where another arm drafted it.
        Impression(11, 200.0, 2, "recommended", "1w"),
        Impression(12, 300.0, 2, "popular", "1d"),
        Impression(13, 300.0, 9, "recommended", "1w"),
        Impression(14, 300.0, 2, "recommended", "12h"),
        # After its vote: does not count.
        Impression(15, 999.0, 2, "recommended", "1w"),
    ]
    votes = [
        Vote(10, "up", 150.0),
        Vote(11, "down", 250.0),
        Vote(12, "up", 350.0),
        Vote(13, "up", 350.0),
        Vote(14, "neutral", 300.5),
        Vote(15, "up", 400.0),
    ]
    credited, skipped = credit_votes(decks, votes, impressions)
    assert [(c.story_id, c.arm, c.version, c.action) for c in credited] == [
        (10, "joined_all", 1, "up"),
        (11, "joined_all", 2, "down"),
    ]
    assert skipped == {
        "view_popular": 1,
        "deck_not_interleaved": 1,
        "story_not_in_window": 1,
        "no_impression": 1,
    }
    summary = summarize(credited, ["production", "joined_all"], 0.0, 0.05)
    assert summary["credited_by_arm"] == {
        "production": {},
        "joined_all": {"up": 1, "down": 1},
    }
    assert summary["decks_with_credit"] == 2
