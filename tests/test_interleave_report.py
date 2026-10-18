from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, strategies as st

from database import Database, InterleaveDeck, Story
from scripts import interleave_report as report
from scripts.interleave_report import (
    CreditedVote,
    Impression,
    Vote,
    credit_votes,
    load,
    main,
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


def _stored_trial(path: Path) -> None:
    db = Database(str(path))
    try:
        for sid, at in [(10, 100.0), (11, 150.0), (12, 200.0)]:
            db.upsert_story(
                Story(
                    id=sid, title="Test", url=None, score=1, time=1, text_content="Text"
                )
            )
            db.execute(
                "INSERT INTO feedback (user_id, story_id, action, updated_at) VALUES (?, ?, ?, ?)",
                (151, sid, "up", at),
            )
        for version, created in [(1, 99.0), (2, 100.0)]:
            db.insert_interleave_deck(
                InterleaveDeck(
                    151,
                    version,
                    created,
                    ("production", "joined_all"),
                    {
                        "1w": (
                            (10, "production"),
                            (11, "joined_all"),
                            (12, "production"),
                        )
                    },
                )
            )
    finally:
        db.close()


def test_trial_start_excludes_first_deployment_decks(tmp_path: Path) -> None:
    path = tmp_path / "trial.db"
    _stored_trial(path)
    db = Database(str(path), read_only=True)
    try:
        decks, votes, _ = load(db, 151, 100.0, 200.0)
    finally:
        db.close()
    assert [d.version for d in decks] == [2]
    credited, skipped = credit_votes(
        decks, votes, [Impression(10, 99.5, 1, "recommended", "1w")]
    )
    assert credited == []
    assert skipped["deck_not_interleaved"] == 1


def test_window_end_is_exclusive_and_start_inclusive(tmp_path: Path) -> None:
    path = tmp_path / "trial.db"
    _stored_trial(path)
    db = Database(str(path), read_only=True)
    try:
        _, votes, _ = load(db, 151, 100.0, 200.0)
    finally:
        db.close()
    assert [v.at for v in votes] == [100.0, 150.0]


def test_missing_decks_fail_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "trial.db"
    _stored_trial(path)
    db = Database(str(path), read_only=True)
    try:
        with pytest.raises(ValueError, match="no interleaved decks"):
            load(db, 999, 100.0, 200.0)
    finally:
        db.close()


def test_single_challenger_preserves_preregistered_alpha() -> None:
    summary = summarize([], ["production", "joined_all"], 0.0, 0.05)
    assert summary["per_test_alpha"] == 0.025
    assert summary["bonferroni_challengers"] == 2


def test_saved_counts_get_interval_and_fixed_down_tests() -> None:
    credited = []
    for arm, counts in [
        ("production", (7, 8, 8)),
        ("joined_all", (10, 8, 2)),
        ("joined_no_metadata", (8, 3, 7)),
    ]:
        for action, count in zip(("up", "neutral", "down"), counts, strict=True):
            credited.extend(
                CreditedVote(0, 1, "1w", arm, action, 100.0) for _ in range(count)
            )
    summary = summarize(
        credited, ["production", "joined_all", "joined_no_metadata"], 0.0, 0.05
    )
    for arm, p in [
        ("joined_all", 0.19058842408213117),
        ("joined_no_metadata", 0.35535839677044323),
    ]:
        result = cast(dict[str, object], summary[arm])
        assert isinstance(result, dict)
        assert result["rate_p"] == pytest.approx(p)
        interval = cast(tuple[float, float], result["up_rate_diff_ci"])
        assert interval[0] < 0 < interval[1]
        assert not result["down_safety_stop"]
    assert report.greater_rate_p(7, 18, 8, 23) == pytest.approx(0.3937108858, abs=0.002)


def test_downvote_harm_is_directional_and_safety_is_separate() -> None:
    harm_p = report.greater_rate_p(60, 100, 20, 100)
    better_p = report.greater_rate_p(20, 100, 60, 100)
    assert harm_p is not None and harm_p < 0.01
    assert better_p is not None and better_p > 0.99
    credited = [
        CreditedVote(i, 1, "1w", arm, "down" if i < downs else "up", 100.0)
        for arm, downs in [("production", 20), ("joined_all", 60)]
        for i in range(100)
    ]
    summary = summarize(credited, ["production", "joined_all"], 0.0, 0.05)
    result = cast(dict[str, object], summary["joined_all"])
    assert isinstance(result, dict)
    assert result["down_safety_stop"]
    assert result["down_rate_rejects"]


@given(st.integers(1, 500), st.integers(1, 500), st.data())
def test_rate_interval_is_bounded_and_reverses_with_arm_order(
    n1: int, n0: int, data: st.DataObject
) -> None:
    k1 = data.draw(st.integers(0, n1))
    k0 = data.draw(st.integers(0, n0))
    ci = report.rate_difference_ci(k1, n1, k0, n0)
    reverse = report.rate_difference_ci(k0, n0, k1, n1)
    assert ci is not None and reverse is not None
    assert -1 - 1e-12 <= ci[0] <= k1 / n1 - k0 / n0 <= ci[1] <= 1 + 1e-12
    assert ci == pytest.approx((-reverse[1], -reverse[0]))


def test_zero_counts_and_extreme_rates_do_not_create_false_harm() -> None:
    assert report.rate_difference_ci(0, 0, 1, 1) is None
    assert report.greater_rate_p(0, 0, 1, 1) is None
    assert report.greater_rate_p(0, 10, 0, 10) == 1.0
    assert report.greater_rate_p(10, 10, 10, 10) == 1.0
    ci = report.rate_difference_ci(0, 10, 10, 10)
    assert ci is not None and ci[1] < 0


def test_calibration_intervals_are_inclusive_and_do_not_double_count() -> None:
    votes = [Vote(i, "up", float(i)) for i in range(5)]
    assert [
        v.at
        for v in report.exclude_votes(
            votes, [report.ExcludedInterval(1, 2), report.ExcludedInterval(2, 3)]
        )
    ] == [0, 4]


def test_cli_echoes_exclusions_without_writing_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "trial.db"
    _stored_trial(path)
    before = path.read_bytes()
    main(
        [
            "--db",
            str(path),
            "--user-id",
            "151",
            "--since",
            "100",
            "--until",
            "200",
            "--exclude-updated-between",
            "150",
            "150",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["raw_votes"] == 2
    assert result["excluded_votes"] == 1
    assert result["votes"] == 1
    assert result["excluded_updated_intervals"] == [{"start": 150, "end": 150}]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "args",
    [
        ["--until", "nan"],
        ["--until", "99"],
        ["--alpha", "0"],
        ["--exclude-updated-between", "3", "2"],
        ["--arms", "production", "joined_all", "joined_all"],
    ],
)
def test_cli_rejects_invalid_analysis_settings(args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--db", "absent.db", "--user-id", "151", "--since", "100", *args])
    assert error.value.code == 2


def test_missing_deck_table_is_not_silently_swallowed(tmp_path: Path) -> None:
    path = tmp_path / "trial.db"
    _stored_trial(path)
    db = Database(str(path))
    try:
        db.execute("DROP TABLE interleave_decks")
        with pytest.raises(sqlite3.OperationalError, match="interleave_decks"):
            load(db, 151, 100.0, 200.0)
    finally:
        db.close()
