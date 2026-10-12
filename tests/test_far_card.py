from __future__ import annotations

import json
from typing import cast

import numpy as np
from hypothesis import given, settings, strategies as st

from clients.tui.src.hn_rerank.models import Window
from database import CardScore, Database, DeckScores, Story, WindowScores
import pytest

from pipeline import (
    Config,
    DISCOVER_MAX_SIM_UP,
    FAR_ARM,
    RankedStory,
    WindowDeck,
    WindowViews,
    draft_recommended,
    place_far_card,
)
from pipeline.ranking import (
    VIEW_SIZE,
    ExploreContext,
    assemble_window_deck,
    serve_window,
)

NOW = 2_000_000_000.0
HORIZON = 3600.0
HEAD = VIEW_SIZE - 1


def _story(sid: int, *, score: int = 10, age_s: float = 3600) -> Story:
    return Story(
        id=sid,
        title=f"Story {sid}",
        url=None,
        score=score,
        time=int(NOW - age_s),
        text_content=f"text {sid}",
        source="hn",
        comment_count=1,
    )


def _card(
    sid: int,
    *,
    score: float,
    closest_up: float | None,
    is_discover: bool = False,
    arm: str = "",
    age_s: float = 3600,
) -> RankedStory:
    return RankedStory(
        story=_story(sid, age_s=age_s),
        score=score,
        best_match_title="",
        is_discover=is_discover,
        arm=arm,
        closest_up=closest_up,
    )


def _deck(
    recommended: tuple[RankedStory, ...],
    explore: tuple[RankedStory, ...] = (),
    window: Window = "1w",
) -> WindowDeck:
    return WindowDeck({window: WindowViews(recommended=recommended, explore=explore)})


def _head_cards(
    n: int = HEAD, *, start: int = 1, near: float = 0.9
) -> list[RankedStory]:
    return [_card(start + i, score=float(100 - i), closest_up=near) for i in range(n)]


def test_pipeline_exports_far_arm() -> None:
    assert FAR_ARM == "far"
    assert DISCOVER_MAX_SIM_UP == 0.60


def test_place_far_card_picks_best_far_from_tail_and_discover() -> None:
    head = _head_cards()
    tail = (
        _card(12, score=50.0, closest_up=0.9),
        _card(13, score=40.0, closest_up=0.5),
        _card(14, score=45.0, closest_up=0.5),
    )
    explore = (
        _card(20, score=60.0, closest_up=0.5, is_discover=True),
        _card(21, score=30.0, closest_up=0.5, is_discover=True),
    )
    out = place_far_card(
        _deck(tuple(head + list(tail)), explore), now=NOW, horizon=HORIZON
    )
    rec = out.window("1w").recommended
    assert [r.story.id for r in rec[:HEAD]] == [r.story.id for r in head]
    assert rec[HEAD].story.id == 20
    assert rec[HEAD].arm == FAR_ARM
    assert rec[HEAD].is_discover is True
    assert [r.story.id for r in rec[HEAD + 1 :]] == [12, 13, 14]
    assert [r.story.id for r in out.window("1w").explore] == [21]


def test_place_far_card_takes_best_tail_when_no_discover_eligible() -> None:
    head = _head_cards()
    tail = (
        _card(12, score=40.0, closest_up=0.5),
        _card(13, score=55.0, closest_up=0.5),
    )
    out = place_far_card(_deck(tuple(head + list(tail))), now=NOW, horizon=HORIZON)
    rec = out.window("1w").recommended
    assert rec[HEAD].story.id == 13
    assert rec[HEAD].arm == FAR_ARM
    assert [r.story.id for r in rec[HEAD + 1 :]] == [12]


def test_place_far_card_tie_breaks_by_lower_id() -> None:
    head = _head_cards()
    tail = (
        _card(30, score=50.0, closest_up=0.1),
        _card(20, score=50.0, closest_up=0.1),
    )
    out = place_far_card(_deck(tuple(head + list(tail))), now=NOW, horizon=HORIZON)
    assert out.window("1w").recommended[HEAD].story.id == 20


def test_place_far_card_head_untouched_no_duplicates() -> None:
    head = _head_cards()
    tail = (
        _card(12, score=50.0, closest_up=0.9),
        _card(13, score=40.0, closest_up=0.2),
    )
    explore = (_card(20, score=70.0, closest_up=0.2, is_discover=True),)
    deck = _deck(tuple(head + list(tail)), explore)
    out = place_far_card(deck, now=NOW, horizon=HORIZON)
    for window, views in out.windows.items():
        for view in ("recommended", "explore"):
            ids = [r.story.id for r in getattr(views, view)]
            assert len(ids) == len(set(ids)), (window, view)
    rec = out.window("1w").recommended
    assert [r.story.id for r in rec[:HEAD]] == [r.story.id for r in head]
    assert all(r.arm != FAR_ARM for r in rec[:HEAD])
    far_id = rec[HEAD].story.id
    assert far_id not in {r.story.id for r in out.window("1w").explore}
    assert out.window("1w").popular == deck.window("1w").popular
    assert out.window("1w").pool == deck.window("1w").pool


def test_place_far_card_ignores_head_story_ids_in_explore() -> None:
    head = _head_cards()
    dup = _card(head[0].story.id, score=999.0, closest_up=0.1, is_discover=True)
    tail = (_card(13, score=40.0, closest_up=0.2),)
    out = place_far_card(
        _deck(tuple(head + list(tail)), (dup,)), now=NOW, horizon=HORIZON
    )
    assert out.window("1w").recommended[HEAD].story.id == 13


def test_place_far_card_no_eligible_leaves_deck_unchanged() -> None:
    head = _head_cards()
    tail = (
        _card(12, score=50.0, closest_up=0.9),
        _card(13, score=40.0, closest_up=None),
    )
    explore = (_card(20, score=60.0, closest_up=0.9, is_discover=True),)
    deck = _deck(tuple(head + list(tail)), explore)
    out = place_far_card(deck, now=NOW, horizon=HORIZON)
    assert out.window("1w").recommended == deck.window("1w").recommended
    assert out.window("1w").explore == deck.window("1w").explore


def test_place_far_card_short_recommended_still_places() -> None:
    head = _head_cards(n=3)
    explore = (_card(20, score=60.0, closest_up=0.2, is_discover=True),)
    out = place_far_card(_deck(tuple(head), explore), now=NOW, horizon=HORIZON)
    rec = out.window("1w").recommended
    assert [r.story.id for r in rec[:3]] == [r.story.id for r in head]
    assert rec[3].story.id == 20
    assert rec[3].arm == FAR_ARM
    assert out.window("1w").explore == ()


def test_serve_window_far_card_served_last() -> None:
    head = _head_cards()
    far = _card(20, score=10.0, closest_up=0.2, is_discover=True, arm=FAR_ARM)
    tail = tuple(_card(30 + i, score=float(9 - i), closest_up=0.9) for i in range(5))
    served = serve_window(
        _deck(tuple(head + [far] + list(tail))).window("1w"), "1w", NOW
    )
    ids = [r.story.id for r in served.recommended]
    assert len(ids) == VIEW_SIZE
    assert ids[:HEAD] == [r.story.id for r in head]
    assert ids[-1] == 20


def test_serve_window_head_aged_out_far_still_last() -> None:
    head = _head_cards()
    old_head = _card(head[0].story.id, score=100.0, closest_up=0.9, age_s=8 * 86400)
    rest_head = head[1:]
    far = _card(20, score=10.0, closest_up=0.2, is_discover=True, arm=FAR_ARM)
    filler = _card(99, score=1.0, closest_up=0.9)
    deck = _deck(tuple([old_head] + rest_head + [far, filler]))
    served = serve_window(deck.window("1w"), "1w", NOW)
    ids = [r.story.id for r in served.recommended]
    assert ids[-1] == 20
    assert old_head.story.id not in ids
    assert ids[:HEAD] == [r.story.id for r in rest_head] + [99]


def test_serve_window_far_aged_out_serves_plain_first_view() -> None:
    head = _head_cards()
    far = _card(
        20, score=10.0, closest_up=0.2, is_discover=True, arm=FAR_ARM, age_s=8 * 86400
    )
    tail = tuple(_card(30 + i, score=float(9 - i), closest_up=0.9) for i in range(5))
    deck = _deck(tuple(head + [far] + list(tail)))
    served = serve_window(deck.window("1w"), "1w", NOW)
    ids = [r.story.id for r in served.recommended]
    assert 20 not in ids
    assert ids == [r.story.id for r in head] + [30]


def test_serve_window_no_far_unchanged() -> None:
    rec = tuple(_card(i + 1, score=float(100 - i), closest_up=0.9) for i in range(20))
    served = serve_window(_deck(rec).window("1w"), "1w", NOW)
    assert [r.story.id for r in served.recommended] == list(range(1, VIEW_SIZE + 1))


@st.composite
def _random_deck(draw: st.DrawFn) -> WindowDeck:
    n_rec = draw(st.integers(0, 30))
    n_exp = draw(st.integers(0, 12))
    rec_ids = draw(
        st.lists(st.integers(1, 200), min_size=n_rec, max_size=n_rec, unique=True)
    )
    exp_ids = draw(
        st.lists(st.integers(1, 200), min_size=n_exp, max_size=n_exp, unique=True)
    )
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    rec: list[RankedStory] = []
    for sid in rec_ids:
        aged = bool(rng.integers(0, 4) == 0)
        rec.append(
            _card(
                sid,
                score=float(rng.integers(0, 100)),
                closest_up=(
                    None
                    if rng.integers(0, 4) == 0
                    else float(
                        rng.random() * 0.59
                        if rng.integers(0, 2)
                        else 0.6 + rng.random() * 0.4
                    )
                ),
                is_discover=bool(rng.integers(0, 2)),
                age_s=8 * 86400 if aged else 3600,
            )
        )
    # Highest score first, like a real Recommended view.
    rec.sort(key=lambda r: r.score, reverse=True)
    exp: list[RankedStory] = []
    for sid in exp_ids:
        aged = bool(rng.integers(0, 4) == 0)
        exp.append(
            _card(
                sid,
                score=float(rng.integers(0, 100)),
                closest_up=(None if rng.integers(0, 4) == 0 else float(rng.random())),
                is_discover=bool(rng.integers(0, 2)),
                age_s=8 * 86400 if aged else 3600,
            )
        )
    return _deck(tuple(rec), tuple(exp))


@settings(deadline=None)
@given(_random_deck())
def test_serve_window_far_invariants(deck: WindowDeck) -> None:
    placed = place_far_card(deck, now=NOW, horizon=HORIZON)
    served = serve_window(placed.window("1w"), "1w", NOW)
    ids = [r.story.id for r in served.recommended]
    assert len(ids) == len(set(ids))
    assert len(ids) <= VIEW_SIZE
    fars = [r for r in served.recommended if r.arm == FAR_ARM]
    assert len(fars) <= 1
    if fars:
        assert ids[-1] == fars[0].story.id
    current = [
        r for r in placed.window("1w").recommended if r.story.time > NOW - 7 * 86400
    ]
    current_non_far = [r for r in current if r.arm != FAR_ARM]
    if fars:
        assert [r.story.id for r in served.recommended if r.arm != FAR_ARM] == [
            r.story.id for r in current_non_far[:HEAD]
        ]
        assert fars[0].story.id not in {r.story.id for r in served.explore}
    else:
        assert ids == [r.story.id for r in current[:VIEW_SIZE]]


def test_assemble_window_deck_sets_closest_up() -> None:
    now = int(NOW)
    ranked = [
        RankedStory(
            story=Story(
                id=i,
                title=f"t{i}",
                url=None,
                score=10,
                time=now - 3600,
                text_content=f"text {i}",
            ),
            score=float(10 - i),
            best_match_title="",
        )
        for i in range(5)
    ]
    closest = np.array([0.1, 0.7, 0.3, 0.95, 0.0], dtype=np.float32)
    explore = ExploreContext(
        cand_closest_up=closest,
        cand_interest=np.zeros(5, dtype=np.int64),
        interest_sizes=np.zeros(0, dtype=np.int64),
        row_of={r.story.id: i for i, r in enumerate(ranked)},
    )
    deck = assemble_window_deck(ranked, config=Config(), now=NOW, explore=explore)
    for views in deck.windows.values():
        for view in ("recommended", "popular", "explore"):
            for r in getattr(views, view):
                assert r.closest_up == float(closest[explore.row_of[r.story.id]])


def test_assemble_window_deck_leaves_closest_up_none_without_explore() -> None:
    ranked = [
        RankedStory(story=_story(i), score=float(i), best_match_title="")
        for i in range(3)
    ]
    deck = assemble_window_deck(ranked, config=Config(), now=NOW)
    for views in deck.windows.values():
        for view in ("recommended", "popular", "explore"):
            for r in getattr(views, view):
                assert r.closest_up is None


def test_record_interleave_far_only_writes_no_row() -> None:
    from server import Handler

    class TestHandler(Handler):
        pass

    db = Database(":memory:")
    try:
        TestHandler.db = db
        head = _head_cards(n=3)
        far = _card(20, score=1.0, closest_up=0.1, is_discover=True, arm=FAR_ARM)
        TestHandler._record_interleave(7, 9, _deck(tuple(head + [far])))
        assert db.get_interleave_decks(7) == []
    finally:
        db.close()


def test_record_interleave_stores_far_card_with_far_arm() -> None:
    from pipeline.interleave import PRODUCTION_ARM
    from server import Handler

    class TestHandler(Handler):
        pass

    db = Database(":memory:")
    try:
        TestHandler.db = db
        rec = tuple(
            RankedStory(
                story=_story(sid),
                score=float(10 - i),
                best_match_title="",
                arm=PRODUCTION_ARM if i % 2 == 0 else "joined_all",
            )
            for i, sid in enumerate([1, 2, 3])
        ) + (_card(20, score=0.5, closest_up=0.1, is_discover=True, arm=FAR_ARM),)
        TestHandler._record_interleave(7, 9, _deck(rec))
        (row,) = db.get_interleave_decks(7)
        assert row.windows["1w"][-1] == (20, "far")
        assert PRODUCTION_ARM in row.arms
    finally:
        db.close()


def test_interleave_report_far_vote_credited_to_no_arm() -> None:
    from database import InterleaveDeck
    from scripts.interleave_report import Impression, Vote, credit_votes, summarize

    decks = [
        InterleaveDeck(
            151,
            1,
            0.0,
            ("production", "joined_all", "far"),
            {"1w": ((10, "production"), (11, "far"))},
        )
    ]
    impressions = [
        Impression(10, 100.0, 1, "recommended", "1w"),
        Impression(11, 100.0, 1, "recommended", "1w"),
    ]
    votes = [Vote(10, "up", 150.0), Vote(11, "up", 150.0)]
    credited, skipped = credit_votes(decks, votes, impressions)
    assert [(c.story_id, c.arm) for c in credited] == [(10, "production")]
    assert skipped["far_card"] == 1
    summary = summarize(credited, ["production", "joined_all"], 0.0, 0.05)
    by_arm = cast("dict[str, dict[str, int]]", summary["credited_by_arm"])
    assert set(by_arm) == {"production", "joined_all"}
    assert by_arm["production"] == {"up": 1}
    assert by_arm["joined_all"] == {}
    assert summary["decks_with_credit"] == 1


def test_deck_scores_arm_roundtrip() -> None:
    db = Database(":memory:")
    try:
        deck = DeckScores(
            1,
            7,
            10.0,
            {
                "1d": WindowScores(
                    1,
                    1,
                    {},
                    (
                        CardScore(
                            5, "recommended", 0, 0.5, None, 0.7, None, None, 0, 0, "far"
                        ),
                    ),
                )
            },
        )
        db.insert_deck_scores(deck)
        assert db.get_deck_scores(1) == [deck]
    finally:
        db.close()


def test_deck_scores_old_row_without_arm_loads_empty() -> None:
    db = Database(":memory:")
    try:
        payload = {
            "1d": {
                "pool_size": 1,
                "candidates_total": 1,
                "quantiles": {},
                "cards": [
                    {
                        "story_id": 5,
                        "view": "recommended",
                        "position": 0,
                        "score": 0.5,
                        "svm_margin": None,
                        "prob_up": 0.7,
                        "prob_neutral": None,
                        "prob_down": None,
                        "window_rank": 0,
                        "global_rank": 0,
                    }
                ],
            }
        }
        db.execute(
            "INSERT OR REPLACE INTO deck_scores "
            "(user_id, version, created_at, windows_json) VALUES (?, ?, ?, ?)",
            (1, 7, 10.0, json.dumps(payload)),
        )
        (row,) = db.get_deck_scores(1)
        assert row.windows["1d"].cards[0].arm == ""
    finally:
        db.close()


def _deck_scores_payload_with_arm(arm: object) -> dict[str, object]:
    return {
        "1d": {
            "pool_size": 1,
            "candidates_total": 1,
            "quantiles": {},
            "cards": [
                {
                    "story_id": 5,
                    "view": "recommended",
                    "position": 0,
                    "score": 0.5,
                    "svm_margin": None,
                    "prob_up": 0.7,
                    "prob_neutral": None,
                    "prob_down": None,
                    "window_rank": 0,
                    "global_rank": 0,
                    "arm": arm,
                }
            ],
        }
    }


def test_deck_scores_arm_null_raises() -> None:
    db = Database(":memory:")
    try:
        db.execute(
            "INSERT OR REPLACE INTO deck_scores "
            "(user_id, version, created_at, windows_json) VALUES (?, ?, ?, ?)",
            (1, 7, 10.0, json.dumps(_deck_scores_payload_with_arm(None))),
        )
        with pytest.raises(ValueError):
            db.get_deck_scores(1)
    finally:
        db.close()


def test_deck_scores_arm_int_raises() -> None:
    db = Database(":memory:")
    try:
        db.execute(
            "INSERT OR REPLACE INTO deck_scores "
            "(user_id, version, created_at, windows_json) VALUES (?, ?, ?, ?)",
            (1, 7, 10.0, json.dumps(_deck_scores_payload_with_arm(3))),
        )
        with pytest.raises(ValueError):
            db.get_deck_scores(1)
    finally:
        db.close()


def test_place_far_card_prefers_staying_within_horizon() -> None:
    head = _head_cards()
    old_age = 12 * 3600 - 600
    leaving = _card(50, score=90.0, closest_up=0.1, age_s=old_age)
    staying = _card(51, score=80.0, closest_up=0.1, age_s=3600)
    deck = _deck(tuple(head + [leaving, staying]), window="12h")
    out = place_far_card(deck, now=NOW, horizon=HORIZON)
    rec = out.window("12h").recommended
    assert rec[HEAD].story.id == 51
    assert rec[HEAD].arm == FAR_ARM


def test_place_far_card_falls_back_to_best_when_none_stays() -> None:
    head = _head_cards()
    old_age_best = 12 * 3600 - 600
    old_age_next = 12 * 3600 - 700
    best = _card(50, score=90.0, closest_up=0.1, age_s=old_age_best)
    nxt = _card(51, score=80.0, closest_up=0.1, age_s=old_age_next)
    deck = _deck(tuple(head + [best, nxt]), window="12h")
    out = place_far_card(deck, now=NOW, horizon=HORIZON)
    rec = out.window("12h").recommended
    assert rec[HEAD].story.id == 50
    assert rec[HEAD].arm == FAR_ARM


def test_no_upvotes_leaves_closest_up_none_and_places_nothing() -> None:
    ranked = [
        RankedStory(story=_story(i), score=float(10 - i), best_match_title="")
        for i in range(5)
    ]
    closest = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    explore = ExploreContext(
        cand_closest_up=closest,
        cand_interest=np.zeros(5, dtype=np.int64),
        interest_sizes=np.zeros(0, dtype=np.int64),
        row_of={r.story.id: i for i, r in enumerate(ranked)},
        has_upvotes=False,
    )
    deck = assemble_window_deck(ranked, config=Config(), now=NOW, explore=explore)
    for views in deck.windows.values():
        for view in ("recommended", "popular", "explore"):
            for r in getattr(views, view):
                assert r.closest_up is None
    placed = place_far_card(deck, now=NOW, horizon=HORIZON)
    for views in placed.windows.values():
        for r in views.recommended:
            assert r.arm != FAR_ARM


def test_interleaved_draft_then_far_card_keeps_head_and_records_far() -> None:
    from pipeline.interleave import PRODUCTION_ARM
    from server import Handler

    class TestHandler(Handler):
        pass

    rec = tuple(_card(i, score=float(100 - i), closest_up=0.9) for i in range(1, 25))
    explore = (
        _card(100, score=90.0, closest_up=0.1, is_discover=True),
        _card(101, score=10.0, closest_up=0.1, is_discover=True),
    )
    deck = _deck(rec, explore, window="1w")
    arm_rankings = {"joined_all": [r.story.id for r in reversed(rec)]}
    rng = np.random.default_rng(0)
    drafted = draft_recommended(deck, arm_rankings, rng)
    placed = place_far_card(drafted, now=NOW, horizon=HORIZON)
    drafted_head = [
        (r.story.id, r.arm) for r in drafted.window("1w").recommended[:HEAD]
    ]
    placed_head = [(r.story.id, r.arm) for r in placed.window("1w").recommended[:HEAD]]
    assert placed_head == drafted_head
    far = placed.window("1w").recommended[HEAD]
    assert far.story.id == 100
    assert far.arm == FAR_ARM
    db = Database(":memory:")
    try:
        TestHandler.db = db
        TestHandler._record_interleave(7, 9, placed)
        (row,) = db.get_interleave_decks(7)
        picks = row.windows["1w"]
        assert (100, FAR_ARM) in picks
        assert all(arm == FAR_ARM or sid != 100 for sid, arm in picks)
        assert PRODUCTION_ARM in {arm for _, arm in picks}
    finally:
        db.close()


def test_same_story_head_in_one_window_far_in_another() -> None:
    shared = _card(500, score=100.0, closest_up=0.1, age_s=3600)
    head_1d = [shared] + [
        _card(i, score=float(99 - i), closest_up=0.9, age_s=3600) for i in range(10)
    ]
    tail_1d = [
        _card(600, score=50.0, closest_up=0.1, age_s=3600),
        _card(601, score=40.0, closest_up=0.1, age_s=3600),
    ]
    head_1w = [
        _card(101 + i, score=float(200 - i), closest_up=0.9, age_s=2 * 86400)
        for i in range(HEAD)
    ]
    tail_1w = [
        shared,
        _card(600, score=50.0, closest_up=0.1, age_s=3600),
        _card(601, score=40.0, closest_up=0.1, age_s=3600),
    ]
    deck = WindowDeck(
        {
            "1d": WindowViews(recommended=tuple(head_1d + tail_1d)),
            "1w": WindowViews(recommended=tuple(head_1w + tail_1w)),
        }
    )
    out = place_far_card(deck, now=NOW, horizon=HORIZON)
    rec_1d = out.window("1d").recommended
    rec_1w = out.window("1w").recommended
    assert rec_1d[HEAD].story.id == 600
    assert rec_1d[HEAD].arm == FAR_ARM
    assert rec_1w[HEAD].story.id == 500
    assert rec_1w[HEAD].arm == FAR_ARM
    head_ids_1d = {r.story.id for r in rec_1d[:HEAD]}
    assert 500 in head_ids_1d
    head_card_1d = next(r for r in rec_1d[:HEAD] if r.story.id == 500)
    assert head_card_1d.arm != FAR_ARM
