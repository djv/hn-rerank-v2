from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from database import Action, CardScore, Database, DeckScores, Story, WindowScores
from pipeline import Config, ranking, unsure_calibration
from pipeline.config import ModelConfig
from pipeline.ranking import RankedStory, RankTrace, rerank_candidates, unsure_entropy
from pipeline.unsure_calibration import (
    MAX_SLOPE,
    PlattCalibrator,
    absorb,
    calibrator_for,
    fit,
    labelled_cards,
)
from tests.test_joined_classifier import _NoiseEmbedder, _seed, _stories
from tests.test_pipeline import (
    _UNSURE_BASE,
    _make_explore_inputs,
    _served,
    _served_explore_ids,
)

prob = st.floats(0.0, 1.0, allow_nan=False)


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    unsure_calibration._CACHE.clear()
    yield
    unsure_calibration._CACHE.clear()


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


@given(st.floats(-5.0, 5.0), st.floats(-5.0, 5.0), prob, prob, prob)
def test_calibrated_keeps_a_distribution_and_down_neutral_ratio(
    slope: float, intercept: float, down: float, neutral: float, up: float
) -> None:
    total = down + neutral + up
    if total == 0:
        down = neutral = up = 1 / 3
    else:
        down, neutral, up = down / total, neutral / total, up / total
    out = PlattCalibrator(slope, intercept, 100).calibrated(down, neutral, up)
    if down + neutral <= 0:
        assert out is None
        return
    assert out is not None
    d, n, u = out
    assert min(d, n, u) >= 0
    assert d + n + u == pytest.approx(1.0)
    assert u == pytest.approx(
        1 / (1 + np.exp(-(slope * _logit(up) + intercept))), abs=1e-9
    )
    assert d * neutral == pytest.approx(n * down, abs=1e-9)


@pytest.mark.parametrize(
    "probs", [(float("nan"), 0.5, 0.5), (0.5, 0.5, float("inf")), (0.0, 0.0, 1.0)]
)
def test_unusable_probabilities_keep_raw_entropy(
    probs: tuple[float, float, float],
) -> None:
    cal = PlattCalibrator(1.0, 3.0, 100)
    assert cal.calibrated(*probs) is None
    ranked, _, _ = _make_explore_inputs()
    some = next(r for r in ranked if r.prob_up is not None)
    r = replace(some, prob_down=probs[0], prob_neutral=probs[1], prob_up=probs[2])
    assert unsure_entropy(r, cal) == ranking.get_entropy(r)


@settings(deadline=None)
@given(st.integers(0, 2**32 - 1), st.floats(0.3, 2.0), st.floats(-2.0, 1.0))
def test_fit_recovers_the_generating_platt_curve(
    seed: int, slope: float, intercept: float
) -> None:
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.02, 0.98, 4000)
    true_up = 1 / (1 + np.exp(-(slope * np.log(probs / (1 - probs)) + intercept)))
    ups = rng.random(4000) < true_up
    cal = fit(probs, ups, min_cards=100)
    assert cal is not None and cal.n == 4000
    assert cal.slope == pytest.approx(slope, abs=0.25)
    assert cal.intercept == pytest.approx(intercept, abs=0.25)


def test_fit_needs_enough_cards_and_both_outcomes() -> None:
    probs = np.linspace(0.1, 0.9, 50)
    # Ups thin out toward low P(up) but overlap: a positive, finite slope.
    ups = (np.arange(50) % 3 == 0) | (probs > 0.7)
    assert fit(probs, ups, min_cards=51) is None
    assert fit(probs, np.ones(50, dtype=bool), min_cards=2) is None
    few_ups = np.zeros(50, dtype=bool)
    few_ups[-9:] = True
    assert fit(probs, few_ups, min_cards=2) is None
    cal = fit(probs, ups, min_cards=50)
    assert cal is not None and 0 < cal.slope <= MAX_SLOPE


@settings(deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_fit_rejects_inverted_or_saturating_curves(seed: int) -> None:
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.02, 0.98, 400)
    # Separable: the fit either saturates past MAX_SLOPE or is refused.
    separable = fit(probs, probs > 0.5, min_cards=100)
    assert separable is None or 0 < separable.slope <= MAX_SLOPE
    # Inverted: the confident ups are mostly not ups.
    inverted = rng.random(400) < 1 - probs
    assert fit(probs, inverted, min_cards=100) is None


def test_labelled_cards_use_the_deck_and_only_later_votes() -> None:
    first = {1: (0.8, 10.0), 2: (0.3, 10.0), 3: (0.6, 20.0)}
    votes: dict[int, tuple[Action, float]] = {
        1: ("up", 11.0),
        2: ("down", 9.0),  # voted before the deck that held it
        3: ("neutral", 25.0),
        4: ("up", 30.0),  # never in a recorded deck
    }
    probs, ups = labelled_cards(first, votes)
    assert sorted(zip(probs.tolist(), ups.tolist())) == [(0.6, False), (0.8, True)]


def _card(story_id: int, prob_up: float | None) -> CardScore:
    return CardScore(
        story_id=story_id,
        view="recommended",
        position=0,
        score=0.0,
        svm_margin=None,
        prob_up=prob_up,
        prob_neutral=None if prob_up is None else (1 - prob_up) / 2,
        prob_down=None if prob_up is None else (1 - prob_up) / 2,
        window_rank=None,
        global_rank=None,
    )


def _deck(
    user_id: int, version: int, created_at: float, cards: list[CardScore]
) -> DeckScores:
    return DeckScores(
        user_id, version, created_at, {"1w": WindowScores(10, 10, {}, tuple(cards))}
    )


@given(st.permutations(range(4)))
def test_absorb_keeps_the_earliest_deck_in_any_read_order(order: list[int]) -> None:
    decks = [
        _deck(1, v, 10.0 * (v + 1), [_card(7, 0.1 * (v + 1)), _card(8, None)])
        for v in range(4)
    ]
    first: dict[int, tuple[float, float]] = {}
    for i in order:
        first = absorb(first, [decks[i]])
    assert first == {7: (pytest.approx(0.1), 10.0)}
    assert absorb(first, decks) == first


def test_get_feedback_times_filters_by_change_time() -> None:
    db = Database(":memory:")
    try:
        uid = db.create_user("times").id
        for i in (1, 2):
            db.upsert_story(
                Story(id=i, title=f"S{i}", url=None, score=1, time=1, text_content="x")
            )
        before = time.time() - 1
        db.upsert_feedback(uid, 1, "down")
        db.upsert_feedback(uid, 2, "up")
        times = db.get_feedback_times(uid)
        assert {k: v[0] for k, v in times.items()} == {1: "down", 2: "up"}
        assert all(v[1] >= before for v in times.values())
        assert db.get_feedback_times(uid, since=time.time() + 60) == {}
        # A changed vote moves its time; the same vote again does not.
        db.upsert_feedback(uid, 1, "up")
        db.upsert_feedback(uid, 2, "up")
        again = db.get_feedback_times(uid)
        assert again[1][0] == "up" and again[1][1] >= times[1][1]
        assert again[2] == times[2]
    finally:
        db.close()


def _seed_overconfident(db: Database, uid: int, created: float) -> None:
    """Forty stories at raw P(up) 0.9 (one in four voted up) and forty at
    0.1 (one in eight up), all in one deck built at *created* and voted
    afterwards."""
    for i in range(1, 81):
        db.upsert_story(
            Story(id=i, title=f"S{i}", url=None, score=1, time=1, text_content="x")
        )
    cards = [_card(i, 0.9) for i in range(1, 41)]
    cards += [_card(i, 0.1) for i in range(41, 81)]
    db.insert_deck_scores(_deck(uid, 5, created, [*cards, _card(200, None)]))
    for i in range(1, 41):
        db.upsert_feedback(uid, i, "up" if i % 4 == 0 else "down")
    for i in range(41, 81):
        db.upsert_feedback(uid, i, "up" if i % 8 == 0 else "neutral")


def test_calibrator_for_fits_reads_late_decks_and_is_off_at_zero() -> None:
    db = Database(":memory:")
    try:
        uid = db.create_user("calibration").id
        now = time.time()
        _seed_overconfident(db, uid, now - 600)
        assert calibrator_for(db, uid, 0.0, 10, 7.0) is None
        cal = calibrator_for(db, uid, 1.0, 10, 7.0)
        assert cal is not None and cal.n == 80
        out = cal.calibrated(0.05, 0.05, 0.9)
        assert out is not None and out[2] == pytest.approx(0.25, abs=0.05)

        # A lower version committed later, built earlier but within the
        # re-read slack, still counts and gives story 1 its earlier P(up).
        db.insert_deck_scores(_deck(uid, 4, now - 700, [_card(1, 0.5)]))
        assert calibrator_for(db, uid, 1.0, 10, 7.0) is not None
        assert unsure_calibration._CACHE[uid].first[1] == (0.5, now - 700)

        # A deck built after the votes labels nothing new.
        db.insert_deck_scores(_deck(uid, 6, time.time() + 60, [_card(200, 0.5)]))
        late = calibrator_for(db, uid, 1.0, 10, 7.0)
        assert late is not None and late.n == 80

        # Past the window, nothing counts and the cache is pruned.
        assert calibrator_for(db, uid, 1.0, 10, 7.0, now=now + 8 * 86400) is None
        assert unsure_calibration._CACHE[uid].first == {}
    finally:
        db.close()


def test_calibrator_for_logs_and_returns_none_on_db_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenDecks:
        def get_deck_scores(self, user_id: int, since: float = 0.0) -> list[DeckScores]:
            raise RuntimeError("boom")

    class BrokenVotes:
        def get_deck_scores(self, user_id: int, since: float = 0.0) -> list[DeckScores]:
            return []

        def get_feedback_times(
            self, user_id: int, since: float = 0.0
        ) -> dict[int, tuple[Action, float]]:
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        for broken in (BrokenDecks(), BrokenVotes()):
            assert calibrator_for(cast(Database, broken), 1, 1.0, 10, 7.0) is None
    assert caplog.text.count("unsure calibration failed") == 2


def test_unsure_picks_by_calibrated_entropy_and_cards_keep_raw_probs() -> None:
    ranked, explore, now = _make_explore_inputs()
    raw_unsure, _, _ = _served_explore_ids(
        ranking.assemble_window_deck(
            ranked, config=Config(count=40), now=now, explore=explore
        ),
        now,
    )
    # A fitted lift of P(up): the fixture's near-uniform stories become
    # confident ups, and its confident downs the mixed ones.
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.02, 0.98, 2000)
    lift = 1 / (1 + np.exp(-(np.log(probs / (1 - probs)) + 3.0)))
    cal = fit(probs, rng.random(2000) < lift, min_cards=100)
    assert cal is not None
    deck = ranking.assemble_window_deck(
        ranked,
        config=Config(count=40),
        now=now,
        explore=replace(explore, unsure_calibrator=cal),
    )
    cal_unsure, _, _ = _served_explore_ids(deck, now)
    assert cal_unsure and cal_unsure != raw_unsure

    by_id = {r.story.id: r for r in ranked}
    pool = [r for r in ranked if r.prob_up is not None and r.story.id >= _UNSURE_BASE]
    best = max(unsure_entropy(r, cal) for r in pool)
    assert any(unsure_entropy(by_id[i], cal) == best for i in cal_unsure)
    for r in _served(deck, "1w", now).explore:
        src = by_id[r.story.id]
        assert (r.prob_down, r.prob_neutral, r.prob_up) == (
            src.prob_down,
            src.prob_neutral,
            src.prob_up,
        )


@pytest.mark.parametrize("since", [0.0, 1.0])
def test_rerank_wires_the_calibrator_into_explore(since: float) -> None:
    db = Database(":memory:")
    try:
        user_id = _seed(db, [0, 1, 2] * 14)
        # A deck built before the seeded votes, with ups placed higher.
        rng = np.random.default_rng(1)
        cards = [
            _card(
                sid,
                float(rng.uniform(0.3, 0.95) if a == "up" else rng.uniform(0.05, 0.7)),
            )
            for sid, (a, _) in db.get_feedback_times(user_id).items()
        ]
        db.insert_deck_scores(_deck(user_id, 1, time.time() - 600, cards))
        expected = fit(
            *labelled_cards(
                absorb({}, db.get_deck_scores(user_id)),
                db.get_feedback_times(user_id),
            ),
            min_cards=20,
        )
        assert expected is not None

        config = Config(
            model=ModelConfig(
                min_up_for_svm=2,
                min_down_for_svm=2,
                classifier="joined_logistic",
                linear_blend_enabled=False,
                unsure_calibration_since=since,
                unsure_calibration_min_cards=20,
            )
        )
        now = int(time.time())
        candidates = [
            replace(s, time=now - 600 * i)
            for i, s in enumerate(_stories(rng, list(rng.integers(0, 3, 24)), 5000))
        ]
        for s in candidates:
            db.upsert_story(s)
        embs = rng.standard_normal((len(candidates), 384)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        trace = RankTrace()
        rerank_candidates(
            db, config, _NoiseEmbedder(), candidates, embs, user_id=user_id, trace=trace
        )
        assert "unsure_calibration" in trace.timings_ms
        want = "off" if since == 0.0 else f"n{expected.n}"
        assert trace.labels["unsure_calibration"] == want
    finally:
        db.close()


def test_unsure_entropy_falls_back_to_raw_without_probabilities() -> None:
    ranked, _, _ = _make_explore_inputs()
    cal = PlattCalibrator(1.0, -4.0, 100)
    no_probs: RankedStory = next(r for r in ranked if r.prob_up is None)
    assert unsure_entropy(no_probs, cal) == ranking.get_entropy(no_probs) == 0.0
    some = next(r for r in ranked if r.prob_up is not None)
    assert unsure_entropy(some, None) == ranking.get_entropy(some)
