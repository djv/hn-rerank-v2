from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from clients.tui.src.hn_rerank.models import VIEWS, WINDOWS, View
from database import (
    CardScore,
    Database,
    DeckScores,
    ScoreQuantiles,
    Story,
    WindowScores,
)
from pipeline import Config
from pipeline.ranking import (
    RankedStory,
    VIEW_SIZE,
    assemble_window_deck,
    in_window,
    serve_window,
)

METRICS: tuple[str, ...] = ("score", "svm_margin", "prob_up", "prob_down")

finite = st.integers(-100000, 100000).map(lambda n: n / 1000)
optional = st.one_of(st.none(), finite)
quantiles = st.builds(ScoreQuantiles, finite, finite, finite, finite, finite, finite)


@st.composite
def _window_scores(draw: st.DrawFn) -> WindowScores:
    pool_size = draw(st.integers(0, 4))
    candidates_total = draw(st.integers(pool_size, pool_size + 4))
    quantile_map = draw(
        st.dictionaries(st.sampled_from(METRICS), quantiles, max_size=2)
    )
    max_cards = 0 if pool_size == 0 else 3
    n_cards = draw(st.integers(0, max_cards))
    views_drawn: list[View] = [draw(st.sampled_from(VIEWS)) for _ in range(n_cards)]
    counts: dict[str, int] = {}
    cards: list[CardScore] = []
    for view in views_drawn:
        # Sequential positions stay below the final per-view length.
        position = counts.get(view, 0)
        counts[view] = position + 1
        if pool_size == 0:
            window_rank: int | None = None
            global_rank: int | None = None
        else:
            window_rank = draw(st.one_of(st.none(), st.integers(0, pool_size - 1)))
            global_rank = draw(st.one_of(st.none(), st.integers(0, pool_size - 1)))
        cards.append(
            CardScore(
                story_id=draw(st.integers(1, 100)),
                view=view,
                position=position,
                score=draw(finite),
                svm_margin=draw(optional),
                prob_up=draw(optional),
                prob_neutral=draw(optional),
                prob_down=draw(optional),
                window_rank=window_rank,
                global_rank=global_rank,
            )
        )
    return WindowScores(pool_size, candidates_total, quantile_map, tuple(cards))


windows = st.dictionaries(st.sampled_from(WINDOWS), _window_scores(), max_size=2)


@settings(deadline=None)
@given(windows, windows)
def test_deck_scores_roundtrip_and_replace(
    original: dict[str, WindowScores],
    replacement: dict[str, WindowScores],
) -> None:
    db = Database(":memory:")
    try:
        first = DeckScores(1, 7, 10.0, original)
        second = DeckScores(1, 8, 11.0, replacement)
        db.insert_deck_scores(first)
        db.insert_deck_scores(second)
        assert db.get_deck_scores(1) == [first, second]
        updated = replace(first, created_at=12.0, windows=replacement)
        db.insert_deck_scores(updated)
        assert db.get_deck_scores(1) == [updated, second]
        assert db.get_deck_scores(1, since=11.5) == [updated]
        assert db.get_deck_scores(2) == []
    finally:
        db.close()


@settings(deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, 1000), st.integers(0, 10 * 86400)),
        min_size=VIEW_SIZE + 1,
        max_size=40,
    )
)
def test_pool_statistics_and_views(rows: list[tuple[int, int]]) -> None:
    now = 2_000_000
    ranked = [
        RankedStory(
            Story(i + 1, "t", None, 100, now - age, ""),
            float(score),
            "",
            svm_margin=float(score) - 500,
            prob_up=score / 1000,
            prob_down=1 - score / 1000,
        )
        for i, (score, age) in enumerate(rows)
    ]
    deck = assemble_window_deck(ranked, config=Config(), now=now, margin=1)
    global_order = sorted(ranked, key=lambda r: r.score, reverse=True)
    global_rank = {r.story.id: i for i, r in enumerate(global_order)}
    for window, views in deck.windows.items():
        expected = [r for r in global_order if in_window(r.story.time, window, now)]
        pool = views.pool
        assert pool is not None
        assert pool.pool_size == len(expected)
        assert pool.candidates_total == len(rows)
        assert {sid: rank.global_rank for sid, rank in pool.ranks.items()} == {
            r.story.id: global_rank[r.story.id] for r in expected
        }
        assert {sid: rank.window_rank for sid, rank in pool.ranks.items()} == {
            r.story.id: i for i, r in enumerate(expected)
        }
        for metric in ("score", "svm_margin", "prob_up", "prob_down"):
            if expected:
                actual = pool.quantiles[metric]
                np.testing.assert_allclose(
                    [
                        actual.p50,
                        actual.p75,
                        actual.p90,
                        actual.p95,
                        actual.p99,
                        actual.max,
                    ],
                    np.percentile(
                        [getattr(r, metric) for r in expected],
                        [50, 75, 90, 95, 99, 100],
                    ),
                )
            else:
                assert metric not in pool.quantiles
        served = serve_window(views, window, now + 60)
        bare = serve_window(replace(views, pool=None), window, now + 60)
        assert replace(served, pool=None) == bare
        assert deck.capped(2).window(window).pool == pool


def test_all_none_metrics_are_omitted() -> None:
    ranked = [RankedStory(Story(1, "t", None, 1, 1000, ""), 0.5, "")]
    pool = assemble_window_deck(ranked, config=Config(), now=1000).window("1d").pool
    assert pool is not None
    assert set(pool.quantiles) == {"score"}


@pytest.mark.parametrize("precomputed", [False, True])
def test_svm_margin_matches_decision_and_original_score(
    monkeypatch: pytest.MonkeyPatch,
    precomputed: bool,
) -> None:
    from numpy.typing import NDArray
    from sklearn.svm import SVC
    from pipeline import Embedder, ModelConfig
    from pipeline import ranking

    class SyntheticEmbedder(Embedder):
        def __init__(self) -> None:
            pass

        def encode(
            self, texts: list[str], batch_size: int | None = None
        ) -> NDArray[np.float32]:
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            for i, text in enumerate(texts):
                vectors[i, 0 if "up" in text else 1] = 1
            return vectors

    decisions: list[NDArray[np.float64]] = []
    original_decision = SVC.decision_function

    def capture(self: SVC, features: NDArray[np.float32]) -> NDArray[np.float64]:
        result = original_decision(self, features)
        if result.ndim == 1:
            up_col = result if list(self.classes_)[-1] == 2 else -result
        else:
            up_col = result[:, list(self.classes_).index(2)]
        decisions.append(np.asarray(up_col, dtype=np.float64).copy())
        return result

    monkeypatch.setattr(SVC, "decision_function", capture)
    config = Config(
        model=ModelConfig(
            min_up_for_svm=2,
            min_down_for_svm=2,
            tier2_blend_window=1,
            tier3_blend_window=1,
            linear_blend_enabled=False,
            svm_precomputed_enabled=precomputed,
            svm_precomputed_chunk_size=2,
            svm_gamma=0.03,
        )
    )
    db = Database(":memory:")
    ranking._MODEL_CACHE.clear()
    try:
        user = db.create_user("margin")
        for i in range(12):
            text = "up" if i % 2 else "down"
            story = Story(100 + i, text, None, 10, 1000, text)
            db.upsert_story(story)
            db.upsert_feedback(user.id, story.id, "up" if i % 2 else "down")
        candidates = [
            Story(i + 1, text, None, i + 1, 1000, text)
            for i, text in enumerate(
                [
                    "up candidate",
                    "down candidate",
                    "up other",
                    "down other",
                    "up extra",
                ]
            )
        ]
        embedder = SyntheticEmbedder()
        embeddings = embedder.encode([s.text_content for s in candidates])
        outputs = []
        for _ in range(2):
            start = len(decisions)
            outputs.append(
                ranking._score_and_rank(
                    candidates, embeddings, db, config, embedder, user_id=user.id
                )
            )
            full = np.concatenate(decisions[start:], axis=0)
            assert len(full) == len(candidates)
            expected_scores = ranking._minmax01(full).astype(np.float32)
            for r in outputs[-1]:
                index = r.story.id - 1
                assert r.svm_margin == float(full[index])
                assert r.score == float(expected_scores[index])
        assert tuple(r.svm_margin for r in outputs[0]) == tuple(
            r.svm_margin for r in outputs[1]
        )
        assert tuple(r.score for r in outputs[0]) == tuple(r.score for r in outputs[1])
    finally:
        db.close()
        ranking._MODEL_CACHE.clear()


@pytest.mark.parametrize("trained,fail", [(True, False), (True, True), (False, False)])
def test_warm_logging_without_socket(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    trained: bool,
    fail: bool,
) -> None:
    import pipeline
    from pipeline import Embedder
    from server import Handler

    class DummyEmbedder(Embedder):
        def __init__(self) -> None:
            pass

    class TestHandler(Handler):
        pass

    db = Database(":memory:")
    try:
        user = db.create_user("memory-warm")
        if trained:
            db.upsert_story(Story(999, "previous vote", None, 1, 1, ""))
            db.upsert_feedback(user.id, 999, "up")
        TestHandler.db = db
        TestHandler.config = Config(db_path=":memory:")
        TestHandler.embedder = DummyEmbedder()
        TestHandler._decks = {}
        TestHandler._dashboard_versions = {}
        card = RankedStory(
            Story(1, "card", None, 1, 1000, ""),
            0.5,
            "",
            prob_up=0.7 if trained else None,
        )
        deck = assemble_window_deck([card], config=Config(), now=1000)
        from pipeline.ranking import WindowDeck

        deck = WindowDeck(
            {w: replace(v, explore=v.recommended) for w, v in deck.windows.items()}
        )
        monkeypatch.setattr(
            pipeline, "fast_rerank_for_user", lambda *args, **kwargs: deck
        )
        monkeypatch.setattr(
            TestHandler, "_trigger_warm", classmethod(lambda *args: None)
        )

        def broken(scores: DeckScores) -> None:
            raise RuntimeError("write failed")

        if fail:
            monkeypatch.setattr(db, "insert_deck_scores", broken)
        TestHandler._run_warm_attempt(user, TestHandler._dashboard_version(user.id))
        assert TestHandler._decks[user.id].deck == deck
        if trained:
            assert TestHandler._deck_for_user(user).deck == deck
        records = db.get_deck_scores(user.id)
        assert len(records) == int(trained and not fail)
        if records:
            assert sum(len(w.cards) for w in records[0].windows.values()) == sum(
                len(v.recommended) + len(v.popular) + len(v.explore)
                for v in deck.windows.values()
            )
        if fail:
            assert "RuntimeError: write failed" in caplog.text
            assert any(r.exc_info for r in caplog.records)
    finally:
        db.close()


def test_canonical_card_keeps_scored_pool_rank() -> None:
    from server import Handler

    class TestHandler(Handler):
        pass

    db = Database(":memory:")
    try:
        TestHandler.db = db
        original = RankedStory(
            Story(1, "original", None, 1, 1000, ""), 0.7, "", prob_up=0.8
        )
        deck = assemble_window_deck([original], config=Config(), now=1000)
        canonical = replace(original.story, id=99)
        mapped = deck.map_views(
            lambda _w, _v, rows: (replace(r, story=canonical) for r in rows)
        )
        TestHandler._record_deck_scores(1, 4, mapped)
        (record,) = db.get_deck_scores(1)
        for scores in record.windows.values():
            for card in scores.cards:
                assert card.story_id == 99
                assert card.score == original.score
                assert card.window_rank == card.global_rank == 0
    finally:
        db.close()


def test_pool_rank_fallback_and_missing() -> None:
    from pipeline.ranking import PoolRank, WindowDeck, WindowPool, WindowViews
    from server import Handler

    class TestHandler(Handler):
        pass

    db = Database(":memory:")
    try:
        TestHandler.db = db
        pool = WindowPool(
            pool_size=2,
            candidates_total=2,
            quantiles={},
            ranks={1: PoolRank(0, 0), 2: PoolRank(1, 1)},
        )
        views = WindowViews(
            recommended=(
                RankedStory(
                    Story(1, "one", None, 1, 1000, ""),
                    0.7,
                    "",
                    prob_up=0.8,
                    pool_rank=None,
                ),
                RankedStory(
                    Story(9999, "missing", None, 1, 1000, ""),
                    0.6,
                    "",
                    prob_up=0.7,
                    pool_rank=None,
                ),
            ),
            popular=(),
            explore=(),
            pool=pool,
        )
        TestHandler._record_deck_scores(1, 5, WindowDeck({"1d": views}))
        (record,) = db.get_deck_scores(1)
        by_id = {c.story_id: c for c in record.windows["1d"].cards}
        assert by_id[1].window_rank == 0
        assert by_id[1].global_rank == 0
        assert by_id[9999].window_rank is None
        assert by_id[9999].global_rank is None
    finally:
        db.close()


@pytest.mark.parametrize("bad", ["none", "raising"])
def test_skipped_window_is_isolated(bad: str, caplog: pytest.LogCaptureFixture) -> None:
    from pipeline.ranking import PoolRank, WindowDeck, WindowPool, WindowViews
    from server import Handler

    class TestHandler(Handler):
        pass

    db = Database(":memory:")
    try:
        TestHandler.db = db
        good_pool = WindowPool(
            pool_size=1,
            candidates_total=1,
            quantiles={},
            ranks={1: PoolRank(0, 0)},
        )
        good_views = WindowViews(
            recommended=(
                RankedStory(
                    Story(1, "one", None, 1, 1000, ""),
                    0.7,
                    "",
                    prob_up=0.8,
                ),
            ),
            popular=(),
            explore=(),
            pool=good_pool,
        )
        if bad == "none":
            bad_views = replace(good_views, pool=None)
        else:

            class ExplodingDict(dict):  # type: ignore[type-arg]
                def get(self, *args: object, **kwargs: object) -> object:
                    raise RuntimeError("boom")

            exploding_pool = WindowPool(
                pool_size=1,
                candidates_total=1,
                quantiles={},
                ranks=ExplodingDict({1: PoolRank(0, 0)}),  # type: ignore[arg-type]
            )
            bad_views = WindowViews(
                recommended=(
                    RankedStory(
                        Story(1, "one", None, 1, 1000, ""),
                        0.7,
                        "",
                        prob_up=0.8,
                        pool_rank=None,
                    ),
                ),
                popular=(),
                explore=(),
                pool=exploding_pool,
            )
        deck = WindowDeck({"1d": good_views, "1w": bad_views})
        with caplog.at_level("WARNING"):
            TestHandler._record_deck_scores(7, 9, deck)
        records = db.get_deck_scores(7)
        assert len(records) == 1
        assert set(records[0].windows) == {"1d"}
        assert "user_id=7" in caplog.text
        assert "version=9" in caplog.text
        assert "1w" in caplog.text
        assert any(r.levelname == "WARNING" and r.exc_info for r in caplog.records)
    finally:
        db.close()


def test_nonfinite_roundtrip_as_none() -> None:
    db = Database(":memory:")
    try:
        deck = DeckScores(
            1,
            1,
            10.0,
            {
                "1d": WindowScores(
                    2,
                    2,
                    {
                        "score": ScoreQuantiles(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                    },
                    (
                        CardScore(
                            1,
                            "recommended",
                            0,
                            0.5,
                            float("nan"),
                            float("inf"),
                            float("-inf"),
                            None,
                            0,
                            0,
                        ),
                        CardScore(
                            2,
                            "popular",
                            0,
                            0.6,
                            None,
                            float("nan"),
                            float("inf"),
                            float("-inf"),
                            1,
                            1,
                        ),
                    ),
                )
            },
        )
        db.insert_deck_scores(deck)
        (record,) = db.get_deck_scores(1)
        cards = {c.story_id: c for c in record.windows["1d"].cards}
        assert cards[1].svm_margin is None
        assert cards[1].prob_up is None
        assert cards[1].prob_down is None
        assert cards[1].prob_neutral is None
        assert cards[2].prob_up is None
        assert cards[2].prob_neutral is None
        assert cards[2].prob_down is None
        assert cards[1].score == 0.5
    finally:
        db.close()
