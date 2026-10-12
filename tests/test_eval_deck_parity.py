"""Regression coverage for evaluator deck parity after Interest was added."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pytest

from database import Story
from pipeline import Config, finalize_ranked_deck
from pipeline.ranking import (
    SELECT_MARGIN,
    VIEW_SIZE,
    RankedStory,
    RankScoreContext,
    WindowDeck,
    assemble_ranked_deck,
)
from scripts.eval_ranker_variants import (
    FoldData,
    _fold_database,
    _FrozenEmbedder,
    _recommended,
)


class EvaluationDeckMismatch(AssertionError):
    """Evaluation deck assembly diverges from the production contract."""


@dataclass(frozen=True)
class _DeckScenario:
    fold: FoldData
    config: Config
    scores: NDArray[np.float64]


@pytest.fixture
def deck_scenario(monkeypatch: pytest.MonkeyPatch) -> _DeckScenario:
    """An HN Interest crosspost can suppress an RSS Recommended card.

    Distinct URLs prevent URL-group isolation from removing the crosspost.
    Its embedding matches the RSS card, but both are below the cosine
    threshold against the training upvote, so feedback does not suppress
    them. Popular and Recommended reserves are full without the crosspost.
    All embeddings and SQLite rows are fold-local; no model or real DB opens.
    """
    now = 1_800_000_000
    monkeypatch.setattr("time.time", lambda: float(now))
    basis = np.eye(384, dtype=np.float32)
    candidates: list[Story] = []
    vectors: list[NDArray[np.float32]] = []
    scores: list[float] = []
    for i in range(32):
        candidates.append(
            Story(
                id=i + 1,
                title=f"RSS item {i}",
                url=f"https://rss.example/{i}",
                score=0,
                time=now - 3600,
                text_content="content",
                source="rss_test",
            )
        )
        vectors.append(basis[i + 2].copy() if i else basis[0] * 0.8 + basis[1] * 0.6)
        scores.append(1.0 - i / 100)
    for i in range(32):
        candidates.append(
            Story(
                id=100 + i,
                title=f"HN item {i}",
                url=f"https://hn.example/{i}",
                score=100,
                time=now - 3600,
                text_content="content",
            )
        )
        vectors.append(basis[100 + i])
        scores.append(0.4 - i / 100)
    candidates.append(
        Story(
            id=999,
            title="HN crosspost",
            url="https://hn.example/crosspost",
            score=1,
            time=now - 3600,
            text_content="content",
        )
    )
    vectors.append(vectors[0].copy())
    scores.append(0.6)
    embeddings = np.asarray(vectors, dtype=np.float32)
    upvote = Story(
        id=2000,
        title="Liked separate story",
        url="https://liked.example",
        score=10,
        time=now - 7200,
        text_content="liked",
    )
    up_embeddings = basis[0:1]
    zeros = np.zeros(len(candidates), dtype=np.float32)
    fold = FoldData(
        candidates=candidates,
        cand_emb=embeddings,
        train_stories=[upvote],
        test_stories=[],
        test_actions=np.array([], dtype=int),
        train_vote_times=np.array([now - 7200], dtype=np.float64),
        x_train_base=np.empty((0, 0), dtype=np.float32),
        x_cand_base=np.empty((0, 0), dtype=np.float32),
        y_train=np.array([2], dtype=int),
        tier2_scores=zeros,
        train_emb=up_embeddings,
        similarities={
            2: (embeddings @ up_embeddings.T).ravel(),
            0: zeros,
            1: zeros,
        },
    )
    return _DeckScenario(fold, Config(), np.asarray(scores, dtype=np.float64))


def _production_deck(scenario: _DeckScenario) -> WindowDeck:
    """Use the context the live ranker supplies, without fitting a model."""
    fold, config = scenario.fold, scenario.config
    ranked = [
        RankedStory(fold.candidates[i], float(scenario.scores[i]), "")
        for i in np.argsort(-scenario.scores, kind="stable")
    ]
    assert fold.train_emb is not None
    context = RankScoreContext(
        cand_closest_up=fold.similarities[2],
        cand_closest_down=fold.similarities[0],
        cand_closest_neutral=fold.similarities[1],
        fb_up_embeddings=fold.train_emb[fold.y_train == 2],
    )
    with _fold_database(fold, config, None) as db:
        embedder = _FrozenEmbedder(config.embedding_model_version)
        deck = assemble_ranked_deck(
            ranked,
            fold.candidates,
            fold.cand_emb,
            db,
            config,
            embedder,
            user_id=1,
            score_context=context,
        )
        return finalize_ranked_deck(
            deck, fold.candidates, fold.cand_emb, db, config, embedder, 1
        )


def test_production_interest_crosspost_suppresses_recommended_duplicate(
    deck_scenario: _DeckScenario,
) -> None:
    """Control: this topology exercises an actual Interest/dedup effect."""
    views = _production_deck(deck_scenario).window("1w")
    assert any(r.is_interest and r.story.id == 999 for r in views.explore)
    assert not any(r.story.id == 1 for r in views.recommended)
    assert len(views.recommended) == VIEW_SIZE * SELECT_MARGIN - 1


def test_evaluation_preserves_interest_with_cached_similarities(
    deck_scenario: _DeckScenario,
) -> None:
    deck = _recommended(
        deck_scenario.scores,
        deck_scenario.fold,
        deck_scenario.config,
        None,
        None,
    )
    if not any(r.is_interest and r.story.id == 999 for r in deck.window("1w").explore):
        raise EvaluationDeckMismatch("The evaluator omitted the Interest crosspost")


def test_evaluation_recommended_matches_production_global_dedup(
    deck_scenario: _DeckScenario,
) -> None:
    deck = _recommended(
        deck_scenario.scores,
        deck_scenario.fold,
        deck_scenario.config,
        None,
        None,
    )
    production = _production_deck(deck_scenario)
    expected = {
        name: tuple(r.story.id for r in views.recommended)
        for name, views in production.windows.items()
    }
    observed = {
        name: tuple(r.story.id for r in views.recommended)
        for name, views in deck.windows.items()
    }
    if observed != expected:
        raise EvaluationDeckMismatch(
            f"Recommended order differs: expected {expected!r}, observed {observed!r}"
        )
