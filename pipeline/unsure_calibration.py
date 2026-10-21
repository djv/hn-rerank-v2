"""Platt-calibrated probabilities for Explore's Unsure entropy
(`ModelConfig.unsure_calibration_since`).

The classifier's raw P(up) can be overconfident at the top, so
the highest raw entropy marks mostly downvotes rather than mixed stories.
Each profile gets its own one-dimensional Platt fit of logit(P(up)) against
"was the vote up", from the raw probabilities its recorded decks held
(`deck_scores`, the earliest deck read with each story, within the last
``days``) and the votes last changed after that deck was built. P(down) and
P(neutral) are rescaled to keep their ratio. Only Unsure's entropy uses the
result: scores, ranking, the stored probabilities and `deck_scores` stay raw.
Calibrated Unsure still changes which stories get votes, so later fits see
its selection (monitor the slope). Feedback keeps only each story's final
vote: a changed vote counts once, at its change time; an undone one drops.

No calibrator (Unsure keeps raw entropy) below ``min_cards`` labelled cards,
with fewer than ``MIN_PER_OUTCOME`` ups or non-ups, or when the fit does not
converge or its slope falls outside (0, ``MAX_SLOPE``].
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from database import Action, Database, DeckScores

_EPS = 1e-6
MIN_PER_OUTCOME = 10
MAX_SLOPE = 5.0
# Decks are recorded at the end of a warm, so concurrent warms can commit
# slightly out of build-time order; re-read this much before the newest
# deck already seen (absorbing is idempotent).
_REREAD_SECONDS = 300.0


@dataclass(frozen=True)
class PlattCalibrator:
    """P(up)' = sigmoid(slope * logit(P(up)) + intercept), fit on *n* cards."""

    slope: float
    intercept: float
    n: int

    def calibrated(
        self, prob_down: float, prob_neutral: float, prob_up: float
    ) -> tuple[float, float, float] | None:
        """(down, neutral, up) with up recalibrated and down:neutral kept;
        None when the input is not finite or has no down or neutral mass
        to rescale."""
        if not all(math.isfinite(p) for p in (prob_down, prob_neutral, prob_up)):
            return None
        rest = prob_down + prob_neutral
        if rest <= 0.0:
            return None
        p = min(max(prob_up, _EPS), 1.0 - _EPS)
        z = self.slope * math.log(p / (1.0 - p)) + self.intercept
        up = 1.0 / (1.0 + math.exp(-min(max(z, -50.0), 50.0)))
        scale = (1.0 - up) / rest
        return prob_down * scale, prob_neutral * scale, up


@dataclass
class _Served:
    """Per user: each story's raw P(up) from the earliest deck read that
    held it, with that deck's build time; ``newest`` is the latest build
    time read."""

    since: float
    newest: float = 0.0
    first: dict[int, tuple[float, float]] = field(default_factory=dict)


_CACHE: dict[int, _Served] = {}
_LOCK = threading.Lock()


def absorb(
    first: dict[int, tuple[float, float]], decks: list[DeckScores]
) -> dict[int, tuple[float, float]]:
    """*first* updated with *decks*: the earliest finite P(up) per story.
    Reading a deck twice, or out of order, gives the same result."""
    out = dict(first)
    for deck in decks:
        for window in deck.windows.values():
            for card in window.cards:
                p = card.prob_up
                if p is None or not math.isfinite(p):
                    continue
                seen = out.get(card.story_id)
                if seen is None or deck.created_at < seen[1]:
                    out[card.story_id] = (float(p), deck.created_at)
    return out


def labelled_cards(
    first: dict[int, tuple[float, float]], votes: dict[int, tuple[Action, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """(raw P(up), up?) for every recorded story voted after its deck."""
    probs: list[float] = []
    ups: list[bool] = []
    for story_id, (action, voted_at) in votes.items():
        served = first.get(story_id)
        if served is None or voted_at <= served[1]:
            continue
        probs.append(served[0])
        ups.append(action == "up")
    return np.asarray(probs, dtype=np.float64), np.asarray(ups, dtype=bool)


def fit(probs: np.ndarray, ups: np.ndarray, min_cards: int) -> PlattCalibrator | None:
    """Platt fit of up-vs-rest on logit(P(up)), or None (module docstring)."""
    n_up = int(ups.sum())
    if len(probs) < min_cards or min(n_up, len(ups) - n_up) < MIN_PER_OUTCOME:
        return None
    p = np.clip(probs, _EPS, 1.0 - _EPS)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=200)
    model.fit(x, ups.astype(np.int64))
    slope = float(model.coef_[0, 0])
    iters = int(model.n_iter_[0])
    if iters >= model.max_iter or not 0.0 < slope <= MAX_SLOPE:
        logging.info(
            "unsure calibration rejected n=%d slope=%.3f iters=%d",
            len(probs),
            slope,
            iters,
        )
        return None
    return PlattCalibrator(
        slope=slope, intercept=float(model.intercept_[0]), n=len(probs)
    )


def calibrator_for(
    db: Database,
    user_id: int,
    since: float,
    min_cards: int,
    days: float,
    now: float | None = None,
) -> PlattCalibrator | None:
    """The user's calibrator from decks built at or after *since* (0 = off)
    and within the last *days*. After the first call it reads only recent
    decks; DB reads run outside the lock. A failure logs and returns None,
    leaving Unsure on raw entropy."""
    if since <= 0.0:
        return None
    try:
        start = max(since, (time.time() if now is None else now) - days * 86400.0)
        with _LOCK:
            state = _CACHE.get(user_id)
            newest = state.newest if state is not None and state.since == since else 0.0
        decks = db.get_deck_scores(user_id, since=max(start, newest - _REREAD_SECONDS))
        votes = db.get_feedback_times(user_id, since=start)
        with _LOCK:
            state = _CACHE.get(user_id)
            if state is None or state.since != since:
                state = _CACHE[user_id] = _Served(since=since)
            merged = absorb(state.first, decks)
            state.first = {s: v for s, v in merged.items() if v[1] >= start}
            state.newest = max([state.newest, *(d.created_at for d in decks)])
            first = state.first
        return fit(*labelled_cards(first, votes), min_cards)
    except Exception:
        logging.exception("unsure calibration failed user_id=%s", user_id)
        return None
