"""Team-draft interleaving of several rankings into one served list
(ROADMAP B2; `interleave_user_ids`).

Each round visits the arms in a fresh random order and every arm adds its
best story not yet in the list, so any prefix that ends a round holds the
same number of stories from each arm. A vote on a story credits the arm
that added it; with equally good arms each arm expects the same credit.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import numpy as np

from .config import Config
from .joined_classifier import JoinedFeatures

T = TypeVar("T", bound=Hashable)

PRODUCTION_ARM = "production"
# Challenger arm -> the joined classifier's inputs (shortlist #2 and #4).
CHALLENGER_FEATURES: dict[str, JoinedFeatures] = {
    "joined_all": "all",
    "joined_no_metadata": "no_metadata",
}


def challenger_configs(config: Config, user_id: int) -> list[tuple[str, Config]]:
    """The interleaving arms ranking *user_id*'s deck besides production,
    each with its own config; none unless the user is in
    ``config.interleave_user_ids``."""
    if user_id not in config.interleave_user_ids:
        return []
    return [
        (
            arm,
            replace(
                config,
                model=replace(
                    config.model,
                    classifier="joined_logistic",
                    joined_features=CHALLENGER_FEATURES[arm],
                    linear_blend_enabled=False,
                ),
            ),
        )
        for arm in config.interleave_arms
    ]


@dataclass(frozen=True)
class Pick(Generic[T]):
    item: T
    arm: int


def team_draft(
    rankings: Sequence[Sequence[T]], limit: int, rng: np.random.Generator
) -> list[Pick[T]]:
    """Up to *limit* distinct items from *rankings* (best first), each
    tagged with the index of the ranking that contributed it. Arms that
    run out of new items stop picking; the others continue."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    picks: list[Pick[T]] = []
    taken: set[T] = set()
    heads = [0] * len(rankings)
    while len(picks) < limit:
        progressed = False
        for arm in rng.permutation(len(rankings)):
            if len(picks) >= limit:
                break
            ranking = rankings[arm]
            while heads[arm] < len(ranking) and ranking[heads[arm]] in taken:
                heads[arm] += 1
            if heads[arm] < len(ranking):
                item = ranking[heads[arm]]
                taken.add(item)
                picks.append(Pick(item, int(arm)))
                progressed = True
        if not progressed:
            break
    return picks
