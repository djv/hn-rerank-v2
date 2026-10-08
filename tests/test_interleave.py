from __future__ import annotations

from collections import Counter

import numpy as np
from hypothesis import given, strategies as st

from pipeline.interleave import team_draft


@st.composite
def rankings(draw: st.DrawFn) -> list[list[int]]:
    """Two to four rankings that mostly share their items, the way arms
    rank one candidate pool, plus a few items only some arms rank."""
    shared = draw(st.lists(st.integers(0, 60), min_size=0, max_size=30, unique=True))
    arms = draw(st.integers(2, 4))
    out = []
    for _ in range(arms):
        extra = draw(st.lists(st.integers(61, 80), max_size=4, unique=True))
        items = shared + extra
        out.append(draw(st.permutations(items)))
    return out


@given(rankings(), st.integers(0, 40), st.integers(0, 2**32 - 1))
def test_team_draft_picks_distinct_items_in_each_arms_order(
    ranked: list[list[int]], limit: int, seed: int
) -> None:
    picks = team_draft(ranked, limit, np.random.default_rng(seed))
    items = [p.item for p in picks]

    assert len(items) == len(set(items))
    universe = set().union(*map(set, ranked))
    assert len(items) == min(limit, len(universe))
    for arm, ranking in enumerate(ranked):
        own = [p.item for p in picks if p.arm == arm]
        assert all(item in ranking for item in own)
        # Each arm adds its own items best first.
        assert own == sorted(own, key=ranking.index)
        # When an arm picks, everything it ranks higher is already taken.
        for i, pick in enumerate(picks):
            if pick.arm == arm:
                better = ranking[: ranking.index(pick.item)]
                assert set(better) <= set(items[:i])


@given(st.integers(2, 4), st.integers(1, 30), st.integers(0, 2**32 - 1))
def test_team_draft_rounds_balance_arms_on_a_shared_pool(
    arms: int, size: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    pool = list(range(size * arms))
    ranked = [list(rng.permutation(pool)) for _ in range(arms)]
    picks = team_draft(ranked, len(pool), rng)

    for end in range(arms, len(picks) + 1, arms):
        counts = Counter(p.arm for p in picks[:end])
        assert set(counts.values()) == {end // arms}


def test_team_draft_identical_arms_split_credit_evenly() -> None:
    ranking = list(range(30))
    first = Counter()
    for seed in range(400):
        picks = team_draft([ranking] * 3, 12, np.random.default_rng(seed))
        assert [p.item for p in picks] == ranking[:12]
        first[picks[0].arm] += 1
    assert set(first) == {0, 1, 2}
    assert min(first.values()) > 100
