from __future__ import annotations

from collections import Counter

import numpy as np
from hypothesis import given, strategies as st

from pipeline.interleave import challenger_configs, team_draft


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


_NOW = 1_800_000_000.0


@given(st.integers(0, 2**32 - 1), st.integers(4, 60))
def test_interleaved_recommended_takes_turns_and_keeps_arm_orders(
    seed: int, size: int
) -> None:
    from dataclasses import replace

    from database import Story
    from pipeline import Config, RankedStory
    from pipeline.interleave import PRODUCTION_ARM
    from pipeline.ranking import SELECT_MARGIN, VIEW_SIZE, assemble_window_deck

    rng = np.random.default_rng(seed)
    pool = [
        RankedStory(
            Story(i, f"s{i}", None, 10, int(_NOW) - int(rng.integers(0, 40_000)), ""),
            float(size - i),
            "",
        )
        for i in range(size)
    ]
    production = [r.story.id for r in pool]
    rankings = {
        "joined_all": production[::-1],
        "joined_no_metadata": [int(i) for i in rng.permutation(production)],
    }
    deck = assemble_window_deck(
        pool, config=Config(), now=_NOW, arm_rankings=rankings, rng=rng
    )
    plain = assemble_window_deck(pool, config=Config(), now=_NOW)

    for window, views in deck.windows.items():
        served = views.recommended
        expected = plain.window(window).recommended
        assert len(served) == min(len(expected), VIEW_SIZE * SELECT_MARGIN)
        assert len({r.story.id for r in served}) == len(served)
        orders = {PRODUCTION_ARM: production, **rankings}
        for arm, order in orders.items():
            own = [r.story.id for r in served if r.arm == arm]
            assert own == sorted(own, key=order.index)
        for end in range(3, len(served) + 1, 3):
            counts = Counter(r.arm for r in served[:end])
            assert set(counts.values()) == {end // 3}
        # Cards keep production's score, probabilities and badges.
        by_id = {r.story.id: r for r in pool}
        for r in served:
            assert replace(r, arm="") == by_id[r.story.id]
        assert all(r.arm == "" for r in views.popular + views.explore)
        assert all(r.arm == "" for r in plain.window(window).recommended)


def test_only_listed_users_get_challenger_arms() -> None:
    from pipeline import Config

    config = Config(interleave_user_ids=(151,))
    assert challenger_configs(config, 7) == []
    arms = challenger_configs(config, 151)
    assert [arm for arm, _ in arms] == ["joined_all", "joined_no_metadata"]
    for (_, arm_config), features in zip(arms, ["all", "no_metadata"], strict=True):
        assert arm_config.model.classifier == "joined_logistic"
        assert arm_config.model.joined_features == features
        assert not arm_config.model.linear_blend_enabled
        assert arm_config.interleave_user_ids == config.interleave_user_ids
    assert challenger_configs(Config(), 151) == []


def test_interleave_decks_round_trip_and_replace_by_version() -> None:
    from database import Database, InterleaveDeck

    db = Database(":memory:")
    try:
        first = InterleaveDeck(
            user_id=151,
            version=7,
            created_at=100.0,
            arms=("production", "joined_all"),
            windows={"1w": ((5, "joined_all"), (3, "production")), "1d": ()},
        )
        db.insert_interleave_deck(first)
        db.insert_interleave_deck(
            InterleaveDeck(151, 8, 200.0, ("production",), {"1w": ((9, "production"),)})
        )
        db.insert_interleave_deck(InterleaveDeck(2, 7, 300.0, ("production",), {}))
        assert db.get_interleave_decks(151) == [
            first,
            InterleaveDeck(
                151, 8, 200.0, ("production",), {"1w": ((9, "production"),)}
            ),
        ]
        assert [d.version for d in db.get_interleave_decks(151, since=150.0)] == [8]
        rebuilt = InterleaveDeck(151, 7, 400.0, ("production",), {})
        db.insert_interleave_deck(rebuilt)
        assert db.get_interleave_decks(151)[0] == rebuilt
    finally:
        db.close()
