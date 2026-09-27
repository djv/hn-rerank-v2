from __future__ import annotations

import sqlite3
from pathlib import Path

from database import Database, Story
from pipeline.ranking import source_category_onehot
from scripts.relabel_legacy_sources import new_label, plan


def test_new_label_maps_legacy_names_to_current_feed_names() -> None:
    assert new_label("tildes", "https://example.com/a") == "rss_tildes_net"
    assert new_label("reddit_programming", None) == "rss_reddit_programming"
    assert new_label("rss", "https://www.latent.space/p/ainews") == "rss_latent_space"
    reddit_url = "https://www.reddit.com/r/MachineLearning/comments/1/x/"
    assert new_label("rss_reddit_com", reddit_url) == "rss_reddit_machinelearning"
    assert new_label("rss_reddit_com", "https://example.com/x") is None
    assert new_label("hn", "https://example.com/x") is None
    assert new_label("rss_tildes_net", "https://example.com/x") is None


def test_every_mapped_label_gets_a_source_category() -> None:
    samples = [
        ("digg", "https://x.com/a"),
        ("reddit_compsci", None),
        ("rss", "https://simonwillison.net/2026/May/20/x/"),
        ("rss_reddit_com", "https://www.reddit.com/r/programming/comments/1/x/"),
    ]
    for source, url in samples:
        label = new_label(source, url)
        assert label is not None
        assert source_category_onehot(label).sum() == 1
    # Legacy reddit_* rows were counted as generic RSS; now they are Reddit.
    assert source_category_onehot("reddit_compsci")[3] == 1
    assert source_category_onehot("rss_reddit_compsci")[2] == 1
    # Most other legacy labels had no category at all.
    assert source_category_onehot("digg").sum() == 0


def test_plan_only_touches_legacy_rows(tmp_path: Path) -> None:
    path = tmp_path / "r.db"
    db = Database(str(path))
    rows = [
        (1, "tildes", "https://a.org/1"),
        (2, "reddit_rust", "https://www.reddit.com/r/rust/comments/1/x/"),
        (3, "hn", "https://b.org/3"),
        (4, "rss_tildes_net", "https://c.org/4"),
        (5, "rss", "https://www.construction-physics.com/p/x"),
    ]
    for sid, source, url in rows:
        db.upsert_story(
            Story(
                id=sid,
                title="t",
                url=url,
                score=1,
                time=1,
                text_content="t",
                source=source,
            )
        )
    db.close()
    conn = sqlite3.connect(path)
    try:
        assert sorted(plan(conn)) == [
            (1, "tildes", "rss_tildes_net"),
            (2, "reddit_rust", "rss_reddit_rust"),
            (5, "rss", "rss_construction-physics_com"),
        ]
    finally:
        conn.close()
