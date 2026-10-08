"""Warm starts saved across restarts (pipeline/warm_store.py) and the first
regen waiting for the startup warms."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from cachetools import LRUCache

import server
from database import Story
from pipeline import joined_classifier, linear_blend, warm_store
from pipeline.joined_classifier import JoinedLogistic

EMB_DIM = 6
WORDS = {
    0: "rust kernel compiler systems database",
    1: "weather museum report city council",
    2: "garden recipe travel bread hiking",
}


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Path]:
    warm_store.configure(tmp_path)
    try:
        yield tmp_path
    finally:
        warm_store.configure(None)


def _stories(rng: np.random.Generator, labels: list[int]) -> list[Story]:
    out = []
    for i, label in enumerate(labels):
        words = WORDS[label].split()
        out.append(
            Story(
                id=1000 + i,
                title=" ".join(rng.choice(words, size=3)),
                url=f"https://site{label}.example/{i}",
                score=1,
                time=1,
                text_content=" ".join(rng.choice(words + ["misc", "news"], size=8)),
            )
        )
    return out


def test_store_round_trip(store: Path) -> None:
    arrays: warm_store.Arrays = {"coef": np.arange(6.0).reshape(2, 3)}
    warm_store.save("fit_1", arrays)
    # Queued fits are readable before the background write.
    loaded = warm_store.load("fit_1")
    assert loaded is not None
    np.testing.assert_array_equal(loaded["coef"], arrays["coef"])

    warm_store.flush()
    assert (store / "fit_1.npz").exists()
    assert not list(store.glob(".*tmp*"))
    warm_store.configure(store)  # a restart: nothing queued, read the file
    loaded = warm_store.load("fit_1")
    assert loaded is not None
    np.testing.assert_array_equal(loaded["coef"], arrays["coef"])

    (store / "bad.npz").write_bytes(b"not a zip")
    assert warm_store.load("bad") is None
    assert warm_store.load("missing") is None


def test_store_is_off_until_configured(tmp_path: Path) -> None:
    warm_store.configure(None)
    warm_store.save("fit_1", {"coef": np.zeros(3)})
    warm_store.flush()
    assert warm_store.load("fit_1") is None
    assert not list(tmp_path.iterdir())


def test_saved_joined_fit_warm_starts_like_the_in_memory_one(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(5)
    labels = [int(v) for v in rng.permutation(np.repeat([0, 1, 2], 30))]
    stories = _stories(rng, labels)
    x = rng.normal(size=(len(labels), EMB_DIM + 3))
    weights = np.ones(len(labels))

    def model() -> JoinedLogistic:
        return JoinedLogistic(
            c=4.0,
            features="all",
            embedding_dim=EMB_DIM,
            embedding_weight=16.0,
            numeric_scale=0.158,
            word_scale=1.0,
        )

    # The previous fit lacks the newest vote, as after a swipe.
    previous = model().fit(
        x[:-1], labels[:-1], sample_weight=weights[:-1], stories=stories[:-1]
    )
    joined_classifier.remember(7, "all", previous)
    warm_store.flush()
    monkeypatch.setattr(joined_classifier, "_LATEST", LRUCache(maxsize=64))

    restored = joined_classifier.latest(7, "all")

    assert restored is not None and restored is not previous
    from_memory = model().fit(
        x, labels, sample_weight=weights, stories=stories, warm=previous
    )
    from_disk = model().fit(
        x, labels, sample_weight=weights, stories=stories, warm=restored
    )
    cold = model().fit(x, labels, sample_weight=weights, stories=stories)
    assert from_disk.estimator.n_iter_[0] <= cold.estimator.n_iter_[0]
    np.testing.assert_allclose(
        from_disk.predict_proba(x, stories),
        from_memory.predict_proba(x, stories),
        atol=5e-3,
    )


def test_saved_linear_blend_fit_warm_starts(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(6)
    labels = [int(v) for v in rng.permutation(np.repeat([0, 2], 60))]
    stories = _stories(rng, labels)
    dense = rng.standard_normal((len(labels), 16)).astype(np.float32)
    weights = linear_blend.balanced_weights(labels)

    def fit(
        n: int, warm: linear_blend.LinearBlendModels | None = None
    ) -> linear_blend.LinearBlendModels:
        return linear_blend.fit_linear_blend(
            dense[:n],
            labels[:n],
            weights[:n],
            stories[:n],
            labels[:n],
            dense_c=0.1,
            tfidf_c=4.0,
            warm=warm,
        )

    monkeypatch.setattr(linear_blend, "_CACHE", LRUCache(maxsize=10_000))
    monkeypatch.setattr(linear_blend, "_LATEST", {})
    linear_blend.set_cached((7, "sig", 1), fit(len(labels) - 1), 10)
    warm_store.flush()
    monkeypatch.setattr(linear_blend, "_CACHE", LRUCache(maxsize=10_000))
    monkeypatch.setattr(linear_blend, "_LATEST", {})

    restored = linear_blend.latest(7)

    assert restored is not None
    cold = fit(len(labels))
    warm = fit(len(labels), warm=restored)
    x = cold.idf.transform(linear_blend.count_rows(stories)[:, cold.keep])
    np.testing.assert_allclose(
        linear_blend.up_minus_down(warm.tfidf, x),
        linear_blend.up_minus_down(cold.tfidf, x),
        atol=0.02,
    )
    assert warm.tfidf.n_iter_[0] <= cold.tfidf.n_iter_[0]


def test_first_regen_waits_for_startup_warms(monkeypatch: pytest.MonkeyPatch) -> None:
    decks: dict[int, object] = {}
    monkeypatch.setattr(server.Handler, "_decks", decks)
    threading.Timer(0.1, lambda: decks.__setitem__(7, object())).start()

    started = time.monotonic()
    assert server.wait_for_startup_warms([7], timeout_s=5.0, poll_s=0.01)
    assert time.monotonic() - started < 2.0

    assert not server.wait_for_startup_warms([7, 8], timeout_s=0.05, poll_s=0.01)
