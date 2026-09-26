"""Shared pytest fixtures for the hn-rewrite test suite."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator

import pytest
from hypothesis import settings

from reddit_feed_cache import cache as reddit_feed_cache
from reddit_fetch_queue import queue as reddit_fetch_queue
from reddit_limiter import limiter as reddit_limiter
from llm_limiter import limiter as llm_limiter

# Example budgets (HYPOTHESIS_PROFILE):
# - unset, local: a quarter of each test's examples (at least 10, or all of
#   them if fewer). A 10x run on 2026-09-26 found nothing, so everyday runs
#   trade depth for speed.
# - unset under CI: Hypothesis auto-selects the `ci` profile below (300 for
#   tests without their own @settings); explicit counts stay unscaled.
# - `deep`: 10x each test's count, at least 1000, no deadline. A one-off
#   search, several minutes: `HYPOTHESIS_PROFILE=deep uv run pytest tests/ -n 4`.
# - `dev` / `ci`: plain Hypothesis profiles (50 / 300 examples) for tests
#   without their own @settings; unscaled.
# A test's own @settings(max_examples=...) overrides any profile, so the
# scaled modes rewrite each test's final settings at collection instead.
settings.register_profile("dev", max_examples=50)
settings.register_profile("ci", max_examples=300, deadline=None, print_blob=True)
settings.register_profile("deep", deadline=None, print_blob=True)
_PROFILE = os.environ.get("HYPOTHESIS_PROFILE") or (
    "" if os.environ.get("CI") else "fast"
)
if _PROFILE not in ("", "fast"):
    settings.load_profile(_PROFILE)


def _scaled_examples(n: int) -> int:
    if _PROFILE == "deep":
        return max(1000, n * 10)
    return max(math.ceil(n / 4), min(n, 10))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _PROFILE not in ("fast", "deep"):
        return
    seen: set[int] = set()
    for item in items:
        test = getattr(item, "obj", None)
        test = getattr(test, "__func__", test)  # test-class methods are bound
        current = getattr(test, "_hypothesis_internal_use_settings", None)
        # Parametrized items share one function: scale it once.
        if test is None or current is None or id(test) in seen:
            continue
        seen.add(id(test))
        test._hypothesis_internal_use_settings = settings(
            current,
            max_examples=_scaled_examples(current.max_examples),
            **({"deadline": None} if _PROFILE == "deep" else {}),
        )


@pytest.fixture(autouse=True)
def reset_reddit_singletons() -> Iterator[None]:
    """Reset module-level singletons before and after every test.

    Also overrides the reddit_fetch_queue spread window and poll
    interval to near-zero for the duration of each test. Production
    defaults (50s MIN_FETCH_SPACING, 1s poll) would make any test that
    enqueues into the singleton block for the full timeout
    (~10-90 min). The per-instance RedditFetchQueue objects created
    by tests/test_reddit_fetch_queue.py are unaffected.

    POLL_INTERVAL is set on the CLASS (not the singleton instance) so
    that any new RedditFetchQueue() created during a test also uses
    the test value. The worker thread starts inside __init__ and
    reads POLL_INTERVAL on its first iteration, so setting only the
    instance attribute after construction is too late — the worker
    would already be in a 1-second sleep before the test can enqueue.
    """
    from reddit_fetch_queue import RedditFetchQueue

    llm_limiter.reset()
    reddit_limiter.reset()
    reddit_feed_cache.reset()
    reddit_fetch_queue.reset()
    orig_min_spacing = RedditFetchQueue.MIN_FETCH_SPACING
    orig_topfeeds = reddit_fetch_queue.SPREAD_WINDOW_TOPFEEDS
    orig_prewarm = reddit_fetch_queue.SPREAD_WINDOW_PREWARM
    orig_poll = RedditFetchQueue.POLL_INTERVAL
    RedditFetchQueue.MIN_FETCH_SPACING = 0.01
    reddit_fetch_queue.MIN_FETCH_SPACING = 0.01
    reddit_fetch_queue.SPREAD_WINDOW_TOPFEEDS = 0.01
    reddit_fetch_queue.SPREAD_WINDOW_PREWARM = 0.01
    RedditFetchQueue.POLL_INTERVAL = 0.001
    yield
    RedditFetchQueue.MIN_FETCH_SPACING = orig_min_spacing
    reddit_fetch_queue.MIN_FETCH_SPACING = orig_min_spacing
    reddit_fetch_queue.SPREAD_WINDOW_TOPFEEDS = orig_topfeeds
    reddit_fetch_queue.SPREAD_WINDOW_PREWARM = orig_prewarm
    RedditFetchQueue.POLL_INTERVAL = orig_poll
    reddit_fetch_queue.reset()
    reddit_limiter.reset()
    llm_limiter.reset()
    reddit_feed_cache.reset()
