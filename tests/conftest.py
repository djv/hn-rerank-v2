"""Shared pytest fixtures for the hn-rewrite test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from reddit_feed_cache import cache as reddit_feed_cache
from reddit_fetch_queue import queue as reddit_fetch_queue
from reddit_limiter import limiter as reddit_limiter
from llm_limiter import limiter as llm_limiter


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
