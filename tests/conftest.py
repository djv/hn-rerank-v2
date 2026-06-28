"""Shared pytest fixtures for the hn-rewrite test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from reddit_feed_cache import cache as reddit_feed_cache
from reddit_limiter import limiter as reddit_limiter


@pytest.fixture(autouse=True)
def reset_reddit_singletons() -> Iterator[None]:
    """Reset module-level singletons before and after every test."""
    reddit_limiter.reset()
    reddit_feed_cache.reset()
    yield
    reddit_limiter.reset()
    reddit_feed_cache.reset()
