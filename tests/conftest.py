"""Shared pytest fixtures for the hn-rewrite test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from reddit_limiter import limiter as reddit_limiter


@pytest.fixture(autouse=True)
def reset_reddit_limiter() -> Iterator[None]:
    """Reset the module-level RedditRateLimiter before and after every test."""
    reddit_limiter.reset()
    yield
    reddit_limiter.reset()
