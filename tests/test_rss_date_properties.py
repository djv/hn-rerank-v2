"""UTC feed dates must not depend on the server's local timezone."""

from __future__ import annotations

import asyncio
import calendar
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import Literal

import pytest
from hypothesis import example, given, settings, strategies as st

from pipeline.enrichment import _fetch_and_parse_feed


class FeedDateMismatch(AssertionError):
    """The parsed feed timestamp differs from the declared UTC instant."""


@contextmanager
def _local_timezone(offset_minutes: int) -> Iterator[None]:
    """Use a fixed POSIX offset, then restore the process timezone."""
    previous = os.environ.get("TZ")
    sign = "+" if offset_minutes >= 0 else "-"
    hours, minutes = divmod(abs(offset_minutes), 60)
    os.environ["TZ"] = f"UTC{sign}{hours:02d}:{minutes:02d}"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _parsed_timestamp(
    published_at: datetime,
    date_kind: Literal["published", "updated"],
    offset_minutes: int,
) -> int:
    expected = calendar.timegm(published_at.utctimetuple())
    if date_kind == "published":
        feed = (
            '<rss version="2.0"><channel><title>Example</title><item>'
            "<title>Article</title><link>https://example.com/article</link>"
            f"<pubDate>{format_datetime(published_at, usegmt=True)}</pubDate>"
            "<description>Article content</description></item></channel></rss>"
        )
    else:
        feed = (
            '<feed xmlns="http://www.w3.org/2005/Atom"><title>Example</title>'
            "<entry><title>Article</title><id>https://example.com/article</id>"
            '<link href="https://example.com/article"/>'
            f"<updated>{published_at.isoformat()}</updated>"
            "<summary>Article content</summary></entry></feed>"
        )

    async def fetch(*args: object, **kwargs: object) -> tuple[int, str, dict[str, str]]:
        return 200, feed, {}

    with _local_timezone(offset_minutes), pytest.MonkeyPatch.context() as patch:
        patch.setattr("http_fetch.fetch_with_urllib_fallback", fetch)
        stories = asyncio.run(
            _fetch_and_parse_feed(
                "https://example.com/feed", 1, expected - 86400, expected + 86400, set()
            )
        )
    if len(stories) != 1:
        raise RuntimeError(f"Expected one feed entry, got {len(stories)}")
    return stories[0].time


@pytest.mark.parametrize("date_kind", ["published", "updated"])
@settings(max_examples=40, deadline=None)
@given(
    published_at=st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2035, 12, 31),
        timezones=st.just(timezone.utc),
    ).map(lambda value: value.replace(microsecond=0)),
    offset_minutes=st.integers(-12 * 60, 14 * 60),
)
@example(published_at=datetime(2000, 1, 1, tzinfo=timezone.utc), offset_minutes=1)
def test_feed_dates_preserve_the_utc_instant_in_any_local_timezone(
    date_kind: Literal["published", "updated"],
    published_at: datetime,
    offset_minutes: int,
) -> None:
    """Vary calendar dates and offsets for both feed date fields."""
    observed = _parsed_timestamp(published_at, date_kind, offset_minutes)
    expected = calendar.timegm(published_at.utctimetuple())
    if observed != expected:
        raise FeedDateMismatch(f"UTC timestamp {expected} was parsed as {observed}")


@pytest.mark.parametrize("date_kind", ["published", "updated"])
def test_feed_date_property_has_a_utc_control(
    date_kind: Literal["published", "updated"],
) -> None:
    published_at = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
    assert _parsed_timestamp(published_at, date_kind, 0) == int(
        published_at.timestamp()
    )
