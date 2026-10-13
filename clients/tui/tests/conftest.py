from __future__ import annotations

from datetime import UTC, datetime

import pytest

import hn_rerank.app as reader_module
from hn_rerank.queue import default_sort


@pytest.fixture(autouse=True)
def fixed_startup_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep existing reader tests deterministic as the real UTC day changes."""
    day = datetime.fromordinal(739900).replace(tzinfo=UTC)  # even ordinal
    monkeypatch.setattr(
        reader_module, "default_sort", lambda now=None: default_sort(now or day)
    )
