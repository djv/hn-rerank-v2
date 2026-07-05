"""Tests for http_fetch helpers."""

from __future__ import annotations

import io
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from http_fetch import urllib_fetch


class _FakeResp:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _empty_msg() -> Message:
    return Message()


def test_urllib_fetch_returns_200_with_body() -> None:
    body = b"hello world"
    with patch("http_fetch.urlopen", return_value=_FakeResp(200, body)):
        status, text = urllib_fetch("https://example.com/x", "ua")
    assert status == 200
    assert text == "hello world"


def test_urllib_fetch_handles_403_without_raising() -> None:
    """A 403 (or any 4xx/5xx) must return ``(status, \"\")``, not raise.

    Regression: pre-fix, the uncaught ``HTTPError`` propagated from
    ``urlopen`` and silently dropped topfeed tasks when both httpx and
    urllib got 403 from a Reddit block. The live service's topfeed
    was stalled for 25+ minutes because of this. See WORKLOG
    2026-06-29 for the diagnosis.
    """
    err = HTTPError(
        "https://www.reddit.com/r/MachineLearning/top/.rss",
        403,
        "Forbidden",
        _empty_msg(),
        io.BytesIO(b"block page"),
    )
    with patch("http_fetch.urlopen", side_effect=err):
        status, text = urllib_fetch("https://www.reddit.com/x", "ua")
    assert status == 403
    assert text == ""


def test_urllib_fetch_handles_429_without_raising() -> None:
    msg = _empty_msg()
    msg["Retry-After"] = "30"
    err = HTTPError(
        "https://www.reddit.com/r/x",
        429,
        "Too Many Requests",
        msg,
        io.BytesIO(b""),
    )
    with patch("http_fetch.urlopen", side_effect=err):
        status, text = urllib_fetch("https://www.reddit.com/x", "ua")
    assert status == 429
    assert text == ""


def test_urllib_fetch_handles_500_without_raising() -> None:
    err = HTTPError(
        "https://example.com/x",
        500,
        "Internal Server Error",
        _empty_msg(),
        io.BytesIO(b""),
    )
    with patch("http_fetch.urlopen", side_effect=err):
        status, text = urllib_fetch("https://example.com/x", "ua")
    assert status == 500
    assert text == ""


def test_urllib_fetch_propagates_network_errors() -> None:
    """URLError (network down, DNS, timeout) should still propagate so
    the caller's broad ``except Exception`` in the factory body can
    log it. Only HTTPError is special-cased."""
    with patch("http_fetch.urlopen", side_effect=URLError("name resolution failed")):
        with pytest.raises(URLError):
            urllib_fetch("https://example.com/x", "ua")
