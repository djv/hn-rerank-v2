"""Tests for http_fetch helpers."""

from __future__ import annotations

import io
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import httpx
import pytest

from http_fetch import fetch_with_urllib_fallback, urllib_fetch


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


# ---------------------------------------------------------------------------
# fetch_with_urllib_fallback: transport-error retry (2026-08-27)
#
# Six RSS feeds produced ~112 logging.error failures in a week -- all
# httpx.TransportError subclasses (RemoteProtocolError/ReadError/
# ConnectError), all against hosts confirmed live on manual probe. Before
# this fix, a transport error from client.get() propagated straight past
# fetch_with_urllib_fallback to the caller's bare `except Exception`, with
# no urllib retry -- unlike the existing 403/503 status-code fallback,
# which already had one. See WORKLOG 2026-08-27.
# ---------------------------------------------------------------------------


class _FakeHttpxResponse:
    def __init__(self, status_code: int, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient: .get() either raises or returns a
    canned _FakeHttpxResponse, one per call, in the order given."""

    def __init__(self, side_effects: list) -> None:
        self._side_effects = list(side_effects)
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url: str, headers: dict):
        self.calls.append((url, headers))
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


@pytest.mark.asyncio
async def test_fetch_with_urllib_fallback_returns_200_without_fallback() -> None:
    client = _FakeAsyncClient([_FakeHttpxResponse(200, "body text", {"x": "1"})])
    status, body, headers = await fetch_with_urllib_fallback(
        client, "https://example.com/feed", {"User-Agent": "ua"}
    )
    assert (status, body, headers) == (200, "body text", {"x": "1"})


@pytest.mark.asyncio
async def test_fetch_with_urllib_fallback_retries_on_transport_error() -> None:
    """A RemoteProtocolError from httpx must fall back to urllib, not
    propagate -- this is the exact bug: the transport error previously
    skipped the fallback entirely."""
    client = _FakeAsyncClient(
        [httpx.RemoteProtocolError("peer closed connection")]
    )
    with patch("http_fetch.urllib_fetch", return_value=(200, "recovered")) as mock_fetch:
        status, body, headers = await fetch_with_urllib_fallback(
            client, "https://example.com/feed", {"User-Agent": "ua"}
        )
    assert (status, body, headers) == (200, "recovered", {})
    mock_fetch.assert_called_once_with("https://example.com/feed", "ua")


@pytest.mark.asyncio
async def test_fetch_with_urllib_fallback_transport_error_then_urllib_also_fails() -> None:
    """If the urllib fallback also fails (non-200), return that status
    cleanly instead of raising -- callers treat any non-200 the same way
    regardless of which path produced it."""
    client = _FakeAsyncClient([httpx.ConnectError("connection refused")])
    with patch("http_fetch.urllib_fetch", return_value=(503, "")):
        status, body, headers = await fetch_with_urllib_fallback(
            client, "https://example.com/feed", {"User-Agent": "ua"}
        )
    assert (status, body, headers) == (503, "", {})


@pytest.mark.asyncio
async def test_fetch_with_urllib_fallback_still_falls_back_on_403() -> None:
    """Existing status-code fallback path is unchanged by the new
    transport-error handling."""
    client = _FakeAsyncClient([_FakeHttpxResponse(403)])
    with patch("http_fetch.urllib_fetch", return_value=(200, "via urllib")) as mock_fetch:
        status, body, headers = await fetch_with_urllib_fallback(
            client, "https://example.com/feed", {"User-Agent": "ua"}
        )
    assert (status, body, headers) == (200, "via urllib", {})
    mock_fetch.assert_called_once_with("https://example.com/feed", "ua")


@pytest.mark.asyncio
async def test_fetch_with_urllib_fallback_non_fallback_status_passes_through() -> None:
    """A status outside both `fallback_statuses` and 200 (e.g. 404) is
    returned as-is, with headers, and never triggers a urllib retry."""
    client = _FakeAsyncClient([_FakeHttpxResponse(404, headers={"x": "y"})])
    with patch("http_fetch.urllib_fetch") as mock_fetch:
        status, body, headers = await fetch_with_urllib_fallback(
            client, "https://example.com/feed", {"User-Agent": "ua"}
        )
    assert (status, body, headers) == (404, "", {"x": "y"})
    mock_fetch.assert_not_called()
