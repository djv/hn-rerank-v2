"""Tests for http_fetch helpers."""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.message import Message
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import httpx
import pytest

import http_fetch
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


@pytest.mark.parametrize("code", [403, 429, 500, 503])
def test_urllib_fetch_returns_error_status_without_raising(code: int) -> None:
    """Any 4xx/5xx returns ``(status, "")`` instead of raising.

    Regression: pre-fix, the uncaught ``HTTPError`` propagated from
    ``urlopen`` and silently dropped topfeed tasks when both httpx and
    urllib got 403 from a Reddit block. The live service's topfeed
    was stalled for 25+ minutes because of this. See WORKLOG
    2026-06-29 for the diagnosis.
    """
    msg = _empty_msg()
    msg["Retry-After"] = "30"
    err = HTTPError("https://example.com/x", code, "err", msg, io.BytesIO(b"body"))
    with patch("http_fetch.urlopen", side_effect=err):
        status, text = urllib_fetch("https://example.com/x", "ua")
    assert (status, text) == (code, "")


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
    client = _FakeAsyncClient([httpx.RemoteProtocolError("peer closed connection")])
    with patch(
        "http_fetch.urllib_fetch", return_value=(200, "recovered")
    ) as mock_fetch:
        status, body, headers = await fetch_with_urllib_fallback(
            client, "https://example.com/feed", {"User-Agent": "ua"}
        )
    assert (status, body, headers) == (200, "recovered", {})
    mock_fetch.assert_called_once_with("https://example.com/feed", "ua")


@pytest.mark.asyncio
async def test_fetch_with_urllib_fallback_transport_error_then_urllib_also_fails() -> (
    None
):
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
    with patch(
        "http_fetch.urllib_fetch", return_value=(200, "via urllib")
    ) as mock_fetch:
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


# --- SSRF guard + size cap (guarded_get / guarded_urllib_fetch) ---
#
# These run against real local sockets: the guard lives in the connection
# layer (httpcore network backend, http.client connection), which a
# MockTransport would bypass entirely.


@dataclass
class _Route:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b"ok"


@dataclass
class _LocalServer:
    host: str
    port: int
    routes: dict[str, _Route]
    seen: list[tuple[str, str]]  # (path, Host header)

    def url(self, path: str, host: str | None = None) -> str:
        return f"http://{host or self.host}:{self.port}{path}"


@contextmanager
def _local_server(
    host: str = "127.0.0.1", routes: dict[str, _Route] | None = None
) -> Iterator[_LocalServer]:
    table = routes if routes is not None else {}
    seen: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append((self.path, self.headers.get("Host", "")))
            route = table.get(self.path, _Route(404, body=b""))
            self.send_response(route.status)
            for key, value in route.headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(route.body)))
            self.end_headers()
            self.wfile.write(route.body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer((host, 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _LocalServer(host, server.server_address[1], table, seen)
    finally:
        server.shutdown()
        server.server_close()


def _allow(monkeypatch: pytest.MonkeyPatch, *allowed: str) -> None:
    """Treat exactly ``allowed`` addresses as public (test servers are local)."""
    monkeypatch.setattr(http_fetch, "_is_public_ip", lambda ip: ip in allowed)


@pytest.mark.parametrize(
    ("addr", "public"),
    [
        ("93.184.215.14", True),
        ("2606:2800:21f:cb07:6820:80da:af6b:8b2c", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("100.100.100.100", False),  # Tailscale CGNAT
        ("0.0.0.0", False),
        ("::1", False),
        ("fe80::1%eth0", False),
        ("fd7a:115c:a1e0::1", False),  # Tailscale ULA
        ("::ffff:127.0.0.1", False),  # v4-mapped loopback
        ("224.0.0.1", False),
    ],
)
def test_is_public_ip(addr: str, public: bool) -> None:
    assert http_fetch._is_public_ip(addr) is public


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/x",
        "http:///nohost",
        "http://example.com:99999/",
    ],
)
async def test_guarded_fetches_reject_bad_scheme_host_or_port(url: str) -> None:
    with pytest.raises(http_fetch.UnsafeUrlError):
        await http_fetch.guarded_get(url, {})
    with pytest.raises(http_fetch.UnsafeUrlError):
        http_fetch.guarded_urllib_fetch(url, "ua")


async def test_loopback_is_refused_before_any_request() -> None:
    with _local_server(routes={"/": _Route()}) as srv:
        for url in (srv.url("/"), srv.url("/", host="localhost")):
            with pytest.raises(http_fetch.UnsafeUrlError):
                await http_fetch.guarded_get(url, {})
            with pytest.raises(http_fetch.UnsafeUrlError):
                http_fetch.guarded_urllib_fetch(url, "ua")
        assert srv.seen == []


async def test_connection_goes_to_the_checked_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS rebinding: the check and the connect share one lookup. The host
    below doesn't exist in real DNS, so the fetch only works if the client
    dials the validated address instead of resolving the name again; the
    Host header still carries the name (and TLS would verify it)."""
    lookups: list[str] = []

    def resolve(host: str, port: int) -> list[str]:
        lookups.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(http_fetch, "_resolve_host", resolve)
    _allow(monkeypatch, "127.0.0.1")
    with _local_server(routes={"/a": _Route(body=b"article")}) as srv:
        url = srv.url("/a", host="rebind.invalid")
        resp = await http_fetch.guarded_get(url, {})
        assert (resp.status, resp.text) == (200, "article")
        assert http_fetch.guarded_urllib_fetch(url, "ua") == (200, "article")
        host_header = f"rebind.invalid:{srv.port}"
        assert srv.seen == [("/a", host_header), ("/a", host_header)]
    assert lookups == ["rebind.invalid", "rebind.invalid"]


async def test_any_private_record_refuses_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        http_fetch, "_resolve_host", lambda host, port: ["127.0.0.1", "10.0.0.1"]
    )
    _allow(monkeypatch, "127.0.0.1")
    with _local_server(routes={"/": _Route()}) as srv:
        url = srv.url("/", host="mixed.invalid")
        with pytest.raises(http_fetch.UnsafeUrlError):
            await http_fetch.guarded_get(url, {})
        with pytest.raises(http_fetch.UnsafeUrlError):
            http_fetch.guarded_urllib_fetch(url, "ua")
        assert srv.seen == []


async def test_redirect_to_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """127.0.0.2 plays the public site, 127.0.0.1 the internal one."""
    _allow(monkeypatch, "127.0.0.2")
    with _local_server(routes={"/admin": _Route(body=b"secret")}) as internal:
        target = internal.url("/admin")
        routes = {"/a": _Route(302, {"Location": target})}
        with _local_server("127.0.0.2", routes) as public:
            with pytest.raises(http_fetch.UnsafeUrlError):
                await http_fetch.guarded_get(public.url("/a"), {})
            with pytest.raises(http_fetch.UnsafeUrlError):
                http_fetch.guarded_urllib_fetch(public.url("/a"), "ua")
            assert [path for path, _ in public.seen] == ["/a", "/a"]
        assert internal.seen == []


async def test_redirect_to_non_http_scheme_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow(monkeypatch, "127.0.0.1")
    routes = {"/a": _Route(302, {"Location": "ftp://127.0.0.1/x"})}
    with _local_server(routes=routes) as srv:
        with pytest.raises(http_fetch.UnsafeUrlError):
            await http_fetch.guarded_get(srv.url("/a"), {})
        with pytest.raises(http_fetch.UnsafeUrlError):
            http_fetch.guarded_urllib_fetch(srv.url("/a"), "ua")


async def test_guarded_get_truncates_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow(monkeypatch, "127.0.0.1")
    with _local_server(routes={"/big": _Route(body=b"a" * 5000)}) as srv:
        resp = await http_fetch.guarded_get(srv.url("/big"), {}, max_bytes=1000)
        status, text = http_fetch.guarded_urllib_fetch(
            srv.url("/big"), "ua", max_bytes=1000
        )
    assert (resp.status, resp.truncated, resp.text) == (200, True, "a" * 1000)
    assert (status, text) == (200, "a" * 1000)


async def test_guarded_get_follows_public_redirect_and_decodes_charset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow(monkeypatch, "127.0.0.1")
    routes = {
        "/old": _Route(301, {"Location": "/new"}),
        "/new": _Route(
            200,
            {"Content-Type": "text/html; charset=iso-8859-1"},
            "café".encode("latin-1"),
        ),
    }
    with _local_server(routes=routes) as srv:
        resp = await http_fetch.guarded_get(srv.url("/old"), {})
        assert http_fetch.guarded_urllib_fetch(srv.url("/old"), "ua") == (200, "café")
    assert (resp.status, resp.text, resp.truncated) == (200, "café", False)


async def test_guarded_get_caps_redirect_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow(monkeypatch, "127.0.0.1")
    routes = {"/loop": _Route(302, {"Location": "/loop"})}
    with _local_server(routes=routes) as srv:
        with pytest.raises(http_fetch.UnsafeUrlError, match="too many redirects"):
            await http_fetch.guarded_get(srv.url("/loop"), {}, max_redirects=3)
        assert len(srv.seen) == 4
