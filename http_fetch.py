"""Shared HTTP fetch helpers.

Used by both the RSS pipeline (pipeline.py) and the article-body fetcher
(server.py). The httpx → urllib fallback is needed because httpx's TLS
fingerprint is often blocked by Cloudflare, but the system OpenSSL
handshake (via urllib) gets through.
"""

from __future__ import annotations

import asyncio
import codecs
import http.client
import ipaddress
import logging
import socket
import ssl
from collections.abc import Iterable
from dataclasses import dataclass
from http.client import HTTPMessage
from typing import IO, Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
    urlopen,
)

import httpcore
import httpx
from httpcore import SOCKET_OPTION

# Article URLs are chosen by whoever submitted the story, so fetches of them
# must not reach loopback/private/tailnet hosts (SSRF) or buffer an unbounded
# body. 5 MB comfortably exceeds real article HTML; ARTICLE_BODY_CHAR_LIMIT
# truncates the extracted text far below that anyway.
ARTICLE_MAX_BYTES = 5_000_000
ARTICLE_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class _HttpGetClient(Protocol):
    """The slice of ``httpx.AsyncClient`` the fallback helper uses."""

    async def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response: ...


class UnsafeUrlError(ValueError):
    """URL scheme or resolved address is not allowed for outbound fetches."""


def _resolve_host(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _is_public_ip(raw: str) -> bool:
    ip = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # is_global excludes loopback, RFC1918, link-local (cloud metadata),
    # and 100.64.0.0/10 (Tailscale CGNAT range).
    return ip.is_global and not ip.is_multicast


def _public_addresses(host: str, port: int) -> list[str]:
    """Resolve ``host`` once and return its addresses, or raise UnsafeUrlError
    if any is non-public. Callers connect to these exact addresses, so a DNS
    answer can't change between the check and the connect (rebinding)."""
    addrs = _resolve_host(host, port)
    if not addrs or not all(_is_public_ip(a) for a in addrs):
        raise UnsafeUrlError(f"non-public address for host {host!r}")
    return addrs


def check_url(url: str) -> None:
    """Raise UnsafeUrlError unless ``url`` is http(s) with a host and valid
    port. Addresses are checked at connect time (``_public_addresses``)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"scheme not allowed: {parts.scheme!r}")
    if not parts.hostname:
        raise UnsafeUrlError("missing host")
    try:
        parts.port
    except ValueError as e:
        raise UnsafeUrlError(f"bad port: {e}") from e


class _PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """httpcore network backend that connects only to validated public
    addresses. TLS still verifies the certificate against the URL's host:
    httpcore passes the origin hostname to start_tls, not the address."""

    def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        addrs = await asyncio.to_thread(_public_addresses, host, port)
        return await self._inner.connect_tcp(
            addrs[0], port, timeout, local_address, socket_options
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise UnsafeUrlError("unix sockets not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _public_only_transport() -> httpx.AsyncHTTPTransport:
    transport = httpx.AsyncHTTPTransport()
    # httpx has no public hook for the network backend; wrap its pool's.
    # tests/test_http_fetch.py exercises this against real sockets, so an
    # httpx upgrade that moves these attributes fails loudly there.
    pool = transport._pool
    if not isinstance(pool, httpcore.AsyncConnectionPool):  # proxy pools
        raise TypeError("unexpected httpx transport pool")
    pool._network_backend = _PublicOnlyBackend(pool._network_backend)
    return transport


def _decode_body(body: bytes, charset: str | None) -> str:
    encoding = "utf-8"
    if charset:
        try:
            encoding = codecs.lookup(charset).name
        except LookupError:
            pass
    return body.decode(encoding, errors="replace")


@dataclass(frozen=True)
class GuardedResponse:
    status: int
    headers: httpx.Headers
    text: str
    truncated: bool = False


async def guarded_get(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 10.0,
    max_bytes: int = ARTICLE_MAX_BYTES,
    max_redirects: int = ARTICLE_MAX_REDIRECTS,
) -> GuardedResponse:
    """GET an untrusted URL: public addresses only, a streamed size cap.

    Every connection (including each redirect hop, followed here so its
    scheme is checked) goes only to addresses validated by that same
    connection's DNS lookup. Non-200 responses return an empty body. Bodies
    over ``max_bytes`` are cut off (``truncated=True``) rather than rejected:
    the extractor only needs the first part of the page. Env proxies are
    ignored (a proxy would hide the destination from the check).
    """
    async with httpx.AsyncClient(
        transport=_public_only_transport(),
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for _ in range(max_redirects + 1):
            check_url(url)
            async with client.stream("GET", url, headers=headers) as resp:
                location = resp.headers.get("location")
                if resp.status_code in _REDIRECT_STATUSES and location:
                    url = urljoin(url, location)
                    continue
                if resp.status_code != 200:
                    return GuardedResponse(resp.status_code, resp.headers, "")
                buf = bytearray()
                truncated = False
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        del buf[max_bytes:]
                        truncated = True
                        break
                text = _decode_body(bytes(buf), resp.charset_encoding)
                return GuardedResponse(200, resp.headers, text, truncated)
    raise UnsafeUrlError(f"too many redirects (>{max_redirects})")


def _connect_public(
    address: tuple[str, int],
    timeout: float | None = None,
    source_address: tuple[str, int] | None = None,
    *args: object,
    **kwargs: object,
) -> socket.socket:
    host, port = address
    ip = _public_addresses(host, port)[0]
    return socket.create_connection((ip, port), timeout, source_address)


class _PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_public


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    # connect() dials via _create_connection, then wraps TLS with
    # server_hostname=self.host, so the certificate is checked by name.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_public


class _PublicHTTPHandler(HTTPHandler):
    def http_open(self, req: Request) -> http.client.HTTPResponse:
        return self.do_open(_PublicHTTPConnection, req)


class _PublicHTTPSHandler(HTTPSHandler):
    def __init__(self) -> None:
        self._ssl_context = ssl.create_default_context()
        super().__init__(context=self._ssl_context)

    def https_open(self, req: Request) -> http.client.HTTPResponse:
        return self.do_open(_PublicHTTPSConnection, req, context=self._ssl_context)


class _CheckedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        check_url(newurl)  # e.g. no ftp:// hop, which has its own handler
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def guarded_urllib_fetch(
    url: str, user_agent: str, *, max_bytes: int = ARTICLE_MAX_BYTES
) -> tuple[int, str]:
    """``urllib_fetch`` for untrusted URLs: same connect-time address check
    as ``guarded_get`` on every hop, same size cap, no env proxies."""
    check_url(url)
    opener = build_opener(
        ProxyHandler({}),
        _PublicHTTPHandler,
        _PublicHTTPSHandler,
        _CheckedRedirectHandler,
    )
    req = Request(url, headers={"User-Agent": user_agent})
    try:
        with opener.open(req, timeout=15) as resp:
            body = resp.read(max_bytes)
            return resp.status, _decode_body(body, resp.headers.get_content_charset())
    except HTTPError as e:
        return e.code, ""
    except URLError as e:
        # urllib wraps connect errors, including our UnsafeUrlError.
        if isinstance(e.reason, UnsafeUrlError):
            raise e.reason from e
        raise


def urllib_fetch(url: str, user_agent: str) -> tuple[int, str]:
    """Sync fetch via urllib (used as fallback when httpx is blocked).

    Returns ``(status, body)``. On HTTP error status (4xx/5xx), the
    ``HTTPError`` is caught and the status code is returned with an
    empty body so callers can handle 403/429/etc. uniformly without
    exceptions. Network-level errors (URLError, TimeoutError) still
    propagate.
    """
    req = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        return e.code, ""


async def _retry_via_urllib(
    url: str, user_agent: str, *, reason: str
) -> tuple[int, str, dict[str, str]]:
    """Shared urllib retry + logging for both fallback triggers below."""
    logging.info("%s: %s, retrying with urllib", url, reason)
    status, body = await asyncio.to_thread(urllib_fetch, url, user_agent)
    if status == 200:
        return status, body, {}
    logging.warning("%s: urllib fallback returned %d", url, status)
    return status, "", {}


async def fetch_with_urllib_fallback(
    client: _HttpGetClient,
    url: str,
    headers: dict[str, str],
    *,
    fallback_statuses: tuple[int, ...] = (403, 503),
) -> tuple[int, str, dict[str, str]]:
    """Try an httpx client.get(); on a fallback status code, retry via urllib.

    Also falls back to urllib on an httpx transport-level failure
    (RemoteProtocolError, ReadError, ConnectError, ...) -- these were
    previously uncaught here, so they propagated to the caller's generic
    ``except Exception`` and were logged as a hard feed failure with no
    retry, even though the same rationale for the status-code fallback
    applies: urllib's system-OpenSSL handshake often succeeds where
    httpx's TLS fingerprint gets blocked or reset (see module docstring).
    Six feeds produced ~112 such failures in a week, all against hosts
    that were live when probed manually -- these were transient, not
    dead feeds (WORKLOG 2026-08-27). A genuine network-down/DNS failure
    still propagates from the urllib attempt (see urllib_fetch) so the
    caller's broad except still catches it.

    Returns (status, body, response_headers). body is "" for non-200
    responses, but headers are always populated (relevant for
    x-ratelimit-* on 429s); a transport-error fallback has no httpx
    response to draw headers from, so it returns {} like the
    status-fallback branch below.
    """
    try:
        resp = await client.get(url, headers=headers)
    except httpx.TransportError as e:
        return await _retry_via_urllib(
            url, headers["User-Agent"], reason=f"httpx transport error ({e!r})"
        )

    if resp.status_code == 200:
        return resp.status_code, resp.text, dict(resp.headers)
    if resp.status_code in fallback_statuses:
        return await _retry_via_urllib(
            url, headers["User-Agent"], reason=f"httpx {resp.status_code}"
        )
    return resp.status_code, "", dict(resp.headers)
