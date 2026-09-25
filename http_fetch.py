"""Shared HTTP fetch helpers.

Used by both the RSS pipeline (pipeline.py) and the article-body fetcher
(server.py). The httpx → urllib fallback is needed because httpx's TLS
fingerprint is often blocked by Cloudflare, but the system OpenSSL
handshake (via urllib) gets through.
"""

from __future__ import annotations

import asyncio
import codecs
import ipaddress
import logging
import socket
from dataclasses import dataclass
from http.client import HTTPMessage
from typing import IO, Protocol
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

import httpx

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


def check_public_url(url: str) -> None:
    """Raise UnsafeUrlError unless ``url`` is http(s) to public addresses only.

    DNS failures propagate as OSError, like a failed connect would. There is
    a residual DNS-rebinding window between this check and the connect;
    closing it would need a pinned-IP transport, which isn't worth it here.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"scheme not allowed: {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise UnsafeUrlError("missing host")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as e:
        raise UnsafeUrlError(f"bad port: {e}") from e
    addrs = _resolve_host(host, port)
    if not addrs or not all(_is_public_ip(a) for a in addrs):
        raise UnsafeUrlError(f"non-public address for host {host!r}")


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
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    *,
    max_bytes: int = ARTICLE_MAX_BYTES,
    max_redirects: int = ARTICLE_MAX_REDIRECTS,
) -> GuardedResponse:
    """GET with an SSRF check on every hop and a streamed body-size cap.

    ``client`` must not follow redirects itself; they are followed here so
    each Location is re-checked. Non-200 responses return an empty body.
    Bodies over ``max_bytes`` are cut off (``truncated=True``) rather than
    rejected: the extractor only needs the first part of the page.
    """
    for _ in range(max_redirects + 1):
        await asyncio.to_thread(check_public_url, url)
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
        check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def guarded_urllib_fetch(
    url: str, user_agent: str, *, max_bytes: int = ARTICLE_MAX_BYTES
) -> tuple[int, str]:
    """``urllib_fetch`` with the same SSRF check (every hop) and size cap."""
    check_public_url(url)
    opener = build_opener(_CheckedRedirectHandler)
    req = Request(url, headers={"User-Agent": user_agent})
    try:
        with opener.open(req, timeout=15) as resp:
            body = resp.read(max_bytes)
            return resp.status, _decode_body(body, resp.headers.get_content_charset())
    except HTTPError as e:
        return e.code, ""


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
