"""Shared HTTP fetch helpers.

Used by both the RSS pipeline (pipeline.py) and the article-body fetcher
(server.py). The httpx → urllib fallback is needed because httpx's TLS
fingerprint is often blocked by Cloudflare, but the system OpenSSL
handshake (via urllib) gets through.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx


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
    client: Any,
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
