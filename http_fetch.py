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
from urllib.request import Request, urlopen


def urllib_fetch(url: str, user_agent: str) -> tuple[int, str]:
    """Sync fetch via urllib (used as fallback when httpx is blocked)."""
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=15) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


async def fetch_with_urllib_fallback(
    client: Any,
    url: str,
    headers: dict[str, str],
    *,
    fallback_statuses: tuple[int, ...] = (403, 503),
) -> tuple[int, str, dict[str, str]]:
    """Try an httpx client.get(); on a fallback status code, retry via urllib.

    Returns (status, body, response_headers). body is "" for non-200
    responses, but headers are always populated (relevant for
    x-ratelimit-* on 429s).
    """
    resp = await client.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.status_code, resp.text, dict(resp.headers)
    if resp.status_code in fallback_statuses:
        logging.info(
            "%s: httpx %d, retrying with urllib",
            url,
            resp.status_code,
        )
        status, body = await asyncio.to_thread(urllib_fetch, url, headers["User-Agent"])
        if status == 200:
            return status, body, {}
        logging.warning("%s: urllib fallback returned %d", url, status)
        return status, "", {}
    return resp.status_code, "", dict(resp.headers)
