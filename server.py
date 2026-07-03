from __future__ import annotations

import asyncio
import gc
import hashlib
import html
import json
import logging
import os
import random
import re
import secrets
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from http import HTTPStatus
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

import feedparser
import justext
import trafilatura
from bs4 import BeautifulSoup
from flask import Flask, Response, jsonify, redirect, request
from flask.typing import ResponseReturnValue
import httpx

from database import Database, Story, User
from pipeline import Config, Embedder, RankedStory, is_hn_source
from llm_limiter import limiter as llm_limiter
from reddit_limiter import limiter as reddit_limiter
from http_fetch import fetch_with_urllib_fallback

ARTICLE_BODY_CHAR_LIMIT = 15_000
SELF_TEXT_PROMPT_CHAR_LIMIT = 8_000
COMMENT_PROMPT_CHAR_LIMIT = 12_000
SELF_TEXT_PROMPT_MIN_CHARS = 300
REDDIT_COMMENTS_CACHE_CHAR_LIMIT = 10_000
REDDIT_COMMENT_LIMIT = 40
REDDIT_RSS_USER_AGENT = "hn-rewrite/1.0 personal RSS reader; contact: local dashboard"
TLDR_PROMPT_VERSION = "detail-v4"
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROMPT_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int = 0


class FixedWindowLimiter:
    """Thread-safe fixed-window limiter for local public-demo protection."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    def try_acquire(
        self, checks: Sequence[tuple[str, int, int]], now: float | None = None
    ) -> RateLimitResult:
        now = time.monotonic() if now is None else now
        with self._lock:
            retry_after = 0
            normalized: list[tuple[str, int, int, deque[float]]] = []
            for key, limit, window_seconds in checks:
                if limit <= 0 or window_seconds <= 0:
                    continue
                bucket = self._buckets.setdefault(key, deque())
                cutoff = now - window_seconds
                while bucket and bucket[0] <= cutoff:
                    bucket.popleft()
                normalized.append((key, limit, window_seconds, bucket))
                if len(bucket) >= limit:
                    oldest = bucket[0]
                    retry_after = max(
                        retry_after,
                        max(1, int((oldest + window_seconds) - now + 0.999)),
                    )
            if retry_after > 0:
                return RateLimitResult(False, retry_after)
            for _key, _limit, _window_seconds, bucket in normalized:
                bucket.append(now)
        return RateLimitResult(True)


def _load_prompt(name: str) -> str:
    """Read a prompt template from the prompts/ directory (cached after first load)."""
    if name not in _PROMPT_CACHE:
        _PROMPT_CACHE[name] = (_PROMPTS_DIR / name).read_text(encoding="utf-8")
    return _PROMPT_CACHE[name]


def _parse_retry_after(header_value: str | None, default: float = 1.0) -> float:
    if not header_value:
        return default
    try:
        seconds = float(header_value)
    except ValueError:
        return default
    return max(0.0, min(seconds, 10.0))


def _normalize_tldr_markdown(text: str) -> str:
    """Make LLM summary Markdown predictable for the compact dashboard renderer."""
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if _looks_like_plain_heading(stripped):
            lines.append(f"### {stripped}")
            continue
        label_match = re.match(r"^([A-Z][A-Za-z ]{1,40}):\s*(.*)$", stripped)
        if label_match:
            label, body = label_match.groups()
            lines.append(f"- **{label}:** {body}" if body else f"- **{label}:**")
        else:
            lines.append(line)

    normalized = "\n".join(lines)
    normalized = re.sub(r"([.!;?:])\s+-\s+(?=\S)", r"\1\n- ", normalized)
    return normalized.strip()


def _looks_like_plain_heading(line: str) -> bool:
    if not line or len(line) > 48:
        return False
    if line.startswith(("#", "-", "*", ">", "`")) or line.endswith((".", ":", ",")):
        return False
    words = line.split()
    return len(words) <= 5 and any(ch.isalpha() for ch in line)


def load_env() -> None:
    # Try local .env
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path("/home/dev/hn_rerank/.env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")


@dataclass(frozen=True)
class RedditRssContext:
    self_text: str = ""
    top_comments: str = ""
    comment_count: int = 0


@dataclass(frozen=True)
class LessWrongContext:
    self_text: str = ""
    top_comments: str = ""
    comment_count: int = 0
    score: int = 0


@dataclass(frozen=True)
class ArticleFetchResult:
    body: str | None = None
    status: int | None = None
    error: str | None = None
    permanent: bool = False


@dataclass(frozen=True)
class LlmChatResult:
    content: str
    ok: bool
    status: int | None = None


@dataclass(frozen=True)
class TldrResult:
    kind: Literal["ok", "no_content", "llm_error"]
    tldr: str = ""
    error_status: int | None = None
    error_text: str = ""


def _llm_error_from(r: LlmChatResult) -> TldrResult:
    return TldrResult(kind="llm_error", error_status=r.status, error_text=r.content)


MIN_ARTICLE_CHARS = 200
MIN_GOOD_PARAS = 2


def _normalize_article_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _looks_bad_extraction(text: str) -> bool:
    """Structural quality check — rejects extractions where the tier itself
    indicates the text is likely dominated by boilerplate.

    No content-based blocklists; only structural signal (length, word count).
    """
    if not text or len(text) < MIN_ARTICLE_CHARS:  # noqa: SIM103
        return True
    words = text.split()
    if len(words) < 80:
        return True
    return False


def _extract_with_justext(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paragraphs = justext.justext(str(soup), justext.get_stoplist("English"))
    good = [
        p.text.strip()
        for p in paragraphs
        if not p.is_boilerplate and p.text and p.text.strip()
    ]
    if len(good) < MIN_GOOD_PARAS:
        return None
    text = _normalize_article_text("\n\n".join(good))
    if _looks_bad_extraction(text):
        return None
    return text


def _extract_with_bs_semantic(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    candidates: list = []

    # Prefer unclassed <article> (main content on e.g. The Register)
    for art in soup.find_all("article"):
        if not art.get("class"):
            candidates.append(art)

    main = soup.find("main")
    if main:
        candidates.append(main)

    candidates.extend(soup.find_all("article"))

    for node in candidates:
        text = _normalize_article_text(node.get_text(separator=" ", strip=True))
        if not _looks_bad_extraction(text):
            return text
    return None


def _extract_with_trafilatura(html: str) -> str | None:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        deduplicate=True,
    )
    if not text:
        return None
    text = _normalize_article_text(text)
    if _looks_bad_extraction(text):
        return None
    return text


def _extract_raw_body(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = _normalize_article_text(soup.get_text(separator=" ", strip=True))
    if len(text) < MIN_ARTICLE_CHARS:
        return None
    return text


def _extract_article_body(html: str) -> str | None:
    """Try each extraction strategy in order of quality.

    jusText (statistical boilerplate classification) → BS semantic
    (unclassed <article>/<main>) → trafilatura (precision mode) → raw body.
    """
    for extractor in (
        _extract_with_justext,
        _extract_with_bs_semantic,
        _extract_with_trafilatura,
        _extract_raw_body,
    ):
        text = extractor(html)
        if text:
            return text
    return None


async def _fetch_article_body_with_result(url: str) -> ArticleFetchResult:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
    }

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code in (429, 503) and attempt == 0:
                    await asyncio.sleep(1)
                    continue
                if resp.status_code == 200:
                    ct = (resp.headers.get("content-type") or "").lower()
                    if ct and not ct.startswith(
                        ("text/html", "application/xhtml+xml", "text/xml", "text/plain")
                    ):
                        return ArticleFetchResult(
                            status=200, error="non_html", permanent=True
                        )
                    html = resp.text
                    if "\x00" in html:
                        return ArticleFetchResult(
                            status=200, error="non_html", permanent=True
                        )
                elif resp.status_code in (403, 503) and attempt == 0:
                    status, html, _headers = await fetch_with_urllib_fallback(
                        client, url, headers
                    )
                    if status != 200:
                        return ArticleFetchResult(
                            status=status,
                            error=f"http_{status}",
                        )
                    if "\x00" in html:
                        return ArticleFetchResult(
                            status=200, error="non_html", permanent=True
                        )
                else:
                    return ArticleFetchResult(
                        status=resp.status_code,
                        error=f"http_{resp.status_code}",
                        permanent=resp.status_code in (404, 410),
                    )
        except Exception as e:
            return ArticleFetchResult(error=type(e).__name__)

        try:
            text = _extract_article_body(html)
        except Exception as e:
            logging.warning("extract_article_body failed for %s: %s", url, e)
            return ArticleFetchResult(error=f"extract_{type(e).__name__}")
        if text:
            return ArticleFetchResult(body=text[:ARTICLE_BODY_CHAR_LIMIT], status=200)
        return ArticleFetchResult(status=200, error="empty_extraction")

    return ArticleFetchResult(status=503, error="retry_exhausted")


async def _fetch_article_body(url: str) -> str | None:
    return (await _fetch_article_body_with_result(url)).body


def _reddit_post_rss_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in {"reddit.com", "old.reddit.com"}:
        return None
    path = parsed.path.rstrip("/")
    if "/comments/" not in path:
        return None
    if path.endswith(".rss"):
        rss_path = path
    else:
        rss_path = f"{path}/.rss"
    return urlunparse((parsed.scheme or "https", parsed.netloc, rss_path, "", "", ""))


def _clean_reddit_rss_html(raw_html: str) -> str:
    soup = BeautifulSoup(html.unescape(raw_html or ""), "html.parser")
    md = soup.find("div", class_="md")
    node = md if md else soup
    text = node.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _is_low_signal_reddit_comment(author: str, text: str) -> bool:
    normalized_author = author.lower().removeprefix("/u/")
    normalized_text = text.lower()
    if normalized_author in {"automoderator", "withoutreason1729"}:
        return True
    if normalized_text in {"[deleted]", "[removed]"}:
        return True
    if "i am a bot and this action was performed automatically" in normalized_text:
        return True
    if "your post is getting popular" in normalized_text:
        return True
    return len(text) < 30


LESSWRONG_COMMENT_LIMIT = 20
MAX_CONTENT_LENGTH = 10**6  # 1MB cap on POST bodies


def _extract_lesswrong_post_id(url: str | None) -> str | None:
    if not url:
        return None
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "posts":
        return parts[1]
    return None


def _clean_lesswrong_html(raw_html: str | None) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(html.unescape(raw_html), "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


async def _fetch_reddit_rss_context(url: str | None) -> RedditRssContext | None:
    rss_url = _reddit_post_rss_url(url)
    if not rss_url:
        return None

    if not await reddit_limiter.acquire():
        return None

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(
                rss_url, headers={"User-Agent": REDDIT_RSS_USER_AGENT}
            )
    except Exception:
        return None

    if resp.status_code == 429:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        rl_reset_raw = resp.headers.get("x-ratelimit-reset")
        rl_reset = float(rl_reset_raw) if rl_reset_raw else None
        reddit_limiter.on_429(retry_after, rate_limit_reset=rl_reset)
        return None
    if resp.status_code != 200:
        return None
    reddit_limiter.on_success()

    parsed = feedparser.parse(resp.text)
    entries = list(parsed.entries)
    if not entries:
        return None

    first = entries[0]
    post_html = ""
    if "content" in first and first.content:
        post_html = first.content[0].value
    elif "summary" in first:
        post_html = first.summary
    self_text = _clean_reddit_rss_html(post_html)

    comments: list[str] = []
    total_len = 0
    for entry in entries[1:]:
        comment_html = ""
        if "content" in entry and entry.content:
            comment_html = entry.content[0].value
        elif "summary" in entry:
            comment_html = entry.summary
        text = _clean_reddit_rss_html(comment_html)
        author = str(entry.get("author", "")).strip()
        if _is_low_signal_reddit_comment(author, text):
            continue
        label = author if author.startswith("/u/") else f"/u/{author}" if author else ""
        formatted = f"{label}: {text}" if label else text
        remaining = REDDIT_COMMENTS_CACHE_CHAR_LIMIT - total_len
        if remaining <= 0 or len(comments) >= REDDIT_COMMENT_LIMIT:
            break
        formatted = formatted[:remaining]
        comments.append(formatted)
        total_len += len(formatted) + 1

    return RedditRssContext(
        self_text=self_text[:SELF_TEXT_PROMPT_CHAR_LIMIT],
        top_comments=" ".join(comments)[:REDDIT_COMMENTS_CACHE_CHAR_LIMIT],
        comment_count=len(comments),
    )


async def _fetch_lesswrong_context(post_id: str) -> LessWrongContext | None:
    query = """
    query($id: String!) {
      post(input: { selector: { _id: $id } }) {
        result { _id commentCount baseScore contents { html } }
      }
      comments(input: { terms: { view: "postCommentsTop", postId: $id } }) {
        results { _id author baseScore htmlBody postedAt }
      }
    }
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://www.lesswrong.com/graphql",
                json={"query": query, "variables": {"id": post_id}},
            )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception as e:
        logging.error("LessWrong fetch failed: %r", e)
        return None
    if not payload.get("data"):
        return None
    data = payload["data"]
    post = (data.get("post") or {}).get("result") or {}
    comments_data = data.get("comments") or {}
    if not post:
        return None

    self_text = _clean_lesswrong_html((post.get("contents") or {}).get("html", ""))[
        :SELF_TEXT_PROMPT_CHAR_LIMIT
    ]

    comments: list[str] = []
    total_len = 0
    for c in comments_data.get("results") or []:
        text = _clean_lesswrong_html(c.get("htmlBody") or "")
        author = (c.get("author") or "").strip()
        label = f"/u/{author}" if author else ""
        line = f"{label}: {text}" if label else text
        remaining = COMMENT_PROMPT_CHAR_LIMIT - total_len
        if remaining <= 0 or len(comments) >= LESSWRONG_COMMENT_LIMIT:
            break
        line = line[:remaining]
        comments.append(line)
        total_len += len(line) + 1

    return LessWrongContext(
        self_text=self_text,
        top_comments=" ".join(comments)[:COMMENT_PROMPT_CHAR_LIMIT],
        comment_count=int(post.get("commentCount") or len(comments)),
        score=int(post.get("baseScore") or 0),
    )


async def _call_llm_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> LlmChatResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            for attempt in range(4):
                await llm_limiter.acquire()
                resp = await client.post(
                    base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                llm_limiter.record_response(
                    status=resp.status_code,
                    headers=resp.headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return LlmChatResult(
                        content=data["choices"][0]["message"]["content"],
                        ok=True,
                    )
                if resp.status_code == 429 and attempt < 3:
                    continue
                if resp.status_code == 503 and attempt < 3:
                    base = 2 ** (attempt + 1)
                    jitter = random.uniform(0, base * 0.5)
                    delay = _parse_retry_after(
                        resp.headers.get("Retry-After"), default=base + jitter
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            return LlmChatResult(
                content=f"Error from LLM Provider: HTTP {resp.status_code} - {resp.text}",
                ok=False,
                status=resp.status_code,
            )
    except Exception as e:
        logging.exception("_call_llm_chat: unexpected exception")
        return LlmChatResult(content=str(e), ok=False)


def _llm_cache_identity() -> str:
    provider = os.environ.get("LLM_PROVIDER", "mistral").lower()
    model = (
        "mistral-small-latest" if provider == "mistral" else "llama-3.3-70b-versatile"
    )
    return f"{provider}:{model}:{TLDR_PROMPT_VERSION}"


def _tldr_cache_key(
    *,
    title: str,
    self_text: str,
    top_comments: str,
    article_body: str,
) -> str:
    payload = {
        "identity": _llm_cache_identity(),
        "title": title,
        "self_text": self_text[:SELF_TEXT_PROMPT_CHAR_LIMIT],
        "top_comments": top_comments[:COMMENT_PROMPT_CHAR_LIMIT],
        "article_body": article_body[:ARTICLE_BODY_CHAR_LIMIT],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def generate_detailed_tldr(
    title: str,
    self_text: str = "",
    top_comments: str = "",
    article_body: str = "",
) -> TldrResult:
    provider = os.environ.get("LLM_PROVIDER", "mistral").lower()
    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        base_url = "https://api.mistral.ai/v1/chat/completions"
        model = "mistral-small-latest"
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        base_url = "https://api.groq.com/openai/v1/chat/completions"
        model = "llama-3.3-70b-versatile"

    if not api_key:
        return TldrResult(
            kind="llm_error",
            error_text="LLM API key not configured in environment.",
        )

    article_section = ""
    if self_text and len(self_text) >= SELF_TEXT_PROMPT_MIN_CHARS:
        article_section += f"Author's text:\n{self_text[:SELF_TEXT_PROMPT_CHAR_LIMIT]}"
    if article_body:
        article_section += (
            f"\n\nArticle body:\n{article_body[:ARTICLE_BODY_CHAR_LIMIT]}"
        )
    comments_section = top_comments[:COMMENT_PROMPT_CHAR_LIMIT]

    if not article_section and not top_comments:
        return TldrResult(kind="no_content")

    if article_section and comments_section:
        article_prompt = _load_prompt("article_v4.txt").format(
            title=title, article_section=article_section
        )
        discussion_prompt = _load_prompt("discussion_v4.txt").format(
            title=title, comments_section=comments_section
        )
        article_task = _call_llm_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=article_prompt,
            max_tokens=900,
        )
        discussion_task = _call_llm_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=discussion_prompt,
            max_tokens=900,
        )
        article_result, discussion_result = await asyncio.gather(
            article_task,
            discussion_task,
        )

        if article_result.ok and discussion_result.ok:
            article_text = _normalize_tldr_markdown(article_result.content)
            discussion_text = _normalize_tldr_markdown(discussion_result.content)
            if not article_text.strip() and not discussion_text.strip():
                return TldrResult(kind="llm_error", error_text="empty LLM response")
            return TldrResult(
                kind="ok",
                tldr=f"### Article\n{article_text}\n\n### Discussion\n{discussion_text}",
            )

        if article_result.ok and not discussion_result.ok:
            logging.warning(
                "tldr: discussion call failed (status=%s), salvaging article-only",
                discussion_result.status,
            )
            article_text = _normalize_tldr_markdown(article_result.content)
            if not article_text.strip():
                return TldrResult(kind="llm_error", error_text="empty LLM response")
            return TldrResult(kind="ok", tldr=f"### Article\n{article_text}")

        if not article_result.ok and discussion_result.ok:
            logging.warning(
                "tldr: article call failed (status=%s), salvaging discussion-only",
                article_result.status,
            )
            discussion_text = _normalize_tldr_markdown(discussion_result.content)
            if not discussion_text.strip():
                return TldrResult(kind="llm_error", error_text="empty LLM response")
            return TldrResult(kind="ok", tldr=f"### Discussion\n{discussion_text}")

        return TldrResult(
            kind="llm_error",
            error_status=article_result.status or discussion_result.status,
            error_text=f"Article: {article_result.content}; Discussion: {discussion_result.content}",
        )

    if article_section and not comments_section:
        prompt = _load_prompt("article_only_v4.txt").format(
            title=title, article_section=article_section
        )
    else:
        prompt = _load_prompt("discussion_only_v4.txt").format(
            title=title, comments_section=comments_section
        )

    result = await _call_llm_chat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=2000,
    )
    if result.ok:
        tldr_text = _normalize_tldr_markdown(result.content)
        if not tldr_text.strip():
            return TldrResult(kind="llm_error", error_text="empty LLM response")
        return TldrResult(kind="ok", tldr=tldr_text)
    return _llm_error_from(result)


async def _prefetch_tldrs_for_ranked(
    ranked_stories: list[RankedStory],
    db: Database,
    per_combo: int,
    stale_per_run: int = 0,
) -> int:
    if (per_combo <= 0 and stale_per_run <= 0) or not ranked_stories:
        return 0

    combo_groups: dict[str, list[int]] = {}
    if per_combo > 0:
        for rs in ranked_stories:
            for combo_key in rs.combo_keys.split():
                if combo_key.endswith("_mixed"):
                    continue
                group = combo_groups.setdefault(combo_key, [])
                if len(group) < per_combo:
                    group.append(rs.story.id)
                break

    seen: set[int] = set()
    story_ids: list[int] = []
    for combo_key in ["recent_hn", "recent_non-hn", "archive_hn", "archive_non-hn"]:
        for sid in combo_groups.get(combo_key, []):
            if sid not in seen:
                seen.add(sid)
                story_ids.append(sid)

    stale_added = 0
    if stale_per_run > 0:
        remaining = [rs for rs in ranked_stories if rs.story.id not in seen]
        cached_keys = db.get_tldr_cache_keys([rs.story.id for rs in remaining])
        for rs in remaining:
            if stale_added >= stale_per_run:
                break
            stored_key = cached_keys.get(rs.story.id)
            if stored_key is None:
                continue
            current_key = _tldr_cache_key(
                title=rs.story.title,
                self_text=rs.story.self_text or "",
                top_comments=rs.story.top_comments or "",
                article_body=rs.story.article_body or "",
            )
            if current_key != stored_key:
                seen.add(rs.story.id)
                story_ids.append(rs.story.id)
                stale_added += 1

    if not story_ids:
        return 0

    sem = asyncio.Semaphore(2)

    async def _prefetch_one(story_id: int) -> bool:
        story = db.get_story(story_id)
        if not story:
            return False

        title = story.title
        self_text = story.self_text or ""
        top_comments = story.top_comments or ""
        article_body = story.article_body or ""

        if not (self_text or top_comments or article_body):
            return False

        cache_key = _tldr_cache_key(
            title=title,
            self_text=self_text,
            top_comments=top_comments,
            article_body=article_body,
        )
        if db.get_tldr_cache(story_id, cache_key):
            return False

        async with sem:
            result = await generate_detailed_tldr(
                title,
                self_text=self_text,
                top_comments=top_comments,
                article_body=article_body,
            )
        if result.kind != "ok":
            return False
        db.upsert_tldr_cache(story_id, cache_key, result.tldr)
        return True

    results = await asyncio.gather(
        *(_prefetch_one(sid) for sid in story_ids), return_exceptions=True
    )
    generated = sum(1 for r in results if r is True)

    if generated:
        logging.info(
            "tldr_prefetch generated=%s candidates=%s per_combo=%s stale_added=%s",
            generated,
            len(story_ids),
            per_combo,
            stale_added,
        )
    return generated


SKELETON_HTML = b"""<!DOCTYPE html>
<html><head><meta http-equiv="refresh" content="1"></head>
<body><p>Loading your personalized dashboard...</p></body></html>"""


class Handler:
    config: Config
    db: Database
    embedder: Embedder
    regen_event: threading.Event
    _dashboard_cache: dict[str, tuple[bytes, float, int]] = {}
    _dashboard_versions: dict[int, int] = {}
    _dashboard_versions_guard = threading.Lock()
    _cold_stories: list[RankedStory] = []
    _article_fetch_in_flight: set[int] = set()
    _warm_bg_lock = threading.Lock()
    _render_locks: dict[int, threading.Lock] = {}
    _render_locks_guard = threading.Lock()
    _warmup_requested_versions: dict[int, int] = {}
    _warmup_last_request_at: dict[int, float] = {}
    _warmup_timers: dict[int, threading.Timer] = {}
    _warmup_running_users: set[int] = set()
    _warmup_in_flight_guard = threading.Lock()
    _WARM_DEBOUNCE_S: float = 1.0
    _public_demo_limiter = FixedWindowLimiter()

    @classmethod
    def reset_public_demo_limiter(cls) -> None:
        cls._public_demo_limiter = FixedWindowLimiter()

    @classmethod
    def _render_dashboard_for_user(
        cls, user: User, expected_version: int | None = None
    ) -> bytes:
        """Render personalized dashboard with SWR semantics."""
        request_start = time.perf_counter()
        cache_key = f"dashboard_{user.id}"
        if expected_version is None:
            expected_version = cls._dashboard_version(user.id)

        cached = cls._dashboard_cache.get(cache_key)

        # Cache hit with matching version → return immediately
        if cached and cached[2] == expected_version:
            logging.info(
                "dashboard_render user_id=%s version=%s result=cache_hit elapsed_ms=%.1f cache_age_s=%.1f",
                user.id,
                expected_version,
                (time.perf_counter() - request_start) * 1000,
                time.time() - cached[1],
            )
            return cached[0]

        # Stale cache (wrong version) → return stale, trigger warm
        if cached:
            logging.info(
                "dashboard_render user_id=%s version=%s result=stale_hit cache_version=%s elapsed_ms=%.1f",
                user.id,
                expected_version,
                cached[2],
                (time.perf_counter() - request_start) * 1000,
            )
            cls._trigger_warm(user, expected_version)
            return cached[0]

        # No per-user cache → render the cold deck, then warm the
        # personalized version in the background.
        n_feedback = sum(cls.db.count_feedback_by_action(user.id).values())
        if n_feedback > 0:
            from pipeline import build_cold_deck

            cold_stories = build_cold_deck(cls.db, user_id=user.id)
        else:
            cold_stories = cls._cold_stories
        if cold_stories:
            from pipeline import generate_dashboard_bytes

            html = generate_dashboard_bytes(
                cold_stories,
                cls.config,
                cls.db,
                user.id,
                user.token,
                dashboard_version=0,
                dashboard_latest_version=cls._dashboard_version(user.id),
            )
            logging.info(
                "dashboard_render user_id=%s version=%s result=cold_deck stories=%s elapsed_ms=%.1f",
                user.id,
                expected_version,
                len(cold_stories),
                (time.perf_counter() - request_start) * 1000,
            )
            if n_feedback > 0:
                cls._trigger_warm(user, expected_version)
            return html

        # No cache and no cold deck → return skeleton, trigger warm
        logging.info(
            "dashboard_render user_id=%s version=%s result=skeleton elapsed_ms=%.1f",
            user.id,
            expected_version,
            (time.perf_counter() - request_start) * 1000,
        )
        cls._trigger_warm(user, expected_version)
        return SKELETON_HTML

    @classmethod
    def _get_render_lock(cls, user_id: int) -> threading.Lock:
        with cls._render_locks_guard:
            lock = cls._render_locks.get(user_id)
            if lock is None:
                lock = threading.Lock()
                cls._render_locks[user_id] = lock
            return lock

    @classmethod
    def _dashboard_version(cls, user_id: int) -> int:
        with cls._dashboard_versions_guard:
            return cls._dashboard_versions.get(user_id, 0)

    @classmethod
    def _invalidate_dashboard_cache(cls, user_id: int) -> int:
        with cls._dashboard_versions_guard:
            version = cls._dashboard_versions.get(user_id, 0) + 1
            cls._dashboard_versions[user_id] = version
            logging.info(
                "dashboard_cache_invalidated user_id=%s version=%s", user_id, version
            )
            return version

    @classmethod
    def _trigger_warm(cls, user: User, version: int) -> None:
        current_version = cls._dashboard_version(user.id)
        effective_version = max(version, current_version)
        with cls._warmup_in_flight_guard:
            previous_version = cls._warmup_requested_versions.get(user.id)
            if previous_version is not None and effective_version <= previous_version:
                return
            cls._warmup_requested_versions[user.id] = effective_version
            cls._warmup_last_request_at[user.id] = time.monotonic()
            cls._schedule_warm_timer_locked(user, cls._WARM_DEBOUNCE_S)

    @classmethod
    def _schedule_warm_timer_locked(
        cls, user: User, delay_seconds: float
    ) -> threading.Timer:
        previous_timer = cls._warmup_timers.get(user.id)
        if previous_timer is not None:
            previous_timer.cancel()
        timer = threading.Timer(
            delay_seconds,
            cls._warm_timer_fired,
            args=(user,),
        )
        timer.daemon = True
        cls._warmup_timers[user.id] = timer
        timer.start()
        return timer

    @classmethod
    def _warm_timer_fired(cls, user: User) -> None:
        with cls._warmup_in_flight_guard:
            timer = cls._warmup_timers.get(user.id)
            if timer is not threading.current_thread():
                return
            requested_version = cls._warmup_requested_versions.get(user.id)
            last_request_at = cls._warmup_last_request_at.get(user.id)
            if requested_version is None or last_request_at is None:
                cls._warmup_timers.pop(user.id, None)
                return
            elapsed_s = time.monotonic() - last_request_at
            remaining_s = cls._WARM_DEBOUNCE_S - elapsed_s
            if remaining_s > 0:
                cls._schedule_warm_timer_locked(user, remaining_s)
                return
            cls._warmup_timers.pop(user.id, None)
            if user.id in cls._warmup_running_users:
                return
            cls._warmup_running_users.add(user.id)

        try:
            cls._run_warm_attempt(user, requested_version)
        except Exception as e:
            logging.exception(
                "Failed warming dashboard cache for user_id=%s: %s", user.id, e
            )
        finally:
            try:
                cls._finish_warm_attempt(user, requested_version)
            finally:
                cls._collect_after_warm_attempt()

    @classmethod
    def _collect_after_warm_attempt(cls) -> None:
        try:
            collected = gc.collect()
        except Exception:
            logging.debug("warm_gc result=failed", exc_info=True)
            return
        logging.debug("warm_gc result=completed collected=%s", collected)

    @classmethod
    def _run_warm_attempt(cls, user: User, requested_version: int) -> None:
        warm_start = time.perf_counter()
        cache_key = f"dashboard_{user.id}"

        cached = cls._dashboard_cache.get(cache_key)
        if cached and cached[2] >= requested_version:
            return

        with cls._dashboard_versions_guard:
            if cls._dashboard_versions.get(user.id, 0) < requested_version:
                logging.info(
                    "dashboard_warm user_id=%s version=%s result=skipped_stale elapsed_ms=%.1f",
                    user.id,
                    requested_version,
                    (time.perf_counter() - warm_start) * 1000,
                )
                return

        lock = cls._get_render_lock(user.id)
        with lock:
            cached = cls._dashboard_cache.get(cache_key)
            if cached and cached[2] >= requested_version:
                return

            from pipeline import (
                RankTrace,
                fast_rerank_for_user,
                generate_dashboard_bytes,
            )

            trace = RankTrace()
            render_start = time.perf_counter()
            with trace.stage("rank_total"):
                final = fast_rerank_for_user(
                    cls.db,
                    cls.config,
                    cls.embedder,
                    user.id,
                    trace=trace,
                )
            rank_ms = (time.perf_counter() - render_start) * 1000

            html_start = time.perf_counter()
            html = generate_dashboard_bytes(
                final,
                cls.config,
                cls.db,
                user.id,
                user.token,
                dashboard_version=requested_version,
                dashboard_latest_version=cls._dashboard_version(user.id),
            )
            html_ms = (time.perf_counter() - html_start) * 1000

            with cls._dashboard_versions_guard:
                cached = cls._dashboard_cache.get(cache_key)
                if cached and cached[2] > requested_version:
                    logging.info(
                        "dashboard_warm user_id=%s version=%s result=skipped_newer_cache_after_rank elapsed_ms=%.1f cache_version=%s",
                        user.id,
                        requested_version,
                        (time.perf_counter() - warm_start) * 1000,
                        cached[2],
                    )
                    return
                cls._dashboard_cache[cache_key] = (
                    html,
                    time.time(),
                    requested_version,
                )
            cls._enforce_cache_cap()

            logging.info(
                "dashboard_warm user_id=%s version=%s result=completed rank_ms=%.1f html_ms=%.1f stories=%s",
                user.id,
                requested_version,
                rank_ms,
                html_ms,
                len(final),
            )
            logging.info("rank_perf %s", trace.format_log_fields())

            per_combo = cls.config.tldr_prefetch_per_combo
            stale_per_run = cls.config.tldr_prefetch_stale_per_run
            if cls.config.article_fetch_max_per_run > 0 or (
                (per_combo > 0 or stale_per_run > 0) and final
            ):
                t = threading.Thread(
                    target=lambda: cls._warm_background_tasks(
                        final,
                        cls.db,
                        cls.embedder,
                        cls.config,
                        per_combo,
                    ),
                    daemon=True,
                )
                t.start()

    @classmethod
    def _finish_warm_attempt(cls, user: User, completed_version: int) -> None:
        with cls._warmup_in_flight_guard:
            user_id = user.id
            cls._warmup_running_users.discard(user_id)
            requested_version = cls._warmup_requested_versions.get(user_id)
            if requested_version is None:
                cls._warmup_last_request_at.pop(user_id, None)
                cls._warmup_timers.pop(user_id, None)
                return
            if requested_version == completed_version:
                cls._warmup_requested_versions.pop(user_id, None)
                cls._warmup_last_request_at.pop(user_id, None)
                timer = cls._warmup_timers.pop(user_id, None)
                if timer is not None:
                    timer.cancel()
                return

            last_request_at = cls._warmup_last_request_at.get(user_id, time.monotonic())
            remaining_s = max(
                0.0, cls._WARM_DEBOUNCE_S - (time.monotonic() - last_request_at)
            )
            cls._schedule_warm_timer_locked(user, remaining_s)

    @classmethod
    def _enforce_cache_cap(cls, max_entries: int = 100) -> None:
        if len(cls._dashboard_cache) <= max_entries:
            return
        keys = sorted(
            cls._dashboard_cache.keys(),
            key=lambda k: cls._dashboard_cache[k][1],
        )
        for k in keys[:-max_entries]:
            del cls._dashboard_cache[k]

    @classmethod
    def _bump_all_cached_versions(cls) -> None:
        with cls._dashboard_versions_guard:
            for uid in list(cls._dashboard_versions.keys()):
                cls._dashboard_versions[uid] += 1
        logging.info(
            "bump_all_cached_versions count=%s",
            len(cls._dashboard_versions),
        )

    @classmethod
    def _rebuild_cold_deck(cls) -> None:
        from pipeline import build_cold_deck

        cold_stories = build_cold_deck(cls.db)
        cls._cold_stories = cold_stories
        logging.info("cold_deck_rebuilt stories=%s", len(cold_stories))

    @classmethod
    def _warm_background_tasks(
        cls,
        final: list[RankedStory],
        db: Database,
        embedder: Embedder,
        config: Config,
        per_combo: int,
    ) -> None:
        """Article body fetch (deduped by story ID) -> TLDR prefetch."""
        from pipeline import (
            select_article_fetch_candidates,
            fetch_and_cache_article_bodies,
        )

        if config.article_fetch_max_per_run > 0 and final:
            fetch_targets = select_article_fetch_candidates(
                ranked=final,
                dashboard_selected=final[: config.count],
                db=db,
                max_per_run=config.article_fetch_max_per_run,
                max_age_days=config.article_fetch_max_age_days,
            )
            runnable: list[Story] = []
            with cls._warm_bg_lock:
                for s in fetch_targets:
                    if s.id not in cls._article_fetch_in_flight:
                        runnable.append(s)
                cls._article_fetch_in_flight.update(s.id for s in runnable)
            try:
                if runnable:
                    asyncio.run(
                        fetch_and_cache_article_bodies(
                            db=db,
                            embedder=embedder,
                            stories=runnable,
                            concurrency=config.article_fetch_concurrency,
                        )
                    )
            except Exception:
                logging.exception("warm_background article_fetch failed")
            finally:
                with cls._warm_bg_lock:
                    cls._article_fetch_in_flight.difference_update(
                        s.id for s in runnable
                    )

        stale_per_run = config.tldr_prefetch_stale_per_run
        if (per_combo > 0 or stale_per_run > 0) and final:
            asyncio.run(_prefetch_tldrs_for_ranked(final, db, per_combo, stale_per_run))


def _no_cache_dashboard_response(
    html: bytes, *, user: User | None = None, set_session_cookie: bool = False
) -> Response:
    response = Response(
        html, status=HTTPStatus.OK, content_type="text/html; charset=utf-8"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    if set_session_cookie and user is not None:
        response.set_cookie(
            "hn_token",
            user.token,
            max_age=31536000,
            path="/",
            samesite="Lax",
            httponly=True,
        )
    return response


def _flask_json_response(
    data: dict[str, object],
    status: int | HTTPStatus = HTTPStatus.OK,
    headers: dict[str, str] | None = None,
) -> Response:
    response = jsonify(data)
    response.status_code = int(status)
    response.headers["Access-Control-Allow-Origin"] = "*"
    for name, value in (headers or {}).items():
        response.headers[name] = value
    return response


def _flask_text_error(code: int | HTTPStatus, message: str | None = None) -> Response:
    return Response(
        message or HTTPStatus(code).phrase,
        status=int(code),
        content_type="text/plain; charset=utf-8",
    )


def _flask_user(runtime: type[Handler]) -> User | None:
    token = request.cookies.get("hn_token")
    if not token:
        return None
    return runtime.db.get_user_by_token(token)


def _flask_client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        first = forwarded_for.split(",", 1)[0].strip()
        if first:
            return first
    return request.remote_addr or "127.0.0.1"


def _acquire_session_create_quota(runtime: type[Handler]) -> RateLimitResult:
    config = runtime.config
    return runtime._public_demo_limiter.try_acquire(
        (
            (
                f"session-create:ip:{_flask_client_ip()}",
                config.session_create_per_ip_limit,
                config.session_create_per_ip_window_seconds,
            ),
        )
    )


def _acquire_profile_link_quota(runtime: type[Handler]) -> RateLimitResult:
    config = runtime.config
    return runtime._public_demo_limiter.try_acquire(
        (
            (
                f"profile-link:ip:{_flask_client_ip()}",
                config.profile_link_per_ip_limit,
                config.profile_link_per_ip_window_seconds,
            ),
        )
    )


def _flask_request_origin() -> tuple[str, str] | None:
    host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
    if not host:
        return None
    proto = request.headers.get("X-Forwarded-Proto") or "http"
    if "," in proto:
        proto = proto.split(",", 1)[0].strip()
    return proto, host


def _flask_is_same_origin_post() -> bool:
    fetch_site = request.headers.get("Sec-Fetch-Site")
    if fetch_site == "cross-site":
        return False

    origin = request.headers.get("Origin")
    if not origin:
        return True
    parsed = urlparse(origin)
    if not parsed.scheme or not parsed.netloc:
        return False
    request_origin = _flask_request_origin()
    if request_origin is None:
        return False
    request_scheme, request_host = request_origin
    if request.headers.get("X-Forwarded-Host"):
        return parsed.netloc == request_host
    return parsed.scheme == request_scheme and parsed.netloc == request_host


def _flask_cross_site_post_response() -> Response | None:
    if _flask_is_same_origin_post():
        return None
    return _flask_json_response(
        {"error": "Cross-site POSTs are not allowed"}, status=HTTPStatus.FORBIDDEN
    )


def _flask_rate_limit_response(message: str, retry_after_seconds: int) -> Response:
    return _flask_json_response(
        {"error": message, "retry_after": retry_after_seconds},
        status=HTTPStatus.TOO_MANY_REQUESTS,
        headers={"Retry-After": str(retry_after_seconds)},
    )


def _acquire_feedback_quota(runtime: type[Handler], user: User) -> RateLimitResult:
    config = runtime.config
    return runtime._public_demo_limiter.try_acquire(
        (
            (
                f"feedback:user:{user.id}",
                config.feedback_per_user_limit,
                config.feedback_per_user_window_seconds,
            ),
            (
                "feedback:global",
                config.feedback_global_limit,
                config.feedback_global_window_seconds,
            ),
        )
    )


def _flask_session_limit_key(user: User | None) -> str:
    if user is not None:
        return f"user:{user.id}"
    return f"addr:{_flask_client_ip()}"


def _acquire_tldr_uncached_quota(
    runtime: type[Handler], user: User | None
) -> RateLimitResult:
    session_key = _flask_session_limit_key(user)
    config = runtime.config
    return runtime._public_demo_limiter.try_acquire(
        (
            (
                f"tldr:user:{session_key}",
                config.tldr_uncached_per_user_limit,
                config.tldr_uncached_per_user_window_seconds,
            ),
            (
                "tldr:global",
                config.tldr_uncached_global_limit,
                config.tldr_uncached_global_window_seconds,
            ),
        )
    )


def _handle_flask_feedback(runtime: type[Handler]) -> Response:
    user = _flask_user(runtime)
    if not user:
        return _flask_json_response(
            {"error": "No session"}, status=HTTPStatus.UNAUTHORIZED
        )

    try:
        body = request.get_data(cache=True)
        if len(body) > MAX_CONTENT_LENGTH:
            return _flask_text_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return _flask_json_response(
                {"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST
            )

        story_id = data.get("story_id")
        action = data.get("action")
        if (
            not isinstance(story_id, int)
            or isinstance(story_id, bool)
            or action not in {"up", "neutral", "down", "clear"}
        ):
            return _flask_json_response(
                {"error": "Invalid feedback"}, status=HTTPStatus.BAD_REQUEST
            )

        quota = _acquire_feedback_quota(runtime, user)
        if not quota.allowed:
            return _flask_rate_limit_response(
                "Too many votes from this demo session. Please try again later.",
                quota.retry_after_seconds,
            )

        if action == "clear":
            runtime.db.delete_feedback(user.id, story_id)
        else:
            runtime.db.upsert_feedback(user.id, story_id, action)

        # Every vote invalidates and warms the next dashboard version so refillQueue
        # cannot reintroduce a story the user has just acted on.
        version = runtime._invalidate_dashboard_cache(user.id)
        runtime._trigger_warm(user, version)
        runtime.regen_event.set()

        return _flask_json_response(
            {
                "ok": True,
                "ranking_refresh_queued": True,
                "target_version": version,
            }
        )
    except Exception:
        logging.exception("Error handling feedback")
        return _flask_json_response(
            {"error": "Internal error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR
        )


def _handle_flask_tldr_detail(runtime: type[Handler]) -> Response:
    try:
        user = _flask_user(runtime)
        body = request.get_data(cache=True)
        if len(body) > MAX_CONTENT_LENGTH:
            return _flask_text_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return _flask_json_response(
                {"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST
            )

        story_id_raw = data.get("story_id")
        if not isinstance(story_id_raw, int) or isinstance(story_id_raw, bool):
            return _flask_json_response(
                {"error": "Invalid request: story_id must be an integer"},
                status=HTTPStatus.BAD_REQUEST,
            )
        story_id: int = story_id_raw

        story = runtime.db.get_story(story_id)
        if not story:
            return _flask_json_response(
                {"error": "Story not found in database"}, status=HTTPStatus.NOT_FOUND
            )

        def _stale_tldr_fallback(reason: str) -> Response | None:
            stale_tldr = runtime.db.get_any_tldr_for_story(story.id)
            if not stale_tldr:
                return None
            logging.info(
                "tldr_detail story_id=%s result=stale_fallback reason=%s",
                story.id,
                reason,
            )
            return _flask_json_response(
                {"ok": True, "tldr": stale_tldr, "cached": True, "stale": True}
            )

        article_body = story.article_body or None
        cache_key = _tldr_cache_key(
            title=story.title,
            self_text=story.self_text or "",
            top_comments=story.top_comments or "",
            article_body=article_body or "",
        )
        cached_tldr = runtime.db.get_tldr_cache(story.id, cache_key)
        if cached_tldr:
            logging.info(
                "tldr_detail story_id=%s result=cache_hit cache_key=%s",
                story.id,
                cache_key[:12],
            )
            return _flask_json_response(
                {"ok": True, "tldr": cached_tldr, "cached": True}
            )

        quota = _acquire_tldr_uncached_quota(runtime, user)
        if not quota.allowed:
            fallback = _stale_tldr_fallback("quota_denied")
            if fallback:
                return fallback
            return _flask_rate_limit_response(
                "Demo TLDR quota reached. Cached summaries still work; please try a new summary later.",
                quota.retry_after_seconds,
            )

        # If an HN story has comments but no cached comment text, fetch them lazily.
        if (
            is_hn_source(story.source)
            and not story.top_comments
            and (story.comment_count or 0) > 0
        ):
            try:
                from pipeline import fetch_story

                async def do_fetch() -> Story | None:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        return await fetch_story(client, story_id, runtime.db)

                updated = asyncio.run(do_fetch())
                if updated:
                    story = updated
            except Exception as e:
                logging.error("Failed to dynamically fetch comments for TLDR: %r", e)

        article_body = story.article_body or article_body

        if (
            story.source.startswith("rss_reddit_")
            and story.url
            and (not story.self_text or not story.top_comments)
        ):
            reddit_context = asyncio.run(_fetch_reddit_rss_context(story.url))
            if reddit_context and (
                reddit_context.self_text or reddit_context.top_comments
            ):
                from pipeline import _merge_source_context

                story = _merge_source_context(
                    story, reddit_context, article_body, prefer_longer_comments=True
                )
                runtime.db.upsert_story(story)

        if (
            story.source == "rss_lesswrong_com"
            and story.url
            and (not story.self_text or not story.top_comments)
        ):
            post_id = _extract_lesswrong_post_id(story.url)
            if post_id:
                lw_context = asyncio.run(_fetch_lesswrong_context(post_id))
                if lw_context and (lw_context.self_text or lw_context.top_comments):
                    from pipeline import _merge_source_context

                    story = _merge_source_context(
                        story, lw_context, article_body, prefer_longer_comments=True
                    )
                    runtime.db.upsert_story(story)

        if (
            article_body is None
            and story.url
            and not story.source.startswith("rss_reddit_")
            and story.source != "rss_lesswrong_com"
            and len(story.self_text) < 500
        ):
            from pipeline import _is_fetchable_article_url

            if _is_fetchable_article_url(story.url):
                from pipeline import (
                    _article_failure_retry_time,
                    compose_story_text,
                )

                result = asyncio.run(_fetch_article_body_with_result(story.url))
                if result.body:
                    article_body = result.body[:ARTICLE_BODY_CHAR_LIMIT]
                    new_text = compose_story_text(
                        story.title,
                        story.self_text,
                        story.top_comments,
                        article_body,
                    )
                    updated_story = replace(
                        story,
                        article_body=article_body,
                        text_content=new_text,
                    )
                    runtime.db.upsert_story(updated_story)
                    runtime.db.clear_article_fetch_failure(story.id)

                    embedder = runtime.embedder
                    if embedder is not None:
                        model_version = "all-MiniLM-L6-v2|mean|norm|256"
                        new_vec = embedder.encode([new_text])[0]
                        new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
                        runtime.db.upsert_embedding(
                            story.id, model_version, new_hash, new_vec
                        )

                    story = updated_story
                else:
                    now_ts = time.time()
                    previous = runtime.db.get_article_fetch_failure(story.id)
                    previous_count = int(previous["failure_count"]) if previous else 0
                    failure_count = previous_count + 1
                    permanent = (
                        result.permanent
                        or (result.error == "empty_extraction" and failure_count >= 3)
                        or (
                            result.error is not None
                            and result.error in ("http_401", "http_403")
                            and failure_count >= 3
                        )
                    )
                    next_retry_at = (
                        now_ts + 3650 * 86400
                        if permanent
                        else _article_failure_retry_time(failure_count, now_ts)
                    )
                    runtime.db.record_article_fetch_failure(
                        story.id,
                        story.url or "",
                        status=result.status,
                        error=result.error,
                        permanent=permanent,
                        next_retry_at=next_retry_at,
                    )

        cache_key = _tldr_cache_key(
            title=story.title,
            self_text=story.self_text or "",
            top_comments=story.top_comments or "",
            article_body=article_body or "",
        )
        cached_tldr = runtime.db.get_tldr_cache(story.id, cache_key)
        if cached_tldr:
            logging.info(
                "tldr_detail story_id=%s result=post_enrich_cache_hit cache_key=%s",
                story.id,
                cache_key[:12],
            )
            return _flask_json_response(
                {"ok": True, "tldr": cached_tldr, "cached": True}
            )

        result = asyncio.run(
            generate_detailed_tldr(
                story.title,
                self_text=story.self_text or "",
                top_comments=story.top_comments or "",
                article_body=article_body or "",
            )
        )
        if result.kind == "no_content":
            return _flask_json_response(
                {
                    "ok": True,
                    "tldr": "No article body or discussion available to summarize for this story.",
                    "cached": False,
                }
            )
        if result.kind == "llm_error":
            logging.warning(
                "tldr_detail story_id=%s result=llm_error cache_key=%s status=%s error=%s",
                story.id,
                cache_key[:12],
                result.error_status,
                result.error_text,
            )
            fallback = _stale_tldr_fallback("llm_error")
            if fallback:
                return fallback
            if result.error_status == 429:
                error = "Rate limit exceeded. Please try again in a moment."
            else:
                error = "Failed to generate TLDR. Please try again later."
            return _flask_json_response(
                {"error": error}, status=HTTPStatus.SERVICE_UNAVAILABLE
            )
        runtime.db.upsert_tldr_cache(story.id, cache_key, result.tldr)
        logging.info(
            "tldr_detail story_id=%s result=generated cache_key=%s",
            story.id,
            cache_key[:12],
        )
        return _flask_json_response({"ok": True, "tldr": result.tldr, "cached": False})
    except Exception:
        logging.exception("Error handling tldr-detail")
        return _flask_json_response(
            {"error": "Internal error"}, status=HTTPStatus.INTERNAL_SERVER_ERROR
        )


def _invalid_version_response() -> Response:
    return _flask_json_response(
        {"error": "Invalid version"}, status=HTTPStatus.BAD_REQUEST
    )


def _parse_non_negative_version(raw: str) -> int | None:
    try:
        version = int(raw)
    except ValueError:
        return None
    if version < 0:
        return None
    return version


def _handle_flask_ranking_ready(runtime: type[Handler]) -> Response:
    user = _flask_user(runtime)
    if not user:
        return _flask_json_response(
            {"error": "No session"}, status=HTTPStatus.UNAUTHORIZED
        )

    raw_min_versions = request.args.getlist("min_version")
    raw_legacy_versions = request.args.getlist("version")
    if raw_min_versions and raw_legacy_versions:
        return _invalid_version_response()
    raw_versions = raw_min_versions or raw_legacy_versions
    if len(raw_versions) != 1:
        return _invalid_version_response()

    min_version = _parse_non_negative_version(raw_versions[0])
    if min_version is None:
        return _invalid_version_response()

    raw_target_versions = request.args.getlist("target_version")
    if len(raw_target_versions) > 1:
        return _invalid_version_response()
    target_version = min_version
    if raw_target_versions:
        parsed_target = _parse_non_negative_version(raw_target_versions[0])
        if parsed_target is None:
            return _invalid_version_response()
        target_version = parsed_target

    current_version = runtime._dashboard_version(user.id)
    cached = runtime._dashboard_cache.get(f"dashboard_{user.id}")
    cached_version = cached[2] if cached is not None else None
    ready = cached is not None and cached[2] >= min_version
    ready_version = cached_version if ready else None
    warm_version = current_version
    if (cached_version is None or cached_version < warm_version) and (
        current_version >= min_version
    ):
        runtime._trigger_warm(user, warm_version)

    return _flask_json_response(
        {
            "ok": True,
            "ready": ready,
            "ready_version": ready_version,
            "min_version": min_version,
            "target_version": target_version,
            "current_version": current_version,
            "cached_version": cached_version,
        }
    )


def create_app(runtime: type[Handler] = Handler) -> Flask:
    app = Flask(__name__)

    @app.get("/u/<path:token>")
    def profile_link(token: str) -> ResponseReturnValue:
        token = token.strip("/")
        if not token:
            return Response(status=HTTPStatus.BAD_REQUEST)

        quota = _acquire_profile_link_quota(runtime)
        if not quota.allowed:
            return _flask_rate_limit_response(
                "Too many profile-link attempts. Please try again later.",
                quota.retry_after_seconds,
            )

        user = runtime.db.get_user_by_token(token)
        if not user:
            return Response(status=HTTPStatus.NOT_FOUND)

        response = redirect("../", code=302)
        response.set_cookie(
            "hn_token",
            user.token,
            max_age=31536000,
            path="/",
            samesite="Lax",
            httponly=True,
        )
        return response

    @app.get("/api/user")
    def api_user() -> ResponseReturnValue:
        user = _flask_user(runtime)
        if user:
            return _flask_json_response({"user_id": user.id, "token": user.token})
        return _flask_json_response(
            {"error": "No session"}, status=HTTPStatus.UNAUTHORIZED
        )

    @app.get("/api/ranking-ready")
    def ranking_ready() -> ResponseReturnValue:
        return _handle_flask_ranking_ready(runtime)

    @app.get("/")
    @app.get("/index.html")
    def dashboard() -> ResponseReturnValue:
        user = _flask_user(runtime)
        if not user:
            quota = _acquire_session_create_quota(runtime)
            if not quota.allowed:
                return _flask_rate_limit_response(
                    "Too many new demo sessions from this address. Please try again later.",
                    quota.retry_after_seconds,
                )

            token = secrets.token_hex(16)
            user = runtime.db.create_user(token)
            html = runtime._render_dashboard_for_user(user)
            return _no_cache_dashboard_response(
                html, user=user, set_session_cookie=True
            )

        html = runtime._render_dashboard_for_user(user)
        return _no_cache_dashboard_response(html)

    @app.route("/api/feedback", methods=["POST"], provide_automatic_options=False)
    def feedback() -> ResponseReturnValue:
        cross_site_response = _flask_cross_site_post_response()
        if cross_site_response is not None:
            return cross_site_response
        return _handle_flask_feedback(runtime)

    @app.route("/api/tldr-detail", methods=["POST"], provide_automatic_options=False)
    def tldr_detail() -> ResponseReturnValue:
        cross_site_response = _flask_cross_site_post_response()
        if cross_site_response is not None:
            return cross_site_response
        return _handle_flask_tldr_detail(runtime)

    @app.route("/api/feedback", methods=["OPTIONS"])
    @app.route("/api/tldr-detail", methods=["OPTIONS"])
    def api_options() -> Response:
        response = Response(status=HTTPStatus.NO_CONTENT)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    return app


def regen_loop(config: Config, event: threading.Event, db: Database) -> None:
    logging.info("Starting background regeneration loop...")
    embedder = Handler.embedder
    if config.regen_initial_delay_seconds > 0:
        logging.info(
            "Deferring first regen for %ds (avoid contention with first warm)",
            config.regen_initial_delay_seconds,
        )
        time.sleep(config.regen_initial_delay_seconds)
        event.set()
    while True:
        # Wait on event or timeout
        triggered = event.wait(timeout=config.regen_interval_seconds)
        if triggered:
            event.clear()
            # Debounce click storms
            time.sleep(2)

        logging.info("Regeneration triggered. Fetching candidates...")
        try:
            from pipeline import fetch_candidates_only

            asyncio.run(fetch_candidates_only(config, db, embedder=embedder))
            Handler._rebuild_cold_deck()
            Handler._bump_all_cached_versions()

            if Handler._cold_stories:
                t = threading.Thread(
                    target=lambda: Handler._warm_background_tasks(
                        list(Handler._cold_stories),
                        db,
                        embedder,
                        config,
                        per_combo=config.tldr_prefetch_per_combo,
                    ),
                    daemon=True,
                )
                t.start()
            logging.info("Regeneration complete.")
        except Exception as e:
            logging.exception("Background regeneration failed: %r", e)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    load_env()
    config = Config.load()
    db = Database(config.db_path)

    regen_event = threading.Event()
    from pipeline import Embedder

    embedder = Embedder(
        config.onnx_model_dir,
        batch_size=config.embedding_batch_size,
        ort_variant=config.embedding_ort_variant,
    )
    Handler.config = config
    Handler.db = db
    Handler.embedder = embedder
    Handler.regen_event = regen_event
    Handler._rebuild_cold_deck()

    # Start regen thread
    t = threading.Thread(target=regen_loop, args=(config, regen_event, db), daemon=True)
    t.start()

    # Start HTTP server
    server_host = "127.0.0.1"
    app = create_app(Handler)
    logging.info("Serving on http://%s:%d", server_host, config.server_port)
    try:
        app.run(
            host=server_host,
            port=config.server_port,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        db.close()


if __name__ == "__main__":
    main()
