from __future__ import annotations

import os
import asyncio
import hashlib
import html
import json
import logging
import re
import secrets
import sys
import threading
import time

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import feedparser
import trafilatura
from bs4 import BeautifulSoup
import httpx

from dataclasses import dataclass, replace
from database import Database, User
from pipeline import Config, Embedder, is_hn_source
from reddit_limiter import limiter as reddit_limiter
from http_fetch import fetch_with_urllib_fallback

ARTICLE_BODY_CHAR_LIMIT = 15_000
SELF_TEXT_PROMPT_CHAR_LIMIT = 8_000
COMMENT_PROMPT_CHAR_LIMIT = 12_000
REDDIT_COMMENTS_CACHE_CHAR_LIMIT = 10_000
REDDIT_COMMENT_LIMIT = 40
REDDIT_RSS_USER_AGENT = "hn-rewrite/1.0 personal RSS reader; contact: local dashboard"
TLDR_PROMPT_VERSION = "detail-v4"
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROMPT_CACHE: dict[str, str] = {}


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
    normalized = re.sub(r"(\S)\s+-\s+(?=\S)", r"\1\n- ", normalized)
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
    error: str = ""
    permanent: bool = False


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
                    html = resp.text
                elif resp.status_code in (403, 503) and attempt == 0:
                    status, html, _headers = await fetch_with_urllib_fallback(
                        client, url, headers
                    )
                    if status != 200:
                        return ArticleFetchResult(
                            status=status,
                            error=f"http_{status}",
                        )
                else:
                    return ArticleFetchResult(
                        status=resp.status_code,
                        error=f"http_{resp.status_code}",
                        permanent=resp.status_code in (404, 410),
                    )
        except Exception as e:
            return ArticleFetchResult(error=type(e).__name__)

        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text and len(text) > 200:
            return ArticleFetchResult(body=text[:ARTICLE_BODY_CHAR_LIMIT], status=200)

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        for tag in soup.find_all(["article", "main"]):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return ArticleFetchResult(
                    body=text[:ARTICLE_BODY_CHAR_LIMIT],
                    status=200,
                )
        text = soup.get_text(separator=" ", strip=True)
        if len(text) > 200:
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
    query = (
        '{ post(input: { selector: { _id: "%s" } }) { result { _id '
        "commentCount baseScore contents { html } } } "
        'comments(input: { terms: { view: "postCommentsTop", '
        'postId: "%s" } }) { results { _id author baseScore '
        "htmlBody postedAt } } }"
    ) % (post_id, post_id)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://www.lesswrong.com/graphql",
                json={"query": query},
            )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception as e:
        logging.error(f"LessWrong fetch failed: {e}")
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
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        for attempt in range(3):
            resp = await client.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            if resp.status_code in (429, 503) and attempt < 2:
                delay = _parse_retry_after(
                    resp.headers.get("Retry-After"), default=2**attempt
                )
                await asyncio.sleep(delay)
                continue
            break
        return f"Error from LLM Provider: HTTP {resp.status_code} - {resp.text}"


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
) -> str:
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
        return "Error: LLM API key not configured in environment."

    article_section = ""
    if self_text:
        article_section += f"Author's text:\n{self_text[:SELF_TEXT_PROMPT_CHAR_LIMIT]}"
    if article_body:
        article_section += (
            f"\n\nArticle body:\n{article_body[:ARTICLE_BODY_CHAR_LIMIT]}"
        )
    comments_section = top_comments[:COMMENT_PROMPT_CHAR_LIMIT]

    if not article_section and not top_comments:
        return "No article body or discussion available to summarize for this story."

    if article_section and comments_section:
        article_prompt = _load_prompt("article_v4.txt").format(
            title=title, article_section=article_section
        )
        discussion_prompt = _load_prompt("discussion_v4.txt").format(
            title=title, comments_section=comments_section
        )
        try:
            article_result, discussion_result = await asyncio.gather(
                _call_llm_chat(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    prompt=article_prompt,
                    max_tokens=900,
                ),
                _call_llm_chat(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    prompt=discussion_prompt,
                    max_tokens=900,
                ),
            )
            if article_result.startswith("Error"):
                return article_result
            if discussion_result.startswith("Error"):
                return discussion_result
            article_result = _normalize_tldr_markdown(article_result)
            discussion_result = _normalize_tldr_markdown(discussion_result)
            return (
                f"### Article\n{article_result}\n\n### Discussion\n{discussion_result}"
            )
        except Exception as e:
            return f"Error executing LLM call: {str(e)}"

    article_section_str = ""
    if self_text:
        article_section_str += (
            f"\n\nAuthor's text:\n{self_text[:SELF_TEXT_PROMPT_CHAR_LIMIT]}"
        )
    if article_body:
        article_section_str += (
            f"\n\nArticle body:\n{article_body[:ARTICLE_BODY_CHAR_LIMIT]}"
        )
    comments_section_str = top_comments[:COMMENT_PROMPT_CHAR_LIMIT]

    # Article-only: no Discussion/Consensus mention in prompt at all
    if article_section_str and not comments_section_str:
        prompt = _load_prompt("article_only_v4.txt").format(
            title=title, article_section=article_section_str
        )
    elif comments_section_str and not article_section_str:
        prompt = _load_prompt("discussion_only_v4.txt").format(
            title=title, comments_section=comments_section_str
        )
    else:
        content_section = f"Title: {title}"
        if self_text:
            content_section += (
                f"\n\nAuthor's text:\n{self_text[:SELF_TEXT_PROMPT_CHAR_LIMIT]}"
            )
        if article_body:
            content_section += (
                f"\n\nArticle body:\n{article_body[:ARTICLE_BODY_CHAR_LIMIT]}"
            )
        if top_comments:
            content_section += (
                f"\n\nComments:\n{top_comments[:COMMENT_PROMPT_CHAR_LIMIT]}"
            )

        prompt = _load_prompt("combined_v4.txt").format(content_section=content_section)

    try:
        result = await _call_llm_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=2000,
        )
        return _normalize_tldr_markdown(result)
    except Exception as e:
        return f"Error executing LLM call: {str(e)}"


SKELETON_HTML = b"""<!DOCTYPE html>
<html><head><meta http-equiv="refresh" content="1"></head>
<body><p>Loading your personalized dashboard...</p></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "HNRewrite/1.0"
    config: Config
    db: Database
    embedder: Embedder
    regen_event: threading.Event
    _dashboard_cache: dict[str, tuple[bytes, float, int]] = {}
    _dashboard_versions: dict[int, int] = {}
    _dashboard_versions_guard = threading.Lock()
    _render_locks: dict[int, threading.Lock] = {}
    _render_locks_guard = threading.Lock()
    _warmup_in_flight: set[tuple[int, int]] = set()
    _warmup_in_flight_guard = threading.Lock()
    _WARM_DEBOUNCE_S: float = 0.2

    def _get_user(self) -> User | None:
        """Extract user from cookie token."""
        cookie_header = self.headers.get("Cookie", "")
        for part in cookie_header.split(";"):
            kv = part.strip().split("=", 1)
            if len(kv) == 2 and kv[0] == "hn_token":
                return self.db.get_or_create_user(kv[1].strip())
        return None

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        # Token-based user session
        if path.startswith("/u/"):
            token = path[3:].strip("/")
            if token:
                user = self.db.get_or_create_user(token)
                self.send_response(302)
                self.send_header("Location", "../")
                self.send_header(
                    "Set-Cookie", f"hn_token={user.token}; Path=/; Max-Age=31536000"
                )
                self.end_headers()
                return
            self.send_error(HTTPStatus.BAD_REQUEST)
            return

        # User info API
        if path == "/api/user":
            user = self._get_user()
            if user:
                self._json_response({"user_id": user.id, "token": user.token})
            else:
                self._json_response({"error": "No session"}, status=401)
            return

        # Dashboard — dynamic render per-user
        if path in ("/", "/index.html"):
            user = self._get_user()
            if not user:
                token = secrets.token_hex(4)
                self.send_response(302)
                self.send_header("Location", f"u/{token}")
                self.send_header(
                    "Set-Cookie", f"hn_token={token}; Path=/; Max-Age=31536000"
                )
                self.end_headers()
                return

            # Render personalized dashboard
            html = self._render_dashboard(user)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(html)

            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _render_dashboard(self, user: User) -> bytes:
        """Render personalized dashboard for user. Uses short-lived cache."""
        return self.__class__._render_dashboard_for_user(user)

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

        # No cache → return skeleton, trigger warm
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
        key = (user.id, version)
        with cls._warmup_in_flight_guard:
            if key in cls._warmup_in_flight:
                return
            cls._warmup_in_flight.add(key)

        def warm() -> None:
            warm_start = time.perf_counter()
            try:
                time.sleep(cls._WARM_DEBOUNCE_S)
                with cls._dashboard_versions_guard:
                    if cls._dashboard_versions.get(user.id, 0) != version:
                        logging.info(
                            "dashboard_warm user_id=%s version=%s result=skipped_stale elapsed_ms=%.1f",
                            user.id,
                            version,
                            (time.perf_counter() - warm_start) * 1000,
                        )
                        return

                lock = cls._get_render_lock(user.id)
                with lock:
                    cached = cls._dashboard_cache.get(f"dashboard_{user.id}")
                    if cached and cached[2] == version:
                        return

                    from pipeline import fast_rerank_for_user, generate_dashboard_bytes

                    render_start = time.perf_counter()
                    final = fast_rerank_for_user(
                        cls.db, cls.config, cls.embedder, user.id
                    )
                    rank_ms = (time.perf_counter() - render_start) * 1000

                    html_start = time.perf_counter()
                    html = generate_dashboard_bytes(
                        final, cls.config, cls.db, user.id, user.token
                    )
                    html_ms = (time.perf_counter() - html_start) * 1000

                    cls._dashboard_cache[f"dashboard_{user.id}"] = (
                        html,
                        time.time(),
                        version,
                    )
                    cls._enforce_cache_cap()

                    logging.info(
                        "dashboard_warm user_id=%s version=%s result=completed rank_ms=%.1f html_ms=%.1f stories=%s",
                        user.id,
                        version,
                        rank_ms,
                        html_ms,
                        len(final),
                    )
            except Exception as e:
                logging.exception(
                    "Failed warming dashboard cache for user_id=%s: %s", user.id, e
                )
            finally:
                with cls._warmup_in_flight_guard:
                    cls._warmup_in_flight.discard(key)

        threading.Thread(target=warm, daemon=True).start()

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

    def do_POST(self) -> None:
        if self.path == "/api/feedback":
            user = self._get_user()
            if not user:
                self._json_response({"error": "No session"}, status=401)
                return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)

                story_id = data["story_id"]
                action = data["action"]

                if action == "clear":
                    self.db.delete_feedback(user.id, story_id)
                else:
                    self.db.upsert_feedback(
                        user.id,
                        story_id,
                        action,
                    )

                # Invalidate the user's dashboard cache and kick a warm for
                # the new version on every vote, so a subsequent refill cannot
                # serve an HTML deck that includes a story the user just voted
                # on. The previous "defer until queue low / every 5 votes"
                # gating left the cached HTML stale for up to ~9s per burst;
                # the SWR stale-hit path then re-injected already-voted
                # stories via refillQueue (the bug observed on 2026-06-28).
                # Bursty votes still produce N sequential warm threads per
                # user; the existing version-skip check in _trigger_warm
                # discards obsolete results, and the render lock serializes
                # them, so only the latest version lands in the cache.
                version = self._invalidate_dashboard_cache(user.id)
                self._trigger_warm(user, version)

                # Also trigger background regen for candidate updates
                self.regen_event.set()

                self._json_response({"ok": True, "ranking_refresh_queued": True})
            except Exception as e:
                logging.error(f"Error handling feedback: {e}")
                self._json_response({"error": str(e)}, status=400)
        elif self.path == "/api/tldr-detail":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)

                story_id = data["story_id"]

                story = self.db.get_story(story_id)
                if not story:
                    self._json_response(
                        {"error": "Story not found in database"}, status=404
                    )
                    return

                # 1. If HN story has comments but top_comments is empty, dynamically fetch them
                if (
                    is_hn_source(story.source)
                    and not story.top_comments
                    and (story.comment_count or 0) > 0
                ):
                    try:
                        from pipeline import fetch_story

                        async def do_fetch():
                            async with httpx.AsyncClient(timeout=15.0) as client:
                                return await fetch_story(client, story_id, self.db)

                        updated = asyncio.run(do_fetch())
                        if updated:
                            story = updated
                    except Exception as e:
                        logging.error(
                            f"Failed to dynamically fetch comments for TLDR: {e}"
                        )

                article_body = story.article_body or None

                if (
                    story.source.startswith("rss_reddit_")
                    and story.url
                    and (not story.self_text or not story.top_comments)
                ):
                    reddit_context = asyncio.run(_fetch_reddit_rss_context(story.url))
                    if reddit_context and (
                        reddit_context.self_text or reddit_context.top_comments
                    ):
                        from pipeline import compose_story_text

                        self_text = (
                            reddit_context.self_text
                            if len(reddit_context.self_text) > len(story.self_text)
                            else story.self_text
                        )
                        top_comments = (
                            reddit_context.top_comments
                            if len(reddit_context.top_comments)
                            > len(story.top_comments)
                            else story.top_comments
                        )
                        new_text = compose_story_text(
                            story.title,
                            self_text,
                            top_comments,
                            article_body or "",
                        )
                        story = replace(
                            story,
                            self_text=self_text,
                            top_comments=top_comments,
                            text_content=new_text,
                            discussion_url=story.discussion_url or story.url,
                            comment_count=story.comment_count
                            or reddit_context.comment_count
                            or None,
                            comment_count_at_fetch=max(
                                story.comment_count_at_fetch,
                                reddit_context.comment_count,
                            ),
                        )
                        self.db.upsert_story(story)

                if (
                    story.source == "rss_lesswrong_com"
                    and story.url
                    and (not story.self_text or not story.top_comments)
                ):
                    post_id = _extract_lesswrong_post_id(story.url)
                    if post_id:
                        lw_context = asyncio.run(_fetch_lesswrong_context(post_id))
                        if lw_context and (
                            lw_context.self_text or lw_context.top_comments
                        ):
                            from pipeline import compose_story_text

                            self_text = (
                                lw_context.self_text
                                if len(lw_context.self_text) > len(story.self_text)
                                else story.self_text
                            )
                            top_comments = (
                                lw_context.top_comments
                                if len(lw_context.top_comments)
                                > len(story.top_comments)
                                else story.top_comments
                            )
                            new_text = compose_story_text(
                                story.title,
                                self_text,
                                top_comments,
                                article_body or "",
                            )
                            story = replace(
                                story,
                                self_text=self_text,
                                top_comments=top_comments,
                                text_content=new_text,
                                discussion_url=story.discussion_url or story.url,
                                comment_count=(
                                    story.comment_count
                                    or lw_context.comment_count
                                    or None
                                ),
                                comment_count_at_fetch=max(
                                    story.comment_count_at_fetch,
                                    lw_context.comment_count,
                                ),
                                score=max(story.score, lw_context.score),
                            )
                            self.db.upsert_story(story)

                if (
                    article_body is None
                    and story.url
                    and not story.source.startswith("rss_reddit_")
                    and not story.source == "rss_lesswrong_com"
                    and not story.url.startswith("https://news.ycombinator.com")
                    and len(story.self_text) < 500
                ):
                    article_body = asyncio.run(_fetch_article_body(story.url))
                    if article_body:
                        article_body = article_body[:ARTICLE_BODY_CHAR_LIMIT]
                        from pipeline import compose_story_text

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
                        self.db.upsert_story(updated_story)
                        story = updated_story
                cache_key = _tldr_cache_key(
                    title=story.title,
                    self_text=story.self_text or "",
                    top_comments=story.top_comments or "",
                    article_body=article_body or "",
                )
                cached_tldr = self.db.get_tldr_cache(story.id, cache_key)
                if cached_tldr:
                    logging.info(
                        "tldr_detail story_id=%s result=cache_hit cache_key=%s",
                        story.id,
                        cache_key[:12],
                    )
                    self._json_response(
                        {"ok": True, "tldr": cached_tldr, "cached": True}
                    )
                    return

                tldr = asyncio.run(
                    generate_detailed_tldr(
                        story.title,
                        self_text=story.self_text or "",
                        top_comments=story.top_comments or "",
                        article_body=article_body or "",
                    )
                )
                if not tldr:
                    self._json_response(
                        {"error": "Failed to generate TLDR"}, status=500
                    )
                    return
                if tldr.startswith("Error"):
                    logging.warning(
                        "tldr_detail story_id=%s result=llm_error cache_key=%s",
                        story.id,
                        cache_key[:12],
                    )
                    self._json_response({"error": tldr}, status=503)
                    return
                if not tldr.startswith("No article body"):
                    self.db.upsert_tldr_cache(story.id, cache_key, tldr)
                    logging.info(
                        "tldr_detail story_id=%s result=generated cache_key=%s",
                        story.id,
                        cache_key[:12],
                    )
                self._json_response({"ok": True, "tldr": tldr, "cached": False})
            except Exception as e:
                logging.error(f"Error handling tldr-detail: {e}")
                self._json_response({"error": str(e)}, status=400)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        logging.info("%s - - %s" % (self.address_string(), format % args))


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
            Handler._bump_all_cached_versions()
            logging.info("Regeneration complete.")
        except Exception as e:
            logging.exception(f"Background regeneration failed: {e}")


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

    embedder = Embedder(config.onnx_model_dir)
    Handler.config = config
    Handler.db = db
    Handler.embedder = embedder
    Handler.regen_event = regen_event

    # Start regen thread
    t = threading.Thread(target=regen_loop, args=(config, regen_event, db), daemon=True)
    t.start()

    # Start HTTP server
    server = ThreadingHTTPServer(("0.0.0.0", config.server_port), Handler)
    logging.info("Serving on http://0.0.0.0:%d", config.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        server.server_close()
        db.close()


if __name__ == "__main__":
    main()
