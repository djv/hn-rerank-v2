from typing import Any
import pytest
import asyncio
import httpx
from server import (
    _fetch_article_body,
    _fetch_article_body_with_result,
)


@pytest.fixture(autouse=True)
def mock_asyncio_sleep(monkeypatch):
    async def mock_sleep(delay):
        pass

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)


_ARTICLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Article</title></head>
<body>
<article>
<h1>Test Headline</h1>
<p>This is the body of the test article. It contains enough text to
pass the 200-character minimum threshold required by the extractor.
We need to make sure this paragraph is long enough so trafilatura
and the BS4 fallback both consider it meaningful content.</p>
<p>Here is some additional text to pad out the length even further.
Traffic comes from many sources and the extractor needs enough
content to be confident it has found the real article body and not
some sidebar or footer text. This should be more than enough.</p>
</article>
<footer>Footer text that should be stripped.</footer>
</body>
</html>
"""

_JUNK_HTML_WITH_TEXT = """\
<!DOCTYPE html>
<html>
<body>
<div id="comments">
<p>This is a comment-like block of text that trafilatura may skip
because it is inside a generic div without article or main tags.
The text is long enough to be considered content but the structure
is too loose for trafilatura to extract confidently.</p>
<p>More text here to ensure we cross the 200-character boundary so
the BeautifulSoup fallback can pick it up when trafilatura gives up.
This simulates poorly-structured pages that still contain readable
content that we want to extract.</p>
</div>
</body>
</html>
"""

_EMPTY_HTML = """\
<!DOCTYPE html>
<html>
<body>
<p>Short.</p>
</body>
</html>
"""

_CHROME_HTML = """\
<!DOCTYPE html>
<html>
<body>
<script>script chrome that should not be visible</script>
<style>style chrome that should not be visible</style>
<nav>navigation chrome that should not be visible</nav>
<header>header chrome that should not be visible</header>
<aside>aside chrome that should not be visible</aside>
<main>
<p>This main article text should survive extraction. It is deliberately long
enough to pass the extractor threshold and to force the BeautifulSoup fallback
to return real content rather than page chrome. The exact wording matters less
than preserving the core article paragraph and removing navigation or footer
material from the final result.</p>
<p>Additional article text keeps the page comfortably over the extraction
threshold. This paragraph represents the body content that should remain
available to the TLDR generator after all chrome elements are removed.</p>
</main>
<footer>footer chrome that should not be visible</footer>
</body>
</html>
"""


async def _serve(handler, status=200, body=b"", delay=0.0):
    import asyncio
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    class TestHandler(BaseHTTPRequestHandler):
        _call_count = 0

        def do_GET(self):
            type(self)._call_count += 1
            if delay:
                asyncio.run(asyncio.sleep(delay))
            if callable(status):
                s, b = status(self)
            else:
                s, b = status, body
            self.send_response(s)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}/{handler}"
    try:
        result = await _fetch_article_body(url)
    finally:
        server.socket.shutdown(socket.SHUT_RDWR)
        server.shutdown()
    return result, TestHandler._call_count


@pytest.mark.asyncio
async def test_fetch_success():
    body = _ARTICLE_HTML.encode()
    result, calls = await _serve("success", 200, body)
    assert result is not None
    assert "Test Headline" in result
    assert "body of the test article" in result
    assert "Footer text" not in result
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_404():
    result, calls = await _serve("notfound", 404, b"")
    assert result is None
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_retry_on_503():
    state = {"attempts": 0}

    def status_handler(inst):
        state["attempts"] += 1
        if state["attempts"] == 1:
            return 503, b""
        return 200, _ARTICLE_HTML.encode()

    result, calls = await _serve("retry", status_handler)
    assert result is not None
    assert "Test Headline" in result
    assert calls == 2


@pytest.mark.asyncio
async def test_fetch_retry_exhausted():
    state = {"attempts": 0}

    def status_handler(inst):
        state["attempts"] += 1
        return 503, b""

    result, calls = await _serve("exhausted", status_handler)
    assert result is None
    assert calls == 2


@pytest.mark.asyncio
async def test_fetch_bs4_fallback():
    body = _JUNK_HTML_WITH_TEXT.encode()
    result, calls = await _serve("junk", 200, body)
    assert result is not None
    assert "comment-like block" in result
    assert "BeautifulSoup fallback" in result
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_empty_body():
    body = _EMPTY_HTML.encode()
    result, calls = await _serve("empty", 200, body)
    assert result is None
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_strips_chrome_tags():
    body = _CHROME_HTML.encode()
    result, calls = await _serve("chrome", 200, body)

    assert result is not None
    assert "main article text should survive" in result
    assert "script chrome" not in result
    assert "style chrome" not in result
    assert "navigation chrome" not in result
    assert "header chrome" not in result
    assert "aside chrome" not in result
    assert "footer chrome" not in result
    assert calls == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_fetch_real_urls():
    """Smoke test: fetch article bodies for real URLs."""
    urls = [
        "https://example.com",
        "https://news.ycombinator.com",
    ]
    for url in urls:
        result = await _fetch_article_body(url)
        # The test passes as long as no exception is thrown.
        if result is not None:
            assert len(result) >= 100, f"body too short ({len(result)} chars) for {url}"


@pytest.mark.asyncio
async def test_fetch_rejects_non_html_content_type(monkeypatch):
    """Content-Type guard: non-HTML types return non_html/permanent."""
    import server

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            return httpx.Response(
                200,
                headers={"content-type": "application/pdf"},
                text="%PDF-1.4\n\x00\x00\x00",
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", MockClient)

    result = await _fetch_article_body_with_result("https://example.com/doc.pdf")
    assert result.body is None
    assert result.error == "non_html"
    assert result.permanent is True


@pytest.mark.asyncio
async def test_fetch_allows_text_plain(monkeypatch):
    """Content-Type guard: text/plain is allowed."""
    import server

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=_ARTICLE_HTML,
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", MockClient)

    result = await _fetch_article_body_with_result("https://example.com/plain")
    assert result.body is not None
    assert result.error == ""


@pytest.mark.asyncio
async def test_fetch_rejects_uppercase_content_type(monkeypatch):
    """Content-Type guard: mixed-case is handled via .lower()."""
    import server

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            return httpx.Response(
                200,
                headers={"content-type": "Application/PDF"},
                text="%PDF",
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", MockClient)

    result = await _fetch_article_body_with_result("https://example.com/Doc.PDF")
    assert result.error == "non_html"
    assert result.permanent is True


@pytest.mark.asyncio
async def test_fetch_allows_missing_content_type(monkeypatch):
    """Content-Type guard: missing header falls through to extraction."""
    import server

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            return httpx.Response(200, headers={}, text=_ARTICLE_HTML)

    monkeypatch.setattr(server.httpx, "AsyncClient", MockClient)

    result = await _fetch_article_body_with_result("https://example.com/noct")
    assert result.body is not None
    assert result.error == ""
