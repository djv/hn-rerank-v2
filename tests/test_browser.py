"""The dashboard page in a real headless browser, against a throwaway server.

The Node tests in test_client_js.py run single client functions against
stubs; this drives the whole page: summaries, votes, undo, a failed vote,
the ranked-deck poll, sort and time-window changes and keys. Opt-in (`-m browser`); needs
the `browser` dependency group and a system Chrome/Chromium:

    uv run --group browser pytest tests/test_browser.py -m browser
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from werkzeug.serving import make_server

import pipeline
import server
from database import Database, Story
from pipeline import Config, RankedStory, WindowDeck, assemble_window_deck
from server import Handler, TldrResult, create_app

pytestmark = pytest.mark.browser

sync_api = pytest.importorskip("playwright.sync_api")

STORY_COUNT = 40
BASE_ID = 10**9  # never a real HN id


def _chrome() -> str | None:
    explicit = os.environ.get("HN_BROWSER")
    if explicit:
        return explicit
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _deck(ranked: list[RankedStory]) -> WindowDeck:
    return assemble_window_deck(ranked, config=Config(), now=time.time())


def _ranked(db: Database) -> list[RankedStory]:
    now = int(time.time())
    ranked = []
    for i in range(1, STORY_COUNT + 1):
        archive = i > 28
        source = "hn" if i % 5 else "rss_example_com"
        story = Story(
            id=BASE_ID + i,
            title=f"Story {i} <b>not bold</b>",
            url=f"https://example.com/{i}",
            score=500 - i * 7,
            time=now - (200 if archive else 1) * 86400 - i * 3600,
            text_content="text",
            source=source,
            discussion_url=f"https://news.ycombinator.com/item?id={BASE_ID + i}",
            article_body=f"Body of story {i}. " * 20,
        )
        db.upsert_story(story)
        ranked.append(RankedStory(story, score=1.0 - i / 50, best_match_title=""))
    return ranked


@pytest.fixture
def dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A dashboard whose ranking is instant and fake and whose summaries
    come from a fake LLM; nothing leaves the machine."""
    db = Database(str(tmp_path / "browser.db"))
    ranked = _ranked(db)

    def fake_rank(
        db: Database, config: Config, embedder: object, user_id: int, **_: object
    ) -> WindowDeck:
        voted = pipeline._voted_story_ids(db, user_id)
        return _deck(
            [
                replace(r, score=1.0 - r.score)  # the personal ranking inverts
                for r in reversed(ranked)
                if r.story.id not in voted
            ]
        )

    async def fake_llm(title: str, **_: str) -> TldrResult:
        return TldrResult(kind="ok", tldr=f"### Article\n- summary of {title}")

    async def no_fetch(*_: Any, **__: Any) -> None:
        raise AssertionError("the browser test must not fetch anything")

    monkeypatch.setattr(pipeline, "fast_rerank_for_user", fake_rank)
    monkeypatch.setattr(pipeline, "fetch_story", no_fetch)
    monkeypatch.setattr(server, "generate_detailed_tldr", fake_llm)
    monkeypatch.setattr(server, "_fetch_article_body_with_result", no_fetch)

    class Runtime(Handler):
        pass

    Runtime.config = Config(
        db_path=db.db_path,
        dashboard_warm_idle_seconds=0.2,
        article_fetch_max_per_run=0,
        tldr_prefetch_per_view=0,
        tldr_prefetch_stale_per_run=0,
    )
    Runtime.db = db
    # The fake ranking never embeds.
    Runtime.embedder = None  # ty: ignore[invalid-assignment]
    Runtime.regen_event = threading.Event()
    Runtime._cold_deck = _deck(ranked)
    Runtime._decks = {}
    Runtime._dashboard_versions = {}
    Runtime._scheduler = None
    Runtime._feedback_warm_counts = {}
    Runtime._feedback_warm_guard = threading.Lock()
    Runtime.reset_public_demo_limiter()
    monkeypatch.setattr(
        Runtime, "_collect_after_warm_attempt", classmethod(lambda cls: None)
    )

    httpd = make_server("127.0.0.1", 0, create_app(Runtime), threaded=True)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/"
    scheduler = Runtime.__dict__.get("_scheduler")
    if scheduler is not None:
        scheduler.clear_pending()
        scheduler.wait_idle(3.0)
    httpd.shutdown()
    db.close()


@pytest.fixture
def page(dashboard: str) -> Iterator[Any]:
    chrome = _chrome()
    if chrome is None:
        if os.environ.get("CI"):
            pytest.fail("no Chrome/Chromium on PATH (set HN_BROWSER)")
        pytest.skip("no Chrome/Chromium on PATH (set HN_BROWSER)")
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.set_default_timeout(5000)
        problems: list[str] = []
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: (
                problems.append(f"console: {m.text}") if m.type == "error" else None
            ),
        )
        page.on(
            "response",
            lambda r: (
                problems.append(f"{r.status} {r.url}")
                if r.status >= 400 and not r.url.endswith("/favicon.ico")
                else None
            ),
        )
        # Only the throwaway server may be reached.
        page.route(
            "**/*",
            lambda route: (
                route.continue_()
                if route.request.url.startswith(dashboard)
                else route.abort()
            ),
        )
        page.goto(dashboard)
        page.problems = problems
        yield page
        browser.close()


def _state(page: Any) -> dict[str, Any]:
    return page.evaluate(
        """() => ({
            active: activeId,
            head: visible.map(s => s.id),
            rated: [...rated],
            ready: feed.ready,
            sort: currentSort,
            window: currentWindow,
            counts: [...document.querySelectorAll('[data-vote-count]')]
                .map(e => e.textContent).join('/'),
            toast: toastEl.hidden ? '' : toastEl.textContent,
        })"""
    )


def test_dashboard_page_end_to_end(page: Any) -> None:
    page.click("[data-dismiss-tip]")
    page.wait_for_function(
        "() => activeCard()?.querySelector('.tldr-detail-content h3, .tldr-detail-content details')"
    )
    start = _state(page)
    assert len(start["head"]) == 12 and start["active"] == start["head"][0]
    # Titles are text, never markup.
    assert "<b>not bold</b>" in page.locator(".story-card").first.inner_text()
    assert page.locator(".story-card b").count() == 0

    # Votes hide the story and advance; the feed goes stale; undo restores.
    page.keyboard.press("1")
    page.keyboard.press("3")
    voted = _state(page)
    assert len(voted["rated"]) == 2 and voted["active"] not in voted["rated"]
    page.evaluate("() => voteChain")
    assert not _state(page)["ready"]
    page.keyboard.press("u")
    page.evaluate("() => voteChain")
    assert len(_state(page)["rated"]) == 1

    # The server re-ranks after the vote pause; polling picks the deck up.
    deadline = time.monotonic() + 5
    while not _state(page)["ready"] and time.monotonic() < deadline:
        page.wait_for_timeout(100)
        page.evaluate("async () => { await pollFeedVersion(); await reloadInFlight; }")
    ranked = _state(page)
    assert ranked["ready"]
    assert ranked["head"][0] == BASE_ID + 28  # highest once inverted
    assert not set(ranked["rated"]) & set(ranked["head"])

    # A sort change shows its head.
    page.evaluate("() => setFilter('sort', 'popular')")
    popular = _state(page)
    assert popular["active"] == popular["head"][0]
    page.evaluate("() => setFilter('sort', 'recommended')")

    # A vote the server never confirms is reverted, with a toast.
    page.evaluate(
        """() => {
            const real = window.fetch;
            window.__realFetch = real;
            window.fetch = (u, i) => String(u).includes('/api/feedback')
                ? Promise.reject(new TypeError('offline')) : real(u, i);
        }"""
    )
    before = _state(page)
    page.keyboard.press("1")
    page.evaluate("() => voteChain.catch(() => {})")
    failed = _state(page)
    assert failed["rated"] == before["rated"]
    assert failed["counts"] == before["counts"]
    assert failed["active"] == before["active"]
    assert failed["toast"].startswith("Vote not confirmed")
    page.evaluate("() => { window.fetch = window.__realFetch; }")

    # Keys: j moves, l cycles the sort, ? opens help and Escape closes it.
    page.keyboard.press("j")
    moved = _state(page)
    assert moved["active"] == moved["head"][1]
    page.keyboard.press("l")
    assert _state(page)["sort"] != "recommended"
    page.keyboard.press("Shift+Slash")
    assert page.is_visible("#first-time-tip")
    page.keyboard.press("Escape")
    assert page.is_hidden("#first-time-tip")

    # Time windows: a moves to the next one (1m holds 1w's stories too), the
    # picker jumps; each window shows only its own stories.
    page.keyboard.press("d")
    page.wait_for_function("() => currentWindow === '1m' && !feed.loading")
    month = _state(page)
    assert page.input_value("#window-select") == "1m"
    assert month["head"] and all(sid <= BASE_ID + 28 for sid in month["head"])
    page.select_option("#window-select", "archive")
    page.wait_for_function(
        "() => currentWindow === 'archive' && !feed.loading && visible.length > 0"
    )
    archive = _state(page)
    assert all(sid > BASE_ID + 28 for sid in archive["head"])
    assert archive["active"] == archive["head"][0]
    page.select_option("#window-select", "12h")
    page.wait_for_function("() => currentWindow === '12h' && !feed.loading")
    assert _state(page)["head"] == []  # nothing that young: an empty view
    assert "Nothing left in this view" in page.inner_text("#queue-loading")

    assert page.problems == []
