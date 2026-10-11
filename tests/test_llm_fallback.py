"""Primary/fallback LLM routing (LLM_FALLBACK_PROVIDER)."""

from __future__ import annotations

import asyncio
import time
import httpx
from collections.abc import Callable, Iterator
from database import Database, Story
from pipeline import RankedStory
from pipeline.ranking import WindowViews

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import server
from llm_limiter import fallback_limiter, reset_fallback_limiters, reset_primary_breaker

ARTICLE = "Substantial article body. " * 30


@pytest.fixture
def muse_with_mistral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")


def fake_providers(
    monkeypatch: pytest.MonkeyPatch,
    ok: dict[str, bool],
    status: int | None = 503,
    transport_failure: bool = False,
) -> list[tuple[str, int]]:
    """Patch _call_llm_for_config; returns the (provider, max_tokens) log."""
    calls: list[tuple[str, int]] = []

    async def fake_call(
        cfg: server.LlmProviderConfig,
        *,
        prompt: str,
        max_tokens: int,
        on_usage: server.LlmUsageRecorder | None = None,
        fail_fast: bool = False,
    ) -> server.LlmChatResult:
        calls.append((cfg.provider, max_tokens))
        if ok[cfg.provider]:
            return server.LlmChatResult(
                content=f"- {cfg.provider} point", ok=True, provider=cfg.provider
            )
        return server.LlmChatResult(
            content=f"{cfg.provider} down",
            ok=False,
            status=status,
            provider=cfg.provider,
            transport_failure=transport_failure,
        )

    monkeypatch.setattr(server, "_call_llm_for_config", fake_call)
    return calls


@pytest.mark.usefixtures("muse_with_mistral")
async def test_failed_primary_falls_back_per_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[tuple[str, str], int] = {}

    async def fake_call(
        cfg: server.LlmProviderConfig,
        *,
        prompt: str,
        max_tokens: int,
        on_usage: server.LlmUsageRecorder | None = None,
        fail_fast: bool = False,
    ) -> server.LlmChatResult:
        section = "Article" if "Summarize the article" in prompt else "Discussion"
        calls[cfg.provider, section] = max_tokens
        return server.LlmChatResult(
            content=f"- {cfg.provider} point",
            ok=cfg.provider == "mistral",
            provider=cfg.provider,
        )

    monkeypatch.setattr(server, "_call_llm_for_config", fake_call)
    result = await server.generate_detailed_tldr(
        "Story", top_comments="A comment.", article_body="a" * 21000
    )
    assert result.kind == "ok" and result.cacheable
    assert "### Article\n- mistral point" in result.tldr
    assert "### Discussion\n- mistral point" in result.tldr
    assert calls == {
        ("gospark", "Article"): 2675,
        ("mistral", "Article"): 675,
        ("gospark", "Discussion"): 2450,
        ("mistral", "Discussion"): 450,
    }


@pytest.mark.usefixtures("muse_with_mistral")
async def test_healthy_primary_never_calls_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_providers(monkeypatch, {"gospark": True, "mistral": True})

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert result.kind == "ok" and "gospark point" in result.tldr
    assert [p for p, _ in calls] == ["gospark"]


async def test_without_fallback_the_primary_error_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    calls = fake_providers(monkeypatch, {"gospark": False}, status=429)

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert result.kind == "llm_error"
    assert (result.error_status, result.error_provider) == (429, "gospark")
    assert [p for p, _ in calls] == ["gospark"]


@pytest.mark.usefixtures("muse_with_mistral")
async def test_both_failing_reports_the_fallback_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_providers(monkeypatch, {"gospark": False, "mistral": False}, status=402)

    result = await server.generate_detailed_tldr(
        "Story", top_comments="A comment.", article_body=ARTICLE
    )

    assert result.kind == "llm_error"
    assert (result.error_status, result.error_provider) == (402, "mistral")


def test_fallback_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("LLM_MODEL", "custom-primary-model")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    assert server._llm_fallback_config() is None

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", " Mistral ")
    cfg = server._llm_fallback_config()
    assert cfg is not None
    assert (cfg.provider, cfg.api_key) == ("mistral", "mistral-key")
    assert cfg.model == "mistral-small-latest"  # LLM_MODEL is the primary's

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "gospark")
    assert server._llm_fallback_config() is None  # same as the primary

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral")
    monkeypatch.delenv("MISTRAL_API_KEY")
    assert server._llm_fallback_config() is None  # no key

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "nope")
    assert server._llm_fallback_config() is None
    with pytest.raises(ValueError, match="LLM_FALLBACK_PROVIDER"):
        server._validate_llm_config()


@pytest.mark.usefixtures("muse_with_mistral")
async def test_cooling_primary_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_providers(monkeypatch, {"gospark": True, "mistral": True})
    server.llm_limiter.on_429()

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert "mistral point" in result.tldr
    assert [p for p, _ in calls] == ["mistral"]


@pytest.mark.usefixtures("muse_with_mistral")
async def test_each_provider_uses_its_own_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, object]] = []

    async def fake_http(
        *, model: str, limiter: object, **_: object
    ) -> server.LlmChatResult:
        seen.append((model, limiter))
        return server.LlmChatResult(content="- x", ok=True)

    monkeypatch.setattr(server, "_call_llm_responses", fake_http)
    monkeypatch.setattr(server, "_call_llm_chat", fake_http)
    primary, fallback = server._llm_provider_config(), server._llm_fallback_config()
    assert fallback is not None

    first = await server._call_llm_once(primary, prompt="p", max_tokens=10)
    second = await server._call_llm_once(fallback, prompt="p", max_tokens=10)

    assert (first.provider, second.provider) == ("gospark", "mistral")
    assert seen[0][1] is server.llm_limiter
    assert seen[1][1] is fallback_limiter("mistral")
    assert seen[1][1] is not server.llm_limiter


@pytest.mark.usefixtures("muse_with_mistral")
def test_retry_after_waits_only_when_every_provider_cools() -> None:
    assert server._llm_retry_after_seconds() == 0
    server.llm_limiter.on_429()
    assert server._llm_retry_after_seconds() == 0  # Mistral is free
    fallback_limiter("mistral").on_429()
    assert server._llm_retry_after_seconds() > 0


def test_retry_after_without_fallback_is_the_primary_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    server.llm_limiter.on_429()
    assert server._llm_retry_after_seconds() == server.llm_limiter.retry_after_seconds
    assert server._llm_retry_after_seconds() > 0


@pytest.mark.usefixtures("muse_with_mistral")
async def test_primary_402_cools_the_primary_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_providers(monkeypatch, {"gospark": False, "mistral": True}, status=402)

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert result.kind == "ok"
    assert server.primary_breaker.retry_after_seconds > 0
    assert server.llm_limiter.retry_after_seconds == 0
    assert fallback_limiter("mistral").retry_after_seconds == 0


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    clock = Clock()
    reset_primary_breaker(clock)
    return clock


def http_providers(
    monkeypatch: pytest.MonkeyPatch,
    primary: Callable[[httpx.Request], httpx.Response],
) -> tuple[list[str], list[float]]:
    calls: list[str] = []
    timeouts: list[float] = []
    real_client = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == httpx.URL(server._llm_provider_config().base_url).host:
            calls.append("gospark")
            return primary(request)
        calls.append("mistral")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "- Fallback point"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    def client(*, timeout: float) -> httpx.AsyncClient:
        timeouts.append(timeout)
        return real_client(timeout=timeout, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(server.httpx, "AsyncClient", client)
    return calls, timeouts


async def route() -> server.LlmChatResult:
    return await server._call_llm_with_fallback(
        server._llm_provider_config(),
        server._llm_fallback_config(),
        prompt="prompt",
        base_tokens=450,
    )


@pytest.mark.usefixtures("muse_with_mistral")
async def test_week_long_retry_after(
    monkeypatch: pytest.MonkeyPatch, clock: Clock
) -> None:
    calls, timeouts = http_providers(
        monkeypatch,
        lambda _: httpx.Response(
            429, headers={"Retry-After": "604800"}, json={"error": "GoUsageLimitError"}
        ),
    )
    assert (await route()).provider == "mistral"
    assert calls == ["gospark", "mistral"]
    assert server.primary_breaker.retry_after_seconds >= 604800
    assert server.llm_limiter.retry_after_seconds <= 120
    assert timeouts == [45.0, 45.0]
    assert (await route()).provider == "mistral"
    assert calls == ["gospark", "mistral", "mistral"]


@pytest.mark.usefixtures("muse_with_mistral")
@pytest.mark.parametrize("failure", [429, 503, 402, 401, 403, 500, "timeout"])
@pytest.mark.parametrize("api", ["responses", "chat"])
async def test_http_health_failure_is_one_attempt(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, failure: int | str, api: str
) -> None:
    if api == "chat":
        key, url, model, extra = server._LLM_PROVIDERS["gospark"]
        monkeypatch.setitem(
            server._LLM_PROVIDERS,
            "gospark",
            (key, url.replace("/responses", "/chat/completions"), model, extra),
        )

    def primary(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        assert isinstance(failure, int)
        return httpx.Response(failure, json={"error": "down"})

    async def no_sleep(delay: float) -> None:
        pytest.fail(f"primary slept {delay}")

    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)
    calls, _ = http_providers(monkeypatch, primary)
    assert (await route()).provider == "mistral"
    assert calls == ["gospark", "mistral"]
    assert server.primary_breaker.retry_after_seconds == 120
    assert (await route()).provider == "mistral"
    assert calls[-1] == "mistral" and calls.count("gospark") == 1


@pytest.mark.usefixtures("muse_with_mistral")
async def test_no_fallback_keeps_four_429_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER")
    calls, timeouts = http_providers(monkeypatch, lambda _: httpx.Response(429))

    async def acquire(*, estimated_tokens: int = 0, wait_allowed: bool = True) -> bool:
        assert wait_allowed
        return True

    monkeypatch.setattr(server.llm_limiter, "acquire", acquire)
    result = await route()
    assert result.status == 429
    assert calls == ["gospark"] * 4 and timeouts == [90.0]
    assert server.primary_breaker.retry_after_seconds == 0


@pytest.mark.usefixtures("muse_with_mistral")
async def test_acquire_race_fails_fast(
    monkeypatch: pytest.MonkeyPatch, clock: Clock
) -> None:
    calls, _ = http_providers(monkeypatch, lambda _: httpx.Response(200))
    real_acquire = server.llm_limiter.acquire

    async def acquire(*, estimated_tokens: int = 0, wait_allowed: bool = True) -> bool:
        server.llm_limiter.on_429()
        return await real_acquire(
            estimated_tokens=estimated_tokens, wait_allowed=wait_allowed
        )

    async def no_sleep(delay: float) -> None:
        pytest.fail(f"acquire slept {delay}")

    monkeypatch.setattr(server.llm_limiter, "acquire", acquire)
    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)
    assert (await route()).provider == "mistral"
    assert calls == ["mistral"]


def test_breaker_escalation_and_reset(clock: Clock) -> None:
    breaker = server.primary_breaker
    for delay in (120, 300, 900, 3600, 3600):
        lease = breaker.acquire()
        assert lease is not None
        breaker.complete(lease, success=False, health_failure="HTTP 402")
        assert breaker.retry_after_seconds == delay
        assert breaker.acquire() is None
        clock.now += delay
    lease = breaker.acquire()
    assert lease is not None
    breaker.complete(lease, success=True)
    lease = breaker.acquire()
    assert lease is not None
    breaker.complete(lease, success=False, health_failure="transport")
    assert breaker.retry_after_seconds == 120


@pytest.mark.usefixtures("muse_with_mistral")
async def test_half_open_single_http_probe(
    monkeypatch: pytest.MonkeyPatch, clock: Clock
) -> None:
    lease = server.primary_breaker.acquire()
    assert lease is not None
    server.primary_breaker.complete(lease, success=False, health_failure="transport")
    clock.now += 120
    started, release = asyncio.Event(), asyncio.Event()
    calls: list[str] = []
    real_client = httpx.AsyncClient

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == httpx.URL(server._llm_provider_config().base_url).host:
            calls.append("gospark")
            started.set()
            await release.wait()
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "- Primary point"}
                            ],
                        }
                    ],
                },
            )
        calls.append("mistral")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "- Fallback point"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kw: real_client(**kw, transport=httpx.MockTransport(handle)),
    )
    first = asyncio.create_task(route())
    await started.wait()
    second = await route()
    assert second.provider == "mistral"
    assert calls == ["gospark", "mistral"]
    assert server.primary_breaker.retry_after_seconds == 0  # half-open doesn't gate
    release.set()
    assert (await first).provider == "gospark"
    assert server.primary_breaker.retry_after_seconds == 0
    assert (await route()).provider == "gospark"


@pytest.mark.usefixtures("muse_with_mistral")
@pytest.mark.parametrize(
    "content,status",
    [("", "completed"), ("prose", "completed"), ("- Cut off", "incomplete")],
)
async def test_http_content_failure_never_opens(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, content: str, status: str
) -> None:
    calls, _ = http_providers(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": content}],
                    }
                ],
            },
        ),
    )
    assert (await route()).provider == "mistral"
    assert server.primary_breaker.retry_after_seconds == 0
    assert (await route()).provider == "mistral"
    assert calls == ["gospark", "mistral"] * 2


@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    outcomes=st.lists(
        st.tuples(st.integers(0, 7), st.integers(0, 4000)), min_size=1, max_size=30
    )
)
def test_routing_property(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[tuple[int, int]]
) -> None:
    """Sequences span outages, content failures and single-probe recovery."""
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    clock = Clock()
    reset_primary_breaker(clock)
    server.llm_limiter.reset()
    reset_fallback_limiters()
    for outcome, advance in outcomes:
        clock.now += advance
        status = (None, None, 402, 429, 401, 403, 500, None)[outcome]
        ok = outcome == 0
        open_before = server.primary_breaker.retry_after_seconds > 0
        calls = fake_providers(
            monkeypatch,
            {"gospark": ok, "mistral": True},
            status=status,
            transport_failure=outcome == 7,
        )
        result = asyncio.run(route())
        assert result.ok
        assert [p for p, _ in calls] == (
            ["mistral"] if open_before else ["gospark"] + ([] if ok else ["mistral"])
        )
        if not open_before:
            if ok or outcome == 1:
                assert server.primary_breaker.retry_after_seconds == 0
            else:
                assert server.primary_breaker.retry_after_seconds >= 120
        # When an open expires, only one lease may own the recovery probe.
        if server.primary_breaker.retry_after_seconds > 0:
            clock.now += server.primary_breaker.retry_after_seconds
            lease = server.primary_breaker.acquire()
            assert lease is not None and server.primary_breaker.acquire() is None
            server.primary_breaker.complete(lease, success=False)


def test_startup_rejects_unknown_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "load_env", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "bad-fallback")
    with pytest.raises(ValueError, match="LLM_FALLBACK_PROVIDER: bad-fallback"):
        server.main()
    monkeypatch.setenv("LLM_PROVIDER", "bad-primary")
    with pytest.raises(ValueError, match="LLM_PROVIDER: bad-primary"):
        server.main()


def test_missing_fallback_key_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "   ")
    server._validate_llm_config()
    assert "MISTRAL_API_KEY is empty" in caplog.text
    assert server._llm_fallback_config() is None


def test_bad_fallback_logs_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "distinct-bad-value")
    monkeypatch.setattr(server, "_bad_fallback_values", set())
    assert server._llm_fallback_config() is None
    assert server._llm_fallback_config() is None
    assert caplog.text.count("Unsupported LLM_FALLBACK_PROVIDER") == 1


@pytest.fixture
def memory_db() -> "Iterator[Database]":
    db = Database(":memory:")
    try:
        yield db
    finally:
        db.close()


def stored_story(db: "Database", *, comments: str = "") -> "Story":
    story = Story(
        id=1234,
        title="Routing story",
        url=None,
        score=20,
        time=int(time.time()),
        text_content=ARTICLE,
        source="hn",
        article_body=ARTICLE,
        top_comments=comments,
    )
    db.upsert_story(story)
    return story


@pytest.mark.usefixtures("muse_with_mistral")
@pytest.mark.parametrize("both_unavailable", [False, True])
async def test_prefetch_uses_fallback_or_skips(
    monkeypatch: pytest.MonkeyPatch,
    clock: Clock,
    memory_db: "Database",
    both_unavailable: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    story = stored_story(memory_db)
    deck = server.WindowDeck(
        {
            "1w": WindowViews(
                recommended=(RankedStory(story=story, score=1.0, best_match_title=""),)
            )
        }
    )
    lease = server.primary_breaker.acquire()
    assert lease is not None
    server.primary_breaker.complete(lease, success=False, health_failure="HTTP 402")
    if both_unavailable:
        fallback_limiter("mistral").on_429()
    calls, _ = http_providers(
        monkeypatch, lambda _: pytest.fail("unavailable primary called")
    )
    with caplog.at_level("INFO"):
        generated = await server._prefetch_tldrs_for_ranked(
            deck,
            memory_db,
            per_view=1,
            stagger_s=0,
        )
    assert generated == (0 if both_unavailable else 1)
    assert calls == ([] if both_unavailable else ["mistral"])
    if both_unavailable:
        assert "provider cooldown" in caplog.text


@pytest.mark.usefixtures("muse_with_mistral")
async def test_mixed_provider_result_is_cached(
    monkeypatch: pytest.MonkeyPatch,
    memory_db: "Database",
) -> None:
    story = stored_story(memory_db, comments="An interesting comment.")

    async def fake_call(
        cfg: server.LlmProviderConfig,
        *,
        prompt: str,
        max_tokens: int,
        on_usage: server.LlmUsageRecorder | None = None,
        fail_fast: bool = False,
    ) -> server.LlmChatResult:
        return server.LlmChatResult(
            content=f"- {cfg.provider} point",
            ok=cfg.provider == "mistral" or "Summarize the article" in prompt,
            provider=cfg.provider,
        )

    monkeypatch.setattr(server, "_call_llm_for_config", fake_call)
    result = await server.generate_detailed_tldr(
        story.title,
        article_body=story.article_body or "",
        top_comments=story.top_comments or "",
    )
    assert result.cacheable and result.kind == "ok"
    assert (
        result.tldr == "### Article\n- gospark point\n\n### Discussion\n- mistral point"
    )
    assert server._maybe_cache_tldr(memory_db, story.id, "key", result)
    assert memory_db.get_tldr_cache(story.id, "key") == result.tldr


@pytest.mark.usefixtures("muse_with_mistral")
def test_tap_primary_402_opens_only_primary(
    memory_db: "Database", clock: Clock
) -> None:
    story = stored_story(memory_db)
    result = server.TldrResult(
        kind="llm_error", error_status=402, error_provider="gospark"
    )
    reply = server._tldr_result_reply(memory_db, story.id, result)
    assert reply.status == 429
    assert server.primary_breaker.retry_after_seconds == 120
    assert server.llm_limiter.retry_after_seconds == 0
    assert fallback_limiter("mistral").retry_after_seconds == 0
    # Two simultaneous failed halves must not advance the escalation twice.
    server._tldr_result_reply(memory_db, story.id, result)
    assert server.primary_breaker.retry_after_seconds == 120


def test_cached_tap_survives_bad_fallback(
    monkeypatch: pytest.MonkeyPatch,
    memory_db: "Database",
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "not-a-provider")
    monkeypatch.setattr(server.Handler, "config", server.Config(), raising=False)
    monkeypatch.setattr(server.Handler, "db", memory_db, raising=False)
    story = stored_story(memory_db)
    user = memory_db.create_user("cached-test-token")
    key = server._tldr_cache_key(
        title=story.title,
        self_text="",
        top_comments="",
        article_body=ARTICLE,
        source="hn",
    )
    assert server._maybe_cache_tldr(
        memory_db, story.id, key, server.TldrResult(kind="ok", tldr="- Cached point")
    )
    client = server.create_app().test_client()
    client.set_cookie("hn_token", user.token)
    response = client.post("/api/tldr-detail", json={"story_id": story.id})
    assert response.status_code == 200
    assert response.get_json()["tldr"] == "- Cached point"


@pytest.mark.parametrize(
    "article_status,discussion_status,expected",
    [
        (429, 402, 402),
        (503, 429, 429),
        (None, 503, 503),
        (402, 429, 402),
    ],
)
@pytest.mark.parametrize("has_fallback", [True, False])
async def test_failed_halves_status_priority(
    monkeypatch: pytest.MonkeyPatch,
    article_status: int | None,
    discussion_status: int | None,
    expected: int,
    has_fallback: bool,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "key")
    monkeypatch.setenv("MISTRAL_API_KEY", "key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral" if has_fallback else "")
    if not has_fallback:
        original_status = article_status or discussion_status
        assert original_status is not None
        expected = original_status

    async def fake_call(
        cfg: server.LlmProviderConfig,
        *,
        prompt: str,
        max_tokens: int,
        on_usage: server.LlmUsageRecorder | None = None,
        fail_fast: bool = False,
    ) -> server.LlmChatResult:
        article = "Summarize the article" in prompt
        return server.LlmChatResult(
            content="failed",
            ok=False,
            status=article_status if article else discussion_status,
            provider="article-provider" if article else "discussion-provider",
        )

    monkeypatch.setattr(server, "_call_llm_for_config", fake_call)
    result = await server.generate_detailed_tldr(
        "Story", article_body=ARTICLE, top_comments="Comment"
    )
    assert result.error_status == expected
    assert result.error_provider == (
        "article-provider" if article_status == expected else "discussion-provider"
    )


def test_half_open_is_single_probe_across_threads(clock: Clock) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    breaker = server.primary_breaker
    lease = breaker.acquire()
    assert lease is not None
    breaker.complete(lease, success=False, health_failure="HTTP 402")
    clock.now += 120
    barrier = Barrier(2)

    def take_probe() -> int | None:
        barrier.wait(timeout=5)
        return breaker.acquire()

    with ThreadPoolExecutor(max_workers=2) as pool:
        probes = list(pool.map(lambda _: take_probe(), range(2)))
    assert sum(probe is not None for probe in probes) == 1
    lease = next(probe for probe in probes if probe is not None)
    breaker.complete(lease, success=True)
    assert breaker.acquire() is not None


def test_concurrent_failure_preserves_long_retry_after(clock: Clock) -> None:
    breaker = server.primary_breaker
    first = breaker.acquire()
    second = breaker.acquire()
    assert first is not None and second is not None
    breaker.complete(first, success=False, health_failure="HTTP 402")
    breaker.complete(
        second, success=False, health_failure="HTTP 429", retry_after_s=604800
    )
    assert breaker.retry_after_seconds == 604800
    clock.now += 604800
    probe = breaker.acquire()
    assert probe is not None
    breaker.complete(probe, success=False, health_failure="HTTP 402")
    assert breaker.retry_after_seconds == 300  # concurrent failures opened only once


@pytest.mark.usefixtures("muse_with_mistral")
async def test_empty_shaped_primary_falls_back_without_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_call(
        cfg: server.LlmProviderConfig,
        *,
        prompt: str,
        max_tokens: int,
        on_usage: server.LlmUsageRecorder | None = None,
        fail_fast: bool = False,
    ) -> server.LlmChatResult:
        calls.append(cfg.provider)
        return server.LlmChatResult(
            ok=True,
            content="\n  " if cfg.provider == "gospark" else "- Fallback point",
            provider=cfg.provider,
        )

    monkeypatch.setattr(server, "_call_llm_for_config", fake_call)
    assert (await route()).provider == "mistral"
    assert calls == ["gospark", "mistral"]
    assert server.primary_breaker.retry_after_seconds == 0
