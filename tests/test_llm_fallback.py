"""Primary/fallback LLM routing (LLM_FALLBACK_PROVIDER)."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import server
from llm_limiter import fallback_limiter, reset_fallback_limiters

ARTICLE = "Substantial article body. " * 30


@pytest.fixture
def muse_with_mistral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")


def fake_providers(
    monkeypatch: pytest.MonkeyPatch, ok: dict[str, bool], status: int | None = 503
) -> list[tuple[str, int]]:
    """Patch _call_llm_for_config; returns the (provider, max_tokens) log."""
    calls: list[tuple[str, int]] = []

    async def fake_call(cfg, *, prompt, max_tokens, on_usage=None):
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
        )

    monkeypatch.setattr(server, "_call_llm_for_config", fake_call)
    return calls


@pytest.mark.usefixtures("muse_with_mistral")
async def test_failed_primary_falls_back_per_section(monkeypatch) -> None:
    calls = fake_providers(monkeypatch, {"gospark": False, "mistral": True})

    result = await server.generate_detailed_tldr(
        "Story", top_comments="A comment.", article_body=ARTICLE
    )

    assert result.kind == "ok" and result.cacheable
    assert "### Article\n- mistral point" in result.tldr
    assert "### Discussion\n- mistral point" in result.tldr
    assert sorted(p for p, _ in calls) == ["gospark", "gospark", "mistral", "mistral"]
    # Each provider gets its own token cap: Muse adds reasoning headroom.
    assert {t for p, t in calls if p == "gospark"} == {
        t + 2000 for p, t in calls if p == "mistral"
    }


@pytest.mark.usefixtures("muse_with_mistral")
async def test_healthy_primary_never_calls_fallback(monkeypatch) -> None:
    calls = fake_providers(monkeypatch, {"gospark": True, "mistral": True})

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert result.kind == "ok" and "gospark point" in result.tldr
    assert [p for p, _ in calls] == ["gospark"]


async def test_without_fallback_the_primary_error_is_returned(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    calls = fake_providers(monkeypatch, {"gospark": False}, status=429)

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert result.kind == "llm_error"
    assert (result.error_status, result.error_provider) == (429, "gospark")
    assert [p for p, _ in calls] == ["gospark"]


@pytest.mark.usefixtures("muse_with_mistral")
async def test_both_failing_reports_the_fallback_error(monkeypatch) -> None:
    fake_providers(monkeypatch, {"gospark": False, "mistral": False}, status=402)

    result = await server.generate_detailed_tldr(
        "Story", top_comments="A comment.", article_body=ARTICLE
    )

    assert result.kind == "llm_error"
    assert (result.error_status, result.error_provider) == (402, "mistral")


def test_fallback_config(monkeypatch) -> None:
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
    with pytest.raises(ValueError, match="LLM_FALLBACK_PROVIDER"):
        server._llm_fallback_config()


@pytest.mark.usefixtures("muse_with_mistral")
async def test_cooling_primary_is_skipped(monkeypatch) -> None:
    calls = fake_providers(monkeypatch, {"gospark": True, "mistral": True})
    server.llm_limiter.on_429()

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert "mistral point" in result.tldr
    assert [p for p, _ in calls] == ["mistral"]


@pytest.mark.usefixtures("muse_with_mistral")
async def test_each_provider_uses_its_own_limiter(monkeypatch) -> None:
    seen: list[tuple[str, object]] = []

    async def fake_http(*, model, limiter, **_):
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


def test_retry_after_without_fallback_is_the_primary_cooldown(monkeypatch) -> None:
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    server.llm_limiter.on_429()
    assert server._llm_retry_after_seconds() == server.llm_limiter.retry_after_seconds
    assert server._llm_retry_after_seconds() > 0


@pytest.mark.usefixtures("muse_with_mistral")
async def test_primary_402_cools_the_primary_only(monkeypatch) -> None:
    fake_providers(monkeypatch, {"gospark": False, "mistral": True}, status=402)

    result = await server.generate_detailed_tldr("Story", article_body=ARTICLE)

    assert result.kind == "ok"
    assert server.llm_limiter.retry_after_seconds > 0
    assert fallback_limiter("mistral").retry_after_seconds == 0


@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    has_fallback=st.booleans(),
    cooling=st.booleans(),
    primary_ok=st.booleans(),
    fallback_ok=st.booleans(),
    status=st.sampled_from([None, 402, 429, 500, 503]),
)
def test_routing_property(
    monkeypatch, has_fallback, cooling, primary_ok, fallback_ok, status
) -> None:
    """The primary runs unless it cools with a fallback present; the fallback
    runs only when the primary was skipped or failed; the result is the
    last provider's."""
    monkeypatch.setenv("LLM_PROVIDER", "gospark")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "mistral" if has_fallback else "")
    server.llm_limiter.reset()
    reset_fallback_limiters()
    if cooling:
        server.llm_limiter.on_429()
    calls = fake_providers(
        monkeypatch, {"gospark": primary_ok, "mistral": fallback_ok}, status=status
    )

    result = asyncio.run(
        server._call_llm_with_fallback(
            server._llm_provider_config(),
            server._llm_fallback_config(),
            prompt="p",
            base_tokens=100,
        )
    )

    skip_primary = has_fallback and cooling
    expected = [] if skip_primary else ["gospark"]
    if has_fallback and (skip_primary or not primary_ok):
        expected.append("mistral")
    assert [p for p, _ in calls] == expected
    assert result.provider == expected[-1]
    assert result.ok == (fallback_ok if expected[-1] == "mistral" else primary_ok)
    if has_fallback and not cooling and not primary_ok and status == 402:
        assert server.llm_limiter.retry_after_seconds > 0
