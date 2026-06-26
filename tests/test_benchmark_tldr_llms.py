from __future__ import annotations

import json

import pytest

from scripts.benchmark_tldr_llms import (
    StoryRecord,
    _call_openrouter_chat,
    _count_bullets,
    _has_nested_lists,
    _sanitize_model_id,
    _source_bucket,
    _word_count,
    generate_tldr_for_story,
    load_partial,
    save_partial,
    score_compliance,
    select_sample,
)


# ── Source bucket grouping ──────────────────────────────────────────────


def test_source_bucket_hn():
    assert _source_bucket("hn") == "hn"


def test_source_bucket_rss():
    assert _source_bucket("rss_lobste_rs") == "rss"
    assert _source_bucket("rss_lesswrong_com") == "rss"
    assert _source_bucket("rss_reddit_machinelearning") == "rss"


def test_source_bucket_seed():
    assert _source_bucket("ch_seed") == "seed"
    assert _source_bucket("bq_seed") == "seed"


def test_source_bucket_other():
    assert _source_bucket("unknown_source") == "other"


# ── Model id sanitization ───────────────────────────────────────────────


def test_sanitize_model_id_slashes_and_colons():
    assert (
        _sanitize_model_id("google/gemma-4-26b-a4b-it:free")
        == "google_gemma-4-26b-a4b-it_free"
    )


def test_sanitize_model_id_already_clean():
    assert _sanitize_model_id("llama-3-70b") == "llama-3-70b"


# ── Sample selection ────────────────────────────────────────────────────


def test_select_sample_respects_limit(tmp_path):
    db_path = tmp_path / "test.db"
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stories (
            id INTEGER PRIMARY KEY, title TEXT, url TEXT, score INTEGER,
            time INTEGER, text_content TEXT, source TEXT, comment_count INTEGER,
            discussion_url TEXT, fetched_at REAL, comment_count_at_fetch INTEGER DEFAULT 0,
            self_text TEXT DEFAULT '', top_comments TEXT DEFAULT '',
            article_body TEXT DEFAULT ''
        )
    """)
    for i in range(8):
        conn.execute(
            "INSERT INTO stories (id, title, text_content, source, self_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (i + 1, f"HN Story {i + 1}", "Content", "hn", "Self text"),
        )
    for i in range(8, 14):
        conn.execute(
            "INSERT INTO stories (id, title, text_content, source, self_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (i + 1, f"RSS Story {i + 1}", "Content", "rss_lobste_rs", "Self text"),
        )
    for i in range(14, 22):
        conn.execute(
            "INSERT INTO stories (id, title, text_content, source, self_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (i + 1, f"Seed Story {i + 1}", "Content", "ch_seed", "Self text"),
        )
    conn.commit()
    conn.close()

    sample = select_sample(str(db_path), limit=10)
    assert len(sample) == 10
    assert all(isinstance(s, StoryRecord) for s in sample)


# ── OpenRouter call ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_openrouter_chat_success(monkeypatch):
    async def mock_post(cls, url, *, headers, json, **kw):
        class MockResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "Mock TLDR summary"}}]}

        return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)
    result = await _call_openrouter_chat(
        api_key="sk-test",
        model="test/model",
        prompt="Summarize",
        max_tokens=900,
    )
    assert result == "Mock TLDR summary"


@pytest.mark.asyncio
async def test_call_openrouter_chat_error(monkeypatch):
    async def mock_post(cls, url, *, headers, json, **kw):
        class MockResponse:
            status_code = 429
            text = "Rate limited"

            def json(self):
                return {"error": "rate_limited"}

        return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)
    result = await _call_openrouter_chat(
        api_key="sk-test",
        model="test/model",
        prompt="Summarize",
        max_tokens=900,
    )
    assert "Error: HTTP 429" in result


# ── Utility: word count, bullet count, nested lists ─────────────────────


def test_word_count_empty():
    assert _word_count("") == 0


def test_word_count_typical():
    assert _word_count("one two three four") == 4


def test_bullet_count_none():
    assert _count_bullets("plain text") == 0


def test_bullet_count_some():
    text = "- First\n- Second\n- Third"
    assert _count_bullets(text) == 3


def test_bullet_count_with_headings():
    text = "### Article\n- First\n\n### Discussion\n- Second"
    assert _count_bullets(text) == 2


def test_has_nested_lists_true():
    assert _has_nested_lists("  - Nested bullet")


def test_has_nested_lists_false():
    assert not _has_nested_lists("- Top level\n- Another")


# ── Format compliance scoring (unified path) ────────────────────────────


def test_score_compliance_perfect():
    tldr = (
        "### Article\n"
        "- **Key finding:** Great results\n"
        "- Second important point\n"
        "- Third bullet for completeness\n"
        "\n"
        "### Discussion\n"
        "- **Consensus:** People agree\n"
        "- Important nuance emerges\n"
    )
    result = score_compliance(tldr)
    assert result.passes, f"Expected no violations, got: {result.violations}"
    assert result.score == 1.0


def test_score_compliance_accepts_no_discussion():
    """Discussion section is optional; only ### Article is required."""
    tldr = (
        "### Article\n- Only point one\n- Only point two\n- Point three\n- Point four\n"
    )
    result = score_compliance(tldr)
    assert result.passes, f"Expected no violations, got: {result.violations}"
    assert result.score == 1.0


def test_score_compliance_missing_article_heading():
    tldr = "### Discussion\n- Comment\n- Another comment\n"
    result = score_compliance(tldr)
    assert not result.passes
    assert "missing_article_heading" in result.violations


def test_score_compliance_no_bullets():
    tldr = "### Article\nJust a paragraph without any bullets."
    result = score_compliance(tldr)
    assert not result.passes
    assert "no_bullets" in result.violations


def test_score_compliance_word_cap_exceeded():
    tldr = "### Article\n" + "- " + "word " * 250 + "\n"
    result = score_compliance(tldr)
    assert not result.passes
    assert any(v.startswith("word_count_") for v in result.violations)


def test_score_compliance_nested_lists():
    tldr = "### Article\n- Top bullet\n  - Nested bullet\n- Another top bullet\n"
    result = score_compliance(tldr)
    assert not result.passes
    assert "nested_lists" in result.violations


def test_score_compliance_nonempty_fail():
    result = score_compliance("Error: HTTP 500")
    assert not result.passes
    assert result.score == 0.0


def test_score_compliance_unexpected_format():
    tldr = "### Article\n- Valid bullet\norphan text without prefix\n- Another bullet\n"
    result = score_compliance(tldr)
    assert "unexpected_format" in str(result.violations)


def test_score_compliance_idempotent_check():
    tldr = (
        "### Article\n"
        "- Normalized line\n"
        "- Another bullet\n"
        "- Third point\n"
        "\n"
        "### Discussion\n"
        "- Good summary\n"
        "- Second point\n"
    )
    result = score_compliance(tldr)
    assert "normalization_not_idempotent" not in result.violations


# ── generate_tldr_for_story (unified path) ──────────────────────────────


@pytest.mark.asyncio
async def test_generate_tldr_for_story_success(monkeypatch):
    async def mock_call(*, api_key, model, prompt, max_tokens, base_url):
        return (
            "### Article\n"
            "- Article point one\n"
            "- Article point two\n"
            "- Article point three\n"
        )

    monkeypatch.setattr("scripts.benchmark_tldr_llms._call_openrouter_chat", mock_call)
    story = StoryRecord(
        id=1,
        title="Test Article",
        url="https://example.com",
        source="hn",
        self_text="Author wrote something",
        top_comments="User comment",
        article_body="Article body text",
        text_content="Full content",
    )
    result = await generate_tldr_for_story(
        story,
        api_key="sk-test",
        model="test/model",
        base_url="https://api.example.com/v1",
        rate_sleep=0.0,
    )
    assert result.status == "ok"
    assert "### Article" in result.tldr
    assert "### Article" in result.raw_response
    assert result.compliance is not None


@pytest.mark.asyncio
async def test_generate_tldr_for_story_includes_discussion_when_comments_present(
    monkeypatch,
):
    async def mock_call(*, api_key, model, prompt, max_tokens, base_url):
        return (
            "### Article\n"
            "- Article point\n"
            "- Second article point\n"
            "- Third article point\n"
            "\n"
            "### Discussion\n"
            "- Comment summary\n"
            "- Nuance point\n"
        )

    monkeypatch.setattr("scripts.benchmark_tldr_llms._call_openrouter_chat", mock_call)
    story = StoryRecord(
        id=2,
        title="With Comments",
        url="https://example.com",
        source="hn",
        self_text="Some author text",
        top_comments="/u/a: Good comment\n/u/b: Another one",
        article_body="Some article",
        text_content="Full content",
    )
    result = await generate_tldr_for_story(
        story,
        api_key="sk-test",
        model="test/model",
        base_url="https://api.example.com/v1",
        rate_sleep=0.0,
    )
    assert result.status == "ok"
    assert "### Article" in result.tldr
    assert "### Discussion" in result.tldr


@pytest.mark.asyncio
async def test_generate_tldr_for_story_exception(monkeypatch):
    async def mock_call(*, api_key, model, prompt, max_tokens, base_url):
        raise RuntimeError("Connection error")

    monkeypatch.setattr("scripts.benchmark_tldr_llms._call_openrouter_chat", mock_call)
    story = StoryRecord(
        id=3,
        title="Failing",
        url=None,
        source="ch_seed",
        self_text="",
        top_comments="",
        article_body="",
        text_content="Just a title.",
    )
    result = await generate_tldr_for_story(
        story,
        api_key="sk-test",
        model="test/model",
        base_url="https://api.example.com/v1",
        rate_sleep=0.0,
    )
    assert result.status == "exception"
    assert result.tldr == ""


# ── Partial cache ───────────────────────────────────────────────────────


def test_load_partial_missing(tmp_path):
    p = tmp_path / "nonexistent.json"
    assert load_partial(p) == {}


def test_load_partial_present(tmp_path):
    data = [{"story_id": 1, "status": "ok"}, {"story_id": 2, "status": "skipped"}]
    p = tmp_path / "partial.json"
    p.write_text(json.dumps(data))
    loaded = load_partial(p)
    assert 1 in loaded
    assert loaded[1]["status"] == "ok"
    assert 2 in loaded
    assert len(loaded) == 2


def test_save_and_load_partial_roundtrip(tmp_path):
    data = [{"story_id": 99, "status": "ok", "tldr": "test"}]
    p = tmp_path / "partial.json"
    save_partial(p, data)
    assert p.exists()
    loaded = load_partial(p)
    assert loaded[99]["tldr"] == "test"
