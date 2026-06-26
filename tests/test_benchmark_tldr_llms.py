from __future__ import annotations

import json

import pytest

from scripts.benchmark_tldr_llms import (
    StoryRecord,
    _build_article_section,
    _build_content_section,
    _call_openrouter_chat,
    _count_bullets,
    _has_nested_lists,
    _sanitize_model_id,
    _source_bucket,
    _split_sections,
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
    # Insert a mix of sources
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


# ── Prompt building ─────────────────────────────────────────────────────


def test_build_article_section_only_self_text():
    result = _build_article_section("Author text", "")
    assert "Author's text:" in result
    assert "Author text" in result
    assert "Article body:" not in result


def test_build_article_section_both():
    result = _build_article_section("Author text", "Article body text")
    assert "Author's text:" in result
    assert "Article body:" in result


def test_build_content_section_all_fields():
    result = _build_content_section("My Title", "Self", "Body", "Comments")
    assert "Title: My Title" in result
    assert "Self" in result
    assert "Body" in result
    assert "Comments" in result


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


# ── Utility: word count, bullet count, nested lists, sections ───────────


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


def test_split_sections_dual():
    text = "### Article\n- Bullet 1\n\n### Discussion\n- Bullet 2"
    sections = _split_sections(text)
    assert "article" in sections
    assert "discussion" in sections
    assert "Bullet 1" in sections["article"]
    assert "Bullet 2" in sections["discussion"]


def test_split_sections_preamble():
    text = "### Article\n- Point"
    sections = _split_sections(text)
    # preamble should be empty; only article present
    assert len(sections) == 1
    assert "article" in sections


# ── Format compliance scoring ───────────────────────────────────────────


def test_score_compliance_dual_perfect():
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
    result = score_compliance(tldr, path="dual")
    assert result.passes, f"Expected no violations, got: {result.violations}"
    assert result.score == 1.0


def test_score_compliance_dual_missing_headings():
    tldr = "- Some orphan bullet\n- Another"
    result = score_compliance(tldr, path="dual")
    assert not result.passes
    assert "missing_article_heading" in result.violations
    assert "missing_discussion_heading" in result.violations


def test_score_compliance_dual_nested_lists():
    tldr = (
        "### Article\n"
        "- Top bullet\n"
        "  - Nested bullet\n"
        "- Another top bullet\n"
        "\n"
        "### Discussion\n"
        "- Comment summary\n"
        "- More discussion\n"
    )
    result = score_compliance(tldr, path="dual")
    assert not result.passes
    assert "nested_lists" in result.violations


def test_score_compliance_fallback_perfect():
    tldr = (
        "### Summary\n"
        "- **Key insight:** Something important\n"
        "- Supporting point here\n"
        "- Another relevant fact\n"
        "- Final thought to round out\n"
    )
    result = score_compliance(tldr, path="fallback")
    assert result.passes, f"Expected no violations, got: {result.violations}"
    assert result.score == 1.0


def test_score_compliance_fallback_no_bullets():
    tldr = "### Summary\nJust a paragraph without any bullets."
    result = score_compliance(tldr, path="fallback")
    assert not result.passes
    assert "fallback_no_bullets" in result.violations


def test_score_compliance_nonempty_fail():
    result = score_compliance("Error: HTTP 500", path="dual")
    assert not result.passes
    assert result.score == 0.0


def test_score_compliance_article_line_not_bullet():
    tldr = (
        "### Article\n"
        "This is a prose line, not a bullet\n"
        "- Valid bullet\n"
        "- Another\n"
        "\n"
        "### Discussion\n"
        "- Comment\n"
        "- Another comment\n"
    )
    result = score_compliance(tldr, path="dual")
    assert not result.passes
    assert "article_line_not_bullet" in result.violations


# ── generate_tldr_for_story dual path ───────────────────────────────────


@pytest.mark.asyncio
async def test_generate_tldr_for_story_dual_path_success(monkeypatch):
    async def mock_call(*, api_key, model, prompt, max_tokens, base_url):
        if "Summarize the article" in prompt:
            return "- Article summary mock"
        if "Summarize the discussion" in prompt:
            return "- Discussion summary mock"
        return "- Fallback mock"

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
    assert result.path == "dual"
    assert "### Article" in result.tldr
    assert "### Discussion" in result.tldr
    assert result.raw_article_response == "- Article summary mock"
    assert result.raw_discussion_response == "- Discussion summary mock"
    assert result.raw_fallback_response == ""
    assert result.compliance is not None


@pytest.mark.asyncio
async def test_generate_tldr_for_story_fallback_path(monkeypatch):
    async def mock_call(*, api_key, model, prompt, max_tokens, base_url):
        return "- Single fallback bullet\n- Second bullet"

    monkeypatch.setattr("scripts.benchmark_tldr_llms._call_openrouter_chat", mock_call)
    story = StoryRecord(
        id=2,
        title="Fallback Only",
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
    assert result.status == "ok"
    assert result.path == "fallback"
    assert result.raw_fallback_response is not None
    assert result.raw_article_response == ""
    assert result.raw_discussion_response == ""
    assert result.compliance is not None


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


# ── Score compliance edge cases ─────────────────────────────────────────


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
    result = score_compliance(tldr, path="dual")
    assert "normalization_not_idempotent" not in result.violations


def test_score_compliance_unexpected_format():
    tldr = (
        "### Article\n"
        "- Valid bullet\n"
        "orphan text without prefix\n"
        "- Another bullet\n"
        "\n"
        "### Discussion\n"
        "- Comment\n"
        "- More\n"
    )
    result = score_compliance(tldr, path="dual")
    assert "unexpected_format" in str(result.violations)
