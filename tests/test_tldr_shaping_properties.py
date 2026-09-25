"""Property tests for TLDR Markdown shaping (server) and its client mirror."""

from __future__ import annotations

import json

from hypothesis import given, settings, strategies as st

import server
from tests.test_client_js import _inline_script, js_functions, run_node

# LLM-shaped lines: headings, labels, bullets with either marker, emphasis
# with single markers, and inline " - " bullet runs after punctuation.
_LINE = st.one_of(
    st.sampled_from(
        [
            "",
            "### Discussion",
            "#### Detail",
            "Summary",
            "Key Point: detail here",
            "Why it matters:",
            "Plain Heading",
        ]
    ),
    st.builds(
        lambda prefix, body: prefix + body,
        st.sampled_from(["", "- ", "* ", "  - ", "  * "]),
        st.text(alphabet="abc XY*_:.;!?-", min_size=1, max_size=30),
    ),
)
_DOC = st.lists(_LINE, max_size=16).map("\n".join)


@settings(max_examples=300)
@given(text=_DOC)
def test_normalize_is_idempotent(text: str) -> None:
    """Cached summaries are normalized again on read; that must be a no-op."""
    once = server._normalize_tldr_markdown(text)
    assert server._normalize_tldr_markdown(once) == once


@settings(max_examples=300)
@given(text=_DOC, max_bullets=st.integers(1, 6), max_subheadings=st.integers(0, 2))
def test_cap_bounds_each_section_and_only_drops_lines(
    text: str, max_bullets: int, max_subheadings: int
) -> None:
    capped = server._cap_tldr_structure(
        text, max_bullets=max_bullets, max_subheadings=max_subheadings
    )
    # Only drops lines: the output is an ordered subsequence of the input.
    remaining = iter(text.split("\n"))
    kept = capped.split("\n") if capped else []
    assert all(any(line == r for r in remaining) for line in kept)
    # Every ### section respects both caps.
    bullets = subheadings = 0
    for line in capped.split("\n"):
        if line.startswith("### "):
            bullets = subheadings = 0
        elif line.startswith("####"):
            subheadings += 1
        elif line.startswith("- "):
            bullets += 1
        assert bullets <= max_bullets and subheadings <= max_subheadings
    assert (
        server._cap_tldr_structure(
            capped, max_bullets=max_bullets, max_subheadings=max_subheadings
        )
        == capped
    )


@settings(max_examples=8, deadline=None)
@given(docs=st.lists(_DOC, min_size=25, max_size=40))
def test_client_normalization_is_a_no_op_on_server_output(docs: list[str]) -> None:
    """index.html re-normalizes cached/stale TLDRs; its mirror of the server
    rules must leave already-normalized text unchanged, or cached and fresh
    summaries render differently."""
    shaped = [server._normalize_tldr_markdown(doc) for doc in docs]
    functions = js_functions(_inline_script(), "normalizeTldrMarkdown")
    client = run_node(
        functions
        + f"\nconsole.log(JSON.stringify({json.dumps(shaped)}.map(normalizeTldrMarkdown)));\n"
    )
    for server_text, client_text in zip(shaped, client, strict=True):
        assert client_text == server_text
