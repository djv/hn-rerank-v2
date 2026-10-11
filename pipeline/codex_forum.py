"""Source-specific content policy for the Codex Forum feed.

The Codex community category (``https://community.openai.com/c/codex/37.rss``,
source ``rss_community_openai_com``) is a Discourse forum, not a publisher:

- Topic pages render threaded discussion (up to 20 posts per crawler page),
  so fetching a topic URL as an "article" mixes non-opening replies into
  ``article_body`` and frames them as article text. This source is excluded
  from proactive article lanes by ``_NO_ARTICLE_SOURCES`` and from taps
  by ``is_codex_source``; cards carry the opening post only.
- Every RSS description ends with generated terminal markup: a count
  paragraph (``<p><small>N posts - M participants</small></p>``) plus an
  adjacent ``Read full topic`` anchor. That metadata is stripped before
  plain-text conversion. Only the complete terminal footer is recognized;
  authored small text, bare counts, bare links, and unknown markup stay.

Legacy rows stored before this policy (mixed ``article_body``, flattened
footer in ``self_text``/``text_content``) are handled nondestructively:
:func:`effective_codex_story` returns the clean summarization/embedding
view while the stored row (and its votes/identity) is left untouched.
Callers must use the effective view for ranking inputs, embedding hashes,
and summary keys, and must never write it back as if it were the raw row
(writing it back would drop the preserved ``article_body``).

No new runtime dependencies. All helpers are pure and idempotent.
"""

from __future__ import annotations

import re
from dataclasses import replace

from database import Story

CODEX_FORUM_SOURCE = "rss_community_openai_com"
"""Source ID for the Codex Forum feed (``_rss_source_name`` of the category RSS)."""

CODEX_SUMMARY_INPUT_VERSION = "codex-inputs-v1"
"""Version marker for Codex summary cache keys.

Bumped inputs (footer stripped, article omitted) hash differently from
legacy mixed-input keys, so summaries generated from contaminated text
are never reused. Stale/error fallback rejects non-matching Codex keys.
"""


def is_codex_source(source: str) -> bool:
    """Whether *source* is the Codex Forum feed."""
    return source == CODEX_FORUM_SOURCE


# Terminal generated footer: a count paragraph plus the adjacent
# "Read full topic" anchor. Both halves are required — an authored
# lookalike with only one half is preserved untouched.
_COUNT_PARAGRAPH_TAIL_RE = re.compile(
    r"<p[^>]*>\s*<small[^>]*>\s*\d+\s+posts?\s*-\s*\d+\s+participants?\s*</small>\s*(?:</p>)?\s*$"
)
_READ_FULL_TOPIC_TAIL_RE = re.compile(
    r"(?:<p[^>]*>\s*)?<a\b[^>]*>\s*Read full topic\s*</a>\s*(?:</p>)?\s*$"
)


def strip_discourse_footer_html(raw_html: str) -> str:
    """Remove the terminal Discourse count/"Read full topic" footer markup.

    Operates on raw RSS description HTML *before* plain-text conversion.
    Only a complete terminal footer (count paragraph adjacent to the
    anchor, in either paragraph layout) is removed; anything else —
    including a bare count, a bare link, mid-post markup, or unknown
    shapes — is returned unchanged. Idempotent.
    """
    if not raw_html:
        return raw_html
    without_link = _READ_FULL_TOPIC_TAIL_RE.sub("", raw_html)
    if without_link == raw_html:
        return raw_html
    without_count = _COUNT_PARAGRAPH_TAIL_RE.sub("", without_link)
    if without_count == without_link:
        # Anchor without the matching count: not the generated footer.
        return raw_html
    return without_count


# Flattened legacy form (``clean_text`` joins the footer with spaces):
# "... 3 posts - 2 participants Read full topic" at the very end.
_FOOTER_TEXT_TAIL_RE = re.compile(
    r"\s*\d+\s+posts?\s*-\s*\d+\s+participants?\s+Read full topic\s*$"
)


def strip_discourse_footer_text(plain: str) -> str:
    """Remove a terminal flattened Discourse footer from plain text.

    Requires the complete footer (count plus link text); authored
    lookalikes with only one half, and mid-text occurrences followed by
    real content, are preserved. Idempotent.
    """
    if not plain:
        return plain
    return _FOOTER_TEXT_TAIL_RE.sub("", plain)


def effective_codex_story(story: Story) -> Story:
    """Clean summarization/embedding view of a (possibly legacy) Codex row.

    Non-Codex stories are returned unchanged (identity). For Codex rows
    the view omits ``article_body`` (threaded topic HTML is never article
    text), strips a confidently recognized terminal footer from
    ``self_text``, and recomposes ``text_content`` from the clean fields;
    ``top_comments`` are passed through untouched. Clean rows are
    returned unchanged so their hashes and cache keys do not churn.

    Field-less copies (ranking-pool style objects carrying only
    ``text_content``) get a best-effort terminal footer strip of that
    text; already-composed article residue cannot be separated there, so
    prefer full-field reads.
    """
    if story.source != CODEX_FORUM_SOURCE:
        return story
    self_text = story.self_text or ""
    article_body = story.article_body or ""
    top_comments = story.top_comments or ""
    if not self_text and not top_comments and not article_body:
        cleaned = strip_discourse_footer_text(story.text_content or "")
        if cleaned == (story.text_content or ""):
            return story
        return replace(story, text_content=cleaned)
    stripped_self = strip_discourse_footer_text(self_text)
    if not article_body and stripped_self == self_text and story.text_content:
        return story
    from .ranking import compose_story_text

    return replace(
        story,
        self_text=stripped_self,
        article_body="",
        text_content=compose_story_text(story.title, stripped_self, top_comments, ""),
    )
