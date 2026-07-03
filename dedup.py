from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Iterable, NewType
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
from numpy.typing import NDArray

from database import FeedbackRecord, Story


logger = logging.getLogger(__name__)


# Mirror the source string constants from pipeline.py. Kept in sync by
# the dedup_ranked test; if you rename these, update both places.
BQ_ARCHIVE_SOURCE: Final[str] = "bq_seed"
CH_ARCHIVE_SOURCE: Final[str] = "ch_seed"


NormalizedUrl = NewType("NormalizedUrl", str)


_TRACKING_QUERY_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_name",
        "utm_brand",
        "utm_social",
        "utm_creative_format",
        "utm_marketing_tactic",
        "fbclid",
        "gclid",
        "gclsrc",
        "gbraid",
        "wbraid",
        "msclkid",
        "dclid",
        "yclid",
        "twclid",
        "li_fat_id",
        "igshid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "source",
        "src",
        "feature",
        "si",
    }
)


def normalize_url(raw: str | None) -> NormalizedUrl | None:
    """Return a canonical form of *raw* suitable for equality comparisons.

    The transformation is intentionally lossy — anything below is fair game:
      * Scheme is dropped.
      * Host is lowercased; leading ``www.`` is stripped.
      * Default ports and trailing slashes are removed.
      * Empty path becomes ``/``.
      * Known tracking query parameters are dropped (see
        :data:`_TRACKING_QUERY_PARAMS`).
      * Fragment is dropped.
      * Remaining query parameters are sorted lexicographically for
        determinism.

    Returns ``None`` for inputs that can't be parsed (missing scheme/host,
    empty string, ``None``). The function is idempotent:

        >>> normalize_url(normalize_url(x)) == normalize_url(x)
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None

    # If the caller hands back a previously-normalized URL it has no
    # scheme, so urlparse won't recognize the authority. Re-add a
    # synthetic scheme to make parsing work for idempotency.
    parse_target = raw if "://" in raw else f"//{raw}"
    parsed = urlparse(parse_target)
    if not parsed.netloc:
        return None
    if "." not in parsed.netloc:
        return None

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
        if not path:
            path = "/"

    filtered: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in _TRACKING_QUERY_PARAMS:
            continue
        filtered.append((key, value))
    filtered.sort()
    query = urlencode(filtered)

    rebuilt = urlunparse(("", host, path, "", query, ""))
    if rebuilt.startswith("//"):
        rebuilt = rebuilt[2:]
    return NormalizedUrl(rebuilt)


@dataclass(frozen=True)
class DedupConfig:
    """Tunable knobs for :func:`dedup_ranked`."""

    render_enabled: bool = True
    embedding_cosine_enabled: bool = True
    embedding_cosine_threshold: float = 0.87
    exclude_actions: tuple[str, ...] = ("up", "neutral")
    source_preference: tuple[str, ...] = (
        "hn",
        BQ_ARCHIVE_SOURCE,
        CH_ARCHIVE_SOURCE,
        "rss_reddit_",
        "rss_lesswrong_com",
    )
    extra_source_preference: tuple[str, ...] = ("rss_",)


def _source_preference_rank(source: str, cfg: DedupConfig) -> tuple[int, int, int]:
    """Return a sort key: lower wins.

    Two-tier preference list: explicit ``source_preference`` entries first
    (matched on prefix for source-family buckets like ``rss_reddit_``),
    then a fallback ``extra_source_preference`` for any other ``rss_*``
    sources, then everything else.
    """
    for i, token in enumerate(cfg.source_preference):
        if source == token or (token.endswith("_") and source.startswith(token)):
            return (i, 0, 0)
    for i, token in enumerate(cfg.extra_source_preference):
        if source.startswith(token):
            return (len(cfg.source_preference), i, 0)
    return (len(cfg.source_preference) + len(cfg.extra_source_preference), 0, 0)


def _story_sort_key(
    story: Story, position: int, cfg: DedupConfig
) -> tuple[int, int, int, int, int]:
    """Stable order key for ``Story`` used inside a duplicate bucket.

    Lower wins.  Components: source preference, negative score (higher
    score first), position (preserve caller-provided order as final
    tiebreak), id (deterministic).
    """
    pref_rank = _source_preference_rank(story.source, cfg)
    return (pref_rank[0], pref_rank[1], pref_rank[2], -story.score, -position)


def dedup_ranked(
    ranked: list[Story],
    feedback: Iterable[FeedbackRecord],
    cfg: DedupConfig,
    *,
    user_id: int | None = None,
    embeddings: dict[int, NDArray[np.float32]] | None = None,
) -> list[Story]:
    """Deduplicate *ranked* preserving the caller's order outside duplicate
    buckets.

    Pipeline:

      1. **URL dedup** — bucket by :func:`normalize_url`. Within each bucket
         keep the best story per :func:`_story_sort_key`. Stories whose
         normalized URL is missing (``url is None`` or unparsable) are
         passed through unchanged — they can't be deduped by URL.
      2. **Feedback URL exclusion** — drop any remaining story whose
         normalized URL matches a feedback record's normalized URL *and*
         whose action is in ``cfg.exclude_actions``. Stories with missing
         URL are skipped here.
      3. **Embedding cosine dedup** (only if ``cfg.embedding_cosine_enabled``
         and *embeddings* is provided) — within the remaining pool, check
         each story's full-content MiniLM embedding against all already-kept
         stories. If cosine similarity >=
         ``cfg.embedding_cosine_threshold``, the two stories are considered
         the same article from different sources. Source preference
         tiebreaking (identical to URL dedup) picks the winner.

    Side effects: emits one ``INFO`` summary line per call and a ``DEBUG``
    line per suppressed story. The ``user_id`` keyword is used only for
    log context (so a multi-user render can be diffed); it does not
    influence the result. Flip the ``dedup`` logger to ``DEBUG`` to see
    per-suppression details (``logging.getLogger("dedup").setLevel(logging.DEBUG)``).
    """
    if not cfg.render_enabled or not ranked:
        if cfg.render_enabled and ranked:
            logger.info(
                "dedup user_id=%s in=%d out=%d suppressed=0",
                _format_user_id(user_id),
                len(ranked),
                len(ranked),
            )
        return list(ranked)

    feedback_list = list(feedback)
    actions_excluded = set(cfg.exclude_actions)
    fb_norm_urls: set[NormalizedUrl] = {
        n
        for n in (
            normalize_url(f.url) for f in feedback_list if f.action in actions_excluded
        )
        if n is not None
    }

    url_buckets: dict[NormalizedUrl, list[tuple[int, Story]]] = defaultdict(list)
    none_url_stories: list[tuple[int, Story]] = []
    for pos, s in enumerate(ranked):
        nurl = normalize_url(s.url)
        if nurl is None:
            none_url_stories.append((pos, s))
        else:
            url_buckets[nurl].append((pos, s))

    # Per-call suppression tracking (for DEBUG logging).
    url_dup_suppressed: list[tuple[Story, Story]] = []  # (dropped, kept_winner)
    fb_url_excluded: list[tuple[Story, NormalizedUrl]] = []  # (dropped, fb_url)
    embedding_suppressed: list[
        tuple[Story, Story, float]
    ] = []  # (dropped, kept, cosine_sim)

    survivors: list[tuple[int, Story]] = []
    for nurl, members in url_buckets.items():
        members.sort(key=lambda m: _story_sort_key(m[1], m[0], cfg))
        kept_pos, kept_story = members[0]
        survivors.append((kept_pos, kept_story))
        for _, dropped in members[1:]:
            url_dup_suppressed.append((dropped, kept_story))
    survivors.extend(none_url_stories)
    survivors.sort(key=lambda ps: ps[0])

    # Step 2: Embedding cosine dedup (runs before FB URL exclusion so that
    # feedback stories can suppress cross-source duplicates via embedding).
    if cfg.embedding_cosine_enabled and embeddings is not None:
        threshold = cfg.embedding_cosine_threshold
        kept: list[tuple[int, Story, NDArray[np.float32] | None]] = []

        max_kept = len(survivors)
        K = np.empty((max_kept, 384), dtype=np.float32)
        n_vecs = 0
        vec_to_kept: list[int] = []

        for pos, s in survivors:
            vec = embeddings.get(s.id)
            if vec is None:
                kept.append((pos, s, None))
                continue

            discarded = False
            if n_vecs > 0:
                sims = K[:n_vecs] @ vec
                best = int(np.argmax(sims))
                cos_sim = float(sims[best])
                if cos_sim >= threshold:
                    ki = vec_to_kept[best]
                    kpos, kstory, kvec = kept[ki]
                    skey = _story_sort_key(s, pos, cfg)
                    ckey = _story_sort_key(kstory, kpos, cfg)
                    if skey < ckey:
                        embedding_suppressed.append((kstory, s, cos_sim))
                        kept[ki] = (pos, s, vec)
                        K[best] = vec
                    else:
                        embedding_suppressed.append((s, kstory, cos_sim))
                    discarded = True

            if not discarded:
                K[n_vecs] = vec
                vec_to_kept.append(len(kept))
                n_vecs += 1
                kept.append((pos, s, vec))

        survivors = [(pos, s) for pos, s, _ in kept]
        survivors.sort(key=lambda ps: ps[0])

    # Step 3: Feedback URL exclusion — drop any survivor whose normalized URL
    # matches a story the user has already voted on.
    if fb_norm_urls:
        excluded_ids: set[int] = set()
        for pos, s in survivors:
            nurl = normalize_url(s.url)
            if nurl is not None and nurl in fb_norm_urls:
                fb_url_excluded.append((s, nurl))
                excluded_ids.add(s.id)
        survivors = [(pos, s) for pos, s in survivors if s.id not in excluded_ids]
        survivors.sort(key=lambda ps: ps[0])

    result = [s for _, s in survivors]

    _log_summary(
        user_id=user_id,
        n_in=len(ranked),
        n_out=len(result),
        n_url_dups=len(url_dup_suppressed),
        n_fb_url=len(fb_url_excluded),
        embedding_cosine_enabled=cfg.embedding_cosine_enabled
        and embeddings is not None,
        n_embedding=len(embedding_suppressed),
        n_buckets_gt1=sum(1 for v in url_buckets.values() if len(v) > 1),
        largest_bucket=max((len(v) for v in url_buckets.values()), default=0),
        fb_norm_urls=len(fb_norm_urls),
    )
    _log_suppressions(
        user_id,
        url_dup_suppressed,
        fb_url_excluded,
        embedding_suppressed,
    )
    return result


def _format_user_id(user_id: int | None) -> str:
    return str(user_id) if user_id is not None else "?"


def _truncate(text: str | None, limit: int = 80) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "\u2026"


def _log_summary(
    *,
    user_id: int | None,
    n_in: int,
    n_out: int,
    n_url_dups: int,
    n_fb_url: int,
    embedding_cosine_enabled: bool,
    n_embedding: int,
    n_buckets_gt1: int,
    largest_bucket: int,
    fb_norm_urls: int,
) -> None:
    """Emit a single INFO line summarising this dedup pass.

    Format is key=value for grep-ability::

        dedup user_id=3 in=75 out=57 suppressed=18 url_dups=4 fb_url=2
              embedding=on embedding_dups=10 buckets>1=3 largest_bucket=3
              fb_url_pool=12
    """
    logger.info(
        "dedup user_id=%s in=%d out=%d suppressed=%d url_dups=%d fb_url=%d "
        "embedding=%s embedding_dups=%d buckets>1=%d "
        "largest_bucket=%d fb_url_pool=%d",
        _format_user_id(user_id),
        n_in,
        n_out,
        n_in - n_out,
        n_url_dups,
        n_fb_url,
        "on" if embedding_cosine_enabled else "off",
        n_embedding,
        n_buckets_gt1,
        largest_bucket,
        fb_norm_urls,
    )


def _log_suppressions(
    user_id: int | None,
    url_dup: list[tuple[Story, Story]],
    fb_url: list[tuple[Story, NormalizedUrl]],
    embedding: list[tuple[Story, Story, float]],
) -> None:
    """Emit one DEBUG line per suppressed story for forensic debugging."""
    uid = _format_user_id(user_id)
    for dropped, kept in url_dup:
        logger.debug(
            "dedup-suppress user_id=%s reason=url_dup dropped_id=%d "
            "dropped_source=%s dropped_url=%s kept_id=%d kept_source=%s",
            uid,
            dropped.id,
            dropped.source,
            _truncate(dropped.url, 100),
            kept.id,
            kept.source,
        )
    for dropped, fb_norm_url in fb_url:
        logger.debug(
            "dedup-suppress user_id=%s reason=fb_url dropped_id=%d "
            "dropped_source=%s dropped_url=%s fb_url=%s",
            uid,
            dropped.id,
            dropped.source,
            _truncate(dropped.url, 100),
            fb_norm_url,
        )
    for dropped, kept, cos_sim in embedding:
        logger.debug(
            "dedup-suppress user_id=%s reason=embedding_cosine dropped_id=%d "
            "dropped_source=%s dropped_title=%r dropped_url=%s "
            "kept_id=%d kept_source=%s cosine_sim=%.4f",
            uid,
            dropped.id,
            dropped.source,
            _truncate(dropped.title, 60),
            _truncate(dropped.url, 100),
            kept.id,
            kept.source,
            cos_sim,
        )
