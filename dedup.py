from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Iterable, NewType
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from database import FeedbackRecord, Story


logger = logging.getLogger(__name__)


# Mirror the source string constants from pipeline.py. Kept in sync by
# the dedup_ranked test; if you rename these, update both places.
BQ_ARCHIVE_SOURCE: Final[str] = "bq_seed"
CH_ARCHIVE_SOURCE: Final[str] = "ch_seed"


NormalizedUrl = NewType("NormalizedUrl", str)
NormalizedTitle = NewType("NormalizedTitle", str)
SimHash64 = NewType("SimHash64", int)
Domain = NewType("Domain", str)


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


_TITLE_LEADIN_RE = re.compile(
    r"^\s*(?:show\s+hn[:\-\s]|ask\s+hn[:\-\s]|tell\s+hn[:\-\s]|launch\s+hn[:\-\s])",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


def canonical_domain(raw: str | None) -> Domain | None:
    """Return a poor-man's eTLD+1 for *raw* (last two host labels).

    Examples::

        >>> canonical_domain("https://www.theverge.com/x")
        Domain('theverge.com')
        >>> canonical_domain("https://old.reddit.com/r/foo")
        Domain('reddit.com')

    Returns ``None`` for inputs with no host. The function is deliberately
    conservative — a real eTLD+1 implementation (Mozilla's Public Suffix
    List) is intentionally avoided to keep the project dep-free.
    """
    if not raw:
        return None
    parsed = urlparse(raw.strip())
    host = parsed.netloc.lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return None
    return Domain(".".join(parts[-2:]))


def normalize_title(raw: str) -> NormalizedTitle:
    """Return a canonical form of *raw* for near-duplicate title comparison.

    The transformation:
      * Lowercases.
      * Strips common HN lead-ins (``Show HN:``, ``Ask HN:``, ...).
      * Removes punctuation.
      * Collapses whitespace.

    Idempotent and total over ``str``.
    """
    if not raw:
        return NormalizedTitle("")
    lowered = raw.lower()
    stripped = _TITLE_LEADIN_RE.sub("", lowered)
    no_punct = _PUNCT_RE.sub(" ", stripped)
    collapsed = _WS_RE.sub(" ", no_punct).strip()
    return NormalizedTitle(collapsed)


def simhash64(text: str) -> SimHash64:
    """Return a 64-bit SimHash of *text* suitable for hamming-distance dedup.

    Tokens are lowercased word-shaped runs (alphanumerics). Bit votes are
    weighted by token frequency in *text* — repeated tokens move the hash
    more than unique ones, so two titles that differ only by a single
    appended token end up with hamming distance at most a few bits.

    Empty / whitespace-only inputs return ``0``.
    """
    if not text:
        return SimHash64(0)
    tokens = _TITLE_TOKEN_RE.findall(text.lower())
    if not tokens:
        return SimHash64(0)

    weights: dict[str, int] = defaultdict(int)
    for tok in tokens:
        weights[tok] += 1

    votes = [0] * 64
    for tok, weight in weights.items():
        h = hashlib.md5(tok.encode("utf-8")).digest()
        # Use the first 8 bytes of the digest to form a 64-bit hash.
        token_hash = int.from_bytes(h[:8], "big")
        for bit in range(64):
            if token_hash & (1 << bit):
                votes[bit] += weight
            else:
                votes[bit] -= weight

    fingerprint = 0
    for bit in range(64):
        if votes[bit] > 0:
            fingerprint |= 1 << bit
    return SimHash64(fingerprint)


def hamming64(a: SimHash64, b: SimHash64) -> int:
    """Popcount of ``a ^ b``. Symmetric, total, 0..64."""
    return bin(int(a) ^ int(b)).count("1")


@dataclass(frozen=True)
class DedupConfig:
    """Tunable knobs for :func:`dedup_ranked`."""

    render_enabled: bool = True
    title_fuzzy_enabled: bool = False
    title_fuzzy_hamming: int = 2
    require_same_domain_for_fuzzy: bool = True
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
      3. **Title fuzzy dedup** (only if ``cfg.title_fuzzy_enabled``) —
         within the remaining pool, cluster by SimHash hamming distance
         ``<= cfg.title_fuzzy_hamming``; if
         ``cfg.require_same_domain_for_fuzzy``, only stories sharing a
         canonical domain (or both having ``url=None``) can cluster.
         Within each cluster keep the best story per sort key.
      4. **Feedback title exclusion** (only if
         ``cfg.title_fuzzy_enabled``) — drop remaining stories whose title
         SimHash is within the threshold of a feedback record's title
         SimHash *and* whose canonical domain matches (or both missing).

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
    fb_norm_titles_by_domain: dict[Domain | None, list[NormalizedTitle]] = defaultdict(
        list
    )
    for f in feedback_list:
        if f.action not in actions_excluded:
            continue
        ntitle = normalize_title(f.title)
        if not ntitle:
            continue
        fdomain = canonical_domain(f.url)
        fb_norm_titles_by_domain[fdomain].append(ntitle)

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
    title_fuzzy_suppressed: list[tuple[Story, Story]] = []  # (dropped, kept_winner)
    fb_title_excluded: list[tuple[Story, str]] = []  # (dropped, matched_feedback_title)

    survivors: list[tuple[int, Story]] = []
    for nurl, members in url_buckets.items():
        members.sort(key=lambda m: _story_sort_key(m[1], m[0], cfg))
        if nurl in fb_norm_urls:
            for _, dropped in members:
                fb_url_excluded.append((dropped, nurl))
            continue
        kept_pos, kept_story = members[0]
        survivors.append((kept_pos, kept_story))
        for _, dropped in members[1:]:
            url_dup_suppressed.append((dropped, kept_story))
    survivors.extend(none_url_stories)

    survivors.sort(key=lambda ps: ps[0])

    if not cfg.title_fuzzy_enabled:
        result = [s for _, s in survivors]
        _log_summary(
            user_id=user_id,
            n_in=len(ranked),
            n_out=len(result),
            n_url_dups=len(url_dup_suppressed),
            n_fb_url=len(fb_url_excluded),
            title_fuzzy_enabled=False,
            n_title_fuzzy=0,
            n_fb_title=0,
            n_buckets_gt1=sum(1 for v in url_buckets.values() if len(v) > 1),
            largest_bucket=max((len(v) for v in url_buckets.values()), default=0),
            fb_norm_urls=len(fb_norm_urls),
        )
        _log_suppressions(
            user_id,
            url_dup_suppressed,
            fb_url_excluded,
            title_fuzzy_suppressed,
            fb_title_excluded,
        )
        return result

    hamming_limit = cfg.title_fuzzy_hamming
    require_same_domain = cfg.require_same_domain_for_fuzzy
    survivors_with_meta: list[tuple[int, Story, SimHash64, Domain | None]] = []
    for pos, s in survivors:
        ntitle = normalize_title(s.title)
        sh = simhash64(ntitle) if ntitle else SimHash64(0)
        domain = canonical_domain(s.url) if s.url else None
        survivors_with_meta.append((pos, s, sh, domain))

    clustered: list[tuple[int, Story, SimHash64, Domain | None]] = []
    for pos, s, sh, domain in survivors_with_meta:
        matched_ci: int | None = None
        for ci, (_, cstory, csh, cdomain) in enumerate(clustered):
            if require_same_domain and domain != cdomain:
                continue
            if hamming64(sh, csh) > hamming_limit:
                continue
            matched_ci = ci
            break
        if matched_ci is not None:
            cpos, cstory, _csh, _cdomain = clustered[matched_ci]
            ckey = _story_sort_key(cstory, cpos, cfg)
            skey = _story_sort_key(s, pos, cfg)
            if skey < ckey:
                title_fuzzy_suppressed.append((cstory, s))
                clustered[matched_ci] = (pos, s, sh, domain)
            else:
                title_fuzzy_suppressed.append((s, cstory))
        else:
            clustered.append((pos, s, sh, domain))

    final: list[tuple[int, Story]] = []
    for pos, s, sh, domain in clustered:
        excluded = False
        matched_fb_title: NormalizedTitle | None = None
        if sh != 0:
            for fb_domain, fb_titles in fb_norm_titles_by_domain.items():
                if require_same_domain and domain != fb_domain:
                    continue
                for fb_title in fb_titles:
                    if not fb_title:
                        continue
                    fb_sh = simhash64(fb_title)
                    if fb_sh == 0:
                        continue
                    if hamming64(sh, fb_sh) <= hamming_limit:
                        excluded = True
                        matched_fb_title = fb_title
                        break
                if excluded:
                    break
        if excluded:
            assert matched_fb_title is not None
            fb_title_excluded.append((s, matched_fb_title))
        else:
            final.append((pos, s))

    final.sort(key=lambda ps: ps[0])
    result = [s for _, s in final]

    _log_summary(
        user_id=user_id,
        n_in=len(ranked),
        n_out=len(result),
        n_url_dups=len(url_dup_suppressed),
        n_fb_url=len(fb_url_excluded),
        title_fuzzy_enabled=True,
        n_title_fuzzy=len(title_fuzzy_suppressed),
        n_fb_title=len(fb_title_excluded),
        n_buckets_gt1=sum(1 for v in url_buckets.values() if len(v) > 1),
        largest_bucket=max((len(v) for v in url_buckets.values()), default=0),
        fb_norm_urls=len(fb_norm_urls),
    )
    _log_suppressions(
        user_id,
        url_dup_suppressed,
        fb_url_excluded,
        title_fuzzy_suppressed,
        fb_title_excluded,
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
    title_fuzzy_enabled: bool,
    n_title_fuzzy: int,
    n_fb_title: int,
    n_buckets_gt1: int,
    largest_bucket: int,
    fb_norm_urls: int,
) -> None:
    """Emit a single INFO line summarising this dedup pass.

    Format is key=value for grep-ability::

        dedup user_id=3 in=75 out=57 suppressed=18 url_dups=4 fb_url=2
              title_fuzzy=on title_fuzzy_dups=10 fb_title=2 buckets>1=3
              largest_bucket=3 fb_url_pool=12
    """
    logger.info(
        "dedup user_id=%s in=%d out=%d suppressed=%d url_dups=%d fb_url=%d "
        "title_fuzzy=%s title_fuzzy_dups=%d fb_title=%d buckets>1=%d "
        "largest_bucket=%d fb_url_pool=%d",
        _format_user_id(user_id),
        n_in,
        n_out,
        n_in - n_out,
        n_url_dups,
        n_fb_url,
        "on" if title_fuzzy_enabled else "off",
        n_title_fuzzy,
        n_fb_title,
        n_buckets_gt1,
        largest_bucket,
        fb_norm_urls,
    )


def _log_suppressions(
    user_id: int | None,
    url_dup: list[tuple[Story, Story]],
    fb_url: list[tuple[Story, NormalizedUrl]],
    title_fuzzy: list[tuple[Story, Story]],
    fb_title: list[tuple[Story, str]],
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
    for dropped, kept in title_fuzzy:
        logger.debug(
            "dedup-suppress user_id=%s reason=title_fuzzy dropped_id=%d "
            "dropped_source=%s dropped_title=%r kept_id=%d kept_source=%s",
            uid,
            dropped.id,
            dropped.source,
            _truncate(dropped.title, 60),
            kept.id,
            kept.source,
        )
    for dropped, fb_norm_title in fb_title:
        logger.debug(
            "dedup-suppress user_id=%s reason=fb_title dropped_id=%d "
            "dropped_source=%s dropped_title=%r fb_title=%r",
            uid,
            dropped.id,
            dropped.source,
            _truncate(dropped.title, 60),
            _truncate(fb_norm_title, 60),
        )
