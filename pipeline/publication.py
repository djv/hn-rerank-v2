"""Experimental, duplicate-aware publication affinity from past votes only."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from typing import NewType
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

from database import Story
from dedup import normalize_url

PublicationKey = NewType("PublicationKey", str)
IDENTITY_VERSION = 2
# A host is not an author on these shared platforms. Preserve other subdomains.
_SHARED_HOSTS = frozenset(
    {
        "news.ycombinator.com",
        "reddit.com",
        "old.reddit.com",
        "github.com",
        "medium.com",
        "substack.com",
        "youtube.com",
        "youtu.be",
    }
)


@dataclass(frozen=True)
class Identity:
    publication: PublicationKey
    article: str


def identity(story: Story) -> Identity | None:
    try:
        url = urlsplit(story.url or "")
        if url.scheme not in {"http", "https"} or not url.hostname:
            return None
        host = url.hostname.encode("idna").decode().lower().removeprefix("www.")
        if host in _SHARED_HOSTS:
            return None
        # Keep nonstandard ports distinct; fragments do not identify articles.
        port = url.port
        authority = host if port in {None, 80, 443} else f"{host}:{port}"
        article = normalize_url(story.url)
        if article is None:
            return None
        return Identity(PublicationKey(authority), str(article))
    except (ValueError, UnicodeError):
        return None


def publication_features(
    feedback: list[Story],
    labels: list[int],
    vote_times: list[float],
    candidates: list[Story],
    *,
    strength: float = 10.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return training/candidate metadata: three prior contrasts and support.

    Training rows use strictly earlier votes, excluding their own URL group.
    Equal-time votes cannot see each other. Repeated URLs count once, using
    the latest earlier vote. Candidate evidence uses all supplied training
    history, excluding its own URL group. Caller must supply fold-local history.
    """
    if len(feedback) != len(labels) or len(labels) != len(vote_times):
        raise ValueError("Feedback, labels and times must have equal lengths")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("Publication smoothing strength must be finite and positive")
    if any(label not in {0, 1, 2} for label in labels):
        raise ValueError("Publication labels must be down/neutral/up (0/1/2)")
    if any(not np.isfinite(t) for t in vote_times):
        raise ValueError("Vote times must be finite")
    totals = np.zeros(3, dtype=np.float64)
    counts: dict[PublicationKey, NDArray[np.float64]] = defaultdict(
        lambda: np.zeros(3, dtype=np.float64)
    )
    seen: dict[str, tuple[PublicationKey, int]] = {}
    identities = [identity(s) for s in feedback]

    def row(key: Identity | None) -> NDArray[np.float32]:
        if key is None:
            return np.zeros(4, dtype=np.float32)
        global_counts = totals.copy()
        local = counts[key.publication].copy()
        previous = seen.get(key.article)
        if previous is not None:
            _, label = previous
            global_counts[label] -= 1
            local[label] -= 1
        n = local.sum()
        # Weak symmetric prior keeps all three classes defined in tiny folds.
        prior = (global_counts + 1) / (global_counts.sum() + 3)
        posterior = (local + strength * prior) / (n + strength)
        return np.asarray([*(posterior - prior), n / (n + strength)], dtype=np.float32)

    train = np.zeros((len(feedback), 4), dtype=np.float32)
    ordered = sorted(
        range(len(feedback)), key=lambda i: (vote_times[i], feedback[i].id)
    )
    for _, batch_iter in groupby(ordered, key=lambda i: vote_times[i]):
        batch = list(batch_iter)
        for i in batch:
            train[i] = row(identities[i])
        for i in batch:
            key = identities[i]
            if key is None:
                continue
            previous = seen.get(key.article)
            if previous is not None:
                pub, old_label = previous
                totals[old_label] -= 1
                counts[pub][old_label] -= 1
            totals[labels[i]] += 1
            counts[key.publication][labels[i]] += 1
            seen[key.article] = (key.publication, labels[i])
    cand = np.zeros((len(candidates), 4), dtype=np.float32)
    for i, story in enumerate(candidates):
        cand[i] = row(identity(story))
    return train, cand
