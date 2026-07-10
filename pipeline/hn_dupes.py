from __future__ import annotations

import html
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Literal, Protocol, TypeAlias, TypeVar, cast

import httpx

from database import Database, HnDupeResolution, Story
from dedup import NormalizedUrl, normalize_url
from .ranking import RankedStory, clean_text, compose_story_text


FirebaseItem: TypeAlias = Mapping[str, object]
FetchItem: TypeAlias = Callable[[int], FirebaseItem | None]
CacheValue = TypeVar("CacheValue")

FIREBASE_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{sid}.json"
MAX_DUPE_RESOLVE_WORKERS = 8
MAX_SOURCE_DESCENDANTS = 8
TITLE_RATIO_THRESHOLD = 0.50
TOKEN_JACCARD_THRESHOLD = 0.20
MIN_SHARED_TITLE_TOKENS = 2
HN_ITEM_LINK_RE = re.compile(
    r"(?:https?://)?news\.ycombinator\.com/item\?id=(\d+)|"
    r"(?:href=[\"'])/?item\?id=(\d+)|"
    r"(?:^|[\s\"'>])/?item\?id=(\d+)",
    re.IGNORECASE,
)
TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "ask",
        "at",
        "be",
        "before",
        "by",
        "for",
        "from",
        "has",
        "have",
        "hn",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "launch",
        "new",
        "of",
        "on",
        "or",
        "over",
        "re",
        "show",
        "the",
        "to",
        "with",
        "why",
        "will",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class FeedbackDupeContext:
    story_ids: set[int]
    urls: set[NormalizedUrl]
    hn_stories: list[Story]


@dataclass(frozen=True)
class HnDupeOutcome:
    """A complete negative inspection is distinct from a transient failure."""
    status: Literal["canonical", "no_match", "retry"]
    canonical_story: Story | None = None
    error: str = ""


def extract_hn_story_link_ids(comment_text: str, *, source_id: int) -> list[int]:
    """Return HN story links from comment text, excluding self-links."""
    if not comment_text:
        return []

    unescaped = html.unescape(comment_text)
    target_ids: list[int] = []
    seen: set[int] = set()
    for match in HN_ITEM_LINK_RE.finditer(unescaped):
        raw_target = match.group(1) or match.group(2) or match.group(3)
        if raw_target is None:
            continue
        target_id = _coerce_positive_int(raw_target)
        if target_id is None or target_id == source_id or target_id in seen:
            continue
        target_ids.append(target_id)
        seen.add(target_id)
    return target_ids


def extract_hn_dupe_target_id(comment_text: str, *, source_id: int) -> int | None:
    """Compatibility wrapper for older tests; returns the first HN story link."""
    target_ids = extract_hn_story_link_ids(comment_text, source_id=source_id)
    return target_ids[0] if target_ids else None


def story_from_firebase_item(item: FirebaseItem) -> Story | None:
    """Normalize a validated Firebase story item without writing it to SQLite."""
    story_id = _coerce_positive_int(item.get("id"))
    if story_id is None or not _is_live_story(item):
        return None

    title = clean_text(html.unescape(str(item.get("title") or "")))
    if not title:
        return None

    self_text = clean_text(html.unescape(str(item.get("text") or "")))
    text_content = compose_story_text(title=title, self_text=self_text)
    if not text_content:
        return None

    descendants = _coerce_nonnegative_int(item.get("descendants"))
    return Story(
        id=story_id,
        title=title,
        url=_coerce_optional_str(item.get("url")),
        score=_coerce_nonnegative_int(item.get("score")),
        time=_coerce_nonnegative_int(item.get("time")),
        text_content=text_content,
        source="hn",
        comment_count=descendants,
        discussion_url=f"https://news.ycombinator.com/item?id={story_id}",
        comment_count_at_fetch=descendants,
        self_text=self_text,
        top_comments="",
        article_body="",
    )


class HnDupeResolver:
    """Best-effort resolver for low-comment HN stories linking canonical threads."""

    def __init__(
        self,
        *,
        fetch_item: FetchItem | None = None,
        ttl_seconds: float = 15 * 60,
        max_kids: int = 8,
        max_source_descendants: int = MAX_SOURCE_DESCENDANTS,
        timeout_seconds: float = 1.5,
        max_cache_entries: int = 512,
    ) -> None:
        self._fetch_item = fetch_item or self._fetch_item_from_firebase
        self._ttl_seconds = ttl_seconds
        self._max_kids = max_kids
        self._max_source_descendants = max_source_descendants
        self._timeout_seconds = timeout_seconds
        self._max_cache_entries = max_cache_entries
        self._item_cache: dict[int, tuple[float, FirebaseItem | None]] = {}
        self._canonical_cache: dict[int, tuple[float, int | None]] = {}
        self._lock = threading.Lock()

    def resolve(self, story_id: int) -> HnDupeOutcome:
        cached = self._get_cached_canonical(story_id)
        if cached is not _CACHE_MISS:
            target_id = cast(int | None, cached)
            if target_id is None:
                return HnDupeOutcome("no_match")
            target = self.get_story(target_id)
            if target is None:
                return HnDupeOutcome("retry", error="canonical target unavailable")
            return HnDupeOutcome("canonical", canonical_story=target)

        canonical_id: int | None = None
        try:
            item = self._get_item(story_id)
            if item is not None and _is_live_story(item):
                source_descendants = _coerce_nonnegative_int(item.get("descendants"))
                if source_descendants > self._max_source_descendants:
                    self._set_cached_canonical(story_id, None)
                    return HnDupeOutcome("no_match")

                targets: list[FirebaseItem] = []
                for child_id in _first_kid_ids(item, self._max_kids):
                    child = self._get_item(child_id)
                    if child is None or _is_dead_or_deleted(child):
                        continue
                    if child.get("type") != "comment":
                        continue
                    for target_id in extract_hn_story_link_ids(
                        str(child.get("text") or ""),
                        source_id=story_id,
                    ):
                        target = self._get_item(target_id)
                        if (
                            target is not None
                            and _is_live_story(target)
                            and _target_is_stronger(item, target)
                            and _titles_are_similar(item, target)
                        ):
                            targets.append(target)
                if targets:
                    canonical_id = _coerce_positive_int(
                        max(
                            targets,
                            key=lambda target: (
                                _coerce_nonnegative_int(target.get("descendants")),
                                _coerce_nonnegative_int(target.get("score")),
                            ),
                        ).get("id")
                    )
        except Exception as exc:
            logging.debug("hn_dupe_resolver story_id=%s error=%r", story_id, exc)
            return HnDupeOutcome("retry", error=repr(exc))

        self._set_cached_canonical(story_id, canonical_id)
        if canonical_id is None:
            return HnDupeOutcome("no_match")
        target = self.get_story(canonical_id)
        if target is None:
            return HnDupeOutcome("retry", error="canonical target unavailable")
        return HnDupeOutcome("canonical", canonical_story=target)

    def find_canonical_story_id(self, story_id: int) -> int | None:
        """Compatibility wrapper for callers outside the persisted worker."""
        outcome = self.resolve(story_id)
        return outcome.canonical_story.id if outcome.canonical_story else None

    def get_story(self, story_id: int) -> Story | None:
        item = self._get_item(story_id)
        if item is None:
            return None
        return story_from_firebase_item(item)

    def _get_item(self, story_id: int) -> FirebaseItem | None:
        now = time.monotonic()
        with self._lock:
            cached = self._item_cache.get(story_id)
            if cached is not None and now - cached[0] <= self._ttl_seconds:
                return cached[1]

        item = self._fetch_item(story_id)
        with self._lock:
            self._item_cache[story_id] = (now, item)
            self._trim_cache_locked(self._item_cache)
        return item

    def _get_cached_canonical(self, story_id: int) -> object:
        now = time.monotonic()
        with self._lock:
            cached = self._canonical_cache.get(story_id)
            if cached is None:
                return _CACHE_MISS
            if now - cached[0] > self._ttl_seconds:
                del self._canonical_cache[story_id]
                return _CACHE_MISS
            return cached[1]

    def _set_cached_canonical(self, story_id: int, target_id: int | None) -> None:
        with self._lock:
            self._canonical_cache[story_id] = (time.monotonic(), target_id)
            self._trim_cache_locked(self._canonical_cache)

    def _trim_cache_locked(
        self, cache: dict[int, tuple[float, CacheValue]]
    ) -> None:
        overflow = len(cache) - self._max_cache_entries
        if overflow <= 0:
            return
        for key, _value in sorted(cache.items(), key=lambda item: item[1][0])[
            :overflow
        ]:
            del cache[key]

    def _fetch_item_from_firebase(self, story_id: int) -> FirebaseItem | None:
        response = httpx.get(
            FIREBASE_ITEM_URL.format(sid=story_id),
            timeout=self._timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Firebase HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("invalid Firebase JSON") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("invalid Firebase response")
        return cast(FirebaseItem, payload)


_CACHE_MISS = object()
_DEFAULT_RESOLVER = HnDupeResolver()


class HnDupeResolutionWorker:
    """One coalescing daemon that keeps Firebase out of ranking requests."""

    def __init__(self, db: Database, *, resolver: HnDupeResolver | None = None) -> None:
        self._db = db
        self._resolver = resolver or HnDupeResolver()
        self._lock = threading.Lock()
        self._pending: list[Story] | None = None
        self._active = False

    def submit(self, candidates: Sequence[Story]) -> None:
        snapshot = [story for story in candidates if story.source == "hn"]
        if not snapshot:
            return
        with self._lock:
            self._pending = snapshot
            if self._active:
                return
            self._active = True
        threading.Thread(target=self._run, name="hn-dupe-resolver", daemon=True).start()

    def _run(self) -> None:
        while True:
            with self._lock:
                snapshot, self._pending = self._pending, None
            if snapshot:
                try:
                    self._resolve_snapshot(snapshot)
                except Exception:
                    logging.exception("hn_dupe_worker batch failed")
            with self._lock:
                if self._pending is None:
                    self._active = False
                    return

    def _resolve_snapshot(self, candidates: Sequence[Story]) -> None:
        by_id = {story.id: story for story in candidates if story.id > 0}
        due_ids = self._db.get_due_hn_dupe_candidate_ids(list(by_id), limit=250)
        if not due_ids:
            return
        now = time.time()
        with ThreadPoolExecutor(max_workers=min(MAX_DUPE_RESOLVE_WORKERS, len(due_ids))) as pool:
            futures = {pool.submit(self._resolver.resolve, sid): sid for sid in due_ids}
            for future in as_completed(futures):
                source_id = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = HnDupeOutcome("retry", error=repr(exc))
                prior = self._db.get_hn_dupe_resolutions([source_id]).get(source_id)
                failures = (prior.failure_count if prior else 0) + (outcome.status == "retry")
                if outcome.status == "canonical" and outcome.canonical_story is not None:
                    # Persist target first: a cache hit can always render locally.
                    self._db.upsert_story(outcome.canonical_story)
                    canonical_id = outcome.canonical_story.id
                    next_check = now + 30 * 86400
                elif outcome.status == "no_match":
                    canonical_id, failures, next_check = None, 0, now + 24 * 3600
                else:
                    canonical_id = None
                    next_check = now + min(6 * 3600, 15 * 60 * (2 ** max(0, failures - 1)))
                self._db.upsert_hn_dupe_resolution(HnDupeResolution(
                    source_id, canonical_id, outcome.status, now, next_check,
                    int(failures), outcome.error,
                ))


def canonicalize_hn_dupes(
    ranked: list[RankedStory],
    db: Database,
    *,
    candidate_stories: Sequence[Story] = (),
    selected_limit: int | None = None,
    user_id: int | None = None,
    feedback_actions: tuple[str, ...] = ("up", "neutral"),
    resolver: HnDupeResolver | None = None,
    trace: _TraceCounter | None = None,
) -> list[RankedStory]:
    """Replace or suppress selected HN cards with canonical duplicate targets."""
    if not ranked:
        return ranked

    selected_count = len(ranked) if selected_limit is None else max(0, selected_limit)
    if selected_count == 0:
        return ranked

    candidate_by_id = {story.id: story for story in candidate_stories}
    original_output_ids = {item.story.id for item in ranked}
    feedback_context = _load_feedback_context(
        db,
        user_id=user_id,
        actions=feedback_actions,
    )
    source_ids = list(dict.fromkeys(
        item.story.id for item in ranked[:selected_count]
        if item.story.source == "hn" and item.story.id > 0
    ))
    resolutions = db.get_hn_dupe_resolutions(source_ids)
    _set_trace_count(trace, "hn_dupes_cache_hit", len(resolutions))
    _set_trace_count(trace, "hn_dupes_cache_miss", len(source_ids) - len(resolutions))
    _set_trace_count(trace, "hn_dupes_retry", sum(r.status == "retry" for r in resolutions.values()))
    emitted_ids: set[int] = set()
    output: list[RankedStory] = []

    for index, item in enumerate(ranked):
        story = item.story
        if index >= selected_count or story.source != "hn" or story.id <= 0:
            if story.id not in emitted_ids:
                output.append(item)
                emitted_ids.add(story.id)
            continue

        resolution = resolutions.get(story.id)
        target_id = resolution.canonical_story_id if resolution and resolution.status == "canonical" else None
        if _matches_feedback(story, feedback_context):
            logging.info(
                "hn_dupe_resolver source_id=%s result=dropped_feedback_match",
                story.id,
            )
            emitted_ids.add(story.id)
            continue

        if target_id is None:
            if story.id not in emitted_ids:
                output.append(item)
                emitted_ids.add(story.id)
            continue

        if target_id in feedback_context.story_ids:
            logging.info(
                "hn_dupe_resolver source_id=%s target_id=%s result=dropped_feedback_target",
                story.id,
                target_id,
            )
            emitted_ids.add(story.id)
            continue

        if target_id in original_output_ids or target_id in emitted_ids:
            logging.info(
                "hn_dupe_resolver source_id=%s target_id=%s result=dropped_existing",
                story.id,
                target_id,
            )
            emitted_ids.add(story.id)
            continue

        target_story = _lookup_canonical_story(
            target_id,
            db,
            candidate_by_id=candidate_by_id,
        )
        if target_story is None:
            _increment_trace_count(trace, "hn_dupes_target_missing")
            if story.id not in emitted_ids:
                output.append(item)
                emitted_ids.add(story.id)
            continue

        logging.info(
            "hn_dupe_resolver source_id=%s target_id=%s result=replaced",
            story.id,
            target_id,
        )
        output.append(replace(item, story=target_story))
        emitted_ids.add(target_id)

    return output


def _resolve_selected_targets(
    ranked: Sequence[RankedStory],
    *,
    selected_count: int,
    resolver: HnDupeResolver,
) -> dict[int, int | None]:
    source_ids = list(
        dict.fromkeys(
            item.story.id
            for item in ranked[:selected_count]
            if item.story.source == "hn" and item.story.id > 0
        )
    )
    if not source_ids:
        return {}
    if len(source_ids) == 1:
        story_id = source_ids[0]
        return {story_id: resolver.find_canonical_story_id(story_id)}

    target_by_source: dict[int, int | None] = {}
    worker_count = min(MAX_DUPE_RESOLVE_WORKERS, len(source_ids))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(resolver.find_canonical_story_id, story_id): story_id
            for story_id in source_ids
        }
        for future in as_completed(futures):
            story_id = futures[future]
            try:
                target_by_source[story_id] = future.result()
            except Exception as exc:
                logging.debug(
                    "hn_dupe_resolver story_id=%s parallel_error=%r", story_id, exc
                )
                target_by_source[story_id] = None
    return target_by_source


def _lookup_canonical_story(
    target_id: int,
    db: Database,
    *,
    candidate_by_id: Mapping[int, Story],
) -> Story | None:
    from . import is_summarizable

    target_story = candidate_by_id.get(target_id)
    if target_story is None:
        target_story = db.get_story(target_id)
    if target_story is None or not is_summarizable(target_story):
        return None
    return target_story


class _TraceCounter(Protocol):
    def set_count(self, name: str, value: int) -> None: ...


def _set_trace_count(trace: _TraceCounter | None, key: str, value: int) -> None:
    if trace is not None:
        trace.set_count(key, value)


def _increment_trace_count(trace: _TraceCounter | None, key: str) -> None:
    if trace is not None:
        # RankTrace intentionally exposes set_count but no get_count.
        # Counters are only informational, so target-missing is reported once.
        trace.set_count(key, 1)


def _load_feedback_context(
    db: Database,
    *,
    user_id: int | None,
    actions: tuple[str, ...],
) -> FeedbackDupeContext:
    if user_id is None or not actions:
        return FeedbackDupeContext(story_ids=set(), urls=set(), hn_stories=[])

    action_set = set(actions)
    story_ids: set[int] = set()
    urls: set[NormalizedUrl] = set()
    hn_stories: list[Story] = []
    for record in db.get_all_feedback(user_id=user_id):
        if record.action not in action_set:
            continue
        story_ids.add(record.story_id)
        norm_url = normalize_url(record.url)
        if norm_url is not None:
            urls.add(norm_url)
        if record.source == "hn" and record.title:
            hn_stories.append(
                Story(
                    id=record.story_id,
                    title=record.title,
                    url=record.url,
                    score=0,
                    time=0,
                    text_content=record.text_content,
                    source=record.source,
                )
            )
    return FeedbackDupeContext(story_ids=story_ids, urls=urls, hn_stories=hn_stories)


def _matches_feedback(story: Story, feedback: FeedbackDupeContext) -> bool:
    if story.id in feedback.story_ids:
        return True
    story_url = normalize_url(story.url)
    if story_url is not None and story_url in feedback.urls:
        return True
    if story.source != "hn":
        return False
    if _story_comment_count(story) > MAX_SOURCE_DESCENDANTS:
        return False
    return any(_story_titles_are_similar(story.title, fb.title) for fb in feedback.hn_stories)


def _first_kid_ids(item: FirebaseItem, limit: int) -> list[int]:
    raw_kids = item.get("kids")
    if not isinstance(raw_kids, Sequence) or isinstance(raw_kids, str):
        return []
    kid_ids: list[int] = []
    for raw_id in raw_kids[:limit]:
        kid_id = _coerce_positive_int(raw_id)
        if kid_id is not None:
            kid_ids.append(kid_id)
    return kid_ids


def _is_live_story(item: FirebaseItem) -> bool:
    return item.get("type") == "story" and not _is_dead_or_deleted(item)


def _is_dead_or_deleted(item: FirebaseItem) -> bool:
    return bool(item.get("dead")) or bool(item.get("deleted"))


def _target_is_stronger(source: FirebaseItem, target: FirebaseItem) -> bool:
    return (
        _coerce_nonnegative_int(target.get("score"))
        > _coerce_nonnegative_int(source.get("score"))
        or _coerce_nonnegative_int(target.get("descendants"))
        > _coerce_nonnegative_int(source.get("descendants"))
    )


def _titles_are_similar(source: FirebaseItem, target: FirebaseItem) -> bool:
    return _story_titles_are_similar(
        str(source.get("title") or ""),
        str(target.get("title") or ""),
    )


def _story_titles_are_similar(source_title: str, target_title: str) -> bool:
    source_norm = _normalize_title(source_title)
    target_norm = _normalize_title(target_title)
    if not source_norm or not target_norm:
        return False
    ratio = SequenceMatcher(None, source_norm, target_norm).ratio()
    if ratio >= TITLE_RATIO_THRESHOLD:
        return True

    source_tokens = _informative_title_tokens(source_norm)
    target_tokens = _informative_title_tokens(target_norm)
    if not source_tokens or not target_tokens:
        return False
    shared = source_tokens & target_tokens
    if len(shared) < MIN_SHARED_TITLE_TOKENS:
        return False
    jaccard = len(shared) / len(source_tokens | target_tokens)
    return jaccard >= TOKEN_JACCARD_THRESHOLD


def _normalize_title(raw: str) -> str:
    return " ".join(TITLE_TOKEN_RE.findall(html.unescape(raw).lower()))


def _informative_title_tokens(normalized: str) -> set[str]:
    return {
        token
        for token in TITLE_TOKEN_RE.findall(normalized)
        if len(token) > 1 and token not in TITLE_STOP_WORDS
    }


def _story_comment_count(story: Story) -> int:
    if story.comment_count is not None:
        return max(0, story.comment_count)
    return max(0, story.comment_count_at_fetch)


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed > 0 else None


def _coerce_nonnegative_int(value: object) -> int:
    parsed = _coerce_positive_int(value)
    return parsed if parsed is not None else 0


def _coerce_optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
