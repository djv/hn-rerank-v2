"""Process-wide cache for the shared candidate pool + embedding matrix.

``load_production_candidate_stories(..., exclude_feedback=False)`` plus
``get_or_compute_embeddings`` together load ~8-9k rows from SQLite and
materialize a ``(N, 384)`` embedding matrix. That pool is identical for
every user — per-user personalization only excludes already-voted stories,
a boolean mask over the same rows. Before this cache, every dashboard
warm and every cold-deck build re-paid the full SQL + embedding cost
(measured at 2-8s median, up to 23s) even though the SVM fit itself takes
single-digit milliseconds (see WORKLOG 2026-07-28 friend-session
investigation). This module caches the pool once per regen generation and
lets callers apply the feedback exclusion as an in-memory mask instead.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from database import Database, Story

from .config import Config
from .ranking import NULL_TRACE, Embedder, RankTrace, _NullTrace


@dataclass(frozen=True)
class CandidatePool:
    """A shared snapshot of production candidate stories and their embeddings.

    ``stories[i]`` and ``embeddings[i]`` always describe the same story.
    """

    stories: tuple[Story, ...]
    embeddings: NDArray[np.float32]
    index_by_id: Mapping[int, int]
    generation: int
    built_at: float

    def without_feedback(
        self, voted_ids: frozenset[int]
    ) -> tuple[list[Story], NDArray[np.float32]]:
        """Return (stories, embeddings) excluding any story in *voted_ids*.

        Order is preserved relative to the pool. When *voted_ids* is empty
        this still copies the embeddings array (callers may mutate rows
        via numpy views elsewhere) but avoids the boolean-mask pass.
        """
        if not voted_ids:
            return list(self.stories), self.embeddings.copy()

        keep_mask = np.fromiter(
            (story.id not in voted_ids for story in self.stories),
            dtype=bool,
            count=len(self.stories),
        )
        kept_stories = [s for s, keep in zip(self.stories, keep_mask) if keep]
        return kept_stories, self.embeddings[keep_mask]


_lock = threading.Lock()
_pool: CandidatePool | None = None
_pool_db: Database | None = None
_generation = 0


def invalidate_candidate_pool() -> None:
    """Mark the cached pool stale; the next ``get_candidate_pool`` rebuilds it.

    Call this at regen time, before the next ``build_cold_deck``/
    ``fast_rerank_for_user`` call, so the rebuilt pool reflects newly
    fetched stories.
    """
    global _generation, _pool, _pool_db
    with _lock:
        _generation += 1
        _pool = None
        _pool_db = None


def get_candidate_pool(
    db: Database,
    config: Config,
    embedder: Embedder,
    trace: RankTrace | _NullTrace = NULL_TRACE,
) -> CandidatePool:
    """Return the shared candidate pool, building it on first use or after
    ``invalidate_candidate_pool()``.

    Safe to call concurrently: only one thread builds; others block on the
    lock and reuse the result. Sets a ``pool_cache: hit|miss`` label on
    *trace* when provided, so ``rank_perf`` history shows whether a given
    rank paid the rebuild cost.

    Scoped to *db* by identity: a different ``Database`` instance (e.g. a
    fresh in-memory DB in tests) always rebuilds rather than serving a
    stale pool from a previous instance. Production runs one long-lived
    ``Database`` for the process lifetime, so this never rebuilds there
    except via explicit ``invalidate_candidate_pool()``.
    """
    global _pool, _pool_db

    with _lock:
        if _pool is not None and _pool_db is db:
            trace.set_label("pool_cache", "hit")
            return _pool
        trace.set_label("pool_cache", "miss")

        # Imported here (not at module load) to avoid a circular import:
        # pipeline/__init__.py imports this module. Importing the pipeline
        # package's own re-exported `get_or_compute_embeddings` (rather than
        # `pipeline.ranking`'s copy directly) also keeps this patchable via
        # `monkeypatch.setattr("pipeline.get_or_compute_embeddings", ...)`,
        # the convention existing tests already use.
        from . import get_or_compute_embeddings, load_production_candidate_stories

        stories = load_production_candidate_stories(
            db, config, user_id=None, exclude_feedback=False
        )
        embeddings = get_or_compute_embeddings(stories, embedder, db)
        index_by_id = {story.id: idx for idx, story in enumerate(stories)}
        _pool = CandidatePool(
            stories=tuple(stories),
            embeddings=embeddings,
            index_by_id=index_by_id,
            generation=_generation,
            built_at=time.time(),
        )
        _pool_db = db
        return _pool
