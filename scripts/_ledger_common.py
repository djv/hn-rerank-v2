"""Shared helpers for interaction-ledger diagnostics.

Lifted out of `narrowing_report.py` (2026-07-15) so `ledger_report.py`
(2026-08-14) can reuse the same current-pool-embedding and cosine/week
utilities without duplication. Behavior-preserving: `narrowing_report.py`'s
output on a fixed `--since` is unchanged by this move.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
from numpy.typing import NDArray

from database import Database
from pipeline import Config, load_production_candidate_stories
from pipeline.ranking import story_embedding_text

SINCE_LEDGER_FIX = "2026-07-15"


def heaviest_user_id(db: Database) -> int:
    """Return the user_id with the most feedback rows."""
    rows = db.execute(
        "SELECT user_id, COUNT(*) AS n FROM feedback "
        "GROUP BY user_id ORDER BY n DESC LIMIT 1"
    )
    if not rows:
        raise SystemExit("No feedback rows found; nothing to report on.")
    return int(rows[0][0])


def pool_embeddings(
    db: Database, config: Config, user_id: int
) -> tuple[list[int], NDArray[np.float32]]:
    """Return (story_ids, embedding matrix) for the current candidate pool.

    Read-only: stories with no cached embedding under the current
    model_version/text_hash are dropped, not computed.
    """
    stories = load_production_candidate_stories(
        db, config, user_id=user_id, exclude_feedback=True
    )
    hashes = {
        s.id: hashlib.sha256(
            story_embedding_text(s).encode("utf-8")
        ).hexdigest()
        for s in stories
    }
    ids = [s.id for s in stories]
    cached = db.get_embeddings_batch(ids, config.embedding_model_version, hashes)
    kept_ids = [sid for sid in ids if sid in cached]
    dropped = len(ids) - len(kept_ids)
    if dropped:
        print(
            f"note: {dropped} of {len(ids)} pool stories have no current "
            f"embedding (uncached / stale hash) — excluded from clustering."
        )
    matrix = np.array([cached[sid] for sid in kept_ids], dtype=np.float32)
    return kept_ids, matrix


def iso_week(ts: float) -> str:
    d = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def parse_since(since: str) -> float:
    """Parse an ISO date (YYYY-MM-DD) into a UTC unix timestamp."""
    return (
        dt.datetime.strptime(since, "%Y-%m-%d")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )
