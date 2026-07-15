"""Coverage / narrowing report over the interaction ledger.

Read-only diagnostic: is the set of stories actually shown to a user
collapsing toward a shrinking region of the candidate pool over time?
See notes/improvement-brainstorm-2026-07-15.md (Fable 4.2) for the
motivating question.

Method: cluster the *current* candidate pool with KMeans, then measure
the impressed subset (from interaction_events) against those clusters
per ISO week — cluster coverage (fraction of clusters ever shown) and
centroid drift (cosine distance of the week's shown centroid from the
pool centroid).

Caveats (also printed at runtime):
  - Impressions before 2026-07-15 are HN-survivor-biased: the ledger
    silently dropped every non-HN event before commit 3a5a77c. Use
    --since 2026-07-15 to exclude that window.
  - The reference pool is the *current* candidate pool, not the pool as
    it existed at impression time (historical pools aren't stored).
  - The candidate pool is currently HN-only (non-HN legs disabled in
    load_production_candidate_stories), so this measures topical
    narrowing within HN, not cross-source coverage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database  # noqa: E402
from pipeline import Config, load_production_candidate_stories  # noqa: E402
from pipeline.ranking import story_embedding_text  # noqa: E402

SINCE_LEDGER_FIX = "2026-07-15"


def _heaviest_user_id(db: Database) -> int:
    rows = db.execute(
        "SELECT user_id, COUNT(*) AS n FROM feedback "
        "GROUP BY user_id ORDER BY n DESC LIMIT 1"
    )
    if not rows:
        raise SystemExit("No feedback rows found; nothing to report on.")
    return int(rows[0][0])


def _pool_embeddings(
    db: Database, config: Config, user_id: int
) -> tuple[list[int], NDArray[np.float32]]:
    """Return (story_ids, embedding matrix) for the current candidate pool.

    Read-only: stories with no cached embedding under the current
    model_version/text_hash are dropped, not computed.
    """
    import hashlib

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


def _fit_clusters(
    embeddings: NDArray[np.float32], k: int
) -> NDArray[np.float32]:
    """KMeans over the pool, mirroring pipeline/ranking.py's
    _positive_cluster_centers: dedup, fall back to k=n_unique when the
    pool is smaller than k, L2-normalize centers."""
    unique = np.unique(embeddings, axis=0)
    if len(unique) <= k:
        centers = unique
    else:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        kmeans.fit(unique)
        centers = kmeans.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.clip(norms, a_min=1e-12, a_max=None)
    return centers.astype(np.float32)


def _assign(
    embeddings: NDArray[np.float32], centers: NDArray[np.float32]
) -> NDArray[np.intp]:
    """Nearest-centroid assignment by cosine similarity (both sides are
    L2-normalized, so dot product suffices)."""
    sims = embeddings @ centers.T
    return np.argmax(sims, axis=1)


def _iso_week(ts: float) -> str:
    d = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def _cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report cluster coverage and shown-vs-pool centroid drift "
            "over the interaction_events impression ledger."
        )
    )
    parser.add_argument("--db", default="hn_rewrite.db")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int)
    parser.add_argument(
        "--k", type=int, default=12, help="Number of pool clusters (KMeans)."
    )
    parser.add_argument(
        "--since",
        default=None,
        help=(
            "ISO date (YYYY-MM-DD); only impressions on/after this date "
            f"are considered. Recommend --since {SINCE_LEDGER_FIX} to "
            "exclude the pre-fix, HN-survivor-biased ledger window."
        ),
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    db = Database(args.db, read_only=True)
    try:
        user_id = args.user_id if args.user_id is not None else _heaviest_user_id(db)

        print(f"user_id={user_id}  k={args.k}")
        print(
            "CAVEATS: reference pool is CURRENT (not historical at "
            "impression time); pool is HN-only (non-HN legs disabled in "
            "load_production_candidate_stories); pre-2026-07-15 ledger "
            "entries are HN-survivor-biased (per-event ingestion fix "
            "landed in commit 3a5a77c)."
        )
        since_ts: float | None = None
        if args.since:
            since_ts = dt.datetime.strptime(args.since, "%Y-%m-%d").replace(
                tzinfo=dt.timezone.utc
            ).timestamp()
            print(f"filtering impressions to occurred_at >= {args.since}")
        else:
            print(
                "warning: no --since filter; pre-fix HN-survivor-biased "
                f"events are included (recommend --since {SINCE_LEDGER_FIX})"
            )
        print()

        pool_ids, pool_emb = _pool_embeddings(db, config, user_id)
        if len(pool_ids) == 0:
            print("no candidate pool stories with embeddings; nothing to report.")
            return
        id_to_row = {sid: i for i, sid in enumerate(pool_ids)}

        centers = _fit_clusters(pool_emb, args.k)
        n_clusters = len(centers)
        pool_assignments = _assign(pool_emb, centers)
        pool_centroid = pool_emb.mean(axis=0)

        query = (
            "SELECT story_id, occurred_at FROM interaction_events "
            "WHERE user_id = ? AND event_type = 'impression'"
        )
        params: tuple = (user_id,)
        if since_ts is not None:
            query += " AND occurred_at >= ?"
            params = (user_id, since_ts)
        rows = db.execute(query, params)

        total_impressions = len(rows)
        in_pool: list[tuple[int, float]] = []
        dropped_no_pool = 0
        for story_id, occurred_at in rows:
            if story_id in id_to_row:
                in_pool.append((story_id, occurred_at))
            else:
                dropped_no_pool += 1

        print(
            f"impressions considered: {total_impressions} total, "
            f"{len(in_pool)} matched to current pool, "
            f"{dropped_no_pool} dropped (aged out / not in current pool)"
        )
        print()

        if not in_pool:
            print("no impressions matched the current pool; nothing to report.")
            return

        weekly: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for story_id, occurred_at in in_pool:
            weekly[_iso_week(occurred_at)].append((story_id, occurred_at))

        # Per-week coverage + drift
        header = f"{'week':>9} {'n_impr':>7} {'clusters_hit':>13} {'coverage':>9} {'drift':>7}"
        print(header)
        all_time_clusters_hit: set[int] = set()
        for week in sorted(weekly):
            entries = weekly[week]
            rows_idx = [id_to_row[sid] for sid, _ in entries]
            clusters_hit = set(int(c) for c in pool_assignments[rows_idx])
            all_time_clusters_hit |= clusters_hit
            coverage = len(clusters_hit) / n_clusters
            week_centroid = pool_emb[rows_idx].mean(axis=0)
            drift = _cosine_distance(week_centroid, pool_centroid)
            print(
                f"{week:>9} {len(entries):>7} {len(clusters_hit):>13} "
                f"{coverage:>9.2f} {drift:>7.3f}"
            )
        print()

        all_time_coverage = len(all_time_clusters_hit) / n_clusters
        print(f"all-time cluster coverage: {all_time_coverage:.2f} "
              f"({len(all_time_clusters_hit)}/{n_clusters} clusters ever shown)")
        print()

        # Per-cluster impression share (all-time)
        all_rows_idx = [id_to_row[sid] for sid, _ in in_pool]
        all_assignments = pool_assignments[all_rows_idx]
        counts = np.bincount(all_assignments, minlength=n_clusters)
        shares = counts / counts.sum()
        print(f"{'cluster':>7} {'impressions':>11} {'share':>7}")
        for c in np.argsort(-counts):
            if counts[c] == 0:
                continue
            print(f"{c:>7} {counts[c]:>11} {shares[c]:>7.2f}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
