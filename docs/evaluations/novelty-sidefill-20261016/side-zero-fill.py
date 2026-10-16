"""Read-only offline analysis: how much does zero-filling missing side vectors demote fresh stories, and which fallback fixes it?

Baseline scores the production candidate pool for USER with the production
ranker at full current side coverage. Each fresh-minutes F then simulates the
serve-time gap (side-encode timer runs every ~30 min) by hiding the side
vectors of candidates younger than F and rescoring three ways: production
zero-fill, renormalize-to-unit-norm, and kNN-imputed side vectors (L2-normalized
mean of the 5 nearest covered stories by stored-embedding cosine). Hiding and
imputation only wrap the in-memory side-vector lookup / model-space join; the
DB is opened read-only and production code is untouched.
Run from the live checkout: uv run python revote/side-zero-fill.py USER OUT.json [--now TS] [--fresh-minutes 30,60,120]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, ".")
from database import Database, Story  # noqa: E402
from pipeline import Config, load_production_candidate_stories  # noqa: E402
from pipeline.config import BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE  # noqa: E402
from pipeline.ranking import WINDOW_SECONDS  # noqa: E402
from pipeline.side_embeddings import SideVectorCache  # noqa: E402
import pipeline.side_embeddings as side_mod  # noqa: E402
from scripts.eval_ranker_variants import (  # noqa: E402
    FoldData,
    _embedding_text_hashes,
    _production_scores,
    _validated_embeddings,
)

DB = "/home/dev/hn-rewrite/main/hn_rewrite.db"
REVOTE = (1791763112.0, 1791763181.0)
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<30m", 0.0, 30.0),
    ("30-60m", 30.0, 60.0),
    ("1-6h", 60.0, 360.0),
    ("6-24h", 360.0, 1440.0),
    ("1-3d", 1440.0, 4320.0),
    (">3d", 4320.0, float("inf")),
)
WINDOWS = ("12h", "1d")
TOP_KS = (12, 40)
KNN_K = 5
VARIANTS = ("zero", "renorm", "knn")


@dataclass(frozen=True)
class Candidate:
    story: Story
    age_min: float
    category: str
    has_side: bool


def _category(source: str) -> str:
    if source == "hn":
        return "hn_live"
    if source in (BQ_ARCHIVE_SOURCE, CH_ARCHIVE_SOURCE):
        return "archive"
    if source.startswith("rss_reddit_"):
        return "reddit"
    if source.startswith("rss_") or source.startswith("reddit_"):
        return "rss"
    return "other"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-vector zero-fill gap analysis (read-only).")
    p.add_argument("user", type=int)
    p.add_argument("out", type=str)
    p.add_argument("--now", type=float, default=None)
    p.add_argument("--fresh-minutes", type=str, default="30,60,120")
    return p.parse_args()


def _score_fold(fold: FoldData, config: Config, db: Database) -> NDArray[np.float64]:
    scores, _ = _production_scores(fold, config, source_db=db)
    return np.asarray(scores, dtype=np.float64)


def _ranks(scores: NDArray[np.float64]) -> NDArray[np.int64]:
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank


def _average_ranks(x: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if len(a) < 2:
        raise ValueError("spearman needs at least 2 scores")
    ra, rb = _average_ranks(a), _average_ranks(b)
    sa, sb = float(ra.std()), float(rb.std())
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    cov = float(((ra - ra.mean()) * (rb - rb.mean())).mean())
    return cov / (sa * sb)


def _hiding_get(orig_get: object, hidden: set[int]) -> object:
    def _get(self: SideVectorCache, db: Database, config: Config, stories: list[Story]) -> dict[int, NDArray[np.float32]]:
        found = orig_get(self, db, config, stories)  # type: ignore[operator]
        return {sid: v for sid, v in found.items() if sid not in hidden}

    return _get


def _knn_get(
    orig_get: object,
    hidden: set[int],
    neigh_stored: NDArray[np.float64],
    neigh_side: NDArray[np.float64],
    cand_emb: NDArray[np.float64],
    id2row: dict[int, int],
) -> object:
    def _get(self: SideVectorCache, db: Database, config: Config, stories: list[Story]) -> dict[int, NDArray[np.float32]]:
        found = {sid: v for sid, v in orig_get(self, db, config, stories).items() if sid not in hidden}  # type: ignore[operator]
        if neigh_stored.shape[0] == 0:
            return found
        for sid in hidden:
            q = cand_emb[id2row[sid]]
            top = np.argsort(neigh_stored @ q, kind="stable")[-KNN_K:]
            mean = neigh_side[top].mean(axis=0)
            n = float(np.linalg.norm(mean))
            if n > 1e-12:
                found[sid] = (mean / n).astype(np.float32)
        return found

    return _get


def _renorm_space(orig_space: object, hidden: set[int]) -> object:
    def _space(
        candidates: list[Story],
        candidate_embeddings: NDArray[np.float32],
        feedback: list[Story],
        feedback_embeddings: NDArray[np.float32],
        db: Database,
        config: Config,
        cache: SideVectorCache = side_mod._CACHE,
    ) -> side_mod.SideSpace | None:
        space = orig_space(candidates, candidate_embeddings, feedback, feedback_embeddings, db, config, cache=cache)  # type: ignore[operator]
        if space is None:
            return None
        for i, s in enumerate(candidates):
            if s.id in hidden:
                n = float(np.linalg.norm(space.candidates[i]))
                if n > 1e-12:
                    space.candidates[i] /= n
        return space

    return _space


def _top_dropout(
    mask: NDArray[np.bool_],
    base: NDArray[np.float64],
    var: NDArray[np.float64],
    hidden_mask: NDArray[np.bool_],
    k: int,
) -> tuple[int, int]:
    pw = np.flatnonzero(mask)
    if pw.size == 0:
        return 0, 0
    top_base = pw[np.argsort(-base[pw], kind="stable")][:k]
    in_top = int(hidden_mask[top_base].sum())
    pos = np.empty(len(base), dtype=np.int64)
    pos[pw[np.argsort(-var[pw], kind="stable")]] = np.arange(1, pw.size + 1)
    dropout = sum(1 for i in top_base if hidden_mask[i] and pos[i] > k)
    return in_top, dropout


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def main() -> None:
    ns = _parse_args()
    fresh = sorted({int(t) for t in ns.fresh_minutes.split(",") if t.strip()})
    if not fresh or any(f <= 0 for f in fresh):
        raise ValueError(f"--fresh-minutes must be positive ints, got {ns.fresh_minutes!r}")

    config = Config.load("config.toml")
    if not config.model.side_embedding_enabled:
        raise ValueError("config.toml has side_embedding_enabled=false: this analysis needs the side-vector ranker")
    db = Database(DB, read_only=True)
    try:
        fb_stories, fb_labels, fb_times = db.get_feedback_for_training(user_id=ns.user)
        keep = [i for i, t in enumerate(fb_times) if not REVOTE[0] <= t <= REVOTE[1]]
        fb_stories = [fb_stories[i] for i in keep]
        fb_y = np.array([fb_labels[i] for i in keep], int)
        fb_t = np.array([fb_times[i] for i in keep], float)
        n_up, n_down = int((fb_y == 2).sum()), int((fb_y == 0).sum())
        print(f"training votes: {len(fb_y)} ({n_up} up / {n_down} down), REVOTE-window excluded", flush=True)
        if not fb_stories:
            print("WARNING: no training votes: cold path, fallbacks are untestable", flush=True)

        raw_pool = load_production_candidate_stories(
            db, config, user_id=ns.user, exclude_feedback=True,
            now_ts=int(ns.now) if ns.now is not None else None,
        )
        pool_hash = dict(zip([s.id for s in raw_pool], _embedding_text_hashes(raw_pool), strict=True))
        stored = db.get_embeddings_batch([s.id for s in raw_pool], config.embedding_model_version, pool_hash)
        pool = [s for s in raw_pool if s.id in stored]
        print(f"pool: {len(raw_pool)} production candidates, {len(pool)} with stored embeddings", flush=True)
        if not pool:
            raise ValueError("empty candidate pool")
        cand_emb = _validated_embeddings(pool, {s.id: stored[s.id] for s in pool}).astype(np.float64)
        id2row = {s.id: i for i, s in enumerate(pool)}

        newest = max(s.time for s in pool)
        now = ns.now if ns.now is not None else float(max(int(time.time()), newest))
        ages = np.array([max(0.0, (now - s.time) / 60.0) for s in pool])
        n_clamped = sum(1 for s in pool if s.time > now)
        if n_clamped:
            print(f"WARNING: {n_clamped} stories newer than --now, age clamped to 0", flush=True)

        union = list({s.id: s for s in pool + fb_stories}.values())
        side_full = SideVectorCache().get(db, config, union)
        cov_union = sum(s.id in side_full for s in union) / len(union)
        cands = [Candidate(s, float(a), _category(s.source), s.id in side_full) for s, a in zip(pool, ages)]
        missing = sum(1 for c in cands if not c.has_side)
        print(f"side coverage: pool {len(cands) - missing}/{len(cands)}, union {cov_union:.3f} (gate {config.model.side_embedding_min_coverage})", flush=True)
        for label, lo, hi in BUCKETS:
            ii = [c for c in cands if lo <= c.age_min < hi]
            cov = sum(c.has_side for c in ii) / len(ii) if ii else float("nan")
            print(f"  age {label:7} n={len(ii):5} covered={cov:.3f}", flush=True)
        for cat in ("hn_live", "archive", "reddit", "rss", "other"):
            ii = [c for c in cands if c.category == cat]
            if ii:
                print(f"  src {cat:8} n={len(ii):5} covered={sum(c.has_side for c in ii) / len(ii):.3f}", flush=True)

        fold = FoldData(
            candidates=pool, cand_emb=cand_emb.astype(np.float32),
            train_stories=fb_stories, test_stories=[], test_actions=np.array([], int),
            train_vote_times=fb_t, x_train_base=np.empty((0, 0)), x_cand_base=np.empty((0, 0)),
            y_train=fb_y, tier2_scores=np.empty(0),
            train_emb=_validated_embeddings(fb_stories, db.get_embeddings_batch(
                [s.id for s in fb_stories], config.embedding_model_version,
                dict(zip([s.id for s in fb_stories], _embedding_text_hashes(fb_stories), strict=True)))) if fb_stories else None,
        )
        base = _score_fold(fold, config, db)
        base_rank = _ranks(base)
        n = len(pool)
        print(f"baseline scored ({n} stories)", flush=True)

        orig_get, orig_space = SideVectorCache.get, side_mod.model_space
        hidden_by_f = {f: {s.id for s, a in zip(pool, ages) if a < f} for f in fresh}
        summary: list[dict] = []
        runs: dict[tuple[int, str], NDArray[np.float64]] = {}
        for f in fresh:
            hidden = hidden_by_f[f]
            hidden_mask = np.array([s.id in hidden for s in pool])
            uncovered = sum(1 for c in cands if c.story.id in hidden and not c.has_side)
            print(f"F={f}min: hidden={len(hidden)} (already missing a side vector: {uncovered})", flush=True)
            if cov_union - len([h for h in hidden if h in side_full]) / len(union) < config.model.side_embedding_min_coverage:
                print("  WARNING: hiding trips the coverage gate: production falls back to stored-only", flush=True)
            covered = [sid for sid in id2row if sid in side_full and sid not in hidden]
            ctx = (np.stack([cand_emb[id2row[s]] for s in covered]).astype(np.float64) if covered else np.zeros((0, cand_emb.shape[1])),
                   np.stack([side_full[s].astype(np.float64) for s in covered]) if covered else np.zeros((0, config.side_embedding_dim)))
            for v in VARIANTS:
                if v == "zero":
                    with patch.object(SideVectorCache, "get", _hiding_get(orig_get, hidden)):
                        sc = _score_fold(fold, config, db)
                elif v == "renorm":
                    with patch.object(SideVectorCache, "get", _hiding_get(orig_get, hidden)), patch.object(
                        side_mod, "model_space", _renorm_space(orig_space, hidden)
                    ):
                        sc = _score_fold(fold, config, db)
                elif v == "knn":
                    with patch.object(SideVectorCache, "get", _knn_get(orig_get, hidden, *ctx, cand_emb, id2row)):
                        sc = _score_fold(fold, config, db)
                else:
                    raise ValueError(f"unknown variant {v}")
                runs[(f, v)] = sc
                rk = _ranks(sc)
                pct = 100.0 * (n - rk) / (n - 1) if n > 1 else np.full(n, 100.0)
                pct0 = 100.0 * (n - base_rank) / (n - 1) if n > 1 else np.full(n, 100.0)
                dp, ds = pct - pct0, sc - base
                h = np.flatnonzero(hidden_mask)
                entry: dict = {"fresh_minutes": f, "variant": v, "n_hidden": len(h),
                    "n_hidden_uncovered": uncovered,
                    "d_pct_median": float(np.median(dp[h])) if h.size else float("nan"),
                    "d_pct_mean": float(dp[h].mean()) if h.size else float("nan"),
                    "d_score_median": float(np.median(ds[h])) if h.size else float("nan"),
                    "d_score_mean": float(ds[h].mean()) if h.size else float("nan"),
                    "spearman_all": _spearman(base, sc),
                    "nonhidden_max_abs_rank_change": int(np.abs(rk[~hidden_mask] - base_rank[~hidden_mask]).max()) if (~hidden_mask).any() else 0,
                    "windows": {}}
                for w in WINDOWS:
                    wmask = ages * 60.0 <= WINDOW_SECONDS[w]
                    entry["windows"][w] = {}
                    for k in TOP_KS:
                        inn, drop = _top_dropout(wmask, base, sc, hidden_mask, k)
                        entry["windows"][w][f"top{k}"] = {"in_baseline": inn, "dropout": drop}
                summary.append(entry)
                wparts = []
                for w in WINDOWS:
                    wparts.append(w + ":" + "/".join(
                        f"{entry['windows'][w][f'top{k}']['in_baseline']}in,{entry['windows'][w][f'top{k}']['dropout']}drop" for k in TOP_KS))
                wl = " ".join(wparts)
                print(f"  {v:6} dPct med {_fmt(entry['d_pct_median'])} mean {_fmt(entry['d_pct_mean'])} "
                      f"dScore med {_fmt(entry['d_score_median'])} mean {_fmt(entry['d_score_mean'])} "
                      f"spear {_fmt(entry['spearman_all'])} nonhidMaxdR {entry['nonhidden_max_abs_rank_change']} {wl}", flush=True)

        run_ranks = {(f, v): _ranks(runs[(f, v)]) for f in fresh for v in VARIANTS}
        rows = [{
            "story_id": s.id, "source": s.source, "category": c.category, "age_minutes": c.age_min,
            "has_side_vector": c.has_side, "baseline_score": float(base[i]), "baseline_rank": int(base_rank[i]),
            "variants": {str(f): {"hidden": s.id in hidden_by_f[f],
                **{v: {"score": float(runs[(f, v)][i]), "rank": int(run_ranks[(f, v)][i])} for v in VARIANTS}} for f in fresh},
        } for i, (s, c) in enumerate(zip(pool, cands))]
        out = {"meta": {"user": ns.user, "now": now, "n_pool": n, "n_train": len(fb_y),
                "n_up": n_up, "n_down": n_down, "pool_coverage": (n - missing) / n,
                "union_coverage": cov_union, "min_coverage": config.model.side_embedding_min_coverage,
                "n_truly_missing_side": missing, "fresh_minutes": fresh, "variants": list(VARIANTS)},
            "summary": summary, "rows": rows}
        with open(ns.out, "w") as fh:
            json.dump(out, fh)
        print(f"wrote {len(rows)} per-story rows to {ns.out}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
