"""Facet-residual step 1: production OOF baseline over frozen blocks.

- Fits block k (k=2..5) on blocks 1..k-1 with production config, threads=1,
  cached embeddings only (text-hash validated; drift reported, never
  recomputed; no local model inference).
- Native ordinal = P(up)-P(down); percentile rank computed over ALL voted
  stories in the block (full block), not just the sampled 70.
- 99 dev: percentiles from prior replay_scores.json ordinals (no refit).
- Snapshot: original is read-only input; fits run against a task-owned COPY
  so pipeline cache writes never touch shared/main/live paths.
- Writes (this dir only): oof_scores.json, oof_metrics.json, settings_oof.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
MAIN = Path("/home/dev/hn-rewrite/main")
SRC_SNAPSHOT = Path("/tmp/opencode/cc-replay/snapshot.db")
WORK_SNAPSHOT = Path("/tmp/opencode/facet-residual-20261009/snapshot.db")
USER_ID = 151

sys.path.insert(0, str(MAIN))

from database import Database, Story  # noqa: E402
from pipeline import Embedder, story_embedding_text  # noqa: E402
from pipeline.config import Config  # noqa: E402
from pipeline.ranking import _score_and_rank  # noqa: E402

LABEL = {"down": 0, "neutral": 1, "up": 2}


class _FrozenEmbedder(Embedder):
    def __init__(self, model_version: str, embedding_dim: int = 384) -> None:
        self.model_version = model_version
        self.embedding_dim = embedding_dim

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        raise RuntimeError("OOF attempted to compute an uncaptured embedding")


def _text_hashes(stories: list[Story]) -> dict[int, str]:
    return {
        s.id: hashlib.sha256(story_embedding_text(s).encode("utf-8")).hexdigest()
        for s in stories
    }


def _valid_vectors(
    stories: list[Story],
    cached: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int], int]:
    dims = {v.shape for v in cached.values()}
    dim = dims.pop()[0] if len(dims) == 1 else 384
    bad = [
        s.id
        for s in stories
        if s.id not in cached
        or cached[s.id].shape != (dim,)
        or not np.isfinite(cached[s.id]).all()
        or not np.isclose(np.linalg.norm(cached[s.id]), 1.0, atol=1e-3)
    ]
    mat = np.asarray(
        [cached[s.id] for s in stories if s.id not in bad], dtype=np.float32
    ).reshape(-1, dim)
    return mat, bad, dim


def _pct_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    n = len(values)
    return (order + 1) / n if n > 1 else np.ones_like(values, dtype=float)


def main() -> None:
    t0 = time.time()
    assert SRC_SNAPSHOT.exists(), f"missing snapshot {SRC_SNAPSHOT}"
    WORK_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    if not WORK_SNAPSHOT.exists():
        shutil.copyfile(SRC_SNAPSHOT, WORK_SNAPSHOT)
    manifest = json.loads((HERE / "manifest.json").read_text())
    sample = json.loads((HERE / "sample.json").read_text())
    assert manifest["feasible"], "infeasible: refusing to fit"

    config = Config.load(str(MAIN / "config.toml"))
    db = Database(str(WORK_SNAPSHOT), read_only=False)
    blocks: dict[int, list[int]] = {}
    with db.conn() as conn:
        for b in range(1, 6):
            ids = sample["blocks"][f"block{b}"]
            rows = {
                r[0]: r
                for r in conn.execute(
                    "SELECT id, title, url, text_content, source, score, time,"
                    " comment_count, self_text, top_comments, article_body"
                    " FROM stories WHERE id IN (%s)" % ",".join("?" * len(ids)),
                    ids,
                )
            }
            assert len(rows) == len(ids), f"block{b}: missing stories"
            blocks[b] = ids
    for k in [k for k in sample if k.startswith("_block_rows_")]:
        del sample[k]

    def _stories(
        ids: list[int], conn: sqlite3.Connection
    ) -> tuple[list[Story], list[int], list[float]]:
        idlist = list(ids)
        m = {
            r[0]: r
            for r in conn.execute(
                "SELECT f.story_id, f.action, f.updated_at, s.title, s.url,"
                " s.text_content, s.source, s.score, s.time, s.comment_count,"
                " s.self_text, s.top_comments, s.article_body FROM feedback f"
                " LEFT JOIN stories s ON s.id=f.story_id"
                " WHERE f.user_id=? AND f.story_id IN (%s)"
                % ",".join("?" * len(idlist)),
                (USER_ID, *idlist),
            )
        }
        assert len(m) == len(idlist), "feedback/story row missing"
        out: list[Story] = []
        for sid in idlist:
            (sid2, _a, _t, title, url, text, src, score, st, cc, stx, tc, ab) = m[sid]
            out.append(
                Story(
                    id=sid2,
                    title=title or "",
                    url=url,
                    score=score or 0,
                    time=st or 0,
                    text_content=text or "",
                    source=src or "hn",
                    comment_count=cc,
                    self_text=stx or "",
                    top_comments=tc or "",
                    article_body=ab or "",
                )
            )
        return out, [LABEL[m[s][1]] for s in idlist], [float(m[s][2]) for s in idlist]

    mv = config.embedding_model_version
    oof_rows: list[dict] = []
    drift: dict[str, list[int]] = {}
    fit_secs: dict[str, float] = {}
    with db.conn() as conn:
        for k in (2, 3, 4, 5):
            train_ids = [s for b in range(1, k) for s in blocks[b]]
            test_ids = blocks[k]
            tr_stories, tr_y, tr_t = _stories(train_ids, conn)
            te_stories, te_y, te_t = _stories(test_ids, conn)
            tr_cached = db.get_embeddings_batch(
                [s.id for s in tr_stories], mv, _text_hashes(tr_stories)
            )
            tr_mat, tr_bad, dim = _valid_vectors(tr_stories, tr_cached)
            te_cached = db.get_embeddings_batch(
                [s.id for s in te_stories], mv, _text_hashes(te_stories)
            )
            te_mat, te_bad, _ = _valid_vectors(te_stories, te_cached)
            drift[f"train_bad_b{k}"] = sorted(set(tr_bad))
            drift[f"test_bad_b{k}"] = sorted(set(te_bad))
            keep = [i for i, s in enumerate(tr_stories) if s.id not in set(tr_bad)]
            fb = (
                [tr_stories[i] for i in keep],
                [int(tr_y[i]) for i in keep],
                [float(tr_t[i]) for i in keep],
            )
            scored = [i for i, s in enumerate(te_stories) if s.id not in set(te_bad)]
            cands = [te_stories[i] for i in scored]
            emb = _FrozenEmbedder(mv, tr_mat.shape[1] if len(keep) else te_mat.shape[1])
            f0 = time.time()
            ranked = _score_and_rank(
                cands, te_mat, db, config, emb, user_id=USER_ID, training_feedback=fb
            )
            fit_secs[f"block{k}"] = round(time.time() - f0, 1)
            by_id = {r.story.id: r for r in ranked}
            ordinals: list[float] = []
            for s in cands:
                prob_up = by_id[s.id].prob_up
                prob_down = by_id[s.id].prob_down
                assert prob_up is not None and prob_down is not None, (
                    f"missing class probs for {s.id}"
                )
                ordinals.append(float(prob_up - prob_down))
            ords = np.array(ordinals)
            pct = _pct_rank(ords)
            for i, s in enumerate(cands):
                r = by_id[s.id]
                oof_rows.append(
                    {
                        "story_id": s.id,
                        "block": k,
                        "true_label": int(te_y[scored[i]]),
                        "vote_time": float(te_t[scored[i]]),
                        "prob_up": r.prob_up,
                        "prob_neutral": r.prob_neutral,
                        "prob_down": r.prob_down,
                        "ordinal_up_minus_down": float(ords[i]),
                        "oof_percentile_in_block": float(pct[i]),
                        "production_score": r.score,
                    }
                )
    db.close()

    # 99 dev percentiles from frozen prior replay ordinals (no refit).
    replay = json.loads(
        Path(
            "/home/dev/hn-rewrite/llm-labels/cc-replay-20261009/replay_scores.json"
        ).read_text()
    )
    dev_ord = np.array([r["ordinal_up_minus_down"] for r in replay])
    dev_pct = _pct_rank(dev_ord)
    dev_rows = [
        {
            "story_id": r["story_id"],
            "true_label": r["true_label"],
            "vote_time": r["vote_time"],
            "prob_up": r["prob_up"],
            "prob_neutral": r["prob_neutral"],
            "prob_down": r["prob_down"],
            "ordinal_up_minus_down": r["ordinal_up_minus_down"],
            "oof_percentile_in_block": float(dev_pct[i]),
            "production_score": r["production_score"],
        }
        for i, r in enumerate(replay)
    ]

    (HERE / "oof_scores.json").write_text(
        json.dumps({"blocks": oof_rows, "dev99": dev_rows}, indent=1)
    )
    metrics = {
        "n_oof": len(oof_rows),
        "n_dev": len(dev_rows),
        "drift": drift,
        "fit_seconds": fit_secs,
        "total_seconds": round(time.time() - t0, 1),
        "config": {
            "svm_c": config.model.svm_c,
            "svm_kernel": config.model.svm_kernel,
            "svm_gamma": config.model.svm_gamma,
            "linear_blend_enabled": config.model.linear_blend_enabled,
            "side_embedding_enabled": config.model.side_embedding_enabled,
            "engagement_features_enabled": config.model.engagement_features_enabled,
            "classifier": config.model.classifier,
            "embedding_model_version": mv,
        },
        "note": "percentiles rank full-block voted stories (k=2..5) and the 99 "
        "dev set; fits see only earlier blocks (k-1..........); cached "
        "embeddings only, drift reported never recomputed.",
    }
    (HERE / "oof_metrics.json").write_text(json.dumps(metrics, indent=1))
    (HERE / "settings_oof.json").write_text(
        json.dumps(
            {
                "threads": "1",
                "snapshot_src": str(SRC_SNAPSHOT),
                "snapshot_work": str(WORK_SNAPSHOT),
                "oof_script_sha16": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest()[:16],
            },
            indent=1,
        )
    )
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
