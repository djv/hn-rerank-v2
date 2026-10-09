"""CC-approved retrospective chronological ML replay (step 1, exploratory).

Honest best-effort replay, NOT exact historical production replication:
- Live DB is never opened. All reads go to the task-owned snapshot copy at
  SNAPSHOT (writable copy; pipeline cache writes land there, never live).
- Training: user-151 feedback with updated_at <= EVAL_NOW (frozen cutoff
  1791401987.9873054, the original evaluation's now). Candidates (99 window
  votes) are never in training (feedback PK is (user_id, story_id)).
- Production config from main/config.toml, single bounded fit, threads=1
  (runner exports thread caps). Cached embeddings reused; any text-hash
  mismatch/missing vector is reported as content drift and never recomputed
  (no local model inference).
- Mutable-content limits: story.score / comment counts / text_content /
  top_comments / article_body in the snapshot are CURRENT values, not
  pre-vote values (stories table has no history). With 3848 training votes
  the tier-1 gravity weight is ~0 and engagement features are off, so HN
  score/comments do not enter replay scores; text_len + TF-IDF words use
  current text (drift quantified via embedding-hash mismatch).
- Ordinal endpoint: primary P(up)-P(down) (native up>neutral>down order,
  same rank rule as JoinedLogistic); UP-vs-rest P(up) reported too.

Writes (this dir only): replay_scores.json, replay_metrics.json,
settings.json. No main/ edits, no DB writes outside snapshot, no network.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = Path("/home/dev/hn-rewrite/main")
SNAPSHOT = Path("/tmp/opencode/cc-replay/snapshot.db")
USER_ID = 151
EVAL_NOW = 1791401987.9873054
WINDOW_END = 1791511537.886615

sys.path.insert(0, str(MAIN))

from database import Database, Story  # noqa: E402
from pipeline import Embedder, story_embedding_text  # noqa: E402
from pipeline.config import Config  # noqa: E402
from pipeline.ranking import _score_and_rank  # noqa: E402

LABEL = {"down": 0, "neutral": 1, "up": 2}


class _ReplayRow(TypedDict):
    story_id: int
    title: str
    source: str
    vote_time: float
    true_label: int
    pred_label: int
    prob_up: float
    prob_neutral: float
    prob_down: float
    ordinal_up_minus_down: float
    production_score: float


class _FrozenEmbedder(Embedder):
    def __init__(self, model_version: str, embedding_dim: int = 384) -> None:
        self.model_version = model_version
        self.embedding_dim = embedding_dim

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        raise RuntimeError("Replay attempted to compute an uncaptured embedding")


def _text_hashes(stories: list[Story]) -> dict[int, str]:
    return {
        s.id: hashlib.sha256(story_embedding_text(s).encode("utf-8")).hexdigest()
        for s in stories
    }


def _valid_vectors(
    stories: list[Story], cached: dict[int, np.ndarray]
) -> tuple[np.ndarray, list[int]]:
    """Stack cached vectors; return matrix + ids with missing/drifted vectors."""
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
    return mat, bad


def _auc_binary(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, int]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n = len(pos) * len(neg)
    if n == 0:
        return None, 0
    gt = sum(np.sum(p > neg) for p in pos)
    eq = sum(np.sum(p == neg) for p in pos)
    return float((gt + 0.5 * eq) / n), int(n)


def _auc_ordinal(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, int]:
    conc = 0.0
    n = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                continue
            n += 1
            ds = float(np.sign(scores[i] - scores[j]))
            dl = float(np.sign(labels[i] - labels[j]))
            conc += 1.0 if ds == dl else (0.5 if ds == 0.0 else 0.0)
    return (float(conc / n), n) if n else (None, 0)


def main() -> None:
    t0 = time.time()
    assert SNAPSHOT.exists(), f"missing snapshot {SNAPSHOT}"
    config = Config.load(str(MAIN / "config.toml"))
    db = Database(str(SNAPSHOT), read_only=False)

    with db.conn() as conn:
        train_rows = conn.execute(
            """SELECT f.story_id, f.action, f.updated_at, s.title, s.url,
                      s.text_content, s.source, s.score, s.time, s.comment_count,
                      s.self_text, s.top_comments, s.article_body
               FROM feedback f LEFT JOIN stories s ON s.id = f.story_id
               WHERE f.user_id = ? AND f.updated_at <= ?
               ORDER BY f.updated_at""",
            (USER_ID, EVAL_NOW),
        ).fetchall()
        cand_rows = conn.execute(
            """SELECT f.story_id, f.action, f.updated_at, s.title, s.url,
                      s.text_content, s.source, s.score, s.time, s.comment_count,
                      s.self_text, s.top_comments, s.article_body
               FROM feedback f LEFT JOIN stories s ON s.id = f.story_id
               WHERE f.user_id = ? AND f.updated_at > ? AND f.updated_at <= ?
               ORDER BY f.updated_at""",
            (USER_ID, EVAL_NOW, WINDOW_END),
        ).fetchall()

    def _story(r: tuple) -> Story:
        (sid, _a, _t, title, url, text, src, score, st, cc, stx, tc, ab) = r
        return Story(
            id=sid,
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

    train_stories = [_story(r) for r in train_rows]
    y_train = np.array([LABEL[r[1]] for r in train_rows], dtype=int)
    t_train = np.array([r[2] for r in train_rows], dtype=float)
    cand_stories = [_story(r) for r in cand_rows]
    y_cand = np.array([LABEL[r[1]] for r in cand_rows], dtype=int)
    t_cand = np.array([r[2] for r in cand_rows], dtype=float)
    assert len(cand_stories) == 99, f"expected 99 candidates, got {len(cand_stories)}"
    assert not (set(s.id for s in cand_stories) & set(s.id for s in train_stories)), (
        "candidate leaked into training"
    )

    # --- cached-embedding coverage (drift report; never recompute) ---
    mv = config.embedding_model_version
    train_cached = db.get_embeddings_batch(
        [s.id for s in train_stories], mv, _text_hashes(train_stories)
    )
    train_mat, train_bad = _valid_vectors(train_stories, train_cached)
    cand_cached = db.get_embeddings_batch(
        [s.id for s in cand_stories], mv, _text_hashes(cand_stories)
    )
    cand_mat, cand_bad = _valid_vectors(cand_stories, cand_cached)

    # Drop drifted training rows (reported); drop unscorable candidates (reported).
    keep = [i for i, s in enumerate(train_stories) if s.id not in set(train_bad)]
    fb = (
        [train_stories[i] for i in keep],
        [int(y_train[i]) for i in keep],
        [float(t_train[i]) for i in keep],
    )
    scored_idx = [i for i, s in enumerate(cand_stories) if s.id not in set(cand_bad)]
    candidates = [cand_stories[i] for i in scored_idx]
    cand_emb = cand_mat
    y_scored = y_cand[scored_idx]

    embedder = _FrozenEmbedder(mv, cand_emb.shape[1])
    ranked = _score_and_rank(
        candidates,
        cand_emb,
        db,
        config,
        embedder,
        user_id=USER_ID,
        training_feedback=fb,
    )
    by_id = {r.story.id: r for r in ranked}
    assert set(by_id) == {s.id for s in candidates}, "ranked/candidate id mismatch"

    rows: list[_ReplayRow] = []
    for i, s in enumerate(candidates):
        r = by_id[s.id]
        assert (
            r.prob_up is not None
            and r.prob_down is not None
            and r.prob_neutral is not None
        ), "missing class probs"
        ordinal = float(r.prob_up - r.prob_down)
        pred = int(np.argmax([r.prob_down, r.prob_neutral, r.prob_up]))
        rows.append(
            {
                "story_id": s.id,
                "title": s.title,
                "source": s.source,
                "vote_time": float(t_cand[scored_idx[i]]),
                "true_label": int(y_scored[i]),
                "pred_label": pred,
                "prob_up": r.prob_up,
                "prob_neutral": r.prob_neutral,
                "prob_down": r.prob_down,
                "ordinal_up_minus_down": ordinal,
                "production_score": r.score,
            }
        )
    rows.sort(key=lambda d: d["vote_time"])

    ordinal = np.array([d["ordinal_up_minus_down"] for d in rows])
    pup = np.array([d["prob_up"] for d in rows])
    y = np.array([d["true_label"] for d in rows])
    auc_ord, n_ord = _auc_ordinal(ordinal, y)
    auc_up, n_up = _auc_binary(pup, (y == 2).astype(int))
    pred = np.array([d["pred_label"] for d in rows])
    acc = float((pred == y).mean())

    metrics = {
        "eval_now": EVAL_NOW,
        "window_end": WINDOW_END,
        "user_id": USER_ID,
        "n_train": len(fb[0]),
        "n_train_dropped_drift": len(train_stories) - len(fb[0]),
        "train_bad_ids": sorted(set(train_bad)),
        "n_candidates": len(cand_stories),
        "n_scored": len(rows),
        "cand_unscored_ids": sorted(set(cand_bad)),
        "label_counts": {k: int((y == v).sum()) for k, v in LABEL.items()},
        "ordinal_up_minus_down_auc": auc_ord,
        "ordinal_pairs": n_ord,
        "up_vs_rest_auc": auc_up,
        "up_vs_rest_pairs": n_up,
        "pred_class_accuracy": acc,
        "confusion_true_pred": {
            f"{t}>{p}": int(((y == t) & (pred == p)).sum())
            for t in (0, 1, 2)
            for p in (0, 1, 2)
        },
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
        "mutable_content_note": "story.score/comments/text fields are snapshot-current, "
        "not pre-vote values; tier-1 gravity weight ~0 at this feedback volume; "
        "text_len/TF-IDF use current text (drift above). NOT exact production replication.",
        "fit_seconds": round(time.time() - t0, 1),
    }
    (HERE / "replay_scores.json").write_text(json.dumps(rows, indent=1))
    (HERE / "replay_metrics.json").write_text(json.dumps(metrics, indent=1))
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    (HERE / "settings.json").write_text(
        json.dumps(
            {
                "eval_now": EVAL_NOW,
                "window_end": WINDOW_END,
                "user_id": USER_ID,
                "model": "replay-is-ML-only-no-LLM",
                "replay_script_sha16": code_hash,
                "snapshot": str(SNAPSHOT),
                "threads": "1",
            },
            indent=1,
        )
    )
    db.close()

    print(json.dumps(metrics, indent=1))
    print("--- mining: confident ML errors (true vs pred, margin) ---")
    for d in sorted(rows, key=lambda d: abs(d["ordinal_up_minus_down"]), reverse=True):
        if d["true_label"] != d["pred_label"]:
            print(
                f"{d['story_id']} true={d['true_label']} pred={d['pred_label']} "
                f"ord={d['ordinal_up_minus_down']:+.3f} pup={d['prob_up']:.3f} "
                f"src={d['source']} {d['title'][:90]}"
            )
    print(
        f"--- scored={len(rows)} errors={sum(1 for d in rows if d['true_label'] != d['pred_label'])}"
    )


if __name__ == "__main__":
    main()
