#!/usr/bin/env python3
"""Compare embeddings with simple probes that are not tuned to any of them.

The production ranker (and the eval's challenger) was tuned on the stored
embeddings, so a new embedding can lose there only because C/gamma/blend
suit the old geometry. This scores every embedding file the same way, each
probe's hyperparameters picked by inner cross-validation per embedding:

- taste (user's votes, down/neutral/up): logistic regression, RBF SVM and
  cosine kNN. Score = P(up) - P(down) (SVM: decision up - down). Reports
  AUC up vs rest, up vs down, up vs neutral, and the upvote / downvote share
  of the top 12 in each held-out fold.
- topic (non-HN stories whose source has >= --min-source stories): logistic
  regression macro-F1 predicting the source (subreddit, blog, site), and
  k-means V-measure against the source. A generic in-domain classification
  and clustering check that ignores the user's taste.

Inputs are replay-embedding .npz files (story_ids + embeddings) from
encode_replay_embeddings.py; only stories present in every file are used,
so files from the same sample compare on identical stories. ``A+B`` joins
two files side by side (each half scaled by 1/sqrt(2)). ``meta`` is a
non-text baseline: source one-hot plus log points, log comments and log
text length. Embeddings are never standard-scaled. Reads a read-only
snapshot; never writes to the DB.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, v_measure_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config, story_embedding_text  # noqa: E402
from scripts.eval_ranker_variants import frozen_database  # noqa: E402

DOWN, NEUTRAL, UP = 0, 1, 2
# Archive seeders and catch-all labels carry no topic.
UNTOPICAL_SOURCES = {"hn", "ch_seed", "bq_seed", "rss", "digg"}
LOGREG_C = (0.3, 1.0, 3.0, 10.0, 30.0)
SVM_C = (0.5, 1.0, 2.0, 4.0, 8.0)
SVM_GAMMA = (0.5, 1.0, 2.0)  # multiples of 1/(dim * var), sklearn's "scale"
KNN_K = (10, 20, 40, 80)


@dataclass(frozen=True)
class Probe:
    name: str
    grid: tuple[tuple[float, ...], ...]


def _load(path: str) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    data = np.load(path, allow_pickle=False)
    vectors = np.asarray(data["embeddings"], dtype=np.float32)
    return np.asarray(data["story_ids"], dtype=np.int64), vectors


def _unit(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _fit_score(
    probe: str,
    params: tuple[float, ...],
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float32],
) -> NDArray[np.float64]:
    if probe == "logreg":
        model = LogisticRegression(C=params[0], max_iter=2000)
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)
        return proba[:, UP] - proba[:, DOWN]
    if probe == "svm":
        gamma = params[1] / (x_train.shape[1] * x_train.var())
        svm = SVC(C=params[0], gamma=gamma, decision_function_shape="ovr")
        svm.fit(x_train, y_train)
        decision = svm.decision_function(x_test)
        return decision[:, UP] - decision[:, DOWN]
    knn = KNeighborsClassifier(
        n_neighbors=int(params[0]), weights="distance", metric="cosine"
    )
    knn.fit(x_train, y_train)
    proba = knn.predict_proba(x_test)
    return proba[:, UP] - proba[:, DOWN]


def _auc(
    y: NDArray[np.int64], score: NDArray[np.float64], a: int, b: int | None
) -> float:
    keep = (y == a) | (y == b) if b is not None else np.ones(len(y), dtype=bool)
    return float(roc_auc_score(y[keep] == a, score[keep]))


def _grid(probe: str) -> list[tuple[float, ...]]:
    if probe == "logreg":
        return [(c,) for c in LOGREG_C]
    if probe == "svm":
        return [(c, g) for c in SVM_C for g in SVM_GAMMA]
    return [(float(k),) for k in KNN_K]


def _taste(
    x: NDArray[np.float32], y: NDArray[np.int64], probe: str, folds: int, seed: int
) -> dict[str, float]:
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []
    for train, test in outer.split(x, y):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 1)
        best, best_auc = _grid(probe)[0], -1.0
        for params in _grid(probe):
            aucs = []
            for fit, val in inner.split(x[train], y[train]):
                score = _fit_score(
                    probe, params, x[train][fit], y[train][fit], x[train][val]
                )
                aucs.append(_auc(y[train][val], score, UP, None))
            if np.mean(aucs) > best_auc:
                best, best_auc = params, float(np.mean(aucs))
        score = _fit_score(probe, best, x[train], y[train], x[test])
        top = np.argsort(-score)[:12]
        rows.append(
            {
                "auc_up_rest": _auc(y[test], score, UP, None),
                "auc_up_down": _auc(y[test], score, UP, DOWN),
                "auc_up_neutral": _auc(y[test], score, UP, NEUTRAL),
                "up12": float((y[test][top] == UP).mean()),
                "down12": float((y[test][top] == DOWN).mean()),
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _topic(
    x: NDArray[np.float32], sources: list[str], min_source: int, seed: int
) -> tuple[float, float, int, int]:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source] = counts.get(source, 0) + 1
    kept = [
        index
        for index, source in enumerate(sources)
        if source not in UNTOPICAL_SOURCES and counts[source] >= min_source
    ]
    labels = np.array([sources[index] for index in kept])
    classes = sorted(set(labels.tolist()))
    xs = x[kept]
    predicted = np.empty(len(kept), dtype=object)
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(
        xs, labels
    ):
        model = LogisticRegression(C=10.0, max_iter=3000)
        model.fit(xs[train], labels[train])
        predicted[test] = model.predict(xs[test])
    f1 = float(f1_score(labels, predicted.astype(str), average="macro"))
    clusters = KMeans(len(classes), n_init=10, random_state=seed).fit_predict(xs)
    return f1, float(v_measure_score(labels, clusters)), len(kept), len(classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", nargs="+", help="NPZ file, A+B, or meta")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--probes", default="logreg,svm,knn")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-source", type=int, default=12)
    args = parser.parse_args()

    files = {
        part for spec in args.embeddings if spec != "meta" for part in spec.split("+")
    }
    loaded = {path: _load(path) for path in files}
    common = set.intersection(*(set(ids.tolist()) for ids, _ in loaded.values()))

    config = Config.load(args.config)
    with frozen_database(config.db_path) as (db, _snapshot_hash):
        stories, labels, _times = db.get_feedback_for_training(user_id=args.user_id)
    by_id = {
        story.id: (story, label)
        for story, label in zip(stories, labels, strict=True)
        if story.id in common
    }
    ids = np.array(sorted(by_id), dtype=np.int64)
    y = np.array([by_id[int(i)][1] for i in ids], dtype=np.int64)
    sources = [by_id[int(i)][0].source for i in ids]
    print(
        f"stories {len(ids)} (down/neutral/up {np.bincount(y, minlength=3).tolist()})",
        flush=True,
    )

    def aligned(path: str) -> NDArray[np.float32]:
        file_ids, vectors = loaded[path]
        row = {int(story_id): index for index, story_id in enumerate(file_ids)}
        return _unit(vectors[[row[int(i)] for i in ids]])

    def meta() -> NDArray[np.float32]:
        names = sorted(set(sources))
        onehot = np.array([[s == n for n in names] for s in sources], dtype=np.float32)
        numeric = np.array(
            [
                [
                    np.log1p(max(by_id[int(i)][0].score, 0)),
                    np.log1p(max(by_id[int(i)][0].comment_count or 0, 0)),
                    np.log1p(len(story_embedding_text(by_id[int(i)][0]))),
                ]
                for i in ids
            ],
            dtype=np.float32,
        )
        # Metadata columns only; embeddings are never standard-scaled.
        return np.hstack([onehot, StandardScaler().fit_transform(numeric)]).astype(
            np.float32
        )

    header = (
        f"{'embedding':38} {'probe':6} {'AUCup':>6} {'up/dn':>6} {'up/nt':>6} "
        f"{'up12':>5} {'dn12':>5}"
    )
    print(header, flush=True)
    topic_lines = []
    for spec in args.embeddings:
        if spec == "meta":
            x = meta()
        else:
            parts = [aligned(path) for path in spec.split("+")]
            x = np.hstack(parts) / np.sqrt(len(parts))
        label = (
            "meta"
            if spec == "meta"
            else "+".join(Path(p).stem for p in spec.split("+"))
        )
        for probe in args.probes.split(","):
            r = _taste(x, y, probe, args.folds, args.seed)
            print(
                f"{label:38} {probe:6} {r['auc_up_rest']:6.3f} {r['auc_up_down']:6.3f} "
                f"{r['auc_up_neutral']:6.3f} {r['up12']:5.2f} {r['down12']:5.2f}",
                flush=True,
            )
        if spec != "meta":
            f1, vm, n, k = _topic(x, sources, args.min_source, args.seed)
            topic_lines.append(
                f"{label:38} source F1 {f1:.3f}  k-means V {vm:.3f}  ({n} stories, {k} sources)"
            )
    print("\n".join(["topic:"] + topic_lines), flush=True)


if __name__ == "__main__":
    main()
