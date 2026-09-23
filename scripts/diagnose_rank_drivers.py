"""Offline SVM driver analysis; disposable read-only DB snapshot only.

Reuses the production-trained model artifacts (SVM + scaler + cluster
centers) via the model cache, then ablates feature groups on the fixed
candidate matrix. Never touches ranking code paths or the live database.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from database import Database  # noqa: E402
from pipeline import Config, Embedder, load_production_candidate_stories  # noqa: E402
from pipeline.ranking import (  # noqa: E402
    _knn_mean_and_max,
    _score_and_rank,
    _similarity_to_positive_cluster_centers,
    _svm_personalization_features,
    get_or_compute_embeddings,
    source_category_stack,
)


def _candidate_feature_matrix(
    stories: list,
    embeddings: np.ndarray,
    feedback_stories: list,
    feedback_labels: list[int],
    fb_embeddings: np.ndarray,
    centers: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    up_mask = np.array(feedback_labels) == 2
    down_mask = np.array(feedback_labels) == 0
    k = config.model.knn_k
    sim_up, close_up, _ = _knn_mean_and_max(embeddings, fb_embeddings[up_mask], k)
    sim_down, close_down, _ = _knn_mean_and_max(embeddings, fb_embeddings[down_mask], k)
    cluster_sim = _similarity_to_positive_cluster_centers(embeddings, centers)
    onehot = source_category_stack([s.source for s in stories])
    feats = _svm_personalization_features(
        embeddings,
        text_lengths=np.array([len(s.text_content) for s in stories]),
        sim_to_upvoted=sim_up,
        sim_to_downvoted=sim_down,
        closest_upvoted=close_up,
        closest_downvoted=close_down,
        positive_cluster_similarity=cluster_sim,
        is_hn_live=onehot[:, 0],
        is_archive=onehot[:, 1],
        is_reddit=onehot[:, 2],
        is_rss=onehot[:, 3],
    )
    return feats, sim_up


META_NAMES = [
    "text_length",
    "sim_to_up",
    "sim_to_down",
    "closest_up",
    "closest_down",
    "cluster_sim",
    "is_hn",
    "is_archive",
    "is_reddit",
    "is_rss",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-db", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--model-dir", required=False, default=None)
    args = parser.parse_args()
    db = Database(args.snapshot_db)
    try:
        config = Config.load()
        embedder = Embedder(
            args.model_dir or config.onnx_model_dir,
            model_version=config.embedding_model_version,
            max_tokens=config.embedding_max_tokens,
            batch_size=config.embedding_batch_size,
        )
        stories = load_production_candidate_stories(
            db, config, user_id=args.user_id, exclude_feedback=True
        )
        embeddings = get_or_compute_embeddings(stories, embedder, db)
        baseline = _score_and_rank(
            stories, embeddings, db, config, embedder, user_id=args.user_id
        )
        base_ids = [r.story.id for r in baseline[:10]]
        lw_ids = [s.id for s in stories if s.source == "rss_lesswrong_com"]
        print(
            json.dumps(
                {
                    "n_candidates": len(stories),
                    "n_lw": len(lw_ids),
                    "top10_ids": base_ids,
                    "top10_sources": [
                        next(s.source for s in stories if s.id == i) for i in base_ids
                    ],
                }
            )
        )

        feedback_stories, feedback_labels, _ = db.get_feedback_for_training(
            user_id=args.user_id
        )
        fb_embs = get_or_compute_embeddings(feedback_stories, embedder, db)
        from pipeline.ranking import _MODEL_CACHE, _positive_cluster_centers

        cached = next(iter(_MODEL_CACHE.values()), None)
        assert cached is not None, "model cache empty after ranking"
        svm, scaler, centers = cached
        if centers is None:
            centers = _positive_cluster_centers(
                fb_embs[(np.array(feedback_labels) == 2)],
                config.model.positive_cluster_k,
            )
        feats, sim_up = _candidate_feature_matrix(
            stories,
            embeddings,
            feedback_stories,
            list(feedback_labels),
            fb_embs,
            centers,
            config,
        )
        emb_dim = embeddings.shape[1]
        scaled = np.hstack(
            [
                feats[:, :emb_dim],
                np.clip(scaler.transform(feats[:, emb_dim:]), -2.5, 2.5),
            ]
        )

        def margins(matrix: np.ndarray) -> np.ndarray:
            return np.asarray(svm.decision_function(matrix))[
                :, list(svm.classes_).index(2)
            ]

        base_margin = margins(scaled)
        # Fidelity: reconstructed margins must reproduce production ordering.
        index_by_id = {s.id: i for i, s in enumerate(stories)}
        prod_order = [index_by_id[r.story.id] for r in baseline]
        my_order = list(np.argsort(base_margin)[::-1])
        overlap = len(set(my_order[:100]) & set(prod_order[:100]))
        prod_scores = np.array([r.score for r in baseline])
        prod_by_pos = np.empty(len(stories))
        for pos, idx in enumerate(prod_order):
            prod_by_pos[idx] = prod_scores[pos]
        order_corr = float(
            np.corrcoef(
                np.argsort(np.argsort(base_margin)), np.argsort(np.argsort(prod_by_pos))
            )[0, 1]
        )
        print(
            json.dumps(
                {
                    "fidelity_top100_overlap": overlap,
                    "rank_correlation": order_corr,
                    "distinct_margins": int(len(np.unique(np.round(base_margin, 9)))),
                }
            )
        )
        if order_corr < 0.999:
            print(json.dumps({"fidelity": "MISMATCH — ablations skipped"}))
            return
        lw_idx = [i for i, s in enumerate(stories) if s.id in set(lw_ids)]
        rss_idx = [
            i
            for i, s in enumerate(stories)
            if s.source.startswith("rss_") and s.id not in set(lw_ids)
        ]
        hn_idx = [i for i, s in enumerate(stories) if s.source == "hn"]
        for name, idx in (("lw", lw_idx), ("other_rss", rss_idx), ("hn", hn_idx)):
            vals = base_margin[idx]
            print(
                json.dumps(
                    {
                        "raw_margin": name,
                        "n": len(vals),
                        "median": float(np.median(vals)),
                        "sim_to_up_median": float(np.median(sim_up[idx]))
                        if len(idx)
                        else None,
                    }
                )
            )

        # Group ablations: neutralize embedding block vs meta block (scaled ~0).
        variants: dict[str, np.ndarray] = {
            "embeddings_zeroed": np.hstack(
                [
                    np.zeros_like(feats[:, :emb_dim]),
                    np.clip(scaler.transform(feats[:, emb_dim:]), -2.5, 2.5),
                ]
            )
        }

        def gap_report(name: str, m: np.ndarray) -> None:
            order = list(np.argsort(m)[::-1])
            lw_ranks = sorted(order.index(i) for i in lw_idx) if lw_idx else []
            print(
                json.dumps(
                    {
                        "ablation": name,
                        "lw_minus_hn_gap": (
                            float(np.median(m[lw_idx])) - float(np.median(m[hn_idx]))
                            if lw_idx and hn_idx
                            else None
                        ),
                        "lw_mean_rank": (
                            float(sum(lw_ranks) / len(lw_ranks)) if lw_ranks else None
                        ),
                        "lw_in_top10": sum(1 for r in lw_ranks if r < 10),
                    }
                )
            )

        gap_report("baseline", base_margin)
        # Scaled-space neutral: training means map to ~0 under the scaler.
        neutral_meta = np.zeros_like(feats[:, emb_dim:])
        variants["meta_neutralized"] = np.hstack([feats[:, :emb_dim], neutral_meta])
        for j, name in enumerate(META_NAMES):
            single = np.clip(scaler.transform(feats[:, emb_dim:]), -2.5, 2.5)
            single[:, j] = 0.0
            variants[f"meta_drop_{name}"] = np.hstack([feats[:, :emb_dim], single])
        for name, matrix in variants.items():
            gap_report(name, margins(matrix))
    finally:
        db.close()


if __name__ == "__main__":
    main()
