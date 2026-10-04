"""Do votes on the stories the model is least sure about (Explore's Unsure)
teach the ranker more than other votes?

Replays one profile's votes in time order on a frozen snapshot (the DB file
is copied and opened read-only). For each split the model trains on the
oldest votes (base), then takes N more votes from the next slice (pool) by a
strategy, and the production ranker is scored on the newest votes (test):

- entropy: highest entropy of the base model's down/neutral/up probabilities
  on the pool (what Unsure picks);
- random: N random pool votes, several seeds;
- top: the pool votes the base model scores highest (what Recommended shows);
- none / all: no pool votes / every pool vote (bounds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Config  # noqa: E402
from scripts.eval_ranker_variants import (  # noqa: E402
    _embedding_text_hashes,
    _make_fold,
    _metrics,
    _production_scores,
    _validated_embeddings,
    frozen_database,
)

METRICS = ("auc_up_vs_rest", "auc_up_vs_down", "ndcg_at_12", "up_recall_at_40")


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-db", required=True)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["0.6:0.2", "0.4:0.2"],
        help="base_frac:pool_frac per split; test is the next 20%% after the pool",
    )
    parser.add_argument("--budgets", nargs="*", type=int, default=[50, 150])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="eval_unsure_votes.json")
    args = parser.parse_args()

    config = Config.load()
    with frozen_database(args.snapshot_db) as (db, digest):
        stories, labels, times = db.get_feedback_for_training(user_id=args.user_id)
        y = np.array(labels, dtype=int)
        vote_times = np.array(times, dtype=np.float64)
        hashes = dict(
            zip([s.id for s in stories], _embedding_text_hashes(stories), strict=True)
        )
        emb = _validated_embeddings(
            stories,
            db.get_embeddings_batch(
                [s.id for s in stories], config.embedding_model_version, hashes
            ),
        )
        order = np.argsort(vote_times, kind="stable")
        positions = np.arange(len(stories))

        def fold(train: np.ndarray, test: np.ndarray):
            return _make_fold(
                stories,
                emb,
                stories,
                positions,
                vote_times,
                y,
                positions,
                train,
                test,
                config,
                feedback_embeddings=emb,
                needs_experimental=False,
                judged_only=True,
            )

        def evaluate(train: np.ndarray, test: np.ndarray) -> dict[str, float]:
            f = fold(train, test)
            scores, probs = _production_scores(f, config, db)
            raw = _metrics(scores, f, config, probs, source_db=db)["raw"]
            return {name: raw[name] for name in METRICS}

        split_results: dict[str, dict[str, object]] = {}
        n = len(order)
        for spec in args.splits:
            base_frac, pool_frac = (float(x) for x in spec.split(":"))
            base_end = int(n * base_frac)
            pool_end = int(n * (base_frac + pool_frac))
            test_end = min(n, int(n * (base_frac + pool_frac + 0.2)))
            base, pool, test = (
                order[:base_end],
                order[base_end:pool_end],
                order[pool_end:test_end],
            )
            print(
                f"split {spec}: base {len(base)}, pool {len(pool)} "
                f"(up {int((y[pool] == 2).sum())}), test {len(test)}",
                flush=True,
            )
            # The base model's view of the pool: scores and probabilities.
            pool_fold = fold(base, pool)
            pool_scores, pool_probs = _production_scores(pool_fold, config, db)
            if pool_probs is None:
                raise RuntimeError("Base model has no probabilities (SVM not fit)")
            by_id = {s.id: i for i, s in enumerate(pool_fold.candidates)}
            kept = np.array([p for p in pool if stories[p].id in by_id])
            rows = np.array([by_id[stories[p].id] for p in kept])
            entropy_rank = kept[np.argsort(-_entropy(pool_probs[rows]), kind="stable")]
            top_rank = kept[np.argsort(-pool_scores[rows], kind="stable")]

            results: dict[str, object] = {
                "none": evaluate(base, test),
                "all": evaluate(np.concatenate([base, pool]), test),
            }
            print(f"  none {results['none']}\n  all  {results['all']}", flush=True)
            for budget in args.budgets:
                picks = {
                    "entropy": entropy_rank[:budget],
                    "top": top_rank[:budget],
                }
                for name, chosen in picks.items():
                    results[f"{name}_{budget}"] = evaluate(
                        np.concatenate([base, chosen]), test
                    ) | {"ups": int((y[chosen] == 2).sum())}
                    print(
                        f"  {name}_{budget} {results[f'{name}_{budget}']}", flush=True
                    )
                randoms = []
                for seed in range(args.seeds):
                    chosen = np.random.default_rng(seed).choice(
                        kept, size=min(budget, len(kept)), replace=False
                    )
                    randoms.append(
                        evaluate(np.concatenate([base, chosen]), test)
                        | {"ups": int((y[chosen] == 2).sum())}
                    )
                summary = {
                    name: {
                        "mean": float(np.mean([r[name] for r in randoms])),
                        "sd": float(np.std([r[name] for r in randoms], ddof=1))
                        if len(randoms) > 1
                        else 0.0,
                    }
                    for name in (*METRICS, "ups")
                }
                results[f"random_{budget}"] = {"runs": randoms, "summary": summary}
                print(f"  random_{budget} {summary}", flush=True)
            split_results[spec] = results
    report = {"snapshot_sha256": digest, "splits": split_results}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
