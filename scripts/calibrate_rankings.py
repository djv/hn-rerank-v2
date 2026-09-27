#!/usr/bin/env python3
"""Order batches of 5 stories by hand to compare two rankers.

Each batch comes from unvoted current candidates where production and the
challenger blend (tuned SVM rank-blended with logistic regression, from the
2026-09-25 study) disagree most near the top. Type the story numbers best to
worst (e.g. ``31524``); ``s`` skips a batch, ``q`` quits. Orders are appended
to a JSONL file together with both rankers' scores, so ``--report`` can say
which ranker agrees with your orderings more often.

Runs against a read-only DB snapshot (never the live dashboard DB):

    uv run python scripts/calibrate_rankings.py --db ~/.local/state/hn-rerank-eval/snapshot-20260925.db
    uv run python scripts/calibrate_rankings.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Story  # noqa: E402
from pipeline import Config, load_production_candidate_stories  # noqa: E402
from scripts.eval_ranker_variants import (  # noqa: E402
    _embedding_text_hashes,
    _make_fold,
    _percentile_scores,
    _production_scores,
    _scores_logreg,
    _weighted_rank_blend,
    _scores_knn_up_minus_down,
    frozen_database,
)

DEFAULT_LOG = Path.home() / ".local/state/hn-rerank-eval/calibration.jsonl"
DEFAULT_CACHE = Path.home() / ".local/state/hn-rerank-eval/calibration-scores.json"
BATCH = 5
TOP_N = 60  # disagreements only matter near the top of either ranking
CHALLENGER = {"svm_c": 4.0, "svm_gamma": 0.05}
CHALLENGER_LR_WEIGHT = 0.4


@dataclass(frozen=True)
class Scored:
    story: Story
    production: float  # percentile, 1 = top
    challenger: float
    # Unit-length story embedding; spreads a batch across topics.
    embedding: np.ndarray | None = field(default=None, compare=False, repr=False)


def score_candidates(db_path: Path, config: Config, user_id: int) -> list[Scored]:
    config = replace(config, db_path=str(db_path))
    with frozen_database(str(db_path)) as (db, _digest):
        fb_stories, labels, vote_times = db.get_feedback_for_training(user_id=user_id)
        candidates = load_production_candidate_stories(
            db, config, user_id=user_id, exclude_feedback=True
        )
        version = config.embedding_model_version

        def vectors(stories: list[Story]) -> dict[int, np.ndarray]:
            hashes = dict(
                zip(
                    [s.id for s in stories],
                    _embedding_text_hashes(stories),
                    strict=True,
                )
            )
            return db.get_embeddings_batch([s.id for s in stories], version, hashes)

        fb_vectors = vectors(fb_stories)
        keep = [i for i, s in enumerate(fb_stories) if s.id in fb_vectors]
        fb_stories = [fb_stories[i] for i in keep]
        y = np.array([labels[i] for i in keep], dtype=int)
        times = np.array([vote_times[i] for i in keep], dtype=np.float64)
        cand_vectors = vectors(candidates)
        candidates = [s for s in candidates if s.id in cand_vectors]
        cand_emb = np.stack([cand_vectors[s.id] for s in candidates])
        fold = _make_fold(
            candidates,
            cand_emb,
            fb_stories,
            np.full(len(fb_stories), -1),
            times,
            y,
            np.arange(len(fb_stories)),
            np.arange(len(fb_stories)),
            np.empty(0, dtype=int),
            config,
            feedback_embeddings=np.stack([fb_vectors[s.id] for s in fb_stories]),
        )
        production = _production_scores(fold, config, db)[0]
        tuned = replace(config, model=replace(config.model, **CHALLENGER))
        challenger = _weighted_rank_blend(
            {
                "production": _production_scores(fold, tuned, db)[0],
                "logreg": _scores_logreg(fold, config, target="up_minus_down")[0],
                "knn": _scores_knn_up_minus_down(fold, config)[0],
            },
            lr_weight=CHALLENGER_LR_WEIGHT,
            knn_weight=0.0,
        )
    prod_pct = _percentile_scores(production)
    chal_pct = _percentile_scores(challenger)
    return [
        Scored(story, float(p), float(c), emb)
        for story, p, c, emb in zip(
            fold.candidates, prod_pct, chal_pct, fold.cand_emb, strict=True
        )
    ]


def cached_scores(
    db_path: Path, config_path: str, config: Config, user_id: int, cache: Path
) -> list[Scored]:
    """score_candidates, reused while the snapshot, config and challenger match."""
    db_stat = db_path.stat()
    key = {
        "db": str(db_path.resolve()),
        "db_size": db_stat.st_size,
        "db_mtime": db_stat.st_mtime,
        "config": Path(config_path).read_text() if Path(config_path).exists() else "",
        "user_id": user_id,
        "challenger": CHALLENGER,
        "lr_weight": CHALLENGER_LR_WEIGHT,
    }
    vectors_path = cache.with_suffix(".npy")
    if cache.exists() and vectors_path.exists():
        data = json.loads(cache.read_text())
        if data["key"] == key:
            vectors = np.load(vectors_path, allow_pickle=False)
            return [
                Scored(Story(**row["story"]), row["production"], row["challenger"], emb)
                for row, emb in zip(data["scored"], vectors, strict=True)
            ]
    print("Scoring candidates with both rankers (cached after this run)…", flush=True)
    scored = score_candidates(db_path, config, user_id)
    rows = [
        {
            # Only the snippet is ever shown; keep the cache small.
            "story": asdict(replace(s.story, text_content=s.story.text_content[:400]))
            | {"self_text": "", "top_comments": "", "article_body": ""},
            "production": s.production,
            "challenger": s.challenger,
        }
        for s in scored
    ]
    vectors = [s.embedding for s in scored if s.embedding is not None]
    if len(vectors) != len(scored):
        raise ValueError("score_candidates returned stories without embeddings")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, np.stack(vectors).astype(np.float32))
    cache.write_text(json.dumps({"key": key, "scored": rows}))
    return scored


def disagreement_pool(scored: list[Scored], seen: set[int]) -> list[Scored]:
    """Stories in either ranker's top TOP_N, most disagreement first."""
    n = len(scored)
    cutoff = 1.0 - TOP_N / max(n, 1)
    pool = [
        s
        for s in scored
        if s.story.id not in seen and max(s.production, s.challenger) >= cutoff
    ]
    return sorted(pool, key=lambda s: abs(s.production - s.challenger), reverse=True)


def _discordant(a: Scored, b: Scored) -> bool:
    """The two rankers put a and b in opposite order."""
    return (a.production - b.production) * (a.challenger - b.challenger) < 0


def next_batch(pool: list[Scored], rng: np.random.Generator) -> list[Scored]:
    """BATCH stories whose ordering tells us the most, shuffled.

    Query-by-committee: a pair the two rankers order differently is the
    only kind that separates them, and it is where a label moves either
    model most. Greedily add the story that makes the most such pairs with
    the batch so far, breaking ties toward a new topic (embedding distance)
    so one ordering doesn't spend five labels on near-duplicates.
    """
    head = pool[: max(4 * BATCH, len(pool) // 3)]
    if len(head) <= BATCH:
        return [head[int(i)] for i in rng.permutation(len(head))]
    batch = [head[int(rng.integers(min(BATCH, len(head))))]]
    while len(batch) < BATCH:

        def gain(item: Scored) -> float:
            pairs = sum(_discordant(item, other) for other in batch)
            spread = min(
                (
                    1.0 - float(item.embedding @ other.embedding)
                    for other in batch
                    if item.embedding is not None and other.embedding is not None
                ),
                default=0.0,
            )
            return pairs + 0.5 * spread + 0.1 * abs(item.production - item.challenger)

        rest = [s for s in head if all(s is not b for b in batch)]
        batch.append(max(rest, key=gain))
    return [batch[int(i)] for i in rng.permutation(len(batch))]


def parse_order(text: str, size: int) -> list[int] | None:
    """``"31524"`` -> [2, 0, 4, 1, 3]; None unless a permutation of 1..size."""
    digits = [c for c in text if not c.isspace() and c != ","]
    if len(digits) != size or not all(c.isdigit() for c in digits):
        return None
    order = [int(c) - 1 for c in digits]
    return order if sorted(order) == list(range(size)) else None


def pair_agreement(order: list[int], scores: list[float]) -> tuple[int, int]:
    """(agreeing pairs, total pairs) between a best-first order and scores."""
    rank = {item: position for position, item in enumerate(order)}
    agree = total = 0
    for a, b in combinations(range(len(order)), 2):
        if scores[a] == scores[b]:
            continue
        total += 1
        user_prefers_a = rank[a] < rank[b]
        agree += user_prefers_a == (scores[a] > scores[b])
    return agree, total


def show(batch: list[Scored]) -> None:
    now = time.time()
    for number, item in enumerate(batch, 1):
        s = item.story
        domain = urlsplit(s.url or "").hostname or s.source
        hours = max(0.0, (now - s.time) / 3600)
        age = (
            f"{hours:.0f}h"
            if hours < 48
            else f"{hours / 24:.0f}d"
            if hours < 24 * 365
            else f"{hours / 24 / 365:.1f}y"
        )
        print(f"\n[{number}] {s.title}")
        print(f"    {domain} · {s.score} pts · {s.comment_count or 0} comments · {age}")
        snippet = " ".join((s.text_content or "").split())[:220]
        if snippet:
            print(textwrap.indent(textwrap.fill(snippet, 88), "    "))


def run(args: argparse.Namespace) -> None:
    config = Config.load(args.config)
    scored = cached_scores(args.db, args.config, config, args.user_id, args.cache)
    seen: set[int] = set()
    if args.log.exists():
        for line in args.log.read_text().splitlines():
            seen.update(json.loads(line)["story_ids"])
    rng = np.random.default_rng()
    done = 0
    while True:
        pool = disagreement_pool(scored, seen)
        if len(pool) < BATCH:
            print("\nNo more disagreeing stories to order.")
            break
        batch = next_batch(pool, rng)
        print("\n" + "=" * 90)
        show(batch)
        answer = input("\nBest→worst (e.g. 31524), s=skip, q=quit: ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            seen.update(item.story.id for item in batch)
            continue
        order = parse_order(answer, len(batch))
        if order is None:
            print("Type each number 1-5 once, best first.")
            continue
        record = {
            "at": time.time(),
            "db": str(args.db),
            "story_ids": [item.story.id for item in batch],
            "order": order,
            "production": [item.production for item in batch],
            "challenger": [item.challenger for item in batch],
        }
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        seen.update(record["story_ids"])
        done += 1
        print(f"Saved ({done} this session).")
    report(args.log)


def report(log: Path) -> None:
    if not log.exists():
        print(f"No orderings yet in {log}")
        return
    records = [json.loads(line) for line in log.read_text().splitlines()]
    totals = {"production": [0, 0], "challenger": [0, 0]}
    wins = {"production": 0, "challenger": 0, "tie": 0}
    for record in records:
        batch = {}
        for name in totals:
            agree, total = pair_agreement(record["order"], record[name])
            totals[name][0] += agree
            totals[name][1] += total
            batch[name] = agree / total if total else 0.5
        if batch["production"] == batch["challenger"]:
            wins["tie"] += 1
        else:
            wins[max(batch, key=lambda k: batch[k])] += 1
    print(f"\n{len(records)} batches ordered ({log})")
    for name, (agree, total) in totals.items():
        share = agree / total if total else float("nan")
        print(f"  {name:11} agrees with you on {agree}/{total} pairs ({share:.0%})")
    print(
        f"  per batch: challenger better {wins['challenger']}, "
        f"production better {wins['production']}, tie {wins['tie']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, help="Read-only DB snapshot to score")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", action="store_true", help="Only print the report")
    args = parser.parse_args()
    if args.report:
        report(args.log)
    elif args.db is None:
        parser.error("--db is required unless --report")
    else:
        run(args)


if __name__ == "__main__":
    main()
