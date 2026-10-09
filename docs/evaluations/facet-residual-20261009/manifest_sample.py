"""Facet-residual step 0: freeze rubric + OOF blocks + 280 sample + 99 dev + repeats.

Agreed design (facet-design review overrides alternatives on conflicts):
- 3848 older votes cut into 5 chronological EQUAL-COUNT blocks by vote time.
  99 prior-replay dev stories excluded from the old pool FIRST (disjoint check).
- Seed 20261009 uniform 70 stories per block from blocks 2-5, outcome-blind
  (no labels, no source, no scores involved): 280 total. Then 20 repeats
  drawn uniformly from the 379 (280+99) via seeded stream. No redraws.
- Shuffle 399 items into 20 batches (~20 each) with opaque IDs. Labels never
  enter batches. Feasibility gate BEFORE any model call: each training class
  (b2-3, b2-4, all280) >=10, each test class (b4, b5, dev99) >=5, else print
  INFEASIBLE and stop (no annotation).
- rubric.json frozen + hashed before any calls. Actual datetime.utcnow freeze.
- Snapshot is existing task-owned input: opened READONLY here. No writes to
  shared cache / main / live DB. Writes stay in this dir (sample.json,
  rubric.json, manifest.json).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import random
import sqlite3
import string
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path("/tmp/opencode/cc-replay/snapshot.db")
SEED = 20261009
USER_ID = 151
EVAL_NOW = 1791401987.9873054
WINDOW_END = 1791511537.886615
N_BLOCKS = 5
PER_BLOCK = 70
N_REPEATS = 20
N_BATCHES = 20

LABEL = {"down": 0, "neutral": 1, "up": 2}

RUBRIC = {
    "sees": ["title", "domain", "source", "body1500"],
    "never_sees": [
        "comments",
        "top_comments",
        "votes",
        "labels",
        "profile",
        "baseline_scores",
        "block_ids",
        "story_ids",
    ],
    "body_chars": 1500,
    "body_priority": ["self_text", "article_body", "text_content"],
    "output": "strict JSON array, one object per opaque item id",
    "fields": {
        "artifact": [
            "paper",
            "repo_tool",
            "product_launch",
            "news_report",
            "essay_opinion",
            "personal_narrative",
            "howto",
            "question_discussion",
            "social_post",
            "data_visual",
            "media",
            "other",
            "unclear",
        ],
        "claim": [
            "new_result",
            "release",
            "incident_event",
            "argument",
            "explainer",
            "retrospective",
            "unclear",
        ],
        "technical_depth": ["0", "1", "2", "3", "unclear"],
        "voice": ["firsthand", "aggregator_commentary", "unclear"],
        "actionable": ["0", "1", "2", "unclear"],
        "conflict_framing": ["0", "1", "2", "unclear"],
        "entity_focus": [
            "company",
            "person",
            "government",
            "oss",
            "science",
            "consumer_product",
            "none",
            "unclear",
        ],
    },
    "rule": "pick exactly one option per field; use unclear only when the "
    "visible title+body genuinely does not decide it; judge ONLY "
    "the item text, never votes or popularity.",
}


def _opaque(n: int, rng: random.Random) -> list[str]:
    alphabet = string.ascii_lowercase + string.digits
    out: list[str] = []
    while len(out) < n:
        cand = "F-" + "".join(rng.choice(alphabet) for _ in range(6))
        if cand not in out:
            out.append(cand)
    return out


def main() -> int:
    assert SNAPSHOT.exists(), f"missing snapshot {SNAPSHOT}"
    frozen_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (HERE / "rubric.json").write_text(json.dumps(RUBRIC, indent=1, sort_keys=True))

    con = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro", uri=True)
    old = con.execute(
        "SELECT story_id, action, updated_at FROM feedback "
        "WHERE user_id=? AND updated_at<=? ORDER BY updated_at",
        (USER_ID, EVAL_NOW),
    ).fetchall()
    dev = con.execute(
        "SELECT story_id, action, updated_at FROM feedback "
        "WHERE user_id=? AND updated_at>? AND updated_at<=? ORDER BY updated_at",
        (USER_ID, EVAL_NOW, WINDOW_END),
    ).fetchall()
    con.close()
    assert len(old) == 3848, f"expected 3848 old votes, got {len(old)}"
    assert len(dev) == 99, f"expected 99 dev votes, got {len(dev)}"
    old_ids = [r[0] for r in old]
    dev_ids = [r[0] for r in dev]
    assert not (set(old_ids) & set(dev_ids)), "dev leaked into old pool"
    assert max(r[2] for r in old) <= EVAL_NOW < min(r[2] for r in dev), (
        "replay cutoff is not before every dev vote"
    )
    labels = {r[0]: LABEL[r[1]] for r in old + dev}

    # 5 equal-count chronological blocks over the 3848 (quintiles).
    n = len(old)
    bounds: list[int] = [n * k // N_BLOCKS for k in range(N_BLOCKS + 1)]
    blocks: dict[str, list[int]] = {}
    block_of: dict[int, int] = {}
    for b in range(1, N_BLOCKS + 1):
        ids = old_ids[bounds[b - 1] : bounds[b]]
        blocks[f"block{b}"] = ids
        for sid in ids:
            block_of[sid] = b
    cut_times = [old[bounds[k]][2] for k in range(1, N_BLOCKS)]

    # Uniform outcome-blind draw: RNG over position-shuffled id lists only.
    rng = random.Random(SEED)
    sample: dict[str, list[int]] = {}
    for b in (2, 3, 4, 5):
        pool = sorted(blocks[f"block{b}"])
        rng.shuffle(pool)
        sample[f"block{b}"] = sorted(pool[:PER_BLOCK])
    s280 = sorted(s for b in (2, 3, 4, 5) for s in sample[f"block{b}"])
    assert len(s280) == 280 and len(set(s280)) == 280
    assert not (set(s280) & set(dev_ids)), "sample hit dev pool"

    pool379 = sorted(s280 + dev_ids)
    rng_r = random.Random(SEED + 1)
    order = list(pool379)
    rng_r.shuffle(order)
    repeats = sorted(order[:N_REPEATS])

    # Feasibility gate on labels (checked once, NO redrawing).
    def counts(ids: list[int]) -> dict[str, int]:
        c = {"down": 0, "neutral": 1, "up": 2}
        inv = {v: k for k, v in c.items()}
        out = {"down": 0, "neutral": 0, "up": 0}
        for sid in ids:
            out[inv[labels[sid]]] += 1
        return out

    train_sets = {
        "train_b2_b3": sorted(sample["block2"] + sample["block3"]),
        "train_b2_b4": sorted(sample["block2"] + sample["block3"] + sample["block4"]),
        "train_all280": s280,
    }
    test_sets = {
        "test_b4": sorted(sample["block4"]),
        "test_b5": sorted(sample["block5"]),
        "test_dev99": sorted(dev_ids),
    }
    coverage = {k: counts(v) for k, v in {**train_sets, **test_sets}.items()}
    feasible = all(v >= 10 for k in train_sets for v in coverage[k].values()) and all(
        v >= 5 for k in test_sets for v in coverage[k].values()
    )

    # 20 shuffled batches over 399 items (379 unique + 20 repeat occurrences).
    items = [
        {
            "story_id": sid,
            "kind": "dev" if sid in set(dev_ids) else "sample",
            "repeat_of": None,
        }
        for sid in pool379
    ]
    items += [{"story_id": sid, "kind": "repeat", "repeat_of": sid} for sid in repeats]
    rng_b = random.Random(SEED + 2)
    rng_b.shuffle(items)
    opaques = _opaque(len(items), random.Random(SEED + 3))
    for it, op in zip(items, opaques):
        it["opaque"] = op
    batches = [items[i::N_BATCHES] for i in range(N_BATCHES)]
    assert all(len(b) <= 20 for b in batches), "batch exceeds 20 stories"
    assert sum(len(b) for b in batches) == 399

    def _h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    sample_doc = {
        "seed": SEED,
        "user_id": USER_ID,
        "eval_now": EVAL_NOW,
        "window_end": WINDOW_END,
        "n_old": len(old),
        "n_dev": len(dev),
        "block_sizes": {k: len(v) for k, v in blocks.items()},
        "block_cut_times": cut_times,
        "blocks": {k: sorted(blocks[k]) for k in blocks},
        "sample70": {k: sample[k] for k in sorted(sample)},
        "s280": s280,
        "dev99": sorted(dev_ids),
        "repeats20": repeats,
        "batches": [
            [{k: it[k] for k in ("opaque", "story_id", "kind")} for it in b]
            for b in batches
        ],
        "coverage": coverage,
        "feasible": feasible,
        "provenance": {
            "old_time_range": [old[0][2], old[-1][2]],
            "dev_time_range": [dev[0][2], dev[-1][2]],
            "dev_disjoint_from_old": True,
            "replay_cutoff_before_all_dev": True,
            "no_redraws": True,
            "outcome_blind_draw": True,
        },
    }
    (HERE / "sample.json").write_text(json.dumps(sample_doc, indent=1))
    manifest = {
        "frozen_at": frozen_at,
        "seed": SEED,
        "feasible": feasible,
        "hashes": {
            "rubric.json": _h(HERE / "rubric.json"),
            "sample.json": _h(HERE / "sample.json"),
        },
        "quota": "annotation <=20 batches/requests incl retries; "
        "20 seeded repeats as separate calls",
        "gates": "arm3-arm1>=+0.02 pooled; arm3>p95 null; arm4-arm2>=+0.01; "
        "dev99 delta>=0; stability>=85% (repeats-only, pre-metrics); "
        "arm3>=arm0. STOP if arm3-arm1<+0.01 or arm3<=proxy. "
        "CI-includes-0 => promising, inconclusive. GO = larger "
        "labeling only, no production claim.",
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {"frozen_at": frozen_at, "feasible": feasible, "coverage": coverage},
            indent=1,
        )
    )
    if not feasible:
        print("INFEASIBLE: class coverage gate failed; STOP, no annotation.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
