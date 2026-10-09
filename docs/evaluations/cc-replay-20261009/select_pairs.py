"""Step-2 pair selection (OUTCOME-AWARE by approved design; development only).

Reads replay_scores.json (ML scores + observed labels) and the task-owned
SNAPSHOT (ro) for the training-only taste profile. Live DB never opened.

Pool: informative ordinal pairs (different true labels), same source;
hn-hn requires >=1 shared title token (stopwords removed) as topic proxy;
ML margin |dOrdinal| in [0.10, 0.90) (moderate: excludes ties and extremes).
Errors: ML order disagrees with label order. Controls: ML order agrees.
Target 5 errors (all available: 3 hn + 2 non-hn) + 5 controls (hn), one
appearance per story (cap 1), seeded greedy. Reversed checks: 2 err + 2 ctl.

Writes (this dir only): selection.json, preregistration.json (freeze),
profile.txt (PRIVATE, training votes only), settings_pairs.json.
Prompts built later by score_muse.py stay outcome-blinded (no labels,
scores, strata); opaque IDs; labels/scores never enter prompts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path("/tmp/opencode/cc-replay/snapshot.db")
USER_ID = 151
EVAL_NOW = 1791401987.9873054
SEED = 20261011
LO, HI = 0.10, 0.90
N_REV_ERR, N_REV_CTL = 2, 2

STOP = set(
    "the a an of to in on for and or with is are was were be by as at "
    "from that this it its into how what when why who do does did can "
    "could should would will you your i we they he she them their our "
    "my me us out up over under more most less new vs v s".split()
)


class _PoolEntry(TypedDict):
    a: int
    b: int
    margin: float
    err: bool
    hn: bool
    dirs: tuple[int, int]


class _StratumPair(_PoolEntry):
    stratum: str


class _PairFull(_StratumPair):
    pair_id: str
    order: list[str]


def toks(t: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2
    }


def sgn(x: float) -> int:
    return (x > 0) - (x < 0)


def domain_of(url: str | None) -> str:
    return urlparse(url).netloc.removeprefix("www.") if url else ""


def main() -> None:
    rng = random.Random(SEED)
    rows = json.loads((HERE / "replay_scores.json").read_text())
    by_id = {r["story_id"]: r for r in rows}

    pool: list[_PoolEntry] = []
    ids = sorted(by_id)
    for a_id, b_id in itertools.combinations(ids, 2):
        a, b = by_id[a_id], by_id[b_id]
        if a["true_label"] == b["true_label"] or a["source"] != b["source"]:
            continue
        if a["source"] == "hn" and not (toks(a["title"]) & toks(b["title"])):
            continue
        m = abs(a["ordinal_up_minus_down"] - b["ordinal_up_minus_down"])
        if not (LO <= m < HI):
            continue
        err = sgn(a["ordinal_up_minus_down"] - b["ordinal_up_minus_down"]) != sgn(
            a["true_label"] - b["true_label"]
        )
        pool.append(
            {
                "a": a_id,
                "b": b_id,
                "margin": m,
                "err": err,
                "hn": a["source"] == "hn",
                "dirs": (
                    min(a["true_label"], b["true_label"]),
                    max(a["true_label"], b["true_label"]),
                ),
            }
        )

    errs = sorted(
        [p for p in pool if p["err"]], key=lambda p: (p["margin"], p["a"], p["b"])
    )
    ctls = sorted(
        [p for p in pool if not p["err"]], key=lambda p: (p["margin"], p["a"], p["b"])
    )
    print(f"pool={len(pool)} err={len(errs)} ctl={len(ctls)}")

    # --- greedy cap-1 pick: all errors, then controls avoiding used stories ---
    used: set[int] = set()
    picked_err: list[_PoolEntry] = []
    picked_ctl: list[_PoolEntry] = []
    for p in errs:
        if p["a"] not in used and p["b"] not in used:
            picked_err.append(p)
            used |= {p["a"], p["b"]}
    # controls: direction-diverse first, then seeded shuffle
    ctl_pref = sorted(ctls, key=lambda p: (p["dirs"] != (1, 2), rng.random()))
    for p in ctl_pref:
        if len(picked_ctl) >= len(picked_err):
            break
        if p["a"] not in used and p["b"] not in used:
            picked_ctl.append(p)
            used |= {p["a"], p["b"]}
    print(
        f"picked err={len(picked_err)} ctl={len(picked_ctl)} "
        f"max_reuse=1 stories={len(used)}"
    )

    base_pairs: list[_StratumPair] = [
        _StratumPair(
            a=p["a"],
            b=p["b"],
            margin=p["margin"],
            err=p["err"],
            hn=p["hn"],
            dirs=p["dirs"],
            stratum="error",
        )
        for p in picked_err
    ] + [
        _StratumPair(
            a=p["a"],
            b=p["b"],
            margin=p["margin"],
            err=p["err"],
            hn=p["hn"],
            dirs=p["dirs"],
            stratum="control",
        )
        for p in picked_ctl
    ]
    pairs: list[_PairFull] = []
    for k, p in enumerate(base_pairs, 1):
        r = random.Random(SEED + k)
        pairs.append(
            _PairFull(
                a=p["a"],
                b=p["b"],
                margin=p["margin"],
                err=p["err"],
                hn=p["hn"],
                dirs=p["dirs"],
                stratum=p["stratum"],
                pair_id=f"p-{k:02d}",
                order=["a", "b"] if r.random() < 0.5 else ["b", "a"],
            )
        )

    rev_pool_err = [p["pair_id"] for p in pairs if p["stratum"] == "error"]
    rev_pool_ctl = [p["pair_id"] for p in pairs if p["stratum"] == "control"]
    revs = rng.sample(rev_pool_err, min(N_REV_ERR, len(rev_pool_err))) + rng.sample(
        rev_pool_ctl, min(N_REV_CTL, len(rev_pool_ctl))
    )

    # --- training-only taste profile from snapshot (ro); PRIVATE ---
    con = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro", uri=True)
    blocks = []
    for action, limit in (("up", 50), ("down", 50), ("neutral", 10)):
        rs = con.execute(
            """SELECT s.source, s.url, s.title FROM feedback f
               JOIN stories s ON s.id = f.story_id
               WHERE f.user_id = ? AND f.action = ? AND f.updated_at <= ?
               ORDER BY f.updated_at DESC LIMIT ?""",
            (USER_ID, action, EVAL_NOW, limit),
        ).fetchall()
        lines = [
            f"- [{src}{' ' + domain_of(url) if domain_of(url) else ''}] {t}"
            for src, url, t in rs
        ]
        blocks.append(
            f"## {action.upper()} ({len(lines)} most recent)\n" + "\n".join(lines)
        )
    con.close()
    (HERE / "profile.txt").write_text("\n\n".join(blocks))

    selection = {
        "seed": SEED,
        "margin_band": [LO, HI],
        "pool": {"n": len(pool), "err": len(errs), "ctl": len(ctls)},
        "note": "OUTCOME-AWARE selection (errors/controls need labels): "
        "selection-biased DEVELOPMENT ONLY; OLD99 already inspected, "
        "not independent validation. Prompts stay outcome-blinded.",
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "stratum": p["stratum"],
                "margin": round(p["margin"], 4),
                "order": p["order"],
                "stories": {
                    "a": {k: by_id[p["a"]][k] for k in ("story_id", "title", "source")},
                    "b": {k: by_id[p["b"]][k] for k in ("story_id", "title", "source")},
                },
            }
            for p in pairs
        ],
        "reversed_pair_ids": revs,
    }
    (HERE / "selection.json").write_text(json.dumps(selection, indent=1))

    def _h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    prereg = {
        "frozen_at": "2026-10-09T12:30:00Z",
        "baseline": "replay.py production-config replay (settings.json); "
        "ordinal endpoint P(up)-P(down), UP-vs-rest P(up)",
        "go_gate": "Muse corrects >=60% of selected error pairs AND agrees "
        "with ML on >=70% of control pairs AND reversed stability "
        ">=3/4. Exploratory, not proof.",
        "feature_decision_rule": "If gate passes: cluster Muse-fixed errors by "
        "observable text/source/novelty-vs-prior-ups pattern; lock ONE "
        "deterministic feature already expressible in pipeline (e.g. "
        "age-at-vote recency, title-novelty vs training ups, text-length) "
        "with pre-signed direction; general source onehots already exist "
        "and must not be reinvented; never fit to LLM rationales alone; "
        "sign BEFORE opening validation outcomes. Else REJECT, no challenger.",
        "quota_cap": "fresh Muse ranking: <=4 batches total, <=24 comparisons "
        "incl reversed (this step: <=10 fwd + <=4 rev).",
        "hashes": {
            "selection.json": _h(HERE / "selection.json"),
            "profile.txt": _h(HERE / "profile.txt"),
            "replay_scores.json": _h(HERE / "replay_scores.json"),
        },
    }
    (HERE / "preregistration.json").write_text(json.dumps(prereg, indent=1))
    (HERE / "settings_pairs.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "margin_band": [LO, HI],
                "n_pairs": len(pairs),
                "n_rev": len(revs),
                "model": "opencode-go/muse-spark-1.3-contributor",
            },
            indent=1,
        )
    )

    for p in pairs:
        a, b = by_id[p["a"]], by_id[p["b"]]
        hi = a if a["true_label"] > b["true_label"] else b
        lo_s = b if hi is a else a
        print(
            f"{p['pair_id']} {p['stratum']:7s} m={p['margin']:.3f} "
            f"dirs={p['dirs']} MLpick={'AB'[int((a['ordinal_up_minus_down'] < b['ordinal_up_minus_down']))]} "
            f"true-hi={hi['story_id']} [{hi['source']}] {hi['title'][:70]}"
        )
        print(
            f"           true-lo={lo_s['story_id']} [{lo_s['source']}] {lo_s['title'][:70]}"
        )
    print("reversed:", revs)


if __name__ == "__main__":
    main()
