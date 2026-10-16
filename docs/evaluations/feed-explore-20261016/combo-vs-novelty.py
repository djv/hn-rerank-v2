"""Read-only Combo (MMR-0.8 + far card) vs novelty-term replay.

Analysis only, no model training. Loads the day-forward scored rows of
revote/novelty-replay.py (E1.json rows plus E1.npz L2-normalized
embeddings) and, per UTC day in two scopes (all views pooled, Recommended
only), picks top K by seven rules: base; nov10/nov15 rankpct blends;
mmr80 greedy MMR lambda 0.8; far1 base head + far card; combo MMR head +
far card; combo+nov10 combo on the nov10 blend. Reports per-day ups /
downs / neutrals / novel ups / pairwise distance / topic spread, day-block
bootstrap deltas vs base, and a 200-draw placebo per rule. Read-only by
construction: the E1 files are the only input (DB never opened), raw
embeddings are never scaled, stdout holds aggregates only.
Run: uv run python revote/combo-vs-novelty.py E1.json E1.npz OUT.json [--k 12]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

NOVEL_CUT = 0.60
MMR_LAM = 0.8
N_CLUSTERS = 40
N_BOOT, BOOT_SEED = 2000, 0
N_PLACEBO, PLACEBO_SEED = 200, 1
NOV_LAM = {"nov10": 0.10, "nov15": 0.15}
RULES = ("base", "nov10", "nov15", "mmr80", "far1", "combo", "combo+nov10")
METRICS = ("ups", "downs", "neutrals", "novel", "div", "topics")
VIEWS = ("recommended", "popular", "explore")
OUTCOMES = ("up", "down", "neutral", "none")


@dataclass
class Row:
    story_id: int
    day: str
    view: str
    outcome: str
    score: float
    maxsim: float | None


def rankpct(vals: np.ndarray) -> np.ndarray:
    """Average-rank percentiles in [0, 1]; ties share their mean rank."""
    n = len(vals)
    if n == 1:
        return np.full(1, 0.5)
    _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg = (ends - counts + ends - 1) / 2.0 / (n - 1)
    return avg[inv]


def pct_fill(vals: np.ndarray, fill: float = 0.5) -> np.ndarray:
    out = np.full(len(vals), fill)
    mask = ~np.isnan(vals)
    if mask.any():
        out[mask] = rankpct(vals[mask])
    return out


def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or float(np.isnan(x)) else f"{x:.{nd}f}"


def nanmean(xs: list[float] | np.ndarray) -> float:
    v = np.asarray(xs, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def jnum(x: float) -> float | None:
    return None if float(np.isnan(x)) else float(x)


def load(e1_json: str, e1_npz: str) -> tuple[list[Row], dict[int, np.ndarray]]:
    data = json.loads(Path(e1_json).read_text())
    rows = [
        Row(int(r["story_id"]), str(r["day"]), str(r["view"]), str(r["outcome"]),
            float(r["score"]), None if r["maxsim_up"] is None else float(r["maxsim_up"]))
        for r in data["rows"]
    ]
    if not rows:
        raise ValueError(f"No rows in {e1_json}")
    for r in rows:
        if r.view not in VIEWS:
            raise ValueError(f"Unexpected view {r.view!r}")
        if r.outcome not in OUTCOMES:
            raise ValueError(f"Unexpected outcome {r.outcome!r}")
    z = np.load(e1_npz)
    ids = np.asarray(z["story_ids"])
    mat = np.asarray(z["embeddings"], dtype=np.float64)
    if len(ids) != mat.shape[0] or set(ids.tolist()) != {r.story_id for r in rows}:
        raise ValueError("E1.npz story_ids do not match E1.json rows")
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    if not np.all(n > 0):
        raise ValueError("Zero-norm embedding in E1.npz")
    mat = mat / n
    return rows, {int(s): mat[k] for k, s in enumerate(ids)}


def fit_topics(vec: dict[int, np.ndarray]) -> dict[int, int]:
    ids = sorted(vec)
    k = min(N_CLUSTERS, len(ids))
    if k < 2:
        raise ValueError("Need at least 2 vectors for topic clusters")
    lab = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(
        np.stack([vec[i] for i in ids])
    )
    return {s: int(v) for s, v in zip(ids, lab)}


def analyze_scope(
    rows: list[Row], vec: dict[int, np.ndarray], lab: dict[int, int],
    idx: list[int], k: int, scope: str,
) -> dict:
    brng = np.random.default_rng(BOOT_SEED)
    prng = np.random.default_rng(PLACEBO_SEED)
    days = sorted({rows[i].day for i in idx})
    pools = {d: [i for i in idx if rows[i].day == d] for d in days}
    score = {i: rows[i].score for i in idx}
    novel = {i: (rows[i].maxsim is None or rows[i].maxsim < NOVEL_CUT) for i in idx}
    print(f"[{scope}] {len(days)} days, {len(idx)} stories, k={k}", flush=True)

    def topk(pool: list[int], val: dict[int, float]) -> list[int]:
        return sorted(pool, key=lambda i: (-val[i], rows[i].story_id))[: min(k, len(pool))]

    def blend(pool: list[int], lam: float) -> dict[int, float]:
        sc = np.array([rows[i].score for i in pool])
        nv = np.array([np.nan if rows[i].maxsim is None else 1.0 - rows[i].maxsim for i in pool])
        b = (1.0 - lam) * pct_fill(sc) + lam * pct_fill(nv)
        return {i: float(v) for i, v in zip(pool, b)}

    def pick_mmr(pool: list[int], val: dict[int, float]) -> list[int]:
        # Greedy lambda-MMR. Production mmr_filter (ranking.py:1919) is
        # threshold-discard (no lambda), so loop + min-max scaling follow
        # diversity-phase0b.py pick_mmr; cosine is the raw L2 dot as in
        # mmr_filter, tie normalisation matches _minmax01 (ranking.py:1191).
        order = sorted(pool, key=lambda i: (-val[i], rows[i].story_id))
        kk = min(k, len(order))
        if kk <= 1:
            return order[:kk]
        lo = min(val[i] for i in pool)
        hi = max(val[i] for i in pool)
        sc = {i: 0.5 if hi == lo else (val[i] - lo) / (hi - lo) for i in pool}
        picked = [order[0]]
        rest = set(order[1:])
        while len(picked) < kk:
            best: int | None = None
            bestkey: tuple[float, float, int] | None = None
            for c in sorted(rest):
                ms = max(float(vec[rows[c].story_id] @ vec[rows[p].story_id]) for p in picked)
                key = (MMR_LAM * sc[c] - (1.0 - MMR_LAM) * ms, val[c], -rows[c].story_id)
                if bestkey is None or key > bestkey:
                    best, bestkey = c, key
            if best is None:
                raise ValueError("MMR pick stalled with cards remaining")
            picked.append(best)
            rest.remove(best)
        return picked

    def pick_far(pool: list[int], val: dict[int, float]) -> list[int]:
        # Approximation of place_far_card (ranking.py:2373): far eligibility
        # is closest_up < 0.60 (:328,:2396), best-scored eligible wins
        # (:2419); no story times/horizon in the replay, so scan day order.
        order = sorted(pool, key=lambda i: (-val[i], rows[i].story_id))
        kk = min(k, len(order))
        if kk <= 1:
            return order[:kk]
        head = order[: kk - 1]
        hs = set(head)
        for c in order:
            if c not in hs and novel[c]:
                return head + [c]
        return order[:kk]

    def pick_combo(pool: list[int], val: dict[int, float]) -> list[int]:
        m = pick_mmr(pool, val)
        kk = min(k, len(pool))
        if kk <= 1:
            return m
        head = m[: kk - 1]
        hs = set(head)
        for c in sorted(pool, key=lambda i: (-val[i], rows[i].story_id)):
            if c not in hs and novel[c]:
                return head + [c]
        return m[:kk]

    def agg(sel: list[int]) -> dict:
        ups = sum(1 for i in sel if rows[i].outcome == "up")
        dns = sum(1 for i in sel if rows[i].outcome == "down")
        neu = sum(1 for i in sel if rows[i].outcome == "neutral")
        nov = sum(1 for i in sel if rows[i].outcome == "up" and novel[i])
        if len(sel) < 2:
            dv = float("nan")
        else:
            m = np.stack([vec[rows[i].story_id] for i in sel])
            iu = np.triu_indices(len(sel), 1)
            dv = float(np.mean(1.0 - (m @ m.T)[iu]))
        top = float(len({lab[rows[i].story_id] for i in sel})) if sel else float("nan")
        return {"n": len(sel), "ups": ups, "downs": dns, "neutrals": neu,
                "novel": nov, "div": dv, "topics": top}

    sel: dict[str, dict[str, list[int]]] = {r: {} for r in RULES}
    for d in days:
        pool = pools[d]
        b10 = blend(pool, NOV_LAM["nov10"])
        b15 = blend(pool, NOV_LAM["nov15"])
        day_sel = {
            "base": topk(pool, score), "nov10": topk(pool, b10), "nov15": topk(pool, b15),
            "mmr80": pick_mmr(pool, score), "far1": pick_far(pool, score),
            "combo": pick_combo(pool, score), "combo+nov10": pick_combo(pool, b10),
        }
        for r, s in day_sel.items():
            sel[r][d] = s

    aggs = {r: {d: agg(sel[r][d]) for d in days} for r in RULES}
    means = {r: {m: nanmean([aggs[r][d][m] for d in days]) for m in METRICS} for r in RULES}

    def boot_ci(diffs: list[float]) -> tuple[float, float, float]:
        v = np.array([x for x in diffs if not np.isnan(x)], dtype=float)
        if v.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        rep = v[brng.integers(0, v.size, size=(N_BOOT, v.size))].mean(axis=1)
        return (float(v.mean()), float(np.percentile(rep, 2.5)), float(np.percentile(rep, 97.5)))

    delta = {}
    for r in RULES:
        delta[r] = {}
        for m in METRICS:
            delta[r][m] = (0.0, 0.0, 0.0) if r == "base" else boot_ci(
                [aggs[r][d][m] - aggs["base"][d][m] for d in days])

    pb: dict[str, dict[str, float]] = {}
    for r in RULES[1:]:
        hist_u: list[float] = []
        hist_d: list[float] = []
        for _ in range(N_PLACEBO):
            su = 0.0
            sd = 0.0
            for d in days:
                pool = pools[d]
                kk = min(k, len(pool))
                base = sel["base"][d]
                if r in NOV_LAM:
                    lam = NOV_LAM[r]
                    ms = np.array([np.nan if rows[i].maxsim is None else float(rows[i].maxsim) for i in pool])
                    nv = 1.0 - ms[prng.permutation(len(pool)).tolist()]
                    sc = np.array([rows[i].score for i in pool])
                    b = (1.0 - lam) * pct_fill(sc) + lam * pct_fill(nv)
                    jp = sorted(range(len(pool)), key=lambda j: (-b[j], rows[pool[j]].story_id))
                    psel = [pool[j] for j in jp[:kk]]
                else:
                    rep = len(sel[r][d]) - len(set(sel[r][d]) & set(base))
                    out = [i for i in pool if i not in set(base)]
                    rr = min(rep, len(base), len(out))
                    if rr <= 0:
                        psel = list(base)
                    else:
                        drop = set(int(x) for x in prng.choice(sorted(base), rr, replace=False).tolist())
                        add = [int(x) for x in prng.choice(sorted(out), rr, replace=False).tolist()]
                        psel = [i for i in base if i not in drop] + add
                a = agg(psel)
                su += a["ups"] - aggs["base"][d]["ups"]
                sd += a["downs"] - aggs["base"][d]["downs"]
            hist_u.append(su / len(days))
            hist_d.append(sd / len(days))
        ru = means[r]["ups"] - means["base"]["ups"]
        rd = means[r]["downs"] - means["base"]["downs"]
        pb[r] = {
            "ups_pct": float(np.mean(np.array(hist_u) <= ru)),
            "downs_pct": float(np.mean(np.array(hist_d) <= rd)),
        }

    print(f"[{scope}] per-day means over {len(days)} days:", flush=True)
    print("rule         ups downs neut novel   div topics | d_ups [95%]       d_downs [95%]     | plac_up plac_dn", flush=True)
    for r in RULES:
        m = means[r]
        du, dd = delta[r]["ups"], delta[r]["downs"]
        p = pb.get(r)
        pstr = "n/a     n/a" if p is None else f"{p['ups_pct']:.3f}   {p['downs_pct']:.3f}"
        print(f"{r:11} {m['ups']:5.2f} {m['downs']:5.2f} {m['neutrals']:4.2f} {m['novel']:5.2f}"
              f" {fmt(m['div']):>5} {fmt(m['topics']):>6} | {du[0]:+.2f} [{du[1]:+.2f},{du[2]:+.2f}]"
              f" {dd[0]:+.2f} [{dd[1]:+.2f},{dd[2]:+.2f}] | {pstr}", flush=True)

    rules_out = {}
    for r in RULES:
        rules_out[r] = {
            "per_day": {d: {kk: (v if isinstance(v, int) else jnum(v)) for kk, v in aggs[r][d].items()} for d in days},
            "mean": {m: jnum(means[r][m]) for m in METRICS},
            "delta_vs_base": {m: [jnum(x) for x in delta[r][m]] for m in METRICS},
        }
    return {"days": days, "rules": rules_out, "placebo": pb}


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only Combo vs novelty-term replay.")
    p.add_argument("e1_json", type=str)
    p.add_argument("e1_npz", type=str)
    p.add_argument("out", type=str)
    p.add_argument("--k", type=int, default=12)
    ns = p.parse_args()
    if ns.k < 1:
        raise ValueError("--k must be >= 1")
    print(f"Loading {ns.e1_json} + {ns.e1_npz} ...", flush=True)
    rows, vec = load(ns.e1_json, ns.e1_npz)
    print(f"{len(rows)} rows, {len(vec)} vectors", flush=True)
    lab = fit_topics(vec)
    print(f"Topics: k={len(set(lab.values()))} (MiniBatchKMeans, seed 0)", flush=True)
    scopes = {
        "all": list(range(len(rows))),
        "rec": [i for i, r in enumerate(rows) if r.view == "recommended"],
    }
    if not scopes["rec"]:
        raise ValueError("Recommended scope is empty")
    out_scopes = {name: analyze_scope(rows, vec, lab, idx, ns.k, name) for name, idx in scopes.items()}
    payload = {
        "e1_json": ns.e1_json, "e1_npz": ns.e1_npz, "k": ns.k, "rules": list(RULES),
        "novel_cut": NOVEL_CUT, "mmr_lambda": MMR_LAM, "nov_lambdas": NOV_LAM,
        "n_clusters": N_CLUSTERS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "n_placebo": N_PLACEBO, "placebo_seed": PLACEBO_SEED,
        "approximations": "mmr80 is greedy lambda-MMR per diversity-phase0b.py "
        "(production mmr_filter is threshold-discard, no lambda); far card "
        "scans the full day order (no window times/horizon); None maxsim "
        "counts as novel; ties break by story_id.",
        "caveat": "Exposure-limited replay: only shown cards have outcomes; "
        "deeper or unshown picks are unscored.",
        "scopes": out_scopes,
    }
    Path(ns.out).write_text(json.dumps(payload))
    print(f"Wrote {ns.out}", flush=True)
    print("Caveat: exposure-limited replay: only shown cards have outcomes.", flush=True)


if __name__ == "__main__":
    main()
