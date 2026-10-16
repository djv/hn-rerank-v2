"""Robustness for novelty-replay: Recommended-only per-day top-K, and a placebo
with novelty shuffled within each day (200 shuffles). Reads OUT.json only."""

import json
import sys

import numpy as np

rows = json.load(open(sys.argv[1]))["rows"]
LAMS = (0.0, 0.1, 0.15, 0.2)


def rankpct(v):
    o = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return o / max(len(v) - 1, 1)


def run(sel, k, rng=None):
    days = sorted({r["day"] for r in sel})
    res = {lam: np.zeros(3) for lam in LAMS}
    for d in days:
        rs = [r for r in sel if r["day"] == d]
        sc = np.array([r["score"] for r in rs])
        nv = np.array([1 - (r["maxsim_up"] if r["maxsim_up"] is not None else 0.4) for r in rs])
        if rng is not None:
            nv = rng.permutation(nv)
        up = np.array([r["outcome"] == "up" for r in rs])
        dn = np.array([r["outcome"] == "down" for r in rs])
        novel = up & np.array([(r["maxsim_up"] or 0) < 0.6 for r in rs])
        for lam in LAMS:
            b = (1 - lam) * rankpct(sc) + lam * rankpct(nv)
            top = np.argsort(-b, kind="stable")[:k]
            res[lam] += [up[top].sum(), dn[top].sum(), novel[top].sum()]
    return {lam: v / len(days) for lam, v in res.items()}, len(days)


rec = [r for r in rows if r["view"] == "recommended"]
for name, sel, k in (("recommended top-6", rec, 6), ("recommended top-12", rec, 12), ("all top-12", rows, 12)):
    real, nd = run(sel, k)
    rng = np.random.default_rng(0)
    plac = [run(sel, k, rng)[0] for _ in range(200)]
    print(f"{name} ({nd} days)  per-day means ups/downs/novel; placebo = novelty shuffled within day")
    for lam in LAMS:
        p = np.array([x[lam] - x[0.0] for x in plac])
        dr = real[lam] - real[0.0]
        ups_ci = f"{np.percentile(p[:, 0], 2.5):+.2f},{np.percentile(p[:, 0], 97.5):+.2f}"
        dn_ci = f"{np.percentile(p[:, 1], 2.5):+.2f},{np.percentile(p[:, 1], 97.5):+.2f}"
        print(
            f"  lam={lam:.2f} real {real[lam][0]:.2f}/{real[lam][1]:.2f}/{real[lam][2]:.2f}"
            f"  d_ups {dr[0]:+.2f} (placebo 95% {ups_ci})  d_downs {dr[1]:+.2f} (placebo {dn_ci})"
        )
