"""Read-only sameness trend (no model training): is the served feed getting
more same-y, and do far stories get upvoted less? First showings, outcomes as
in short-window-scores.py, maxsim_up plus weekly diversity and clusters from
stored embeddings. Run: revote/sameness-trend.py USER SINCE_ISO OUT.json
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, ".")

DB = "/home/dev/hn-rewrite/main/hn_rewrite.db"
REVOTE = (1791763112.0, 1791763181.0)
NOVEL_CUT = 0.60
VIEWS = ("recommended", "explore", "popular", "other")
BUCKET_LABELS = ("<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", ">=0.8")
K, SEED, CHUNK, MAX_PAIR_N = 40, 0, 500, 400


@dataclass
class Shown:
    story_id: int
    at: float
    view: str
    window: str
    outcome: str
    maxsim_up: float | None
    cluster: int | None


def parse_since(s: str) -> float:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC).timestamp()


def iso_week(ts: float) -> str:
    y, w, _ = dt.datetime.fromtimestamp(ts, dt.UTC).isocalendar()
    return f"{y}-W{w:02d}"


def bucket(ms: float) -> str:
    return BUCKET_LABELS[bisect.bisect_right((0.5, 0.6, 0.7, 0.8), ms)]


def wilson(
    k: int, n: int, z: float = 1.96
) -> tuple[float | None, float | None, float | None]:
    if not n:
        return (None, None, None)
    p, den = k / n, 1.0 + z * z / n
    mid = (p + z * z / (2.0 * n)) / den
    half = z * float(np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / den
    return (p, max(0.0, mid - half), min(1.0, mid + half))


def _avg_ranks(v: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - counts + ends - 1.0)[inv] / 2.0 + 1.0


def spearman(xs: list[float], ys: list[float]) -> float:
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    cx, cy = _avg_ranks(x) - 1.0, _avg_ranks(y) - 1.0
    cx, cy = cx - cx.mean(), cy - cy.mean()
    d = float(np.sqrt(float(cx @ cx) * float(cy @ cy)))
    return float((cx @ cy) / d) if d > 0 else float("nan")


def fmt(x: float | None, nd: int = 3) -> str:
    bad = x is None or (isinstance(x, float) and float(np.isnan(x)))
    return "n/a" if bad else f"{x:.{nd}f}"


def pairwise_div(vecs: list[np.ndarray], rng: np.random.Generator) -> float | None:
    if len(vecs) < 2:
        return None
    if len(vecs) > MAX_PAIR_N:
        vecs = [vecs[i] for i in rng.choice(len(vecs), MAX_PAIR_N, replace=False)]
    m = np.stack(vecs)
    return float(np.mean(1.0 - (m @ m.T)[np.triu_indices(len(vecs), 1)]))


def main() -> None:
    if len(sys.argv) != 4:
        raise ValueError("usage: sameness-trend.py USER SINCE_ISO OUT.json")
    user, since_iso, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    since = parse_since(since_iso)
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    imps = c.execute(
        "select story_id, occurred_at, sort_mode, age_filter from interaction_events"
        " where user_id=? and event_type='impression' and occurred_at>=? order by occurred_at",
        (user, since),
    ).fetchall()
    votes = {
        sid: (a, t)
        for sid, a, t in c.execute(
            "select story_id, action, updated_at from feedback where user_id=?", (user,)
        )
        if not REVOTE[0] <= t <= REVOTE[1]
    }
    for sid, (a, _t) in votes.items():
        if a not in ("up", "down", "neutral"):
            raise ValueError(f"Unexpected action {a!r} on story {sid}")
    if not imps:
        raise ValueError(f"No impressions for user {user} since {since_iso}")
    first: dict[int, tuple[float, str, str]] = {}
    for sid, at, sm, win in imps:
        if sid not in first:
            sv = (sm or "").strip().lower()
            view = sv if sv in ("recommended", "explore", "popular") else "other"
            first[sid] = (float(at), view, str(win) if win else "unknown")
    prevoted = 0
    pool: list[tuple[int, float, str, str, str]] = []
    for sid, (at, view, win) in first.items():
        vote = votes.get(sid)
        if vote is not None and vote[1] < at - 1:
            prevoted += 1
            continue
        pool.append((sid, at, view, win, vote[0] if vote is not None else "none"))

    ups = sorted(
        ((sid, t) for sid, (a, t) in votes.items() if a == "up"), key=lambda x: x[1]
    )
    ver = c.execute(
        "select model_version, count(*) from embeddings group by 1 order by 2 desc"
    ).fetchone()
    if ver is None:
        raise ValueError("No embeddings in DB")
    version = str(ver[0])
    ids = sorted({s for s, _, _, _, _ in pool} | {s for s, _ in ups})
    vec: dict[int, np.ndarray] = {}
    for i in range(0, len(ids), CHUNK):
        q = "select story_id, embedding from embeddings where model_version=?"
        q += f" and story_id in ({','.join('?' * len(ids[i : i + CHUNK]))})"
        for sid, blob in c.execute(q, (version, *ids[i : i + CHUNK])):
            arr = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
            n = float(np.linalg.norm(arr))
            if n > 0:
                vec[sid] = arr / n
    c.close()
    n_novec = sum(1 for sid, _, _, _, _ in pool if sid not in vec)
    print(
        f"{len(pool)} in pool ({prevoted} prevoted); model {version}:",
        f"{len(vec)} vectors, {n_novec} pool stories without one",
        flush=True,
    )

    uv = [(sid, t) for sid, t in ups if sid in vec]
    ut = [t for _, t in uv]
    U = np.stack([vec[sid] for sid, _ in uv]) if uv else np.empty((0, 0))
    def maxsim_to_earlier(arr: np.ndarray, at: float) -> float | None:
        n_early = bisect.bisect_left(ut, at)
        return float(np.max(U[:n_early] @ arr)) if n_early else None
    rows = [
        Shown(sid, at, view, win, oc, None, None) for sid, at, view, win, oc in pool
    ]
    pids = sorted({r.story_id for r in rows if r.story_id in vec})
    if len(pids) < 2:
        raise ValueError("Need at least 2 pool vectors for clusters")
    n_topics = min(K, len(pids))
    km = MiniBatchKMeans(n_clusters=n_topics, random_state=SEED, n_init=3)
    lab = dict(zip(pids, map(int, km.fit_predict(np.stack([vec[i] for i in pids])))))
    for r in rows:
        if r.story_id in vec:
            r.maxsim_up = maxsim_to_earlier(vec[r.story_id], r.at)
        r.cluster = lab.get(r.story_id)
    print(
        f"clusters: k={n_topics} fit once on {len(pids)} pool vectors (seed {SEED})",
        flush=True,
    )

    weeks = sorted({iso_week(r.at) for r in rows})
    widx = {w: i for i, w in enumerate(weeks)}
    cells: dict[tuple[str, str], list[Shown]] = defaultdict(list)
    for r in rows:
        cells[(iso_week(r.at), r.view)].append(r)
    rng = np.random.default_rng(SEED)

    weekly: dict[str, dict] = {}
    print(
        "\nWeekly x view (week of first showing; sim over stories with maxsim_up):\n"
        "week     view        |     n  nvec   mean    med novel%    div   cl/n"
        " | up/neu/dn/none up/vot",
        flush=True,
    )
    for wk in weeks:
        for vw in VIEWS:
            rs = cells.get((wk, vw), [])
            if not rs:
                continue
            sims = [r.maxsim_up for r in rs if r.maxsim_up is not None]
            cnt = Counter(r.outcome for r in rs)
            voted = cnt["up"] + cnt["down"] + cnt["neutral"]
            mean = float(np.mean(sims)) if sims else None
            med = float(np.median(sims)) if sims else None
            novel = sum(1 for x in sims if x < NOVEL_CUT) / len(sims) if sims else None
            here = [vec[r.story_id] for r in rs if r.story_id in vec]
            div = pairwise_div(here, rng)
            distinct = len({r.cluster for r in rs if r.cluster is not None})
            upv = cnt["up"] / voted if voted else None
            weekly[f"{wk}|{vw}"] = {
                "n": len(rs),
                "n_vec": len(here),
                "n_sim": len(sims),
                "mean": mean,
                "share_novel": novel,
                "diversity": div,
                "distinct_over_n": distinct / len(rs),
                "up_rate_voted": upv,
            }
            ns = "n/a" if novel is None else f"{100 * novel:5.1f}"
            print(
                f"{wk:8} {vw:11} | {len(rs):5} {len(here):5} {fmt(mean):>6} {fmt(med):>6} {ns:>6}",
                f"{fmt(div):>6} {distinct / len(rs):6.3f} | {cnt['up']:2}/{cnt['neutral']:2}/{cnt['down']:2}/{cnt['none']:3} {fmt(upv):>6}",
                flush=True,
            )

    scopes: dict[str, list[Shown]] = {"all": rows}
    scopes.update({v: [r for r in rows if r.view == v] for v in VIEWS})
    buckets: dict[str, dict[str, dict]] = {}
    print(
        "\nOutcome by maxsim_up bucket (stories with a maxsim_up only):\n"
        "scope       bucket  |     n | up/voted [95%] | up/shown [95%] | counts",
        flush=True,
    )
    for scope, rs in scopes.items():
        buckets[scope] = {}
        for b in BUCKET_LABELS:
            br = [r for r in rs if r.maxsim_up is not None and bucket(r.maxsim_up) == b]
            cnt = Counter(r.outcome for r in br)
            pv, lo, hi = wilson(cnt["up"], cnt["up"] + cnt["down"] + cnt["neutral"])
            ps, los, his = wilson(cnt["up"], len(br))
            buckets[scope][b] = {
                "n": len(br),
                "up_rate_voted": [pv, lo, hi],
                "up_rate_shown": [ps, los, his],
            }
            print(
                f"{scope:11} {b:7} | {len(br):5} | {fmt(pv):>8} [{fmt(lo)},{fmt(hi)}]",
                f"{fmt(ps):>8} [{fmt(los)},{fmt(his)}] | {cnt['up']:2}/{cnt['neutral']:2}/{cnt['down']:2}/{cnt['none']:3}",
                flush=True,
            )

    trends: dict[str, dict] = {}
    print(
        "\nTrend per view (Spearman of weekly stat vs week index):",
        flush=True,
    )
    for vw in VIEWS:
        has = [(widx[w], weekly[f"{w}|{vw}"]) for w in weeks if f"{w}|{vw}" in weekly]
        wm = [(i, d["mean"]) for i, d in has if d["mean"] is not None]
        wn = [(i, d["share_novel"]) for i, d in has if d["share_novel"] is not None]
        trends[vw] = {
            "rho_mean_maxsim": spearman([x for x, _ in wm], [y for _, y in wm]),
            "n_weeks_mean": len(wm),
            "rho_share_novel": spearman([x for x, _ in wn], [y for _, y in wn]),
            "n_weeks_novel": len(wn),
        }
        tr = trends[vw]
        print(
            f"  {vw:11} mean rho={fmt(tr['rho_mean_maxsim'])} ({len(wm)}w)",
            f"novel rho={fmt(tr['rho_share_novel'])} ({len(wn)}w)",
            flush=True,
        )

    upvote_weekly: dict[str, dict] = {}
    print(
        "\nUp-vote profile per vote week (maxsim of each new up to earlier ups):\n"
        "week     |     n  nsim   mean    med novel%",
        flush=True,
    )
    by_uvw: dict[str, list[float | None]] = defaultdict(list)
    for sid, t in ups:
        if t < since:
            continue
        arr = vec.get(sid)
        ms = maxsim_to_earlier(arr, t) if arr is not None else None
        by_uvw[iso_week(t)].append(ms)
    for wk in sorted(by_uvw):
        vals = [x for x in by_uvw[wk] if x is not None]
        mean = float(np.mean(vals)) if vals else None
        med = float(np.median(vals)) if vals else None
        novel = sum(1 for x in vals if x < NOVEL_CUT) / len(vals) if vals else None
        upvote_weekly[wk] = {
            "n": len(by_uvw[wk]),
            "n_sim": len(vals),
            "mean": mean,
            "share_novel": novel,
        }
        nov = "n/a" if novel is None else f"{100 * novel:5.1f}"
        print(
            f"{wk:8} | {len(by_uvw[wk]):5} {len(vals):5} {fmt(mean):>6} {fmt(med):>6} {nov:>6}",
            flush=True,
        )
    uw_ok = [w for w in sorted(upvote_weekly) if upvote_weekly[w]["mean"] is not None]
    rho_up = spearman(
        list(range(len(uw_ok))), [upvote_weekly[w]["mean"] for w in uw_ok]
    )
    print(
        f"up-vote mean-maxsim trend: rho={fmt(rho_up)} over {len(uw_ok)} weeks",
        flush=True,
    )

    payload = {
        "user": user,
        "since_iso": since_iso,
        "model_version": version,
        "novel_cut": NOVEL_CUT,
        "k": n_topics,
        "n_shown": len(rows),
        "n_prevoted_dropped": prevoted,
        "n_no_embedding": n_novec,
        "rows": [
            {
                "story_id": r.story_id,
                "first_shown": dt.datetime.fromtimestamp(r.at, dt.UTC).isoformat(),
                "view": r.view,
                "window": r.window,
                "outcome": r.outcome,
                "maxsim_up": r.maxsim_up,
                "cluster": r.cluster,
            }
            for r in rows
        ],
        "weekly": weekly,
        "buckets": buckets,
        "trends": trends,
        "upvote_weekly": upvote_weekly,
        "upvote_trend_rho": rho_up,
    }
    json.dump(payload, open(out, "w"))
    print(f"\nWrote {len(rows)} rows to {out} (aggregates above)", flush=True)


if __name__ == "__main__":
    main()
