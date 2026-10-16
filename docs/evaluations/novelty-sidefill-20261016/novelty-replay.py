"""Read-only novelty replay: does a novelty term keep hits while widening topics?

Day-forward replay over one profile's shown cards: each day's first-shown cards
are re-scored by the production ranker trained only on earlier votes, with
maxsim_up to earlier up-votes (stored L2-normalized embeddings). Phase 2 blends
within-day rankpct(score)/rankpct(1-maxsim_up); reports AUCs, per-day top-12/24
hits/diversity/topics, day-block bootstrap deltas vs lambda=0. Read-only DB;
per-story rows only to OUT.json/NPZ.
Run: uv run python revote/novelty-replay.py USER SINCE_ISO OUT.json [--analyze-only]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")

from database import Database  # noqa: E402
from pipeline import Config  # noqa: E402
from scripts.eval_ranker_variants import FoldData, _embedding_text_hashes  # noqa: E402
from scripts.eval_ranker_variants import _production_scores, _validated_embeddings  # noqa: E402

DB = "/home/dev/hn-rewrite/main/hn_rewrite.db"
REVOTE = (1791763112.0, 1791763181.0)
VIEWS = ("recommended", "popular", "explore")
LAMBDAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
NOVEL_CUT = 0.60
N_CLUSTERS = 40
N_BOOT, BOOT_SEED = 2000, 0
TOP_KS = (12, 24)
METRICS = ("ups", "downs", "novel", "div", "topics")


@dataclass
class FirstShown:
    story_id: int
    at: float
    view: str
    window: str


@dataclass
class ScoredRow:
    story_id: int
    day: str
    view: str
    window: str
    outcome: str
    score: float
    maxsim_up: float | None


def npz_path(out: str) -> str:
    return out[:-5] + ".npz" if out.endswith(".json") else out + ".npz"


def parse_since(s: str) -> float:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()


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


def auc_or_nan(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or float(np.isnan(x)) else f"{x:.{nd}f}"


def nanmean(xs: list[float] | np.ndarray) -> float:
    v = np.asarray(xs, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def diversity(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    iu = np.triu_indices(mat.shape[0], 1)
    return float(np.mean(1.0 - (mat @ mat.T)[iu]))


def score_phase(
    user: int, since_iso: str, out: str
) -> tuple[list[ScoredRow], np.ndarray, np.ndarray]:
    since = parse_since(since_iso)
    raw = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    imps = raw.execute(
        "select story_id, occurred_at, sort_mode, age_filter from interaction_events"
        " where user_id=? and event_type='impression' and occurred_at>=?"
        " and sort_mode in ('recommended','popular','explore') order by occurred_at",
        (user, since),
    ).fetchall()
    votes = {
        sid: (a, t)
        for sid, a, t in raw.execute(
            "select story_id, action, updated_at from feedback where user_id=?", (user,)
        )
        if not REVOTE[0] <= t <= REVOTE[1]
    }
    raw.close()
    if not imps:
        raise ValueError(f"No impressions for user {user} since {since_iso}")

    first: dict[int, FirstShown] = {}
    for sid, at, view, win in imps:
        if view not in VIEWS:
            raise ValueError(f"Unexpected sort_mode {view!r}")
        if not win:
            raise ValueError(f"Missing age_filter for story {sid}")
        first.setdefault(sid, FirstShown(sid, float(at), view, str(win)))

    def outcome(fs: FirstShown) -> str:
        v = votes.get(fs.story_id)
        if v is None:
            return "none"
        action, vt = v
        if action not in ("up", "down", "neutral"):
            raise ValueError(f"Unexpected action {action!r} on story {fs.story_id}")
        return "prevoted" if vt < fs.at - 1 else action

    config = Config.load("config.toml")
    db = Database(DB, read_only=True)
    fb_stories, fb_labels, fb_times = db.get_feedback_for_training(user_id=user)
    keep = [i for i, t in enumerate(fb_times) if not REVOTE[0] <= t <= REVOTE[1]]
    fb_stories = [fb_stories[i] for i in keep]
    fb_y = np.array([fb_labels[i] for i in keep], int)
    fb_t = np.array([fb_times[i] for i in keep], float)
    fb_hash = dict(
        zip([s.id for s in fb_stories], _embedding_text_hashes(fb_stories), strict=True)
    )
    fb_emb = _validated_embeddings(
        fb_stories,
        db.get_embeddings_batch(
            [s.id for s in fb_stories], config.embedding_model_version, fb_hash
        ),
    )

    pool = [s for s in (db.get_story(sid) for sid in sorted(first)) if s is not None]
    pool_hash = dict(
        zip([s.id for s in pool], _embedding_text_hashes(pool), strict=True)
    )
    stored = db.get_embeddings_batch(
        [s.id for s in pool], config.embedding_model_version, pool_hash
    )
    pool = [s for s in pool if s.id in stored]
    pool_emb = _validated_embeddings(pool, {s.id: stored[s.id] for s in pool})
    pos = {s.id: i for i, s in enumerate(pool)}
    if not pool:
        raise ValueError("Pool has no stories with stored embeddings")
    print(f"{len(first)} stories shown, {len(pool)} scorable, {len(fb_y)} votes", flush=True)

    rows: list[ScoredRow] = []
    show_day = {
        s.id: dt.datetime.fromtimestamp(first[s.id].at, dt.UTC).date() for s in pool
    }
    days = sorted(set(show_day.values()))
    for day in days:
        cut = dt.datetime.combine(day, dt.time(), dt.UTC).timestamp()
        tr = np.flatnonzero(fb_t < cut)
        trained = {fb_stories[i].id for i in tr}
        cand = [i for i, s in enumerate(pool) if s.id not in trained]
        fold = FoldData(
            candidates=[pool[i] for i in cand], cand_emb=pool_emb[cand],
            train_stories=[fb_stories[i] for i in tr], test_stories=[],
            test_actions=np.array([], int), train_vote_times=fb_t[tr],
            x_train_base=np.empty((0, 0)), x_cand_base=np.empty((0, 0)),
            y_train=fb_y[tr], tier2_scores=np.empty(0), train_emb=fb_emb[tr]
        )
        sc, _ = _production_scores(fold, config, source_db=db)
        by_id = dict(zip([pool[i].id for i in cand], map(float, sc), strict=True))
        todays = [s.id for s in pool if show_day[s.id] == day]
        up_idx = tr[fb_y[tr] == 2]
        base = pool_emb[[pos[sid] for sid in todays]]
        sim_up = base @ fb_emb[up_idx].T if len(up_idx) else np.empty((len(todays), 0))
        d = day.isoformat()
        n = 0
        for k, sid in enumerate(todays):
            if sid not in by_id:
                continue
            fs = first[sid]
            oc = outcome(fs)
            if oc == "prevoted":
                continue
            ms = float(sim_up[k].max()) if sim_up.shape[1] else None
            rows.append(ScoredRow(sid, d, fs.view, fs.window, oc, by_id[sid], ms))
            n += 1
        print(f"{day}: trained on {len(tr)} votes, scored {n} first shown that day", flush=True)
    db.close()
    if not rows:
        raise ValueError("No scored rows after outcome filtering")
    ids = np.array(sorted({r.story_id for r in rows}))
    mat = np.stack([pool_emb[pos[int(i)]] for i in ids]).astype(np.float64)
    payload = {"user": user, "since_iso": since_iso, "rows": [asdict(r) for r in rows]}
    Path(out).write_text(json.dumps(payload))
    np.savez(npz_path(out), story_ids=ids, embeddings=mat)
    print(f"Saved {len(rows)} rows to {out} and {len(ids)} vectors to {npz_path(out)}", flush=True)
    return rows, ids, mat


def load_artifacts(
    out: str,
) -> tuple[int, str, list[ScoredRow], np.ndarray, np.ndarray]:
    data = json.loads(Path(out).read_text())
    rows = [ScoredRow(**r) for r in data["rows"]]
    if not rows:
        raise ValueError(f"No rows in {out}")
    z = np.load(npz_path(out))
    ids = np.asarray(z["story_ids"])
    mat = np.asarray(z["embeddings"], dtype=np.float64)
    if len(ids) != mat.shape[0] or set(ids.tolist()) != {r.story_id for r in rows}:
        raise ValueError("OUT.npz story_ids do not match OUT.json rows")
    return int(data["user"]), str(data["since_iso"]), rows, ids, mat


def _is_novel(r: ScoredRow) -> bool:
    return r.outcome == "up" and (r.maxsim_up is None or r.maxsim_up < NOVEL_CUT)


def analyze(rows: list[ScoredRow], ids: np.ndarray, mat: np.ndarray) -> None:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    if not np.all(n > 0):
        raise ValueError("Zero-norm embedding in pool")
    mat = mat / n
    vmap = {int(sid): mat[k] for k, sid in enumerate(ids)}
    lab = {int(sid): k for k, sid in enumerate(ids)}
    days = sorted({r.day for r in rows})
    by_day = {d: [i for i, r in enumerate(rows) if r.day == d] for d in days}
    is_up = np.array([r.outcome == "up" for r in rows])
    is_down = np.array([r.outcome == "down" for r in rows])
    novel = np.array([_is_novel(r) for r in rows])
    nov_score = np.array(
        [np.nan if r.maxsim_up is None else 1.0 - r.maxsim_up for r in rows]
    )

    blended: dict[float, np.ndarray] = {}
    for lam in LAMBDAS:
        b = np.empty(len(rows))
        for d, idx in by_day.items():
            sc = np.array([rows[i].score for i in idx])
            b[idx] = (1 - lam) * pct_fill(sc) + lam * pct_fill(nov_score[idx])
        blended[lam] = b

    print(f"Per-day n ({len(days)} days; first-shown, scored, not prevoted):", flush=True)
    for d in days:
        vs = [rows[i].view for i in by_day[d]]
        print(f"  {d} n={len(vs)} rec={vs.count('recommended')} pop={vs.count('popular')} exp={vs.count('explore')}", flush=True)

    scopes = {"all": np.ones(len(rows), bool)}
    scopes.update({v: np.array([r.view == v for r in rows]) for v in VIEWS})
    print("\nAUC per lambda (ud=up vs down, ur=up vs rest, nd=novel-up vs down):", flush=True)
    for name, m in scopes.items():
        print(f"[{name}]", flush=True)
        for lam, b in blended.items():
            m_ud = m & (is_up | is_down)
            ud = auc_or_nan(is_up[m_ud].astype(int), b[m_ud])
            ur = auc_or_nan(is_up[m].astype(int), b[m])
            m_nd = m & (novel | is_down)
            nd = auc_or_nan(novel[m_nd].astype(int), b[m_nd])
            print(f"  lam={lam:4.2f} ud={fmt(ud)} ur={fmt(ur)} nd={fmt(nd)} (n={m.sum()})", flush=True)

    k = min(N_CLUSTERS, len(ids))
    if k < 2:
        raise ValueError("Need at least 2 pool vectors for topic clusters")
    cl = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(mat)
    rng = np.random.default_rng(BOOT_SEED)
    for K in TOP_KS:
        per: dict[float, dict[str, np.ndarray]] = {}
        for lam in LAMBDAS:
            acc: dict[str, list[float]] = {k: [] for k in METRICS}
            lam_b = blended[lam]
            for d in days:
                idx = by_day[d]
                order = np.argsort(-lam_b[idx], kind="stable")[: min(K, len(idx))]
                sids = [rows[idx[j]].story_id for j in order]
                acc["ups"].append(sum(is_up[idx[j]] for j in order))
                acc["downs"].append(sum(is_down[idx[j]] for j in order))
                acc["novel"].append(sum(novel[idx[j]] for j in order))
                vecs = [vmap[s] for s in sids]
                dv = diversity(np.stack(vecs)) if len(vecs) > 1 else float("nan")
                acc["div"].append(dv)
                acc["topics"].append(float(len({cl[lab[s]] for s in sids})))
            per[lam] = {k2: np.array(v, float) for k2, v in acc.items()}
        print(f"\nTop-{K} per-day means ({len(days)} days):", flush=True)
        print("lam    ups    downs  novel  div    topics", flush=True)
        for lam, m in per.items():
            print(f"{lam:4.2f} {nanmean(m['ups']):5.2f} {nanmean(m['downs']):5.2f} {nanmean(m['novel']):5.2f} {fmt(nanmean(m['div']))} {nanmean(m['topics']):5.1f}", flush=True)
        print(f"Top-{K} deltas vs lam=0, day-block bootstrap 95% ({N_BOOT} resamples):", flush=True)
        for lam in LAMBDAS[1:]:
            parts = []
            for key in ("ups", "downs", "novel", "div"):
                d = per[lam][key] - per[0.0][key]
                ok = d[~np.isnan(d)]
                if not ok.size:
                    parts.append(f"d_{key}=n/a")
                    continue
                rep = ok[rng.integers(0, ok.size, size=(N_BOOT, ok.size))].mean(axis=1)
                parts.append(f"d_{key}={ok.mean():+.2f} [{np.percentile(rep, 2.5):+.2f},{np.percentile(rep, 97.5):+.2f}]")
            print(f"  lam={lam:4.2f} " + " ".join(parts), flush=True)
    print("Caveat: outcomes exist only for stories that were shown (exposure-limited replay).", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only novelty replay.")
    p.add_argument("user", type=int)
    p.add_argument("since_iso", type=str)
    p.add_argument("out", type=str)
    p.add_argument("--analyze-only", action="store_true")
    ns = p.parse_args()
    if ns.analyze_only:
        user, since_iso, rows, ids, mat = load_artifacts(ns.out)
    else:
        rows, ids, mat = score_phase(ns.user, ns.since_iso, ns.out)
        user, since_iso = ns.user, ns.since_iso
    print(f"\nProfile {user}, first showings since {since_iso}: {len(rows)} rows", flush=True)
    analyze(rows, ids, mat)


if __name__ == "__main__":
    main()
