"""Read-only label-noise probe: how noisy are one profile's vote labels, and how
close is the production ranker to a crude kNN noise ceiling?

Votes are the user's feedback (down=0, neutral=1, up=2), REVOTE window
excluded, stored L2-normalized embeddings only (missing vectors are counted
as skips, never recomputed). Cosine pairs use chunked matrix products; raw
embeddings are only re-normalized to unit norm, never standard-scaled.
No DB writes; per-story IDs go only to OUT.json, stdout gets aggregates.
Run from the live checkout: uv run python revote/label-noise.py USER OUT.json
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from database import Database  # noqa: E402
from dedup import normalize_url  # noqa: E402
from pipeline import Config  # noqa: E402
from scripts.eval_ranker_variants import (  # noqa: E402
    FoldData,
    _embedding_text_hashes,
    _production_scores,
    _validated_embeddings,
)

USER = int(sys.argv[1])
OUT = sys.argv[2]
DB = "/home/dev/hn-rewrite/main/hn_rewrite.db"
REVOTE = (1791763112.0, 1791763181.0)
SIM_CUTS = (0.90, 0.95)
KS = (1, 5, 15)
K_HEADLINE = 5
CHUNK = 500
DAY = 86400.0
WINDOW_DAYS = 30
PERIOD_DAYS = 30
NAMES = {0: "down", 1: "neutral", 2: "up"}


@dataclass
class Vote:
    story_id: int
    label: int
    at: float
    url: str | None


@dataclass
class ConflictPair:
    story_a: int
    story_b: int
    sim: float
    label_a: str
    label_b: str
    gap_days: float
    same_url: bool


def auc_or_nan(y: np.ndarray, s: np.ndarray) -> float:
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return float("nan")
    return float(roc_auc_score(y, s))


def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or float(np.isnan(x)) else f"{x:.{nd}f}"


def knn_upshare(
    q: np.ndarray, k_mat: np.ndarray, key_up: np.ndarray, k: int, exclude_diag: bool
) -> np.ndarray:
    """Mean up-share of the k nearest key rows per query row (stable ties)."""
    out = np.empty(q.shape[0])
    for a0 in range(0, q.shape[0], CHUNK):
        s = q[a0 : a0 + CHUNK] @ k_mat.T
        rows = s.shape[0]
        if exclude_diag:
            s[np.arange(rows), np.arange(a0, a0 + rows)] = -np.inf
        kk = min(k, s.shape[1] - (1 if exclude_diag else 0))
        if kk < 1:
            out[a0 : a0 + rows] = np.nan
            continue
        top = np.argsort(-s, axis=1, kind="stable")[:, :kk]
        out[a0 : a0 + rows] = key_up[top].mean(axis=1)
    return out


config = Config.load("config.toml")
db = Database(DB, read_only=True)
fb_stories, fb_labels, fb_times = db.get_feedback_for_training(user_id=USER)
keep = [i for i, t in enumerate(fb_times) if not REVOTE[0] <= float(t) <= REVOTE[1]]
if any(int(fb_labels[i]) not in (0, 1, 2) for i in keep):
    raise ValueError("Unexpected label value in feedback")
vstories = [fb_stories[i] for i in keep]
vhash = dict(zip([s.id for s in vstories], _embedding_text_hashes(vstories), strict=True))
stored = db.get_embeddings_batch(
    [s.id for s in vstories], config.embedding_model_version, vhash
)
has_vec = np.array([s.id in stored for s in vstories])
skipped = int((~has_vec).sum())
stories = [s for s, ok in zip(vstories, has_vec) if ok]
votes = [
    Vote(s.id, int(lb), float(t), s.url)
    for s, lb, t, ok in zip(vstories, [fb_labels[i] for i in keep],
                            [fb_times[i] for i in keep], has_vec)
    if ok
]
if not votes:
    raise ValueError(f"No voted stories with stored embeddings for user {USER}")
E = np.asarray(
    _validated_embeddings(stories, {s.id: stored[s.id] for s in stories}),
    dtype=np.float64,
)
norms = np.linalg.norm(E, axis=1)
if not np.all(norms > 0):
    raise ValueError("Zero-norm stored embedding")
E = E / norms[:, None]
y = np.array([v.label for v in votes], int)
vt = np.array([v.at for v in votes], float)
sids = np.array([v.story_id for v in votes], int)
n = len(votes)
lc = {name: int((y == code).sum()) for code, name in NAMES.items()}
print(f"Profile {USER}: {len(keep)} votes, {n} with stored embeddings "
      f"({skipped} skipped), up={lc['up']} neutral={lc['neutral']} down={lc['down']}", flush=True)

# --- Near-duplicate pairs (all labels; cosine >= 0.90 / 0.95). ---
thr: dict[float, dict[str, int]] = {
    c: {"pairs": 0, "up_down": 0, "up_neutral": 0, "down_neutral": 0, "same": 0}
    for c in SIM_CUTS
}
bucket_n = [0, 0, 0, 0]
bucket_c = [0, 0, 0, 0]
conflicts: list[ConflictPair] = []
normed = [normalize_url(v.url) for v in votes]
for a0 in range(0, n, CHUNK):
    S = E[a0 : a0 + CHUNK] @ E.T
    for r in range(S.shape[0]):
        i = a0 + r
        for j in np.flatnonzero(S[r, i + 1 :] >= SIM_CUTS[0]) + i + 1:
            j = int(j)
            sim = float(S[r, j])
            li, lj = int(y[i]), int(y[j])
            gap = abs(float(vt[i]) - float(vt[j])) / DAY
            b = 0 if gap < 1.0 else (1 if gap < 7.0 else (2 if gap < 30.0 else 3))
            bucket_n[b] += 1
            if li == lj:
                for c in SIM_CUTS:
                    if sim >= c:
                        thr[c]["pairs"] += 1
                        thr[c]["same"] += 1
            else:
                key = ("up_down" if {li, lj} == {0, 2}
                       else ("up_neutral" if {li, lj} == {1, 2} else "down_neutral"))
                for c in SIM_CUTS:
                    if sim >= c:
                        thr[c]["pairs"] += 1
                        thr[c][key] += 1
                bucket_c[b] += 1
                same = normed[i] is not None and normed[i] == normed[j]
                conflicts.append(ConflictPair(int(sids[i]), int(sids[j]), round(sim, 4),
                                              NAMES[li], NAMES[lj], round(gap, 2), bool(same)))
    print(f"pairs: scanned rows {min(a0 + CHUNK, n)}/{n}", flush=True)
conflicts.sort(key=lambda p: (-p.sim, p.story_a, p.story_b))

print("\nNear-duplicate pairs (cosine on stored embeddings):")
print(f"{'cut':>6} {'pairs':>6} {'conflict':>8} {'up-down':>7} {'up-neut':>7} {'down-neut':>9} {'same':>6}")
thr_json: dict[str, dict] = {}
for c in SIM_CUTS:
    d = thr[c]
    conf = d["up_down"] + d["up_neutral"] + d["down_neutral"]
    thr_json[f">={c:.2f}"] = {**d, "conflict_share": (conf / d["pairs"] if d["pairs"] else float("nan"))}
    print(f"{c:>6.2f} {d['pairs']:6} {conf / d['pairs'] if d['pairs'] else float('nan'):8.3f} "
          f"{d['up_down']:7} {d['up_neutral']:7} {d['down_neutral']:9} {d['same']:6}")

groups: dict[str, list[int]] = {}
for i, u in enumerate(normed):
    if u is not None:
        groups.setdefault(str(u), []).append(i)
url_pairs = url_conf = 0
for idxs in groups.values():
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            url_pairs += 1
            url_conf += int(y[idxs[a]] != y[idxs[b]])
print(f"\nSame normalized URL: {url_pairs} pairs over {sum(1 for v in groups.values() if len(v) > 1)} groups, "
      f"conflict share {url_conf / url_pairs if url_pairs else float('nan'):.3f}")

print("\nConflict share by vote time gap (pairs >= 0.90):")
print(f"{'gap':>6} {'pairs':>6} {'conflict':>8}")
gap_json: dict[str, dict] = {}
for name, bn, bc in zip(("<1d", "1-7d", "7-30d", ">30d"), bucket_n, bucket_c):
    gap_json[name] = {"pairs": bn, "conflict_share": (bc / bn if bn else float("nan"))}
    print(f"{name:>6} {bn:6} {bc / bn if bn else float('nan'):8.3f}")

# --- Leave-one-out kNN agreement over up/down votes (crude noise ceiling). ---
ud = np.flatnonzero((y == 0) | (y == 2))
if len(ud) < 2:
    raise ValueError("Need at least 2 up/down votes for LOO kNN")
Eu, yu = E[ud], y[ud]
is_up = (yu == 2).astype(int)
gs = {k: knn_upshare(Eu, Eu, is_up, k, True) for k in KS}
print("\nLOO kNN agreement (neighbour = other up/down votes, score = up-share):")
print(f"{'k':>3} {'AUC':>6} {'mean up-share | true up':>24} {'true down':>9}")
knn_json: dict[str, dict] = {}
for k in KS:
    auc = auc_or_nan(is_up, gs[k])
    mu = float(gs[k][is_up == 1].mean()) if (is_up == 1).any() else float("nan")
    md = float(gs[k][is_up == 0].mean()) if (is_up == 0).any() else float("nan")
    knn_json[str(k)] = {"auc": auc, "mean_upshare_true_up": mu, "mean_upshare_true_down": md}
    print(f"{k:3} {fmt(auc):>6} {mu:24.3f} {md:9.3f}")
db.close()

# --- Production day-forward AUC on the last 30 days of up/down votes. ---
db = Database(DB, read_only=True)
t_ud = vt[ud]
cutoff = float(t_ud.max()) - WINDOW_DAYS * DAY
recent = ud[vt[ud] >= cutoff]
days = sorted({dt.datetime.fromtimestamp(float(vt[i]), dt.UTC).date() for i in recent})
prod_y: list[int] = []
prod_s: list[float] = []
for day in days:
    cut = dt.datetime.combine(day, dt.time(), dt.UTC).timestamp()
    tr = np.flatnonzero(vt < cut)
    cand = [int(i) for i in recent
            if dt.datetime.fromtimestamp(float(vt[int(i)]), dt.UTC).date() == day]
    if not cand:
        continue
    fold = FoldData(
        candidates=[stories[i] for i in cand], cand_emb=E[cand],
        train_stories=[stories[i] for i in tr], test_stories=[],
        test_actions=np.array([], int), train_vote_times=vt[tr],
        x_train_base=np.empty((0, 0)), x_cand_base=np.empty((0, 0)),
        y_train=y[tr], tier2_scores=np.empty(0), train_emb=E[tr],
    )
    sc, _ = _production_scores(fold, config, source_db=db)
    for i, s in zip(cand, map(float, sc)):
        prod_y.append(1 if y[i] == 2 else 0)
        prod_s.append(s)
    print(f"{day}: trained on {len(tr)} votes, scored {len(cand)} voted stories", flush=True)
db.close()
prod_auc = auc_or_nan(np.array(prod_y), np.array(prod_s))
mask30 = vt[ud] >= cutoff
print(f"\nProduction day-forward, last {WINDOW_DAYS}d: {len(days)} days, "
      f"scored {len(prod_y)} (up={sum(prod_y)} down={len(prod_y) - sum(prod_y)}), "
      f"up-vs-down AUC {fmt(prod_auc)}")
print("Noise-ceiling read (same 30d votes, neighbours still all votes):")
for k in KS:
    print(f"  kNN k={k} AUC {fmt(auc_or_nan(is_up[mask30], gs[k][mask30]))} vs production {fmt(prod_auc)}")

# --- Temporal drift: consecutive 30-day periods, within vs cross neighbours. ---
period = ((t_ud - float(t_ud.min())) // (PERIOD_DAYS * DAY)).astype(int)
print(f"\nDrift over consecutive {PERIOD_DAYS}d periods (k={K_HEADLINE} up-share AUC):")
print(f"{'period from':>11} {'n':>4} {'up':>3} {'down':>4} {'within':>6} {'cross':>6}")
drift: list[dict] = []
for p in sorted(set(period)):
    idx = np.flatnonzero(period == p)
    out = np.flatnonzero(period != p)
    lab = is_up[idx]
    w = knn_upshare(Eu[idx], Eu[idx], is_up[idx], K_HEADLINE, True)
    c = knn_upshare(Eu[idx], Eu[out], is_up[out], K_HEADLINE, False) if len(out) else np.full(len(idx), np.nan)
    aw, ac = auc_or_nan(lab, w), auc_or_nan(lab, c)
    start = (dt.datetime.fromtimestamp(float(t_ud.min()), dt.UTC).date()
             + dt.timedelta(days=int(p) * PERIOD_DAYS)).isoformat()
    drift.append({"period_from": start, "n": len(idx), "n_up": int(lab.sum()),
                  "n_down": int((lab == 0).sum()), "auc_within": aw, "auc_cross": ac})
    print(f"{start:>11} {len(idx):4} {int(lab.sum()):3} {int((lab == 0).sum()):4} "
          f"{fmt(aw):>6} {fmt(ac):>6}")

json.dump({
    "user": USER, "n_votes": len(keep), "n_scored": n, "n_skipped_no_embedding": skipped,
    "label_counts": lc, "thresholds": thr_json,
    "same_url": {"pairs": url_pairs, "conflict_share": (url_conf / url_pairs if url_pairs else float("nan"))},
    "gap_buckets_ge090": gap_json,
    "knn_loo": {"n": len(ud), "ks": knn_json,
                "auc_last30d": {str(k): auc_or_nan(is_up[mask30], gs[k][mask30]) for k in KS}},
    "production_dayforward_last30d": {"days": len(days), "n_scored": len(prod_y),
        "n_up": sum(prod_y), "n_down": len(prod_y) - sum(prod_y), "auc_up_down": prod_auc},
    "drift_30d_k5": drift,
    "conflicting_pairs": [asdict(p) for p in conflicts],
}, open(OUT, "w"))
print(f"\nWrote aggregates + {len(conflicts)} conflicting pair IDs to {OUT}", flush=True)
