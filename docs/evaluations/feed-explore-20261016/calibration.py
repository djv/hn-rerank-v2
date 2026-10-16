"""Read-only, day-forward calibration and exposure-limited Unsure audit."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from database import Database, Story  # noqa: E402
from pipeline import Config  # noqa: E402
from pipeline.codex_forum import effective_codex_story  # noqa: E402
from pipeline.ranking import RankTrace, _score_and_rank  # noqa: E402
from scripts.eval_ranker_variants import (  # noqa: E402
    _FrozenEmbedder, _embedding_text_hashes, _validated_embeddings,
)

DB = "/home/dev/hn-rewrite/main/hn_rewrite.db"
REVOTE = (1791763112.0, 1791763181.0)
CLASSES = ("down", "neutral", "up")
MIN_CAL = 30
EPS = 1e-12


@dataclass
class Row:
    story_id: int
    day: str
    outcome: str
    voted_today: bool
    first_shown: bool
    probs: dict[str, list[float]]
    entropy: dict[str, float] = field(default_factory=dict)


def day_of(at: float) -> str:
    return dt.datetime.fromtimestamp(at, dt.UTC).date().isoformat()


def vectors(db: Database, stories: list[Story], config: Config) -> np.ndarray:
    hashes = dict(zip([s.id for s in stories], _embedding_text_hashes(stories), strict=True))
    cached = db.get_embeddings_batch([s.id for s in stories], config.embedding_model_version, hashes)
    # The shared validator includes IDs in errors; keep diagnostics aggregate-only.
    try:
        mat = _validated_embeddings(stories, cached)
    except ValueError:
        raise ValueError("Missing, stale or invalid stored embeddings; replay aborted") from None
    if mat.shape[1] != 384:
        raise ValueError("Expected stored 384-d embeddings")
    return mat


def score_day(db: Database, config: Config, pool: list[Story], mat: np.ndarray,
              train: tuple[list[Story], list[int], list[float]], cut: float) -> np.ndarray | None:
    trace = RankTrace()
    # Direct production path: unlike _production_scores, no writable fold DB.
    # None user_id disables model-cache signatures derived from live feedback.
    with patch("pipeline.ranking.time.time", return_value=cut):
        ranked = _score_and_rank(pool, mat, db, config,
                                 _FrozenEmbedder(config.embedding_model_version, 384),
                                 trace=trace, training_feedback=train)
    if any(v == "error" for v in trace.labels.values()):
        raise RuntimeError("Production scoring failed; refusing fallback probabilities")
    if config.model.side_embedding_enabled and trace.labels.get("side_embeddings") == "off":
        raise RuntimeError("Insufficient side-vector coverage; refusing stored-only fallback")
    by_id = {r.story.id: r for r in ranked}
    if set(by_id) != {s.id for s in pool}:
        raise RuntimeError("Production candidate coverage mismatch")
    values = [[by_id[s.id].prob_down, by_id[s.id].prob_neutral, by_id[s.id].prob_up] for s in pool]
    if all(all(v is None for v in row) for row in values):
        return None  # Sparse/cold days have no production Unsure probabilities.
    if any(v is None for row in values for v in row):
        raise RuntimeError("Partial probability coverage")
    probs = np.asarray(values, dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1):
        raise ValueError("Invalid production probability distribution")
    return probs


def temperature(p: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(p, EPS, 1)) / t
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean())


def logit_up(p: np.ndarray) -> np.ndarray:
    u = np.clip(p[:, 2], EPS, 1 - EPS)
    return np.log(u / (1 - u)).reshape(-1, 1)


def recalibrate(p: np.ndarray, history: list[Row]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    variants = {"raw": p}
    params: dict[str, object] = {"history_n": len(history)}
    if len(history) < MIN_CAL:
        return variants, params
    hp = np.array([r.probs["raw"] for r in history])
    hy = np.array([CLASSES.index(r.outcome) for r in history])
    if len(np.unique(hy)) < 2:
        return variants, params
    # Bounded deterministic log-grid search; include identity T exactly.
    grid = np.unique(np.r_[np.geomspace(0.05, 20, 241), 1.0])
    t = float(min(grid, key=lambda v: (loss(temperature(hp, float(v)), hy), abs(np.log(v)))))
    variants["temperature"] = temperature(p, t)
    params["temperature"] = t
    up = (hy == 2).astype(int)
    if len(np.unique(up)) == 2:
        model = LogisticRegression(C=1e6, random_state=0, max_iter=1000).fit(logit_up(hp), up)
        if (model.n_iter_ >= 1000).any():
            raise RuntimeError("Platt fit did not converge")
        u = model.predict_proba(logit_up(p))[:, 1]
        mass = p[:, :2].sum(1, keepdims=True)
        if (mass <= 0).any():
            raise ValueError("Platt redistribution needs positive non-up probability mass")
        other = p[:, :2] / mass
        variants["platt"] = np.column_stack([other * (1 - u[:, None]), u])
        params["platt"] = {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
    return variants, params


def reliability(p: np.ndarray, y: np.ndarray) -> dict[str, object]:
    bins = []
    for ix in np.array_split(np.argsort(p, kind="stable"), min(10, len(p))):
        bins.append({"n": len(ix), "predicted": float(p[ix].mean()), "observed": float(y[ix].mean())})
    ece = sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in bins) / len(p)
    return {"ece": ece, "bins": bins}


def metrics(p: np.ndarray, y: np.ndarray) -> dict[str, object]:
    rel = {c: reliability(p[:, j], (y == j).astype(int)) for j, c in enumerate(CLASSES)}
    ud = y != 1
    binary: dict[str, object] | None = None
    if ud.any():
        q = p[ud, 2] / np.maximum(p[ud, 0] + p[ud, 2], EPS)
        b = (y[ud] == 2).astype(int)
        binary = {**reliability(q, b), "n": len(b), "brier": float(np.mean((q - b) ** 2)),
                  "log_loss": loss(np.column_stack([1 - q, q]), b)}
    return {"n": len(y), "classwise": rel,
            "ece": float(np.mean([r["ece"] for r in rel.values()])),
            "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
            "log_loss": loss(p, y), "up_down": binary}


def entropy_mix(rows: list[Row], variant: str) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda r: (r.entropy[variant], r.day, r.story_id))
    result = []
    for q, ix in enumerate(np.array_split(np.arange(len(ordered)), min(5, len(ordered))), 1):
        rs = [ordered[int(i)] for i in ix]
        result.append({"quintile": q, "n": len(rs),
                       "mean_entropy": float(np.mean([r.entropy[variant] for r in rs])),
                       **{c: sum(r.outcome == c for r in rs) / len(rs) for c in CLASSES}})
    return result


def top_check(rows: list[Row], variant: str) -> dict[str, object]:
    raw = sorted(rows, key=lambda r: (-r.entropy["raw"], r.story_id))[:10]
    cal = sorted(rows, key=lambda r: (-r.entropy[variant], r.story_id))[:10]
    a, b = {r.story_id for r in raw}, {r.story_id for r in cal}
    result: dict[str, object] = {"pool_n": len(rows), "jaccard": len(a & b) / len(a | b)}
    for name, picked in (("raw", raw), (variant, cal)):
        voted = [r for r in picked if r.outcome != "none"]
        ups = sum(r.outcome == "up" for r in voted)
        result[name] = {"n": len(picked), "voted": len(voted), "ups": ups,
                        "up_rate": ups / len(voted) if voted else None}
    return result


def summarize(rows: list[Row]) -> dict[str, object]:
    report: dict[str, object] = {}
    for variant in ("raw", "temperature", "platt"):
        voted = [r for r in rows if r.voted_today and variant in r.probs]
        if not voted:
            continue
        y = np.array([CLASSES.index(r.outcome) for r in voted])
        result = metrics(np.array([r.probs[variant] for r in voted]), y)
        baseline = metrics(np.array([r.probs["raw"] for r in voted]), y)
        result["paired_raw"] = baseline
        result["delta"] = {k: result[k] - baseline[k] for k in ("ece", "log_loss", "brier")}
        result["class_ece_delta"] = {c: result["classwise"][c]["ece"] - baseline["classwise"][c]["ece"] for c in CLASSES}
        if result["up_down"] is not None:
            result["up_down_delta"] = {k: result["up_down"][k] - baseline["up_down"][k]
                                       for k in ("ece", "log_loss", "brier")}
        result["entropy_quintiles"] = entropy_mix(voted, variant)
        result["paired_raw_entropy_quintiles"] = entropy_mix(voted, "raw")
        print(f"{variant:12} n={len(voted):5} ECE={result['ece']:.4f} "
              f"LL={result['log_loss']:.4f} Brier={result['brier']:.4f} delta={result['delta']}", flush=True)
        print("  class ECE: " + " ".join(f"{c}={result['classwise'][c]['ece']:.4f}" for c in CLASSES), flush=True)
        binary = result["up_down"]
        if binary:
            print(f"  up/down n={binary['n']} ECE={binary['ece']:.4f} "
                  f"LL={binary['log_loss']:.4f} Brier={binary['brier']:.4f} "
                  f"delta={result['up_down_delta']}", flush=True)
        if variant == "raw":
            print("  equal-count bin: down pred/obs/n | neutral pred/obs/n | up pred/obs/n | up/down pred/obs/n", flush=True)
            rels = [result["classwise"][c] for c in CLASSES] + [binary]
            for i in range(10):
                cells = []
                for rel in rels:
                    bins = rel["bins"] if rel else []
                    cells.append(f"{bins[i]['predicted']:.3f}/{bins[i]['observed']:.3f}/{bins[i]['n']}" if i < len(bins) else "n/a")
                print(f"  {i + 1:2} " + " | ".join(cells), flush=True)
        print("  entropy quintile: n / mean bits / down neutral up", flush=True)
        for q in result["entropy_quintiles"]:
            print(f"  Q{q['quintile']} {q['n']:4} {q['mean_entropy']:.3f} "
                  + " ".join(f"{q[c]:.3f}" for c in CLASSES), flush=True)
        report[variant] = result
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user", type=int)
    parser.add_argument("since_iso")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    since_dt = dt.datetime.fromisoformat(args.since_iso)
    since = since_dt.replace(tzinfo=dt.UTC).timestamp() if since_dt.tzinfo is None else since_dt.timestamp()
    if not np.isfinite(since):
        raise ValueError("SINCE must be a finite timestamp")
    if args.out.suffix != ".json" or args.out.exists():
        raise ValueError("OUT must be a new .json path (no overwrites)")
    config = Config.load("config.toml")
    rows: list[Row] = []
    days_report: list[dict[str, object]] = []
    db = Database(DB, read_only=True)
    try:
        with db.conn() as conn:
            # Earliest recorded impression across every view, not just since SINCE.
            first = dict(conn.execute("SELECT story_id, MIN(occurred_at) FROM interaction_events "
                                      "WHERE user_id=? AND event_type='impression' GROUP BY story_id", (args.user,)))
        stories, labels, times = db.get_feedback_for_training(user_id=args.user)
        keep = sorted((i for i, t in enumerate(times) if not REVOTE[0] <= t <= REVOTE[1]),
                      key=lambda i: (times[i], stories[i].id))
        fb = [effective_codex_story(stories[i]) for i in keep]
        y, ts = np.array([labels[i] for i in keep]), np.array([times[i] for i in keep])
        if not np.isfinite(ts).all() or not set(y.tolist()) <= {0, 1, 2}:
            raise ValueError("Invalid training labels/timestamps")
        votes = {s.id: (CLASSES[int(a)], float(t)) for s, a, t in zip(fb, y, ts, strict=True)}
        vectors(db, fb, config)  # Fail before fitting if any training vector is missing.
        eligible = {sid: v for sid, v in votes.items() if v[1] >= since and v[1] >= first.get(sid, v[1]) - 1}
        days = sorted({day_of(v[1]) for v in eligible.values()})
        coverage = {"feedback_n": len(stories), "revote_excluded": len(stories) - len(keep),
                    "eligible_votes": len(eligible), "vote_days": len(days),
                    "known_first_impressions": len(first)}
        print(f"Coverage: {coverage}", flush=True)
        for day in days:
            cut = dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC).timestamp()
            voted_ids = {sid for sid, v in eligible.items() if day_of(v[1]) == day}
            shown = {sid for sid, at in first.items() if at >= since and day_of(at) == day
                     and (sid not in votes or votes[sid][1] >= at - 1)}
            tr = np.flatnonzero(ts < cut)
            ids = sorted((voted_ids | shown) - {fb[int(i)].id for i in tr})
            pool = [db.get_story(sid) for sid in ids]
            if any(s is None for s in pool):
                raise ValueError("Missing candidate stories")
            pool = [effective_codex_story(s) for s in pool if s is not None]
            # Like short-window-scores.py: score only candidates whose stored
            # vector matches the current text; count the rest.
            hashes = dict(zip([s.id for s in pool], _embedding_text_hashes(pool), strict=True))
            stored = db.get_embeddings_batch([s.id for s in pool], config.embedding_model_version, hashes)
            skipped = len(pool) - sum(s.id in stored for s in pool)
            pool = [s for s in pool if s.id in stored]
            ids = [s.id for s in pool]
            if skipped:
                print(f"{day}: skipped {skipped} candidates without a current stored vector", flush=True)
            raw = score_day(db, config, pool, vectors(db, pool, config),
                            ([fb[int(i)] for i in tr], y[tr].tolist(), ts[tr].tolist()), cut)
            if raw is None:
                days_report.append({"day": day, "skipped_no_probs": len(pool)})
                print(f"{day}: no production probabilities; skipped {len(pool)} candidates", flush=True)
                continue
            # Only prior vote-day predictions enter calibration; never eventual exposure labels.
            history = [r for r in rows if r.voted_today]
            variants, params = recalibrate(raw, history)
            today = []
            for i, sid in enumerate(ids):
                r = Row(sid, day, votes.get(sid, ("none", 0))[0], sid in voted_ids, sid in shown,
                        {k: p[i].tolist() for k, p in variants.items()})
                r.entropy = {k: float(-np.sum(np.where(p > 1e-9, p * np.log2(np.maximum(p, EPS)), 0)))
                             for k, p in ((k, v[i]) for k, v in variants.items())}
                today.append(r)
            top = {k: top_check([r for r in today if r.first_shown], k) for k in variants
                   if any(r.first_shown for r in today)}
            days_report.append({"day": day, "train_n": len(tr), "calibration": params, "top10": top})
            rows.extend(today)
            print(f"{day}: train={len(tr)} voted={len(voted_ids)} first_shown={len(shown)} "
                  f"cal_history={len(history)}", flush=True)
        if not rows:
            raise ValueError("No scorable days with votes")
        summary = summarize(rows)
        for variant in ("raw", "temperature", "platt"):
            checks = [d["top10"][variant] for d in days_report if variant in d.get("top10", {})]
            for name in dict.fromkeys(("raw", variant)):
                voted = sum(c[name]["voted"] for c in checks)
                ups = sum(c[name]["ups"] for c in checks)
                mean_j = float(np.mean([c["jaccard"] for c in checks])) if checks else None
                print(f"top10 {variant}/{name}: days={len(checks)} mean_J={mean_j} "
                      f"picks={sum(c[name]['n'] for c in checks)} voted={voted} up_rate={ups / voted if voted else None}", flush=True)
        payload = {"user": args.user, "since_iso": args.since_iso, "classes": CLASSES,
                   "min_calibration": MIN_CAL, "coverage": coverage,
                   "summary": summary, "days": days_report,
                   "rows": [asdict(r) for r in rows]}
        with os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as out:
            json.dump(payload, out, allow_nan=False)
    finally:
        db.close()


if __name__ == "__main__":
    main()
