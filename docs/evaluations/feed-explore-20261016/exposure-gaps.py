"""Read-only exposure-gap analysis: which topics does the feed under-show
relative to how often the user upvotes them? No model training.

Universe = summarizable stories with stored embeddings whose story time is
>= SINCE, plus every story the user voted on or was shown since SINCE.
MiniBatchKMeans(k, seed 0) runs on the stored L2-normalized embeddings with
no scaling. Per cluster: pool vs shown shares, exposure ratios, vote
outcomes, Wilson 95% intervals, and mean max-similarity to pre-SINCE ups.
Stdout gets aggregate tables only; per-story titles go only to OUT.json.
Run from the live checkout: uv run python revote/exposure-gaps.py USER SINCE_ISO OUT.json [--k 40]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, ".")

from database import Database  # noqa: E402
from pipeline import Config, is_summarizable  # noqa: E402
from scripts.eval_ranker_variants import _embedding_text_hashes  # noqa: E402
from scripts.eval_ranker_variants import _validated_embeddings  # noqa: E402

DB = "/home/dev/hn-rewrite/main/hn_rewrite.db"
REVOTE = (1791763112.0, 1791763181.0)
CHUNK = 500
Z95 = 1.96
UNDER_EXP = 0.7
OVER_EXP = 1.3
NEAREST = 8
ACTIONS = ("up", "down", "neutral")


@dataclass
class ClusterStat:
    idx: int
    n_pool: int
    n_shown: int
    n_shown_rec: int
    n_up: int
    n_down: int
    n_neu: int
    n_none: int
    n_prevoted: int
    fe_up: int
    fe_down: int
    fe_neu: int
    pool_share: float
    shown_share: float
    shown_rec_share: float
    exp_ratio: float | None
    exp_rec_ratio: float | None
    up_rate: float | None
    w_lo: float | None
    w_hi: float | None
    mean_maxsim: float | None
    flag: str
    nearest_titles: list[str]


def parse_since(s: str) -> float:
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.UTC)
    return d.timestamp()


def wilson(up: int, n: int) -> tuple[float | None, float | None, float | None]:
    if n == 0:
        return None, None, None
    p = up / n
    z = Z95
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    m = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return p, max(0.0, (c - m) / d), min(1.0, (c + m) / d)


def fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only exposure-gap analysis.")
    p.add_argument("user", type=int)
    p.add_argument("since_iso", type=str)
    p.add_argument("out", type=str)
    p.add_argument("--k", type=int, default=40)
    ns = p.parse_args()
    if ns.k < 2:
        raise ValueError(f"--k must be >= 2, got {ns.k}")
    since = parse_since(ns.since_iso)

    config = Config.load("config.toml")
    db = Database(DB, read_only=True)
    imps = db.execute(
        "select story_id, occurred_at, sort_mode from interaction_events "
        "where user_id=? and event_type='impression' and occurred_at>=? "
        "order by occurred_at",
        (ns.user, since),
    )
    votes: dict[int, tuple[str, float]] = {}
    for sid, action, at in db.execute(
        "select story_id, action, updated_at from feedback where user_id=?",
        (ns.user,),
    ):
        if REVOTE[0] <= float(at) <= REVOTE[1]:
            continue
        if action not in ACTIONS:
            raise ValueError(f"Unexpected action {action!r} on story {sid}")
        votes[int(sid)] = (str(action), float(at))
    print(f"{len(imps)} impressions, {len(votes)} votes (REVOTE excluded)", flush=True)

    first: dict[int, float] = {}
    rec_shown: set[int] = set()
    for sid, at, mode in imps:
        sid = int(sid)
        if sid not in first or float(at) < first[sid]:
            first[sid] = float(at)
        if mode == "recommended":
            rec_shown.add(sid)

    def outcome(sid: int) -> str:
        v = votes.get(sid)
        if v is None:
            return "none"
        action, vt = v
        return "prevoted" if vt < first[sid] - 1 else action

    base_rows = db.execute(
        "select id, title, url, score, time, text_content, source, comment_count, "
        "discussion_url, comment_count_at_fetch, self_text, top_comments, "
        "article_body from stories where time >= ?",
        (int(since),),
    )
    base = [Database._row_to_story(r) for r in base_rows]
    base_kept = [s for s in base if is_summarizable(s)]
    print(
        f"{len(base)} stories with time >= SINCE, {len(base_kept)} summarizable",
        flush=True,
    )
    have = {s.id for s in base_kept}
    # Review fix: only votes cast since SINCE join the universe.
    recent_votes = {sid for sid, (_a, t) in votes.items() if t >= since}
    extra_ids = sorted((recent_votes | set(first)) - have)
    extra = db.get_stories(extra_ids)
    print(f"+{len(extra)}/{len(extra_ids)} voted/shown extras fetched", flush=True)
    stories = base_kept + extra
    if not stories:
        raise ValueError("Empty universe; nothing to cluster")

    hashes = dict(
        zip(
            [s.id for s in stories],
            [str(h) for h in _embedding_text_hashes(stories)],
            strict=True,
        )
    )
    stored: dict[int, np.ndarray] = {}
    ids = [s.id for s in stories]
    for i in range(0, len(ids), CHUNK):
        stored.update(
            db.get_embeddings_batch(
                ids[i : i + CHUNK], config.embedding_model_version, hashes
            )
        )
    stories = [s for s in stories if s.id in stored]
    print(f"{len(stories)}/{len(ids)} universe stories with stored embeddings", flush=True)
    if not stories:
        raise ValueError("No universe story has a stored embedding")
    mat = _validated_embeddings(stories, {s.id: stored[s.id] for s in stories})
    mat64 = mat.astype(np.float64)
    pos = {s.id: i for i, s in enumerate(stories)}
    titles = {s.id: s.title for s in stories}
    sources = {s.id: s.source for s in stories}

    k = min(ns.k, len(stories))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3)
    labels = km.fit_predict(mat64)
    print(f"Clustered {len(stories)} vectors into k={k} (seed 0)", flush=True)

    prior_ids = [sid for sid, (a, t) in votes.items() if a == "up" and t < since]
    # Review fix: pre-SINCE ups are no longer in the universe; fetch their vectors.
    missing_prior = [sid for sid in prior_ids if sid not in stored]
    if missing_prior:
        pst = db.get_stories(missing_prior)
        ph = dict(zip([s.id for s in pst], [str(h) for h in _embedding_text_hashes(pst)], strict=True))
        for i in range(0, len(pst), CHUNK):
            stored.update(db.get_embeddings_batch(
                [s.id for s in pst[i : i + CHUNK]], config.embedding_model_version, ph))
    prior_vecs = [stored[sid] for sid in prior_ids if sid in stored]
    print(
        f"{len(prior_vecs)}/{len(prior_ids)} pre-SINCE ups with vectors",
        flush=True,
    )
    maxsim = np.full(len(stories), np.nan)
    if prior_vecs:
        pup = np.stack([np.asarray(v, dtype=np.float64) for v in prior_vecs])
        prior_with_vec = [sid for sid in prior_ids if sid in stored]
        prior_col = {sid: j for j, sid in enumerate(prior_with_vec)}
        for i in range(0, len(stories), CHUNK):
            sub = mat64[i : i + CHUNK] @ pup.T
            # Review fix: a reference up never counts as its own neighbour.
            for r in range(sub.shape[0]):
                j = prior_col.get(stories[i + r].id)
                if j is not None:
                    sub[r, j] = -np.inf
            maxsim[i : i + CHUNK] = sub.max(axis=1)

    members: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        members[int(lab)].append(stories[i].id)
    n_pool = len(stories)
    shown_ids = [sid for sid in first if sid in pos]
    rec_ids = [sid for sid in rec_shown if sid in pos]
    n_shown = len(shown_ids)
    ou = sum(1 for sid in shown_ids if outcome(sid) == "up")
    on = sum(1 for sid in shown_ids if outcome(sid) in ACTIONS)
    overall, o_lo, o_hi = wilson(ou, on)
    print(
        f"Overall shown up rate {fmt(overall)} [{fmt(o_lo)}, {fmt(o_hi)}] "
        f"({ou}/{on} with outcome, {n_shown} shown)",
        flush=True,
    )

    stats: list[ClusterStat] = []
    for c in range(k):
        mids = members.get(c, [])
        shown = [sid for sid in mids if sid in first]
        shown_rec = [sid for sid in shown if sid in rec_shown]
        outs = [outcome(sid) for sid in shown]
        n_up = sum(o == "up" for o in outs)
        n_down = sum(o == "down" for o in outs)
        n_neu = sum(o == "neutral" for o in outs)
        rate, lo, hi = wilson(n_up, n_up + n_down + n_neu)
        fe = [votes[sid][0] for sid in mids if sid not in first and sid in recent_votes]
        pool_share = len(mids) / n_pool if n_pool else 0.0
        shown_share = len(shown) / n_shown if n_shown else 0.0
        rec_share = len(shown_rec) / len(rec_ids) if rec_ids else 0.0
        exp = (shown_share / pool_share) if pool_share > 0 else None
        exp_rec = (rec_share / pool_share) if pool_share > 0 else None
        ms = [float(maxsim[pos[sid]]) for sid in mids if not np.isnan(maxsim[pos[sid]])]
        flag = ""
        if rate is not None and overall is not None and exp is not None:
            if lo is not None and lo > overall and exp < UNDER_EXP:
                flag = "UNDER-SHOWN HIT"
            elif hi is not None and hi < overall and exp > OVER_EXP:
                flag = "OVER-SHOWN MISS"
        near: list[str] = []
        if mids:
            idx = [pos[sid] for sid in mids]
            cent = mat64[idx].mean(axis=0)
            nrm = float(np.linalg.norm(cent))
            if nrm <= 0:
                raise ValueError(f"Zero-norm centroid for cluster {c}")
            sims = mat64[idx] @ (cent / nrm)
            near = [titles[mids[j]] for j in np.argsort(-sims)[:NEAREST]]
        stats.append(
            ClusterStat(
                idx=c, n_pool=len(mids), n_shown=len(shown),
                n_shown_rec=len(shown_rec), n_up=n_up, n_down=n_down,
                n_neu=n_neu, n_none=sum(o == "none" for o in outs),
                n_prevoted=sum(o == "prevoted" for o in outs),
                fe_up=sum(a == "up" for a in fe),
                fe_down=sum(a == "down" for a in fe),
                fe_neu=sum(a == "neutral" for a in fe),
                pool_share=pool_share, shown_share=shown_share,
                shown_rec_share=rec_share, exp_ratio=exp,
                exp_rec_ratio=exp_rec, up_rate=rate, w_lo=lo, w_hi=hi,
                mean_maxsim=float(np.mean(ms)) if ms else None,
                flag=flag, nearest_titles=near,
            )
        )
    stats.sort(key=lambda c: (c.exp_ratio is None, c.exp_ratio or 0.0))
    db.close()

    print(f"\nProfile {ns.user}, exposure since {ns.since_iso}: clusters by exposure ratio", flush=True)
    print(
        "clust n_pool pool% n_shown shown% exp n_rec expR up/d/n rate [lo,hi] maxsim flag",
        flush=True,
    )
    for c in stats:
        print(
            f"{c.idx:5d} {c.n_pool:6d} {100 * c.pool_share:5.1f} {c.n_shown:6d} "
            f"{100 * c.shown_share:5.1f} {fmt(c.exp_ratio, 2):>4} {c.n_shown_rec:5d} "
            f"{fmt(c.exp_rec_ratio, 2):>4} {c.n_up}/{c.n_down}/{c.n_neu} "
            f"{fmt(c.up_rate):>5} [{fmt(c.w_lo)},{fmt(c.w_hi)}] "
            f"{fmt(c.mean_maxsim):>6} {c.flag}",
            flush=True,
        )
    hits = [c.idx for c in stats if c.flag == "UNDER-SHOWN HIT"]
    misses = [c.idx for c in stats if c.flag == "OVER-SHOWN MISS"]
    print(f"\nUnder-shown hits (lo > overall, exp < {UNDER_EXP}): {hits}", flush=True)
    print(f"Over-shown misses (hi < overall, exp > {OVER_EXP}): {misses}", flush=True)

    by_src: dict[str, list[int]] = defaultdict(list)
    for s in stories:
        by_src[sources[s.id]].append(s.id)
    print("\nPer-source (any-view shown; up rate among shown-with-outcome):", flush=True)
    print("source n_pool pool% n_shown shown% exp up/d/n rate [lo,hi]", flush=True)
    src_json: dict[str, dict] = {}
    for src in sorted(by_src):
        sids = by_src[src]
        sh = [sid for sid in sids if sid in first]
        u = sum(outcome(sid) == "up" for sid in sh)
        dd = sum(outcome(sid) == "down" for sid in sh)
        nn = sum(outcome(sid) == "neutral" for sid in sh)
        rate, lo, hi = wilson(u, u + dd + nn)
        ps, ss = len(sids) / n_pool, (len(sh) / n_shown if n_shown else 0.0)
        print(
            f"{src} {len(sids)} {100 * ps:.1f} {len(sh)} {100 * ss:.1f} "
            f"{fmt(ss / ps if ps else None, 2)} {u}/{dd}/{nn} "
            f"{fmt(rate)} [{fmt(lo)},{fmt(hi)}]",
            flush=True,
        )
        src_json[src] = {
            "n_pool": len(sids), "pool_share": ps, "n_shown": len(sh),
            "shown_share": ss, "exp_ratio": (ss / ps if ps else None),
            "n_up": u, "n_down": dd, "n_neutral": nn,
            "up_rate": rate, "w_lo": lo, "w_hi": hi,
        }

    with open(ns.out, "w") as f:
        json.dump(
            {
                "user": ns.user, "since_iso": ns.since_iso, "k": k,
                "n_pool": n_pool, "n_shown_any": n_shown,
                "n_shown_rec": len(rec_ids),
                "overall": {"up_rate": overall, "w_lo": o_lo, "w_hi": o_hi,
                            "n_up": ou, "n_with_outcome": on},
                "n_prior_ups": len(prior_vecs),
                "clusters": [asdict(c) for c in stats],
                "sources": src_json,
            },
            f,
        )
    print(f"\nWrote {ns.out} (per-cluster nearest titles are in OUT.json only)", flush=True)


if __name__ == "__main__":
    main()
