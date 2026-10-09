"""Facet-residual step 3: stability gate + residual evaluation (no Muse calls).

Frozen design (facet-design review overrides alternatives on conflicts):
- Field retention SOLELY from 20 repeat agreement >=0.85, BEFORE evaluation.
- Arms (C=1 balanced multinomial, score P(up)-P(down)):
    0 native OOF percentile (no fit)
    1 OOF percentile only
    2 + deterministic proxy (frozen PROXY_RULES below)
    3 + facets (retained fields, one-hot incl unclear)
    4 + proxy + facets
    5 null: arm3 on 200 whole-facet-row permutations within (source, block),
      train/test permuted independently (leakage-safe; note: earlier draft said
      "across train and test together" — NOT used, documented in REPORT).
- Temporal tests: test b4 train b2-3; test b5 train b2-4; test dev99 train 280.
  OOF percentiles rank full-block voted stories (consistent with oof_baseline).
- Uncertainty: paired story bootstrap 2000, stratified by block, conditional
  on fits (no refit), for arm3-arm1 and arm4-arm2 + per-block deltas.
- Metric: ordinal AUC (Down<Neutral<Up pairs within test block, ties 0.5);
  pool b4+b5 by pair count; dev99 reported separately. Up-vs-rest AUC is a
  secondary metric on the same fitted scores (no learner change, no gates).
- Repairs (frozen before results, implementation hash + UTC frozen in-report
  before the first fit): one row per story (n x p); true one-hot per frozen
  train-only vocab key (zeros for unseen test values); arm4 concatenates
  BOTH proxy and facet blocks; null mapping filled group-by-group within
  (raw source, block), train/test separately with independent RNG streams,
  multiset + same-source/block asserts, class-free; proxy body length is
  len(clean(self_text OR article_body OR EMPTY)[:1500]); raw sources map
  rss_reddit_*->reddit, rss_lesswrong*->lesswrong, hn/bq_seed/ch_seed->hn,
  else other (coarse proxy field types unchanged).
- No hyperparameter search. Current snapshot content scores historical votes
  (same footing all arms; reported as risk). All 280+99 are development data:
  result is exploratory, GO means larger labeling only, no production claim.
- Reads (ro): sample.json, rubric.json, facet-cache.json, oof_scores.json,
  snapshot stories columns (title/url/source/self_text/article_body only,
  never votes). Writes (this dir only): facet_fields.json, eval_report.json,
  REPORT.md. No live DB / main / shared-cache writes. Threads=1.

If repeats <20 pairs complete or items <399 parsed: print INFEASIBLE/PARTIAL
and exit nonzero WITHOUT fitting (no fishing for more calls).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path("/tmp/opencode/facet-residual-20261009/snapshot.db")


# --- Frozen deterministic proxy rules (frozen before eval; no tuning) ---
class _ProxyRules(TypedDict):
    version: str
    title_prefix: list[str]
    url_kind: list[str]
    domain_class: list[str]
    source_values: list[str]
    text_len_bins: list[int]
    note: str


PROXY_RULES: _ProxyRules = {
    "version": "proxy-v1-20261009",
    "title_prefix": ["ask hn:", "show hn:", "launch hn:", "tell hn:"],
    "url_kind": ["github.com", "arxiv", "youtube", "twitter/x", "other/none"],
    "domain_class": ["code", "paper", "video", "social", "news", "other"],
    "source_values": ["hn", "reddit", "lesswrong"],
    "text_len_bins": [0, 1, 200, 800, 1500],
    "note": "regexes fixed here; text_len from repaired body length only",
}

N_BOOT = 2000
N_NULL = 200
SEED = 20261009


def _domain(url: str | None) -> str:
    try:
        return (urlparse(url).netloc or "").removeprefix("www.").lower() if url else ""
    except Exception:
        return ""


def clean(text: str | None) -> str:
    # MUST mirror label_facets.clean: strip tags, collapse whitespace.
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())


def normalize_source(source: str | None) -> str:
    # Repair: map raw snapshot sources onto the frozen coarse categories
    # (hn / reddit / lesswrong / other). Field types unchanged.
    s = (source or "").lower()
    if s.startswith("rss_reddit_"):
        return "reddit"
    if s.startswith("rss_lesswrong"):
        return "lesswrong"
    if s in ("hn", "bq_seed", "ch_seed"):
        return "hn"
    return "other"


def proxy_features(
    title: str, url: str | None, source: str, body_len: int
) -> dict[str, str]:
    t = (title or "").lower()
    d = _domain(url)
    if "ask hn:" in t[:12]:
        prefix = "ask hn:"
    elif "show hn:" in t[:12]:
        prefix = "show hn:"
    elif "launch hn:" in t[:12]:
        prefix = "launch hn:"
    elif "tell hn:" in t[:12]:
        prefix = "tell hn:"
    else:
        prefix = "other"
    if "github.com" in d:
        ukind = "github.com"
    elif "arxiv" in d:
        ukind = "arxiv"
    elif d in ("youtube.com", "youtu.be"):
        ukind = "youtube"
    elif d in ("twitter.com", "x.com"):
        ukind = "twitter/x"
    else:
        ukind = "other/none"
    if "github" in d or "gitlab" in d:
        dclass = "code"
    elif "arxiv" in d or "biorxiv" in d or "medrxiv" in d:
        dclass = "paper"
    elif ukind == "youtube":
        dclass = "video"
    elif ukind == "twitter/x" or "reddit.com" in d:
        dclass = "social"
    elif "nytimes" in d or "bbc" in d or "reuters" in d:
        dclass = "news"
    else:
        dclass = "other"
    bins = PROXY_RULES["text_len_bins"]
    bl = (
        "len0"
        if body_len <= 0
        else next(
            (
                f"len>{bins[i]}"
                for i in range(len(bins) - 1, 0, -1)
                if body_len > bins[i]
            ),
            "len1+",
        )
    )
    return {
        "prefix": prefix,
        "ukind": ukind,
        "dclass": dclass,
        "source": normalize_source(source),
        "blen": bl,
    }


def ordinal_auc(y: list[int], s: list[float]) -> float:
    n = len(y)
    num = den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            den += 1
            si, sj = s[i], s[j]
            want = 1 if y[i] > y[j] else -1
            got = (si > sj) - (si < sj)
            num += 1.0 if got == want else (0.5 if got == 0 else 0.0)
    return num / den if den else float("nan")


def _pair_count(y: list[int]) -> int:
    n = len(y)
    tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] != y[j]:
                tot += 1
    return tot


def uprest_auc(y: list[int], s: list[float]) -> float:
    # Secondary metric ONLY (same fitted scores, no learner change):
    # P(up) vs rest discrimination under the same pair-counting rule.
    return ordinal_auc([1 if v == 2 else 0 for v in y], s)


def _env_snapshot() -> dict:
    out: dict[str, str] = {"threads": "1"}
    for cmd in (["uptime"], ["free", "-m"]):
        try:
            out[" ".join(cmd)] = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            ).stdout.strip()[:400]
        except Exception as e:  # noqa: BLE001 - diagnostic only
            out[" ".join(cmd)] = f"unavailable: {e}"
    return out


def _check_onehot(
    sids: list[int],
    X: "object",
    pkeys: list[str],
    prow: dict[int, dict[str, str]],
    is_train: bool,
) -> None:
    # Focused invariant: proxy block holds a true one-hot per frozen vocab
    # key (1 iff the row's value equals the key; zeros for unseen test
    # values). Train rows must show exactly one 1 per proxy field.
    import numpy as _np  # noqa: PLC0415 - local to keep import surface stable

    Xa = _np.asarray(X, dtype=float)
    pblock = Xa[:, 1 : 1 + len(pkeys)]
    fields = ("prefix", "ukind", "dclass", "source", "blen")
    for ri, s in enumerate(sids):
        for f in fields:
            idx = [i for i, k in enumerate(pkeys) if k.startswith(f + "=")]
            got = pblock[ri, idx].sum()
            want = 1.0 if f"{f}={prow[s][f]}" in pkeys else 0.0
            assert got == want, (s, f, got, want)
        if is_train:
            assert pblock[ri].sum() == len(fields), (s, pblock[ri].sum())


def main() -> int:
    import numpy as np

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("INFEASIBLE: sklearn unavailable; no fits attempted.")
        return 2

    sample = json.loads((HERE / "sample.json").read_text())
    rubric = json.loads((HERE / "rubric.json").read_text())
    cache_p = (
        HERE / "facet_labels.json"
        if (HERE / "facet_labels.json").exists()
        else HERE / "facet-cache.json"
    )
    cache_raw = json.loads(cache_p.read_text())
    cache = cache_raw["results"] if "results" in cache_raw else cache_raw
    oof = json.loads((HERE / "oof_scores.json").read_text())

    flat = [it for b in sample["batches"] for it in b]
    fields: list[str] = sorted(rubric["fields"].keys())

    # --- Stability gate: SOLELY 20 repeat pairs, BEFORE any eval ---
    by_sid: dict[int, list[dict]] = defaultdict(list)
    for it in flat:
        by_sid[it["story_id"]].append(it)
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    missing_pairs = 0
    for sid in sample["repeats20"]:
        ops = by_sid.get(sid, [])
        ps = [(op, (cache.get(op["opaque"], {}) or {}).get("parsed")) for op in ops]
        if len(ops) != 2 or any(p is None for _, p in ps):
            missing_pairs += 1
            continue
        first_parsed, second_parsed = ps[0][1], ps[1][1]
        assert first_parsed is not None and second_parsed is not None, (
            f"repeat {sid} missing parsed facets"
        )
        pairs.append((first_parsed, second_parsed))
    agree = {f: sum(1 for a, b in pairs if a[f] == b[f]) for f in fields}
    rate = {f: (agree[f] / len(pairs) if pairs else 0.0) for f in fields}
    retained = sorted(f for f in fields if rate.get(f, 0) >= 0.85)
    # Invariant: retention from repeats ALONE, before any fit/outcome use.
    assert set(retained) <= set(fields) and len(pairs) == 20, (retained, len(pairs))
    (HERE / "facet_fields.json").write_text(
        json.dumps(
            {
                "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "n_pairs_complete": len(pairs),
                "n_pairs_missing": missing_pairs,
                "agree": agree,
                "rate": {k: round(v, 4) for k, v in rate.items()},
                "retained": retained,
                "rule": "retain iff repeat agreement >=0.85",
                "proxy_rules": PROXY_RULES,
                "proxy_sha16": hashlib.sha256(
                    json.dumps(PROXY_RULES, sort_keys=True).encode()
                ).hexdigest()[:16],
            },
            indent=1,
        )
    )

    n_ok = sum(1 for it in flat if (cache.get(it["opaque"], {}) or {}).get("parsed"))
    if len(pairs) < 20 or n_ok < 399:
        print(
            json.dumps(
                {
                    "status": "INFEASIBLE/PARTIAL",
                    "n_ok": n_ok,
                    "need": 399,
                    "pairs_complete": len(pairs),
                    "pairs_need": 20,
                    "agree": agree,
                    "rate": {k: round(v, 4) for k, v in rate.items()},
                },
                indent=1,
            )
        )
        print(
            "INFEASIBLE: need 399 parsed + 20 repeat pairs before evaluation; "
            f"have {n_ok}/399 and {len(pairs)}/20. No fits run, no gates assessed."
        )
        return 2

    # --- Full evaluation (complete data only) ---
    env = _env_snapshot()
    print(f"env before fits: {json.dumps(env)[:600]}", flush=True)

    # OOF lookup: story_id -> (true_label, percentile, block_key)
    oof_by_sid: dict[int, dict] = {}
    for r in oof["blocks"]:
        oof_by_sid[r["story_id"]] = {
            "y": r["true_label"],
            "pct": r["oof_percentile_in_block"],
            "block": f"b{r['block']}",
        }
    for r in oof["dev99"]:
        oof_by_sid[r["story_id"]] = {
            "y": r["true_label"],
            "pct": r["oof_percentile_in_block"],
            "block": "dev",
        }
    # Original (non-repeat) opaque per story for feature join
    orig_opaque: dict[int, str] = {}
    for it in flat:
        if it["kind"] != "repeat" and it["story_id"] not in orig_opaque:
            orig_opaque[it["story_id"]] = it["opaque"]
    sid_block: dict[int, str] = {}
    for bname in ("block2", "block3", "block4", "block5"):
        for sid in sample["sample70"][bname]:
            sid_block[sid] = bname
    for sid in sample["dev99"]:
        sid_block[sid] = "dev"

    # Snapshot columns for proxy only (never votes/labels/scores)
    con = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro", uri=True)
    meta: dict[int, dict] = {}
    for sid in orig_opaque:
        row = con.execute(
            "SELECT title, url, source, self_text, article_body FROM stories WHERE id=?",
            (sid,),
        ).fetchone()
        assert row is not None, f"story {sid} missing"
        title, url, source, stx, ab = row
        # REPAIRED body length: same visible input the annotator saw —
        # clean(self_text OR article_body OR EMPTY)[:1500]. NEVER
        # comments/text_content, and NOT the sum of both body fields.
        body_len = len(clean(stx or ab or "")[:1500])
        meta[sid] = {
            "title": title or "",
            "url": url,
            "source": source or "hn",
            "body_len": body_len,
        }
    con.close()

    # Design matrices per split
    splits = {
        "b4": (
            sample["sample70"]["block2"] + sample["sample70"]["block3"],
            sample["sample70"]["block4"],
        ),
        "b5": (
            sample["sample70"]["block2"]
            + sample["sample70"]["block3"]
            + sample["sample70"]["block4"],
            sample["sample70"]["block5"],
        ),
        "dev99": (sample["s280"], sample["dev99"]),
    }
    # Fixed one-hot vocabularies from train only per split (no test leak).
    # Invariants (asserted per split below): one row per story (n x p);
    # true one-hot per frozen pkey (zeros for unseen test values); arm4
    # width == arm1 width + len(pkeys) + len(fkeys) (concatenates BOTH).
    results: dict = {
        "arms": {},
        "env": env,
        "proxy_sha16": hashlib.sha256(
            json.dumps(PROXY_RULES, sort_keys=True).encode()
        ).hexdigest()[:16],
        "impl_sha16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "impl_frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    print(
        f"impl {results['impl_sha16']} frozen at {results['impl_frozen_at']} "
        f"BEFORE first fit; retained={retained}",
        flush=True,
    )

    def build_matrix(tr: list[int], te: list[int]):
        facets = {sid: cache[orig_opaque[sid]]["parsed"] for sid in tr + te}
        prow = {
            sid: proxy_features(
                meta[sid]["title"],
                meta[sid]["url"],
                meta[sid]["source"],
                meta[sid]["body_len"],
            )
            for sid in tr + te
        }
        pkeys = sorted({f"{k}={v}" for s in tr for k, v in prow[s].items()})
        fkeys = sorted({f"{fld}={facets[s][fld]}" for s in tr for fld in retained})

        def vec(sids: list[int], use_p: bool, use_f: bool) -> np.ndarray:
            rows: list[list[float]] = []
            for s in sids:
                po = {f"{k}={v}" for k, v in prow[s].items()}
                fo = {f"{fld}={facets[s][fld]}" for fld in retained}
                row = [oof_by_sid[s]["pct"]]
                if use_p:
                    row += [1.0 if k in po else 0.0 for k in pkeys]
                if use_f:
                    row += [1.0 if k in fo else 0.0 for k in fkeys]
                rows.append(row)
            X = np.asarray(rows, dtype=float)
            assert X.shape == (
                len(sids),
                1 + (len(pkeys) if use_p else 0) + (len(fkeys) if use_f else 0),
            ), X.shape
            return X

        ytr = [oof_by_sid[s]["y"] for s in tr]
        yte = [oof_by_sid[s]["y"] for s in te]
        return vec, ytr, yte, facets, prow, pkeys, fkeys

    def fit_score(Xtr: np.ndarray, ytr: list[int], Xte: np.ndarray) -> list[float]:
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED,
        )
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        cls = list(clf.classes_)
        iu, idn = cls.index(2), cls.index(0)
        return (proba[:, iu] - proba[:, idn]).tolist()

    arm_aucs: dict[str, dict[str, float]] = {}
    arm_y: dict[str, list[int]] = {}
    arm_scores: dict[str, dict[str, list[float]]] = {}
    arm_uprest: dict[str, dict[str, float]] = {}
    for name, (tr, te) in splits.items():
        vec, ytr, yte, _, prow_sp, pkeys_sp, fkeys_sp = build_matrix(tr, te)
        X1tr, X1te = vec(tr, False, False), vec(te, False, False)
        X2tr, X2te = vec(tr, True, False), vec(te, True, False)
        X3tr, X3te = vec(tr, False, True), vec(te, False, True)
        X4tr, X4te = vec(tr, True, True), vec(te, True, True)
        # Focused invariants: n x p shapes; arm4 concatenates BOTH blocks.
        assert X1tr.shape == (len(tr), 1) and X1te.shape == (len(te), 1)
        assert X2tr.shape == (len(tr), 1 + len(pkeys_sp))
        assert X3tr.shape == (len(tr), 1 + len(fkeys_sp))
        assert X4tr.shape == (len(tr), 1 + len(pkeys_sp) + len(fkeys_sp)), (
            X4tr.shape,
            len(pkeys_sp),
            len(fkeys_sp),
        )
        assert X4te.shape == (len(te), 1 + len(pkeys_sp) + len(fkeys_sp))
        _check_onehot(tr, X2tr, pkeys_sp, prow_sp, True)
        _check_onehot(te, X2te, pkeys_sp, prow_sp, False)
        s0 = [oof_by_sid[s]["pct"] for s in te]
        s1 = fit_score(X1tr, ytr, X1te)
        s2 = fit_score(X2tr, ytr, X2te)
        s3 = fit_score(X3tr, ytr, X3te)
        s4 = fit_score(X4tr, ytr, X4te)
        arm_scores[name] = {"s0": s0, "s1": s1, "s2": s2, "s3": s3, "s4": s4}
        arm_aucs[name] = {k: ordinal_auc(yte, v) for k, v in arm_scores[name].items()}
        arm_uprest[name] = {k: uprest_auc(yte, v) for k, v in arm_scores[name].items()}
        arm_y[name] = yte
    # Pooled b4+b5 by pair count
    pooled: dict[str, float] = {}
    for arm in ("s0", "s1", "s2", "s3", "s4"):
        num = den = 0.0
        for name in ("b4", "b5"):
            yte = arm_y[name]
            auc, pc = arm_aucs[name][arm], _pair_count(yte)
            num += auc * pc
            den += pc
        pooled[arm] = num / den if den else float("nan")
    results["auc"] = {
        k: {a: v for a, v in d.items() if a != "y"} for k, d in arm_aucs.items()
    }
    results["auc"]["pooled_b4_b5"] = pooled
    # Secondary up-vs-rest (same scores, no learner change, no gates).
    uprest_pooled: dict[str, float] = {}
    for arm in ("s0", "s1", "s2", "s3", "s4"):
        num = den = 0.0
        for name in ("b4", "b5"):
            yte = arm_y[name]
            pc = _pair_count([1 if v == 2 else 0 for v in yte])
            num += arm_uprest[name][arm] * pc
            den += pc
        uprest_pooled[arm] = num / den if den else float("nan")
    results["auc_uprest"] = arm_uprest
    results["auc_uprest"]["pooled_b4_b5"] = uprest_pooled
    results["delta_pooled_arm3_arm1"] = pooled["s3"] - pooled["s1"]
    results["delta_pooled_arm4_arm2"] = pooled["s4"] - pooled["s2"]
    results["delta_dev99_arm3_arm1"] = arm_aucs["dev99"]["s3"] - arm_aucs["dev99"]["s1"]

    # Null: 200 whole-facet-row perms within (RAW source, block).
    # Train and test strata permuted SEPARATELY (no cross transfer), each
    # with an INDEPENDENT RNG stream per (rep, split, part). Mapping is
    # filled group-by-group: zip(group sids, shuffled group facet rows).
    # Class-free: only sids/facets/sources/blocks enter the mapping — never
    # labels/outcomes/scores (structural: no y is passed or read here).
    def _permute_group(
        grp: list[int],
        get_row: Callable[[int], dict[str, str]],
        rng: np.random.Generator,
    ) -> dict[int, dict[str, str]]:
        import numpy as _np  # noqa: PLC0415 - local to keep import surface stable

        original = [get_row(s) for s in grp]
        order = list(_np.asarray(rng.permutation(len(grp))).tolist())
        assigned = [original[i] for i in order]
        before = sorted(json.dumps(r, sort_keys=True) for r in original)
        after = sorted(json.dumps(r, sort_keys=True) for r in assigned)
        assert before == after, "stratum facet-row multiset changed"
        return dict(zip(grp, assigned))

    null_deltas: list[float] = []
    for rep in range(N_NULL):
        num = den = 0.0
        for si, (name, (tr, te)) in enumerate(
            (("b4", splits["b4"]), ("b5", splits["b5"]))
        ):
            assert not (set(tr) & set(te)), "train/test sid overlap"
            get_row = lambda s: cache[orig_opaque[s]]["parsed"]  # noqa: E731
            pmap: dict[int, dict] = {}
            for part, sids in (("tr", tr), ("te", te)):
                rng = np.random.default_rng(
                    (SEED, 1000, rep, si, 0 if part == "tr" else 1)
                )
                strata: dict[tuple, list[int]] = defaultdict(list)
                for s in sids:
                    strata[(meta[s]["source"], sid_block[s])].append(s)
                for (raw_src, blk), grp in strata.items():
                    assert all(
                        meta[s]["source"] == raw_src and sid_block[s] == blk
                        for s in grp
                    )
                    pmap.update(_permute_group(grp, get_row, rng))
            assert set(pmap) == set(tr) | set(te)
            # rebuild facet one-hots from permuted rows (train vocab only)
            fkeys = sorted({f"{fld}={pmap[s][fld]}" for s in tr for fld in retained})

            def pvec(sids: list[int]) -> np.ndarray:
                def _row(s: int) -> list[float]:
                    rowset = {f"{f}={pmap[s][f]}" for f in retained}
                    return [oof_by_sid[s]["pct"]] + [
                        1.0 if k in rowset else 0.0 for k in fkeys
                    ]

                P = np.asarray([_row(s) for s in sids], dtype=float)
                assert P.shape == (len(sids), 1 + len(fkeys)), P.shape
                return P

            ytr = [oof_by_sid[s]["y"] for s in tr]
            yte = [oof_by_sid[s]["y"] for s in te]
            s1 = fit_score(
                np.asarray([[oof_by_sid[s]["pct"]] for s in tr]),
                ytr,
                np.asarray([[oof_by_sid[s]["pct"]] for s in te]),
            )
            s3 = fit_score(pvec(tr), ytr, pvec(te))
            a1, a3 = ordinal_auc(yte, s1), ordinal_auc(yte, s3)
            pc = _pair_count(yte)
            num += (a3 - a1) * pc
            den += pc
        null_deltas.append(num / den if den else float("nan"))
    results["null_pooled_delta_mean"] = float(np.mean(null_deltas))
    results["null_pooled_delta_p95"] = float(np.percentile(null_deltas, 95))
    results["null_n"] = N_NULL

    # Bootstrap 2000 paired-story, stratified by block, conditional on fits.
    # One index draw per block per rep; the same draw feeds per-block and
    # pooled deltas (paired within rep).
    boots: dict[str, list[float]] = {"d31": [], "d42": []}
    per_block: dict[str, dict[str, list[float]]] = {}
    for rep in range(N_BOOT):
        r = np.random.default_rng((SEED, 5000, rep))
        y4 = arm_y["b4"]
        y5 = arm_y["b5"]
        i4 = r.integers(0, len(y4), len(y4))
        i5 = r.integers(0, len(y5), len(y5))
        blk_d: dict[str, dict[str, float]] = {}
        for name, yy, ii in (("b4", y4, i4), ("b5", y5, i5)):
            ys = [yy[i] for i in ii]
            d31 = ordinal_auc(
                ys, [arm_scores[name]["s3"][i] for i in ii]
            ) - ordinal_auc(ys, [arm_scores[name]["s1"][i] for i in ii])
            d42 = ordinal_auc(
                ys, [arm_scores[name]["s4"][i] for i in ii]
            ) - ordinal_auc(ys, [arm_scores[name]["s2"][i] for i in ii])
            blk_d[name] = {"d31": d31, "d42": d42}
            per_block.setdefault(name, {"d31": [], "d42": []})["d31"].append(d31)
            per_block.setdefault(name, {"d42": []})["d42"].append(d42)
        for ak, bk, out in (("s3", "s1", "d31"), ("s4", "s2", "d42")):
            a4 = ordinal_auc(
                [y4[i] for i in i4], [arm_scores["b4"][ak][i] for i in i4]
            ) - ordinal_auc([y4[i] for i in i4], [arm_scores["b4"][bk][i] for i in i4])
            a5 = ordinal_auc(
                [y5[i] for i in i5], [arm_scores["b5"][ak][i] for i in i5]
            ) - ordinal_auc([y5[i] for i in i5], [arm_scores["b5"][bk][i] for i in i5])
            pc4 = _pair_count([y4[i] for i in i4])
            pc5 = _pair_count([y5[i] for i in i5])
            boots[out].append(
                (a4 * pc4 + a5 * pc5) / (pc4 + pc5) if pc4 + pc5 else float("nan")
            )
    results["boot"] = {
        k: {
            "mean": float(np.mean(v)),
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)),
        }
        for k, v in boots.items()
    }
    results["boot_per_block"] = {
        b: {
            k: {
                "mean": float(np.mean(v)),
                "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5)),
            }
            for k, v in d.items()
        }
        for b, d in per_block.items()
    }

    # Gates
    g = {
        "arm3-arm1>=+0.02 pooled": results["delta_pooled_arm3_arm1"] >= 0.02,
        "arm3>p95 null": results["delta_pooled_arm3_arm1"]
        > results["null_pooled_delta_p95"],
        "arm4-arm2>=+0.01": results["delta_pooled_arm4_arm2"] >= 0.01,
        "dev99 delta>=0": results["delta_dev99_arm3_arm1"] >= 0,
        "arm3>=arm0": pooled["s3"] >= pooled["s0"],
    }
    results["gates"] = g
    results["stop"] = not (
        results["delta_pooled_arm3_arm1"] >= 0.01 and pooled["s3"] > pooled["s2"]
    )
    results["verdict"] = (
        "GO larger labeling only"
        if all(g.values())
        else "STOP facet line"
        if results["stop"]
        else "promising, inconclusive"
    )
    (HERE / "eval_report.json").write_text(json.dumps(results, indent=1))
    print(
        json.dumps(
            {k: v for k, v in results.items() if k not in ("boot_per_block",)}, indent=1
        )[:4000]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
