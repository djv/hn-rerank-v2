"""Bounded source diagnostic (frozen protocol 2026-10-09, PLAN.md).

Artifact-only: joins existing baseline predictions with read-only snapshot
metadata (id/source/time) and writes aggregate JSON. No ranking, fitting,
embeddings, or production reads/writes. All computation runs on the VPS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

FROZEN_CLOCK = 1791401987.9873054
HN_SOURCES = frozenset({"hn", "ch_seed", "bq_seed"})
CLASSES = ("HN", "non-HN")
AGE_BINS = ("0-7", "8-30", "31-90", "90+")
ALL_BINS = (*AGE_BINS, "unknown")
SUPPORT_MIN = 10
SUPPORT_MASS_MIN = 0.50
BOTTOM_FRAC = 0.20
UP, NEUTRAL, DOWN = 2, 1, 0
J = dict[str, Any]  # JSON aggregate assembly (serialization boundary only)


@dataclass(frozen=True)
class BaselineRow:
    block: int
    story_id: int
    score: float
    label: int
    vote_time: float


@dataclass(frozen=True)
class JoinedRow:
    base: BaselineRow
    source: str
    story_time: int | None


def source_class(source: str) -> str:
    return "HN" if source in HN_SOURCES else "non-HN"


def age_days(story_time: int | None, ref: float) -> float | None:
    if story_time is None or story_time <= 0:
        return None
    age = (ref - story_time) / 86400.0
    return age if age >= 0 else None


def age_bin(age: float | None) -> str:
    if age is None:
        return "unknown"
    if age <= 7:
        return "0-7"
    if age <= 30:
        return "8-30"
    if age <= 90:
        return "31-90"
    return "90+"


def binlabel(label: int, endpoint: str) -> int | None:
    if endpoint == "up_down":
        if label == UP:
            return 1
        return 0 if label == DOWN else None
    return 1 if label == UP else 0


def pairs(sub: list[JoinedRow], endpoint: str) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for j in sub:
        v = binlabel(j.base.label, endpoint)
        if v is not None:
            scores.append(j.base.score)
            labels.append(v)
    return scores, labels


def auc_sparse(scores: list[float], labels: list[int]) -> tuple[float | None, bool]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    sparse = n_pos < SUPPORT_MIN or n_neg < SUPPORT_MIN
    if n_pos == 0 or n_neg == 0:
        return None, sparse
    return float(roc_auc_score(labels, scores)), sparse


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wal_bytes(snapshot: Path) -> int:
    wal = Path(str(snapshot) + "-wal")
    return wal.stat().st_size if wal.exists() else 0


def load_baseline(path: Path, expected_prefix: str) -> tuple[list[BaselineRow], str]:
    digest = sha256_file(path)
    if not digest.startswith(expected_prefix):
        raise ValueError(f"baseline sha mismatch: {digest}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("baseline JSON must be a list")
    rows: list[BaselineRow] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("baseline row must be an object")
        try:
            rows.append(
                BaselineRow(
                    int(item["block"]),
                    int(item["story_id"]),
                    float(item["production_score"]),
                    int(item["true_label"]),
                    float(item["vote_time"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bad baseline row: {item!r:.120}") from exc
    return rows, digest


def load_source_meta(
    snapshot: Path, ids: list[int]
) -> dict[int, tuple[str, int | None]]:
    meta: dict[int, tuple[str, int | None]] = {}
    con = sqlite3.connect(f"file:{snapshot}?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA query_only=ON")
    try:
        for lo in range(0, len(ids), 500):
            chunk = ids[lo : lo + 500]
            holes = ",".join("?" for _ in chunk)
            cur = con.execute(
                f"SELECT id, source, time FROM stories WHERE id IN ({holes})", chunk
            )
            for sid, source, stime in cur:
                meta[int(sid)] = (
                    str(source),
                    int(stime) if stime is not None else None,
                )
    finally:
        con.close()
    return meta


def score_summary(scores: list[float]) -> J:
    arr = np.asarray(scores, dtype=float)
    q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return {
        "min": float(arr.min()),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "max": float(arr.max()),
    }


def class_entry(block: int, cls: str, sub: list[JoinedRow]) -> J:
    lb = [j.base.label for j in sub]
    n_up = lb.count(UP)
    a_ud, sparse_ud = auc_sparse(*pairs(sub, "up_down"))
    a_ur, sparse_ur = auc_sparse(*pairs(sub, "up_rest"))
    return {
        "block": block,
        "class": cls,
        "n": len(sub),
        "n_up": n_up,
        "n_neutral": lb.count(NEUTRAL),
        "n_down": lb.count(DOWN),
        "up_rate": n_up / len(sub) if sub else None,
        "scores": score_summary([j.base.score for j in sub]),
        "auc_up_down": a_ud,
        "auc_up_rest": a_ur,
        "sparse_up_down": sparse_ud,
        "sparse_up_rest": sparse_ur,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("baseline", "snapshot", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--plan-hash", required=True)
    ap.add_argument("--expected-sha-prefix", required=True)
    args = ap.parse_args()

    snapshot_before = args.snapshot.stat()

    wal_before = wal_bytes(args.snapshot)
    if wal_before != 0:
        raise ValueError(f"snapshot WAL nonempty ({wal_before}B): not immutable")
    rows, digest = load_baseline(args.baseline, args.expected_sha_prefix)

    assert len(rows) == 3079, f"rows={len(rows)}"
    blocks = sorted({r.block for r in rows})
    assert blocks == [2, 3, 4, 5], f"blocks={blocks}"
    labels = [r.label for r in rows]
    assert labels.count(UP) == 683 and labels.count(DOWN) == 1331
    assert all(np.isfinite(r.score) for r in rows), "non-finite score"
    ids = [r.story_id for r in rows]
    assert len(set(ids)) == 3079, "duplicate story ids"
    meta = load_source_meta(args.snapshot, ids)
    assert len(meta) == 3079, f"source_meta coverage={len(meta)}"
    joined = [JoinedRow(b, meta[b.story_id][0], meta[b.story_id][1]) for b in rows]

    comp: dict[str, int] = {}
    for j in joined:
        comp[j.source] = comp.get(j.source, 0) + 1
    out: J = {
        "meta": {
            "baseline_sha256": digest,
            "plan_sha256": args.plan_hash,
            "snapshot": str(args.snapshot),
            "snapshot_wal_bytes_before": wal_before,
            "frozen_clock": FROZEN_CLOCK,
            "n_rows": len(rows),
            "hn_sources": sorted(HN_SOURCES),
        },
        "validation": {
            "n": len(rows),
            "blocks": blocks,
            "n_up": 683,
            "n_down": 1331,
            "meta_coverage": len(meta),
        },
        "composition": {
            "hn": {s: n for s, n in sorted(comp.items()) if s in HN_SOURCES},
            "non_hn": {s: n for s, n in sorted(comp.items()) if s not in HN_SOURCES},
        },
    }
    by_block: dict[int, list[JoinedRow]] = {b: [] for b in blocks}
    for j in joined:
        by_block[j.base.block].append(j)

    block_rows: list[J] = []
    down_rows: list[J] = []
    band_sections: list[J] = []
    for block in blocks:
        bro = by_block[block]
        by_cls = {c: [j for j in bro if source_class(j.source) == c] for c in CLASSES}
        for cls in CLASSES:
            block_rows.append(class_entry(block, cls, by_cls[cls]))
        ordered = sorted(j.base.score for j in bro)
        cutoff = ordered[max(1, int(BOTTOM_FRAC * len(bro))) - 1]
        bottom = [j for j in bro if j.base.score <= cutoff]
        per_class = {}
        for cls in CLASSES:
            d_total = sum(1 for j in by_cls[cls] if j.base.label == DOWN)
            d_in = sum(
                1
                for j in bottom
                if source_class(j.source) == cls and j.base.label == DOWN
            )
            per_class[cls] = {
                "n_down": d_total,
                "down_in_bottom": d_in,
                "placement_rate": d_in / d_total if d_total else None,
            }
        down_rows.append(
            {
                "block": block,
                "n": len(bro),
                "cutoff": cutoff,
                "bottom_size": len(bottom),
                "per_class": per_class,
            }
        )
        edges = np.quantile(np.asarray(ordered), np.linspace(0, 1, 11))
        inner = list(edges[1:-1])
        bands = []
        for bnd in range(10):
            in_band = [
                j for j in bro if int(np.sum(np.asarray(inner) < j.base.score)) == bnd
            ]
            bsc = [j.base.score for j in in_band]
            entry: J = {
                "band": bnd,
                "n": len(in_band),
                "lo": min(bsc) if bsc else None,
                "hi": max(bsc) if bsc else None,
            }
            for cls in CLASSES:
                sub = [j for j in in_band if source_class(j.source) == cls]
                mix: dict[str, int] = dict.fromkeys(ALL_BINS, 0)
                for j in sub:
                    mix[age_bin(age_days(j.story_time, FROZEN_CLOCK))] += 1
                n_up = sum(1 for j in sub if j.base.label == UP)
                entry[cls] = {
                    "n": len(sub),
                    "n_up": n_up,
                    "n_down": sum(j.base.label == DOWN for j in sub),
                    "n_neutral": sum(j.base.label == NEUTRAL for j in sub),
                    "sparse": len(sub) < SUPPORT_MIN,
                    "up_rate": n_up / len(sub) if sub else None,
                    "age_mix_primary": mix,
                }
            bands.append(entry)
        assert sum(b["n"] for b in bands) == len(bro), "band counts must sum to block n"
        band_sections.append(
            {"block": block, "edges": [float(e) for e in edges], "bands": bands}
        )
    out["per_block_class"] = block_rows
    out["down_placement"] = down_rows
    out["score_bands"] = band_sections

    strata_rows: list[J] = []
    support_rows: list[J] = []
    for basis in ("primary", "at_vote"):
        for block in blocks:
            bro = by_block[block]
            binned: dict[str, dict[str, list[JoinedRow]]] = {
                bn: {c: [] for c in CLASSES} for bn in ALL_BINS
            }
            for j in bro:
                ref = FROZEN_CLOCK if basis == "primary" else j.base.vote_time
                binned[age_bin(age_days(j.story_time, ref))][
                    source_class(j.source)
                ].append(j)
            for bn in ALL_BINS:
                for cls in CLASSES:
                    sub = binned[bn][cls]
                    lb = [x.base.label for x in sub]
                    n_up = lb.count(UP)
                    a_ud, sp_ud = auc_sparse(*pairs(sub, "up_down"))
                    a_ur, sp_ur = auc_sparse(*pairs(sub, "up_rest"))
                    sparse = sp_ud or n_up < SUPPORT_MIN
                    strata_rows.append(
                        {
                            "basis": basis,
                            "block": block,
                            "bin": bn,
                            "class": cls,
                            "n": len(sub),
                            "n_up": n_up,
                            "n_down": lb.count(DOWN),
                            "n_neutral": lb.count(NEUTRAL),
                            "up_rate": n_up / len(sub) if sub else None,
                            "auc_up_down": a_ud,
                            "auc_up_rest": a_ur,
                            "sparse": sparse,
                            "sparse_up_down": sp_ud,
                            "sparse_up_rest": sp_ur,
                        }
                    )
            pooled = {bn: sum(len(binned[bn][c]) for c in CLASSES) for bn in AGE_BINS}
            known = sum(pooled.values())
            for endpoint in ("up_down", "up_rest"):
                bin_auc: dict[str, dict[str, float | None]] = {c: {} for c in CLASSES}
                retained = []
                for bn in AGE_BINS:
                    counts = {}
                    for cls in CLASSES:
                        s, v = pairs(binned[bn][cls], endpoint)
                        n1, n0 = sum(v), len(v) - sum(v)
                        counts[cls] = (n1, n0)
                        bin_auc[cls][bn] = (
                            float(roc_auc_score(v, s)) if n1 and n0 else None
                        )
                    if all(
                        p >= SUPPORT_MIN and q >= SUPPORT_MIN
                        for p, q in counts.values()
                    ):
                        retained.append(bn)
                kept = sum(pooled[bn] for bn in retained)
                mass = kept / known if known else 0.0
                conclusive = bool(retained) and mass >= SUPPORT_MASS_MIN
                weighted = {}
                for cls in CLASSES:
                    if conclusive:
                        parts = [bin_auc[cls][bn] for bn in retained]
                        assert all(p is not None for p in parts)
                        wsum = sum(
                            pooled[bn] / kept * p
                            for bn, p in zip(retained, parts)
                            if p is not None
                        )
                        weighted[cls] = float(wsum)
                    else:
                        weighted[cls] = None
                support_rows.append(
                    {
                        "basis": basis,
                        "block": block,
                        "endpoint": endpoint,
                        "retained_bins": retained,
                        "weights": {bn: pooled[bn] / kept for bn in retained}
                        if kept
                        else {},
                        "support_mass": mass,
                        "known_n": known,
                        "conclusive": conclusive,
                        "weighted_auc": weighted,
                    }
                )
    out["age_strata"] = strata_rows
    out["common_support"] = support_rows

    assert sum(r["n"] for r in block_rows) == 3079
    for s in support_rows:
        assert 0.0 <= s["support_mass"] <= 1.0
        assert (
            abs(sum(s["weights"].values()) - (1.0 if s["retained_bins"] else 0.0))
            < 1e-9
        )
    for r in block_rows + strata_rows:
        for key in ("auc_up_down", "auc_up_rest"):
            if r[key] is not None:
                assert 0.0 <= r[key] <= 1.0, (key, r[key])
    wal_after = wal_bytes(args.snapshot)
    out["meta"]["snapshot_wal_bytes_after"] = wal_after
    assert wal_after == 0, "snapshot WAL changed during run"
    snapshot_after = args.snapshot.stat()
    assert (snapshot_before.st_size, snapshot_before.st_mtime_ns) == (
        snapshot_after.st_size,
        snapshot_after.st_mtime_ns,
    ), "snapshot file identity changed during run"
    assert sha256_file(args.baseline) == digest, "baseline changed during run"
    out["meta"]["snapshot_identity_unchanged"] = True
    out["meta"]["snapshot_size"] = snapshot_after.st_size
    out["meta"]["snapshot_mtime_ns"] = snapshot_after.st_mtime_ns
    out["meta"]["script_sha256"] = sha256_file(Path(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} blocks={blocks} wal={wal_before}/{wal_after}")
    for r in block_rows:
        print(
            f"block={r['block']} {r['class']}: n={r['n']} "
            f"up_rate={r['up_rate']:.3f} auc_ud={r['auc_up_down']} auc_ur={r['auc_up_rest']}"
        )


if __name__ == "__main__":
    main()
