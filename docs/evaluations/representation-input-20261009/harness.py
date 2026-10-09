"""Isolated primary-input comparison harness (artifact-local, typed).

Subcommands (all reads stay off the live DB; writes stay in --out-dir
plus a task-owned snapshot copy):

- manifest: read-only input audit on the frozen cohort. Records
  source/input hashes per story, matched-vs-mismatch counts, and
  config/encoder/tokenizer/script hashes. No vectors, no fits.
- validate: reproduce old oof probabilities within 1e-6 with the real
  _score_and_rank on a task-owned snapshot copy, fixed eval_now clock,
  warm_store disabled, frozen embedder (no inference). Gate for any fit.
- compare: refused in this bounded step (needs baseline gate +
  challenger vectors).
- self-check: pure behavioral checks, no DB/model.

Only stdlib + numpy + sklearn (via pipeline) + repo modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any
from dataclasses import asdict, dataclass
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from database import Database, Story  # noqa: E402
from pipeline import Embedder, story_embedding_text  # noqa: E402
from pipeline.config import Config  # noqa: E402
from pipeline.ranking import _score_and_rank, compose_story_text  # noqa: E402
from pipeline import warm_store  # noqa: E402

USER_ID = 151
LABEL = {"down": 0, "neutral": 1, "up": 2}
PROB_TOL = 1e-6
N_BOOT = 2000
BOOT_SEED = 20261009
TOP_K = 12

CHALLENGER_VERSION_SUFFIX = "|nocomments"

TASK_ROOT = "/tmp/opencode/representation-input-20261009"
LIVE_DB_BASENAME = "hn_rewrite.db"
PARITY_PER_BLOCK = 4
PARITY_COS_MIN = 0.9999
ENCODE_CHUNK = 16
# Single-row batches bound peak memory on the shared VPS. Parity was checked
# at this batch size; wider-batch speed and numerical behavior were not tested.
ENCODE_BATCH = 1


def _challenger_version(prod_version: str) -> str:
    return prod_version + CHALLENGER_VERSION_SUFFIX


def _is_task_path(path: str) -> bool:
    return os.path.normpath(path).startswith(TASK_ROOT + os.sep)


def _guard_task_paths(snapshot_src: str, work_snapshot: str, out_dir: str) -> None:
    for p in (snapshot_src, work_snapshot, out_dir):
        if LIVE_DB_BASENAME in p:
            raise RuntimeError(f"live DB path refused: {p}")
    if os.path.normpath(work_snapshot) == os.path.normpath(snapshot_src):
        raise RuntimeError("work snapshot must be a copy, never the frozen source")
    for p in (work_snapshot, out_dir):
        if not _is_task_path(p):
            raise RuntimeError(f"path outside task dir refused: {p}")


def _guard_task_reads(snapshot: str, out_dir: str) -> None:
    """Read-only commands: frozen source may live outside the task dir, but
    outputs must stay inside it and the live DB is never touched."""
    for p in (snapshot, out_dir):
        if LIVE_DB_BASENAME in p:
            raise RuntimeError(f"live DB path refused: {p}")
    if not _is_task_path(out_dir):
        raise RuntimeError(f"path outside task dir refused: {out_dir}")


def _pooled_mean(vals: list[float], weights: list[float]) -> float:
    num = sum(v * w for v, w in zip(vals, weights) if not math.isnan(v) and w > 0)
    den = sum(w for v, w in zip(vals, weights) if not math.isnan(v) and w > 0)
    return num / den if den else float("nan")


def _parity_ids(entries: list[StoryInputHashes], per_block: int) -> list[int]:
    """Deterministic full-text parity ids: sorted matched ids per block, strided.

    IDs only (plus the matched input-policy flag, never vote outcomes).
    """
    by_block: dict[str, list[int]] = {}
    for e in entries:
        if e.matched:
            by_block.setdefault(e.block, []).append(e.story_id)
    picks: list[int] = []
    for b in ("block1", "block2", "block3", "block4", "block5"):
        ids = sorted(by_block.get(b, []))
        if len(ids) < per_block:
            raise RuntimeError(f"block {b}: only {len(ids)} matched rows")
        stride = max(1, len(ids) // per_block)
        picks.extend(ids[min(i * stride, len(ids) - 1)] for i in range(per_block))
    return picks


def _challenger_choice(matched: bool, needs_encoding: bool) -> str:
    """Input policy: only matched-and-changed rows are re-encoded; else reuse."""
    return "encode" if (matched and needs_encoding) else "reuse"


def _parse_manifest_entries(manifest: dict[str, Any]) -> list[StoryInputHashes]:
    raw: list[Any] = manifest["entries"]
    entries: list[StoryInputHashes] = []
    for e in raw:
        d: dict[str, Any] = e
        entries.append(
            StoryInputHashes(
                story_id=int(d["story_id"]),
                source=str(d["source"]),
                matched=bool(d["matched"]),
                source_hash=str(d["source_hash"]),
                nocom_hash=str(d["nocom_hash"]),
                clean_body_empty=bool(d["clean_body_empty"]),
                block=str(d["block"]),
                effective_input_hash=str(d["effective_input_hash"]),
                needs_encoding=bool(d["needs_encoding"]),
            )
        )
    return entries


def _parse_scored_rows(raw: list[Any]) -> list[ScoredRow]:
    rows: list[ScoredRow] = []
    for r in raw:
        d: dict[str, Any] = r
        rows.append(
            ScoredRow(
                story_id=int(d["story_id"]),
                block=int(d["block"]),
                true_label=int(d["true_label"]),
                vote_time=float(d["vote_time"]),
                prob_up=float(d["prob_up"]),
                prob_neutral=float(d["prob_neutral"]),
                prob_down=float(d["prob_down"]),
                production_score=float(d["production_score"]),
            )
        )
    return rows


def _write_challenger_npz(path: Path, acc: dict[str, NDArray[np.float32]]) -> None:
    order = sorted(acc, key=int)
    ids = np.array([int(s) for s in order], dtype=np.int64)
    vecs = np.stack([np.asarray(acc[s], dtype=np.float32) for s in order])
    np.savez(str(path), ids=ids, vecs=vecs)


def _read_challenger_npz(path: str) -> dict[int, NDArray[np.float32]]:
    with np.load(path) as data:
        ids = [int(v) for v in data["ids"].tolist()]
        vecs = data["vecs"]
        return {
            sid: np.asarray(vec, dtype=np.float32).reshape(384)
            for sid, vec in zip(ids, vecs)
        }


def _single_thread_embedder(config: Config) -> Embedder:
    """Production Embedder with the ONNX session rebuilt at 1 intra-op thread.

    Tokenizer, mean-pool, L2, truncation path untouched (encode() reused
    verbatim); only the session thread count differs from the production
    default of 2. No pipeline change.
    """
    import onnxruntime as ort  # noqa: PLC0415 - heavy dep, encode path only

    emb = Embedder(
        config.onnx_model_dir,
        model_version=config.embedding_model_version,
        max_tokens=config.embedding_max_tokens,
        batch_size=ENCODE_BATCH,
    )
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    emb.session = ort.InferenceSession(
        str(Path(config.onnx_model_dir) / "model.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return emb


def _frozen_frame(
    snapshot: str, ids: list[int]
) -> dict[int, tuple[str, str, str, str, str]]:
    """Read-only story frame: id -> (title, self, comments, article, stored)."""
    if _wal_bytes(snapshot) != 0:
        raise RuntimeError("WAL not empty; refusing to read frozen snapshot")
    uri = f"file:{snapshot}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT id, title, self_text, top_comments,"
            " article_body, text_content FROM stories WHERE id IN (%s)"
            % ",".join("?" * len(ids)),
            ids,
        ).fetchall()
    finally:
        con.close()
    frame = {
        r[0]: (r[1] or "", r[2] or "", r[3] or "", r[4] or "", r[5] or "") for r in rows
    }
    missing = [i for i in ids if i not in frame]
    if missing:
        raise RuntimeError(f"{len(missing)} parity/encode ids missing from snapshot")
    return frame


def _stored_vectors(
    snapshot: str, model_version: str, hashes: dict[int, str]
) -> dict[int, NDArray[np.float32]]:
    uri = f"file:{snapshot}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        ids = list(hashes)
        placeholders = ",".join("?" * len(ids))
        found: dict[int, NDArray[np.float32]] = {}
        for row in con.execute(
            "SELECT story_id, text_hash, embedding FROM embeddings"
            f" WHERE model_version = ? AND story_id IN ({placeholders})",
            (model_version, *ids),
        ).fetchall():
            sid, h, blob = int(row[0]), str(row[1]), row[2]
            if hashes.get(sid) == h:
                vec = Database._decode_embedding_blob(
                    blob, expected_dim=384, story_id=sid
                )
                if vec is not None:
                    found[sid] = vec
    finally:
        con.close()
    return found


def _freeze_record(
    sample_path: str, config_path: str, extra: dict[str, str]
) -> dict[str, str]:
    return {
        "sample.json": sha16_file(sample_path),
        "config.toml": sha16_file(config_path),
        "harness.py": sha16_file(__file__),
        **extra,
    }


def sha16_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _FrozenEmbedder(Embedder):
    """Cached-vectors-only embedder: any uncaptured encode is a hard error."""

    def __init__(self, model_version: str, embedding_dim: int = 384) -> None:
        self.model_version = model_version
        self.embedding_dim = embedding_dim

    def encode(
        self, texts: list[str], batch_size: int | None = None
    ) -> NDArray[np.float32]:
        raise RuntimeError("validate attempted to compute an uncaptured embedding")


def _text_hashes(stories: list[Story]) -> dict[int, str]:
    return {s.id: sha256_str(story_embedding_text(s)) for s in stories}


def _valid_vectors(
    stories: list[Story], cached: dict[int, NDArray[np.float32]]
) -> tuple[NDArray[np.float32], list[int], int]:
    dims = {v.shape for v in cached.values()}
    dim = dims.pop()[0] if len(dims) == 1 else 384
    bad = [
        s.id
        for s in stories
        if s.id not in cached
        or cached[s.id].shape != (dim,)
        or not np.isfinite(cached[s.id]).all()
        or not np.isclose(np.linalg.norm(cached[s.id]), 1.0, atol=1e-3)
    ]
    mat = np.asarray(
        [cached[s.id] for s in stories if s.id not in set(bad)],
        dtype=np.float32,
    ).reshape(-1, dim)
    return mat, sorted(set(bad)), dim


def ordinal_auc(y: list[int], s: list[float]) -> float:
    num = den = 0.0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                continue
            den += 1
            want = (y[i] > y[j]) - (y[i] < y[j])
            got = (s[i] > s[j]) - (s[i] < s[j])
            num += 1.0 if got == want else (0.5 if got == 0 else 0.0)
    return num / den if den else float("nan")


def _pair_count(y: list[int]) -> int:
    counts = Counter(y)
    return (len(y) ** 2 - sum(n * n for n in counts.values())) // 2


def uprest_auc(y: list[int], s: list[float]) -> float:
    """Primary metric: UP-vs-rest AUC under the same pair-counting rule."""
    binary = [int(v == 2) for v in y]
    return float(roc_auc_score(binary, s)) if len(set(binary)) == 2 else float("nan")


def uprest_pair_count(y: list[int]) -> int:
    return _pair_count([1 if v == 2 else 0 for v in y])


def top_k_up_fraction(y: list[int], s: list[float], k: int = TOP_K) -> float:
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]
    return sum(1 for i in order if y[i] == 2) / k if k else float("nan")


def top_k_down_fraction(y: list[int], s: list[float], k: int = TOP_K) -> float:
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]
    return sum(1 for i in order if y[i] == 0) / k if k else float("nan")


def paired_story_bootstrap_lo(
    blocks_y: list[list[int]],
    blocks_sa: list[list[float]],
    blocks_sb: list[list[float]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Paired story bootstrap lower bound for pooled UP-vs-rest delta.

    One index draw per block per rep; the same draw feeds both arms
    (paired within rep); blocks pooled by UP/non-UP pair count.
    Conditional on fits (no refit).
    """
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        num = den = 0.0
        for y, sa, sb in zip(blocks_y, blocks_sa, blocks_sb):
            n = len(y)
            idx = rng.integers(0, n, n).tolist()
            ys = [y[i] for i in idx]
            da = uprest_auc(ys, [sa[i] for i in idx])
            db = uprest_auc(ys, [sb[i] for i in idx])
            pc = uprest_pair_count(ys)
            if pc:
                num += (db - da) * pc
                den += pc
        if den:
            deltas.append(num / den)
    if not deltas:
        return {
            "mean": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n_boot": float(n_boot),
        }
    vals = np.asarray(deltas, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
        "n_boot": float(n_boot),
    }


@dataclass(frozen=True)
class ScoredRow:
    story_id: int
    block: int
    true_label: int
    vote_time: float
    prob_up: float
    prob_neutral: float
    prob_down: float
    production_score: float


@dataclass(frozen=True)
class StoryInputHashes:
    story_id: int
    source: str
    matched: bool
    source_hash: str
    nocom_hash: str
    clean_body_empty: bool
    block: str
    effective_input_hash: str
    needs_encoding: bool


def challenger_text(title: str, self_text: str, article_body: str) -> str:
    return compose_story_text(title, self_text, "", article_body)


def _wal_bytes(snapshot: str) -> int:
    try:
        return Path(snapshot + "-wal").stat().st_size
    except FileNotFoundError:
        return 0


def _resource_snapshot() -> dict[str, str]:
    out: dict[str, str] = {}
    for cmd in (["uptime"], ["free", "-m"]):
        try:
            out[" ".join(cmd)] = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            ).stdout.strip()[:400]
        except Exception as e:  # noqa: BLE001 - diagnostic only
            out[" ".join(cmd)] = f"unavailable: {e}"
    return out


def _guard_runtime_headroom() -> None:
    """Stop safely at a checkpoint rather than compete with the live service."""
    available_kb: int | None = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kb = int(line.split()[1])
            break
    load = os.getloadavg()[0]
    if available_kb is None or available_kb < 2 * 1024 * 1024 or load > 4:
        raise RuntimeError(
            f"Resource pause: load={load:.2f}, available_kb={available_kb}; "
            "resume the task-owned checkpoint after headroom returns"
        )


def _encoder_hashes(model_dir: str) -> dict[str, str]:
    out: dict[str, str] = {"model_dir": model_dir}
    for name in ("model.onnx", "config.json", "tokenizer.json", "vocab.txt"):
        p = Path(model_dir) / name
        out[name] = sha16_file(p) if p.exists() else "missing"
    return out


def run_manifest(
    snapshot: str, sample_path: str, config_path: str, out: str | None
) -> int:
    from pipeline.ranking import clean_text  # noqa: PLC0415 - local, stable surface

    wal = _wal_bytes(snapshot)
    if wal != 0:
        raise RuntimeError(f"WAL not empty ({wal} bytes); refusing to read")
    sample = json.loads(Path(sample_path).read_text())
    old_ids = [s for b in range(1, 6) for s in sample["blocks"][f"block{b}"]]
    uri = f"file:{snapshot}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT id, title, source, self_text, top_comments,"
            " article_body, text_content FROM stories WHERE id IN (%s)"
            % ",".join("?" * len(old_ids)),
            old_ids,
        ).fetchall()
    finally:
        con.close()
    by_id = {r[0]: r for r in rows}
    missing = [s for s in old_ids if s not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} frozen stories missing from snapshot")
    entries: list[StoryInputHashes] = []
    block_by_id = {
        int(sid): str(block) for block, ids in sample["blocks"].items() for sid in ids
    }
    for sid in old_ids:
        _, title, source, stx, tc, ab, stored = by_id[sid]
        title_s, self_s, comm_s, art_s = (
            title or "",
            stx or "",
            tc or "",
            ab or "",
        )
        full = compose_story_text(title_s, self_s, comm_s, art_s)
        matched = (stored or "") == full
        nocom = challenger_text(title_s, self_s, art_s)
        entries.append(
            StoryInputHashes(
                story_id=int(sid),
                source=str(source or ""),
                matched=matched,
                source_hash=sha256_str(stored or ""),
                nocom_hash=sha256_str(nocom),
                clean_body_empty=not clean_text(self_s) and not clean_text(art_s),
                block=block_by_id[int(sid)],
                effective_input_hash=sha256_str(nocom if matched else (stored or "")),
                needs_encoding=matched and nocom != (stored or ""),
            )
        )
    matched_n = sum(1 for e in entries if e.matched)
    config = Config.load(config_path)
    payload = {
        "snapshot": snapshot,
        "n_old": len(old_ids),
        "matched": matched_n,
        "mismatched": len(old_ids) - matched_n,
        "clean_body_empty": sum(1 for e in entries if e.clean_body_empty),
        "changed_inputs_if_encoded": sum(e.needs_encoding for e in entries),
        "mismatches_by_block": dict(Counter(e.block for e in entries if not e.matched)),
        "entries": [asdict(e) for e in entries],
        "config": {
            "embedding_model_version": config.embedding_model_version,
            "embedding_max_tokens": config.embedding_max_tokens,
            "svm_c": config.model.svm_c,
            "svm_kernel": config.model.svm_kernel,
            "svm_gamma": config.model.svm_gamma,
            "linear_blend_enabled": config.model.linear_blend_enabled,
            "side_embedding_enabled": config.model.side_embedding_enabled,
            "classifier": config.model.classifier,
        },
        "encoder": _encoder_hashes(config.onnx_model_dir),
        "hashes": {
            "sample.json": sha16_file(sample_path),
            "config.toml": sha16_file(config_path),
            "harness.py": sha16_file(__file__),
        },
    }
    text = json.dumps(payload, indent=1, sort_keys=True)
    if out:
        Path(out).write_text(text + "\n")
    else:
        print(text)
    print(
        f"manifest: n={len(old_ids)} matched={matched_n} "
        f"mismatched={len(old_ids) - matched_n} "
        f"clean_body_empty={payload['clean_body_empty']}",
        flush=True,
    )
    return 0


def _load_cohort(
    conn: sqlite3.Connection, ids: list[int]
) -> tuple[list[Story], list[int], list[float]]:
    idlist = list(ids)
    m = {
        r[0]: r
        for r in conn.execute(
            "SELECT f.story_id, f.action, f.updated_at, s.title, s.url,"
            " s.text_content, s.source, s.score, s.time, s.comment_count,"
            " s.self_text, s.top_comments, s.article_body FROM feedback f"
            " LEFT JOIN stories s ON s.id=f.story_id"
            " WHERE f.user_id=? AND f.story_id IN (%s)" % ",".join("?" * len(idlist)),
            (USER_ID, *idlist),
        )
    }
    assert len(m) == len(idlist), "feedback/story row missing"
    stories: list[Story] = []
    for sid in idlist:
        (_, _a, _t, title, url, text, src, score, st, cc, stx, tc, ab) = m[sid]
        stories.append(
            Story(
                id=sid,
                title=title or "",
                url=url,
                score=score or 0,
                time=st or 0,
                text_content=text or "",
                source=src or "hn",
                comment_count=cc,
                self_text=stx or "",
                top_comments=tc or "",
                article_body=ab or "",
            )
        )
    return (
        stories,
        [LABEL[m[s][1]] for s in idlist],
        [float(m[s][2]) for s in idlist],
    )


def run_validate(
    snapshot_src: str,
    work_snapshot: str,
    sample_path: str,
    oof_scores_path: str,
    config_path: str,
    out_dir: str,
) -> int:
    t0 = time.time()
    env = _resource_snapshot()
    print(f"env before copy: {json.dumps(env)[:500]}", flush=True)
    if warm_store.enabled():
        raise RuntimeError("warm_store enabled; refusing (must be disabled)")
    sample = json.loads(Path(sample_path).read_text())
    old_oof = json.loads(Path(oof_scores_path).read_text())
    eval_now = float(sample["eval_now"])
    blocks: dict[int, list[int]] = {
        b: list(sample["blocks"][f"block{b}"]) for b in range(1, 6)
    }

    work = Path(work_snapshot)
    work.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-shm", "-wal"):
        p = Path(str(work) + suffix)
        if p.exists():
            p.unlink()
    shutil.copyfile(snapshot_src, work)

    import pipeline.ranking as ranking  # noqa: PLC0415 - patched clock below
    from unittest import mock  # noqa: PLC0415 - local, stable surface

    config = Config.load(config_path)
    real_time = time.time
    rows: list[ScoredRow] = []
    drift: dict[str, list[int]] = {}
    fit_secs: dict[str, float] = {}
    with mock.patch.object(ranking.time, "time", return_value=eval_now):
        db = Database(str(work), read_only=False)
        mv = config.embedding_model_version
        with db.conn() as conn:
            for k in (2, 3, 4, 5):
                train_ids = [s for b in range(1, k) for s in blocks[b]]
                test_ids = blocks[k]
                tr_stories, tr_y, tr_t = _load_cohort(conn, train_ids)
                te_stories, te_y, te_t = _load_cohort(conn, test_ids)
                tr_cached = db.get_embeddings_batch(
                    [s.id for s in tr_stories], mv, _text_hashes(tr_stories)
                )
                tr_mat, tr_bad, _ = _valid_vectors(tr_stories, tr_cached)
                te_cached = db.get_embeddings_batch(
                    [s.id for s in te_stories], mv, _text_hashes(te_stories)
                )
                te_mat, te_bad, _ = _valid_vectors(te_stories, te_cached)
                drift[f"train_bad_b{k}"] = tr_bad
                drift[f"test_bad_b{k}"] = te_bad
                keep = [i for i, s in enumerate(tr_stories) if s.id not in set(tr_bad)]
                fb: tuple[list[Story], list[int], list[float]] = (
                    [tr_stories[i] for i in keep],
                    [int(tr_y[i]) for i in keep],
                    [float(tr_t[i]) for i in keep],
                )
                scored = [
                    i for i, s in enumerate(te_stories) if s.id not in set(te_bad)
                ]
                cands = [te_stories[i] for i in scored]
                emb = _FrozenEmbedder(mv, tr_mat.shape[1] if keep else te_mat.shape[1])
                f0 = real_time()
                ranked = _score_and_rank(
                    cands,
                    te_mat,
                    db,
                    config,
                    emb,
                    user_id=USER_ID,
                    training_feedback=fb,
                )
                fit_secs[f"block{k}"] = round(real_time() - f0, 1)
                by_id = {r.story.id: r for r in ranked}
                for i, s in enumerate(cands):
                    r = by_id[s.id]
                    assert (
                        r.prob_up is not None
                        and r.prob_neutral is not None
                        and r.prob_down is not None
                    ), f"missing probs for {s.id}"
                    rows.append(
                        ScoredRow(
                            story_id=s.id,
                            block=k,
                            true_label=int(te_y[scored[i]]),
                            vote_time=float(te_t[scored[i]]),
                            prob_up=float(r.prob_up),
                            prob_neutral=float(r.prob_neutral),
                            prob_down=float(r.prob_down),
                            production_score=float(r.score),
                        )
                    )
        db.close()

    old_by_id = {(r["story_id"], r["block"]): r for r in old_oof["blocks"]}
    assert len(old_by_id) == len(rows), (len(old_by_id), len(rows))
    max_abs = 0.0
    worst: tuple[int, int] | None = None
    score_drift = 0.0
    for r in rows:
        key = (r.story_id, r.block)
        old = old_by_id.get(key)
        assert old is not None, f"old oof row missing for {key}"
        for new_prob, name in (
            (r.prob_up, "prob_up"),
            (r.prob_neutral, "prob_neutral"),
            (r.prob_down, "prob_down"),
        ):
            d = abs(new_prob - float(old[name]))
            if d > max_abs:
                max_abs, worst = d, (r.story_id, r.block)
        score_drift = max(
            score_drift,
            abs(r.production_score - float(old["production_score"])),
        )
    passed = max_abs <= PROB_TOL
    report = {
        "gate": f"max|prob diff| <= {PROB_TOL}",
        "passed": passed,
        "max_abs_prob_diff": max_abs,
        "worst_row": worst,
        "n_rows": len(rows),
        "drift": drift,
        "fit_seconds": fit_secs,
        "total_seconds": round(real_time() - t0, 1),
        "score_drift_max_abs_vs_wallclock": score_drift,
        "score_drift_note": "expected: fixed eval_now clock vs old wall-clock"
        " scores; not gated.",
        "eval_now": eval_now,
        "warm_store_enabled": False,
        "config": {
            "svm_c": config.model.svm_c,
            "svm_kernel": config.model.svm_kernel,
            "svm_gamma": config.model.svm_gamma,
            "linear_blend_enabled": config.model.linear_blend_enabled,
            "side_embedding_enabled": config.model.side_embedding_enabled,
            "classifier": config.model.classifier,
            "embedding_model_version": mv,
        },
        "encoder": _encoder_hashes(config.onnx_model_dir),
        "hashes": {
            "sample.json": sha16_file(sample_path),
            "oof_scores.json": sha16_file(oof_scores_path),
            "config.toml": sha16_file(config_path),
            "harness.py": sha16_file(__file__),
        },
        "env_before_copy": env,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n"
    )
    (out / "oof_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=1, sort_keys=True)[:3000], flush=True)
    if not passed:
        print("BASELINE GATE FAILED: stopping before any model inference.", flush=True)
        return 1
    print("BASELINE GATE PASSED.", flush=True)
    return 0


def run_parity(
    snapshot: str,
    sample_path: str,
    manifest_path: str,
    config_path: str,
    out_dir: str,
    per_block: int = PARITY_PER_BLOCK,
) -> int:
    t0 = time.time()
    _guard_task_reads(snapshot, out_dir)
    env = _resource_snapshot()
    print(f"env before parity: {json.dumps(env)[:500]}", flush=True)
    _guard_runtime_headroom()
    if warm_store.enabled():
        raise RuntimeError("warm_store enabled; refusing (must be disabled)")
    manifest = json.loads(Path(manifest_path).read_text())
    entries = _parse_manifest_entries(manifest)
    picks = _parity_ids(entries, per_block)
    by_id = {e.story_id: e for e in entries}
    frame = _frozen_frame(snapshot, picks)
    config = Config.load(config_path)
    texts: list[str] = []
    expect_hash: dict[int, str] = {}
    for sid in picks:
        title, self_s, comm_s, art_s, stored = frame[sid]
        story = Story(
            id=sid,
            title=title,
            url=None,
            score=0,
            time=0,
            text_content=stored,
            source="hn",
            comment_count=None,
            self_text=self_s,
            top_comments=comm_s,
            article_body=art_s,
        )
        full = story_embedding_text(story)
        if sha256_str(full) != by_id[sid].source_hash:
            raise RuntimeError(f"source hash mismatch for {sid}; refusing")
        texts.append(full)
        expect_hash[sid] = sha256_str(full)
    stored_vecs = _stored_vectors(snapshot, config.embedding_model_version, expect_hash)
    missing = [sid for sid in picks if sid not in stored_vecs]
    if missing:
        raise RuntimeError(f"{len(missing)} stored vectors missing; refusing")
    emb = _single_thread_embedder(config)
    f0 = time.time()
    fresh = emb.encode(texts)
    encode_secs = time.time() - f0
    rows = []
    min_cos = 1.0
    for sid, vec in zip(picks, fresh):
        cos = float(np.dot(vec, stored_vecs[sid]))
        min_cos = min(min_cos, cos)
        rows.append({"story_id": sid, "cosine": cos})
    n_changed = int(manifest["changed_inputs_if_encoded"])
    per_row = encode_secs / len(picks)
    report = {
        "gate": f"cosine >= {PARITY_COS_MIN} on every parity row",
        "passed": all(r["cosine"] >= PARITY_COS_MIN for r in rows),
        "n_rows": len(picks),
        "min_cosine": min_cos,
        "rows": rows,
        "encode_seconds": round(encode_secs, 2),
        "seconds_per_row": round(per_row, 3),
        "projected_seconds_for_changed": round(per_row * n_changed, 1),
        "projected_note": "linear projection for Penn 3237 changed rows; first metrics only",
        "total_seconds": round(time.time() - t0, 1),
        "warm_store_enabled": False,
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "onnx_intra_op_threads": 1,
        "config": {
            "embedding_model_version": config.embedding_model_version,
            "embedding_max_tokens": config.embedding_max_tokens,
        },
        "encoder": _encoder_hashes(config.onnx_model_dir),
        "hashes": _freeze_record(
            sample_path, config_path, {"manifest.json": sha16_file(manifest_path)}
        ),
        "env_before": env,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=1, sort_keys=True)[:2500], flush=True)
    if not report["passed"]:
        print("PARITY GATE FAILED: stopping before any body encoding.", flush=True)
        return 1
    print("PARITY GATE PASSED.", flush=True)
    return 0


def run_encode(
    snapshot: str,
    sample_path: str,
    manifest_path: str,
    config_path: str,
    out_dir: str,
) -> int:
    t0 = time.time()
    _guard_task_reads(snapshot, out_dir)
    env = _resource_snapshot()
    print(f"env before encode: {json.dumps(env)[:500]}", flush=True)
    _guard_runtime_headroom()
    if warm_store.enabled():
        raise RuntimeError("warm_store enabled; refusing (must be disabled)")
    manifest = json.loads(Path(manifest_path).read_text())
    entries = _parse_manifest_entries(manifest)
    todo = [
        e.story_id
        for e in entries
        if _challenger_choice(e.matched, e.needs_encoding) == "encode"
    ]
    by_id = {e.story_id: e for e in entries}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "challenger_done.json"
    npz_path = out / "challenger_vectors.npz"
    done: dict[str, str] = (
        json.loads(done_path.read_text()) if done_path.exists() else {}
    )
    acc: dict[str, NDArray[np.float32]] = {}
    if npz_path.exists():
        with np.load(str(npz_path)) as data:
            ids_arr = [int(v) for v in data["ids"].tolist()]
            vecs_arr = data["vecs"]
            acc = {
                str(sid): np.asarray(vec, dtype=np.float32)
                for sid, vec in zip(ids_arr, vecs_arr)
            }
    for sid_str, h in done.items():
        if sid_str not in acc:
            raise RuntimeError(f"checkpoint row {sid_str} lacks vector; refusing")
        if h != by_id[int(sid_str)].effective_input_hash:
            raise RuntimeError(f"checkpoint input drift for {sid_str}; refusing")
    pending = [s for s in todo if str(s) not in done]
    print(
        f"encode: todo={len(todo)} resumed={len(done)} pending={len(pending)}",
        flush=True,
    )
    config = Config.load(config_path)
    if pending:
        frame = _frozen_frame(snapshot, pending)
        texts: dict[int, str] = {}
        for sid in pending:
            title, self_s, _comm_s, art_s, _stored = frame[sid]
            nocom = challenger_text(title, self_s, art_s)
            if sha256_str(nocom) != by_id[sid].nocom_hash:
                raise RuntimeError(f"nocom hash mismatch for {sid}; refusing")
            texts[sid] = nocom
        emb = _single_thread_embedder(config)
        order = list(pending)
        for c0 in range(0, len(order), ENCODE_CHUNK):
            _guard_runtime_headroom()
            chunk = order[c0 : c0 + ENCODE_CHUNK]
            vecs = emb.encode([texts[s] for s in chunk])
            for sid, vec in zip(chunk, vecs):
                n = float(np.linalg.norm(vec))
                if not np.isfinite(vec).all() or not math.isclose(n, 1.0, abs_tol=1e-3):
                    raise RuntimeError(f"bad vector for {sid} (norm {n}); refusing")
                if vec.shape != (384,):
                    raise RuntimeError(f"bad dim for {sid}: {vec.shape}; refusing")
                acc[str(sid)] = vec.astype(np.float32)
                done[str(sid)] = by_id[sid].effective_input_hash
            _write_challenger_npz(npz_path, acc)
            done_path.write_text(json.dumps(done, indent=1, sort_keys=True) + "\n")
            print(f"encode: checkpoint {len(done)}/{len(todo)}", flush=True)
    entry_hashes = {
        str(sid): {
            "input_hash": done[str(sid)],
            "vector_sha16": hashlib.sha256(
                np.ascontiguousarray(acc[str(sid)], dtype=np.float32).tobytes()
            ).hexdigest()[:16],
        }
        for sid in todo
    }
    report = {
        "n_changed": len(todo),
        "n_vectors": len(acc),
        "complete": len(acc) == len(todo),
        "total_seconds": round(time.time() - t0, 1),
        "warm_store_enabled": False,
        "encoder": _encoder_hashes(config.onnx_model_dir),
        "hashes": _freeze_record(
            sample_path, config_path, {"manifest.json": sha16_file(manifest_path)}
        ),
        "env_before": env,
        "rows": entry_hashes,
    }
    (out / "encode_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"})[:2000], flush=True
    )
    if not report["complete"]:
        print("ENCODE INCOMPLETE: rerun to resume from checkpoint.", flush=True)
        return 1
    print("ENCODE COMPLETE.", flush=True)
    return 0


def run_compare(
    arm: str,
    snapshot_src: str,
    work_snapshot: str,
    sample_path: str,
    manifest_path: str | None,
    challenger_npz: str | None,
    oof_scores_path: str | None,
    config_path: str,
    out_dir: str,
) -> int:
    t0 = time.time()
    _guard_task_paths(snapshot_src, work_snapshot, out_dir)
    env = _resource_snapshot()
    print(f"env before copy ({arm}): {json.dumps(env)[:500]}", flush=True)
    if warm_store.enabled():
        raise RuntimeError("warm_store enabled; refusing (must be disabled)")
    if arm not in ("baseline", "challenger"):
        raise RuntimeError(f"unknown arm: {arm}")
    sample = json.loads(Path(sample_path).read_text())
    eval_now = float(sample["eval_now"])
    blocks: dict[int, list[int]] = {
        b: list(sample["blocks"][f"block{b}"]) for b in range(1, 6)
    }
    work = Path(work_snapshot)
    work.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-shm", "-wal"):
        p = Path(str(work) + suffix)
        if p.exists():
            p.unlink()
    shutil.copyfile(snapshot_src, work)

    import pipeline.ranking as ranking  # noqa: PLC0415 - patched clock below
    from unittest import mock  # noqa: PLC0415 - local, stable surface

    config = Config.load(config_path)
    mv = config.embedding_model_version
    cv = _challenger_version(mv)

    db = Database(str(work), read_only=False)
    try:
        with db.conn() as conn:
            cohort = [s for b in range(1, 6) for s in blocks[b]]
            stories_by_id = {s.id: s for s in _load_cohort(conn, cohort)[0]}
            if arm == "baseline":
                emb: Embedder = _FrozenEmbedder(mv, 384)
                cand_vecs: dict[int, NDArray[np.float32]] | None = None
            else:
                if not manifest_path or not challenger_npz:
                    raise RuntimeError("challenger arm needs manifest + npz")
                manifest = json.loads(Path(manifest_path).read_text())
                by_id = {e.story_id: e for e in _parse_manifest_entries(manifest)}
                npz_vecs = _read_challenger_npz(challenger_npz)
                stored_hash = _text_hashes(list(stories_by_id.values()))
                for sid, h in stored_hash.items():
                    if h != by_id[sid].source_hash:
                        raise RuntimeError(
                            f"source drift for {sid}; refusing challenger upsert"
                        )
                prod_cached = db.get_embeddings_batch(cohort, mv, stored_hash)
                if len(prod_cached) != len(cohort):
                    raise RuntimeError("challenger arm: stored vectors missing")
                chall_map: dict[int, NDArray[np.float32]] = {}
                for sid in cohort:
                    e = by_id[sid]
                    if _challenger_choice(e.matched, e.needs_encoding) == "encode":
                        if sid not in npz_vecs:
                            raise RuntimeError(f"challenger vector missing: {sid}")
                        vec = npz_vecs[sid]
                        db.upsert_embedding(sid, cv, stored_hash[sid], vec)
                        chall_map[sid] = vec
                    else:
                        vec = prod_cached[sid]
                        db.upsert_embedding(sid, cv, stored_hash[sid], vec)
                        chall_map[sid] = vec
                roundtrip = db.get_embeddings_batch(cohort, cv, stored_hash)
                if len(roundtrip) != len(cohort):
                    raise RuntimeError("challenger round-trip failed; refusing")
                emb = _FrozenEmbedder(cv, 384)
                cand_vecs = chall_map
    finally:
        pass

    real_time = time.time
    rows: list[ScoredRow] = []
    fit_secs: dict[str, float] = {}
    with mock.patch.object(ranking.time, "time", return_value=eval_now):
        with db.conn() as conn:
            for k in (2, 3, 4, 5):
                train_ids = [s for b in range(1, k) for s in blocks[b]]
                test_ids = blocks[k]
                tr_stories, tr_y, tr_t = _load_cohort(conn, train_ids)
                te_stories, te_y, te_t = _load_cohort(conn, test_ids)
                if arm == "baseline":
                    tr_cached = db.get_embeddings_batch(
                        [s.id for s in tr_stories], mv, _text_hashes(tr_stories)
                    )
                    _, tr_bad, tr_dim = _valid_vectors(tr_stories, tr_cached)
                    te_cached = db.get_embeddings_batch(
                        [s.id for s in te_stories], mv, _text_hashes(te_stories)
                    )
                    te_mat, te_bad, te_dim = _valid_vectors(te_stories, te_cached)
                    if tr_dim != 384 or te_dim != 384:
                        raise RuntimeError("unexpected embedding dim; refusing")
                else:
                    assert cand_vecs is not None
                    te_mat = np.asarray(
                        [cand_vecs[s.id] for s in te_stories], dtype=np.float32
                    )
                    tr_bad, te_bad = [], []
                if tr_bad or te_bad:
                    raise RuntimeError(f"bad vectors in {arm} block{k}; refusing")
                keep = list(range(len(tr_stories)))
                fb: tuple[list[Story], list[int], list[float]] = (
                    [tr_stories[i] for i in keep],
                    [int(tr_y[i]) for i in keep],
                    [float(tr_t[i]) for i in keep],
                )
                f0 = real_time()
                ranked = _score_and_rank(
                    te_stories,
                    te_mat,
                    db,
                    config,
                    emb,
                    user_id=USER_ID,
                    training_feedback=fb,
                )
                fit_secs[f"block{k}"] = round(real_time() - f0, 1)
                by_rank = {r.story.id: r for r in ranked}
                for i, s in enumerate(te_stories):
                    r = by_rank[s.id]
                    assert (
                        r.prob_up is not None
                        and r.prob_neutral is not None
                        and r.prob_down is not None
                    ), f"missing probs for {s.id}"
                    rows.append(
                        ScoredRow(
                            story_id=s.id,
                            block=k,
                            true_label=int(te_y[i]),
                            vote_time=float(te_t[i]),
                            prob_up=float(r.prob_up),
                            prob_neutral=float(r.prob_neutral),
                            prob_down=float(r.prob_down),
                            production_score=float(r.score),
                        )
                    )
    db.close()
    ref_check: dict[str, object] = {}
    if oof_scores_path:
        old_oof = json.loads(Path(oof_scores_path).read_text())
        old_by_id = {(r["story_id"], r["block"]): r for r in old_oof["blocks"]}
        max_abs = 0.0
        for r in rows:
            old = old_by_id.get((r.story_id, r.block))
            if old is None:
                raise RuntimeError("frozen reference row missing; refusing")
            for name in ("prob_up", "prob_neutral", "prob_down"):
                max_abs = max(max_abs, abs(getattr(r, name) - float(old[name])))
        ref_check = {
            "oof_scores": sha16_file(oof_scores_path),
            "max_abs_prob_diff_vs_frozen": max_abs,
        }
        if not math.isfinite(max_abs) or max_abs > PROB_TOL:
            raise RuntimeError(
                f"Frozen baseline mismatch {max_abs}; stopping before report"
            )
    report = {
        "arm": arm,
        "model_version": cv if arm == "challenger" else mv,
        "n_rows": len(rows),
        "fit_seconds": fit_secs,
        "total_seconds": round(real_time() - t0, 1),
        "eval_now": eval_now,
        "warm_store_enabled": False,
        "config": {
            "embedding_model_version": mv,
            "classifier": config.model.classifier,
            "svm_c": config.model.svm_c,
            "svm_kernel": config.model.svm_kernel,
            "svm_gamma": config.model.svm_gamma,
            "linear_blend_enabled": config.model.linear_blend_enabled,
            "side_embedding_enabled": config.model.side_embedding_enabled,
        },
        "encoder": _encoder_hashes(config.onnx_model_dir),
        "hashes": _freeze_record(sample_path, config_path, {}),
        "env_before_copy": env,
        "frozen_reference": ref_check,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{arm}_rows.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=1, sort_keys=True) + "\n"
    )
    (out / f"{arm}_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=1, sort_keys=True)[:2500], flush=True)
    print(f"COMPARE {arm.upper()} DONE: {len(rows)} rows.", flush=True)
    return 0


def run_report(
    baseline_rows_path: str,
    challenger_rows_path: str,
    sample_path: str,
    out_path: str,
) -> int:
    base = _parse_scored_rows(json.loads(Path(baseline_rows_path).read_text()))
    chall = _parse_scored_rows(json.loads(Path(challenger_rows_path).read_text()))
    test_blocks = (2, 3, 4, 5)
    bb: dict[int, list[ScoredRow]] = {b: [] for b in test_blocks}
    cb: dict[int, list[ScoredRow]] = {b: [] for b in test_blocks}
    for r in base:
        if r.block in bb:
            bb[r.block].append(r)
    for r in chall:
        if r.block in cb:
            cb[r.block].append(r)
    for b in test_blocks:
        if sorted(r.story_id for r in bb[b]) != sorted(r.story_id for r in cb[b]):
            raise RuntimeError(f"block{b}: arm row sets differ; refusing")
    per_block: list[dict[str, float]] = []
    for b in test_blocks:
        order = sorted(range(len(bb[b])), key=lambda i: bb[b][i].story_id)
        y = [bb[b][i].true_label for i in order]
        sa = [bb[b][i].production_score for i in order]
        sb = [cb[b][i].production_score for i in order]
        aa, ab = uprest_auc(y, sa), uprest_auc(y, sb)
        pc = uprest_pair_count(y)
        per_block.append(
            {
                "block": b,
                "n": len(y),
                "up": sum(1 for v in y if v == 2),
                "pairs": pc,
                "auc_baseline": aa,
                "auc_challenger": ab,
                "delta": ab - aa,
                "top12_up_baseline": top_k_up_fraction(y, sa),
                "top12_up_challenger": top_k_up_fraction(y, sb),
                "top12_down_baseline": top_k_down_fraction(y, sa),
                "top12_down_challenger": top_k_down_fraction(y, sb),
                "ordinal_baseline": ordinal_auc(y, sa),
                "ordinal_challenger": ordinal_auc(y, sb),
            }
        )
    pooled_delta = _pooled_mean(
        [p["delta"] for p in per_block], [p["pairs"] for p in per_block]
    )
    pooled_auc = {
        arm: _pooled_mean(
            [p[f"auc_{arm}"] for p in per_block], [p["pairs"] for p in per_block]
        )
        for arm in ("baseline", "challenger")
    }
    top12_up = {
        arm: sum(p[f"top12_up_{arm}"] * TOP_K for p in per_block) / (TOP_K * 4)
        for arm in ("baseline", "challenger")
    }
    top12_down = {
        arm: sum(p[f"top12_down_{arm}"] * TOP_K for p in per_block) / (TOP_K * 4)
        for arm in ("baseline", "challenger")
    }
    boot = paired_story_bootstrap_lo(
        [
            [r.true_label for r in sorted(bb[b], key=lambda r: r.story_id)]
            for b in test_blocks
        ],
        [
            [r.production_score for r in sorted(bb[b], key=lambda r: r.story_id)]
            for b in test_blocks
        ],
        [
            [r.production_score for r in sorted(cb[b], key=lambda r: r.story_id)]
            for b in test_blocks
        ],
    )
    n_pos = sum(1 for p in per_block if p["delta"] > 0)
    gates = {
        "primary_delta_ge_0.01": pooled_delta >= 0.01,
        "positive_ge_3_of_4": n_pos >= 3,
        "bootstrap_lo_gt_0": boot["lo"] > 0,
        "guardrail_down_not_up": top12_down["challenger"] <= top12_down["baseline"],
        "guardrail_up_not_down": top12_up["challenger"] >= top12_up["baseline"],
    }
    report = {
        "endpoint": "UP-vs-rest AUC on production score, pooled by UP/non-UP pairs",
        "per_block": per_block,
        "pooled_auc": pooled_auc,
        "pooled_delta": pooled_delta,
        "n_positive_blocks": n_pos,
        "top12_up_pooled": top12_up,
        "top12_down_pooled": top12_down,
        "bootstrap": boot,
        "gates": gates,
        "passed": all(gates.values()),
        "hashes": {
            "baseline_rows": sha16_file(baseline_rows_path),
            "challenger_rows": sha16_file(challenger_rows_path),
            "sample.json": sha16_file(sample_path),
            "harness.py": sha16_file(__file__),
        },
    }
    Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True)[:3000], flush=True)
    print("GATES PASSED." if report["passed"] else "GATES FAILED.", flush=True)
    return 0 if report["passed"] else 1


def run_self_check() -> int:
    # uprest_auc: perfect separation -> 1.0; ties -> 0.5 parts.
    assert uprest_auc([2, 2, 0, 1], [0.9, 0.8, 0.1, 0.2]) == 1.0
    assert uprest_auc([2, 0], [0.5, 0.5]) == 0.5
    assert uprest_pair_count([2, 2, 0, 1]) == 2 * 2
    # ordinal_auc stays descriptive: full ordering respected.
    assert ordinal_auc([0, 1, 2], [0.1, 0.2, 0.3]) == 1.0
    assert ordinal_auc([0, 1, 2], [0.3, 0.2, 0.1]) == 0.0
    # top-k fractions break score ties by index (deterministic).
    assert top_k_up_fraction([2, 0, 0], [0.9, 0.9, 0.1], k=2) == 0.5
    assert top_k_down_fraction([2, 0, 0], [0.9, 0.9, 0.1], k=2) == 0.5
    # challenger input drops comments only (same cleaner/caps path).
    full = compose_story_text("T", "self", "comments", "article")
    nocom = challenger_text("T", "self", "article")
    assert "comments" in full and "comments" not in nocom
    assert nocom == compose_story_text("T", "self", "", "article")
    assert challenger_text("T", "", "") == "T."
    # bootstrap: identical arms -> delta 0, lo <= 0 <= hi; seeded repeat.
    y = [[2, 0, 1, 2, 0, 1]] * 2
    s = [[0.9, 0.1, 0.5, 0.8, 0.2, 0.4]] * 2
    b1 = paired_story_bootstrap_lo(y, s, s, n_boot=50, seed=BOOT_SEED)
    b2 = paired_story_bootstrap_lo(y, s, s, n_boot=50, seed=BOOT_SEED)
    assert b1 == b2 and b1["lo"] <= 0.0 <= b1["hi"]
    # strictly better arm -> positive mean (baseline ranks a down above
    # an up; challenger separates cleanly).
    imperfect = [[0.5, 0.9, 0.5, 0.4, 0.6, 0.3]] * 2
    better = [[0.95, 0.05, 0.5, 0.85, 0.15, 0.45]] * 2
    b3 = paired_story_bootstrap_lo(y, imperfect, better, n_boot=50, seed=BOOT_SEED)
    assert b3["mean"] > 0.0
    # cache/input policy: challenger namespace differs; per-arm DB copies are
    # mandatory because upsert_embedding keys on story_id alone.
    prod = "mxbai-embed-xsmall-v1|mean|norm|4096"
    assert _challenger_version(prod) == prod + "|nocomments"
    assert _challenger_version(prod) != prod
    assert _challenger_choice(True, True) == "encode"
    assert _challenger_choice(False, True) == "reuse"
    assert _challenger_choice(True, False) == "reuse"
    # pooled mean weights by pair count: hand-weighted toy check.
    assert _pooled_mean([1.0, 0.0], [3, 1]) == 0.75
    assert math.isnan(_pooled_mean([float("nan")], [5]))
    assert math.isnan(_pooled_mean([0.5], [0]))
    # ordinal-vs-binary pooling agree on two-class data (UP-vs-rest pairs
    # equal ordinal pairs when only two labels are present).
    yb = [2, 2, 0, 0]
    assert uprest_pair_count(yb) == _pair_count([1, 1, 0, 0])
    # task-path guard refuses live DB names and non-task work paths.
    assert _is_task_path(TASK_ROOT + "/snapshot_baseline.db")
    assert not _is_task_path("/tmp/other/snapshot.db")
    try:
        _guard_task_paths(
            "/tmp/opencode/cc-replay/snapshot.db", "/tmp/x.db", TASK_ROOT + "/work"
        )
        raise AssertionError("guard should have refused /tmp/x.db")
    except RuntimeError:
        pass
    _guard_task_reads("/tmp/opencode/cc-replay/snapshot.db", TASK_ROOT + "/parity20")
    # parity id selection is deterministic, stratified, IDs-only.
    toy = [
        StoryInputHashes(sid, "hn", True, "s", "n", False, f"block{b}", "e", True)
        for b in range(1, 6)
        for sid in range(b * 100, b * 100 + 10)
    ]
    p1 = _parity_ids(toy, 2)
    p2 = _parity_ids(toy, 2)
    assert p1 == p2 and len(p1) == 10 and len(set(p1)) == 10
    # tied scores in both arms -> zero delta interval covers 0.
    yt = [[2, 0, 2, 0]]
    st = [[0.5, 0.5, 0.5, 0.5]]
    bt = paired_story_bootstrap_lo(yt, st, st, n_boot=20, seed=BOOT_SEED)
    assert bt["mean"] == 0.0 and bt["lo"] <= 0.0 <= bt["hi"]
    print("self-check: 13 groups passed (metrics, input rule, bootstrap, guards).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Primary-input harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="read-only input audit")
    m.add_argument("--snapshot", required=True)
    m.add_argument("--sample", required=True)
    m.add_argument("--config", required=True)
    m.add_argument("--out", required=False, default=None)
    v = sub.add_parser("validate", help="baseline probability reproduction")
    v.add_argument("--snapshot-src", required=True)
    v.add_argument("--work-snapshot", required=True)
    v.add_argument("--sample", required=True)
    v.add_argument("--oof-scores", required=True)
    v.add_argument("--config", required=True)
    v.add_argument("--out-dir", required=True)
    p = sub.add_parser("parity", help="20-row full-text encoder parity gate")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-block", type=int, default=PARITY_PER_BLOCK)
    e = sub.add_parser("encode", help="challenger nocom vectors (resumable)")
    e.add_argument("--snapshot", required=True)
    e.add_argument("--sample", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--config", required=True)
    e.add_argument("--out-dir", required=True)
    c = sub.add_parser("compare", help="one-arm fits with real scorer")
    c.add_argument("--arm", required=True, choices=("baseline", "challenger"))
    c.add_argument("--snapshot-src", required=True)
    c.add_argument("--work-snapshot", required=True)
    c.add_argument("--sample", required=True)
    c.add_argument("--manifest", required=False, default=None)
    c.add_argument("--challenger-npz", required=False, default=None)
    c.add_argument("--oof-scores", required=False, default=None)
    c.add_argument("--config", required=True)
    c.add_argument("--out-dir", required=True)
    r = sub.add_parser("report", help="pool metrics + frozen gates (pure)")
    r.add_argument("--baseline-rows", required=True)
    r.add_argument("--challenger-rows", required=True)
    r.add_argument("--sample", required=True)
    r.add_argument("--out", required=True)
    sub.add_parser("self-check", help="pure behavioral checks")
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        return run_manifest(args.snapshot, args.sample, args.config, args.out)
    if args.cmd == "validate":
        return run_validate(
            args.snapshot_src,
            args.work_snapshot,
            args.sample,
            args.oof_scores,
            args.config,
            args.out_dir,
        )
    if args.cmd == "parity":
        return run_parity(
            args.snapshot,
            args.sample,
            args.manifest,
            args.config,
            args.out_dir,
            args.per_block,
        )
    if args.cmd == "encode":
        return run_encode(
            args.snapshot, args.sample, args.manifest, args.config, args.out_dir
        )
    if args.cmd == "compare":
        if args.arm == "challenger" and not (args.manifest and args.challenger_npz):
            print(
                "compare refused: challenger arm needs manifest + vectors.",
                flush=True,
            )
            return 2
        return run_compare(
            args.arm,
            args.snapshot_src,
            args.work_snapshot,
            args.sample,
            args.manifest,
            args.challenger_npz,
            args.oof_scores,
            args.config,
            args.out_dir,
        )
    if args.cmd == "report":
        return run_report(
            args.baseline_rows, args.challenger_rows, args.sample, args.out
        )
    return run_self_check()


if __name__ == "__main__":
    raise SystemExit(main())
