"""Bounded read-only input audit for the comments-only two-arm question.

Compares stored text_content against the full composer on user 151 voted
rows only. No embedding inference, no fits, no writes, no private text out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from database import Story
from pipeline.embedding_sections import story_section_text
from pipeline.ranking import clean_text, compose_story_text


@dataclass(frozen=True)
class AuditConfig:
    snapshot: str
    sample: str | None
    user_id: int


@dataclass(frozen=True)
class AuditResult:
    snapshot: str
    wal_bytes: int
    user_id: int
    voted_rows: int
    equal_stored_vs_compose: int
    mismatch_stored_vs_compose: int
    empty_stored_text: int
    no_self_no_article_raw: int
    no_self_no_article_clean: int
    nocom_equals_body_section: int
    nocom_empty_or_title_only: int
    source_counts: dict[str, int]
    source_equal_counts: dict[str, int]
    label_counts: dict[str, int]
    block_counts: dict[str, int]
    block_label_counts: dict[str, dict[str, int]]
    unmapped_rows: int
    sample_path: str | None


def _wal_bytes(snapshot: str) -> int:
    wal = snapshot + "-wal"
    try:
        return os.path.getsize(wal)
    except FileNotFoundError:
        return 0


def _load_block_map(sample_path: str | None) -> dict[int, str]:
    if not sample_path:
        return {}
    data = json.loads(Path(sample_path).read_text())
    blocks = data.get("blocks", {})
    out: dict[int, str] = {}
    for bname, sids in blocks.items():
        if isinstance(sids, list):
            for sid in sids:
                out[int(sid)] = str(bname)
    return out


def run_audit(cfg: AuditConfig) -> AuditResult:
    wal = _wal_bytes(cfg.snapshot)
    if wal != 0:
        raise RuntimeError(f"WAL not empty ({wal} bytes); refusing to read")
    block_map = _load_block_map(cfg.sample)
    uri = f"file:{cfg.snapshot}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.execute(
            "SELECT f.story_id, f.action, s.title, s.source,"
            " s.self_text, s.top_comments, s.article_body, s.text_content"
            " FROM feedback f JOIN stories s ON s.id = f.story_id"
            " WHERE f.user_id = ?",
            (cfg.user_id,),
        )
        rows = cur.fetchall()
    finally:
        con.close()

    source_counts: Counter[str] = Counter()
    source_equal: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    block_labels: dict[str, Counter[str]] = {}
    equal = 0
    empty_stored = 0
    no_raw = 0
    no_clean = 0
    nocom_eq_body = 0
    nocom_trivial = 0

    for (
        story_id,
        action,
        title,
        source,
        self_text,
        top_comments,
        article_body,
        text_content,
    ) in rows:
        title_s: str = title or ""
        self_s: str = self_text or ""
        comm_s: str = top_comments or ""
        art_s: str = article_body or ""
        stored: str = text_content or ""
        src: str = source or ""
        act: str = action or ""
        source_counts[src] += 1
        label_counts[act] += 1
        bname = block_map.get(int(story_id), "unmapped")
        block_counts[bname] += 1
        block_labels.setdefault(bname, Counter())[act] += 1

        composed = compose_story_text(title_s, self_s, comm_s, art_s)
        if stored == composed:
            equal += 1
            source_equal[src] += 1
        if not stored:
            empty_stored += 1
        if not self_s.strip() and not art_s.strip():
            no_raw += 1
        if not clean_text(self_s) and not clean_text(art_s):
            no_clean += 1

        story = Story(
            id=int(story_id),
            title=title_s,
            url=None,
            score=0,
            time=0,
            text_content=stored,
            source=src,
            self_text=self_s,
            top_comments=comm_s,
            article_body=art_s,
        )
        nocom = compose_story_text(title_s, self_s, "", art_s)
        if nocom == story_section_text(story, "body"):
            nocom_eq_body += 1
        if not clean_text(self_s) and not clean_text(art_s):
            nocom_trivial += 1

    total = len(rows)
    block_label_out = {k: dict(v) for k, v in block_labels.items()}
    return AuditResult(
        snapshot=cfg.snapshot,
        wal_bytes=wal,
        user_id=cfg.user_id,
        voted_rows=total,
        equal_stored_vs_compose=equal,
        mismatch_stored_vs_compose=total - equal,
        empty_stored_text=empty_stored,
        no_self_no_article_raw=no_raw,
        no_self_no_article_clean=no_clean,
        nocom_equals_body_section=nocom_eq_body,
        nocom_empty_or_title_only=nocom_trivial,
        source_counts=dict(source_counts),
        source_equal_counts=dict(source_equal),
        label_counts=dict(label_counts),
        block_counts=dict(block_counts),
        block_label_counts=block_label_out,
        unmapped_rows=int(block_counts.get("unmapped", 0)),
        sample_path=cfg.sample,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only text-input audit")
    ap.add_argument("--snapshot", required=True, type=str)
    ap.add_argument("--sample", required=False, default=None, type=str)
    ap.add_argument("--user-id", required=False, default=151, type=int)
    ap.add_argument("--out", required=False, default=None, type=str)
    args = ap.parse_args(argv)
    cfg = AuditConfig(snapshot=args.snapshot, sample=args.sample, user_id=args.user_id)
    res = run_audit(cfg)
    payload = asdict(res)
    payload["script_sha16"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[
        :16
    ]
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
