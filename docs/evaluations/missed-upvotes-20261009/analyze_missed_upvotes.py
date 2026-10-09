"""Bounded missed-upvote analysis over EXISTING held-out predictions.

Reads only: baseline/challenger row JSON (held-out predictions), the
immutable task snapshot (read-only SELECTs of aggregate-safe columns),
and cached facet annotations (opaque mapping + parsed labels).

Writes only: aggregates.json (counts, no titles/transcripts/text).

Definitions:
  - UP rows are true_label == 2 (verified: feedback 'up' maps to 2,
    'down' to 0; spot-checked story -2144136158 -> up/2).
  - Within-block rank orders baseline native production_score desc
    (tie-break story_id asc for determinism). Missed upvote = UP row
    with rank > TOP_K (12) in its block.
  - Worst-ranked UP quintile = pooled UP rows sorted by within-block
    rank desc (worst first), first ceil(n_up / 5). Documented in output.
  - Broad source: hn-family -> hn; rss_reddit_* -> reddit;
    rss_lesswrong* -> lesswrong; else other.
  - Content class from aggregate lengths only (never text):
    body (self_text or article_body non-empty), comments_only
    (top_comments non-empty, no body), title_only (neither).

Usage (VPS, project uv, one CPU, read-only snapshot):
  uv run --project /home/dev/hn-rewrite/main python analyze_missed_upvotes.py \\
    --baseline-rows /tmp/opencode/representation-input-20261009/compare_baseline/baseline_rows.json \\
    --challenger-rows /tmp/opencode/representation-input-20261009/compare_challenger/challenger_rows.json \\
    --snapshot /tmp/opencode/cc-replay/snapshot.db \\
    --facet-labels /home/dev/hn-rewrite/llm-labels/facet-residual-20261009/facet_labels.json \\
    --facet-batches /home/dev/hn-rewrite/llm-labels/facet-residual-20261009/facet_batches.json \\
    --out aggregates.json [--check-only ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass, field


UP_LABEL: int = 2
NEUTRAL_LABEL: int = 1
DOWN_LABEL: int = 0
TOP_K: int = 12
SCHEMA_VERSION: str = "missed-upvotes-v1"

RETAINED_FACETS: tuple[str, ...] = (
    "artifact",
    "claim",
    "conflict_framing",
    "entity_focus",
    "voice",
)
DROPPED_FACETS: tuple[str, ...] = ("technical_depth", "actionable")
ALL_FACET_FIELDS: tuple[str, ...] = RETAINED_FACETS + DROPPED_FACETS

SMALL_N_THRESHOLD: int = 10


@dataclass(frozen=True)
class PredRow:
    block: int
    story_id: int
    true_label: int
    production_score: float
    vote_time: float


@dataclass(frozen=True)
class RankedRow:
    row: PredRow
    rank: int
    block_size: int


@dataclass(frozen=True)
class StoryMeta:
    story_id: int
    raw_source: str
    broad_source: str
    self_len: int
    article_len: int
    comments_len: int
    text_len: int
    comment_count: int
    content_class: str


@dataclass(frozen=True)
class FacetAnn:
    story_id: int
    kind: str
    fields: dict[str, str]


@dataclass
class CategoryCount:
    n_up: int = 0
    n_missed: int = 0
    n_quintile: int = 0
    n_down: int = 0
    n_down_top12: int = 0


def broad_source(raw: str) -> str:
    if raw in ("hn", "ch_seed", "bq_seed"):
        return "hn"
    if raw.startswith("rss_reddit_"):
        return "reddit"
    if raw.startswith("rss_lesswrong"):
        return "lesswrong"
    return "other"


def content_class(self_len: int, article_len: int, comments_len: int) -> str:
    if self_len > 0 or article_len > 0:
        return "body"
    if comments_len > 0:
        return "comments_only"
    return "title_only"


def _expect_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be int")
    return value


def _expect_float(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be numeric")
    return float(value)


def _expect_str(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be str")
    return value


def load_pred_rows(path: str) -> list[PredRow]:
    with open(path, encoding="utf-8") as fh:
        raw: object = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError("pred rows JSON must be a list")
    rows: list[PredRow] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("pred row must be an object")
        block = _expect_int(entry.get("block"), "block")
        story_id = _expect_int(entry.get("story_id"), "story_id")
        true_label = _expect_int(entry.get("true_label"), "true_label")
        if true_label not in (DOWN_LABEL, NEUTRAL_LABEL, UP_LABEL):
            raise ValueError(f"unexpected true_label {true_label}")
        score = _expect_float(entry.get("production_score"), "production_score")
        vote_time = _expect_float(entry.get("vote_time"), "vote_time")
        rows.append(
            PredRow(
                block=block,
                story_id=story_id,
                true_label=true_label,
                production_score=score,
                vote_time=vote_time,
            )
        )
    return rows


def rank_within_block(rows: list[PredRow]) -> dict[int, RankedRow]:
    by_block: dict[int, list[PredRow]] = {}
    for row in rows:
        by_block.setdefault(row.block, []).append(row)
    ranked: dict[int, RankedRow] = {}
    for block, members in by_block.items():
        ordered = sorted(members, key=lambda r: (-r.production_score, r.story_id))
        size = len(ordered)
        for i, row in enumerate(ordered):
            ranked[row.story_id] = RankedRow(row=row, rank=i + 1, block_size=size)
    return ranked


def load_story_metas(snapshot_path: str, story_ids: list[int]) -> dict[int, StoryMeta]:
    metas: dict[int, StoryMeta] = {}
    uri = f"file:{snapshot_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.cursor()
        for start in range(0, len(story_ids), 500):
            chunk = story_ids[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            cur.execute(
                "SELECT id, source, LENGTH(self_text), LENGTH(article_body), "
                "LENGTH(top_comments), LENGTH(text_content), comment_count "
                f"FROM stories WHERE id IN ({placeholders})",
                chunk,
            )
            for sid, src, slen, alen, clen, tlen, ccount in cur.fetchall():
                story_id = int(sid)
                raw_source = str(src)
                metas[story_id] = StoryMeta(
                    story_id=story_id,
                    raw_source=raw_source,
                    broad_source=broad_source(raw_source),
                    self_len=int(slen or 0),
                    article_len=int(alen or 0),
                    comments_len=int(clen or 0),
                    text_len=int(tlen or 0),
                    comment_count=int(ccount or 0),
                    content_class=content_class(
                        int(slen or 0), int(alen or 0), int(clen or 0)
                    ),
                )
    finally:
        con.close()
    return metas


def load_facet_anns(labels_path: str, batches_path: str) -> dict[int, FacetAnn]:
    with open(labels_path, encoding="utf-8") as fh:
        labels_raw: object = json.load(fh)
    with open(batches_path, encoding="utf-8") as fh:
        batches_raw: object = json.load(fh)
    if not isinstance(labels_raw, dict) or not isinstance(batches_raw, dict):
        raise ValueError("facet files must be JSON objects")
    results = labels_raw.get("results")
    batches = batches_raw.get("batches")
    if not isinstance(results, dict) or not isinstance(batches, list):
        raise ValueError("facet results must be object, batches must be list")
    opaque_to_story: dict[str, tuple[int, str]] = {}
    kind_priority: dict[str, int] = {"sample": 0, "dev": 1, "repeat": 2}
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("facet batch must be an object")
        mapping = batch.get("mapping")
        if not isinstance(mapping, list):
            raise ValueError("facet batch mapping must be a list")
        for item in mapping:
            if not isinstance(item, dict):
                raise ValueError("facet mapping item must be an object")
            opaque = _expect_str(item.get("opaque"), "opaque")
            story_id = _expect_int(item.get("story_id"), "mapping.story_id")
            kind = _expect_str(item.get("kind"), "kind")
            prev = opaque_to_story.get(opaque)
            if prev is None or kind_priority.get(kind, 9) < kind_priority.get(
                prev[1], 9
            ):
                opaque_to_story[opaque] = (story_id, kind)
    anns: dict[int, FacetAnn] = {}
    for opaque, entry in results.items():
        if not isinstance(entry, dict):
            continue
        parsed = entry.get("parsed")
        if not isinstance(parsed, dict):
            continue
        mapped = opaque_to_story.get(opaque)
        if mapped is None:
            continue
        story_id, kind = mapped
        fields: dict[str, str] = {}
        for facet_field in ALL_FACET_FIELDS:
            value = parsed.get(facet_field, "__missing__")
            fields[facet_field] = value if isinstance(value, str) else "__missing__"
        existing = anns.get(story_id)
        if existing is None or kind_priority.get(kind, 9) < kind_priority.get(
            existing.kind, 9
        ):
            anns[story_id] = FacetAnn(story_id=story_id, kind=kind, fields=fields)
    return anns


def miss_rate(n_missed: int, n_up: int) -> float | None:
    if n_up == 0:
        return None
    return n_missed / n_up


def enrichment(missed_share: float | None, up_share: float | None) -> float | None:
    if missed_share is None or up_share is None or up_share == 0:
        return None
    return missed_share / up_share


@dataclass
class AnalysisResult:
    aggregates: dict[str, object]
    worst_missed_ids: list[int] = field(default_factory=list)


def analyze(
    baseline: list[PredRow],
    challenger: list[PredRow] | None,
    metas: dict[int, StoryMeta],
    anns: dict[int, FacetAnn],
) -> AnalysisResult:
    ranked = rank_within_block(baseline)
    blocks = sorted({r.block for r in baseline})
    up_ranked = [ranked[r.story_id] for r in baseline if r.true_label == UP_LABEL]
    down_ranked = [ranked[r.story_id] for r in baseline if r.true_label == DOWN_LABEL]
    neutral_ranked = [
        ranked[r.story_id] for r in baseline if r.true_label == NEUTRAL_LABEL
    ]
    missed_ids = {rr.row.story_id for rr in up_ranked if rr.rank > TOP_K}
    hit_ids = {rr.row.story_id for rr in up_ranked if rr.rank <= TOP_K}
    down_top12_ids = {rr.row.story_id for rr in down_ranked if rr.rank <= TOP_K}
    neutral_top12_ids = {rr.row.story_id for rr in neutral_ranked if rr.rank <= TOP_K}

    quintile_n = math.ceil(len(up_ranked) / 5) if up_ranked else 0
    worst_first = sorted(up_ranked, key=lambda rr: (-rr.rank, rr.row.story_id))
    quintile_ids = {rr.row.story_id for rr in worst_first[:quintile_n]}

    def meta_of(sid: int) -> StoryMeta | None:
        return metas.get(sid)

    # --- broad source / content tables over UP rows ---
    source_counts: dict[str, CategoryCount] = {}
    content_counts: dict[str, CategoryCount] = {}
    for rr in up_ranked:
        meta = meta_of(rr.row.story_id)
        broad = meta.broad_source if meta else "__missing_meta__"
        cclass = meta.content_class if meta else "__missing_meta__"
        for table, key in ((source_counts, broad), (content_counts, cclass)):
            cell = table.setdefault(key, CategoryCount())
            cell.n_up += 1
            if rr.row.story_id in missed_ids:
                cell.n_missed += 1
            if rr.row.story_id in quintile_ids:
                cell.n_quintile += 1
    for rr in down_ranked:
        meta = meta_of(rr.row.story_id)
        broad = meta.broad_source if meta else "__missing_meta__"
        cclass = meta.content_class if meta else "__missing_meta__"
        for table, key in ((source_counts, broad), (content_counts, cclass)):
            cell = table.setdefault(key, CategoryCount())
            cell.n_down += 1
            if rr.row.story_id in down_top12_ids:
                cell.n_down_top12 += 1

    n_up = len(up_ranked)
    n_missed = len(missed_ids)

    def table_json(
        table: dict[str, CategoryCount], denom_up: int, denom_missed: int
    ) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for key in sorted(table):
            cell = table[key]
            up_share = (cell.n_up / denom_up) if denom_up else None
            missed_share = (cell.n_missed / denom_missed) if denom_missed else None
            out[key] = {
                "n_up": cell.n_up,
                "n_missed": cell.n_missed,
                "miss_rate": miss_rate(cell.n_missed, cell.n_up),
                "up_share": up_share,
                "missed_share": missed_share,
                "enrichment_missed_vs_up": enrichment(missed_share, up_share),
                "n_up_worst_quintile": cell.n_quintile,
                "n_down": cell.n_down,
                "n_down_top12": cell.n_down_top12,
                "down_promotion_rate": (
                    (cell.n_down_top12 / cell.n_down) if cell.n_down else None
                ),
                "small_n": cell.n_up < SMALL_N_THRESHOLD,
            }
        return out

    # --- per-block consistency + per-block source/content tables ---
    # All ranks are within-block; per-block tables use block UP denominators,
    # so block composition differences are read against each block's own base
    # rates, never against pooled absolute scores.
    per_block: dict[str, dict[str, object]] = {}
    per_block_source: dict[str, dict[str, dict[str, object]]] = {}
    per_block_content: dict[str, dict[str, dict[str, object]]] = {}
    for block in blocks:
        b_up = [rr for rr in up_ranked if rr.row.block == block]
        b_missed = [rr for rr in b_up if rr.row.story_id in missed_ids]
        b_down = [rr for rr in down_ranked if rr.row.block == block]
        b_down_top = [rr for rr in b_down if rr.row.story_id in down_top12_ids]
        b_quint = [rr for rr in b_up if rr.row.story_id in quintile_ids]
        b_neutral_top = [
            rr for rr in neutral_ranked if rr.row.block == block and rr.rank <= TOP_K
        ]
        per_block[str(block)] = {
            "n_rows": sum(1 for r in baseline if r.block == block),
            "n_up": len(b_up),
            "n_hit_top12": len(b_up) - len(b_missed),
            "n_missed": len(b_missed),
            "miss_rate": miss_rate(len(b_missed), len(b_up)),
            "n_up_worst_quintile": len(b_quint),
            "worst_quintile_rate_of_block_up": (
                (len(b_quint) / len(b_up)) if b_up else None
            ),
            "n_down": len(b_down),
            "n_down_top12": len(b_down_top),
            "down_promotion_rate": (
                (len(b_down_top) / len(b_down)) if b_down else None
            ),
            "n_neutral_top12": len(b_neutral_top),
        }
        b_source: dict[str, CategoryCount] = {}
        b_content: dict[str, CategoryCount] = {}
        for rr in b_up:
            meta = meta_of(rr.row.story_id)
            broad = meta.broad_source if meta else "__missing_meta__"
            cclass = meta.content_class if meta else "__missing_meta__"
            for table, key in ((b_source, broad), (b_content, cclass)):
                cell = table.setdefault(key, CategoryCount())
                cell.n_up += 1
                if rr.row.story_id in missed_ids:
                    cell.n_missed += 1
                if rr.row.story_id in quintile_ids:
                    cell.n_quintile += 1
        for rr in b_down:
            meta = meta_of(rr.row.story_id)
            broad = meta.broad_source if meta else "__missing_meta__"
            cclass = meta.content_class if meta else "__missing_meta__"
            for table, key in ((b_source, broad), (b_content, cclass)):
                cell = table.setdefault(key, CategoryCount())
                cell.n_down += 1
                if rr.row.story_id in down_top12_ids:
                    cell.n_down_top12 += 1
        per_block_source[str(block)] = table_json(b_source, len(b_up), len(b_missed))
        per_block_content[str(block)] = table_json(b_content, len(b_up), len(b_missed))

    # --- facet subset (no population claims; report coverage) ---
    facet_up = [rr for rr in up_ranked if rr.row.story_id in anns]
    facet_missed = [rr for rr in facet_up if rr.row.story_id in missed_ids]
    facet_quint = [rr for rr in facet_up if rr.row.story_id in quintile_ids]
    facet_down = [rr for rr in down_ranked if rr.row.story_id in anns]
    facet_down_top = [rr for rr in facet_down if rr.row.story_id in down_top12_ids]
    facet_tables: dict[str, dict[str, dict[str, object]]] = {}
    for facet_field in ALL_FACET_FIELDS:
        value_counts: dict[str, CategoryCount] = {}
        for rr in facet_up:
            ann = anns[rr.row.story_id]
            value = ann.fields.get(facet_field, "__missing__")
            cell = value_counts.setdefault(value, CategoryCount())
            cell.n_up += 1
            if rr.row.story_id in missed_ids:
                cell.n_missed += 1
            if rr.row.story_id in quintile_ids:
                cell.n_quintile += 1
        for rr in facet_down:
            ann = anns[rr.row.story_id]
            value = ann.fields.get(facet_field, "__missing__")
            cell = value_counts.setdefault(value, CategoryCount())
            cell.n_down += 1
            if rr.row.story_id in down_top12_ids:
                cell.n_down_top12 += 1
        facet_tables[facet_field] = table_json(
            value_counts, len(facet_up), len(facet_missed)
        )

    facet_value_dist: dict[str, Counter[str]] = {}
    for facet_field in ALL_FACET_FIELDS:
        facet_value_dist[facet_field] = Counter(
            anns[rr.row.story_id].fields.get(facet_field, "__missing__")
            for rr in facet_up
        )

    # --- challenger consistency (descriptive overlap only) ---
    challenger_json: dict[str, object] | None = None
    if challenger is not None:
        ch_ranked = rank_within_block(challenger)
        ch_up = [ch_ranked[r.story_id] for r in challenger if r.true_label == UP_LABEL]
        ch_missed = {rr.row.story_id for rr in ch_up if rr.rank > TOP_K}
        overlap = len(missed_ids & ch_missed)
        ch_per_block: dict[str, dict[str, object]] = {}
        for block in sorted({r.block for r in challenger}):
            b = [rr for rr in ch_up if rr.row.block == block]
            m = [rr for rr in b if rr.row.story_id in ch_missed]
            ch_per_block[str(block)] = {
                "n_up": len(b),
                "n_missed": len(m),
                "miss_rate": miss_rate(len(m), len(b)),
            }
        challenger_json = {
            "n_up": len(ch_up),
            "n_missed": len(ch_missed),
            "miss_rate": miss_rate(len(ch_missed), len(ch_up)),
            "overlap_missed_with_baseline": overlap,
            "baseline_only_missed": len(missed_ids - ch_missed),
            "challenger_only_missed": len(ch_missed - missed_ids),
            "per_block": ch_per_block,
        }

    label_counts = Counter(r.true_label for r in baseline)
    n_slots = TOP_K * len(blocks)
    oracle_max_hits = min(n_slots, n_up)
    unavoidable_misses = n_up - oracle_max_hits
    n_neutral_top12 = len(neutral_top12_ids)
    aggregates: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "definitions": {
            "missed_upvote": f"UP (true_label=={UP_LABEL}) with within-block "
            f"baseline native production_score rank > {TOP_K}",
            "worst_quintile": "pooled UP rows sorted by within-block rank desc "
            f"(worst first), first ceil(n_up/5) = {quintile_n}. Rank-based "
            "only; absolute scores never compared across blocks.",
            "rank": "within-block baseline native production_score desc, "
            "tie-break story_id asc",
            "oracle_ceiling": f"{TOP_K} slots x {len(blocks)} blocks = "
            f"{n_slots} slots for {n_up} UPs: max {oracle_max_hits} hits, "
            f"{unavoidable_misses} misses unavoidable even for a perfect "
            "ranker. Replaceable non-UP picks are counted, not inferred.",
            "broad_source_map": "hn|ch_seed|bq_seed->hn; rss_reddit_*->reddit; "
            "rss_lesswrong*->lesswrong; else other",
            "content_class": "body=self_text|article_body non-empty; "
            "comments_only=top_comments non-empty without body; "
            "title_only=neither (lengths only, no text read)",
            "facet_scope": "UP rows in blocks 2-5 with a cached facet "
            "annotation; repeats deduped (sample>dev>repeat); "
            "subset-only, no population claims",
            "down_mirror_scope": "baseline top-12 placement of DOWN rows "
            "only; not a test of Muse corrections",
            "small_n_threshold": SMALL_N_THRESHOLD,
        },
        "inputs": {
            "n_rows": len(baseline),
            "blocks": blocks,
            "label_counts": {
                "down_0": label_counts.get(DOWN_LABEL, 0),
                "neutral_1": label_counts.get(NEUTRAL_LABEL, 0),
                "up_2": label_counts.get(UP_LABEL, 0),
            },
            "n_story_metas_matched": sum(1 for r in baseline if r.story_id in metas),
            "facet_annotations_total_stories": len(anns),
            "facet_kinds_present": sorted({a.kind for a in anns.values()}),
        },
        "overall": {
            "n_up": n_up,
            "n_hit_top12": len(hit_ids),
            "n_missed": n_missed,
            "miss_rate": miss_rate(n_missed, n_up),
            "n_up_worst_quintile": quintile_n,
            "n_down": len(down_ranked),
            "n_down_top12": len(down_top12_ids),
            "down_promotion_rate": (
                (len(down_top12_ids) / len(down_ranked)) if down_ranked else None
            ),
            "n_neutral_top12": n_neutral_top12,
            "top12_slots": n_slots,
            "oracle_max_up_hits": oracle_max_hits,
            "oracle_unavoidable_misses": unavoidable_misses,
            "replaceable_non_up_top12": n_neutral_top12 + len(down_top12_ids),
        },
        "per_block": per_block,
        "per_block_source": per_block_source,
        "per_block_content": per_block_content,
        "by_broad_source": table_json(source_counts, n_up, n_missed),
        "by_content_class": table_json(content_counts, n_up, n_missed),
        "facet_subset": {
            "n_up_in_blocks_with_facet": len(facet_up),
            "coverage_of_up": ((len(facet_up) / n_up) if n_up else None),
            "n_missed_with_facet": len(facet_missed),
            "n_quintile_with_facet": len(facet_quint),
            "n_down_with_facet": len(facet_down),
            "n_down_top12_with_facet": len(facet_down_top),
            "note": "subset-only denominators; no population claims",
            "by_facet_value": facet_tables,
        },
        "challenger_consistency": challenger_json,
        "confounding_notes": [
            "Ranks are within-block, so block-level score shifts cannot "
            "explain block composition of the pooled worst quintile; read "
            "it against block UP denominators (per_block_source tables).",
            "Broad source and content class are confounded (e.g. hn vs "
            "reddit differ in body/comment availability); read jointly.",
            "Block UP counts drift (b2 196, b3 236, b4 151, b5 100); "
            "blocks with more UPs mechanically place more of them outside "
            "12 fixed slots.",
            "Facet subset is small and selection-shaped (sample+dev kinds "
            "from a prior design); compare within-subset only.",
            "Production scores are uncalibrated ranking scores, not "
            "confidences; low score means low rank, nothing more.",
            "All outcomes are reused development data; descriptive only.",
        ],
    }
    worst_missed_ids = [rr.row.story_id for rr in worst_first[: min(12, quintile_n)]]
    return AnalysisResult(aggregates=aggregates, worst_missed_ids=worst_missed_ids)


def validate_aggregates(aggregates: dict[str, object]) -> list[str]:
    errors: list[str] = []
    blob = json.dumps(aggregates)
    for forbidden in (
        "title",
        "transcript",
        "self_text",
        "article_body",
        "top_comments",
        "text_content",
    ):
        if f'"{forbidden}"' in blob:
            errors.append(f"aggregates leak private key: {forbidden}")
    overall = aggregates.get("overall")
    per_block = aggregates.get("per_block")
    if not isinstance(overall, dict) or not isinstance(per_block, dict):
        errors.append("missing overall/per_block")
        return errors
    n_up = overall.get("n_up")
    n_missed = overall.get("n_missed")
    n_hit = overall.get("n_hit_top12")
    if (
        isinstance(n_up, int)
        and isinstance(n_missed, int)
        and isinstance(n_hit, int)
        and n_up != n_missed + n_hit
    ):
        errors.append("overall n_up != n_missed + n_hit_top12")
    block_missed = 0
    block_up = 0
    for _, cell in per_block.items():
        if isinstance(cell, dict):
            missed = cell.get("n_missed")
            up = cell.get("n_up")
            if isinstance(missed, int):
                block_missed += missed
            if isinstance(up, int):
                block_up += up
    if isinstance(n_missed, int) and block_missed != n_missed:
        errors.append("per-block missed sum != overall missed")
    if isinstance(n_up, int) and block_up != n_up:
        errors.append("per-block up sum != overall up")
    slots = overall.get("top12_slots")
    oracle_max = overall.get("oracle_max_up_hits")
    unavoidable = overall.get("oracle_unavoidable_misses")
    replaceable = overall.get("replaceable_non_up_top12")
    n_down_top = overall.get("n_down_top12")
    n_neutral_top = overall.get("n_neutral_top12")
    if (
        isinstance(slots, int)
        and isinstance(oracle_max, int)
        and isinstance(unavoidable, int)
        and isinstance(n_up, int)
        and isinstance(n_hit, int)
        and isinstance(n_missed, int)
    ):
        if oracle_max != min(slots, n_up):
            errors.append("oracle_max != min(slots, n_up)")
        if unavoidable != n_up - oracle_max:
            errors.append("unavoidable != n_up - oracle_max")
        if n_missed < unavoidable:
            errors.append("missed below oracle unavoidable floor")
    if (
        isinstance(replaceable, int)
        and isinstance(n_down_top, int)
        and isinstance(n_neutral_top, int)
        and replaceable != n_down_top + n_neutral_top
    ):
        errors.append("replaceable != down_top12 + neutral_top12")
    for table_key in ("per_block_source", "per_block_content"):
        table = aggregates.get(table_key)
        if not isinstance(table, dict):
            errors.append(f"missing {table_key}")
            continue
        for block_key, cats in table.items():
            if not isinstance(cats, dict):
                errors.append(f"{table_key}[{block_key}] not an object")
                continue
            cat_up = sum(c.get("n_up", 0) for c in cats.values() if isinstance(c, dict))
            cat_missed = sum(
                c.get("n_missed", 0) for c in cats.values() if isinstance(c, dict)
            )
            cat_quint = sum(
                c.get("n_up_worst_quintile", 0)
                for c in cats.values()
                if isinstance(c, dict)
            )
            cat_down = sum(
                c.get("n_down", 0) for c in cats.values() if isinstance(c, dict)
            )
            cat_down_top = sum(
                c.get("n_down_top12", 0) for c in cats.values() if isinstance(c, dict)
            )
            block_cell = per_block.get(block_key)
            if isinstance(block_cell, dict):
                if cat_up != block_cell.get("n_up"):
                    errors.append(f"{table_key}[{block_key}] n_up sum mismatch")
                if cat_missed != block_cell.get("n_missed"):
                    errors.append(f"{table_key}[{block_key}] missed sum mismatch")
                if cat_quint != block_cell.get("n_up_worst_quintile"):
                    errors.append(f"{table_key}[{block_key}] quintile sum mismatch")
                if cat_down != block_cell.get("n_down"):
                    errors.append(f"{table_key}[{block_key}] down sum mismatch")
                if cat_down_top != block_cell.get("n_down_top12"):
                    errors.append(f"{table_key}[{block_key}] down top12 sum mismatch")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded missed-upvote aggregates from existing artifacts."
    )
    parser.add_argument("--baseline-rows", required=True)
    parser.add_argument("--challenger-rows", default=None)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--facet-labels", required=True)
    parser.add_argument("--facet-batches", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--print-private-ids",
        action="store_true",
        help="Print up to 12 worst-missed story IDs to stdout for VPS-side "
        "private inspection only (never copied to the report).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = load_pred_rows(str(args.baseline_rows))
    challenger: list[PredRow] | None = None
    if args.challenger_rows:
        challenger = load_pred_rows(str(args.challenger_rows))
    story_ids = [r.story_id for r in baseline]
    metas = load_story_metas(str(args.snapshot), story_ids)
    anns = load_facet_anns(str(args.facet_labels), str(args.facet_batches))
    result = analyze(baseline, challenger, metas, anns)
    errors = validate_aggregates(result.aggregates)
    if errors:
        for error in errors:
            print(f"VALIDATION ERROR: {error}")
        return 1
    with open(str(args.out), "w", encoding="utf-8") as fh:
        json.dump(result.aggregates, fh, indent=2, sort_keys=True)
        fh.write("\n")
    overall = result.aggregates["overall"]
    print(f"wrote {args.out}: {json.dumps(overall, sort_keys=True)}")
    if bool(args.print_private_ids):
        print("PRIVATE_INSPECTION_IDS=" + ",".join(map(str, result.worst_missed_ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
