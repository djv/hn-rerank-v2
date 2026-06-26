"""TLDR generation benchmark: unified prompt against an OpenRouter model,
score format compliance, save raw response and normalized TLDR to JSON.

Usage:
    LLM_PROVIDER=openrouter \
    OPENROUTER_API_KEY=sk-or-... \
    OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free \
    uv run python scripts/benchmark_tldr_llms.py --limit 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import _normalize_tldr_markdown  # noqa: E402

PROMPT_VERSION = "unified-v1"

# ── Input char limits ────────────────────────────────────────────────────
SELF_TEXT_PROMPT_CHAR_LIMIT = 8_000
COMMENT_PROMPT_CHAR_LIMIT = 12_000
ARTICLE_BODY_CHAR_LIMIT = 15_000

# ── Single unified prompt template ───────────────────────────────────────
# The model decides whether to include a ### Discussion section based on
# whether the comments input is present and substantive.
UNIFIED_PROMPT_TEMPLATE = """Write a scannable Markdown TLDR for a knowledgeable reader.
Optimized for an 11-inch screen — keep it under 240 words total.

Structure rules:
- Use a "### Article" section with 3-5 flat Markdown bullets summarizing the article.
- Use a "### Discussion" section with 2-4 flat bullets summarizing the comments, ONLY if the comments below are present and substantive. OMIT it otherwise.
- No nested list levels. Use "####" headings if you need to group sub-topics.
- Every non-empty line must start with "- ", "### ", or "#### ".
- Use **bold** key terms.
- Keep each bullet to one short sentence.

Content rules:
- Use ONLY information from the inputs below. Do not expand with outside knowledge.
- If a section's input is empty or trivial, OMIT that section entirely — do not invent content.
- If everything is thin, say so plainly in a single bullet under "### Article".

Title: {title}

Author's text (may be empty):
{self_text}

Article body (may be empty):
{article_body}

Comments (may be empty):
{top_comments}
"""


# ── Data types ───────────────────────────────────────────────────────────


@dataclass
class StoryRecord:
    id: int
    title: str
    url: str | None
    source: str
    self_text: str
    top_comments: str
    article_body: str
    text_content: str


@dataclass
class ComplianceResult:
    passes: bool
    violations: list[str]
    score: float


@dataclass
class TldrResult:
    story_id: int
    title: str
    source: str
    status: str  # "ok", "skipped_rate_limit", "client_error", "exception"
    latency_ms: float
    tldr: str = ""  # post-normalize; empty when status != "ok"
    raw_response: str = ""  # pre-normalize LLM response
    compliance: ComplianceResult | None = None


@dataclass
class BenchmarkReport:
    schema_version: int = 2  # bumped from 1 for unified-v1 shape change
    started_at: str = ""
    finished_at: str = ""
    config: dict = field(default_factory=dict)
    sample: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)


# ── Sample selection ─────────────────────────────────────────────────────


def _source_bucket(source: str) -> str:
    """Group sources into hn / rss / seed buckets."""
    if source == "hn":
        return "hn"
    if source.startswith("rss_"):
        return "rss"
    if source in ("ch_seed", "bq_seed"):
        return "seed"
    return "other"


def select_sample(db_path: str, limit: int = 25) -> list[StoryRecord]:
    """Select a deterministic, mixed-source sample."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT id, title, url, source, self_text, top_comments, article_body, text_content
            FROM stories
            WHERE text_content != ''
              AND source IN ('hn', 'ch_seed', 'bq_seed')
               OR (source LIKE 'rss_%' AND text_content != '')
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    all_stories = [
        StoryRecord(
            id=r[0],
            title=r[1],
            url=r[2],
            source=r[3],
            self_text=r[4] or "",
            top_comments=r[5] or "",
            article_body=r[6] or "",
            text_content=r[7],
        )
        for r in rows
        if r[4] or r[5] or r[6]  # must have some content for TLDR
    ]

    bucket_quota: dict[str, int] = {
        "hn": limit // 2,
        "rss": limit // 5,
        "seed": limit // 5,
    }
    leftovers = limit - sum(bucket_quota.values())

    sampled: list[StoryRecord] = []
    buckets: dict[str, list[StoryRecord]] = {
        "hn": [],
        "rss": [],
        "seed": [],
        "other": [],
    }
    for s in all_stories:
        buckets[_source_bucket(s.source)].append(s)

    for bucket_name in ("hn", "rss", "seed"):
        quota = bucket_quota[bucket_name]
        sampled.extend(buckets[bucket_name][:quota])

    remaining = [
        s for b in ("hn", "rss", "seed") for s in buckets[b] if s not in sampled
    ]
    sampled.extend(remaining[:leftovers])

    return sampled[:limit]


# ── OpenRouter call ──────────────────────────────────────────────────────


async def _call_openrouter_chat(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    base_url: str = "https://openrouter.ai/api/v1/chat/completions",
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8765",
        "X-Title": "hn-rewrite-benchmark",
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(base_url, headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        return f"Error: HTTP {resp.status_code} - {resp.text}"


# ── Format compliance scoring ────────────────────────────────────────────


def _count_bullets(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.strip().startswith("- "))


def _word_count(text: str) -> int:
    return len(text.split())


def _has_nested_lists(text: str) -> bool:
    return bool(re.search(r"^\s{2,}- ", text, re.MULTILINE))


def score_compliance(tldr: str) -> ComplianceResult:
    violations: list[str] = []

    if not tldr or tldr.startswith("Error:"):
        return ComplianceResult(passes=False, violations=["nonempty"], score=0.0)

    # 1. ### Article heading is required
    if "### Article" not in tldr:
        violations.append("missing_article_heading")

    # 2. Every non-empty line must start with "- ", "### ", or "#### "
    for line in tldr.split("\n"):
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped.startswith(("- ", "### ", "#### ")):
            continue
        violations.append(f"unexpected_format: {stripped[:60]}")
        break

    # 3. No nested list levels
    if _has_nested_lists(tldr):
        violations.append("nested_lists")

    # 4. Word cap
    total = _word_count(tldr)
    if total > 240:
        violations.append(f"word_count_{total}")

    # 5. At least one bullet somewhere
    if _count_bullets(tldr) < 1:
        violations.append("no_bullets")

    # 6. Idempotent normalization
    normalized = _normalize_tldr_markdown(tldr)
    if _normalize_tldr_markdown(normalized) != normalized:
        violations.append("normalization_not_idempotent")

    score = len(violations)
    normalized_score = 1.0 - (score / 6) if score > 0 else 1.0
    return ComplianceResult(
        passes=score == 0,
        violations=violations,
        score=round(normalized_score, 4),
    )


# ── Per-story generation ─────────────────────────────────────────────────


async def generate_tldr_for_story(
    story: StoryRecord,
    api_key: str,
    model: str,
    base_url: str,
    rate_sleep: float,
) -> TldrResult:
    prompt = UNIFIED_PROMPT_TEMPLATE.format(
        title=story.title,
        self_text=story.self_text[:SELF_TEXT_PROMPT_CHAR_LIMIT],
        article_body=story.article_body[:ARTICLE_BODY_CHAR_LIMIT],
        top_comments=story.top_comments[:COMMENT_PROMPT_CHAR_LIMIT],
    )

    start = time.perf_counter()
    try:
        raw = await _call_openrouter_chat(
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=900,
            base_url=base_url,
        )
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        return TldrResult(
            story_id=story.id,
            title=story.title,
            source=story.source,
            status="exception",
            latency_ms=round(elapsed, 1),
        )

    elapsed = (time.perf_counter() - start) * 1000
    tldr = _normalize_tldr_markdown(raw)
    compliance = score_compliance(tldr)
    return TldrResult(
        story_id=story.id,
        title=story.title,
        source=story.source,
        status="ok",
        latency_ms=round(elapsed, 1),
        tldr=tldr,
        raw_response=raw,
        compliance=compliance,
    )


# ── Retry / skip logic ───────────────────────────────────────────────────


async def generate_with_retry(
    story: StoryRecord,
    api_key: str,
    model: str,
    base_url: str,
    rate_sleep: float,
    max_retries: int = 3,
) -> TldrResult:
    for attempt in range(max_retries + 1):
        result = await generate_tldr_for_story(
            story,
            api_key,
            model,
            base_url,
            rate_sleep,
        )

        if result.status != "ok":
            raw = result.raw_response or ""
            if "429" in raw or "HTTP 429" in raw or "rate_limit" in raw.lower():
                if attempt < max_retries:
                    sleep_s = 2 ** (attempt + 1) * 5
                    logging.info(
                        "429 on story %d, retry %d/%d in %ds",
                        story.id,
                        attempt + 1,
                        max_retries,
                        sleep_s,
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                result.status = "skipped_rate_limit"
            elif "5" in raw and "HTTP 5" in raw:
                if attempt < max_retries:
                    sleep_s = 2 ** (attempt + 1) * 5
                    await asyncio.sleep(sleep_s)
                    continue
                result.status = "skipped_rate_limit"

        if rate_sleep > 0:
            await asyncio.sleep(rate_sleep)
        return result

    return TldrResult(
        story_id=story.id,
        title=story.title,
        source=story.source,
        status="skipped_rate_limit",
        latency_ms=0.0,
    )


# ── Partial cache (resume) ───────────────────────────────────────────────


def _sanitize_model_id(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _partial_path(output_dir: Path, model: str) -> Path:
    return output_dir / f"tldr_{_sanitize_model_id(model)}.partial.json"


def _final_path(output_dir: Path, model: str) -> Path:
    return output_dir / f"tldr_{_sanitize_model_id(model)}.json"


def load_partial(path: Path) -> dict[int, dict]:
    if path.exists():
        data = json.loads(path.read_text())
        return {r["story_id"]: r for r in data}
    return {}


def save_partial(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def result_to_dict(r: TldrResult) -> dict:
    d = asdict(r)
    if r.compliance:
        d["compliance"] = asdict(r.compliance)
    else:
        d["compliance"] = None
    return d


# ── Model preflight ──────────────────────────────────────────────────────


async def preflight_model_check(
    api_key: str,
    model: str,
    base_url: str,
) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code != 200:
            return False, f"Cannot fetch model list: HTTP {resp.status_code}"
        models = resp.json().get("data", [])
        ids = {m["id"] for m in models}
        if model not in ids:
            suggestion = " (available: ...)"
            similar = sorted(m for m in ids if model.split("/")[0] in m)
            if similar:
                suggestion = f" — did you mean one of {similar[:5]}?"
            return (
                False,
                f"Model '{model}' not found in OpenRouter model list{suggestion}",
            )
        return True, ""
    except Exception as e:
        return False, f"Preflight check failed: {type(e).__name__}: {e}"


# ── Main entry point ─────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TLDR generation quality for an OpenRouter model"
    )
    parser.add_argument(
        "--limit", type=int, default=25, help="Sample size (default 25)"
    )
    parser.add_argument("--db", default="hn_rewrite.db", help="Database path")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory")
    parser.add_argument(
        "--rate-sleep",
        type=float,
        default=2.0,
        help="Fixed sleep (s) between successful calls (default 2.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries on 429/5xx (default 3)",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip model preflight check",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial results if available",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Validate env
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider != "openrouter":
        logging.error("LLM_PROVIDER must be 'openrouter' (got '%s')", provider)
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logging.error("OPENROUTER_API_KEY must be set")
        sys.exit(1)

    model = os.environ.get("OPENROUTER_MODEL", "")
    if not model:
        logging.error(
            "OPENROUTER_MODEL must be set (e.g. google/gemma-4-26b-a4b-it:free)"
        )
        sys.exit(1)

    base_url = os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    started_at = datetime.now(timezone.utc).isoformat()

    # Preflight
    if not args.no_preflight:
        ok, msg = await preflight_model_check(api_key, model, base_url)
        if not ok:
            logging.warning("Preflight: %s (continuing anyway)", msg)
        else:
            logging.info("Preflight: model '%s' found on OpenRouter", model)

    # Select sample
    sample = select_sample(args.db, limit=args.limit)
    logging.info(
        "Selected %d stories: hn=%d rss=%d seed=%d",
        len(sample),
        sum(1 for s in sample if s.source == "hn"),
        sum(1 for s in sample if s.source.startswith("rss_")),
        sum(1 for s in sample if s.source in ("ch_seed", "bq_seed")),
    )

    # Load partial results for resume
    output_dir = Path(args.output_dir)
    partial_path = _partial_path(output_dir, model)
    done: dict[int, dict] = {}
    if args.resume:
        done = load_partial(partial_path)
        logging.info("Loaded %d existing partial results", len(done))

    results: list[dict] = []
    by_source: dict[str, int] = {}
    completed = 0
    skipped_rate_limit = 0
    client_error = 0
    exceptions = 0
    total_latency = 0.0
    compliance_passes = 0

    for story in sample:
        if story.id in done:
            result_data = done[story.id]
            results.append(result_data)
            by_source[story.source] = by_source.get(story.source, 0) + 1
            status = result_data.get("status", "")
            if status == "ok":
                completed += 1
            elif status == "skipped_rate_limit":
                skipped_rate_limit += 1
            elif status == "client_error":
                client_error += 1
            elif status == "exception":
                exceptions += 1
            logging.info("[resumed] story %d (%s): %s", story.id, story.source, status)
            continue

        logging.info("Generating TLDR for story %d (%s)...", story.id, story.source)
        result = await generate_with_retry(
            story,
            api_key,
            model,
            base_url,
            rate_sleep=args.rate_sleep,
            max_retries=args.max_retries,
        )

        result_data = result_to_dict(result)
        results.append(result_data)
        by_source[story.source] = by_source.get(story.source, 0) + 1

        if result.status == "ok":
            completed += 1
            total_latency += result.latency_ms
            if result.compliance and result.compliance.passes:
                compliance_passes += 1
            logging.info(
                "  → %s (%.0fms, compliance=%.2f)",
                result.status,
                result.latency_ms,
                result.compliance.score if result.compliance else 0.0,
            )
        elif result.status == "skipped_rate_limit":
            skipped_rate_limit += 1
            logging.warning(
                "  → skipped (rate limited after %d retries)", args.max_retries
            )
        elif result.status == "client_error":
            client_error += 1
            logging.error("  → client error")
        elif result.status == "exception":
            exceptions += 1
            logging.error("  → exception")

        # Save partial after every story
        save_partial(partial_path, results)

    finished_at = datetime.now(timezone.utc).isoformat()

    # Build summary
    n = len(results)
    check_pass_rates: dict[str, float] = {}
    if n > 0:
        check_pass_rates["nonempty"] = (
            sum(
                1
                for r in results
                if r.get("status") == "ok" and r.get("tldr", "").strip()
            )
            / completed
            if completed > 0
            else 0.0
        )
        for check in (
            "missing_article_heading",
            "nested_lists",
            "normalization_not_idempotent",
            "no_bullets",
        ):
            violations = sum(
                1
                for r in results
                if r.get("compliance") and check in r["compliance"]["violations"]
            )
            check_pass_rates[check] = 1.0 - (violations / n)

    report = BenchmarkReport(
        started_at=started_at,
        finished_at=finished_at,
        config={
            "model": model,
            "provider": "openrouter",
            "limit": args.limit,
            "source_mix": ["hn", "rss", "seed"],
            "prompt_version": PROMPT_VERSION,
        },
        sample={
            "requested": args.limit,
            "sampled": len(sample),
            "by_source": dict(sorted(by_source.items())),
        },
        summary={
            "completed": completed,
            "skipped_rate_limit": skipped_rate_limit,
            "client_error": client_error,
            "exceptions": exceptions,
            "avg_latency_ms": round(total_latency / completed, 1)
            if completed > 0
            else 0.0,
            "compliance_rate": round(compliance_passes / completed, 4)
            if completed > 0
            else 0.0,
            "checks_pass_rate": check_pass_rates,
        },
        results=results,
    )

    output_path = _final_path(output_dir, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str)
    )
    # Remove partial on clean finish
    partial_path.unlink(missing_ok=True)
    logging.info("Report written to %s", output_path)

    # Print summary table
    print()
    print("=" * 60)
    print(f"  TLDR Benchmark: {model}")
    print("=" * 60)
    print(f"  Sampled:       {len(sample)} stories")
    print(f"  Completed:     {completed}")
    print(f"  Skipped (RL):  {skipped_rate_limit}")
    print(f"  Errors:        {client_error}")
    print(f"  Exceptions:    {exceptions}")
    print(f"  Avg latency:   {report.summary['avg_latency_ms']} ms")
    print(f"  Compliance:    {report.summary['compliance_rate']:.1%}")
    print(f"  Report:        {output_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
