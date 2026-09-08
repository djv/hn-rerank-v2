"""Same-story TLDR bakeoff across LLM providers (default: groq vs gemini).

Picks N stories with both halves present, runs generate_detailed_tldr under
each provider, and reports kind/latency/size per story. Read-only against
the DB (generate_detailed_tldr never writes; only _maybe_cache_tldr does,
and it is not called here). Keep N small: Groq free trips fast.

Usage: uv run python scripts/bakeoff_tldr_providers.py [--stories 4]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server


def pick_stories(db_path: str, n: int) -> list[dict[str, str]]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT title, self_text, top_comments, article_body FROM stories "
            "WHERE self_text != '' AND top_comments != '' AND article_body != '' "
            "ORDER BY time DESC LIMIT ?",
            (n,),
        ).fetchall()
    finally:
        con.close()
    return [
        {
            "title": t or "",
            "self_text": s or "",
            "top_comments": c or "",
            "article_body": a or "",
        }
        for t, s, c, a in rows
    ]


async def run_provider(
    provider: str, stories: list[dict[str, str]]
) -> list[dict[str, object]]:
    os.environ["LLM_PROVIDER"] = provider
    # BAKEOFF_MODEL pins one model across providers; otherwise each
    # provider's default is used (LLM_MODEL is ignored to avoid leaks).
    pinned = os.environ.get("BAKEOFF_MODEL")
    if pinned:
        os.environ["LLM_MODEL"] = pinned
    else:
        os.environ.pop("LLM_MODEL", None)
    cfg = server._llm_provider_config()
    if not cfg.api_key:
        return [{"provider": provider, "error": f"missing {provider} key"}]
    print(f"--- {provider} model={cfg.model} ---", flush=True)
    server.llm_limiter.reset()  # cross-provider: one ban must not fail-fast the next
    out: list[dict[str, object]] = []
    for i, s in enumerate(stories):
        t0 = time.perf_counter()
        try:
            res = await server.generate_detailed_tldr(
                s["title"],
                self_text=s["self_text"],
                top_comments=s["top_comments"],
                article_body=s["article_body"],
            )
            ms = (time.perf_counter() - t0) * 1000.0
            row: dict[str, object] = {
                "story": i,
                "kind": res.kind,
                "ms": round(ms),
                "chars": len(res.tldr),
                "status": res.error_status,
            }
            print(
                f"  story {i}: kind={res.kind} ms={ms:.0f} chars={len(res.tldr)} "
                f"status={res.error_status} err={str(res.error_text)[:160]!r}",
                flush=True,
            )
            out.append(row)
        except Exception as e:  # noqa: BLE001 - bakeoff must report, not crash
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"  story {i}: EXCEPTION {type(e).__name__}: {e}", flush=True)
            out.append(
                {"story": i, "kind": "exception", "ms": round(ms), "error": str(e)}
            )
        await asyncio.sleep(6)  # stay under free-tier RPM on both providers
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=4)
    ap.add_argument("--providers", default="groq,gemini")
    args = ap.parse_args()
    server.load_env()
    stories = pick_stories("hn_rewrite.db", args.stories)
    print(f"baking off {len(stories)} stories")
    for provider in args.providers.split(","):
        asyncio.run(run_provider(provider.strip(), stories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
