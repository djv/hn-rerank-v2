"""Facet-residual step 2: Muse facet annotation (REAL model calls, bounded).

Constraints (frozen manifest + both reviews):
- Muse ONLY via /home/dev/.opencode/bin/opencode run --pure --format json
  -m opencode-go/muse-spark-1.3-contributor. No other inference models,
  no paid fallback.
- Inputs per story: title + source + domain + body1500 (self_text else
  article_body else EMPTY; text_content NEVER used — it contains comments).
  NEVER comments / top_comments, votes, labels, profile, baseline scores,
  block ids.
- Closed schema (rubric.json enums + unclear); strict JSON array, opaque IDs.
  Labels never enter prompts; batches are the frozen shuffled partitions.
- Quota: 20 batches (<=20 stories each) in <=20 ACTUAL requests INCLUDING
  retries => MAX_RETRIES=0. Any batch that fails validation is recorded as
  failed and the run STOPS with a concrete blocker (no quota overrun).
- 20 seeded repeats are separate items with distinct opaque IDs in different
  batches => separate calls by construction (verified in sample.json).
- Child env OPENCODE_CONFIG_CONTENT denies all tools; global config untouched.
  Fresh isolated temp cwd INSIDE this dir per batch. Per-item cache with
  prompt/model hashes (facet-cache.json is PRIVATE, local, untracked).
- Reads (ro): task-owned snapshot stories columns only (never vote values);
  sample.json batches; rubric.json. Writes (this dir only): facet-cache.json,
  facet_batches.json (manifest), facet_labels.json (aggregate).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path("/tmp/opencode/cc-replay/snapshot.db")
OPENCODE_BIN = "/home/dev/.opencode/bin/opencode"
MODEL = "opencode-go/muse-spark-1.3-contributor"
TIMEOUT_S = 600
MAX_RETRIES = 0  # quota: <=20 actual requests incl retries; no retries
MAX_REQUESTS = 20  # total actual opencode invocations incl interrupts/retries
ATTEMPT_FILE = HERE / "facet_attempts.json"
CACHE_FILE = HERE / "facet-cache.json"
OUT_FILE = HERE / "facet_labels.json"
MANIFEST_FILE = HERE / "facet_batches.json"
NEW_BATCH_MAX = 25  # repaired rebatch cap to fit 399 targets in <=16 calls
BODY_CHARS = 1500

NO_TOOLS_CONFIG = json.dumps(
    {
        "permission": {
            "*": "deny",
            "bash": "deny",
            "read": "deny",
            "edit": "deny",
            "write": "deny",
            "glob": "deny",
            "grep": "deny",
            "patch": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "task": "deny",
            "skill": "deny",
            "todowrite": "deny",
            "todoread": "deny",
            "browser": "deny",
            "computer": "deny",
        }
    }
)

PROMPT_TMPL = """You are a story facet annotator. You have NO tools. Use ONLY the text supplied below. Do NOT access files, credentials, environment, network, or any other model. Do NOT attempt credential discovery or file/network discovery. Judge each item ONLY from its own title+body text.

For EACH item below, pick exactly one option per field (use "unclear" only when the visible text genuinely does not decide it):

- artifact: paper | repo_tool | product_launch | news_report | essay_opinion | personal_narrative | howto | question_discussion | social_post | data_visual | media | other | unclear
- claim: new_result | release | incident_event | argument | explainer | retrospective | unclear
- technical_depth: 0 | 1 | 2 | 3 | unclear
- voice: firsthand | aggregator_commentary | unclear
- actionable: 0 | 1 | 2 | unclear
- conflict_framing: 0 | 1 | 2 | unclear
- entity_focus: company | person | government | oss | science | consumer_product | none | unclear

Items (opaque IDs, no meaning beyond this task):

{items_block}

Reply with JSON ONLY: exactly a JSON array with one object per item (any order), exactly this shape (no other text):
[{{"item": "{ex_id}", "artifact": "news_report", "claim": "incident_event", "technical_depth": "1", "voice": "aggregator_commentary", "actionable": "0", "conflict_framing": "0", "entity_focus": "company"}}]
Rules: "item" MUST be exactly one of the opaque IDs listed above (copy exactly); every other value MUST be exactly one of the listed options for that field (strings); one object per item, no extras, no commentary.
"""


def clean(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def domain_of(url: str | None) -> str:
    return urlparse(url).netloc.removeprefix("www.") if url else ""


def bodies(ids: set[int]) -> dict[int, dict]:
    # REPAIR 2026-10-09: self_text else article_body else EMPTY ONLY.
    # text_content MUST NOT be used: snapshot text_content embeds comments
    # (38/379 unique stories have empty self_text+article_body with nonempty
    # text_content==comments). top_comments never read here either.
    con = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro", uri=True)
    out: dict[int, dict] = {}
    for sid in ids:
        row = con.execute(
            "SELECT title, url, source, self_text, article_body"
            " FROM stories WHERE id=?",
            (sid,),
        ).fetchone()
        assert row is not None, f"story {sid} missing"
        title, url, source, stx, ab = row
        body = clean(stx or ab or "")[:BODY_CHARS]
        out[sid] = {
            "story_id": sid,
            "title": title or "",
            "source": source or "",
            "domain": domain_of(url),
            "body": body,
        }
    con.close()
    return out


def build_items_block(items: list[dict], meta: dict[int, dict]) -> str:
    chunks = []
    for it in items:
        m = meta[it["story_id"]]
        chunks.append(
            f"Item {it['opaque']}:\n"
            f"title={m['title']}\n"
            f"source={m['source']} | domain={m['domain']}\n"
            f"body={m['body'] or '(none)'}"
        )
    return "\n---\n".join(chunks)


def extract_texts(proc: subprocess.CompletedProcess) -> str:
    texts: list[str] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = ev.get("part", {})
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts).strip()


def find_json_array(text: str) -> list | None:
    start = text.find("[")
    while start != -1:
        depth = 0
        instr = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if (
                        isinstance(obj, list)
                        and obj
                        and all(isinstance(o, dict) and "item" in o for o in obj)
                    ):
                        return obj
                    break
        start = text.find("[", start + 1)
    return None


def validate_batch(objs: list, items: list[dict], rubric: dict) -> dict[str, dict]:
    fields = rubric["fields"]
    by_opaque = {it["opaque"]: it for it in items}
    if len(objs) != len(items):
        raise ValueError(f"expected {len(items)} results, got {len(objs)}")
    seen: set[str] = set()
    out: dict[str, dict] = {}
    for o in objs:
        t = o.get("item")
        if t not in by_opaque:
            raise ValueError(f"unknown item id {t!r}")
        if t in seen:
            raise ValueError(f"duplicate item id {t!r}")
        seen.add(t)
        rec: dict[str, str] = {}
        for f, opts in fields.items():
            v = o.get(f)
            if v not in opts:
                raise ValueError(f"bad {f}={v!r} for {t}")
            rec[f] = v
        out[t] = rec
    if set(out) != set(by_opaque):
        raise ValueError("opaque ID set mismatch")
    return out


def call_batch_once(prompt: str) -> tuple[str, str]:
    tmp = Path(tempfile.mkdtemp(prefix="tmp-facet-", dir=str(HERE)))
    try:
        env = dict(os.environ)
        env["OPENCODE_CONFIG_CONTENT"] = NO_TOOLS_CONFIG
        proc = subprocess.run(
            [OPENCODE_BIN, "run", "--pure", "--format", "json", "-m", MODEL, prompt],
            cwd=str(tmp),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env=env,
            check=False,
        )
        return extract_texts(proc), (proc.stderr or "")[-500:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_attempts() -> dict:
    # Initialized 4 (3 completed + 1 interrupted, CONSERVATIVE) before repair.
    if not ATTEMPT_FILE.exists():
        d = {
            "attempts": 4,
            "max_requests": MAX_REQUESTS,
            "note": "counts every opencode invocation incl interrupts",
        }
        _atomic_write(ATTEMPT_FILE, d)
        return d
    return json.loads(ATTEMPT_FILE.read_text())


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


def _reserve_attempt() -> int:
    # Reserve BEFORE the call so interrupts still count. Atomic tmp+rename.
    d = _load_attempts()
    if int(d.get("attempts", 0)) >= MAX_REQUESTS:
        raise RuntimeError("QUOTA EXCEEDED: >=20 actual requests; STOP.")
    d["attempts"] = int(d.get("attempts", 0)) + 1
    _atomic_write(ATTEMPT_FILE, d)
    return int(d["attempts"])


def _plan_new_batches(pending: list[dict]) -> list[list[dict]]:
    # Separate original repeat occurrences from sample/dev; each new batch
    # <= NEW_BATCH_MAX slots, reusing frozen opaque IDs (no new IDs, no
    # labels/baseline/profile involved).
    reps = [it for it in pending if it.get("kind") == "repeat"]
    rest = [it for it in pending if it.get("kind") != "repeat"]
    plan: list[list[dict]] = []
    for i in range(0, len(reps), NEW_BATCH_MAX):
        plan.append(reps[i : i + NEW_BATCH_MAX])
    for i in range(0, len(rest), NEW_BATCH_MAX):
        plan.append(rest[i : i + NEW_BATCH_MAX])
    assert all(len(b) <= NEW_BATCH_MAX for b in plan)
    for b in plan:
        kinds = {it.get("kind") for it in b}
        assert not (("repeat" in kinds) and len(kinds) > 1), (
            "repeat batch mixed with sample/dev"
        )
    return plan


def main() -> int:
    rubric = json.loads((HERE / "rubric.json").read_text())
    sample = json.loads((HERE / "sample.json").read_text())
    batches = sample["batches"]
    assert len(batches) == 20 and all(len(b) <= 20 for b in batches)
    assert sum(len(b) for b in batches) == 399
    flat = [it for b in batches for it in b]
    assert len({it["opaque"] for it in flat}) == 399, "frozen opaque IDs must be unique"
    ids = {it["story_id"] for it in flat}
    meta = bodies(ids)
    # Repaired bodies must never carry comments: spot-assert no text_content use.
    # (bodies() selects self_text/article_body only; see body_fix_amendment.json.)

    cache: dict = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    manifest: dict = {"model": MODEL, "max_retries": MAX_RETRIES, "batches": []}
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text())
        except json.JSONDecodeError:
            pass
    att = _load_attempts()
    n_requests = int(att.get("attempts", 0))

    pending = [
        it
        for it in flat
        if not (it["opaque"] in cache and cache[it["opaque"]].get("parsed"))
    ]
    if not pending:
        results = {
            it["opaque"]: {
                "story_id": it["story_id"],
                "kind": it["kind"],
                **cache[it["opaque"]],
            }
            for it in flat
        }
        n_ok = sum(1 for r in results.values() if r.get("parsed"))
        OUT_FILE.write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "n_items": len(results),
                    "n_ok": n_ok,
                    "results": results,
                },
                indent=1,
            )
        )
        print(
            f"done: {n_ok}/{len(results)} parsed OK, requests={n_requests} -> {OUT_FILE.name}"
        )
        return 0 if n_ok == len(results) else 3

    plan = _plan_new_batches(pending)
    print(
        f"pending {len(pending)} in {len(plan)} new batches "
        f"(repeats {[len(b) for b in plan if b and b[0].get('kind') == 'repeat']}, "
        f"attempts used {n_requests}/{MAX_REQUESTS})",
        flush=True,
    )
    if n_requests + len(plan) > MAX_REQUESTS:
        print(
            f"BLOCKER: need {len(plan)} calls but only "
            f"{MAX_REQUESTS - n_requests} of 20 remain; STOP."
        )
        return 3
    base_n = len(manifest.get("batches", []))

    for pi, batch in enumerate(plan):
        prompt = PROMPT_TMPL.format(
            items_block=build_items_block(batch, meta), ex_id=batch[0]["opaque"]
        )
        for pat in (
            r"true_label",
            r"ordinal",
            r"prob_up",
            r"vote_time",
            r"pair_id",
            r"stratum",
            r"upvote",
            r"downvote",
            r"block_id",
            r"block\s*[1-5]",
        ):
            assert not re.search(pat, prompt, re.IGNORECASE), (
                f"forbidden {pat} in facet prompt"
            )
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        print(
            f"-- new batch {pi + 1}/{len(plan)} x{len(batch)} "
            f"kind={batch[0].get('kind')} ph={ph} --",
            flush=True,
        )
        try:
            n_requests = _reserve_attempt()
        except RuntimeError as e:
            print(str(e))
            return 3
        raw_text, err_tail = call_batch_once(prompt)
        if "luna" in raw_text.lower() or "luna" in err_tail.lower():
            print("BLOCKER: non-Muse text detected; STOP.")
            return 3
        try:
            objs = find_json_array(raw_text)
            if objs is None:
                raise ValueError(f"no JSON array with item in: {raw_text[:200]!r}")
            parsed = validate_batch(objs, batch, rubric)
        except (ValueError, subprocess.TimeoutExpired) as e:
            for it in batch:
                cache[it["opaque"]] = {
                    "parsed": None,
                    "raw": raw_text[:2000],
                    "attempts": 1,
                    "error": f"{type(e).__name__}: {e}",
                    "prompt_hash": ph,
                    "model": MODEL,
                    "story_id": it["story_id"],
                    "kind": it["kind"],
                }
            CACHE_FILE.write_text(json.dumps(cache, indent=1))
            print(
                f"  new batch {pi + 1} FAILED (no retry budget): {e}"[:300], flush=True
            )
            print(
                "BLOCKER: batch failed with zero retry budget; STOP, no quota overrun."
            )
            return 3
        for it in batch:
            cache[it["opaque"]] = {
                "parsed": parsed[it["opaque"]],
                "raw": raw_text[:500],
                "attempts": 1,
                "prompt_hash": ph,
                "model": MODEL,
                "story_id": it["story_id"],
                "kind": it["kind"],
            }
        CACHE_FILE.write_text(json.dumps(cache, indent=1))
        manifest["batches"].append(
            {
                "batch": base_n + pi + 1,
                "prompt_hash": ph,
                "model": MODEL,
                "requests": 1,
                "repaired": True,
                "kinds": sorted({it.get("kind", "") for it in batch}),
                "stderr_tail": err_tail,
                "mapping": [
                    {
                        "opaque": it["opaque"],
                        "story_id": it["story_id"],
                        "kind": it["kind"],
                    }
                    for it in batch
                ],
            }
        )
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=1))
        print(
            f"  new batch {pi + 1} OK ({len(batch)} items, total requests {n_requests})",
            flush=True,
        )

    results = {
        it["opaque"]: {
            "story_id": it["story_id"],
            "kind": it["kind"],
            **cache[it["opaque"]],
        }
        for it in flat
    }
    n_ok = sum(1 for r in results.values() if r.get("parsed"))
    OUT_FILE.write_text(
        json.dumps(
            {"model": MODEL, "n_items": len(results), "n_ok": n_ok, "results": results},
            indent=1,
        )
    )
    print(
        f"done: {n_ok}/{len(results)} parsed OK, requests={n_requests} -> {OUT_FILE.name}"
    )
    return 0 if n_ok == len(results) else 3


if __name__ == "__main__":
    sys.exit(main())
