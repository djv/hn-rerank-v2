"""Step-2 Muse ranking: real batched opencode subprocesses, ONE frozen profile.

Constraints (per approved brief):
- Model ONLY via /home/dev/.opencode/bin/opencode run --pure --format json
  -m opencode-go/muse-spark-1.3-contributor. No other inference models.
- REAL batches of 6-8 tasks sharing ONE older (training-only) taste profile;
  serial requests; max 1 retry (bounded); cooldown pause on upstream 500.
- Prompt carries story id/title/source/domain/excerpt only. NEVER labels,
  ML scores, strata, or outcomes. Task keys are opaque neutral IDs mapped
  locally. Excerpts come from the task-owned SNAPSHOT (stories columns only;
  this script never reads vote values).
- Reversed re-scores are fresh calls in a separate batch from forward pairs.
- Child env OPENCODE_CONFIG_CONTENT denies all tools; global config untouched.
- Fresh isolated temp cwd INSIDE this dir per batch. JSON-array validation,
  per-task cache with prompt/model hashes.
- Quota: this step <=10 fwd + <=4 rev = 14 comparisons in <=3 batches.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path("/tmp/opencode/cc-replay/snapshot.db")
OPENCODE_BIN = "/home/dev/.opencode/bin/opencode"
MODEL = "opencode-go/muse-spark-1.3-contributor"
BATCH = 7
TIMEOUT_S = 600
MAX_RETRIES = 1
CACHE_FILE = HERE / "muse-cache.json"
OUT_FILE = HERE / "muse_pair_scores.json"
MANIFEST_FILE = HERE / "muse_batches.json"
SEED = 20261011
EXCERPT_CHARS = 600

ALLOWED_STYLES = {
    "substantive",
    "repetitive-question",
    "social-chatter",
    "news-brief",
    "other",
}

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

BATCH_PROMPT_TMPL = """You are a pairwise ranking judge. You have NO tools. Use ONLY the text supplied below. Do NOT access files, credentials, environment, network, or any other model. Do NOT attempt credential discovery or file/network discovery. Base every answer solely on the reader profile and the story texts given.

Reader taste profile (older training votes only; most recent first; infer tastes from it):
---
{profile}
---

You must rank each of the {n} independent pairs below. Presentation order A/B within each pair is randomized and arbitrary; task IDs are opaque identifiers with no meaning.

{pairs_block}

Task: for EACH pair, which story would this reader more likely prefer (wants more like it)?
Reply with JSON ONLY: exactly a JSON array with one object per pair (any order), exactly this shape (no other text):
[{{"task": "{ex_id}", "preferred": "A", "preferred_id": 123, "confidence": 0, "style_A": ["substantive"], "style_B": ["social-chatter"], "rationale": "..."}}]
Rules: "task" MUST be exactly one of the opaque pair IDs listed above (copy exactly); "preferred" is "A" or "B" within that pair; "preferred_id" MUST equal the numeric story id of the story you prefer in that pair (copy exactly); "confidence" is an integer 0-100; style tags per story: zero or more of ["substantive", "repetitive-question", "social-chatter", "news-brief", "other"], judged ONLY from the title+excerpt above; "rationale" is one or two sentences grounded in the profile and the two texts of that pair.
"""


def clean(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def domain_of(url: str | None) -> str:
    return urlparse(url).netloc.removeprefix("www.") if url else ""


def excerpts(ids: set[int]) -> dict[int, dict]:
    con = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro", uri=True)
    out = {}
    for sid in ids:
        title, url, source, self_text, article, text = con.execute(
            "select title, url, source, self_text, article_body, text_content "
            "from stories where id = ?",
            (sid,),
        ).fetchone()
        out[sid] = {
            "id": sid,
            "title": title,
            "source": source,
            "domain": domain_of(url),
            "excerpt": clean(self_text or article or text or "")[:EXCERPT_CHARS],
        }
    con.close()
    return out


def build_pairs_block(items: list[dict]) -> str:
    chunks: list[str] = []
    for it in items:
        a, b = it["a"], it["b"]
        chunks.append(
            f"Pair {it['opaque']}:\n"
            f"A. id={a['id']} | source={a['source']} | domain={a['domain']} | title={a['title']} | excerpt={a['excerpt'] or '(none)'}\n"
            f"B. id={b['id']} | source={b['source']} | domain={b['domain']} | title={b['title']} | excerpt={b['excerpt'] or '(none)'}"
        )
    return "\n---\n".join(chunks)


def build_batch_prompt(profile: str, items: list[dict]) -> str:
    return BATCH_PROMPT_TMPL.format(
        profile=profile,
        n=len(items),
        pairs_block=build_pairs_block(items),
        ex_id=items[0]["opaque"],
    )


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
                        and all(isinstance(o, dict) and "task" in o for o in obj)
                    ):
                        return obj
                    break
        start = text.find("[", start + 1)
    return None


def validate_batch(objs: list, items: list[dict]) -> dict[str, dict]:
    by_opaque = {it["opaque"]: it for it in items}
    if len(objs) != len(items):
        raise ValueError(f"expected {len(items)} results, got {len(objs)}")
    seen: set[str] = set()
    out: dict[str, dict] = {}
    for o in objs:
        t = o.get("task")
        if t not in by_opaque:
            raise ValueError(f"unknown task id {t!r}")
        if t in seen:
            raise ValueError(f"duplicate task id {t!r}")
        seen.add(t)
        it = by_opaque[t]
        a_id, b_id = it["a"]["id"], it["b"]["id"]
        pref, pid, conf = o.get("preferred"), o.get("preferred_id"), o.get("confidence")
        sA, sB, rat = o.get("style_A", []), o.get("style_B", []), o.get("rationale", "")
        if pref not in ("A", "B"):
            raise ValueError(f"bad preferred {pref!r} for {t}")
        if int(pid) != int(a_id if pref == "A" else b_id):
            raise ValueError(f"preferred_id {pid!r} mismatch for {t}")
        conf = int(conf)
        if not (0 <= conf <= 100):
            raise ValueError(f"bad confidence {conf!r} for {t}")
        for name, tags in (("style_A", sA), ("style_B", sB)):
            if not isinstance(tags, list) or not all(
                t2 in ALLOWED_STYLES for t2 in tags
            ):
                raise ValueError(f"bad {name} {tags!r} for {t}")
        if not isinstance(rat, str) or len(rat.strip()) < 10:
            raise ValueError(f"rationale too short for {t}")
        out[t] = {
            "preferred": pref,
            "preferred_id": int(pid),
            "confidence": conf,
            "style_A": list(sA),
            "style_B": list(sB),
            "rationale": rat.strip(),
        }
    if set(out) != set(by_opaque):
        raise ValueError("opaque ID set mismatch")
    return out


def call_batch_once(prompt: str) -> tuple[str, str]:
    tmp = Path(tempfile.mkdtemp(prefix="tmp-batch-", dir=str(HERE)))
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


def opaque_ids(n: int, rng: random.Random) -> list[str]:
    alphabet = string.ascii_lowercase + string.digits
    out: list[str] = []
    while len(out) < n:
        cand = "T-" + "".join(rng.choice(alphabet) for _ in range(6))
        if cand not in out:
            out.append(cand)
    return out


def main() -> None:
    # freeze guard: selection + profile must match preregistration hashes
    prereg = json.loads((HERE / "preregistration.json").read_text())
    for name, want in prereg["hashes"].items():
        if name == "replay_scores.json":
            continue
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        assert got == want, f"freeze violation: {name} changed since preregistration"
    sel = json.loads((HERE / "selection.json").read_text())
    profile = (HERE / "profile.txt").read_text()
    pairs = sel["pairs"]
    rev_ids = list(sel["reversed_pair_ids"])
    assert len(pairs) <= 20 and len(rev_ids) <= 4, "quota: too many pairs"
    by_id = {q["pair_id"]: q for q in pairs}

    ids = {s["story_id"] for q in pairs for s in (q["stories"]["a"], q["stories"]["b"])}
    meta = excerpts(ids)

    all_tasks: list[dict] = []
    for q in pairs:
        first, second = q["order"][0], q["order"][1]
        all_tasks.append(
            {
                "key": q["pair_id"],
                "rev": False,
                "stratum": q["stratum"],
                "a": meta[q["stories"][first]["story_id"]],
                "b": meta[q["stories"][second]["story_id"]],
            }
        )
    for pid in sorted(rev_ids):
        q = by_id[pid]
        first, second = q["order"][1], q["order"][0]
        all_tasks.append(
            {
                "key": pid + ":rev",
                "rev": True,
                "stratum": q["stratum"],
                "a": meta[q["stories"][first]["story_id"]],
                "b": meta[q["stories"][second]["story_id"]],
            }
        )
    assert len(all_tasks) <= 24, "quota: >24 comparisons"

    cache: dict = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    pending = [
        t
        for t in all_tasks
        if not (t["key"] in cache and cache[t["key"]].get("parsed"))
    ]
    print(
        f"{len(all_tasks)} total; {len(pending)} pending, "
        f"{len(all_tasks) - len(pending)} cached reuse",
        flush=True,
    )

    rng = random.Random(SEED)
    batches: list[dict] = []
    fwd = sorted([t for t in pending if not t["rev"]], key=lambda t: t["key"])
    rev = sorted([t for t in pending if t["rev"]], key=lambda t: t["key"])
    for grp in (fwd, rev):
        for i in range(0, len(grp), BATCH):
            chunk = grp[i : i + BATCH]
            if len(chunk) > 8:
                chunk = chunk[:8]
            opaques = opaque_ids(len(chunk), rng)
            batches.append(
                {
                    "rev": grp is rev,
                    "items": [
                        {
                            "opaque": op,
                            "key": t["key"],
                            "rev": t["rev"],
                            "a": t["a"],
                            "b": t["b"],
                        }
                        for t, op in zip(chunk, opaques)
                    ],
                }
            )
    # fwd 10 -> 7+3, rev 4 -> own batch; <=4 batches total
    print(
        "batch requests planned: "
        + ", ".join(
            f"{'rev' if b['rev'] else 'fwd'}x{len(b['items'])}" for b in batches
        ),
        flush=True,
    )
    assert len(batches) <= 4, "quota: >4 batches"

    manifest: dict = {"model": MODEL, "seed": SEED, "batches": []}
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text())
        except json.JSONDecodeError:
            pass
    done_opaque = {
        m["opaque"] for b in manifest.get("batches", []) for m in b.get("mapping", [])
    }

    for bi, batch in enumerate(batches):
        items = batch["items"]
        for it in items:
            while it["opaque"] in done_opaque:
                it["opaque"] = opaque_ids(1, rng)[0]
            done_opaque.add(it["opaque"])
        prompt = build_batch_prompt(profile, items)
        assert not re.search(r"p-\d{2}", prompt), "pair id leaked into batch prompt"
        for tok in ("pair_id", "stratum", "true_label", "ordinal"):
            assert tok not in prompt, f"forbidden {tok} in batch prompt"
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        print(
            f"-- batch {bi + 1}/{len(batches)} {'rev' if batch['rev'] else 'fwd'} "
            f"x{len(items)} ph={ph} --",
            flush=True,
        )
        raw_text, err_tail = "", ""
        parsed_map: dict[str, dict] | None = None
        last_err = ""
        attempts = 0
        for attempt in range(1 + MAX_RETRIES):
            attempts = attempt + 1
            try:
                raw_text, err_tail = call_batch_once(prompt)
                if "500" in err_tail or "cooldown" in err_tail.lower():
                    print(
                        "  upstream 500/cooldown hint; pausing 300s before retry",
                        flush=True,
                    )
                    time.sleep(300)
                objs = find_json_array(raw_text)
                if objs is None:
                    raise ValueError(f"no JSON array with task in: {raw_text[:200]!r}")
                parsed_map = validate_batch(objs, items)
                break
            except (ValueError, subprocess.TimeoutExpired) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(5 * (attempt + 1))
        if parsed_map is None:
            for it in items:
                cache[it["key"]] = {
                    "parsed": None,
                    "raw": raw_text[:2000],
                    "attempts": attempts,
                    "error": last_err,
                    "prompt_hash": ph,
                    "model": MODEL,
                    "opaque": it["opaque"],
                }
            print(f"  batch {bi + 1} FAILED: {last_err[:200]}", flush=True)
        else:
            for it in items:
                cache[it["key"]] = {
                    "parsed": parsed_map[it["opaque"]],
                    "raw": raw_text[:2000],
                    "attempts": attempts,
                    "prompt_hash": ph,
                    "model": MODEL,
                    "opaque": it["opaque"],
                }
                print(
                    f"  {it['key']} -> pref={cache[it['key']]['parsed']['preferred']} "
                    f"conf={cache[it['key']]['parsed']['confidence']}",
                    flush=True,
                )
        CACHE_FILE.write_text(json.dumps(cache, indent=1))
        manifest["batches"].append(
            {
                "batch": bi + 1,
                "rev": batch["rev"],
                "prompt_hash": ph,
                "model": MODEL,
                "stderr_tail": err_tail,
                "mapping": [{"opaque": it["opaque"], "key": it["key"]} for it in items],
            }
        )
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=1))

    results: dict = {}
    for t in all_tasks:
        results[t["key"]] = {
            "rev": t["rev"],
            "stratum": t["stratum"],
            "a_id": t["a"]["id"],
            "b_id": t["b"]["id"],
            **cache.get(t["key"], {}),
        }
    OUT_FILE.write_text(
        json.dumps(
            {"model": MODEL, "n_tasks": len(all_tasks), "results": results}, indent=1
        )
    )
    n_ok = sum(1 for r in results.values() if r.get("parsed"))
    print(f"done: {n_ok}/{len(all_tasks)} parsed OK -> {OUT_FILE.name}")


if __name__ == "__main__":
    sys.exit(main())
