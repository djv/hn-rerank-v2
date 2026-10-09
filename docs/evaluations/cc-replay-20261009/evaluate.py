"""Step-3 evaluation: unblind Muse ranks against frozen labels (read-only).

Reads selection.json, replay_scores.json (labels + ML), muse_pair_scores.json.
Writes results.json (aggregate only) + prints per-pair table for audit.
Deterministic; no LLM calls; no DB access.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = {0: "down", 1: "neutral", 2: "up"}


def main() -> None:
    sel = json.loads((HERE / "selection.json").read_text())
    reps = {
        r["story_id"]: r for r in json.loads((HERE / "replay_scores.json").read_text())
    }
    muse = json.loads((HERE / "muse_pair_scores.json").read_text())["results"]

    rows = []
    for q in sel["pairs"]:
        pid = q["pair_id"]
        orga, orgb = q["stories"]["a"]["story_id"], q["stories"]["b"]["story_id"]
        ra, rb = reps[orga], reps[orgb]
        hi = orga if ra["true_label"] > rb["true_label"] else orgb
        ml_pick = (
            orga if ra["ordinal_up_minus_down"] > rb["ordinal_up_minus_down"] else orgb
        )
        rec = muse[pid]["parsed"]
        # scorer stored a/b in presentation order; map A/B -> story id
        pres = muse[pid]
        muse_pick = pres["a_id"] if rec["preferred"] == "A" else pres["b_id"]
        rev = muse.get(pid + ":rev", {}).get("parsed")
        rev_pick = None
        if rev:
            rpres = muse[pid + ":rev"]
            rev_pick = rpres["a_id"] if rev["preferred"] == "A" else rpres["b_id"]
        rows.append(
            {
                "pair_id": pid,
                "stratum": q["stratum"],
                "hi": hi,
                "ml_pick": ml_pick,
                "muse_pick": muse_pick,
                "muse_correct": muse_pick == hi,
                "ml_correct": ml_pick == hi,
                "muse_conf": rec["confidence"],
                "rev_pick": rev_pick,
                "stable": (rev_pick == muse_pick) if rev_pick else None,
            }
        )

    err = [r for r in rows if r["stratum"] == "error"]
    ctl = [r for r in rows if r["stratum"] == "control"]
    corr = sum(r["muse_correct"] for r in err)
    agree = sum(r["muse_pick"] == r["ml_pick"] for r in ctl)
    stab_rows = [r for r in rows if r["stable"] is not None]
    stab = sum(r["stable"] for r in stab_rows)
    gate = (
        corr / len(err) >= 0.60
        and agree / len(ctl) >= 0.70
        and stab / len(stab_rows) >= 0.75
    )

    out = {
        "n_pairs": len(rows),
        "errors": {
            "n": len(err),
            "muse_corrected": corr,
            "rate": round(corr / len(err), 3),
        },
        "controls": {
            "n": len(ctl),
            "muse_ml_agreement": agree,
            "rate": round(agree / len(ctl), 3),
        },
        "stability": {
            "n": len(stab_rows),
            "stable": stab,
            "rate": round(stab / len(stab_rows), 3),
        },
        "go_gate_pass": gate,
        "note": "Selection-biased development (OLD99 inspected); exploratory, not proof.",
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    for r in rows:
        print(
            f"{r['pair_id']} {r['stratum']:7s} ML={'R' if r['ml_correct'] else 'W'} "
            f"Muse={'R' if r['muse_correct'] else 'W'} conf={r['muse_conf']:3d} "
            f"rev={'-' if r['stable'] is None else ('S' if r['stable'] else 'F')} "
            f"hi={r['hi']} ml={r['ml_pick']} muse={r['muse_pick']}"
        )


if __name__ == "__main__":
    main()
