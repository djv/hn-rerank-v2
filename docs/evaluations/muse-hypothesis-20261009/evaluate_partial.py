"""Freeze completed rankings, then evaluate outcomes read-only on the VPS."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def wilson(correct: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.96
    fraction = correct / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(center - radius, 3), round(center + radius, 3)]


def main() -> None:
    here = Path(__file__).resolve().parent
    names = (
        "preregistration.json",
        "selection.json",
        "profile_fresh.txt",
        "muse_pair_scores_fresh.json",
        "muse-cache-fresh.json",
    )
    frozen = {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in names
    }
    freeze = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "hashes": frozen,
        "note": "Committed completed ranking artifacts before evaluator reads candidate vote outcomes; no further ranking permitted in this partial analysis.",
    }
    freeze_path = here / "evaluation-freeze.json"
    original_freeze = here / "partial-evaluation-freeze.json"
    if freeze_path.exists() and not original_freeze.exists():
        original_freeze.write_bytes(freeze_path.read_bytes())
    previous_result = here / "results.json"
    partial_result = here / "partial-results.json"
    if previous_result.exists() and not partial_result.exists():
        partial_result.write_bytes(previous_result.read_bytes())
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    selection = json.loads((here / "selection.json").read_text())
    ranks = json.loads((here / "muse_pair_scores_fresh.json").read_text())["results"]
    ids = sorted(
        {
            story["id"]
            for pair in selection["pairs"]
            for story in pair["stories"].values()
        }
    )
    db_path = Path.home() / "hn-rewrite/main/hn_rewrite.db"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT story_id, action, updated_at FROM feedback WHERE user_id = ? AND story_id IN ("
            + ",".join("?" for _ in ids)
            + ")",
            (151, *ids),
        ).fetchall()
    outcomes = {
        sid: action
        for sid, action, updated in rows
        if selection["window_start"] <= updated <= selection["window_end"]
    }
    available = 0
    comparable = 0
    correct = 0
    ties = 0
    missing = 0
    precedent_total = 0
    precedent_picks = 0
    conditional: dict[str, list[int]] = {
        "precedent_up": [0, 0],
        "precedent_nonup": [0, 0],
    }
    for pair in selection["pairs"]:
        rank = ranks.get(pair["pair_id"], {})
        parsed = rank.get("parsed")
        if not parsed:
            continue
        available += 1
        preferred = parsed["preferred_id"]
        precedent = pair["precedent_id"]
        if precedent != "tie":
            precedent_total += 1
            precedent_picks += preferred == precedent
        a, b = (pair["stories"][key]["id"] for key in ("s1", "s2"))
        if a not in outcomes or b not in outcomes:
            missing += 1
            continue
        a_up, b_up = outcomes[a] == "up", outcomes[b] == "up"
        if a_up == b_up:
            ties += 1
            continue
        comparable += 1
        truth = a if a_up else b
        won = preferred == truth
        correct += won
        if precedent != "tie":
            key = "precedent_up" if precedent == truth else "precedent_nonup"
            conditional[key][0] += won
            conditional[key][1] += 1
    reverse_ids = selection["reversed_pair_ids"]
    reverse_complete = [
        key
        for key in reverse_ids
        if ranks.get(key + ":rev", {}).get("parsed")
        and ranks.get(key, {}).get("parsed")
    ]
    stable = sum(
        ranks[key]["parsed"]["preferred_id"]
        == ranks[key + ":rev"]["parsed"]["preferred_id"]
        for key in reverse_complete
    )
    primary = precedent_total > 0 and precedent_picks / precedent_total >= 0.67
    counts = list(conditional.values())
    enough = all(total >= 6 for _, total in counts)
    difference = (
        counts[0][0] / counts[0][1] - counts[1][0] / counts[1][1]
        if all(total for _, total in counts)
        else None
    )
    complete = available == len(selection["pairs"]) and len(reverse_complete) == len(
        reverse_ids
    )
    stability_met = len(reverse_complete) == 6 and stable >= 5
    survived = (
        primary
        and enough
        and difference is not None
        and difference >= 0.4
        and stability_met
    )
    decision = (
        (
            "SURVIVES exploratory gate; independent ML comparison still required"
            if survived
            else "NOT SUPPORTED by preregistered gate; no challenger built"
        )
        if complete
        else "INCONCLUSIVE; incomplete protocol; no challenger built"
    )
    result = {
        "status": "complete" if complete else "partial_provider_blocked",
        "model": "opencode-go/muse-spark-1.3-contributor",
        "selected_pairs": len(selection["pairs"]),
        "completed_forward": available,
        "planned_reversed": len(reverse_ids),
        "completed_reversed": len(reverse_complete),
        "stable": stable,
        "up_vs_rest": {
            "correct": correct,
            "comparable_pairs": comparable,
            "outcome_ties_excluded": ties,
            "missing_or_changed_outcomes": missing,
            "wilson_95_percent": wilson(correct, comparable),
            "interval_note": "Descriptive pair interval; reused stories and topic clustering violate independent-trial assumption.",
        },
        "hypothesis": {
            "precedent_picks": precedent_picks,
            "non_tied_precedent_pairs": precedent_total,
            "primary_threshold_met": primary,
            "conditional_correct_total": conditional,
            "conditional_minimum_met": enough,
            "accuracy_difference": difference,
            "stability_requirement_met": stability_met,
            "decision": decision,
        },
        "ml_comparison": "unknown: no saved out-of-sample scores cover these fresh stories",
        "provider_failure": "OpenCode: OpenRouter fallback paused: credit exhausted or unverified. Go retries after cooldown.",
        "retry_protocol_note": "The user authorized retry after partial outcomes were reviewed. Selection, hypothesis and text-only prompts stayed fixed; nine missing rankings used fresh isolated sessions without outcome access. Original partial freeze and results retained.",
        "freeze": freeze,
        "production_changes": False,
    }
    (here / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "freeze"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
