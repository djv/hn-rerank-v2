"""Smoke tests for scripts/eval_ranker_variants.py.

These run the script as a subprocess to verify the CLI works end-to-end.
The end-to-end test uses a tiny config (1 variant, 2 folds, 1000
candidates) so it stays under ~30s on this host.
"""

import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_ranker_variants.py"
TMP_OUTPUT = Path("/tmp/eval_ranker_variants_smoke.json")


def test_leak_check_flag_in_help() -> None:
    """--leak-check must appear in --help output.

    Locks the flag in place; regression test against accidental removal.
    """
    result = subprocess.run(
        ["uv", "run", "python", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "--leak-check" in result.stdout, (
        "--leak-check not in --help output; flag is missing"
    )


def test_leak_check_smoke() -> None:
    """End-to-end: run --leak-check on a tiny config; verify the JSON
    structure and that shuffled NDCG@40 is much lower than normal
    (no data leakage).

    Uses --max-candidates 5000 --folds 2 --variants margin3_up to keep
    runtime under ~30s. The 0.5 ratio threshold is conservative: a
    clean harness typically gives <0.2 (shuffled is ~random baseline).
    """
    if TMP_OUTPUT.exists():
        TMP_OUTPUT.unlink()
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--variants",
            "margin3_up",
            "--folds",
            "2",
            "--max-candidates",
            "5000",
            "--leak-check",
            "--output",
            str(TMP_OUTPUT),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"script failed: {result.stderr}\nstdout: {result.stdout[-500:]}"
    )
    assert TMP_OUTPUT.exists(), "output JSON not written"

    report = json.loads(TMP_OUTPUT.read_text())

    # Sanity: normal variant results present
    assert "variants" in report, "no variants in report"
    assert "margin3_up" in report["variants"], "margin3_up missing from variants"
    normal_n = report["variants"]["margin3_up"]["mean"]["raw"]["ndcg_at_40"]

    # Sanity: leak check section present and structured the same way
    assert "leak_check" in report, "leak_check key missing from report"
    assert "variants" in report["leak_check"], "leak_check.variants missing"
    assert "margin3_up" in report["leak_check"]["variants"], (
        "leak_check missing margin3_up"
    )
    shuffled_n = report["leak_check"]["variants"]["margin3_up"]["mean"]["raw"][
        "ndcg_at_40"
    ]

    # The harness should NOT have leakage: shuffled NDCG should be
    # well below normal. Use a conservative 0.5 ratio threshold
    # (clean runs typically give <0.2).
    assert normal_n > 0.05, f"normal NDCG suspiciously low: {normal_n}"
    ratio = shuffled_n / normal_n
    assert ratio < 0.5, (
        f"Possible data leakage: shuffled/raw ratio = {ratio:.3f} "
        f"(normal={normal_n:.4f}, shuffled={shuffled_n:.4f})"
    )
