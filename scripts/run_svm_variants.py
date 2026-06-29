"""Run manual SVM variant experiments.

This script temporarily edits tracked files while evaluating each variant.
It is intentionally inert when imported.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config.toml"
PIPELINE = REPO_ROOT / "pipeline.py"
EVAL = REPO_ROOT / "eval.py"
REPORT = REPO_ROOT / "eval_report.json"
TEMP_FILES = ("_test_kernels.py", "_diag_scores.py")


class TrialResult(NamedTuple):
    soft_ndcg10: float
    soft_ndcg10_std: float
    soft_p10: float
    soft_median_rank: float


class TrialError(NamedTuple):
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-dirty",
        action="store_true",
        help="allow running when tracked files already have uncommitted changes",
    )
    return parser.parse_args()


def _git_status_short() -> list[str]:
    run = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in run.stdout.splitlines() if line.strip()]


def _tracked_status_lines(status_lines: list[str]) -> list[str]:
    return [line for line in status_lines if not line.startswith("?? ")]


def preflight_dirty_worktree(force_dirty: bool) -> None:
    status_lines = _git_status_short()
    untracked = [line for line in status_lines if line.startswith("?? ")]
    tracked = _tracked_status_lines(status_lines)

    if untracked:
        print("Untracked files present; they do not block this experiment:")
        for line in untracked:
            print(f"  {line}")

    if tracked and not force_dirty:
        print(
            "Refusing to run because tracked files have uncommitted changes.",
            file=sys.stderr,
        )
        print("Re-run with --force-dirty to override.", file=sys.stderr)
        for line in tracked:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(2)


def build_trials() -> list[tuple[str, str, float, str | None, str | None]]:
    trials = []
    for kernel in ("linear", "rbf"):
        for c in (0.1, 1.0, 10.0):
            trials.append((f"{kernel}_C{c:g}_sig", kernel, c, "sigmoid", None))
        trials.append((f"{kernel}_C03_iso", kernel, 0.3, "isotonic", None))
        trials.append((f"{kernel}_softmax", kernel, 0.3, None, "softmax"))
    return trials


def patch_config(config_base: str, kernel: str, c: float) -> None:
    cfg_lines = []
    for line in config_base.splitlines():
        if line.strip().startswith("svm_kernel"):
            cfg_lines.append(f'svm_kernel = "{kernel}"')
        elif line.strip().startswith("svm_c"):
            cfg_lines.append(f"svm_c = {c}")
        elif line.strip().startswith("# svm_c"):
            cfg_lines.append(f"svm_c = {c}")
        elif line.strip().startswith("svm_gamma"):
            if kernel == "linear":
                cfg_lines.append("# svm_gamma = 0.1")
            else:
                cfg_lines.append("svm_gamma = 0.1")
        else:
            cfg_lines.append(line)
    CONFIG.write_text("\n".join(cfg_lines))


def patch_model_code(
    pipeline_base: str,
    eval_base: str,
    method: str | None,
    margin_mode: str | None,
) -> None:
    if margin_mode == "softmax":
        pipe_code = re.sub(
            r"n_train = len\(fb_features_scaled\)\n\s+calibrated = CalibratedClassifierCV\([^)]+\)\n\s+calibrated\.fit\(fb_features_scaled, labels, sample_weight=sample_weights\)",
            "",
            pipeline_base,
        )
        pipe_code = re.sub(
            r"probs = calibrated\.predict_proba\(cand_features_scaled\)\n\s+class_order = list\(calibrated\.classes_\)",
            "margins = svm.decision_function(cand_features_scaled)\n"
            "margins_exp = np.exp(margins - margins.max(axis=1, keepdims=True))\n"
            "probs = margins_exp / margins_exp.sum(axis=1, keepdims=True)\n"
            "class_order = list(svm.classes_)",
            pipe_code,
        )

        eval_code = re.sub(
            r"n_train = len\(X_train_scaled\)\n\s+calibrated = CalibratedClassifierCV\([^)]+\)\n\s+calibrated\.fit\(X_train_scaled, y_train, sample_weight=weights\)",
            "",
            eval_base,
        )
        eval_code = re.sub(
            r"probs = calibrated\.predict_proba\(X_all_scaled\)",
            "margins = svm.decision_function(X_all_scaled)\n"
            "margins_exp = np.exp(margins - margins.max(axis=1, keepdims=True))\n"
            "probs = margins_exp / margins_exp.sum(axis=1, keepdims=True)",
            eval_code,
        )

        PIPELINE.write_text(pipe_code)
        EVAL.write_text(eval_code)
        return

    if method != "sigmoid":
        PIPELINE.write_text(
            pipeline_base.replace('method="sigmoid"', f'method="{method}"')
        )
        EVAL.write_text(eval_base.replace('method="sigmoid"', f'method="{method}"'))
        return

    PIPELINE.write_text(pipeline_base)
    EVAL.write_text(eval_base)


def run_eval() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "eval.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def restore_files(config_base: str, pipeline_base: str, eval_base: str) -> None:
    CONFIG.write_text(config_base)
    PIPELINE.write_text(pipeline_base)
    EVAL.write_text(eval_base)


def cleanup_temp_files() -> None:
    for filename in TEMP_FILES:
        (REPO_ROOT / filename).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    preflight_dirty_worktree(force_dirty=args.force_dirty)

    config_base = CONFIG.read_text()
    pipeline_base = PIPELINE.read_text()
    eval_base = EVAL.read_text()
    results: dict[str, TrialResult | TrialError] = {}

    try:
        for label, kernel, c, method, margin_mode in build_trials():
            print(f"\n{'=' * 60}")
            print(f"TRIAL: {label}")
            print("=" * 60)

            patch_config(config_base, kernel, c)
            patch_model_code(pipeline_base, eval_base, method, margin_mode)

            print("\n--- eval.py ---")
            eval_run = run_eval()
            if eval_run.returncode != 0:
                print(f"ERR: {eval_run.stderr[-300:]}")
                results[label] = TrialError(error=eval_run.stderr[-200:])
                continue

            if REPORT.exists():
                report = json.loads(REPORT.read_text())
                soft = report["formulas"]["soft"]
                result = TrialResult(
                    soft_ndcg10=soft["mean"]["ndcg_at_10"],
                    soft_ndcg10_std=soft["std"]["ndcg_at_10"],
                    soft_p10=soft["mean"]["precision_at_10"],
                    soft_median_rank=soft["mean"]["median_rank"],
                )
                results[label] = result
                print(
                    ">>> soft NDCG@10: "
                    f"{result.soft_ndcg10:.4f} "
                    f"+/- {result.soft_ndcg10_std:.4f}"
                )

            restore_files(config_base, pipeline_base, eval_base)
    finally:
        restore_files(config_base, pipeline_base, eval_base)
        cleanup_temp_files()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Variant':20s} {'NDCG@10':>10s} {'P@10':>8s} {'med_rk':>8s}")
    print("-" * 52)
    for label, data in results.items():
        if isinstance(data, TrialError):
            print(f"{label:20s} ERROR: {data.error[:40]}")
        else:
            print(
                f"{label:20s} {data.soft_ndcg10:10.4f} "
                f"{data.soft_p10:8.3f} {data.soft_median_rank:8.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
