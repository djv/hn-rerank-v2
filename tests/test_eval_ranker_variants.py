"""AST-based checks for scripts/eval_ranker_variants.py."""

import ast
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_ranker_variants.py"


def test_leak_check_flag_in_help() -> None:
    """--leak-check must be wired into argparse.

    Parses the script's source for `argparse.ArgumentParser.add_argument`
    calls containing the literal `--leak-check`. Cheaper than booting a
    subprocess (the script imports sklearn + onnx at top, ~2.5s).
    """
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == "--leak-check":
                    return
    pytest.fail("--leak-check not in any argparse add_argument call")
